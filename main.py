#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import random
import warnings

import numpy as np
import open_clip
import torch
import wandb
import yaml
from tqdm import tqdm

from data import get_coco_dataloaders
from losses import (
    ClipLoss,
    get_alpha,
    get_beta,
    get_cosine_schedule_with_warmup,
    lalign_loss,
    lunif_centroid,
    lunif_loss,
    lunif_modality,
)
from metrics import (
    compute_clustering_metrics,
    compute_gap,
    compute_mean_angular_value_of_a_modality,
    compute_metric_ret,
    mean_distance_of_true_pairs,
    uniformity,
)


warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Per-run artifacts (checkpoints, embeddings, ...) live under RUNS_ROOT/<run_name>/.
RUNS_ROOT = os.path.join(PROJECT_ROOT, 'runs')
os.makedirs(RUNS_ROOT, exist_ok=True)


# ---------------------------------------------------------------------------
# Loss-type identifiers. Keep the long string keys (they're also the values
# users put into config.yaml); use the short names inside the cascade below.
# ---------------------------------------------------------------------------
LOSS_ANCHOR             = "anchor"
LOSS_LUNIF_THEN_FULL    = "only_lunif_n_then_anchor+lalign+lunif(text)+lunif(img)"
LOSS_LUNIF_THEN_CENT    = "only_lunif_n_then_anchor+lalign+lunif(centroids)"
LOSS_LUNIF_THEN_BLUNIF  = "only_lunif_n_then_anchor+lalign+BETA*lunif(centroids)"
LOSS_LUNIF_THEN_AB_FULL = "only_lunif_n_then_anchor+ALPHA*lalign+BETA*(lunif(text)+lunif(img))"
LOSS_LUNIF_THEN_AB_CENT = "only_lunif_n_then_anchor+ALPHA*lalign+BETA*lunif(centroids)"
LOSS_ABL_FULL           = "ANCHOR(IMAGE,TEXT)+LALIGN(IMAGE,TEXT)+LUNIF(CENTROIDS)"
LOSS_ABL_NO_UNIF        = "ANCHOR(IMAGE,TEXT)+LALIGN(IMAGE,TEXT)"
LOSS_ABL_NO_ALIGN       = "ANCHOR(IMAGE,TEXT)+LUNIF(CENTROIDS)"
LOSS_LALIGN_LUNIF       = "only_lunif_n_+lalign+lunif(centroids)"

# Loss types that share the lunif-only warmup phase (`only_lunif_n_then_*`).
_WARMUP_LOSS_TYPES = {
    LOSS_LUNIF_THEN_FULL,
    LOSS_LUNIF_THEN_CENT,
    LOSS_LUNIF_THEN_BLUNIF,
    LOSS_LUNIF_THEN_AB_FULL,
    LOSS_LUNIF_THEN_AB_CENT,
}

# Loss types whose objective contains an alpha-scheduled `lalign` term.
_ALPHA_LOSS_TYPES = {LOSS_LUNIF_THEN_AB_FULL, LOSS_LUNIF_THEN_AB_CENT}

# Loss types whose objective contains a beta-scheduled lunif term.
_BETA_LOSS_TYPES  = {LOSS_LUNIF_THEN_BLUNIF, LOSS_LUNIF_THEN_AB_FULL, LOSS_LUNIF_THEN_AB_CENT}


def _fmt_lr(lr):
    # 0.0001 → "1e-4" (compact scientific, no leading zero in exponent).
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def build_auto_run_name(config):
    """Build run_name from the most-frequently-changed hyperparameters.

    Order: model → batch_size → lr → epochs → temperature mode → precision
    → α/β schedules (only when the chosen loss_type uses them).
    A timestamp suffix is appended to prevent runs/<name>/ collisions when
    two launches share identical hyperparameters.
    """
    from datetime import datetime

    # Resolve precision with backward-compat for legacy `fp16: bool` configs.
    precision = config.get("precision")
    if precision is None:
        precision = "fp16" if config.get("fp16", False) else "fp32"

    parts = [
        config["model"].replace("/", "-"),  # ViT-B/32 → ViT-B-32 (slash unsafe in paths)
        f"bs{config['batch_size']}",
        f"lr{_fmt_lr(config['learning_rate'])}",
        f"ep{config['epochs']}",
        "Tlearn" if config["anchor_temperature_learnable"] else "Tfix",
        precision,
    ]

    loss_type = config["loss_type"]
    if loss_type in _ALPHA_LOSS_TYPES:
        parts.append(f"a{config['alpha_warmup_epoch']}-{config['alpha_increment_epoch']}")
    if loss_type in _BETA_LOSS_TYPES:
        parts.append(f"b{config['beta_warmup_epoch']}-{config['beta_decay_epoch']}")

    parts.append(datetime.now().strftime("%Y%m%d-%H%M%S"))
    return "_".join(parts)


def setup_run_dir(run_name):
    run_dir  = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    emb_dir  = os.path.join(run_dir, 'embeddings')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    return run_dir, ckpt_dir, emb_dir


def per_loss_grad_norms(components, params):
    """L2 norm of the gradient of each loss component w.r.t. params.
    Uses torch.autograd.grad(retain_graph=True) so it does not interfere with
    the subsequent .backward() call. Computed in fp32 for numerical accuracy.
    """
    out = {}
    for name, comp in components.items():
        grads = torch.autograd.grad(comp, params, retain_graph=True, allow_unused=True)
        sq = 0.0
        for g in grads:
            if g is not None:
                sq += g.detach().float().pow(2).sum().item()
        out[name] = sq ** 0.5
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, test_loader, device, model_name, config, clip_loss,
                   epoch=None, emb_dir=None, plot_embeddings=True):
    """Run text↔image retrieval + clustering eval on test_loader.

    Returns the per-key metric dict (also written to wandb under semantic
    prefixes, and dumped to {emb_dir}/epoch_NNN.pt for post-hoc visualization).
    `clip_loss` is the trained `ClipLoss` instance — its temperature (which
    may be learnable) is used for the val_loss/anchor computation.
    """
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    all_image_embeds, all_text_embeds = [], []
    ids_img, ids_txt = [], []
    all_captions = []  # captions actually fed to the encoder

    # Per-batch validation loss (averaged across batches), mirroring train's
    # loss/{anchor,lalign,lunif_centroids} components.
    val_loss_acc = {"anchor": 0.0, "lalign": 0.0,
                    "lunif_centroids": 0.0, "n_batches": 0}

    with torch.no_grad():
        loader_iter = tqdm(test_loader, desc="Evaluating") if plot_embeddings else test_loader
        for images, captions_list, sample_ids in loader_iter:
            images = images.to(device)
            all_captions.extend(list(captions_list))
            text_tokens = tokenizer(captions_list).to(device)

            image_embeds = model.module.encode_image(images)
            text_embeds  = model.module.encode_text(text_tokens)

            # Normalize for val-loss components (mirrors training flow).
            ie_n = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            te_n = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
            val_loss_acc["anchor"]          += clip_loss(ie_n, te_n).item()
            val_loss_acc["lalign"]          += lalign_loss(ie_n, te_n).item()
            val_loss_acc["lunif_centroids"] += lunif_centroid(ie_n, te_n).item()
            val_loss_acc["n_batches"]       += 1

            all_image_embeds.append(image_embeds.cpu())
            all_text_embeds.append(text_embeds.cpu())
            ids_img.extend(sample_ids)
            ids_txt.extend(sample_ids)

    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds  = torch.cat(all_text_embeds, dim=0)

    # Normalize (project onto unit sphere) before retrieval / clustering /
    # gap / true-pair-distance — every downstream metric assumes unit-norm.
    all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
    all_text_embeds  = all_text_embeds  / all_text_embeds.norm(dim=-1, keepdim=True)

    similarity_matrix = torch.matmul(all_text_embeds.to(device),
                                     all_image_embeds.t().to(device))

    log_forward  = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='forward')
    log_backward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='backward')
    gap = compute_gap(all_image_embeds, all_text_embeds)
    mean_ang_image = compute_mean_angular_value_of_a_modality(all_image_embeds)
    mean_ang_text  = compute_mean_angular_value_of_a_modality(all_text_embeds)
    uniformity_metric = uniformity(all_image_embeds, all_text_embeds)
    mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds)
    clustering_metrics = compute_clustering_metrics(
        all_text_embeds, all_image_embeds, ids_txt)

    n = max(val_loss_acc["n_batches"], 1)
    val_loss_log = {
        "val_loss/anchor":          round(val_loss_acc["anchor"]          / n, 6),
        "val_loss/lalign":          round(val_loss_acc["lalign"]          / n, 6),
        "val_loss/lunif_centroids": round(val_loss_acc["lunif_centroids"] / n, 6),
        # Unweighted total — comparable to train loss/total without the
        # per-step alpha/beta scheduling that would obscure the trend.
        "val_loss/total_unweighted": round(
            (val_loss_acc["anchor"] + val_loss_acc["lalign"]
             + val_loss_acc["lunif_centroids"]) / n, 6),
    }

    final_log = {
        **log_forward,
        **log_backward,
        'gap':                              round(gap, 4),
        'mean_angular_value_image':         round(mean_ang_image, 4),
        'mean_angular_value_text':          round(mean_ang_text, 4),
        'uniformity':                       round(uniformity_metric, 4),
        'mean_cosine_similarity_true_pairs': round(mean_cos_true_pairs, 4),
        **clustering_metrics,
        **val_loss_log,
    }

    if plot_embeddings:
        print("Evaluation Results:", final_log)
        print()

    # Per-epoch embedding snapshot (consumed by visualization.py's curves /
    # histogram plots). `epoch` may be int (epoch_NNN.pt) or str (e.g.
    # "final_full" → final_full.pt).
    if emb_dir is not None and epoch is not None:
        os.makedirs(emb_dir, exist_ok=True)
        if isinstance(epoch, str):
            snap_name   = f"{epoch}.pt"
            epoch_value = epoch
        else:
            snap_name   = f"epoch_{int(epoch):03d}.pt"
            epoch_value = int(epoch)
        torch.save({
            "epoch":        epoch_value,
            "image_embeds": all_image_embeds.detach().cpu(),
            "text_embeds":  all_text_embeds.detach().cpu(),
            "ids_img":      list(ids_img),
            "ids_txt":      list(ids_txt),
            "captions":     all_captions,
            "labels":       np.array([]),  # historical key, kept empty for COCO
            "metrics":      final_log,
        }, os.path.join(emb_dir, snap_name))

    # Wandb keys: prefix by section so the UI auto-groups them. Snapshot keys
    # stay flat so visualization.py readers don't have to know about prefixes.
    _embed_keys = {"gap", "uniformity", "mean_angular_value_image",
                   "mean_angular_value_text", "mean_cosine_similarity_true_pairs"}
    def _wandb_key(k):
        if "/" in k:
            return k  # already prefixed (e.g. val_loss/anchor)
        if k.startswith("forward_") or k.startswith("backward_"):
            return f"retrieval/{k}"
        if k in _embed_keys:
            return f"embedding/{k}"
        return f"clustering/{k}"
    eval_log = {_wandb_key(k): v for k, v in final_log.items()}
    # Log epoch for x-axis alignment with training metrics. `epoch` may be
    # int (per-epoch eval) or str (e.g. "final_full") — only log when numeric.
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        eval_log["epoch"] = float(epoch)
    wandb.log(eval_log)

    model.train()
    return final_log


# ---------------------------------------------------------------------------
# Loss dispatch
# ---------------------------------------------------------------------------

def _compute_loss(loss_type, image_embeds, text_embeds,
                  *, in_warmup, clip_loss,
                  current_batch, t_total, config):
    """Compute the training loss for `loss_type`.

    Returns (loss, named, alpha, beta, total_unweighted) where:
      - `loss` is the (possibly weighted) optimization target.
      - `named` is `{component_name: tensor}` for every term in this branch —
        train_model logs `loss/<name>` for each, regardless of loss_type.
      - alpha, beta are the current schedule weights (0.0 if unused).
      - `total_unweighted` is the sum of the un-multiplied components, so it
        stays comparable across phase boundaries even when alpha/beta change.

    `clip_loss` is the shared `ClipLoss` instance — it owns the temperature
    scalar and computes the symmetric contrastive (anchor) term.
    """
    # All "only_lunif_n_then_*" types share the same warmup objective.
    if in_warmup:
        lunif_img = lunif_loss(image_embeds)
        lunif_txt = lunif_loss(text_embeds)
        loss = (lunif_img + lunif_txt) / 2
        named = {"lunif_img": lunif_img, "lunif_txt": lunif_txt}
        return loss, named, 0.0, 0.0, loss

    alpha, beta = 0.0, 0.0

    if loss_type == LOSS_ANCHOR:
        anchor = clip_loss(image_embeds, text_embeds)
        loss = anchor
        named = {"anchor": anchor}
        total_unweighted = anchor

    elif loss_type == LOSS_LUNIF_THEN_FULL:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif  = lunif_modality(image_embeds, text_embeds)
        loss = anchor + lunif + lalign
        named = {"anchor": anchor, "lalign": lalign, "lunif_modality": lunif}
        total_unweighted = anchor + lalign + lunif

    elif loss_type == LOSS_LUNIF_THEN_CENT:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif_cent = lunif_centroid(image_embeds, text_embeds)
        loss = anchor + config["lambda1"] * lalign + config["lambda2"] * lunif_cent
        named = {"anchor": anchor, "lalign": lalign, "lunif_centroids": lunif_cent}
        total_unweighted = anchor + lalign + lunif_cent

    elif loss_type == LOSS_LUNIF_THEN_BLUNIF:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif  = lunif_modality(image_embeds, text_embeds)
        beta = get_beta(current_batch, t_total,
                        config["beta_warmup_epoch"], config["beta_decay_epoch"])
        loss = anchor + lalign + beta * lunif
        named = {"anchor": anchor, "lalign": lalign, "lunif_modality": lunif}
        total_unweighted = anchor + lalign + lunif

    elif loss_type == LOSS_LUNIF_THEN_AB_FULL:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif  = lunif_modality(image_embeds, text_embeds)
        alpha = get_alpha(current_batch, t_total,
                          config["alpha_warmup_epoch"], config["alpha_increment_epoch"])
        beta  = get_beta(current_batch, t_total,
                         config["beta_warmup_epoch"], config["beta_decay_epoch"])
        loss = anchor + alpha * lalign + beta * lunif
        named = {"anchor": anchor, "lalign": lalign, "lunif_modality": lunif}
        total_unweighted = anchor + lalign + lunif

    elif loss_type == LOSS_LUNIF_THEN_AB_CENT:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif_cent = lunif_centroid(image_embeds, text_embeds)
        alpha = get_alpha(current_batch, t_total,
                          config["alpha_warmup_epoch"], config["alpha_increment_epoch"])
        beta  = get_beta(current_batch, t_total,
                         config["beta_warmup_epoch"], config["beta_decay_epoch"])
        loss = anchor + alpha * lalign + beta * lunif_cent
        named = {"anchor": anchor, "lalign": lalign, "lunif_centroids": lunif_cent}
        total_unweighted = anchor + lalign + lunif_cent

    elif loss_type == LOSS_ABL_FULL:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif_cent = lunif_centroid(image_embeds, text_embeds)
        loss = anchor + lalign + lunif_cent
        named = {"anchor": anchor, "lalign": lalign, "lunif_centroids": lunif_cent}
        total_unweighted = loss

    elif loss_type == LOSS_ABL_NO_UNIF:
        anchor = clip_loss(image_embeds, text_embeds)
        lalign = lalign_loss(image_embeds, text_embeds)
        loss = anchor + lalign
        named = {"anchor": anchor, "lalign": lalign}
        total_unweighted = loss

    elif loss_type == LOSS_ABL_NO_ALIGN:
        anchor = clip_loss(image_embeds, text_embeds)
        lunif_cent = lunif_centroid(image_embeds, text_embeds)
        loss = anchor + lunif_cent
        named = {"anchor": anchor, "lunif_centroids": lunif_cent}
        total_unweighted = loss

    elif loss_type == LOSS_LALIGN_LUNIF:
        # Despite the name, this branch has NO anchor and NO warmup phase.
        lalign = lalign_loss(image_embeds, text_embeds)
        lunif_cent = lunif_centroid(image_embeds, text_embeds)
        loss = lalign + lunif_cent
        named = {"lalign": lalign, "lunif_centroids": lunif_cent}
        total_unweighted = loss

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss, named, alpha, beta, total_unweighted


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(config, train_loader, test_loader, device,
                ckpt_dir=None, emb_dir=None):
    # Build OpenCLIP from scratch (no pretrained weights).
    model, _, _ = open_clip.create_model_and_transforms(
        config["model"], pretrained=None, device=device,
    )
    tokenizer = open_clip.get_tokenizer(config["model"])

    model.train()
    for p in model.parameters():
        p.requires_grad = True
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # ClipLoss owns the temperature scalar (Parameter when learnable, buffer
    # otherwise) and is the single source for the anchor (contrastive) term
    # across every loss_type branch.
    clip_loss = ClipLoss(
        temperature=config["anchor_temperature"],
        learnable=config["anchor_temperature_learnable"],
    ).to(device)
    if config["anchor_temperature_learnable"]:
        print("Using learnable temperature parameter")

    start_epoch = 0
    if config["resume_checkpoint"]:
        print(f"Resuming training from {config['resume_checkpoint']} at epoch {config['resume_epoch']}")
        model.load_state_dict(torch.load(config["resume_checkpoint"]))
        start_epoch = config["resume_epoch"]

    # Optimizer sees model params plus clip_loss params (the latter is empty
    # unless temperature is learnable).
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(clip_loss.parameters()),
        lr=config["learning_rate"],
    )

    epochs = config["epochs"]
    t_total = len(train_loader) * epochs
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=t_total, config=config,
    )

    # epoch 0 checkpoint — the "init" state for the paper-style init/CLIP/Ours
    # PCA comparison.
    if ckpt_dir is not None:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "epoch_000.pt"))

    print("Evaluating model before training...")
    evaluate_model(model, test_loader, device, model_name=config["model"],
                   config=config, epoch=0, emb_dir=emb_dir, clip_loss=clip_loss)

    # Save at every-N epochs PLUS the alpha/beta phase-transition boundaries.
    phase_epochs = {
        config["beta_warmup_epoch"],
        config["beta_warmup_epoch"] + config["beta_decay_epoch"],
        config["alpha_warmup_epoch"],
        config["alpha_warmup_epoch"] + config["alpha_increment_epoch"],
    }
    save_every_n   = config["save_checkpoint_every_n_epochs"]
    grad_log_every = int(config.get("grad_log_every_n_steps", 100))

    # Precision resolution: prefer new `precision: fp32|fp16|bf16`; fall back
    # to legacy `fp16: bool` for older config files.
    precision = config.get("precision")
    if precision is None:
        precision = "fp16" if config.get("fp16", False) else "fp32"
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown precision: {precision!r} (expected fp32|fp16|bf16)")

    use_amp   = (precision != "fp32") and device.type == "cuda"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
    # bf16 has fp32-equivalent dynamic range, so no GradScaler needed; only
    # fp16 risks underflow and benefits from loss scaling.
    scaler    = torch.amp.GradScaler('cuda', enabled=(precision == "fp16" and use_amp))
    print(f"Mixed precision: {precision}{' (AMP)' if use_amp else ' (no AMP)'}")

    loss_type         = config["loss_type"]
    only_lunif_epochs = config["only_lunif_epochs"]
    has_warmup_phase  = loss_type in _WARMUP_LOSS_TYPES

    current_batch = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        in_warmup = has_warmup_phase and (epoch < only_lunif_epochs)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, captions, _sample_ids in pbar:
            current_batch += 1
            images = images.to(device)

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                text_tokens  = tokenizer(captions).to(device)
                image_embeds = model.module.encode_image(images)
                text_embeds  = model.module.encode_text(text_tokens)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)

                loss, named_tensors, alpha, beta, total_unweighted = _compute_loss(
                    loss_type, image_embeds, text_embeds,
                    in_warmup=in_warmup, clip_loss=clip_loss,
                    current_batch=current_batch, t_total=t_total, config=config,
                )

            # Per-component values (cheap, every step, every loss_type).
            extra_log = {f"loss/{k}": v.item() for k, v in named_tensors.items()}

            # Per-component grad norms (expensive — one extra backward per
            # term). Only EXP 10 main-phase enables this.
            should_grad_log = (
                loss_type == LOSS_LUNIF_THEN_AB_CENT
                and not in_warmup
                and grad_log_every > 0
                and (current_batch == 1 or current_batch % grad_log_every == 0)
            )
            if should_grad_log:
                grad_params = [p for p in model.module.parameters() if p.requires_grad]
                gn = per_loss_grad_norms(named_tensors, grad_params)
                extra_log.update({f"grad_norm/{k}": v for k, v in gn.items()})

            payload = {
                "loss/total":            loss.item(),             # weighted (optimization target)
                "loss/total_unweighted": total_unweighted.item(), # comparable across α/β phases
                "optim/learning_rate":   scheduler.get_last_lr()[0],
                "optim/alpha":           alpha,
                "optim/beta":            beta,
                # Fractional epoch — usable as wandb x-axis to compare runs
                # with different batch sizes (step count differs, epoch doesn't).
                "epoch":                 start_epoch + current_batch / len(train_loader),
            }
            if config["anchor_temperature_learnable"]:
                payload["optim/temperature"] = clip_loss.temperature.item()
            payload.update(extra_log)
            wandb.log(payload)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        evaluate_model(model, test_loader, device, model_name=config["model"],
                       config=config, epoch=epoch + 1, emb_dir=emb_dir, clip_loss=clip_loss)

        # Save checkpoint at every-N epochs OR at any α/β phase boundary.
        epoch_one_indexed = epoch + 1
        should_save = ((epoch_one_indexed % save_every_n == 0)
                       or (epoch_one_indexed in phase_epochs))
        if should_save and ckpt_dir is not None:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"epoch_{epoch_one_indexed:03d}.pt"))
            print(f"Model saved at epoch {epoch_one_indexed}")

    return model, clip_loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config):
    # Resolve run_name. If left as null/empty/"auto" in config.yaml, generate
    # `{model}_{loss-tag}_{relevant-hparams}_{YYYYMMDD-HHMMSS}` so each launch
    # gets a unique, self-describing directory under runs/ without manual edits.
    if config.get("run_name") in (None, "", "auto"):
        config["run_name"] = build_auto_run_name(config)
        print(f"Auto-generated run_name: {config['run_name']}")

    wandb.init(project=config["project_name"], config=config, name=config['run_name'])
    set_seed(config["seed"])
    print("Config:", config)
    print("Experiment:", config["run_name"])

    device = torch.device(f"cuda:{config['device_id']}" if torch.cuda.is_available() else "cpu")

    run_dir, ckpt_dir, emb_dir = setup_run_dir(config["run_name"])
    print(f"Run artifacts will be saved under: {run_dir}")

    # Authoritative snapshot of "what was actually launched" — written BEFORE
    # training so a crashed run still leaves a trail. visualization.py reads
    # this in priority over the project-root config.yaml (which may have
    # drifted since launch).
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\nLoading the COCO dataset...")
    train_loader, test_loader = get_coco_dataloaders(config)
    print("Dataset loaded.\n")

    print("Training the model...")
    model, clip_loss = train_model(config, train_loader, test_loader, device,
                                   ckpt_dir=ckpt_dir, emb_dir=emb_dir)
    print("Training complete.\n")

    # Final evaluation on the FULL validation set (overrides num_test_samples).
    # Snapshot is written as embeddings/final_full.pt.
    print("Final evaluation on the FULL validation set...")
    full_config = dict(config)
    full_config["num_test_samples"] = -1
    _, full_test_loader = get_coco_dataloaders(full_config)

    final_log = evaluate_model(
        model, full_test_loader, device,
        model_name=config["model"],
        config=config,
        clip_loss=clip_loss,
        epoch="final_full",
        emb_dir=emb_dir,
    )
    print("Evaluation complete.\n")
    print("Final evaluation results:", final_log)

    final_metrics_payload = {
        "run_name":             config["run_name"],
        "config":               config,
        "num_val_samples_used": "full",
        "metrics":              final_log,
    }
    final_metrics_path = os.path.join(run_dir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"Final metrics saved to {final_metrics_path}")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the experiment with a config.yaml file")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the yaml config file")
    parser.add_argument("--device", type=int, required=True, help="GPU id to use")
    args = parser.parse_args()

    if os.path.isfile(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        config["device_id"]     = args.device
        config["learning_rate"] = float(config["learning_rate"])
        main(config)
