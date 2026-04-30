#!/usr/bin/env python
# coding: utf-8
"""Post-hoc visualization for ModalityGap runs.

Reads embedding snapshots dumped by `evaluate_model` (one per epoch) under
`runs/{run_name}/embeddings/`, generates a set of figures, and writes them to
`runs/{run_name}/figures/`.

Usage:
    python visualization.py --run_name <name>
    python visualization.py --run_name <name> --plots curves pca histogram
    python visualization.py --run_name <name> --pca_epochs 0 50 100
    python visualization.py --run_name <name> --hist_epochs 0 100
"""

import argparse
import glob
import os
import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT = os.path.join(PROJECT_ROOT, 'runs')


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def load_snapshots(emb_dir: str) -> List[dict]:
    paths = sorted(glob.glob(os.path.join(emb_dir, "epoch_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No epoch_*.pt found under {emb_dir}")
    snaps = []
    for p in paths:
        snap = torch.load(p, map_location='cpu', weights_only=False)
        snaps.append(snap)
    snaps.sort(key=lambda s: s['epoch'])
    return snaps


def select_snapshots(snapshots: List[dict], epochs: List[int]) -> List[dict]:
    by_epoch = {s['epoch']: s for s in snapshots}
    picked = []
    for e in epochs:
        if e in by_epoch:
            picked.append(by_epoch[e])
        else:
            warnings.warn(f"epoch {e} not found in snapshots; available: {sorted(by_epoch.keys())}")
    return picked


# ----------------------------------------------------------------------------
# Plot 1: training curves (auto-discover all numeric metrics)
# ----------------------------------------------------------------------------

PRIORITY_KEYS = [
    "gap",
    "mean_cosine_similarity_true_pairs",
    "forward_r1",
    "backward_r1",
    "V-measure",
    "K-NN Acc",
    "uniformity",
    "mean_angular_value_image",
    "mean_angular_value_text",
]


def plot_training_curves(snapshots: List[dict], fig_dir: str):
    epochs = [s['epoch'] for s in snapshots]
    # Discover numeric keys present in at least one snapshot
    numeric_keys = set()
    for s in snapshots:
        for k, v in s.get('metrics', {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_keys.add(k)

    ordered = [k for k in PRIORITY_KEYS if k in numeric_keys]
    ordered += sorted(numeric_keys - set(ordered))

    n = len(ordered)
    if n == 0:
        warnings.warn("No numeric metrics found in snapshots; skipping training curves.")
        return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), squeeze=False)

    for i, key in enumerate(ordered):
        ax = axes[i // cols][i % cols]
        vals = [s['metrics'].get(key, np.nan) for s in snapshots]
        ax.plot(epochs, vals, marker='o', linewidth=1.5, markersize=4)
        ax.set_title(key)
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n, rows * cols):
        axes[i // cols][i % cols].axis('off')

    fig.suptitle('Training curves (all logged metrics)', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Plot 2: PCA latent space — runs FRESH inference from saved checkpoints.
# Decoupled from the per-epoch embedding snapshots, so the user can pick any
# `--pca_num_samples` after training has finished.
# ----------------------------------------------------------------------------

def _load_run_config(run_dir: str) -> dict:
    """Resolve the config used for this run, preferring sources in this order:
      1. runs/<name>/run_config.json   (saved at training START — most reliable
         even if the run crashed mid-flight)
      2. runs/<name>/final_metrics.json's "config" field  (saved at training
         END only — same content as run_config.json on successful runs)
      3. project-root config.yaml      (last-resort fallback; risky because
         it may have been edited since this run was launched)
    """
    import json
    run_cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(run_cfg_path):
        with open(run_cfg_path) as f:
            return json.load(f)

    metrics_path = os.path.join(run_dir, "final_metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            cfg = json.load(f).get("config", {})
            if cfg:
                return cfg

    fallback = os.path.join(PROJECT_ROOT, "config.yaml")
    if os.path.isfile(fallback):
        import yaml
        with open(fallback) as f:
            warnings.warn(f"No run_config.json/final_metrics.json under "
                          f"{run_dir}; falling back to project-root config.yaml "
                          f"— may not match the actual training config.")
            return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No run_config.json or final_metrics.json under {run_dir} "
        f"and no config.yaml at project root.")


def _list_available_checkpoint_epochs(ckpt_dir: str) -> List[int]:
    paths = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
    out = []
    for p in paths:
        try:
            out.append(int(os.path.basename(p).split("_")[1].split(".")[0]))
        except Exception:
            pass
    return sorted(out)


def _build_test_loader(config: dict, num_samples: int):
    """Build a fresh test loader with `num_samples` samples.

    Built inline (rather than reused from main.py's `get_coco_dataloaders`) so
    we can set `drop_last=False`. The training pipeline's loader uses
    `drop_last=True` which silently drops the last partial batch — fine when
    `num_samples >> batch_size`, but yields zero batches when `num_samples`
    is small (e.g. 200 < 256).
    """
    import random as _random
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    mean = [0.48145466, 0.4578275, 0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]

    if config.get("dataset") != "coco":
        # Defer to main.py for non-coco datasets (cifar10).
        import sys
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from main import get_cifar10_dataloaders
        cfg = dict(config); cfg["num_test_samples"] = int(num_samples)
        cfg.setdefault("device_id", 0)
        _, test_loader = get_cifar10_dataloaders(cfg)
        return test_loader

    test_image_dir = os.path.join(DATA_ROOT, "coco/images/val2017/")
    test_annotation_file = os.path.join(DATA_ROOT, "coco/annotations/captions_val2017.json")
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    class _CocoCaptionsWithIDs(dset.CocoCaptions):
        def __getitem__(self, index):
            image, captions = super().__getitem__(index)
            return image, captions, self.ids[index]

    test_coco = _CocoCaptionsWithIDs(
        root=test_image_dir,
        annFile=test_annotation_file,
        transform=test_transform,
    )
    if num_samples is not None and num_samples > 0:
        test_coco = Subset(test_coco, list(range(min(num_samples, len(test_coco)))))

    def collate_fn(batch):
        images, captions, sample_ids = zip(*batch)
        images = torch.stack(images, 0)
        sel = [_random.choice(c) for c in captions]
        return images, sel, sample_ids

    batch_size = min(int(config.get("batch_size", 256)), max(1, len(test_coco)))
    return DataLoader(test_coco, batch_size=batch_size, shuffle=False,
                      drop_last=False, collate_fn=collate_fn, num_workers=0)


def _build_model_with_weights(model_name: str, ckpt_path: str, device: str):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=None, device=device)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model


# Class-name lookup for image-only datasets that surface labels-as-ints in the
# loader's "captions" slot. Mirrors main.py's evaluate_model conversion.
_CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']


@torch.no_grad()
def _extract_embeddings(model, test_loader, model_name: str, device: str):
    """Returns (img_embeds, txt_embeds, sample_ids, raw_labels) where
    raw_labels is the un-converted second tuple element from the loader —
    list[str] for COCO captions, list[int] for CIFAR-10 class labels.
    Used by `plot_pca_latent_space_class` to color CIFAR-10 points without
    re-iterating the loader."""
    import open_clip
    tokenizer = open_clip.get_tokenizer(model_name)
    img_chunks, txt_chunks, ids, raw_labels = [], [], [], []
    for images, captions, sample_ids in test_loader:
        images = images.to(device)
        raw_labels.extend(list(captions))
        # Some loaders (CIFAR-10) put int class labels here; tokenizer needs
        # strings, so look up the class name first.
        if captions and (isinstance(captions[0], int)
                          or torch.is_tensor(captions[0])):
            captions = [_CIFAR10_CLASSES[int(lbl)] for lbl in captions]
        text_tokens = tokenizer(list(captions)).to(device)
        ie = model.module.encode_image(images)
        te = model.module.encode_text(text_tokens)
        ie = ie / ie.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        img_chunks.append(ie.cpu())
        txt_chunks.append(te.cpu())
        ids.extend(list(sample_ids))
    return (torch.cat(img_chunks, dim=0).numpy(),
            torch.cat(txt_chunks, dim=0).numpy(),
            ids,
            raw_labels)


def plot_pca_latent_space(run_name: str, fig_dir: str,
                          epochs: List[int] = None,
                          num_samples: int = 1000,
                          device: str = "cuda:0"):
    run_dir = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        warnings.warn(f"No checkpoints dir at {ckpt_dir}; skipping PCA plot.")
        return

    config = _load_run_config(run_dir)
    available = _list_available_checkpoint_epochs(ckpt_dir)
    if not available:
        warnings.warn(f"No epoch_*.pt under {ckpt_dir}; skipping PCA plot.")
        return

    # Default selection: init, ~1/3, ~2/3, final from available checkpoints.
    if epochs is None:
        e_max = max(available)
        targets = [0, e_max // 3, 2 * e_max // 3, e_max]
        # Snap each target to the nearest available checkpoint.
        epochs = sorted({min(available, key=lambda a: abs(a - t)) for t in targets})
    else:
        missing = [e for e in epochs if e not in available]
        for e in missing:
            warnings.warn(f"epoch {e} checkpoint not found; "
                          f"available: {available}")
        epochs = [e for e in epochs if e in available]
    if not epochs:
        warnings.warn("No usable epochs after filtering; skipping PCA plot.")
        return

    print(f"PCA plot: fresh inference for epochs {epochs} on {num_samples} samples")

    # Reproducible caption sampling across the whole run.
    seed = int(config.get("seed", 0))
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    test_loader = _build_test_loader(config, num_samples)

    # Run inference per epoch. Reuse the same loader (so the same images +
    # captions feed every checkpoint, making panels directly comparable).
    embeds = {}
    for e in epochs:
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{e:03d}.pt")
        print(f"  loading {os.path.basename(ckpt_path)} ...")
        model = _build_model_with_weights(config["model"], ckpt_path, device)
        # Re-seed before each loader pass so caption sampling is deterministic
        # AND identical across checkpoints.
        _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        img, txt, _, _ = _extract_embeddings(model, test_loader, config["model"], device)
        gap = float(np.linalg.norm(img.mean(0) - txt.mean(0)))
        embeds[e] = (img, txt, gap)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- plotting (same 3 × N grid as before) -----
    n = len(epochs)
    views = [
        ("perspective",     dict(elev=20, azim=-60)),
        ("front (PC2-PC3)", dict(elev=0,  azim=-90)),
        ("top (PC1-PC2)",   dict(elev=90, azim=-90)),
    ]
    n_views = len(views)
    fig, axes = plt.subplots(n_views, n, figsize=(4.5 * n, 4.5 * n_views),
                             squeeze=False, subplot_kw={'projection': '3d'})

    _u = np.linspace(0, 2 * np.pi, 30)
    _v = np.linspace(0, np.pi, 20)
    _sx = np.outer(np.cos(_u), np.sin(_v))
    _sy = np.outer(np.sin(_u), np.sin(_v))
    _sz = np.outer(np.ones_like(_u), np.cos(_v))

    for col, e in enumerate(epochs):
        img, txt, gap = embeds[e]
        combined = np.concatenate([img, txt], axis=0)
        proj = PCA(n_components=3).fit_transform(combined)
        proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        n_img = len(img)

        for row, (view_name, view_kwargs) in enumerate(views):
            ax = axes[row][col]
            ax.plot_wireframe(_sx, _sy, _sz, color='gray', alpha=0.15, linewidth=0.4)
            ax.scatter(proj[:n_img, 0], proj[:n_img, 1], proj[:n_img, 2],
                       c='tab:blue', alpha=0.5, s=10, label='image')
            ax.scatter(proj[n_img:, 0], proj[n_img:, 1], proj[n_img:, 2],
                       c='tab:orange', alpha=0.5, s=10, label='text')
            ax.view_init(**view_kwargs)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_zlim(-1.05, 1.05)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('PC1', fontsize=7, labelpad=-4)
            ax.set_ylabel('PC2', fontsize=7, labelpad=-4)
            ax.set_zlabel('PC3', fontsize=7, labelpad=-4)
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(f"epoch {e} — gap={gap:.3f}", fontsize=10)
            if col == 0:
                ax.text2D(-0.18, 0.5, view_name, transform=ax.transAxes,
                          rotation=90, ha='center', va='center',
                          fontsize=10, fontweight='bold')
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=7)

    fig.suptitle(
        f'PCA (3D) of image vs text embeddings — projected to unit sphere '
        f'(N={img.shape[0]} per modality)', y=1.00)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "pca_latent_space.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Plot 2b: PCA latent space colored by class (single-object COCO filter).
# Markers follow TempModGap convention: text=*, image=s. Colors are class-id
# via tab20 / hsv depending on the number of unique classes that survive the
# single-object filter.
# ----------------------------------------------------------------------------

# Marker conventions borrowed from TempModGap/avmnist/visualize.py
_MOD_MARKERS = {"image": "s", "text": "*"}
_MOD_SIZES   = {"image": 32,  "text":  46}
_MOD_LABELS  = {"image": "Vision (image)", "text": "Text (caption)"}


def _filter_single_object_coco(image_embeds: np.ndarray,
                               text_embeds: np.ndarray,
                               sample_ids: List[int]):
    """Apply the same per-image filter used by `compute_clustering_metrics`:
    keep only val2017 images whose `instances_val2017.json` annotations contain
    exactly one distinct category. Returns
        (filtered_img, filtered_txt, labels_int, label_names)
    where labels_int is in [0, K-1] and label_names lists the K COCO category
    names corresponding to those ints.
    """
    import contextlib
    import io
    DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
    instances_path = os.path.join(DATA_ROOT, "coco/annotations/instances_val2017.json")
    if not os.path.isfile(instances_path):
        raise FileNotFoundError(f"COCO instances file missing: {instances_path}")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        from pycocotools.coco import COCO
        coco = COCO(instances_path)

    cat_lookup = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    keep_idx = []
    keep_cat = []
    for i, sid in enumerate(sample_ids):
        anns = coco.loadAnns(coco.getAnnIds(imgIds=int(sid)))
        cats = {a["category_id"] for a in anns}
        if len(cats) == 1:
            keep_idx.append(i)
            keep_cat.append(next(iter(cats)))
    if not keep_idx:
        return None
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    unique_cats = sorted(set(keep_cat))
    cat_to_int = {c: i for i, c in enumerate(unique_cats)}
    labels_int = np.asarray([cat_to_int[c] for c in keep_cat], dtype=np.int64)
    names = [cat_lookup[c] for c in unique_cats]
    return (image_embeds[keep_idx], text_embeds[keep_idx], labels_int, names)


def _make_class_palette(num_classes: int) -> np.ndarray:
    """Return an (num_classes, 4) RGBA palette. tab20 up to 20 classes, else hsv."""
    if num_classes <= 10:
        return plt.get_cmap("tab10")(np.arange(num_classes))
    if num_classes <= 20:
        return plt.get_cmap("tab20")(np.arange(num_classes))
    return plt.get_cmap("hsv")(np.linspace(0, 1, num_classes, endpoint=False))


def plot_pca_latent_space_class(run_name: str, fig_dir: str,
                                epochs: List[int] = None,
                                num_samples: int = 1000,
                                device: str = "cuda:0"):
    """Variant of `plot_pca_latent_space` that colors points by class.

    Class labels come from the COCO `instances_val2017.json` filter used by the
    clustering eval: each kept image has exactly ONE distinct object category;
    text and image inherit that category. Markers distinguish modality
    (image=square, text=star). Layout is 3 views × N selected epochs.
    """
    run_dir = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        warnings.warn(f"No checkpoints dir at {ckpt_dir}; skipping.")
        return

    config = _load_run_config(run_dir)
    dataset = config.get("dataset")
    if dataset not in ("coco", "cifar10"):
        warnings.warn(f"plot_pca_latent_space_class supports COCO and CIFAR-10; "
                      f"got dataset={dataset}. Skipping.")
        return
    available = _list_available_checkpoint_epochs(ckpt_dir)
    if not available:
        warnings.warn(f"No epoch_*.pt under {ckpt_dir}; skipping.")
        return

    if epochs is None:
        e_max = max(available)
        targets = [0, e_max // 3, 2 * e_max // 3, e_max]
        epochs = sorted({min(available, key=lambda a: abs(a - t)) for t in targets})
    else:
        missing = [e for e in epochs if e not in available]
        for e in missing:
            warnings.warn(f"epoch {e} checkpoint not found; available: {available}")
        epochs = [e for e in epochs if e in available]
    if not epochs:
        warnings.warn("No usable epochs; skipping.")
        return

    print(f"PCA(class) plot: fresh inference for epochs {epochs} on {num_samples} samples")
    seed = int(config.get("seed", 0))
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    test_loader = _build_test_loader(config, num_samples)

    # Run inference once per epoch. The loader is fixed across epochs, so
    # sample_ids and raw_labels are identical too — only embeddings change.
    raw_embeds = {}  # epoch -> (img, txt, sample_ids, raw_labels)
    for e in epochs:
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{e:03d}.pt")
        print(f"  loading {os.path.basename(ckpt_path)} ...")
        model = _build_model_with_weights(config["model"], ckpt_path, device)
        _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        img, txt, sids, raw = _extract_embeddings(
            model, test_loader, config["model"], device)
        raw_embeds[e] = (img, txt, sids, raw)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _, _, sids0, raw0 = next(iter(raw_embeds.values()))

    if dataset == "coco":
        # COCO: filter to single-object images via instances_val2017.json.
        res = _filter_single_object_coco(raw_embeds[epochs[0]][0],
                                         raw_embeds[epochs[0]][1], sids0)
        if res is None:
            warnings.warn("No single-object images survived the filter; skipping.")
            return
        _, _, labels_int, label_names = res
        # Re-derive the keep_idx (in original order) so we can apply it to
        # every epoch's embeddings.
        import contextlib, io as _io
        with contextlib.redirect_stdout(_io.StringIO()):
            from pycocotools.coco import COCO
            coco = COCO(os.path.join(PROJECT_ROOT, "data",
                                     "coco/annotations/instances_val2017.json"))
        keep_idx_list = []
        for i, sid in enumerate(sids0):
            anns = coco.loadAnns(coco.getAnnIds(imgIds=int(sid)))
            cats = {a["category_id"] for a in anns}
            if len(cats) == 1:
                keep_idx_list.append(i)
        keep_idx = np.asarray(keep_idx_list, dtype=np.int64)
        print(f"  single-object filter kept {len(keep_idx)} / {len(sids0)} "
              f"images across {len(label_names)} classes")
    else:  # cifar10
        # CIFAR-10 already has clean class labels per sample; no filter needed.
        labels_int = np.asarray([int(c) for c in raw0], dtype=np.int64)
        label_names = list(_CIFAR10_CLASSES)
        keep_idx = np.arange(len(labels_int), dtype=np.int64)
        print(f"  cifar10: using all {len(labels_int)} samples "
              f"across {len(label_names)} classes")

    n_classes = len(label_names)
    palette = _make_class_palette(n_classes)

    filt_embeds = {e: (raw_embeds[e][0][keep_idx], raw_embeds[e][1][keep_idx])
                   for e in epochs}

    # ----- plotting -----
    n = len(epochs)
    views = [
        ("perspective",     dict(elev=20, azim=-60)),
        ("front (PC2-PC3)", dict(elev=0,  azim=-90)),
        ("top (PC1-PC2)",   dict(elev=90, azim=-90)),
    ]
    n_views = len(views)
    fig, axes = plt.subplots(n_views, n, figsize=(4.5 * n, 4.5 * n_views),
                             squeeze=False, subplot_kw={'projection': '3d'})

    _u = np.linspace(0, 2 * np.pi, 30)
    _v = np.linspace(0, np.pi, 20)
    _sx = np.outer(np.cos(_u), np.sin(_v))
    _sy = np.outer(np.sin(_u), np.sin(_v))
    _sz = np.outer(np.ones_like(_u), np.cos(_v))

    for col, e in enumerate(epochs):
        img_f, txt_f = filt_embeds[e]
        gap = float(np.linalg.norm(img_f.mean(0) - txt_f.mean(0)))
        combined = np.concatenate([img_f, txt_f], axis=0)
        proj = PCA(n_components=3).fit_transform(combined)
        proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12)
        n_img = len(img_f)
        proj_img = proj[:n_img]
        proj_txt = proj[n_img:]

        for row, (view_name, view_kwargs) in enumerate(views):
            ax = axes[row][col]
            ax.plot_wireframe(_sx, _sy, _sz, color='gray', alpha=0.15, linewidth=0.4)

            face = palette[labels_int].copy()
            edge = palette[labels_int].copy()
            face[:, 3] = 0.55; edge[:, 3] = 0.85
            ax.scatter(proj_img[:, 0], proj_img[:, 1], proj_img[:, 2],
                       facecolors=face, edgecolors=edge,
                       marker=_MOD_MARKERS["image"], s=_MOD_SIZES["image"],
                       linewidths=0.6)
            ax.scatter(proj_txt[:, 0], proj_txt[:, 1], proj_txt[:, 2],
                       facecolors=face, edgecolors=edge,
                       marker=_MOD_MARKERS["text"], s=_MOD_SIZES["text"],
                       linewidths=0.6)
            ax.view_init(**view_kwargs)
            ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-1.05, 1.05)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel('PC1', fontsize=7, labelpad=-4)
            ax.set_ylabel('PC2', fontsize=7, labelpad=-4)
            ax.set_zlabel('PC3', fontsize=7, labelpad=-4)
            ax.tick_params(labelsize=6)

            if row == 0:
                ax.set_title(f"epoch {e} — gap={gap:.3f}", fontsize=10)
            if col == 0:
                ax.text2D(-0.18, 0.5, view_name, transform=ax.transAxes,
                          rotation=90, ha='center', va='center',
                          fontsize=10, fontweight='bold')

    # Legend (modality markers + class colors) at the bottom of the figure.
    from matplotlib.lines import Line2D
    mod_handles = [
        Line2D([0], [0], marker=_MOD_MARKERS["text"], color="none",
               markerfacecolor="white", markeredgecolor="0.35",
               markersize=11, linestyle="None", label=_MOD_LABELS["text"]),
        Line2D([0], [0], marker=_MOD_MARKERS["image"], color="none",
               markerfacecolor="white", markeredgecolor="0.35",
               markersize=9,  linestyle="None", label=_MOD_LABELS["image"]),
    ]
    cls_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=palette[i], markeredgecolor=palette[i],
               markersize=8, linestyle="None", label=label_names[i])
        for i in range(n_classes)
    ]
    fig.legend(handles=mod_handles + cls_handles, loc="lower center",
               ncol=min(8, len(mod_handles) + n_classes),
               frameon=True, bbox_to_anchor=(0.5, -0.01),
               fontsize=8, handlelength=1.0, columnspacing=1.0,
               handletextpad=0.4, borderpad=0.6)

    fig.suptitle(
        f'PCA (3D) by class — image vs text on unit sphere '
        f'(N_kept={len(labels_int)}, classes={n_classes})', y=1.00)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "pca_latent_space_class.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Plot 3: pair-distance histograms (true-pair vs random-pair cosine similarity)
# ----------------------------------------------------------------------------

def plot_pair_distance_histogram(snapshots: List[dict], fig_dir: str,
                                 epochs: List[int] = None,
                                 n_random: int = 5000):
    if not snapshots:
        return
    if epochs is None:
        all_e = [s['epoch'] for s in snapshots]
        # init + final
        epochs = sorted({0, max(all_e)})
    selected = select_snapshots(snapshots, epochs)
    if not selected:
        return

    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    rng = np.random.default_rng(0)

    for i, snap in enumerate(selected):
        ax = axes[0][i]
        img = snap['image_embeds']
        txt = snap['text_embeds']
        # Inputs are unit-normalized at dump time
        N = img.shape[0]
        # True pair cosine similarity (diagonal)
        true_cos = (img * txt).sum(dim=-1).numpy()

        # Random pair cosine similarity (non-diagonal)
        idx_a = rng.integers(0, N, size=n_random)
        idx_b = rng.integers(0, N, size=n_random)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        rand_cos = (img[idx_a] * txt[idx_b]).sum(dim=-1).numpy()

        bins = np.linspace(-1, 1, 60)
        ax.hist(rand_cos, bins=bins, alpha=0.5, label=f'random (n={len(rand_cos)})',
                color='tab:gray', density=True)
        ax.hist(true_cos, bins=bins, alpha=0.7, label=f'true pair (n={N})',
                color='tab:red', density=True)
        ax.axvline(true_cos.mean(), color='tab:red', linestyle='--', linewidth=1,
                   label=f'true mean={true_cos.mean():.3f}')
        ax.axvline(rand_cos.mean(), color='tab:gray', linestyle='--', linewidth=1,
                   label=f'rand mean={rand_cos.mean():.3f}')
        gap = snap.get('metrics', {}).get('gap')
        title = f"epoch {snap['epoch']}"
        if gap is not None:
            title += f" — gap={gap:.3f}"
        ax.set_title(title)
        ax.set_xlabel('cosine similarity (image, text)')
        ax.set_ylabel('density')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Pair-wise cosine similarity distribution', y=1.02)
    plt.tight_layout()
    out_path = os.path.join(fig_dir, "pair_distance_histogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

PLOT_REGISTRY = {
    "curves": plot_training_curves,
    "pca": plot_pca_latent_space,
    "pca_class": plot_pca_latent_space_class,
    "histogram": plot_pair_distance_histogram,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_name", type=str, required=True,
                        help="Name of the run under runs/")
    parser.add_argument("--plots", nargs="*", default=list(PLOT_REGISTRY.keys()),
                        choices=list(PLOT_REGISTRY.keys()),
                        help="Which plots to generate (default: all)")
    parser.add_argument("--pca_epochs", nargs="*", type=int, default=None,
                        help="Specific epochs to include in the PCA plot "
                             "(must match available checkpoints)")
    parser.add_argument("--pca_num_samples", type=int, default=1000,
                        help="Number of validation samples to use for PCA "
                             "post-hoc inference (default: 1000)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for PCA inference (default: cuda:0)")
    parser.add_argument("--hist_epochs", nargs="*", type=int, default=None,
                        help="Specific epochs to include in the histogram plot")
    args = parser.parse_args()

    run_dir = os.path.join(RUNS_ROOT, args.run_name)
    emb_dir = os.path.join(run_dir, "embeddings")
    fig_dir = os.path.join(run_dir, "figures")
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")
    os.makedirs(fig_dir, exist_ok=True)

    # Snapshots are needed by curves + histogram, but NOT by the new PCA plot.
    needs_snapshots = any(p in ("curves", "histogram") for p in args.plots)
    snapshots = []
    if needs_snapshots:
        print(f"Loading snapshots from: {emb_dir}")
        snapshots = load_snapshots(emb_dir)
        print(f"Loaded {len(snapshots)} snapshots; epochs: {[s['epoch'] for s in snapshots]}")
    print(f"Output dir: {fig_dir}")
    print()

    for name in args.plots:
        fn = PLOT_REGISTRY[name]
        if name in ("pca", "pca_class"):
            kwargs = {"num_samples": args.pca_num_samples, "device": args.device}
            if args.pca_epochs is not None:
                kwargs["epochs"] = args.pca_epochs
            fn(args.run_name, fig_dir, **kwargs)
        else:
            kwargs = {}
            if name == "histogram" and args.hist_epochs is not None:
                kwargs["epochs"] = args.hist_epochs
            fn(snapshots, fig_dir, **kwargs)


if __name__ == "__main__":
    main()
