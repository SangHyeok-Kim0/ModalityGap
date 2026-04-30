#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import random

import torch 
import json


import torch
import os
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Subset
from typing import List, Dict
from openTSNE import TSNE
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import wandb
import time
import open_clip
import math
import umap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torch.nn as nn



from metrics import *
from losses import *


import warnings
warnings.filterwarnings("ignore", message=".*'force_all_finite' was renamed to 'ensure_all_finite'.*")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
# Per-run artifacts (checkpoints, embeddings, ...) live under RUNS_ROOT/<run_name>/.
# This directory is git-ignored.
RUNS_ROOT = os.path.join(PROJECT_ROOT, 'runs')
os.makedirs(RUNS_ROOT, exist_ok=True)


def setup_run_dir(run_name: str):
    run_dir = os.path.join(RUNS_ROOT, run_name)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    emb_dir = os.path.join(run_dir, 'embeddings')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    return run_dir, ckpt_dir, emb_dir


def per_loss_grad_norms(components: Dict[str, "torch.Tensor"], params):
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



def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, model_name: str, epoch: int = None, emb_dir: str = None, plot_embeddings=True, loss_fn=None) -> Dict[str, float]:
    """
    Evaluate the (OpenCLIP) model on the given test_loader by computing
    text-to-image and image-to-text retrieval metrics, along with additional metrics.

    Args:
        model (torch.nn.Module): The trained (DataParallel) model.
        test_loader (DataLoader): A DataLoader for the evaluation set.
        device (torch.device): The device (CPU or GPU).

    Returns:
        Dict[str, float]: Dictionary containing all evaluation metrics.
    """
    # Put model into eval mode
    model.eval()

    # Prepare storage for embeddings
    all_image_embeds = []
    all_text_embeds  = []
    all_labels       = []

    # IDs for retrieval
    ids_img = []
    ids_txt = []

    current_index = 0
    
    tokenizer = open_clip.get_tokenizer(model_name)
    
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    all_captions = []  # capture the (string) captions actually fed to the encoder

    # No gradient needed during evaluation
    with torch.no_grad():
        for images, captions_list, sample_ids in (tqdm(test_loader, desc="Evaluating") if plot_embeddings else test_loader):
            # Move images to device
            images = images.to(device)
            
            # Convert numerical labels to text class names (only for CIFAR-10)
            if isinstance(captions_list[0], int) or torch.is_tensor(captions_list[0]):
                # Convert numeric label to textual class name
                numeric_labels = captions_list
                captions_list = [CIFAR10_CLASSES[int(lbl)] for lbl in numeric_labels]
                # Store the numeric labels for color-coding
                all_labels.extend(numeric_labels)
            else:
                # For non-CIFAR dataset, all_labels can remain empty or zero-based
                # The code below simply won't color by label if labels is empty.
                numeric_labels = [0]*len(captions_list)  # or skip altogether

            # Tokenize captions
            all_captions.extend(list(captions_list))
            text_tokens = tokenizer(captions_list).to(device)
            
            # Extract embeddings using the .module references in DataParallel
            image_embeds = model.module.encode_image(images)
            text_embeds = model.module.encode_text(text_tokens)

            # Move embeddings to CPU for later concatenation
            image_embeds = image_embeds.cpu()
            text_embeds = text_embeds.cpu()

            # Store embeddings
            all_image_embeds.append(image_embeds)
            all_text_embeds.append(text_embeds)

            # Assign IDs used by retrieval R@K matching.
            # COCO  → unique sample_id per image (1 positive per query).
            # CIFAR → class label, so same-class samples count as positives;
            #         R@K becomes class-level (see compute_metric_ret).
            bs = images.size(0)
            if config["dataset"] == "cifar10":
                cls_ids = [int(lbl) for lbl in numeric_labels]
                ids_img.extend(cls_ids)
                ids_txt.extend(cls_ids)
            else:
                ids_img.extend(sample_ids)
                ids_txt.extend(sample_ids)
            current_index += bs

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)  # Shape: [N, D]
    all_text_embeds = torch.cat(all_text_embeds, dim=0)    # Shape: [N, D]
    all_labels = np.array(all_labels)
    
    # Time taken for UMAP visualization:  7.042090892791748
    # Time taken for TSNE visualization:  51.96275329589844
    # Time taken for PCA visualization:  0.8503968715667725
    
    # If we are working with cifar10, create a true variable

    #all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
    #all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)

    # Compute pairwise similarity: [N_text, N_image]
    if config["loss_type"] == "harmonic":
        text_embedding_exp = all_text_embeds.unsqueeze(1)  # Shape: (bs, 1, 10)
        vision_embedding_exp = all_image_embeds.unsqueeze(0)  # Shape: (1, bs, 10)
        similarity_matrix = -torch.norm( text_embedding_exp.to(device) - vision_embedding_exp.to(device), dim=-1 )#torch.matmul(text_embedding, vision_embedding.permute(1,0))
    else:
            # Normalize embeddings to map the embeddings in a sphere of radius 1
        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
        all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)

        similarity_matrix = torch.matmul(all_text_embeds.to(device), all_image_embeds.t().to(device))


    """SEQUENTIAL COMPUTATION
    # Compute retrieval and additional metrics
    log_forward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='forward')   # Text-to-Vision
    log_backward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='backward') # Vision-to-Text
    gap = compute_gap(all_image_embeds, all_text_embeds)
    mean_ang_image = compute_mean_angular_value_of_a_modality(all_image_embeds)
    mean_ang_text = compute_mean_angular_value_of_a_modality(all_text_embeds)
    uniformity_metric = uniformity(all_image_embeds, all_text_embeds)
    mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds)
    """
    
    def compute_metrics(all_image_embeds, all_text_embeds, similarity_matrix, ids_img, ids_txt):
        if config["loss_type"] == "harmonic":
            mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds, cosine=False)
        else:
            mean_cos_true_pairs = mean_distance_of_true_pairs(all_image_embeds, all_text_embeds)

        all_image_embeds = all_image_embeds / all_image_embeds.norm(dim=-1, keepdim=True)
        all_text_embeds = all_text_embeds / all_text_embeds.norm(dim=-1, keepdim=True)

        log_forward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='forward')
        log_backward = compute_metric_ret(similarity_matrix, ids_img, ids_txt, direction='backward')
        gap = compute_gap(all_image_embeds, all_text_embeds)
        mean_ang_image = compute_mean_angular_value_of_a_modality(all_image_embeds)
        mean_ang_text = compute_mean_angular_value_of_a_modality(all_text_embeds)
        uniformity_metric = uniformity(all_image_embeds, all_text_embeds)

        if config["dataset"] == "cifar10":
            clustering_metrics = compute_clustering_metrics(
                all_text_embeds, all_image_embeds, ids_txt,
                dataset="cifar10", labels=all_labels)
        else:
            clustering_metrics = compute_clustering_metrics(
                all_text_embeds, all_image_embeds, ids_txt, dataset="coco")

        return log_forward, log_backward, gap, mean_ang_image, mean_ang_text, uniformity_metric, mean_cos_true_pairs, clustering_metrics

    #with ThreadPoolExecutor() as executor:
    #    metrics = executor.submit(compute_metrics)
    #    log_forward, log_backward, gap, mean_ang_image, mean_ang_text, uniformity_metric, mean_cos_true_pairs = metrics.result()
    log_forward, log_backward, gap, mean_ang_image, mean_ang_text, uniformity_metric, mean_cos_true_pairs, clustering_metrics = compute_metrics(all_image_embeds, all_text_embeds, similarity_matrix, ids_img, ids_txt)

    # Combine all metrics into final_log
    final_log = {
        **log_forward,
        **log_backward,
        'gap': round(gap, 4),
        'mean_angular_value_image': round(mean_ang_image, 4), # round to 4 decimal places
        'mean_angular_value_text': round(mean_ang_text, 4),
        'uniformity': round(uniformity_metric, 4),
        'mean_cosine_similarity_true_pairs': round(mean_cos_true_pairs, 4),

        **clustering_metrics
    }

    if plot_embeddings:
        print("Evaluation Results:", final_log)
        print()

    # Dump embeddings + IDs + captions for post-hoc analysis (visualization,
    # vector-DB queries, etc.). One file per evaluate call. Tensors are unit-
    # normalized at this point. `epoch` may be an int (per-epoch dump named
    # epoch_NNN.pt) or a string label (e.g. "final_full" → final_full.pt).
    if emb_dir is not None and epoch is not None:
        os.makedirs(emb_dir, exist_ok=True)
        if isinstance(epoch, str):
            snap_name = f"{epoch}.pt"
            epoch_value = epoch
        else:
            snap_name = f"epoch_{int(epoch):03d}.pt"
            epoch_value = int(epoch)
        snapshot = {
            "epoch": epoch_value,
            "image_embeds": all_image_embeds.detach().cpu(),
            "text_embeds": all_text_embeds.detach().cpu(),
            "ids_img": list(ids_img),
            "ids_txt": list(ids_txt),
            "captions": all_captions,
            "labels": all_labels,
            "metrics": final_log,
        }
        torch.save(snapshot, os.path.join(emb_dir, snap_name))

    # Add semantic prefixes for wandb auto-grouping (snapshot keeps flat keys
    # so visualization.py readers continue to work unchanged).
    _embed_keys = {"gap", "uniformity", "mean_angular_value_image",
                   "mean_angular_value_text", "mean_cosine_similarity_true_pairs"}
    def _wandb_key(k: str) -> str:
        if k.startswith("forward_") or k.startswith("backward_"):
            return f"retrieval/{k}"
        if k in _embed_keys:
            return f"embedding/{k}"
        return f"clustering/{k}"
    wandb.log({_wandb_key(k): v for k, v in final_log.items()})

    model.train()
    return final_log

def train_model(config, train_loader, test_loader, device, ckpt_dir: str = None, emb_dir: str = None):

    # Create model & transforms from scratch (no pretrained weights)
    model, _, preprocess = open_clip.create_model_and_transforms(
        config["model"],
        pretrained=None,
        device=device
    )
    
    # Get the tokenizer from the model
    tokenizer = open_clip.get_tokenizer(config["model"])
    
    # Put the model into training mode
    model.train()

    # Require gradients for all parameters to train from scratch
    for param in model.parameters():
        param.requires_grad = True
        
    # Move the model to given device
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    # Set up training parameters
    lr = config["learning_rate"]
    epochs = config["epochs"]
    start_epoch = 0

    # Load the roberta model for anchor-roberta loss
    if config["loss_type"] == "anchor-roberta":
        roberta = SentenceTransformer('stsb-roberta-large').to(device)

    # ClipLoss owns the temperature scalar (Parameter when learnable, buffer
    # otherwise). All other loss branches that call contrastive_loss(...)
    # reuse loss_fn.temperature so gradients flow into the same tensor.
    loss_fn = ClipLoss(
        temperature=config["anchor_temperature"],
        learnable=config["anchor_temperature_learnable"],
    ).to(device)
    if config["anchor_temperature_learnable"]:
        print("Using learnable temperature parameter")
    temperature = loss_fn.temperature

    # Load checkpoint if resuming
    if config["resume_checkpoint"]:
        print(f"Resuming training from {config['resume_checkpoint']} at epoch {config['resume_epoch']}")
        checkpoint = torch.load(config["resume_checkpoint"])
        model.load_state_dict(checkpoint)
        start_epoch = config["resume_epoch"]

    # Optimizer sees model params plus loss_fn params (the latter contains the
    # temperature Parameter only when learnable=True; empty otherwise).
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=1e-4,
    )
    
    
    # Set up the learning rate scheduler as 20% warmup
    t_total = len(train_loader) * config["epochs"]
    num_warmup_steps = int(0.20 * t_total)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total, config=config)

    # Save the initialization (epoch 0) checkpoint and embeddings BEFORE any
    # training, so the paper-style "init / CLIP / Ours" comparison at PCA-time
    # has the actual init-state encoder.
    if ckpt_dir is not None:
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "epoch_000.pt"))

    # Make a prior evaluation of the model (== epoch 0 snapshot)
    print("Evaluating model before training...")
    evaluate_model(model, test_loader, device, model_name=config["model"],
                   epoch=0, emb_dir=emb_dir, loss_fn=loss_fn)

    # Checkpoint cadence: every-N epochs PLUS the alpha/beta phase-transition
    # boundaries (computed from the schedule config so it adapts if the user
    # changes the schedule).
    phase_epochs = {
        config["beta_warmup_epoch"],                                     # beta starts decay
        config["beta_warmup_epoch"] + config["beta_decay_epoch"],        # beta hits 0
        config["alpha_warmup_epoch"],                                    # alpha starts increase
        config["alpha_warmup_epoch"] + config["alpha_increment_epoch"],  # alpha hits 2
    }
    save_every_n = config["save_checkpoint_every_n_epochs"]
    grad_log_every = int(config.get("grad_log_every_n_steps", 100))

    # Mixed precision (AMP) setup — read fp16 flag from config
    use_amp = bool(config.get("fp16", False)) and device.type == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"Mixed precision (fp16 AMP): {'ENABLED' if use_amp else 'disabled (running fp32)'}")

    # BETA init for EXP 7-8-9-10
    beta = 0.0
    alpha = 0.0
    
    # Record start time
    start_time = time.time()
    remaining_time_formatted = "00:00:00"
    
    CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    current_batch, loss = 0, 0
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, captions_list, sample_ids in pbar:

            current_batch += 1
            extra_log = {}  # per-loss values + grad norms accumulated for this step

            # Move data to the primary device
            images = images.to(device)
            captions = captions_list

            #print(f"Processing batch {current_batch} with {len(captions)} samples")

            #print(f"images shape: {images.shape}")
            #print(f"captions: {captions}")

            if isinstance(captions[0], int) or torch.is_tensor(captions[0]):
               captions = [CIFAR10_CLASSES[label] for label in captions]
            
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                # Tokenize text
                text_tokens = tokenizer(captions)
                text_tokens = text_tokens.to(device)
                
                
    
                # Encode image and text
                image_embeds = model.module.encode_image(images)  # Use .module for methods inside DataParallel
                text_embeds = model.module.encode_text(text_tokens)
                
                
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds  = text_embeds  / text_embeds.norm(dim=-1, keepdim=True)
    
    
                # EXP 1 AND EXP 2
                if config["loss_type"] == "anchor":
                    #if epoch < config["only_lunif_epochs"]:
                    #    #print(f"Used only lunif loss for epoch {epoch}, batch {current_batch}")
                    #    loss = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                    #else:
                        #print(f"Used only anchor loss for epoch {epoch}, batch {current_batch}")
                    #loss = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    loss = loss_fn(image_embeds, text_embeds)
                
                # EXP 3 AND EXP 5
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+lunif(text)+lunif(img)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                        lalign = lalign_loss(image_embeds, text_embeds)
                        lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                        loss = anchor + lunif + lalign
                
                # EXP 4 AND EXP 6
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+lunif(centroids)":
                
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
    
                        centroids = compute_centroids_only(image_embeds, text_embeds)
                        centroids = F.normalize(centroids, dim=-1)
                        lunif_centroids = lunif_loss(centroids)
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
    
                        loss =  anchor + config["lambda1"] * lalign + config["lambda2"] * lunif_centroids
    
    
                # EXP 7
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+BETA*lunif(centroids)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
    
                        lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)
                        
                        loss =  anchor + lalign + beta * lunif
      
                
                # EXP 8
                elif config["loss_type"] == "only_lunif_n_then_anchor+lalign+BETA*lunif(centroids)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
    
                        centroids = compute_centroids_only(image_embeds, text_embeds)
                        centroids = F.normalize(centroids, dim=-1)
                        lunif_centroids = lunif_loss(centroids)
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)
                        
                        loss =  anchor + lalign + beta * lunif_centroids
    
                # EXP 9
                elif config["loss_type"] == "only_lunif_n_then_anchor+ALPHA*lalign+BETA*(lunif(text)+lunif(img))":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
    
                        lunif = (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2
                        
                        lalign = lalign_loss(image_embeds, text_embeds)
                        
                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)
    
                        alpha_warmup_epoch = config["alpha_warmup_epoch"]
                        alpha_increment_epoch = config["alpha_increment_epoch"]
    
                        alpha = get_alpha(current_batch,t_total,alpha_warmup_epoch,alpha_increment_epoch)
                        
                        loss =  anchor + alpha * lalign + beta * lunif
      
                
                # EXP 10
                elif config["loss_type"] == "only_lunif_n_then_anchor+ALPHA*lalign+BETA*lunif(centroids)":
                    if epoch < config["only_lunif_epochs"]:
                        lunif_img = lunif_loss(image_embeds)
                        lunif_txt = lunif_loss(text_embeds)
                        loss = (lunif_img + lunif_txt) / 2
                        extra_log["loss/lunif_img"] = lunif_img.item()
                        extra_log["loss/lunif_txt"] = lunif_txt.item()
                    else:
                        anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)

                        centroids = compute_centroids_only(image_embeds, text_embeds)
                        centroids = F.normalize(centroids, dim=-1)
                        lunif_centroids = lunif_loss(centroids)

                        lalign = lalign_loss(image_embeds, text_embeds)

                        beta_warmup_epoch = config["beta_warmup_epoch"]
                        beta_decay_epoch = config["beta_decay_epoch"]
                        beta = get_beta(current_batch,t_total,beta_warmup_epoch,beta_decay_epoch)

                        alpha_warmup_epoch = config["alpha_warmup_epoch"]
                        alpha_increment_epoch = config["alpha_increment_epoch"]

                        alpha = get_alpha(current_batch,t_total,alpha_warmup_epoch,alpha_increment_epoch)

                        loss =  anchor + alpha * lalign + beta * lunif_centroids

                        # Per-loss component values: every step (free).
                        extra_log["loss/anchor"] = anchor.item()
                        extra_log["loss/lalign"] = lalign.item()
                        extra_log["loss/lunif_centroids"] = lunif_centroids.item()

                        # Per-loss gradient norms: every grad_log_every steps
                        # (expensive — runs an extra backward per component).
                        if grad_log_every > 0 and (current_batch == 1 or current_batch % grad_log_every == 0):
                            grad_params = [p for p in model.module.parameters() if p.requires_grad]
                            gn = per_loss_grad_norms(
                                {"anchor": anchor, "lalign": lalign, "lunif_centroids": lunif_centroids},
                                grad_params,
                            )
                            for k, v in gn.items():
                                extra_log[f"grad_norm/{k}"] = v
            
                ###################################
                # ABLATION STUDIES BASED ON EXP 4
                ##################################
                
                # COMPLETE LOSS: ANCHOR(IMAGE,TEXT) + LALIGN(IMAGE,TEXT) + LUNIF(CENTROIDS)
                elif config["loss_type"] == "ANCHOR(IMAGE,TEXT)+LALIGN(IMAGE,TEXT)+LUNIF(CENTROIDS)":
            
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    
                    lalign = lalign_loss(image_embeds, text_embeds)
    
                    centroids = compute_centroids_only(image_embeds, text_embeds)
                    centroids = F.normalize(centroids, dim=-1)
                    lunif_centroids = lunif_loss(centroids)
                    
                    loss =  anchor + lalign + lunif_centroids
                
                # ABLATATION 1: ANCHOR(IMAGE,TEXT) + LALIGN(IMAGE,TEXT)
                elif config["loss_type"] == "ANCHOR(IMAGE,TEXT)+LALIGN(IMAGE,TEXT)":
                    
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    lalign = lalign_loss(image_embeds, text_embeds)
                    
                    loss =  anchor + lalign
                
                # ABLATION 2: ANCHOR(IMAGE,TEXT) + LUNIF(CENTROIDS)
                elif config["loss_type"] == "ANCHOR(IMAGE,TEXT)+LUNIF(CENTROIDS)":
                    
                    anchor = contrastive_loss(image_embeds, text_embeds, temperature=temperature)
                    
                    centroids = compute_centroids_only(image_embeds, text_embeds)
                    centroids = F.normalize(centroids, dim=-1)
                    lunif_centroids = lunif_loss(centroids)
                    
                    loss =  anchor + lunif_centroids   
    
                elif config["loss_type"] == "only_lunif_n_+lalign+lunif(centroids)":
                    
                    centroids = compute_centroids_only(image_embeds, text_embeds)
                    centroids = F.normalize(centroids, dim=-1)
                    lunif_centroids = lunif_loss(centroids)
                    
                    lalign = lalign_loss(image_embeds, text_embeds)
                    
                    loss =  lalign + lunif_centroids

                    
            # Track useful metrics
            if config["anchor_temperature_learnable"]:
                payload = {"loss/total": loss.item(),
                           "optim/temperature": loss_fn.temperature.item(),
                           "optim/learning_rate": scheduler.get_last_lr()[0]}
            else:
                payload = {"loss/total": loss.item(),
                           "optim/learning_rate": scheduler.get_last_lr()[0],
                           "optim/alpha": alpha,
                           "optim/beta": beta}
            payload.update(extra_log)
            wandb.log(payload)
            # Evaluate the model every n batches
            #if current_batch % 100 == 0:
            #    evaluate_model(model, test_loader, device, plot_embeddings=False)
            
            # Zero gradients
            optimizer.zero_grad()

            # AMP-aware backward and optimizer step (no-op scaling when use_amp=False)
            scaler.scale(loss).backward()
            # Add gradient clipping
            #torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            scheduler.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")


        evaluate_model(model, test_loader, device, model_name=config["model"],
                       epoch=epoch + 1, emb_dir=emb_dir, loss_fn=loss_fn)

        # Save checkpoint at every-N epochs OR at any alpha/beta phase boundary.
        epoch_one_indexed = epoch + 1
        should_save_ckpt = (
            (epoch_one_indexed % save_every_n == 0)
            or (epoch_one_indexed in phase_epochs)
        )
        if should_save_ckpt and ckpt_dir is not None:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"epoch_{epoch_one_indexed:03d}.pt"))
            print(f"Model saved at epoch {epoch_one_indexed}")

    return model

class CIFAR10WithIDs(datasets.CIFAR10):
    """CIFAR10 that also returns the sample index as the third tuple element,
    matching the (image, label_or_caption, sample_id) shape expected by the
    rest of the pipeline."""
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, label, index


def get_cifar10_dataloaders(cf, num_workers=4, data_root=DATA_ROOT):
    # Get the CIFAR-10 dataset and dataloaders

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 images to match CLIP's expected size
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))  # CLIP ImageNet normalization
    ])

    # Load CIFAR-10 dataset (with index as sample_id)
    train_dataset = CIFAR10WithIDs(root=data_root, train=True, download=True, transform=transform)
    test_dataset  = CIFAR10WithIDs(root=data_root, train=False, download=True, transform=transform)

    # Optional subsetting (mirror the COCO loader behavior).
    if cf.get("num_train_samples", -1) != -1:
        n = int(cf["num_train_samples"])
        print(f"Subsetting the training dataset to {n} samples")
        train_dataset = Subset(train_dataset, list(range(min(n, len(train_dataset)))))
    if cf.get("num_test_samples", -1) != -1:
        n = int(cf["num_test_samples"])
        print(f"Subsetting the test dataset to {n} samples")
        test_dataset = Subset(test_dataset, list(range(min(n, len(test_dataset)))))

    # CIFAR10 returns label as a Python int already; the eval/train loops
    # accept either ints or strings as the second element ([main.py:133-138],
    # [main.py:389-390]). Use a collate that just stacks images and packs
    # labels + ids as tuples, matching the COCO collate's return shape.
    def collate_fn(batch):
        images, labels, ids = zip(*batch)
        images = torch.stack(images, 0)
        return images, list(labels), tuple(ids)

    batch_size = int(cf.get("batch_size", 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              drop_last=False, collate_fn=collate_fn, num_workers=num_workers)

    print(f"Train Dataloader samples = {len(train_loader)*batch_size}")
    print(f"Test Dataloader samples  = {len(test_loader)*batch_size}")

    return train_loader, test_loader

class CocoCaptionsWithIDs(dset.CocoCaptions):
    def __getitem__(self, index):
        image, captions = super().__getitem__(index)
        sample_id = self.ids[index]  # COCO image ID
        return image, captions, sample_id

def get_coco_dataloaders(config):

    # Path to train images and annotations
    train_image_dir = os.path.join(DATA_ROOT, 'coco/images/train2017/')                          # Path to train2017 images
    train_annotation_file = os.path.join(DATA_ROOT, 'coco/annotations/captions_train2017.json')  # Path to train2017 captions

    # Path to test (val) images and annotations
    test_image_dir = os.path.join(DATA_ROOT, 'coco/images/val2017/')                          # Path to val2017 images
    test_annotation_file = os.path.join(DATA_ROOT, 'coco/annotations/captions_val2017.json')  # Path to val2017 captions
    
    # Fixed mean and std for the dataset
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    # Define the transform to be applied to the images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),  # Resize the image to the model's required input size
        transforms.RandomHorizontalFlip(),         # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Resize(256) + CenterCrop(224) 이렇게도 해볼 수 있음
        transforms.Normalize(mean, std)
    ])

    # Create the training dataset
    train_coco = CocoCaptionsWithIDs(
        root=train_image_dir,
        annFile=train_annotation_file,
        transform=train_transform
    )

    # Create the test dataset
    test_coco = CocoCaptionsWithIDs(
        root=test_image_dir,
        annFile=test_annotation_file,
        transform=test_transform
    )
    
    if config["num_train_samples"] != -1:
        print(f"Subsetting the training dataset to {config['num_train_samples']} samples")
        # Subset the training dataset
        num_training_samples = config["num_train_samples"]
        subset_indices = list(range(num_training_samples))
        train_coco = Subset(train_coco, subset_indices)
    
    if config["num_test_samples"] != -1:
        print(f"Subsetting the test dataset to {config['num_test_samples']} samples")
        # Subset the test dataset
        num_test_samples = config["num_test_samples"]
        subset_indices = list(range(num_test_samples))
        test_coco = Subset(test_coco, subset_indices)

    # Every image has 5 captions at max, we need to sample one of them
    # Create collate function to sample one caption per image
    def collate_fn(batch):
        images, captions, sample_ids = zip(*batch)
        images = torch.stack(images, 0)
        sel_captions = []
        for list_captions in captions:
            caption = random.choice(list_captions)
            sel_captions.append(caption)
        return images, sel_captions, sample_ids

    # Create DataLoader
    train_batch_size = config["batch_size"]
    test_batch_size = config["batch_size"]
    num_workers = int(config.get("num_workers", 4))
    train_loader = DataLoader(train_coco, batch_size=train_batch_size, shuffle=True , drop_last=True, collate_fn=collate_fn, num_workers=num_workers)
    test_loader  = DataLoader(test_coco , batch_size=test_batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn, num_workers=0)

    return train_loader, test_loader


def set_seed(seed: int):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy random module
    torch.manual_seed(seed)  # PyTorch CPU random numbers
    torch.cuda.manual_seed(seed)  # PyTorch GPU random numbers for a single GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random numbers for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.benchmark = False  # Disable benchmark for deterministic behavior


def main(config):

    # run_name = f"{config['run_name']}_lr{config['learning_rate']}_seed{config['seed']}"

    # Initialize your W&B run
    wandb.init(project=config["project_name"], config=config, name=config['run_name']) #, name=f"lambda1_{config['lambda1']}_lambda2_{config['lambda2']}")

    # Set the seed for reproducibility
    set_seed(config["seed"])
    
    # Print the config
    print("Config:", config)
    
    # Print the experiment name
    print("Experiment:", config["run_name"])
    
    # Set the device
    device_id = config["device_id"]
    device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")

    # Per-run output directory (gitignored). All checkpoints / embedding dumps
    # for this run live under run_dir; nothing else writes to disk here.
    run_dir, ckpt_dir, emb_dir = setup_run_dir(config["run_name"])
    print(f"Run artifacts will be saved under: {run_dir}")

    # Load the dataset
    print(f"\nLoading the dataset {config['dataset']}...")
    if config["dataset"] == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(config)
    elif config["dataset"] == "coco":
        train_loader, test_loader = get_coco_dataloaders(config)
    print("Dataset loaded.\n")

    # Train the model
    print("Training the model...")
    model = train_model(config, train_loader, test_loader, device,
                        ckpt_dir=ckpt_dir, emb_dir=emb_dir)
    print("Training complete.\n")

    # Final evaluation on the FULL validation set, regardless of the
    # `num_test_samples` cap used during per-epoch quick eval. Builds a fresh
    # loader by overriding the cap; the snapshot is written as
    # embeddings/final_full.pt (string label avoids colliding with the
    # epoch_NNN.pt dumps).
    print("Final evaluation on the FULL validation set...")
    full_config = dict(config)
    full_config["num_test_samples"] = -1
    if config["dataset"] == "cifar10":
        _, full_test_loader = get_cifar10_dataloaders(full_config)
    elif config["dataset"] == "coco":
        _, full_test_loader = get_coco_dataloaders(full_config)
    else:
        full_test_loader = test_loader

    final_log = evaluate_model(
        model, full_test_loader, device,
        model_name=config["model"],
        epoch="final_full",
        emb_dir=emb_dir,
        loss_fn=None,
    )
    print("Evaluation complete.\n")
    print("Final evaluation results:", final_log)

    final_metrics_payload = {
        "run_name": config["run_name"],
        "config": config,
        "num_val_samples_used": "full",
        "metrics": final_log,
    }
    final_metrics_path = os.path.join(run_dir, "final_metrics.json")
    with open(final_metrics_path, "w") as f:
        json.dump(final_metrics_payload, f, indent=2, ensure_ascii=False)
    print(f"Final metrics saved to {final_metrics_path}")

    # Save the final model weights as final.pt (in addition to the per-epoch
    # checkpoint already written in train_model).
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))

    wandb.finish()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run the experiment with a config.yaml file")
    parser.add_argument("--config", type=str, required=True, help="Path to the yaml config file or to a folder containing multiple config files")
    parser.add_argument("--device", type=int, required=True, help="GPU id to use")
    args = parser.parse_args()
    
    # Load the config file provided from the command line if the path is a file
    if os.path.isfile(args.config):
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
            # Set the device id
        
        config["device_id"] = args.device
        # Convert learning rate to float
        config["learning_rate"] = float(config["learning_rate"])

        # Start the experiment
        main(config)
