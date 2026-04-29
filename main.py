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
MODELS_ROOT = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_ROOT, exist_ok=True)



def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, model_name: str, plot_embeddings=True, loss_fn=None) -> Dict[str, float]:
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

            # Assign unique IDs
            bs = images.size(0)
            #sample_ids = list(range(current_index, current_index + bs))
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

        clustering_metrics = compute_clustering_metrics(all_text_embeds, all_image_embeds, ids_txt)

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
    
    wandb.log(final_log)

    model.train()
    return final_log

def train_model(config, train_loader, test_loader, device):

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

    # Make a prior evaluation of the model
    print("Evaluating model before training...")
    evaluate_model(model, test_loader, device, model_name=config["model"], loss_fn=loss_fn)

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
        for images, captions_list, sample_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, ETA: {remaining_time_formatted}"):

            current_batch += 1
                        
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
                wandb.log({"train_loss": loss.item(),
                           "constrantive_temperature_learnable": loss_fn.temperature.item(),
                           "learning_rate": scheduler.get_last_lr()[0]})
                           
            else:
                wandb.log({"train_loss": loss.item(),
                            #"learning_rate": scheduler.get_last_lr()[0],
                            "beta": beta,
                            "alpha": alpha})
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
            
            
        
        evaluate_model(model, test_loader, device, model_name=config["model"], loss_fn=loss_fn)

        if (epoch+1) % config["save_checkpoint_every_n_epochs"]  == 0:
            torch.save(model.state_dict(), os.path.join(MODELS_ROOT, f"{config['run_name']}_epoch_{epoch+1}.pt"))
            print(f"Model saved at epoch {epoch+1}")
        
    return model

def get_cifar10_dataloaders(cf, batch_size=128, num_workers=4, data_root=DATA_ROOT):
    # Get the CIFAR-10 dataset and dataloaders
    
    # Define transformations
    #transform = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    #])
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize CIFAR-10 images to match CLIP's expected size
        transforms.ToTensor(),
        transforms.Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758))  # CLIP ImageNet normalization
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Print the len of the dataloaders
    print(f"Train Dataloader samples = {len(train_loader)*batch_size}")
    print(f"Test Dataloader samples = {len(test_loader)*batch_size}")
    
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
    train_loader = DataLoader(train_coco, batch_size=train_batch_size, shuffle=True , drop_last=True, collate_fn=collate_fn, num_workers=8)
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
    
    # Load the dataset
    print(f"\nLoading the dataset {config['dataset']}...")
    if config["dataset"] == "cifar10":
        train_loader, test_loader = get_cifar10_dataloaders(config)
    elif config["dataset"] == "coco":
        train_loader, test_loader = get_coco_dataloaders(config)
    print("Dataset loaded.\n")
    
    # Train the model
    print("Training the model...")
    model = train_model(config, train_loader, test_loader, device)
    print("Training complete.\n")
    
    # Final evaluation of the model
    print("Final evaluation of the model...")
    final_log = evaluate_model(model, test_loader, device, model_name=config["model"], loss_fn=None)
    print("Evaluation complete.\n")
    print("Final evaluation results:", final_log)
    
    # Save the model and upload it to W&B
    torch.save(model.state_dict(), os.path.join(MODELS_ROOT, config['run_name'] + ".pt"))
    #wandb.save(config["run_name"] + ".pt")    
    
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
