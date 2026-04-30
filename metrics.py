import contextlib
import io
import os
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, accuracy_score
from sklearn.metrics import v_measure_score
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
import math
from typing import List, Dict


@contextlib.contextmanager
def _silence_stdout():
    """Suppress stdout — used to mute pycocotools' verbose loading messages."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

def compute_metric_ret(score_matrix: torch.Tensor, ids: List[int], ids_txt: List[int], direction: str = 'forward') -> Dict[str, float]:
    """
    Compute retrieval metrics for either text-to-vision or vision-to-text retrieval.

    Args:
        score_matrix (torch.Tensor): Similarity matrix of shape [N_text, N_image].
        ids (List[int]): List of image IDs.
        ids_txt (List[int]): List of text IDs corresponding to images.
        direction (str): 'forward' for text-to-vision, 'backward' for vision-to-text.

    Returns:
        Dict[str, float]: Dictionary containing retrieval metrics.
    """
    assert score_matrix.shape == (len(ids_txt), len(ids)), f"Score matrix shape {score_matrix.shape} does not match (len(ids_txt), len(ids))"

    if direction == 'forward':  # Text-to-Vision Retrieval
        # Sort each row in descending order
        indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            # Multi-positive: every image whose id matches counts as correct.
            # For COCO (unique ids) this is exactly one match, so behavior is
            # unchanged. For CIFAR-10 (ids = class labels) every same-class
            # image counts → metric becomes class-level R@K.
            gt_indices = [j for j, id_img in enumerate(ids) if id_img == ids_txt[i]]
            rank.append(min(indice_matrix[i].index(j) for j in gt_indices))

        rank = torch.tensor(rank).to(score_matrix.device)

        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)

        eval_log = {
            'forward_r1': round(vr_r1 * 100, 4),
            'forward_r5': round(vr_r5 * 100, 4),
            'forward_r10': round(vr_r10 * 100, 4),
            #'forward_recall': f'{round(vr_r1 * 100, 1)}/{round(vr_r5 * 100, 1)}/{round(vr_r10 * 100, 1)}',
            'forward_ravg': round((vr_r1 + vr_r5 + vr_r10) / 3 * 100, 4)
        }

    else:  # Vision-to-Text Retrieval
        # Sort each column in descending order
        indice_matrix = score_matrix.sort(dim=0, descending=True)[1].permute(1, 0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices = [idx for idx, id_txt in enumerate(ids_txt) if id_txt == ids[i]]
            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))

        rank = torch.tensor(rank).to(score_matrix.device)

        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)

        eval_log = {
            'backward_r1': round(tr_r1 * 100, 4),
            'backward_r5': round(tr_r5 * 100, 4),
            'backward_r10': round(tr_r10 * 100, 4),
            #'backward_recall': f'{round(tr_r1 * 100,1)}/{round(tr_r5 * 100,1)}/{round(tr_r10 * 100,1)}',
            'backward_ravg': round((tr_r1 + tr_r5 + tr_r10) / 3 * 100, 4)
        }

    return eval_log

def compute_clustering_metrics(feat_t: torch.Tensor, feat_v: torch.Tensor,
                                ids_txt, dataset: str = "coco",
                                labels=None):
    """Compute clustering / linear-probe / k-NN metrics.

    Two paths:
    * COCO: load `instances_val2017.json`, keep images with exactly one object
      category, use that category as the class label (existing behavior).
    * CIFAR10 (or any dataset where per-sample labels are already known): pass
      `labels` (1-D array, len == feat_t.shape[0]) and skip the COCO filter.
    """
    if dataset == "cifar10" or labels is not None:
        if labels is None:
            raise ValueError(
                "labels must be provided for non-COCO datasets")
        true_labels = list(map(int, labels))
        feat_t_new = feat_t if torch.is_tensor(feat_t) else torch.as_tensor(feat_t)
        feat_v_new = feat_v if torch.is_tensor(feat_v) else torch.as_tensor(feat_v)
    else:
        from pycocotools.coco import COCO
        instances_path = os.path.join(
            DATA_ROOT, 'coco/annotations/instances_val2017.json')
        true_labels = []
        with _silence_stdout():
            coco_instances = COCO(instances_path)
        feat_t_new, feat_v_new = [], []
        n_skipped_multilabel = 0
        for i, id in enumerate(ids_txt):
            ann_ids = coco_instances.getAnnIds(imgIds=id)
            anns = coco_instances.loadAnns(ann_ids)
            local_ids = set([ann['category_id'] for ann in anns])
            if len(local_ids) == 1:
                true_labels.append(list(local_ids)[0])
                feat_t_new.append(feat_t[i])
                feat_v_new.append(feat_v[i])
            else:
                n_skipped_multilabel += 1
                continue
        feat_t_new = torch.stack(feat_t_new)
        feat_v_new = torch.stack(feat_v_new)



    kmeans = KMeans(n_clusters=10, random_state=0)
    cluster_labels = kmeans.fit_predict(torch.vstack((feat_t_new, feat_v_new)))

    cluster_labels.shape

    true_labels_new = true_labels * 2
    ari = adjusted_rand_score(true_labels_new, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels_new, cluster_labels)
    hom = homogeneity_score(true_labels_new, cluster_labels)

    preds = kmeans.labels_


    v = v_measure_score(true_labels_new, cluster_labels)

    embeddings = torch.vstack((feat_t_new, feat_v_new))
    # Get unique labels and create mapping to consecutive integers
    unique_labels = sorted(list(set(true_labels_new)))
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}

    # Apply mapping to create consecutive labels from 0 to num_classes-1
    full_labels = torch.tensor([label_mapping[label] for label in true_labels_new], dtype=torch.long)



    D = embeddings.shape[1]
    num_classes = len(torch.unique(full_labels))

    # Custom dataset
    class EmbeddingDataset(Dataset):
        def __init__(self, embeddings, labels):
            self.embeddings = embeddings
            self.labels = labels

        def __len__(self):
            return len(self.embeddings)

        def __getitem__(self, idx):
            return self.embeddings[idx], self.labels[idx]

    dataset = EmbeddingDataset(embeddings, full_labels)

    # Train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    # Linear classifier model
    class LinearProbe(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_dim, num_classes)

        def forward(self, x):
            return self.fc(x)

    model = LinearProbe(D, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_losses, test_accuracies = [], []

    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(acc)
        #print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {acc:.4f}")

    # k-NN evaluation
    X_train = embeddings[train_set.indices].numpy()
    y_train = full_labels[train_set.indices].numpy()
    X_test = embeddings[test_set.indices].numpy()
    y_test = full_labels[test_set.indices].numpy()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, knn_preds)

    return {
        "ARI": ari,
        "NMI": nmi,
        "Homogeneity": hom,
        "V-measure": v,
        "Max Linear Probe Acc": max(test_accuracies),
        "K-NN Acc": knn_acc
    }


def compute_gap(feat_modality1: torch.Tensor, feat_modality2: torch.Tensor) -> float:
    """
    Compute the Euclidean distance between the centroids of two modalities.

    Args:
        feat_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [N, D].
        feat_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Euclidean distance between centroids.
    """
    # Ensure features are normalized if required
    modality1_centroid = torch.mean(feat_modality1, dim=0)
    modality2_centroid = torch.mean(feat_modality2, dim=0)

    gap = modality1_centroid - modality2_centroid
    norm_gap = torch.norm(gap).item()

    return norm_gap

def compute_mean_angular_value_of_a_modality(feat_modality: torch.Tensor) -> float:
    """
    Compute the mean angular value (mean cosine similarity) of a modality.

    Args:
        feat_modality (torch.Tensor): Feature matrix with shape [N, D].

    Returns:
        float: Mean angular value.
    """
    # Compute cosine similarity matrix
    cos_sim = feat_modality @ feat_modality.T

    # Exclude diagonal elements by creating a mask
    mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
    cos_sim_no_diag = cos_sim[mask]

    mean_cos_sim = cos_sim_no_diag.mean().item()

    return mean_cos_sim

def uniformity(features_modality1: torch.Tensor, features_modality2: torch.Tensor) -> float:
    x = torch.cat([features_modality1, features_modality2], dim=0)
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x - x_center).t(), x - x_center) / N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps

    mS = np.sqrt(np.diag(S.clip(min=0)))

    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return -wasserstein_distance 

def centroid_alignment_loss(img_embeds: torch.Tensor, txt_embeds: torch.Tensor, p=2) -> torch.Tensor:
    """
    Compute the distance between the mean image embedding and the mean text embedding.

    Args:
        img_embeds (torch.Tensor): Image embeddings of shape (batch_size, embed_dim).
        txt_embeds (torch.Tensor): Text embeddings of shape (batch_size, embed_dim).
        p (int): Norm order (2 for Euclidean / L2 norm).

    Returns:
        torch.Tensor: A scalar tensor representing the centroid alignment penalty.
    """
    # Compute centroids along the batch dimension
    centroid_img = img_embeds.mean(dim=0)  # shape (embed_dim,)
    centroid_txt = txt_embeds.mean(dim=0)  # shape (embed_dim,)

    # Compute the L2 distance (default) between the centroids
    dist = torch.norm(centroid_img - centroid_txt, p=p)
    return dist

def mean_distance_of_true_pairs(features_modality1: torch.Tensor, features_modality2: torch.Tensor, cosine = True) -> float:
    """
    Compute the mean cosine similarity of true pairs between two modalities.

    Args:
        features_modality1 (torch.Tensor): Normalized feature matrix of modality 1 with shape [N, D].
        features_modality2 (torch.Tensor): Normalized feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Mean cosine similarity of true pairs.
    """
    # Compute cosine similarity matrix
    if cosine:
        cosine_sim = torch.matmul(features_modality1, features_modality2.T)

        # Extract diagonal elements (true pairs)
        cosine_sim_diag = torch.diag(cosine_sim)

        # Compute mean cosine similarity of true pairs
        cosine_tv_mean = torch.mean(cosine_sim_diag).item()

        return cosine_tv_mean

    else:
        # Compute Euclidean distance matrix
        euclidean_dist = torch.cdist(features_modality1, features_modality2)

        # Extract diagonal elements (true pairs)
        euclidean_dist_diag = torch.diag(euclidean_dist)

        # Compute mean Euclidean distance of true pairs
        euclidean_tv_mean = torch.mean(euclidean_dist_diag).item()

        return euclidean_tv_mean

