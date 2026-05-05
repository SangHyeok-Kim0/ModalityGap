from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_beta(current_step, total_steps, warmup_epoch=20, decay_epoch=50):

    steps_in_one_epoch = total_steps / 100

    if current_step < warmup_epoch*steps_in_one_epoch:
        return 1.0
    elif current_step < (warmup_epoch+decay_epoch)*steps_in_one_epoch:
        return 1.0 - float(current_step - warmup_epoch*steps_in_one_epoch) / float(max(1, decay_epoch*steps_in_one_epoch))
    else:
        return 0.0


def get_alpha(current_step, total_steps, warmup_epoch=20, increment_epoch=50):

    steps_in_one_epoch = total_steps / 100


    if current_step < warmup_epoch*steps_in_one_epoch:
        return 1.0
    elif current_step < (warmup_epoch+increment_epoch)*steps_in_one_epoch:
        return 1.0 + float(current_step - warmup_epoch*steps_in_one_epoch) / float(max(1, increment_epoch*steps_in_one_epoch))
    else:
        return 2.0
    

def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
                                    last_epoch: int = -1, steps_sparsify: int = 462, config: dict = None) -> LambdaLR:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Arguments:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    # Question: If we use this scheduler, the max value of lr will be the one set in the optimizer, but this means
    # that lr will be 1e-4 only for a few steps after the warmup period, but in reality we see that if we use a 
    # constant rate of 1e-4, the model performs good, so why use 1e-4 only for a few steps?
    # Cosine with restarts?

    def lr_lambda(current_step):
        # If we are using a warmup with a sparsity loss, we only want to apply the cosine schedule after 
        # the sparsity loss i.e. we want to keep the learning rate constant during the sparsity loss
        if current_step < steps_sparsify and config["only_lunif_epochs"] > 0:
            return 1.0
        elif current_step < num_warmup_steps:
            return (float(current_step) / float(max(1, num_warmup_steps))) + 1e-5
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)) 
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.1, n=15):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps
        self.n = n
    
    def forward(self, pred, target):
        #pred = pred.log_softmax(dim=1)
        pred = pred ** (-self.n)
        pred = pred / torch.sum(pred, dim=1, keepdim=True)


        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * pred, dim=1).mean()
    
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0, n= 15):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n = n

    def forward(self, x, target):
        #logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        logprobs = x ** (-self.n)
        logprobs = logprobs / torch.sum(logprobs, dim=1, keepdim=True)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ClipLoss(nn.Module):
    """Symmetric contrastive (CLIP) loss with optional learnable temperature.

    The temperature scalar lives here:
      - learnable=True  → nn.Parameter (optimizer picks it up via self.parameters())
      - learnable=False → buffer (moves with .to(device), not optimized)

    Inputs are assumed to be unit-norm — the caller normalizes the encoder
    outputs before invoking the loss. (Re-normalizing inside would be a
    harmless no-op but wastes compute, since every call site already does it.)
    """
    def __init__(self, temperature=0.07, learnable=False):
        super().__init__()
        t = torch.tensor(float(temperature))
        if learnable:
            self.temperature = nn.Parameter(t)
        else:
            self.register_buffer("temperature", t)

    def forward(self, image_features, text_features):
        # image_features, text_features: [B, D], expected unit-norm.
        # Single matmul; the t→i direction reuses the transpose.
        logits = image_features @ text_features.t() / self.temperature

        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        loss_i2t = F.cross_entropy(logits,     labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2


def lunif_loss(x, t=2):
    # Compute pairwise distances between all embeddings
    sq_pdist = torch.pdist(x, p=2).pow(2)

    # Apply the uniformity loss formula
    return sq_pdist.mul(-t).exp().mean().log()


def lalign_loss(x, y, alpha=2):
    return (x - y).norm(dim=1).pow(alpha).mean()


def lunif_modality(image_embeds, text_embeds):
    """Per-modality uniformity, averaged across the two modalities."""
    return (lunif_loss(image_embeds) + lunif_loss(text_embeds)) / 2


def lunif_centroid(image_embeds, text_embeds):
    """Uniformity of the (image, text) centroids on the unit sphere."""
    centroids = compute_centroids(image_embeds, text_embeds)
    centroids = F.normalize(centroids, dim=-1)
    return lunif_loss(centroids)


def compute_centroids(text_embeddings, visual_embeddings):
    """Element-wise centroid for each (text_i, visual_i) pair: mean of the
    two embeddings.

    Parameters:
    - text_embeddings   (torch.Tensor): shape (batch_size, feature_dim)
    - visual_embeddings (torch.Tensor): shape (batch_size, feature_dim)

    Returns:
    - torch.Tensor: shape (batch_size, feature_dim)
    """
    return (text_embeddings + visual_embeddings) / 2.0