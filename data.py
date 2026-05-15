"""COCO captions data loading.

`get_coco_dataloaders(config)` returns (train_loader, test_loader). Each batch
yields `(images, captions, sample_ids)` where `captions` is a list of one
randomly-chosen caption per image (COCO has up to 5 per image).
"""

import os
import random

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')

# CLIP / OpenAI ImageNet normalization stats.
_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD  = [0.26862954, 0.26130258, 0.27577711]


class CocoCaptionsWithIDs(dset.CocoCaptions):
    """CocoCaptions that also returns the COCO image_id as the third element."""
    def __getitem__(self, index):
        image, captions = super().__getitem__(index)
        sample_id = self.ids[index]
        return image, captions, sample_id


def _build_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
    return train_transform, test_transform


def _coco_collate(batch):
    """Stack images, randomly pick one caption per image, keep ids as a tuple."""
    images, captions, sample_ids = zip(*batch)
    images = torch.stack(images, 0)
    sel_captions = [random.choice(c) for c in captions]
    return images, sel_captions, sample_ids


def _coco_collate_first(batch):
    """Test-time collate: always pick captions[0] so the same image's caption
    is identical across epochs. Lets per-epoch embedding drift be attributed
    to model updates rather than caption resampling.
    """
    images, captions, sample_ids = zip(*batch)
    images = torch.stack(images, 0)
    sel_captions = [c[0] for c in captions]
    return images, sel_captions, sample_ids


def _maybe_subset(dataset, n, label):
    if n == -1:
        return dataset
    print(f"Subsetting the {label} dataset to {n} samples")
    return Subset(dataset, list(range(n)))


def get_coco_dataloaders(config):
    train_image_dir = os.path.join(DATA_ROOT, 'coco/images/train2017/')
    train_ann_file  = os.path.join(DATA_ROOT, 'coco/annotations/captions_train2017.json')
    test_image_dir  = os.path.join(DATA_ROOT, 'coco/images/val2017/')
    test_ann_file   = os.path.join(DATA_ROOT, 'coco/annotations/captions_val2017.json')

    train_transform, test_transform = _build_transforms()

    train_coco = CocoCaptionsWithIDs(root=train_image_dir, annFile=train_ann_file,
                                     transform=train_transform)
    test_coco  = CocoCaptionsWithIDs(root=test_image_dir,  annFile=test_ann_file,
                                     transform=test_transform)

    train_coco = _maybe_subset(train_coco, config["num_train_samples"], "training")
    test_coco  = _maybe_subset(test_coco,  config["num_test_samples"],  "test")

    batch_size  = config["batch_size"]
    num_workers = int(config.get("num_workers", 4))

    train_loader = DataLoader(train_coco, batch_size=batch_size, shuffle=True,
                              drop_last=True, collate_fn=_coco_collate,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_coco,  batch_size=batch_size, shuffle=False,
                              drop_last=True, collate_fn=_coco_collate_first,
                              num_workers=0)

    return train_loader, test_loader
