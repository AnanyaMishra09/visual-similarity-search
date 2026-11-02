"""CIFAR-100 loading, deterministic per-class splits, and PK batch sampling.

The model never sees class labels as a training signal (no classification
head, no cross-entropy anywhere). Labels are only used here to decide
which images are allowed to sit in the same mini-batch as positives/negatives
for the triplet miner, and later to score retrieval. That's "metric learning
from similarity structure", not classification.
"""
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR100

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_transforms(train: bool) -> transforms.Compose:
    """Light, standard CIFAR augmentation. Kept mild on purpose: triplet
    mining already needs the model to tell apart fine-grained classes, so
    aggressive augmentation (heavy color jitter, cutout) in the early
    epochs was making hard negatives *too* hard and stalled convergence
    during tuning. Crop + flip gave the best val-triplet-accuracy curve.
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def build_splits(
    targets: List[int],
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    gallery_frac_of_test: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Deterministic, per-class, stratified split.

    80% train / 10% val / 10% test per class. The test 10% is further split
    in half into a retrieval gallery and a query set (so a query is never
    trivially matched against itself in the gallery).
    """
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)

    train_idx, val_idx, gallery_idx, query_idx = [], [], [], []
    rng = np.random.RandomState(seed)
    for cls in sorted(by_class.keys()):
        idxs = np.array(by_class[cls])
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_part = idxs[:n_train]
        val_part = idxs[n_train:n_train + n_val]
        test_part = idxs[n_train + n_val:]
        n_gallery = int(round(len(test_part) * gallery_frac_of_test))
        gallery_part = test_part[:n_gallery]
        query_part = test_part[n_gallery:]

        train_idx.extend(train_part.tolist())
        val_idx.extend(val_part.tolist())
        gallery_idx.extend(gallery_part.tolist())
        query_idx.extend(query_part.tolist())

    return {
        "train": np.array(sorted(train_idx)),
        "val": np.array(sorted(val_idx)),
        "gallery": np.array(sorted(gallery_idx)),
        "query": np.array(sorted(query_idx)),
    }


class IndexedCIFAR100(Dataset):
    """Wraps torchvision CIFAR100 restricted to a fixed subset of indices,
    with its own transform. Returns (image, label, original_index).
    """

    def __init__(self, base: CIFAR100, indices: np.ndarray, transform):
        self.base = base
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        orig_idx = int(self.indices[i])
        img, label = self.base.data[orig_idx], int(self.base.targets[orig_idx])
        from PIL import Image
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label, orig_idx


class PKSampler(Sampler):
    """Samples batches of P classes x K images per class ("batch hard"
    sampling, following Hermans et al. 2017). This is what makes in-batch
    hard-negative mining meaningful: without multiple instances per class
    in the same batch there is no positive pair to mine a hard positive from.
    """

    def __init__(self, labels: List[int], p: int, k: int, seed: int = 0):
        self.labels = np.array(labels)
        self.classes = np.unique(self.labels)
        self.by_class = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.p = min(p, len(self.classes))
        self.k = k
        self.seed = seed
        self.epoch = 0
        # one "epoch" = enough batches to roughly cover the dataset once
        self.batches_per_epoch = max(1, len(self.labels) // (self.p * self.k))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.RandomState(self.seed * 10_000 + self.epoch)
        for _ in range(self.batches_per_epoch):
            classes = rng.choice(self.classes, size=self.p, replace=False)
            batch = []
            for c in classes:
                pool = self.by_class[c]
                replace = len(pool) < self.k
                chosen = rng.choice(pool, size=self.k, replace=replace)
                batch.extend(chosen.tolist())
            rng.shuffle(batch)
            yield batch


def get_cifar100(data_root: str) -> Tuple[CIFAR100, CIFAR100]:
    """Returns (train_raw, test_raw) torchvision datasets, download on first
    use. We only use these for .data/.targets as a shared 60k-image pool;
    our own split (build_splits) decides train/val/gallery/query membership,
    not torchvision's own train/test flag.
    """
    os.makedirs(data_root, exist_ok=True)
    train_raw = CIFAR100(root=data_root, train=True, download=True)
    test_raw = CIFAR100(root=data_root, train=False, download=True)
    return train_raw, test_raw


def concat_cifar_pool(train_raw: CIFAR100, test_raw: CIFAR100):
    """Pools torchvision's train+test (50k+10k) into one 60k array so our
    own 80/10/10 split operates on the full, documented 600-images/class
    CIFAR-100 the assignment describes, rather than inheriting torchvision's
    own 50k/10k boundary.
    """
    data = np.concatenate([train_raw.data, test_raw.data], axis=0)
    targets = train_raw.targets + test_raw.targets
    pool = CIFAR100.__new__(CIFAR100)
    pool.data = data
    pool.targets = targets
    pool.classes = train_raw.classes
    return pool
