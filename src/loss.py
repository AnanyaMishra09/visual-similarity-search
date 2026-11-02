"""Triplet loss with in-batch hard-negative mining ("batch hard", Hermans
et al. 2017), implemented directly on top of raw tensor ops. No
pytorch-metric-learning, no library loss function.

Given an embedded batch and its labels, for every anchor i we look at every
other sample in the *same batch* and pick:
  - the hardest positive: same label as i, maximum distance from i
  - the hardest negative: different label from i, minimum distance from i
and push d(a,p) - d(a,n) + margin below zero. This only works if the batch
actually contains multiple images per class, which is why training uses the
PKSampler (P classes x K images) from data.py rather than plain random
shuffling.
"""
from typing import Tuple

import torch


def pairwise_euclidean(embeddings: torch.Tensor) -> torch.Tensor:
    """NxD embeddings -> NxN euclidean distance matrix. Written out
    explicitly (not torch.cdist) so the mining logic below is fully
    auditable: d(i,j)^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j.
    """
    sq_norms = (embeddings ** 2).sum(dim=1, keepdim=True)  # N x 1
    dot = embeddings @ embeddings.t()  # N x N
    sq_dists = sq_norms + sq_norms.t() - 2 * dot
    sq_dists = sq_dists.clamp(min=0.0)  # guard against tiny negative fp noise
    # sqrt's gradient is infinite at exactly 0. A freshly-initialized network
    # (heavy BatchNorm collapse before any training) often puts several pairs
    # at sq_dist==0, so without this eps the very first backward pass produces
    # NaN gradients that silently poison every parameter for the rest of training.
    return (sq_dists + 1e-12).sqrt()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3
) -> Tuple[torch.Tensor, dict]:
    """Batch-hard triplet loss. Returns (loss, stats) where stats contains
    the fraction of anchors for which d(a,p) < d(a,n), the same quantity
    the assignment's "≥75% of validation triplets" target is measured on.
    """
    device = embeddings.device
    n = embeddings.size(0)
    dist = pairwise_euclidean(embeddings)

    labels = labels.view(-1, 1)
    same_class = labels.eq(labels.t())
    diff_class = ~same_class
    eye = torch.eye(n, dtype=torch.bool, device=device)
    positive_mask = same_class & ~eye

    # Anchors with no positive in this batch can't form a triplet; mask them out.
    has_positive = positive_mask.any(dim=1)
    has_negative = diff_class.any(dim=1)
    valid_anchor = has_positive & has_negative

    # Hardest positive = max distance among same-class (excluding self).
    dist_pos = dist.masked_fill(~positive_mask, float("-inf"))
    hardest_pos, _ = dist_pos.max(dim=1)

    # Hardest negative = min distance among different-class.
    dist_neg = dist.masked_fill(~diff_class, float("inf"))
    hardest_neg, _ = dist_neg.min(dim=1)

    losses = torch.relu(hardest_pos - hardest_neg + margin)
    losses = losses[valid_anchor]

    if losses.numel() == 0:
        loss = torch.zeros((), device=device, requires_grad=True)
        frac_correct = torch.tensor(0.0)
    else:
        loss = losses.mean()
        correct = (hardest_pos[valid_anchor] < hardest_neg[valid_anchor]).float()
        frac_correct = correct.mean()

    stats = {
        "frac_correct_ordering": frac_correct.item(),
        "mean_hardest_pos_dist": hardest_pos[valid_anchor].mean().item() if valid_anchor.any() else float("nan"),
        "mean_hardest_neg_dist": hardest_neg[valid_anchor].mean().item() if valid_anchor.any() else float("nan"),
        "n_valid_anchors": int(valid_anchor.sum().item()),
    }
    return loss, stats


def batch_random_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3,
    generator: torch.Generator = None,
) -> Tuple[torch.Tensor, dict]:
    """Same triplet margin loss, but the positive/negative for each anchor
    are picked *uniformly at random* from the valid same-class / different-
    class pool instead of the hardest one. Used only for the mining-strategy
    ablation (random vs. hard negatives); the hard-mining path above is
    what trains the main model.
    """
    device = embeddings.device
    n = embeddings.size(0)
    dist = pairwise_euclidean(embeddings)

    labels = labels.view(-1, 1)
    same_class = labels.eq(labels.t())
    diff_class = ~same_class
    eye = torch.eye(n, dtype=torch.bool, device=device)
    positive_mask = same_class & ~eye

    has_positive = positive_mask.any(dim=1)
    has_negative = diff_class.any(dim=1)
    valid_anchor = has_positive & has_negative

    rand = torch.rand(n, n, device=device, generator=generator)

    pos_scores = rand.masked_fill(~positive_mask, float("-inf"))
    pos_idx = pos_scores.argmax(dim=1)
    neg_scores = rand.masked_fill(~diff_class, float("-inf"))
    neg_idx = neg_scores.argmax(dim=1)

    arange = torch.arange(n, device=device)
    d_ap = dist[arange, pos_idx]
    d_an = dist[arange, neg_idx]

    losses = torch.relu(d_ap - d_an + margin)[valid_anchor]
    if losses.numel() == 0:
        loss = torch.zeros((), device=device, requires_grad=True)
        frac_correct = torch.tensor(0.0)
    else:
        loss = losses.mean()
        correct = (d_ap[valid_anchor] < d_an[valid_anchor]).float()
        frac_correct = correct.mean()

    stats = {
        "frac_correct_ordering": frac_correct.item(),
        "n_valid_anchors": int(valid_anchor.sum().item()),
    }
    return loss, stats
