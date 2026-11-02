"""Retrieval metrics (Recall@K, mAP) and the ablation driver.

All metrics are computed from a query set against a disjoint gallery set
(see data.build_splits) using retrieval.BruteForceIndex.
"""
from typing import Dict

import numpy as np

from retrieval import BruteForceIndex


def recall_at_k(query_labels: np.ndarray, top_labels: np.ndarray, k: int) -> float:
    """Fraction of queries where >=1 of the first k retrieved gallery items
    shares the query's class label.
    """
    hits = (top_labels[:, :k] == query_labels[:, None]).any(axis=1)
    return float(hits.mean())


def average_precision(query_label: int, ranked_labels: np.ndarray) -> float:
    """Standard AP: precision averaged at each rank where a relevant
    (same-class) item is retrieved, over the full ranked gallery.
    """
    relevant = (ranked_labels == query_label).astype(np.float64)
    n_relevant = relevant.sum()
    if n_relevant == 0:
        return 0.0
    cum_hits = np.cumsum(relevant)
    ranks = np.arange(1, len(ranked_labels) + 1)
    precision_at_i = cum_hits / ranks
    ap = (precision_at_i * relevant).sum() / n_relevant
    return float(ap)


def mean_average_precision(query_labels: np.ndarray, all_ranked_labels: np.ndarray) -> float:
    """all_ranked_labels: (Q, N_gallery) gallery labels ranked by distance
    for each query (the *full* ranking, not just top-k). mAP is defined
    over the whole gallery, not a truncated top-k list.
    """
    aps = [
        average_precision(query_labels[i], all_ranked_labels[i])
        for i in range(len(query_labels))
    ]
    return float(np.mean(aps))


def evaluate_retrieval(
    query_emb: np.ndarray, query_labels: np.ndarray,
    gallery_emb: np.ndarray, gallery_labels: np.ndarray,
    ks=(1, 5, 10),
) -> Dict[str, float]:
    """Full evaluation: Recall@{ks} and mAP over the whole gallery ranking."""
    index = BruteForceIndex(gallery_emb, gallery_labels)
    full_k = gallery_emb.shape[0]
    top_idx, _ = index.query(query_emb, k=full_k)
    ranked_labels = gallery_labels[top_idx]  # Q x N_gallery, full ranking

    results = {}
    for k in ks:
        results[f"recall@{k}"] = recall_at_k(query_labels, ranked_labels, k)
    results["mAP"] = mean_average_precision(query_labels, ranked_labels)
    return results
