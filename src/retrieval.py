"""Retrieval index: brute-force NumPy search (primary path) plus a small
from-scratch KD-tree (sanity-check alternative, per the ground rules:
no Faiss as the primary path). Both operate on plain NumPy arrays so the
trained model doesn't need to be loaded at query time.
"""
from typing import Tuple

import numpy as np


class BruteForceIndex:
    """Stores gallery embeddings + labels; answers top-k queries by
    ascending L2 distance (embeddings are L2-normalised, so this ranks
    identically to cosine similarity, so we expose both for clarity).
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        assert embeddings.shape[0] == labels.shape[0]
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels

    def query(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """queries: (Q, D). Returns (indices, distances), each (Q, k),
        sorted by ascending L2 distance. O(Q * N * D), fine at gallery
        sizes in the thousands; see README for the FAISS stretch-goal note.
        """
        queries = queries.astype(np.float32)
        # ||q - g||^2 = ||q||^2 + ||g||^2 - 2 q.g, computed as one matmul.
        q_sq = (queries ** 2).sum(axis=1, keepdims=True)          # Q x 1
        g_sq = (self.embeddings ** 2).sum(axis=1, keepdims=True).T  # 1 x N
        dot = queries @ self.embeddings.T                          # Q x N
        sq_dist = np.clip(q_sq + g_sq - 2 * dot, a_min=0.0, a_max=None)
        dist = np.sqrt(sq_dist)

        k = min(k, dist.shape[1])
        part = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(dist.shape[0])[:, None]
        part_dist = dist[row_idx, part]
        order = np.argsort(part_dist, axis=1)
        top_idx = part[row_idx, order]
        top_dist = part_dist[row_idx, order]
        return top_idx, top_dist

    def cosine_query(self, queries: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Same thing ranked by cosine similarity (descending) instead of
        L2 distance. Included because the assignment explicitly allows
        "NumPy cosine / dot-product" retrieval and the two are not
        *exactly* identical in finite precision.
        """
        q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        g = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)
        sim = q @ g.T  # Q x N, higher = more similar
        k = min(k, sim.shape[1])
        part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        row_idx = np.arange(sim.shape[0])[:, None]
        part_sim = sim[row_idx, part]
        order = np.argsort(-part_sim, axis=1)
        top_idx = part[row_idx, order]
        top_sim = part_sim[row_idx, order]
        return top_idx, top_sim


class KDNode:
    __slots__ = ("point_idx", "axis", "left", "right")

    def __init__(self, point_idx, axis, left=None, right=None):
        self.point_idx = point_idx
        self.axis = axis
        self.left = left
        self.right = right


class KDTree:
    """Minimal from-scratch KD-tree over the gallery embeddings, used only
    as a sanity check against BruteForceIndex (KD-trees don't actually help
    at D=128: "curse of dimensionality" means brute force degenerates to
    near-linear scan anyway, but the assignment lists it as an allowed
    retrieval component so it's implemented for cross-checking correctness).
    """

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = labels
        self.dim = embeddings.shape[1]
        self.root = self._build(list(range(len(embeddings))), depth=0)

    def _build(self, indices, depth):
        if not indices:
            return None
        axis = depth % self.dim
        indices = sorted(indices, key=lambda i: self.embeddings[i, axis])
        mid = len(indices) // 2
        node = KDNode(point_idx=indices[mid], axis=axis)
        node.left = self._build(indices[:mid], depth + 1)
        node.right = self._build(indices[mid + 1:], depth + 1)
        return node

    def _dist(self, i, query):
        diff = self.embeddings[i] - query
        return float(np.dot(diff, diff)) ** 0.5

    def query_one(self, query: np.ndarray, k: int = 5):
        best = []  # list of (dist, idx), kept sorted, size <= k

        def visit(node):
            if node is None:
                return
            d = self._dist(node.point_idx, query)
            if len(best) < k:
                best.append((d, node.point_idx))
                best.sort(key=lambda x: x[0])
            elif d < best[-1][0]:
                best[-1] = (d, node.point_idx)
                best.sort(key=lambda x: x[0])

            axis = node.axis
            diff = query[axis] - self.embeddings[node.point_idx, axis]
            near, far = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(near)
            if len(best) < k or abs(diff) < best[-1][0]:
                visit(far)

        visit(self.root)
        idx = np.array([b[1] for b in best])
        dist = np.array([b[0] for b in best])
        return idx, dist

    def query(self, queries: np.ndarray, k: int = 5):
        idxs, dists = [], []
        for q in queries:
            i, d = self.query_one(q, k)
            idxs.append(i)
            dists.append(d)
        return np.stack(idxs), np.stack(dists)
