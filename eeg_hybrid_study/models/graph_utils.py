import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _canonical_name(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def load_electrode_positions(path: str) -> Dict[str, np.ndarray]:
    positions: Dict[str, np.ndarray] = {}
    file_path = Path(path)
    if not file_path.exists():
        return positions

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            name = _canonical_name(parts[0])
            coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            positions[name] = coords
    return positions


def channels_to_positions(ch_names: List[str], pos_dict: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    coords = []
    for ch in ch_names:
        key = _canonical_name(ch)
        if key not in pos_dict:
            return None
        coords.append(pos_dict[key])
    return np.stack(coords, axis=0).astype(np.float32)


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    k = max(1, min(k, n - 1))
    mask = np.zeros_like(scores, dtype=np.float32)
    for i in range(n):
        idx = np.argsort(scores[i])[::-1]
        kept = [j for j in idx if j != i][:k]
        mask[i, kept] = 1.0
    mask = np.maximum(mask, mask.T)
    return mask


def build_distance_graph(coords: np.ndarray, sigma: float = 0.6, k: int = 8) -> np.ndarray:
    n = coords.shape[0]
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sim = np.exp(-np.square(dist) / (2.0 * sigma * sigma)).astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    mask = _top_k_mask(sim, k=k)
    graph = sim * mask
    return graph.astype(np.float32)


def build_correlation_graph(train_x: np.ndarray, k: int = 8) -> np.ndarray:
    # train_x: [num_samples, channels, timepoints]
    channels = train_x.shape[1]
    flattened = np.transpose(train_x, (1, 0, 2)).reshape(channels, -1)
    corr = np.corrcoef(flattened).astype(np.float32)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)
    mask = _top_k_mask(corr, k=k)
    graph = corr * mask
    return graph.astype(np.float32)


def degree_preserving_random_graph(adj: np.ndarray, seed: int, swaps_per_edge: int = 10) -> np.ndarray:
    """Randomize an undirected graph with double-edge swaps while preserving node degrees."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square matrix")

    rng = np.random.RandomState(seed)
    n = adj.shape[0]
    edge_pairs = [tuple(pair) for pair in np.argwhere(np.triu(adj > 0, k=1))]
    if len(edge_pairs) < 2:
        raise ValueError("At least two graph edges are required for randomization")

    edges = set(edge_pairs)
    attempts = 0
    accepted = 0
    max_attempts = max(100, swaps_per_edge * len(edges) * 20)
    target = swaps_per_edge * len(edges)
    while accepted < target and attempts < max_attempts:
        attempts += 1
        first, second = rng.choice(len(edges), size=2, replace=False)
        edge_list = list(edges)
        a, b = edge_list[first]
        c, d = edge_list[second]
        if len({a, b, c, d}) != 4:
            continue
        if rng.rand() < 0.5:
            candidate = {(min(a, c), max(a, c)), (min(b, d), max(b, d))}
        else:
            candidate = {(min(a, d), max(a, d)), (min(b, c), max(b, c))}
        if len(candidate) != 2 or any(u == v for u, v in candidate):
            continue
        old = {(a, b), (c, d)}
        if any(edge in edges and edge not in old for edge in candidate):
            continue
        edges.difference_update(old)
        edges.update(candidate)
        accepted += 1

    weights = np.array([adj[i, j] for i, j in edge_pairs], dtype=np.float32)
    rng.shuffle(weights)
    out = np.zeros((n, n), dtype=np.float32)
    for (i, j), weight in zip(sorted(edges), weights):
        out[i, j] = weight
        out[j, i] = weight
    return out


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    adj = adj.astype(np.float32)
    n = adj.shape[0]
    adj = adj + np.eye(n, dtype=np.float32)
    degree = np.sum(adj, axis=1)
    degree = np.clip(degree, a_min=1e-6, a_max=None)
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return (inv_sqrt @ adj @ inv_sqrt).astype(np.float32)
