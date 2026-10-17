"""Distance blocks: a library's distance matrix as a weighted sum of parts you can take apart.

Each block is an (n x n) matrix with values roughly in [0, 1], plus a weight. Blocks need not
come from vectors (the era block is a kernel on years, the creator block is "same or not").
The total is D = sum_k w_k * D_k, and explain(i, j) returns each block's share of D[i, j].

Pure numpy; nothing here knows about books or music, so it can move into mediamap as is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Block:
    name: str
    D: np.ndarray
    weight: float


def combine(blocks: list[Block]) -> np.ndarray:
    total = sum(b.weight for b in blocks)
    D = sum(b.weight * b.D for b in blocks) / total
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0.0)
    return D


def explain(blocks: list[Block], i: int, j: int) -> dict[str, float]:
    """Each block's contribution to D[i, j] (they add up to D[i, j])."""
    total = sum(b.weight for b in blocks)
    return {b.name: float(b.weight * b.D[i, j] / total) for b in blocks}


# ------------------------------------------------------------------- blocks
def cosine_block(vectors: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, clipped to [0, 1]."""
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.clip(1.0 - v @ v.T, 0.0, 1.0)


def weighted_tag_vectors(tags: list[list[str]], tag_vectors: dict[str, np.ndarray]) -> np.ndarray:
    """One vector per item: the mean of its tags' vectors, weighted 1, 1/2, 1/3, ... by rank.

    (MusicBrainz gives vote counts, which would be better weights; rank is the fallback.)
    """
    out = []
    for item_tags in tags:
        weights = np.array([1.0 / (rank + 1) for rank in range(len(item_tags))])
        vectors = np.array([tag_vectors[t] for t in item_tags])
        out.append((weights[:, None] * vectors).sum(axis=0))
    return np.array(out)


def era_block(years: np.ndarray, sigma: float) -> np.ndarray:
    """Kernel distance on years: 0 for the same year, approaching 1 beyond a few sigma."""
    gap = years[:, None] - years[None, :]
    return 1.0 - np.exp(-(gap.astype(float) ** 2) / (2 * sigma**2))


def creator_block(creators: list[str]) -> np.ndarray:
    """0 for the same creator, 1 otherwise."""
    c = np.array(creators)
    return (c[:, None] != c[None, :]).astype(float)
