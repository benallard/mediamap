"""Scores for an ordering. All take the distance matrix D and `order` (a permutation)."""
from __future__ import annotations

import numpy as np

from .ordering import path_length


def mean_step(D, order) -> float:
    """Average distance between consecutive items. Lower is better."""
    return path_length(D, order) / (len(order) - 1)


def neighbour_recall(D, order, k: int = 5, window: int = 5) -> float:
    """Share of each item's k nearest neighbours that sit within +-window shelf positions."""
    n = len(D)
    position = np.empty(n, dtype=int)
    position[np.asarray(order)] = np.arange(n)
    masked = np.array(D, dtype=float)
    np.fill_diagonal(masked, np.inf)
    nearest = np.argsort(masked, axis=1)[:, :k]
    return float((np.abs(position[nearest] - position[:, None]) <= window).mean())


def label_breaks(labels, order) -> int:
    """How often the label changes between neighbours. The minimum possible is
    (number of distinct labels - 1), i.e. every section in one piece."""
    ordered = np.asarray(labels)[np.asarray(order)]
    return int((ordered[1:] != ordered[:-1]).sum())


def worst_joins(D, order, top: int = 5) -> list[tuple[int, int, float]]:
    """The largest steps: (item before, item after, distance)."""
    order = np.asarray(order)
    steps = D[order[:-1], order[1:]]
    return [(int(order[i]), int(order[i + 1]), float(steps[i])) for i in np.argsort(steps)[::-1][:top]]
