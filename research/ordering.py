"""Ordering algorithms: every function maps a distance matrix to a permutation.

    order = leaf(D)          # order[k] = index of the item that goes at position k

All of them see only D, so they work for any mix of distance blocks. Pure numpy/scipy
(plus umap-learn for umap1d), no knowledge of books or music.
"""
from __future__ import annotations

import numpy as np


def path_length(D: np.ndarray, order) -> float:
    order = np.asarray(order)
    return float(D[order[:-1], order[1:]].sum())


def _clean(D: np.ndarray) -> np.ndarray:
    D = (np.asarray(D, dtype=float) + np.asarray(D, dtype=float).T) / 2
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------- baselines
def random_order(D, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).permutation(len(D))


def pcoa1d(D) -> np.ndarray:
    """Classical MDS to one dimension: sort by the first principal coordinate."""
    D = _clean(D)
    n = len(D)
    J = np.eye(n) - 1.0 / n
    B = -0.5 * J @ (D**2) @ J
    _, vectors = np.linalg.eigh(B)
    return np.argsort(vectors[:, -1])


def spectral(D, k: int = 7) -> np.ndarray:
    """Spectral seriation: sort by the Fiedler vector of the normalised graph Laplacian."""
    D = _clean(D)
    n = len(D)
    sigma = np.median(np.sort(D, axis=1)[:, min(k, n - 1)])
    W = np.exp(-((D / sigma) ** 2))
    np.fill_diagonal(W, 0.0)
    degree = W.sum(axis=1)
    L = np.eye(n) - W / np.sqrt(np.outer(degree, degree))
    _, vectors = np.linalg.eigh(L)
    return np.argsort(vectors[:, 1] / np.sqrt(degree))


def umap1d(D, neighbors: int = 10, seed: int = 42) -> np.ndarray:
    """UMAP to one dimension on the precomputed distances, then sort."""
    import umap

    D = _clean(D)
    embedding = umap.UMAP(
        n_components=1, metric="precomputed", n_neighbors=min(neighbors, len(D) - 1),
        min_dist=0.15, random_state=seed,
    ).fit_transform(D)
    return np.argsort(embedding[:, 0])


# ------------------------------------------------------ hierarchical clustering
def leaf(D, method: str = "average") -> np.ndarray:
    """Agglomerative clustering, then the leaf order that minimises the sum of adjacent
    distances (Bar-Joseph et al.). Gives a tree, so sections come for free."""
    from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    D = _clean(D)
    condensed = squareform(D, checks=False)
    Z = optimal_leaf_ordering(linkage(condensed, method=method), condensed)
    return leaves_list(Z)


# ----------------------------------------------------- shortest open path (TSP)
# The path is padded with a dummy node at both ends (index n, at distance 0 from everything),
# which turns "open path" into plain edge arithmetic: no special cases at the ends.
def _padded(D) -> np.ndarray:
    n = len(D)
    DD = np.zeros((n + 1, n + 1))
    DD[:n, :n] = D
    return DD


def _two_opt(P: np.ndarray, DD: np.ndarray) -> bool:
    """Reverse segments while that shortens the path. P is padded; changed in place."""
    n = len(P) - 2
    improved_any, improved = False, True
    while improved:
        improved = False
        for a in range(1, n):
            b = np.arange(a + 1, n + 1)
            delta = (DD[P[a - 1], P[b]] + DD[P[a], P[b + 1]]
                     - DD[P[a - 1], P[a]] - DD[P[b], P[b + 1]])
            k = int(delta.argmin())
            if delta[k] < -1e-12:
                j = b[k]
                P[a:j + 1] = P[a:j + 1][::-1].copy()
                improved = improved_any = True
    return improved_any


def _or_opt(P: np.ndarray, DD: np.ndarray, max_len: int = 3) -> bool:
    """Move segments of 1..max_len items (optionally reversed) to a better place."""
    n = len(P) - 2
    improved_any, improved = False, True
    while improved:
        improved = False
        for L in range(1, max_len + 1):
            for s in range(1, n - L + 2):
                seg = P[s:s + L].copy()
                first, last = seg[0], seg[-1]
                prev, nxt = P[s - 1], P[s + L]
                removal = DD[prev, first] + DD[last, nxt] - DD[prev, nxt]
                Q = np.concatenate([P[:s], P[s + L:]])
                a, b = Q[:-1], Q[1:]
                base = DD[a, b]
                forward = DD[a, first] + DD[last, b] - base
                backward = DD[a, last] + DD[first, b] - base
                kf, kb = int(forward.argmin()), int(backward.argmin())
                if min(forward[kf], backward[kb]) < removal - 1e-12:
                    k, piece = (kf, seg) if forward[kf] <= backward[kb] else (kb, seg[::-1])
                    P[:] = np.concatenate([Q[:k + 1], piece, Q[k + 1:]])
                    improved = improved_any = True
    return improved_any


def refine(D, order, max_len: int = 3) -> np.ndarray:
    """Local search (2-opt + or-opt) from a given order; never makes it longer."""
    D = _clean(D)
    n = len(D)
    P = np.concatenate([[n], np.asarray(order, dtype=int), [n]])
    DD = _padded(D)
    while _two_opt(P, DD) | _or_opt(P, DD, max_len):
        pass
    return P[1:-1].copy()


def _nearest_neighbour(D, start: int) -> np.ndarray:
    n = len(D)
    free = np.ones(n, dtype=bool)
    order = [start]
    free[start] = False
    for _ in range(n - 1):
        nxt = int(np.where(free, D[order[-1]], np.inf).argmin())
        order.append(nxt)
        free[nxt] = False
    return np.array(order)


def tsp(D, restarts: int = 30, seed: int = 0) -> np.ndarray:
    """Shortest open path, heuristically: greedy nearest-neighbour from several random
    starts, each polished with 2-opt and or-opt, best one kept. Not exact."""
    D = _clean(D)
    rng = np.random.default_rng(seed)
    starts = rng.choice(len(D), size=min(restarts, len(D)), replace=False)
    best, best_length = None, np.inf
    for start in starts:
        order = refine(D, _nearest_neighbour(D, int(start)))
        length = path_length(D, order)
        if length < best_length:
            best, best_length = order, length
    return best


# ------------------------------------------------------------ placing one item
def insert(D, order, item: int) -> np.ndarray:
    """Cheapest insertion: put `item` where it lengthens the path least (ends included).

    The relative order of the items already placed never changes, which is what a frozen
    shelf needs. D must cover both the placed items and `item`; `order` holds indices into D.
    """
    order = np.asarray(order)
    d = np.asarray(D)[item, order]
    inner = d[:-1] + d[1:] - np.asarray(D)[order[:-1], order[1:]]
    costs = np.concatenate([[d[0]], inner, [d[-1]]])  # slot k = before order[k]; k = len = at the end
    return np.insert(order, int(costs.argmin()), item)
