"""Dimensionality reduction: the sibling of embedding.py.

Maps embedding vectors to coordinates in the destination space (1-D, 2-D, ...).
Everything UMAP-specific lives here, so trying another method (PCA, a 2-D map
followed by an ordering step, ...) only means providing the same three
functions. A fitted reducer must be picklable and offer
`transform(vectors) -> array of shape (n, dims)`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

MIN_DIST = 0.15
SEED = 42


def fit_reducer(vectors: Any, dims: int, neighbors: int) -> tuple[Any, Any]:
    """Fit on the given vectors. Returns (reducer, coords) with coords of shape (n, dims)."""
    import numpy as np
    import umap  # slow to import, so only when a map is actually fitted

    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=max(2, min(neighbors, len(vectors) - 1)),
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=SEED,
        transform_seed=SEED,
    )
    coords = np.asarray(reducer.fit_transform(vectors)).reshape(len(vectors), dims)
    return reducer, coords


@lru_cache(maxsize=4)
def load_reducer(path: Path) -> Any:
    import joblib

    return joblib.load(path)


def save_reducer(reducer: Any, path: Path) -> None:
    import joblib

    joblib.dump(reducer, path)
    load_reducer.cache_clear()  # a reducer loaded earlier is now stale