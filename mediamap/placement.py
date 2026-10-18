"""Placement: how items get positions on the shelf.

A Placement is fitted once on the whole collection and afterwards places newcomers
without moving anything that is already placed. Collection only talks to this interface,
so the way positions are computed can change without touching the commands.

Rows are mappings with at least "semantic_text" (sqlite3.Row works). A strategy may read
any other column it needs; UMAP only reads the text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

from .embedding import EMBEDDING_MODEL
from .reduction import fit_reducer, load_reducer, save_reducer


class Placement(ABC):
    dims: int          # number of coordinates per item
    saved_to: str      # what the fitted state is called, for the "saved to ..." message

    @abstractmethod
    def exists(self) -> bool:
        """True once `fit` has run, i.e. newcomers can be placed."""

    @abstractmethod
    def fit(self, rows: Sequence[Any], neighbors: int) -> list[tuple[float, ...]]:
        """Compute coordinates for every row (one tuple of `dims` floats each) and keep
        whatever is needed to place newcomers later. `neighbors` is the neighbourhood size
        the layout considers (UMAP's n_neighbors); strategies without such a knob ignore it."""

    @abstractmethod
    def place(self, row: Any) -> tuple[float, ...]:
        """Coordinates for one new row, without moving anything already placed."""


class UmapPlacement(Placement):
    """Embed the rows' text, fit UMAP once (see reduction.py), `transform` newcomers."""

    def __init__(self, model_path: Path, dims: int, embed: Callable[..., Any], label: str = "item"):
        self.model_path = model_path
        self.dims = dims
        self.embed = embed
        self.label = label  # only used in messages

    @property
    def saved_to(self) -> str:
        return self.model_path.name

    def exists(self) -> bool:
        return self.model_path.exists()

    def fit(self, rows: Sequence[Any], neighbors: int) -> list[tuple[float, ...]]:
        print(f"Embedding {len(rows)} {self.label}s with {EMBEDDING_MODEL} …")
        vectors = self.embed(
            [r["semantic_text"] or f"{self.label} with no available description" for r in rows],
            progress=True,
        )
        reducer, coords = fit_reducer(vectors, self.dims, neighbors)
        save_reducer(reducer, self.model_path)
        return [tuple(float(v) for v in point) for point in coords]

    def place(self, row: Any) -> tuple[float, ...]:
        reducer = load_reducer(self.model_path)
        return tuple(float(v) for v in reducer.transform(self.embed([row["semantic_text"]]))[0])
