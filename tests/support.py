"""Shared fixtures for the mediamap tests."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mediamap import collection, reduction


class FakeReducer:
    """Stands in for umap.UMAP: coordinates are 0, 1, 2, ... and projections are 0.25."""

    created = []

    def __init__(self, **options):
        self.options = options
        self.n_components = options["n_components"]
        self.created.append(self)

    def fit_transform(self, vectors):
        coordinates = np.arange(len(vectors), dtype=float)[:, None]
        return np.tile(coordinates, (1, self.n_components))

    def transform(self, vectors):
        return np.full((len(vectors), self.n_components), 0.25)


def fake_encode(texts, progress=False):
    return np.arange(len(texts) * 4, dtype=float).reshape(len(texts), 4)


def with_fake_models(test):
    """Replace the sentence embedder and UMAP with the fakes above for one test."""
    test = patch("mediamap.collection.encode_texts", new=fake_encode)(test)
    return patch.dict("sys.modules", {"umap": SimpleNamespace(UMAP=FakeReducer)})(test)


def count(col, where="1"):
    return col.db.execute(f"SELECT COUNT(*) FROM {col.spec.table} WHERE {where}").fetchone()[0]


class CatalogueTestCase(unittest.TestCase):
    """Every test gets its own project root, silenced output and a fresh reducer cache.

    Anything the commands print can be read back from `self.stdout`.
    """

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.start_patch(patch.object(collection, "ROOT", self.root))
        self.stdout = self.start_patch(patch("sys.stdout", new_callable=io.StringIO))
        FakeReducer.created.clear()
        reduction.load_reducer.cache_clear()
        self.addCleanup(reduction.load_reducer.cache_clear)

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def open_collection(self, spec):
        col = collection.Collection(spec)
        self.addCleanup(col.close)
        return col