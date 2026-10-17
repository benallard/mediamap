from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np

from mediamap import collection, music, reduction
from mediamap.metadata.providers import musicbrainz


class FakeReducer:
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


FAKE_UMAP = {"umap": SimpleNamespace(UMAP=FakeReducer)}
OLD_ALBUMS_SCHEMA = """CREATE TABLE albums (
                                               id INTEGER PRIMARY KEY, title TEXT NOT NULL, artists TEXT NOT NULL DEFAULT '',
                                               year TEXT NOT NULL DEFAULT '', label TEXT NOT NULL DEFAULT '',
                                               release_format TEXT NOT NULL DEFAULT '', country TEXT NOT NULL DEFAULT '',
                                               barcode TEXT NOT NULL DEFAULT '', source_id TEXT NOT NULL UNIQUE,
                                               semantic_text TEXT NOT NULL DEFAULT '', x REAL,
                                               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"""


def album_record(**overrides):
    record = {
        "source_id": "rel-1", "title": "Unknown Pleasures", "artists": ["Joy Division"],
        "year": "1979", "label": "Factory", "release_format": "CD", "country": "GB",
        "barcode": "", "track_count": 9, "release_group_id": "rg", "artist_id": "art",
    }
    return {**record, **overrides}


def count(col, where="1"):
    return col.db.execute(
        f"SELECT COUNT(*) FROM {col.spec.table} WHERE {where}"
    ).fetchone()[0]


def joblib_load(path: Path):
    import joblib

    return joblib.load(path)


class MediaTestCase(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        for target, name, value in [
            (collection, "ROOT", self.root),
            (musicbrainz, "MIN_INTERVAL", 0),
        ]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        quiet = patch("sys.stdout", new_callable=io.StringIO)
        quiet.start()
        self.addCleanup(quiet.stop)
        FakeReducer.created.clear()
        reduction.load_reducer.cache_clear()
        self.addCleanup(reduction.load_reducer.cache_clear)

    def open_collection(self, spec):
        col = collection.Collection(spec)
        self.addCleanup(col.close)
        return col

    def insert_albums(self, col, n, genres="[]", semantic_text="text"):
        col.db.executemany(
            """INSERT INTO albums(title,artists,year,source_id,semantic_text,genres)
               VALUES(?,?,?,?,?,?)""",
            [(f"Album {i}", "Band", "1975", f"{col.spec.media_type}-{i}", semantic_text, genres)
             for i in range(n)],
        )
        col.db.commit()

    def add_record(self, col, record, genres=("post-punk",)):
        with patch.object(music, "choose_album", side_effect=lambda *args: dict(record)), \
                patch.object(music, "fetch_genres", return_value=list(genres)) as fetch, \
                patch.object(music, "fetch_lastfm_tags", return_value=[]), \
                patch("builtins.input", side_effect=["title", "artist"]):
            col.add()
        return fetch

    @staticmethod
    def musicbrainz_response(releases):
        response = Mock()
        response.json.return_value = {"releases": releases}
        response.raise_for_status.return_value = None
        return response

    @staticmethod
    def release(release_id, release_format, title):
        release = {
            "id": release_id,
            "title": title,
            "artist-credit": [{"name": "Artist"}],
            "date": "2001-03-02",
            "media": [{"track-count": 10}],
        }
        if release_format:
            release["media"][0]["format"] = release_format
        return release
