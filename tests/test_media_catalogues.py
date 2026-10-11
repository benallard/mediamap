from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from mediamap import books, cli, music


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


class MediaCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.root_patch = patch.object(music, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_cd_and_lp_have_independent_databases(self):
        cds = music.connect("cds")
        lps = music.connect("lps")
        self.addCleanup(cds.close)
        self.addCleanup(lps.close)

        cds.execute(
            """INSERT INTO albums(title,source_id)
               VALUES('CD release','cd-release')"""
        )
        cds.commit()

        self.assertNotEqual(
            music.collection_paths("cds")[0],
            music.collection_paths("lps")[0],
        )
        self.assertEqual(
            cds.execute("SELECT COUNT(*) FROM albums").fetchone()[0],
            1,
        )
        self.assertEqual(
            lps.execute("SELECT COUNT(*) FROM albums").fetchone()[0],
            0,
        )

    @patch("mediamap.music.requests.get")
    def test_release_search_filters_by_requested_physical_format(self, get):
        get.return_value = self._musicbrainz_response([
            self._release("cd-release", "CD", "CD release"),
            self._release("lp-release", '12" Vinyl', "LP release"),
            self._release("unknown-release", None, "Unknown format"),
        ])

        cds = music.search_releases("cds", "Album", "Artist")
        lps = music.search_releases("lps", "Album", "Artist")

        self.assertEqual([record["source_id"] for record in cds], ["cd-release"])
        self.assertEqual([record["source_id"] for record in lps], ["lp-release"])
        self.assertEqual(cds[0]["track_count"], 10)
        self.assertEqual(cds[0]["artists"], ["Artist"])

    @patch("mediamap.music.requests.get")
    def test_unreported_formats_remain_selectable(self, get):
        get.return_value = self._musicbrainz_response([
            self._release("unknown-release", None, "Unknown format"),
        ])

        records = music.search_releases("lps", "Album", "Artist")

        self.assertEqual([record["source_id"] for record in records], ["unknown-release"])

    @patch("mediamap.music._embedding")
    @patch("mediamap.music.umap.UMAP", FakeReducer)
    def test_cd_and_lp_maps_are_independent_and_one_dimensional(self, embedding):
        embeddings = np.arange(24, dtype=float).reshape(6, 4)
        embedding.side_effect = lambda texts, progress=False: embeddings[:len(texts)]

        for media_type in ("cds", "lps"):
            db = music.connect(media_type)
            self.addCleanup(db.close)
            db.executemany(
                """INSERT INTO albums(title,artists,source_id,semantic_text)
                   VALUES(?,?,?,?)""",
                [
                    (f"Album {index}", "Artist", f"{media_type}-{index}", f"text {index}")
                    for index in range(6)
                ],
            )
            db.commit()

            music.build_map(db, media_type, neighbors=3)

            _, model_path = music.collection_paths(media_type)
            self.assertTrue(model_path.is_file())
            self.assertEqual(joblib_load(model_path).n_components, 1)
            self.assertEqual(
                db.execute("SELECT COUNT(*) FROM albums WHERE x IS NOT NULL").fetchone()[0],
                6,
            )

        self.assertNotEqual(
            music.collection_paths("cds")[1],
            music.collection_paths("lps")[1],
        )

    @patch("mediamap.books.load_embedder")
    @patch.dict("sys.modules", {
        "joblib": SimpleNamespace(dump=lambda model, path: None),
        "umap": SimpleNamespace(UMAP=FakeReducer),
    })
    def test_book_map_remains_two_dimensional(self, load_embedder):
        FakeReducer.created.clear()
        db = sqlite3.connect(":memory:")
        self.addCleanup(db.close)
        db.row_factory = sqlite3.Row
        db.execute(
            "CREATE TABLE books(id INTEGER PRIMARY KEY, semantic_text TEXT, x REAL, y REAL)"
        )
        db.executemany(
            "INSERT INTO books(id,semantic_text) VALUES(?,?)",
            [(index, f"book {index}") for index in range(5)],
        )

        class FakeEmbedder:
            def encode(self, texts, **kwargs):
                return np.ones((len(texts), 4))

        load_embedder.return_value = FakeEmbedder()
        books.build_map(db, neighbors=3)

        self.assertEqual(FakeReducer.created[-1].n_components, 2)
        self.assertEqual(
            db.execute(
                "SELECT COUNT(*) FROM books WHERE x IS NOT NULL AND y IS NOT NULL"
            ).fetchone()[0],
            5,
        )

    def test_cli_routes_books_and_music_to_distinct_handlers(self):
        with patch.object(books, "main") as books_main:
            cli.main(["books", "add"])
        books_main.assert_called_once_with(["add"])

        with patch.object(music, "main") as music_main:
            cli.main(["lps", "list"])
        music_main.assert_called_once_with("lps", ["list"])

    @staticmethod
    def _musicbrainz_response(releases):
        response = Mock()
        response.json.return_value = {"releases": releases}
        response.raise_for_status.return_value = None
        return response

    @staticmethod
    def _release(release_id, release_format, title):
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


def joblib_load(path: Path):
    import joblib

    return joblib.load(path)


if __name__ == "__main__":
    unittest.main()
