import io
import sys
from unittest.mock import patch

import numpy as np

from mediamap import books, music, reduction

from _media_test_helpers import (
    FAKE_UMAP,
    FakeReducer,
    MediaTestCase,
    album_record,
    fake_encode,
    count,
    joblib_load,
)


class MapTests(MediaTestCase):
    @patch("mediamap.collection.encode_texts", new=fake_encode)
    @patch.dict(sys.modules, FAKE_UMAP)
    def test_new_items_are_projected_into_the_existing_map(self):
        col = self.open_collection(music.MusicSpec("cds"))
        self.insert_albums(col, 3)
        col.build_map(neighbors=2)

        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.add_record(col, album_record(source_id="rel-new"))

        row = col.db.execute("SELECT x FROM albums WHERE source_id='rel-new'").fetchone()
        self.assertEqual(row["x"], 0.25)
        self.assertIn(
            "Suggested slot for this release: between Album 0 — Band [1975] "
            "and Album 1 — Band [1975].",
            output.getvalue(),
        )

    @patch("mediamap.collection.encode_texts", new=fake_encode)
    @patch.dict(sys.modules, FAKE_UMAP)
    def test_cd_and_lp_maps_are_independent_and_one_dimensional(self):
        for media_type in ("cds", "lps"):
            col = self.open_collection(music.MusicSpec(media_type))
            self.insert_albums(col, 6)

            col.build_map(neighbors=3)

            self.assertTrue(col.spec.model_path.is_file())
            self.assertEqual(joblib_load(col.spec.model_path).n_components, 1)
            self.assertEqual(count(col, "x IS NOT NULL"), 6)

        self.assertNotEqual(
            music.MusicSpec("cds").model_path, music.MusicSpec("lps").model_path
        )

    @patch("mediamap.collection.encode_texts", new=fake_encode)
    @patch.dict(sys.modules, FAKE_UMAP)
    def test_build_map_rebuilds_text_from_stored_columns(self):
        col = self.open_collection(music.MusicSpec("cds"))
        self.insert_albums(col, 3, genres='["post-punk", "new wave"]', semantic_text="stale")

        col.build_map(neighbors=2)

        texts = {row[0] for row in col.db.execute("SELECT semantic_text FROM albums")}
        self.assertEqual(texts, {"Genres: post-punk, new wave\nArtist: Band\nEra: 1970s"})

    @patch("mediamap.collection.encode_texts", new=fake_encode)
    @patch.dict(sys.modules, FAKE_UMAP)
    def test_book_map_remains_two_dimensional(self):
        col = self.open_collection(books.BookSpec())
        col.db.executemany(
            "INSERT INTO books(title,source_id,semantic_text) VALUES(?,?,?)",
            [(f"Book {i}", f"book-{i}", f"book {i}") for i in range(5)],
        )

        col.build_map(neighbors=3)

        self.assertEqual(FakeReducer.created[-1].n_components, 2)
        self.assertEqual(count(col, "x IS NOT NULL AND y IS NOT NULL"), 5)

    @patch.dict(sys.modules, FAKE_UMAP)
    def test_reducer_neighbors_are_clamped_and_cache_is_invalidated_on_save(self):
        vectors = np.random.default_rng(0).normal(size=(4, 8))
        reducer, coordinates = reduction.fit_reducer(vectors, dims=1, neighbors=8)
        self.assertEqual(coordinates.shape, (4, 1))
        self.assertEqual(reducer.options["n_neighbors"], 3)

        path = self.root / "reducer.joblib"
        reduction.save_reducer(reducer, path)
        first = reduction.load_reducer(path)
        self.assertIs(first, reduction.load_reducer(path))
        reduction.save_reducer(reducer, path)
        self.assertIsNot(first, reduction.load_reducer(path))
