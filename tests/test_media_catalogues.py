from __future__ import annotations

import io
import json
import sqlite3
import sys
import threading
import tempfile
import unittest
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import requests

from mediamap import books, cli, collection, music, reduction, web


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
    return col.db.execute(f"SELECT COUNT(*) FROM {col.spec.table} WHERE {where}").fetchone()[0]


class MediaCatalogueTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        for target, name, value in [(collection, "ROOT", self.root), (music, "MIN_INTERVAL", 0)]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        quiet = patch("sys.stdout", new_callable=io.StringIO)  # the commands print a lot
        quiet.start()
        self.addCleanup(quiet.stop)
        FakeReducer.created.clear()
        reduction.load_reducer.cache_clear()
        self.addCleanup(reduction.load_reducer.cache_clear)

    # -- helpers ----------------------------------------------------------
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
        """Run `add` with provider lookups mocked; return the genre lookup mock."""
        with patch.object(music, "choose_album", side_effect=lambda *args: dict(record)), \
                patch.object(music, "fetch_genres", return_value=list(genres)) as fetch, \
                patch.object(music, "fetch_lastfm_tags", return_value=[]), \
                patch("builtins.input", side_effect=["title", "artist"]):
            col.add()
        return fetch

    # -- storage ----------------------------------------------------------
    def test_cd_and_lp_have_independent_databases(self):
        cds = self.open_collection(music.MusicSpec("cds"))
        lps = self.open_collection(music.MusicSpec("lps"))
        cds.db.execute("INSERT INTO albums(title,source_id) VALUES('CD release','cd-release')")
        cds.db.commit()

        self.assertNotEqual(cds.spec.db_path, lps.spec.db_path)
        self.assertEqual(count(cds), 1)
        self.assertEqual(count(lps), 0)

    def test_old_database_gains_new_columns_and_keeps_its_rows(self):
        spec = music.MusicSpec("cds")
        old = sqlite3.connect(spec.db_path)
        old.execute(OLD_ALBUMS_SCHEMA)
        old.execute("INSERT INTO albums(title,source_id) VALUES('Old','rel-old')")
        old.commit()
        old.close()

        col = self.open_collection(spec)

        columns = {row["name"] for row in col.db.execute("PRAGMA table_info(albums)")}
        self.assertLessEqual({"genres", "lastfm_tags", "release_group_id"}, columns)
        self.assertEqual(count(col), 1)

    # -- MusicBrainz search ----------------------------------------------
    @patch("mediamap.music.requests.get")
    def test_release_search_includes_all_physical_formats(self, get):
        get.return_value = self._musicbrainz_response([
            self._release("cd-release", "CD", "CD release"),
            self._release("lp-release", '12" Vinyl', "LP release"),
            self._release("unknown-release", None, "Unknown format"),
        ])

        records = music.search_releases("Album", "Artist")

        self.assertEqual(
            [record["source_id"] for record in records],
            ["cd-release", "lp-release", "unknown-release"],
        )
        self.assertEqual(records[0]["track_count"], 10)
        self.assertEqual(records[0]["artists"], ["Artist"])

    @patch("mediamap.music.requests.get")
    def test_unreported_formats_remain_selectable(self, get):
        get.return_value = self._musicbrainz_response([
            self._release("unknown-release", None, "Unknown format"),
        ])

        records = music.search_releases("Album", "Artist")

        self.assertEqual([record["source_id"] for record in records], ["unknown-release"])

    @patch("mediamap.music.requests.get")
    def test_release_search_reports_musicbrainz_http_errors(self, get):
        response = Mock()
        response.status_code = 503
        response.raise_for_status.side_effect = requests.HTTPError("503 Server Error")
        get.return_value = response

        with self.assertRaisesRegex(music.MusicBrainzError, "temporarily unavailable"):
            music.search_releases("Album", "Artist")

    def test_web_search_returns_gateway_error_when_musicbrainz_is_unavailable(self):
        handler = type(
            "TestLPHandler", (web.MediaMapHandler,),
            {"spec": music.MusicSpec("lps"), "log_message": lambda *args: None},
        )
        server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port)
        try:
            with patch.object(
                web, "search_releases",
                side_effect=music.MusicBrainzError("MusicBrainz is temporarily unavailable."),
            ):
                connection.request(
                    "POST", "/api/search", body=json.dumps({"title": "Album", "artist": "Artist"}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 502)
                self.assertIn("temporarily unavailable", json.loads(response.read())["error"])
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()

    @patch("mediamap.music.requests.get")
    def test_search_results_carry_the_ids_needed_for_genre_lookup(self, get):
        release = self._release("cd-release", "CD", "CD release")
        release["release-group"] = {"id": "rg-1"}
        release["artist-credit"] = [{"name": "Artist", "artist": {"id": "art-1", "name": "Artist"}}]
        get.return_value = self._musicbrainz_response([release])

        record = music.search_releases("Album", "Artist")[0]

        self.assertEqual((record["release_group_id"], record["artist_id"]), ("rg-1", "art-1"))

    # -- genres -----------------------------------------------------------
    def test_semantic_text_is_genre_led_and_ignores_release_details(self):
        record = album_record(genres=["post-punk", "new wave"])

        self.assertEqual(
            music._semantic_text(record),
            "Genres: post-punk, new wave\nArtist: Joy Division\nEra: 1970s",
        )
        bare = {"title": "Untitled", "artists": [], "year": "", "genres": []}
        self.assertEqual(music._semantic_text(bare), "Album: Untitled")

    def test_semantic_text_includes_lastfm_tags_separately(self):
        record = album_record(genres=["trip-hop"], lastfm_tags=["instrumental", "downtempo"])
        self.assertEqual(
            music._semantic_text(record),
            "Genres: trip-hop\nTags: instrumental, downtempo\nArtist: Joy Division\nEra: 1970s",
        )

    @patch("mediamap.music._mb_get")
    def test_fetch_genres(self, mb_get):
        data = {
            "release-group/rg": {"genres": [{"name": "Post-Punk", "count": 9}, {"name": "New Wave", "count": 4}]},
            "artist/art": {"genres": [{"name": "post-punk", "count": 5}, {"name": "gothic rock", "count": 3}]},
        }
        mb_get.side_effect = lambda url, params: data["/".join(url.split("/")[-2:])]

        # fewer than MIN_GENRES from the release group: topped up from the artist, deduplicated
        self.assertEqual(music.fetch_genres("rg", "art"), ["post-punk", "new wave", "gothic rock"])

        # a failed lookup is None ("unknown"), not [] ("no genres")
        mb_get.side_effect = requests.ConnectionError("down")
        self.assertIsNone(music.fetch_genres("rg", "art"))

    @patch.dict("os.environ", {"MEDIAMAP_LASTFM_API_KEY": "test-key"})
    @patch("mediamap.music.requests.get")
    def test_fetch_lastfm_album_tags(self, get):
        response = Mock()
        response.json.return_value = {
            "toptags": {"tag": [
                {"name": "Trip-Hop", "count": 3},
                {"name": "Electronic", "count": 12},
                {"name": "trip-hop", "count": 2},
            ]}
        }
        response.raise_for_status.return_value = None
        get.return_value = response

        self.assertEqual(
            music.fetch_lastfm_tags("Wax Tailor", "In the Mood for Life"),
            ["electronic", "trip-hop"],
        )
        self.assertEqual(get.call_args.kwargs["params"]["method"], "album.getTopTags")
        self.assertEqual(get.call_args.kwargs["params"]["artist"], "Wax Tailor")

    @patch.dict("os.environ", {"MEDIAMAP_LASTFM_API_KEY": ""})
    @patch("mediamap.music.requests.get")
    def test_missing_lastfm_key_leaves_album_tags_retryable(self, get):
        self.assertIsNone(music.fetch_lastfm_tags("Artist", "Album"))
        get.assert_not_called()

    # -- adding / projecting ---------------------------------------------
    def test_add_stores_genres_and_skips_lookups_for_duplicates(self):
        col = self.open_collection(music.MusicSpec("cds"))

        first = self.add_record(col, album_record())
        second = self.add_record(col, album_record())

        self.assertEqual((first.call_count, second.call_count), (1, 0))
        self.assertEqual(count(col), 1)
        row = col.db.execute("SELECT * FROM albums").fetchone()
        self.assertEqual(json.loads(row["genres"]), ["post-punk"])
        self.assertEqual(row["release_group_id"], "rg")
        self.assertIn("Genres: post-punk", row["semantic_text"])
        self.assertEqual(json.loads(row["lastfm_tags"]), [])

    def test_web_add_route_saves_selected_release_and_lists_it(self):
        spec = music.MusicSpec("lps")
        handler = type(
            "TestLPHandler", (web.MediaMapHandler,),
            {"spec": spec, "log_message": lambda *args: None},
        )
        server = web.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port)
        try:
            with patch.object(web, "fetch_genres", return_value=["post-punk"]), \
                    patch.object(web, "fetch_lastfm_tags", return_value=["instrumental"]):
                connection.request(
                    "POST", "/api/albums",
                    body=json.dumps({"record": album_record(), "genres": []}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 201)
                self.assertEqual(json.loads(response.read())["id"], 1)

            connection.request("GET", "/api/albums")
            response = connection.getresponse()
            albums = json.loads(response.read())
            self.assertEqual(response.status, 200)
            self.assertEqual(albums[0]["title"], "Unknown Pleasures")
            self.assertEqual(albums[0]["genres"], ["post-punk"])
            self.assertEqual(albums[0]["tags"], ["instrumental"])
        finally:
            connection.close()
            server.shutdown()
            thread.join()
            server.server_close()

    @patch("mediamap.collection.encode_texts", new=fake_encode)
    @patch.dict(sys.modules, FAKE_UMAP)
    def test_new_items_are_projected_into_the_existing_map(self):
        col = self.open_collection(music.MusicSpec("cds"))
        self.insert_albums(col, 3)
        col.build_map(neighbors=2)

        with patch("sys.stdout", new_callable=io.StringIO) as output:
            self.add_record(col, album_record(source_id="rel-new"))

        row = col.db.execute("SELECT x FROM albums WHERE source_id='rel-new'").fetchone()
        self.assertEqual(row["x"], 0.25)  # what FakeReducer.transform returns
        self.assertIn(
            "Suggested slot for this release: between Album 0 — Band [1975] "
            "and Album 1 — Band [1975].",
            output.getvalue(),
        )

    @patch("mediamap.music.fetch_lastfm_tags", return_value=["trip-hop"])
    @patch("mediamap.music._mb_get")
    def test_refresh_backfills_missing_metadata_once(self, mb_get, lastfm_tags):
        col = self.open_collection(music.MusicSpec("cds"))
        col.db.executemany(
            "INSERT INTO albums(title,artists,year,source_id,genres) VALUES(?,?,?,?,?)",
            [("Old", "Joy Division", "1979", "rel-a", ""), ("Done", "X", "1990", "rel-b", '["jazz"]')],
        )
        col.db.commit()
        release = {"release-group": {"id": "rg"},
                   "artist-credit": [{"artist": {"id": "art", "name": "Joy Division"}}]}
        genres = {"genres": [{"name": "Post-Punk", "count": 5}, {"name": "new wave", "count": 4},
                             {"name": "goth", "count": 3}]}

        def stored(source_id):
            return col.db.execute("SELECT * FROM albums WHERE source_id=?", (source_id,)).fetchone()

        def genres_down(url, params):
            if "/release/" in url:
                return release
            raise requests.ConnectionError("down")

        mb_get.side_effect = genres_down
        col.refresh()
        self.assertEqual(stored("rel-a")["genres"], "")  # failure leaves it retryable
        self.assertEqual(json.loads(stored("rel-a")["lastfm_tags"]), ["trip-hop"])

        mb_get.side_effect = lambda url, params: release if "/release/" in url else genres
        mb_get.reset_mock()
        col.refresh()
        self.assertEqual(json.loads(stored("rel-a")["genres"]), ["post-punk", "new wave", "goth"])
        self.assertIn("Tags: trip-hop", stored("rel-a")["semantic_text"])
        self.assertEqual(stored("rel-b")["genres"], '["jazz"]')  # untouched
        self.assertEqual(json.loads(stored("rel-b")["lastfm_tags"]), ["trip-hop"])
        self.assertIn("Tags: trip-hop", stored("rel-b")["semantic_text"])
        self.assertEqual(mb_get.call_count, 2)  # release lookup + release-group genres

        mb_get.reset_mock()
        col.refresh()
        mb_get.assert_not_called()
        self.assertEqual(lastfm_tags.call_count, 2)

    # -- maps -------------------------------------------------------------
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
        self.assertEqual(reducer.options["n_neighbors"], 3)  # at most n - 1

        path = self.root / "reducer.joblib"
        reduction.save_reducer(reducer, path)
        first = reduction.load_reducer(path)
        self.assertIs(first, reduction.load_reducer(path))
        reduction.save_reducer(reducer, path)
        self.assertIsNot(first, reduction.load_reducer(path))

    # -- cli --------------------------------------------------------------
    def test_cli_routes_each_collection_to_its_spec(self):
        with patch.object(cli, "run_command") as run_command:
            cli.main(["books", "add"])
            cli.main(["lps", "list"])

        (book_spec, book_args), (lp_spec, lp_args) = [c.args for c in run_command.call_args_list]
        self.assertIsInstance(book_spec, books.BookSpec)
        self.assertEqual(book_args.cmd, "add")
        self.assertIsInstance(lp_spec, music.MusicSpec)
        self.assertEqual(lp_spec.media_type, "lps")
        self.assertEqual(lp_args.cmd, "list")

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