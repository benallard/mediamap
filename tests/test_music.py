import json
import os
import sqlite3
from unittest.mock import Mock, patch

import requests

from mediamap import music

from _media_test_helpers import (
    OLD_ALBUMS_SCHEMA,
    MediaTestCase,
    album_record,
    count,
)


class MusicTests(MediaTestCase):
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

    @patch("mediamap.music.requests.get")
    def test_release_search_includes_all_physical_formats(self, get):
        get.return_value = self.musicbrainz_response([
            self.release("cd-release", "CD", "CD release"),
            self.release("lp-release", '12" Vinyl', "LP release"),
            self.release("unknown-release", None, "Unknown format"),
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
        get.return_value = self.musicbrainz_response([
            self.release("unknown-release", None, "Unknown format"),
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

    @patch("mediamap.music.requests.get")
    def test_search_results_carry_the_ids_needed_for_genre_lookup(self, get):
        release = self.release("cd-release", "CD", "CD release")
        release["release-group"] = {"id": "rg-1"}
        release["artist-credit"] = [{"name": "Artist", "artist": {"id": "art-1", "name": "Artist"}}]
        get.return_value = self.musicbrainz_response([release])

        record = music.search_releases("Album", "Artist")[0]

        self.assertEqual((record["release_group_id"], record["artist_id"]), ("rg-1", "art-1"))

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

        self.assertEqual(music.fetch_genres("rg", "art"), ["post-punk", "new wave", "gothic rock"])

        mb_get.side_effect = requests.ConnectionError("down")
        self.assertIsNone(music.fetch_genres("rg", "art"))

    @patch.dict(os.environ, {"MEDIAMAP_LASTFM_API_KEY": "test-key"})
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

    @patch.dict(os.environ, {"MEDIAMAP_LASTFM_API_KEY": ""})
    @patch("mediamap.music.requests.get")
    def test_missing_lastfm_key_leaves_album_tags_retryable(self, get):
        self.assertIsNone(music.fetch_lastfm_tags("Artist", "Album"))
        get.assert_not_called()

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
        self.assertEqual(stored("rel-a")["genres"], "")
        self.assertEqual(json.loads(stored("rel-a")["lastfm_tags"]), ["trip-hop"])

        mb_get.side_effect = lambda url, params: release if "/release/" in url else genres
        mb_get.reset_mock()
        col.refresh()
        self.assertEqual(json.loads(stored("rel-a")["genres"]), ["post-punk", "new wave", "goth"])
        self.assertIn("Tags: trip-hop", stored("rel-a")["semantic_text"])
        self.assertEqual(stored("rel-b")["genres"], '["jazz"]')
        self.assertEqual(json.loads(stored("rel-b")["lastfm_tags"]), ["trip-hop"])
        self.assertIn("Tags: trip-hop", stored("rel-b")["semantic_text"])
        self.assertEqual(mb_get.call_count, 2)

        mb_get.reset_mock()
        col.refresh()
        mb_get.assert_not_called()
        self.assertEqual(lastfm_tags.call_count, 2)
