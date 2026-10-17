import json
import os
from unittest.mock import Mock, patch

import requests

from mediamap import films

from _media_test_helpers import MediaTestCase, count


class FilmTests(MediaTestCase):
    @patch.dict(os.environ, {"MEDIAMAP_TMDB_API_KEY": "test-key"})
    @patch("mediamap.films.requests.get")
    def test_search_and_details_fetch_tmdb_movie_metadata(self, get):
        search_response = Mock()
        search_response.json.return_value = {
            "results": [{
                "id": 129,
                "title": "Spirited Away",
                "release_date": "2001-07-20",
                "overview": "A young girl enters a world of spirits.",
            }]
        }
        search_response.raise_for_status.return_value = None
        detail_response = Mock()
        detail_response.json.return_value = {
            "id": 129,
            "title": "Spirited Away",
            "release_date": "2001-07-20",
            "overview": "A young girl enters a world of spirits.",
            "genres": [{"name": "Animation"}, {"name": "Fantasy"}],
            "credits": {
                "crew": [{"job": "Director", "name": "Hayao Miyazaki"}],
                "cast": [{"name": "Rumi Hiiragi"}],
            },
        }
        detail_response.raise_for_status.return_value = None
        get.side_effect = [search_response, detail_response]

        matches = films.search_movies("Spirited Away")
        details = films.get_movie_details(matches[0]["source_id"])

        self.assertEqual(matches[0]["year"], "2001")
        self.assertEqual(details["genres"], ["Animation", "Fantasy"])
        self.assertEqual(details["directors"], ["Hayao Miyazaki"])
        self.assertEqual(details["cast"], ["Rumi Hiiragi"])
        self.assertEqual(get.call_args_list[0].kwargs["params"]["api_key"], "test-key")
        self.assertEqual(
            get.call_args_list[1].kwargs["params"]["append_to_response"], "credits"
        )

    @patch.dict(os.environ, {"MEDIAMAP_TMDB_API_KEY": ""})
    @patch("mediamap.films.requests.get")
    def test_missing_tmdb_key_reports_configuration_requirement(self, get):
        with self.assertRaisesRegex(films.FilmMetadataError, "MEDIAMAP_TMDB_API_KEY"):
            films.search_movies("A film")
        get.assert_not_called()

    @patch.dict(os.environ, {"MEDIAMAP_TMDB_API_KEY": "test-key"})
    @patch("mediamap.films.requests.get")
    def test_tmdb_http_failure_is_reported(self, get):
        get.side_effect = requests.ConnectionError("offline")

        with self.assertRaisesRegex(films.FilmMetadataError, "TMDb is unavailable"):
            films.search_movies("A film")

    def test_semantic_text_uses_story_metadata_not_physical_format(self):
        text = films._semantic_text({
            "title": "Spirited Away",
            "overview": "A young girl enters a world of spirits.",
            "genres": ["Animation", "Fantasy"],
            "directors": ["Hayao Miyazaki"],
            "cast": ["Rumi Hiiragi"],
            "owned_formats": ["DVD"],
        })

        self.assertIn("Synopsis:", text)
        self.assertIn("Genres: Animation, Fantasy", text)
        self.assertNotIn("DVD", text)

    def test_duplicate_film_addition_merges_owned_formats(self):
        collection = self.open_collection(films.FilmSpec())
        dvd = {
            "source_id": "129",
            "title": "Spirited Away",
            "year": "2001",
            "overview": "A young girl enters a world of spirits.",
            "genres": ["Animation"],
            "directors": ["Hayao Miyazaki"],
            "cast": ["Rumi Hiiragi"],
            "owned_formats": ["DVD"],
        }
        blu_ray = {**dvd, "owned_formats": ["Blu-ray"]}

        item_id = collection.add_record(dvd, enrich=False)
        same_item_id = collection.add_record(blu_ray, enrich=False)
        row = collection.db.execute("SELECT * FROM films").fetchone()

        self.assertEqual(item_id, same_item_id)
        self.assertEqual(count(collection), 1)
        self.assertEqual(
            json.loads(row["owned_formats"]), ["DVD", "Blu-ray"]
        )
        self.assertNotIn("DVD", row["semantic_text"])
        self.assertIn("DVD, Blu-ray", collection.describe(row))
