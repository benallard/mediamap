import unittest
from unittest.mock import patch

from mediamap.metadata.base import MetadataProvider, MetadataQuery
from mediamap.metadata.providers.googlebooks import GoogleBooksProvider
from mediamap.metadata.providers.lastfm import LastFmProvider
from mediamap.metadata.providers.musicbrainz import MusicBrainzProvider
from mediamap.metadata.providers.openlibrary import OpenLibraryProvider
from mediamap.metadata.providers.tmdb import TMDbProvider


class MetadataProviderTests(unittest.TestCase):
    def test_providers_share_the_common_interface(self):
        providers = (
            GoogleBooksProvider(),
            OpenLibraryProvider(),
            TMDbProvider(),
            MusicBrainzProvider(),
            LastFmProvider(),
        )

        self.assertTrue(all(isinstance(provider, MetadataProvider) for provider in providers))
        self.assertEqual(
            [provider.name for provider in providers],
            ["google", "openlibrary", "tmdb", "musicbrainz", "lastfm"],
        )

    @patch("mediamap.metadata.providers.googlebooks.get_json")
    def test_google_books_provider_uses_common_query_and_normalizes_results(self, get_json):
        get_json.return_value = {
            "items": [{
                "id": "volume-1",
                "volumeInfo": {
                    "title": "Example",
                    "authors": ["A. Writer"],
                    "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9780000000001"}],
                },
            }]
        }

        results = GoogleBooksProvider().search(
            MetadataQuery(title="Example", creator="A. Writer")
        )

        self.assertEqual(results[0]["source"], "google")
        self.assertEqual(results[0]["source_id"], "volume-1")
        self.assertEqual(get_json.call_args.args[1]["q"], 'intitle:"Example" inauthor:"A. Writer"')

    @patch("mediamap.metadata.providers.openlibrary.get_json")
    def test_openlibrary_provider_searches_by_identifier(self, get_json):
        get_json.return_value = {
            "docs": [{
                "key": "/works/OL1W",
                "title": "Example",
                "subject": ["Fiction"],
                "isbn": ["9780000000001"],
            }]
        }

        results = OpenLibraryProvider().search(
            MetadataQuery(identifier="978-0-306-40615-7")
        )

        self.assertEqual(results[0]["source"], "openlibrary")
        self.assertEqual(results[0]["source_id"], "/works/OL1W")
        self.assertEqual(get_json.call_args.args[1]["isbn"], "9780306406157")

    @patch("mediamap.metadata.providers.openlibrary.requests.get")
    def test_openlibrary_provider_fetches_work_details(self, get):
        response = get.return_value
        response.json.return_value = {"description": {"value": "A work description."}}

        result = OpenLibraryProvider.get_work("/works/OL1W")

        self.assertEqual(result["description"]["value"], "A work description.")
        self.assertEqual(get.call_args.args[0], "https://openlibrary.org/works/OL1W.json")


if __name__ == "__main__":
    unittest.main()
