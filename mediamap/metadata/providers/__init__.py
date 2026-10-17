"""Adapters for external metadata services."""

from .googlebooks import GoogleBooksProvider
from .lastfm import LastFmProvider
from .musicbrainz import MusicBrainzProvider
from .openlibrary import OpenLibraryProvider
from .tmdb import TMDbProvider

__all__ = [
    "GoogleBooksProvider",
    "LastFmProvider",
    "MusicBrainzProvider",
    "OpenLibraryProvider",
    "TMDbProvider",
]
