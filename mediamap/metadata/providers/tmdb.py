from __future__ import annotations

import os
import sys
from typing import Any

import requests

from ..base import MetadataProvider, MetadataQuery

TMDB_API = "https://api.themoviedb.org/3"
TIMEOUT = 15


class FilmMetadataError(RuntimeError):
    """A TMDb lookup failed or returned an invalid response."""


class TMDbProvider(MetadataProvider[dict[str, Any]]):
    name = "tmdb"

    @staticmethod
    def _get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        api_key = os.environ.get("MEDIAMAP_TMDB_API_KEY", "").strip()
        if not api_key:
            raise FilmMetadataError(
                "Set MEDIAMAP_TMDB_API_KEY to a TMDb API key before looking up films."
            )
        try:
            response = requests.get(
                f"{TMDB_API}{path}",
                params={"api_key": api_key, **(params or {})},
                headers={"Accept": "application/json"},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"TMDb lookup failed ({type(exc).__name__}).", file=sys.stderr)
            raise FilmMetadataError(
                "TMDb is unavailable or returned invalid data. Please try again shortly."
            ) from exc
        if not isinstance(payload, dict):
            raise FilmMetadataError("TMDb returned an invalid response.")
        return payload

    def search(self, query: MetadataQuery) -> list[dict[str, Any]]:
        if not query.title:
            return []
        payload = self._get(
            "/search/movie", {"query": query.title, "include_adult": "false"}
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise FilmMetadataError("TMDb returned an invalid movie result list.")
        return [
            {
                "source_id": str(movie["id"]),
                "title": str(movie["title"]),
                "year": str(movie.get("release_date") or "")[:4],
                "overview": str(movie.get("overview") or ""),
            }
            for movie in results[:20]
            if isinstance(movie, dict) and movie.get("id") and movie.get("title")
        ]

    def get_movie_details(self, source_id: str) -> dict[str, Any]:
        data = self._get(f"/movie/{source_id}", {"append_to_response": "credits"})
        credits_data = data.get("credits") or {}
        crew = credits_data.get("crew", []) if isinstance(credits_data, dict) else []
        cast = credits_data.get("cast", []) if isinstance(credits_data, dict) else []
        directors = [
            str(person["name"]) for person in crew
            if isinstance(person, dict) and person.get("job") == "Director" and person.get("name")
        ]
        actors = [
            str(person["name"]) for person in cast
            if isinstance(person, dict) and person.get("name")
        ][:8]
        genres = data.get("genres", [])
        return {
            "source_id": str(data.get("id") or source_id),
            "title": str(data.get("title") or ""),
            "year": str(data.get("release_date") or "")[:4],
            "overview": str(data.get("overview") or ""),
            "genres": [
                str(genre["name"]) for genre in genres
                if isinstance(genre, dict) and genre.get("name")
            ] if isinstance(genres, list) else [],
            "directors": directors,
            "cast": actors,
        }
