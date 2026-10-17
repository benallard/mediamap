"""Film catalogue and TMDb metadata lookup."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from typing import Any

import requests

from .collection import MediaSpec

TMDB_API = "https://api.themoviedb.org/3"
TIMEOUT = 15
FORMATS = ("DVD", "Blu-ray")


class FilmMetadataError(RuntimeError):
    """A TMDb lookup failed or returned an invalid response."""


def _tmdb_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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


def search_movies(query: str) -> list[dict[str, Any]]:
    """Find TMDb movie matches by title."""
    payload = _tmdb_get("/search/movie", {"query": query, "include_adult": "false"})
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise FilmMetadataError("TMDb returned an invalid movie result list.")
    movies = []
    for movie in results[:20]:
        if not isinstance(movie, dict) or not movie.get("id") or not movie.get("title"):
            continue
        movies.append({
            "source_id": str(movie["id"]),
            "title": str(movie["title"]),
            "year": str(movie.get("release_date") or "")[:4],
            "overview": str(movie.get("overview") or ""),
        })
    return movies


def get_movie_details(source_id: str) -> dict[str, Any]:
    """Fetch descriptive metadata and director/cast for a selected TMDb movie."""
    data = _tmdb_get(f"/movie/{source_id}", {"append_to_response": "credits"})
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


def _decode_list(value: str) -> list[str]:
    try:
        decoded = json.loads(value) if value else []
    except ValueError:
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _semantic_text(film: dict[str, Any]) -> str:
    parts = []
    if film.get("overview"):
        parts.append("Synopsis: " + film["overview"])
    if film.get("genres"):
        parts.append("Genres: " + ", ".join(film["genres"]))
    if film.get("directors"):
        parts.append("Director: " + ", ".join(film["directors"]))
    if film.get("cast"):
        parts.append("Cast: " + ", ".join(film["cast"]))
    if film.get("title"):
        parts.append("Title: " + film["title"])
    return "\n".join(parts)[:12000]


class FilmSpec(MediaSpec):
    label = "film"
    table = "films"
    creators_col = "directors"
    extra_columns = ("source_id", "overview", "genres", "cast", "owned_formats")
    unique = "source_id"
    db_name = "films.sqlite3"
    model_name = "films_map_model.joblib"
    dims = 1
    min_items = 3
    add_help = "search TMDb and add a DVD or Blu-ray film"

    def prompt_and_choose(self) -> dict[str, Any] | None:
        title = input("Film title: ").strip()
        if not title:
            print("Cancelled: title is required.")
            return None
        movies = search_movies(title)
        if not movies:
            print("No matching films found on TMDb.")
            return None

        print("\nMatching films:")
        for index, movie in enumerate(movies, 1):
            snippet = " ".join(movie["overview"].split())
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            print(
                f" {index:2}. {movie['title']} "
                f"({movie['year'] or 'year unknown'})"
            )
            if snippet:
                print(f"     {snippet}")
        print("  0. Cancel")
        while True:
            raw = input("Choose the matching film number: ").strip()
            if raw == "0":
                return None
            if raw.isdigit() and 1 <= int(raw) <= len(movies):
                selected = get_movie_details(movies[int(raw) - 1]["source_id"])
                return {**movies[int(raw) - 1], **selected, "owned_formats": self._prompt_formats()}
            print("Please enter a number from the list.")

    @staticmethod
    def _prompt_formats() -> list[str]:
        while True:
            raw = input("Owned format (DVD, Blu-ray, or both): ").strip().lower()
            if raw in {"dvd", "d"}:
                return ["DVD"]
            if raw in {"blu-ray", "bluray", "blu ray", "b"}:
                return ["Blu-ray"]
            if raw in {"both", "dvd and blu-ray", "dvd, blu-ray"}:
                return list(FORMATS)
            print("Enter DVD, Blu-ray, or both.")

    def semantic_text(self, record: dict[str, Any]) -> str:
        return _semantic_text(record)

    def to_row(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": record["title"],
            "directors": ", ".join(record.get("directors") or []),
            "year": record.get("year", ""),
            "source_id": record["source_id"],
            "overview": record.get("overview", ""),
            "genres": json.dumps(record.get("genres") or [], ensure_ascii=False),
            "cast": json.dumps(record.get("cast") or [], ensure_ascii=False),
            "owned_formats": json.dumps(record.get("owned_formats") or [], ensure_ascii=False),
        }

    def duplicate_key(self, record: dict[str, Any]) -> dict[str, str] | None:
        return {"source_id": record["source_id"]}

    def duplicate_updates(
        self, existing: sqlite3.Row, record: dict[str, Any],
    ) -> dict[str, Any] | None:
        formats = _decode_list(existing["owned_formats"])
        added = [medium for medium in record.get("owned_formats", []) if medium not in formats]
        if not added:
            return None
        return {"owned_formats": json.dumps(formats + added, ensure_ascii=False)}

    def display_details(self, row: sqlite3.Row) -> str:
        formats = _decode_list(row["owned_formats"])
        return ", ".join(formats) if formats else "format unknown"

    def warn(self, record: dict[str, Any]) -> None:
        if not record.get("overview") and not record.get("genres"):
            print("Warning: this film has no synopsis or genres; placement may be weak.")
