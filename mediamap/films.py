"""Film catalogue and TMDb metadata lookup."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from .collection import MediaSpec
from .metadata.base import MetadataQuery
from .metadata.providers.tmdb import FilmMetadataError, TMDbProvider

__all__ = [
    "FilmMetadataError",
    "FilmSpec",
    "get_movie_details",
    "search_movies",
]

FORMATS = ("DVD", "Blu-ray")
TMDB_PROVIDER = TMDbProvider()


def search_movies(query: str) -> list[dict[str, Any]]:
    return TMDB_PROVIDER.search(MetadataQuery(title=query))


def get_movie_details(source_id: str) -> dict[str, Any]:
    return TMDB_PROVIDER.get_movie_details(source_id)


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
