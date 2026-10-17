"""Books: Google Books / Open Library metadata plugged into the generic collection.

Only the book-specific parts live here: provider lookup, the text to embed,
and the table layout. Everything else is in collection.py.
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from .collection import MediaSpec
from .metadata.base import MetadataProvider, MetadataQuery
from .metadata.providers.googlebooks import GoogleBooksProvider
from .metadata.providers.openlibrary import OpenLibraryProvider

GOOGLE_BOOKS_PROVIDER = GoogleBooksProvider()
OPEN_LIBRARY_PROVIDER = OpenLibraryProvider()
BOOK_PROVIDERS: tuple[MetadataProvider[dict[str, Any]], ...] = (
    GOOGLE_BOOKS_PROVIDER,
    OPEN_LIBRARY_PROVIDER,
)


def enrich_openlibrary(book: dict[str, Any]) -> dict[str, Any]:
    """Add work-level description and subjects when Open Library has them."""
    key = book.get("source_id", "")
    if book.get("source") != "openlibrary" or not key.startswith("/works/"):
        return book
    data = OPEN_LIBRARY_PROVIDER.get_work(key)
    if data:
        description = data.get("description", "")
        if isinstance(description, dict):
            description = description.get("value", "")
        book["description"] = str(description or "")
        if data.get("subjects"):
            book["subjects"] = list(
                dict.fromkeys((book.get("subjects") or []) + data["subjects"])
            )[:40]
    return book


def lookup_metadata(
    isbn: str | None = None,
    title: str | None = None,
    author: str | None = None,
) -> dict[str, Any]:
    """Merge book metadata from configured providers, tolerating provider failures."""
    if not isbn and not title:
        raise ValueError("Provide at least an ISBN or a title.")

    records: list[dict[str, Any]] = []
    query = MetadataQuery(title=title, creator=author, identifier=isbn)
    for provider in BOOK_PROVIDERS:
        try:
            provider_records = provider.search(query)
            if provider_records:
                records.extend(provider_records)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                print(f"Metadata provider unavailable ({provider.name}): {exc}")
        except requests.RequestException as exc:
            print(f"Metadata provider unavailable ({provider.name}): {exc}")
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Could not parse {provider.name} response: {exc}")

    if not records:
        return {
            "title": title,
            "authors": [author] if author else [],
            "description": None,
            "subjects": [],
            "publisher": None,
            "published_date": None,
            "language": None,
            "isbn_10": None,
            "isbn_13": isbn if isbn and len(isbn) == 13 else None,
            "sources": [],
        }

    richest = max(records, key=lambda record: len(record.get("description") or ""))
    merged = dict(richest)
    for field in (
        "title", "authors", "publisher", "published_date",
        "language", "isbn_10", "isbn_13",
    ):
        if not merged.get(field):
            merged[field] = next(
                (record.get(field) for record in records if record.get(field)), None
            )

    merged["subjects"] = list(dict.fromkeys(
        subject
        for record in records
        for subject in record.get("subjects", [])
        if subject
    ))
    merged["sources"] = [record["source"] for record in records]
    merged["source_ids"] = {
        record["source"]: record["source_id"]
        for record in records
        if record.get("source_id")
    }
    merged.pop("source", None)
    merged.pop("source_id", None)
    return merged


def normalize_isbn(value: str) -> str:
    """Normalize and validate an ISBN-10 or ISBN-13 identifier."""
    isbn = re.sub(r"[\s-]", "", value).upper()
    if len(isbn) == 10 and isbn[:9].isdigit() and (isbn[-1].isdigit() or isbn[-1] == "X"):
        checksum = sum((10 - index) * (10 if char == "X" else int(char))
                       for index, char in enumerate(isbn))
        if checksum % 11 == 0:
            return isbn
    if len(isbn) == 13 and isbn.isdigit() and isbn.startswith(("978", "979")):
        checksum = sum(int(char) * (1 if index % 2 == 0 else 3)
                       for index, char in enumerate(isbn))
        if checksum % 10 == 0:
            return isbn
    raise ValueError("Enter a valid ISBN-10 or ISBN-13.")


def make_semantic_text(b: dict[str, Any]) -> str:
    # Do not overweight author/title: content and subject terms carry the semantics.
    parts = []
    if b.get("description"): parts.append("Book description: " + b["description"])
    subjects = b.get("subjects") or []
    if subjects: parts.append("Subjects and categories: " + "; ".join(subjects))
    if b.get("title"): parts.append("Title: " + b["title"])
    if b.get("authors"): parts.append("Author: " + ", ".join(b["authors"]))
    return "\n".join(parts)[:12000]


def choose_candidate(title: str, author: str) -> dict[str, Any] | None:
    metadata = lookup_metadata(title=title, author=author)
    if not metadata["sources"]:
        print("No metadata result found from Google Books or Open Library.")
        print("Check your connection, wait if a provider rate-limited requests, or try a shorter title.")
        print("The book has not been saved yet; you can retry later.")
        return None

    source_ids = metadata.get("source_ids", {})
    source = next(
        (provider for provider in metadata["sources"] if source_ids.get(provider)),
        metadata["sources"][0],
    )
    published_date = metadata.get("published_date") or ""
    candidate = {
        "source": source,
        "source_id": source_ids.get(source, ""),
        "display_source": " + ".join(metadata["sources"]),
        "title": metadata.get("title") or title,
        "authors": metadata.get("authors") or ([author] if author else []),
        "year": published_date[:4],
        "isbn": metadata.get("isbn_13") or metadata.get("isbn_10") or "",
        "description": metadata.get("description") or "",
        "subjects": metadata.get("subjects") or [],
        "language": metadata.get("language") or "",
    }
    candidates = [candidate]

    print("\nMatching catalogue records:")
    for i, b in enumerate(candidates, 1):
        authors = ", ".join(b["authors"]) or "author unknown"
        print(f" {i:2}. {b['title']} — {authors} ({b['year'] or 'year unknown'}; {b.get('display_source', b['source'])})")
        if b.get("description"):
            snippet = re.sub(r"\s+", " ", b["description"])
            print("     " + (snippet[:210] + ("…" if len(snippet) > 210 else "")))
        elif b.get("subjects"):
            print("     Subjects: " + ", ".join(b["subjects"][:8]))
    print("  0. Don't use these results; cancel")
    while True:
        raw = input("Choose the matching edition/work number: ").strip()
        if raw == "0": return None
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return enrich_openlibrary(candidates[int(raw)-1])
        print("Please enter a number from the list.")


class BookSpec(MediaSpec):
    label = "book"
    table = "books"
    creators_col = "authors"
    extra_columns = ("isbn", "source", "source_id", "description", "subjects", "language")
    unique = "source, source_id"
    db_name = "library.sqlite3"
    model_name = "map_model.joblib"
    dims = 2
    min_items = 5
    add_help = "manually enter a title/author and select external metadata"

    def prompt_and_choose(self) -> dict[str, Any] | None:
        title = input("Title: ").strip()
        if not title:
            print("Cancelled: title is required.")
            return None
        author = input("Author (optional): ").strip()
        return choose_candidate(title, author)

    def semantic_text(self, record: dict[str, Any]) -> str:
        return make_semantic_text(record)

    def to_row(self, b: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": b["title"],
            "authors": ", ".join(b.get("authors") or []),
            "year": b.get("year", ""),
            "isbn": b.get("isbn", ""),
            "source": b.get("source", ""),
            "source_id": b.get("source_id", ""),
            "description": b.get("description", ""),
            "subjects": json.dumps(b.get("subjects", []), ensure_ascii=False),
            "language": b.get("language", ""),
        }

    def duplicate_key(self, b: dict[str, Any]) -> dict[str, str] | None:
        if not b.get("source_id"):
            return None
        return {"source": b["source"], "source_id": b["source_id"]}

    def warn(self, b: dict[str, Any]) -> None:
        if not b.get("description") and not b.get("subjects"):
            print("Warning: this record has no description or subjects; placement may be weak.")