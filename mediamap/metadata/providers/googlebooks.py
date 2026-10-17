from __future__ import annotations

import os
from typing import Any

from ..base import MetadataProvider, MetadataQuery
from .books_common import clean, get_json

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"


class GoogleBooksProvider(MetadataProvider[dict[str, Any]]):
    name = "google"

    def search(self, query: MetadataQuery) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"maxResults": 5}
        if query.identifier:
            params["q"] = f"isbn:{query.identifier.replace('-', '').strip()}"
        else:
            terms = []
            if query.title:
                terms.append(f'intitle:"{query.title}"')
            if query.creator:
                terms.append(f'inauthor:"{query.creator}"')
            params["q"] = " ".join(terms)

        key = os.getenv("GOOGLE_BOOKS_API_KEY")
        if key:
            params["key"] = key

        data = get_json(GOOGLE_BOOKS_URL, params)
        items = data.get("items", [])
        if not items:
            return []

        item = items[0]
        info = item.get("volumeInfo", {})
        identifiers = {
            identifier.get("type"): identifier.get("identifier")
            for identifier in info.get("industryIdentifiers", [])
        }
        return [{
            "source_id": item.get("id", ""),
            "title": clean(info.get("title")),
            "authors": info.get("authors", []),
            "description": clean(info.get("description")),
            "subjects": info.get("categories", []),
            "publisher": clean(info.get("publisher")),
            "published_date": clean(info.get("publishedDate")),
            "language": clean(info.get("language")),
            "isbn_10": identifiers.get("ISBN_10"),
            "isbn_13": identifiers.get("ISBN_13"),
            "source": self.name,
        }]
