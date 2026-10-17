from __future__ import annotations

import sys
from typing import Any

import requests

from ..base import MetadataProvider, MetadataQuery
from .books_common import USER_AGENT, clean, get_json

OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"


class OpenLibraryProvider(MetadataProvider[dict[str, Any]]):
    name = "openlibrary"

    def search(self, query: MetadataQuery) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": 5}
        if query.identifier:
            params["isbn"] = query.identifier.replace("-", "").strip()
        else:
            if query.title:
                params["title"] = query.title
            if query.creator:
                params["author"] = query.creator

        data = get_json(OPEN_LIBRARY_SEARCH_URL, params)
        docs = data.get("docs", [])
        if not docs:
            return []

        doc = max(
            docs,
            key=lambda result: (
                bool(result.get("key")),
                bool(result.get("subject")),
                bool(result.get("isbn")),
            ),
        )
        return [{
            "source_id": doc.get("key", ""),
            "title": clean(doc.get("title")),
            "authors": doc.get("author_name", []),
            "description": None,
            "subjects": doc.get("subject", [])[:30],
            "publisher": (doc.get("publisher") or [None])[0],
            "published_date": str(doc.get("first_publish_year") or "") or None,
            "language": None,
            "isbn_10": next(
                (value for value in doc.get("isbn", []) if len(value) == 10), None
            ),
            "isbn_13": next(
                (value for value in doc.get("isbn", []) if len(value) == 13), None
            ),
            "source": self.name,
        }]

    @staticmethod
    def get_work(source_id: str) -> dict[str, Any] | None:
        if not source_id.startswith("/works/"):
            return None
        url = "https://openlibrary.org" + source_id + ".json"
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"Metadata lookup failed for {url}: {exc}", file=sys.stderr)
            return None
        return payload if isinstance(payload, dict) else None
