import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
TIMEOUT = 12
load_dotenv(ROOT / ".env")
CONTACT_EMAIL = os.environ.get("BOOKMAP_CONTACT_EMAIL", "").strip()
USER_AGENT = "MediaMap/0.1" + (
    f" (contact: {CONTACT_EMAIL})" if CONTACT_EMAIL else " (personal, low-volume use)"
)


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    if response.status_code == 429:
        relevant_headers = {
            name: value
            for name, value in response.headers.items()
            if name.lower() in {
                "date", "content-type", "retry-after", "www-authenticate", "server"
            }
            or any(term in name.lower() for term in ("rate", "quota"))
            or name.lower().startswith(("x-goog-", "x-google-"))
        }
        header_details = ", ".join(
            f"{name}: {value}" for name, value in sorted(relevant_headers.items())
        )
        suffix = f" Relevant headers: {header_details}." if header_details else ""
        body = response.text.strip()
        body_suffix = f" Response body: {body[:2000]}" if body else ""
        if len(body) > 2000:
            body_suffix += "…"
        print(
            f"Metadata provider rate-limited this request (HTTP 429): "
            f"{url}.{suffix}{body_suffix}",
            file=sys.stderr,
        )
    response.raise_for_status()
    return response.json()


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip() or None
    return value


def _google_books(isbn: str | None = None, title: str | None = None,
                  author: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"maxResults": 5}
    if isbn:
        params["q"] = f"isbn:{isbn.replace('-', '').strip()}"
    else:
        terms = []
        if title:
            terms.append(f'intitle:"{title}"')
        if author:
            terms.append(f'inauthor:"{author}"')
        params["q"] = " ".join(terms)

    key = os.getenv("GOOGLE_BOOKS_API_KEY")
    if key:
        params["key"] = key

    data = _get_json(GOOGLE_BOOKS_URL, params)
    items = data.get("items", [])
    if not items:
        return {}

    info = items[0].get("volumeInfo", {})
    identifiers = {
        identifier.get("type"): identifier.get("identifier")
        for identifier in info.get("industryIdentifiers", [])
    }
    return {
        "source_id": items[0].get("id", ""),
        "title": _clean(info.get("title")),
        "authors": info.get("authors", []),
        "description": _clean(info.get("description")),
        "subjects": info.get("categories", []),
        "publisher": _clean(info.get("publisher")),
        "published_date": _clean(info.get("publishedDate")),
        "language": _clean(info.get("language")),
        "isbn_10": identifiers.get("ISBN_10"),
        "isbn_13": identifiers.get("ISBN_13"),
        "source": "google",
    }


def _open_library(isbn: str | None = None, title: str | None = None,
                  author: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": 5}
    if isbn:
        params["isbn"] = isbn.replace("-", "").strip()
    else:
        if title:
            params["title"] = title
        if author:
            params["author"] = author

    data = _get_json(OPEN_LIBRARY_SEARCH_URL, params)
    docs = data.get("docs", [])
    if not docs:
        return {}

    # Prefer records with descriptions, subjects, and ISBNs.
    doc = max(
        docs,
        key=lambda result: (
            bool(result.get("key")),
            bool(result.get("subject")),
            bool(result.get("isbn")),
        ),
    )
    return {
        "source_id": doc.get("key", ""),
        "title": _clean(doc.get("title")),
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
        "source": "openlibrary",
    }


def lookup_metadata(isbn: str | None = None, title: str | None = None,
                    author: str | None = None) -> dict[str, Any]:
    """Return merged metadata; gracefully tolerate individual provider failures."""
    if not isbn and not title:
        raise ValueError("Provide at least an ISBN or a title.")

    records = []
    for provider in (_google_books, _open_library):
        try:
            record = provider(isbn=isbn, title=title, author=author)
            if record:
                records.append(record)
        except requests.HTTPError as exc:
            if exc.response is None or exc.response.status_code != 429:
                print(f"Metadata provider unavailable ({provider.__name__}): {exc}")
        except requests.RequestException as exc:
            print(f"Metadata provider unavailable ({provider.__name__}): {exc}")
        except (ValueError, KeyError, TypeError) as exc:
            print(f"Could not parse {provider.__name__} response: {exc}")

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

    # Use the richest description, and union subjects from all providers.
    richest = max(records, key=lambda record: len(record.get("description") or ""))
    merged = dict(richest)
    for field in ("title", "authors", "publisher", "published_date",
                  "language", "isbn_10", "isbn_13"):
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
