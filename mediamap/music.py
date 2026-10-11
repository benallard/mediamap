from __future__ import annotations

import os
import re
import sys
from typing import Any

import requests

from .collection import MediaSpec, run_cli  # imported first: loads .env

MUSICBRAINZ_URL = "https://musicbrainz.org/ws/2/release/"
TIMEOUT = 15
USER_AGENT = "MediaMap/0.1" + (
    f" (contact: {os.environ['BOOKMAP_CONTACT_EMAIL'].strip()})"
    if os.environ.get("BOOKMAP_CONTACT_EMAIL", "").strip()
    else " (personal collection)"
)
FORMAT_FILTERS = {
    "cds": re.compile(r"\bcd\b|compact disc", re.IGNORECASE),
    "lps": re.compile(r"\bvinyl\b|\blp\b", re.IGNORECASE),
}
MEDIA_NAMES = {"cds": "CD", "lps": "LP"}


def _release_formats(release: dict[str, Any]) -> list[str]:
    return [
        str(medium.get("format") or "").strip()
        for medium in release.get("media", [])
        if medium.get("format")
    ]


def _artist_names(release: dict[str, Any]) -> list[str]:
    names = []
    for credit in release.get("artist-credit", []):
        if isinstance(credit, dict) and credit.get("name"):
            names.append(str(credit["name"]))
    return names


def search_releases(media_type: str, title: str, artist: str) -> list[dict[str, Any]]:
    query = f'release:"{title}"'
    if artist:
        query += f' AND artist:"{artist}"'
    try:
        response = requests.get(
            MUSICBRAINZ_URL,
            params={"query": query, "fmt": "json", "limit": 25},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"MusicBrainz lookup failed: {exc}", file=sys.stderr)
        return []
    except ValueError as exc:
        print(f"Could not parse MusicBrainz response: {exc}", file=sys.stderr)
        return []

    releases = payload.get("releases", [])
    if not isinstance(releases, list):
        print("MusicBrainz response did not contain a release list.", file=sys.stderr)
        return []

    formatted = []
    unformatted = []
    format_pattern = FORMAT_FILTERS[media_type]
    for release in releases:
        if not isinstance(release, dict):
            continue
        formats = _release_formats(release)
        record = {
            "source_id": str(release.get("id") or ""),
            "title": str(release.get("title") or title),
            "artists": _artist_names(release) or ([artist] if artist else []),
            "year": str(release.get("date") or "")[:4],
            "label": ", ".join(
                str(label.get("label", {}).get("name", ""))
                for label in release.get("label-info", [])
                if isinstance(label, dict) and label.get("label", {}).get("name")
            ),
            "release_format": ", ".join(formats),
            "country": str(release.get("country") or ""),
            "barcode": str(release.get("barcode") or ""),
            "track_count": sum(
                int(medium.get("track-count") or 0)
                for medium in release.get("media", [])
                if isinstance(medium, dict)
            ),
        }
        if not record["source_id"]:
            continue
        if formats and any(format_pattern.search(value) for value in formats):
            formatted.append(record)
        elif not formats:
            unformatted.append(record)

    # Prefer records explicitly matching the physical format. If MusicBrainz
    # has no format data for the search results, still offer those records.
    return formatted or unformatted


def choose_album(media_type: str, title: str, artist: str) -> dict[str, Any] | None:
    records = search_releases(media_type, title, artist)
    if not records:
        print(f"No {MEDIA_NAMES[media_type]} release matched in MusicBrainz.")
        print("Try a different title or artist; no album has been saved.")
        return None

    print(f"\nMatching {MEDIA_NAMES[media_type]} releases:")
    for index, record in enumerate(records, 1):
        creators = ", ".join(record["artists"]) or "artist unknown"
        details = [record["year"] or "year unknown"]
        if record["release_format"]:
            details.append(record["release_format"])
        if record["country"]:
            details.append(record["country"])
        print(f" {index:2}. {record['title']} — {creators} ({'; '.join(details)})")
        if record["label"]:
            print(f"     Label: {record['label']}")
        if record["track_count"]:
            print(f"     Tracks: {record['track_count']}")

    while True:
        raw = input("Choose the matching release number (0 to cancel): ").strip()
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(records):
            return records[int(raw) - 1]
        print("Please enter a number from the list.")


def _semantic_text(record: dict[str, Any], media_type: str) -> str:
    parts = [
        f"{MEDIA_NAMES[media_type]} album: {record['title']}",
        "Artist: " + ", ".join(record["artists"]),
        ]
    if record["label"]:
        parts.append("Record label: " + record["label"])
    if record["year"]:
        parts.append("Release year: " + record["year"])
    if record["release_format"]:
        parts.append("Physical format: " + record["release_format"])
    if record["track_count"]:
        parts.append(f"Number of tracks: {record['track_count']}")
    return "\n".join(parts)


class MusicSpec(MediaSpec):
    table = "albums"
    creators_col = "artists"
    extra_columns = ("label", "release_format", "country", "barcode", "source_id")
    unique = "source_id"
    dims = 1
    min_items = 3  # very low for UMAP; consider 20+
    add_help = "search MusicBrainz and add a release"

    def __init__(self, media_type: str):
        self.media_type = media_type
        self.label = MEDIA_NAMES[media_type]
        self.db_name = f"{media_type}.sqlite3"
        self.model_name = f"{media_type}_map_model.joblib"

    def prompt_and_choose(self) -> dict[str, Any] | None:
        title = input("Album title: ").strip()
        if not title:
            print("Cancelled: title is required.")
            return None
        artist = input("Artist (optional): ").strip()
        return choose_album(self.media_type, title, artist)

    def semantic_text(self, record: dict[str, Any]) -> str:
        return _semantic_text(record, self.media_type)

    def to_row(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": record["title"],
            "artists": ", ".join(record["artists"]),
            "year": record["year"],
            "label": record["label"],
            "release_format": record["release_format"],
            "country": record["country"],
            "barcode": record["barcode"],
            "source_id": record["source_id"],
        }

    def duplicate_key(self, record: dict[str, Any]) -> dict[str, str] | None:
        return {"source_id": record["source_id"]}


def main(media_type: str, argv: list[str] | None = None) -> None:
    run_cli(MusicSpec(media_type), argv)