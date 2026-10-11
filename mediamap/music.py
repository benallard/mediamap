from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import requests

from .collection import MediaSpec  # imported first: loads .env

MUSICBRAINZ_API = "https://musicbrainz.org/ws/2/"
MUSICBRAINZ_URL = MUSICBRAINZ_API + "release/"
TIMEOUT = 15
MIN_INTERVAL = 1.1  # MusicBrainz allows about one request per second
MAX_GENRES = 8
MIN_GENRES = 3  # fewer release-group genres than this: top up from the artist
VARIOUS_ARTISTS = "Various Artists"
USER_AGENT = "MediaMap/0.1" + (
    f" (contact: {os.environ['MEDIAMAP_CONTACT_EMAIL'].strip()})"
    if os.environ.get("MEDIAMAP_CONTACT_EMAIL", "").strip()
    else " (personal collection)"
)
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


_last_request = 0.0


def _mb_get(url: str, params: dict[str, Any]) -> Any:
    """GET a MusicBrainz JSON document, keeping to ~1 request/second.
    Raises requests.RequestException or ValueError on failure."""
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    try:
        response = requests.get(
            url,
            params={**params, "fmt": "json"},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        print(f"MusicBrainz API error: {exc}", file=sys.stderr)
        print(exc.response.json(), file=sys.stderr)
    finally:
        _last_request = time.monotonic()


def _artist_id(release: dict[str, Any]) -> str:
    """First credited artist's MBID ('Various Artists' has no useful genres)."""
    for credit in release.get("artist-credit", []):
        artist = credit.get("artist") if isinstance(credit, dict) else None
        if artist and artist.get("id") and artist.get("name") != VARIOUS_ARTISTS:
            return str(artist["id"])
    return ""


def _genres(kind: str, mbid: str) -> list[str] | None:
    """Genre names of a release-group or artist, most-voted first.
    [] if there is no MBID or no genres; None if the request failed."""
    if not mbid:
        return []
    try:
        data = _mb_get(f"{MUSICBRAINZ_API}{kind}/{mbid}", {"inc": "genres"})
    except (requests.RequestException, ValueError) as exc:
        print(f"MusicBrainz genre lookup failed ({kind}): {exc}", file=sys.stderr)
        return None
    genres = [g for g in data.get("genres", []) if isinstance(g, dict) and g.get("name")]
    genres.sort(key=lambda g: g.get("count", 0), reverse=True)
    return [str(g["name"]).lower() for g in genres]


def fetch_genres(release_group_id: str, artist_id: str) -> list[str] | None:
    """Genres for an album: its release group's, topped up with the artist's if
    there are only a few. None means a lookup failed (so don't record 'no genres')."""
    genres = _genres("release-group", release_group_id)
    if genres is None:
        return None
    if len(genres) < MIN_GENRES:
        extra = _genres("artist", artist_id)
        if extra is None:
            return None
        genres += [g for g in extra if g not in genres]
    return genres[:MAX_GENRES]


def _load_genres(raw: str) -> list[str]:
    try:
        value = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [str(g) for g in value] if isinstance(value, list) else []


def search_releases(title: str, artist: str) -> list[dict[str, Any]]:
    """Return matching releases without filtering by physical medium."""
    query = f'release:"{title}"'
    if artist:
        query += f' AND artist:"{artist}"'
    try:
        payload = _mb_get(MUSICBRAINZ_URL, {"query": query, "limit": 25})
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

    records = []
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
            "release_group_id": str((release.get("release-group") or {}).get("id") or ""),
            "artist_id": _artist_id(release),
            "track_count": sum(
                int(medium.get("track-count") or 0)
                for medium in release.get("media", [])
                if isinstance(medium, dict)
            ),
        }
        if not record["source_id"]:
            continue
        records.append(record)
    return records


def choose_album(media_type: str, title: str, artist: str) -> dict[str, Any] | None:
    records = search_releases(title, artist)
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


def _decade(year: str) -> str:
    return f"{year[:3]}0s" if len(year) >= 4 and year[:4].isdigit() else ""


def _semantic_text(record: dict[str, Any]) -> str:
    """Genre-led text for the embedder. Label, format and track count are left
    out on purpose: they say nothing about how a record sounds."""
    parts = []
    if record.get("genres"):
        parts.append("Genres: " + ", ".join(record["genres"]))
    if record["artists"]:
        parts.append("Artist: " + ", ".join(record["artists"]))
    if _decade(record["year"]):
        parts.append("Era: " + _decade(record["year"]))
    return "\n".join(parts) or f"Album: {record['title']}"


class MusicSpec(MediaSpec):
    table = "albums"
    creators_col = "artists"
    extra_columns = (
        "label", "release_format", "country", "barcode", "source_id",
        "release_group_id",
        "genres",  # JSON list; "" = never fetched, "[]" = none found
    )
    unique = "source_id"
    dims = 1
    min_items = 3  # very low for UMAP; consider 20+
    add_help = "search MusicBrainz and add a release"
    supports_refresh = True

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
        return _semantic_text(record)

    def enrich(self, record: dict[str, Any]) -> None:
        print("Looking up genres …")
        record["genres"] = fetch_genres(record["release_group_id"], record["artist_id"])
        if record["genres"]:
            print("Genres: " + ", ".join(record["genres"]))
            return
        raw = input("No genres found. Enter some (comma-separated), or press Enter to skip: ")
        manual = [g.strip().lower() for g in raw.split(",") if g.strip()]
        if manual:
            record["genres"] = manual
        # else: stays [] (nothing exists) or None (lookup failed; `refresh` retries)

    def warn(self, record: dict[str, Any]) -> None:
        if not record.get("genres"):
            print("Warning: no genres for this release; placement will be weak.")

    def text_from_row(self, row: Any) -> str | None:
        return _semantic_text({
            "title": row["title"],
            "artists": [row["artists"]] if row["artists"] else [],
            "year": row["year"],
            "genres": _load_genres(row["genres"]),
        })

    def refresh_row(self, row: Any) -> dict[str, Any] | None:
        if row["genres"] != "":  # already fetched, or entered by hand
            return None
        try:
            release = _mb_get(
                f"{MUSICBRAINZ_URL}{row['source_id']}",
                {"inc": "release-groups artist-credits"},
            )
        except (requests.RequestException, ValueError) as exc:
            print(f"MusicBrainz release lookup failed: {exc}", file=sys.stderr)
            return None
        group_id = str((release.get("release-group") or {}).get("id") or "")
        genres = fetch_genres(group_id, _artist_id(release))
        if genres is None:
            return None
        return {"release_group_id": group_id, "genres": json.dumps(genres)}

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
            "release_group_id": record["release_group_id"],
            "genres": "" if record.get("genres") is None else json.dumps(record["genres"]),
        }

    def duplicate_key(self, record: dict[str, Any]) -> dict[str, str] | None:
        return {"source_id": record["source_id"]}