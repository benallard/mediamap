from __future__ import annotations

import json
import sys
from typing import Any

import requests

from .collection import MediaSpec  # imported first: loads .env
from .metadata.base import MetadataQuery
from .metadata.providers.lastfm import LastFmProvider
from .metadata.providers.musicbrainz import MusicBrainzProvider

MUSICBRAINZ_API = "https://musicbrainz.org/ws/2/"
MUSICBRAINZ_URL = MUSICBRAINZ_API + "release/"
MAX_GENRES = 8
MIN_GENRES = 3  # fewer release-group genres than this: top up from the artist
VARIOUS_ARTISTS = "Various Artists"
MEDIA_NAMES = {"cds": "CD", "lps": "LP"}
MUSICBRAINZ_PROVIDER = MusicBrainzProvider()
LASTFM_PROVIDER = LastFmProvider()


class MusicBrainzError(RuntimeError):
    """A MusicBrainz lookup failed or returned an invalid response."""


def _mb_get(url: str, params: dict[str, Any]) -> Any:
    """Fetch a MusicBrainz document through its provider adapter."""
    return MUSICBRAINZ_PROVIDER.get(url, params)


def _artist_id(release: dict[str, Any]) -> str:
    """First credited artist's MBID ('Various Artists' has no useful genres)."""
    for credit in release.get("artist-credit", []):
        artist = credit.get("artist") if isinstance(credit, dict) else None
        if (
            isinstance(artist, dict)
            and artist.get("id")
            and artist.get("name") != VARIOUS_ARTISTS
        ):
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


def _fetch_genres(release_group_id: str, artist_id: str) -> list[str] | None:
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


def fetch_lastfm_tags(artist: str, album: str) -> list[str] | None:
    """Get an album's tags from Last.fm via the provider adapter."""
    return LASTFM_PROVIDER.search(MetadataQuery(title=album, creator=artist))


def _load_genres(raw: str) -> list[str]:
    try:
        value = json.loads(raw) if raw else []
    except ValueError:
        return []
    return [str(g) for g in value] if isinstance(value, list) else []


def search_releases(title: str, artist: str) -> list[dict[str, Any]]:
    """Return matching releases, handling provider failures for the CLI."""
    try:
        results = MUSICBRAINZ_PROVIDER.search(
            MetadataQuery(title=title, creator=artist)
        )
    except requests.RequestException as exc:
        print(f"MusicBrainz lookup failed: {exc}", file=sys.stderr)
        raise MusicBrainzError(
            "MusicBrainz is temporarily unavailable. Please try again shortly."
        ) from exc
    except ValueError as exc:
        print(f"Could not parse MusicBrainz response: {exc}", file=sys.stderr)
        raise MusicBrainzError(
            "MusicBrainz returned an invalid response. Please try again shortly."
        ) from exc
    return results


def fetch_genres(release_group_id: str, artist_id: str) -> list[str] | None:
    return _fetch_genres(release_group_id, artist_id)


def choose_album(media_type: str, title: str, artist: str) -> dict[str, Any] | None:
    try:
        records = search_releases(title, artist)
    except MusicBrainzError as exc:
        print(exc)
        return None
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
    """Genre/tag-led text; physical release details do not describe its sound."""
    parts = []
    if record.get("genres"):
        parts.append("Genres: " + ", ".join(record["genres"]))
    if record.get("lastfm_tags"):
        parts.append("Tags: " + ", ".join(record["lastfm_tags"]))
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
        "lastfm_tags",  # JSON list; "" = never fetched, "[]" = none found
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
        record["lastfm_tags"] = fetch_lastfm_tags(
            ", ".join(record["artists"]), record["title"]
        )
        if record["genres"]:
            print("Genres: " + ", ".join(record["genres"]))
        if record["lastfm_tags"]:
            print("Last.fm tags: " + ", ".join(record["lastfm_tags"]))
        if record["genres"] or record["lastfm_tags"]:
            return
        raw = input("No genres found. Enter some (comma-separated), or press Enter to skip: ")
        manual = [g.strip().lower() for g in raw.split(",") if g.strip()]
        if manual:
            record["genres"] = manual
        # else: stays [] (nothing exists) or None (lookup failed; `refresh` retries)

    def warn(self, record: dict[str, Any]) -> None:
        if not record.get("genres") and not record.get("lastfm_tags"):
            print(
                "Warning: no genres or Last.fm tags for this release; placement will be weak."
            )

    def text_from_row(self, row: Any) -> str | None:
        return _semantic_text({
            "title": row["title"],
            "artists": [row["artists"]] if row["artists"] else [],
            "year": row["year"],
            "genres": _load_genres(row["genres"]),
            "lastfm_tags": _load_genres(row["lastfm_tags"]),
        })

    def refresh_row(self, row: Any) -> dict[str, Any] | None:
        updates: dict[str, Any] = {}
        if row["genres"] == "":
            try:
                release = _mb_get(
                    f"{MUSICBRAINZ_URL}{row['source_id']}",
                    {"inc": "release-groups artist-credits"},
                )
            except (requests.RequestException, ValueError) as exc:
                print(f"MusicBrainz release lookup failed: {exc}", file=sys.stderr)
            else:
                group_id = str((release.get("release-group") or {}).get("id") or "")
                genres = fetch_genres(group_id, _artist_id(release))
                if genres is not None:
                    updates["release_group_id"] = group_id
                    updates["genres"] = json.dumps(genres)

        if row["lastfm_tags"] == "":
            tags = fetch_lastfm_tags(row["artists"], row["title"])
            if tags is not None:
                updates["lastfm_tags"] = json.dumps(tags)

        return updates or None

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
            "lastfm_tags": "" if record.get("lastfm_tags") is None
            else json.dumps(record["lastfm_tags"]),
        }

    def duplicate_key(self, record: dict[str, Any]) -> dict[str, str] | None:
        return {"source_id": record["source_id"]}