from __future__ import annotations

import os
import time
from typing import Any

import requests

from ..base import MetadataProvider, MetadataQuery

API = "https://musicbrainz.org/ws/2/"
TIMEOUT = 15
MIN_INTERVAL = 1.1
CONTACT_EMAIL = os.environ.get("MEDIAMAP_CONTACT_EMAIL", "").strip()
USER_AGENT = "MediaMap/0.1" + (
    f" (contact: {CONTACT_EMAIL})" if CONTACT_EMAIL else " (personal collection)"
)
_last_request = 0.0


class MusicBrainzProvider(MetadataProvider[dict[str, Any]]):
    name = "musicbrainz"

    @staticmethod
    def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
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
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("MusicBrainz returned an invalid JSON document.")
            return payload
        finally:
            _last_request = time.monotonic()

    def search(self, query: MetadataQuery) -> list[dict[str, Any]]:
        release_query = f'release:"{query.title or ""}"'
        if query.creator:
            release_query += f' AND artist:"{query.creator}"'
        payload = self._get(API + "release/", {"query": release_query, "limit": 25})
        releases = payload.get("releases", [])
        if not isinstance(releases, list):
            return []

        records = []
        for release in releases:
            if not isinstance(release, dict):
                continue
            media = release.get("media", [])
            artist_credit = release.get("artist-credit", [])
            formats = [
                str(medium.get("format") or "").strip()
                for medium in media
                if isinstance(medium, dict) and medium.get("format")
            ] if isinstance(media, list) else []
            artists = [
                str(credit["name"])
                for credit in artist_credit
                if isinstance(credit, dict) and credit.get("name")
            ] if isinstance(artist_credit, list) else []
            release_group = release.get("release-group") or {}
            artist_id = next(
                (
                    str((credit.get("artist") or {}).get("id"))
                    for credit in artist_credit
                    if isinstance(credit, dict)
                    and (credit.get("artist") or {}).get("id")
                    and (credit.get("artist") or {}).get("name") != "Various Artists"
                ),
                "",
            ) if isinstance(artist_credit, list) else ""
            record = {
                "source_id": str(release.get("id") or ""),
                "title": str(release.get("title") or query.title or ""),
                "artists": artists or ([query.creator] if query.creator else []),
                "year": str(release.get("date") or "")[:4],
                "label": ", ".join(
                    str(label.get("label", {}).get("name", ""))
                    for label in release.get("label-info", [])
                    if isinstance(label, dict) and label.get("label", {}).get("name")
                ),
                "release_format": ", ".join(formats),
                "country": str(release.get("country") or ""),
                "barcode": str(release.get("barcode") or ""),
                "release_group_id": str(release_group.get("id") or ""),
                "artist_id": artist_id,
                "track_count": sum(
                    int(medium.get("track-count") or 0)
                    for medium in media
                    if isinstance(medium, dict)
                ) if isinstance(media, list) else 0,
            }
            if record["source_id"]:
                records.append(record)
        return records

    def get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._get(url, params)
