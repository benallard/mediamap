from __future__ import annotations

import os
import sys
from typing import Any

import requests

from ..base import MetadataProvider, MetadataQuery

API_URL = "https://ws.audioscrobbler.com/2.0/"
TIMEOUT = 15
MAX_TAGS = 8


class LastFmProvider(MetadataProvider[str]):
    name = "lastfm"

    def search(self, query: MetadataQuery) -> list[str] | None:
        api_key = os.environ.get("MEDIAMAP_LASTFM_API_KEY", "").strip()
        if not api_key:
            print(
                "Last.fm tags unavailable; set MEDIAMAP_LASTFM_API_KEY to enable them.",
                file=sys.stderr,
            )
            return None
        if not (query.creator or "").strip() or not (query.title or "").strip():
            return []

        try:
            response = requests.get(
                API_URL,
                params={
                    "method": "album.getTopTags",
                    "api_key": api_key,
                    "artist": query.creator,
                    "album": query.title,
                    "autocorrect": 1,
                    "format": "json",
                },
                headers={"Accept": "application/json"},
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Last.fm returned an invalid JSON document.")
            if "error" in payload:
                raise ValueError(f"Last.fm API error {payload['error']}.")
            tags = payload.get("toptags", {}).get("tag", [])
            if isinstance(tags, dict):
                tags = [tags]
            if not isinstance(tags, list):
                raise ValueError("Last.fm returned an invalid tag list.")
        except (requests.RequestException, ValueError, AttributeError) as exc:
            print(
                f"Last.fm album-tag lookup failed ({type(exc).__name__}).",
                file=sys.stderr,
            )
            return None

        def tag_count(tag: dict[str, Any]) -> int:
            try:
                return int(tag.get("count") or 0)
            except (TypeError, ValueError):
                return 0

        ranked = sorted(
            (tag for tag in tags if isinstance(tag, dict) and tag.get("name")),
            key=tag_count,
            reverse=True,
        )
        result = []
        for tag in ranked:
            name = str(tag["name"]).strip().lower()
            if name and name not in result:
                result.append(name)
            if len(result) == MAX_TAGS:
                break
        return result
