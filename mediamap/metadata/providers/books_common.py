from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
TIMEOUT = 12
load_dotenv(ROOT / ".env")
CONTACT_EMAIL = os.environ.get("MEDIAMAP_CONTACT_EMAIL", "").strip()
USER_AGENT = "MediaMap/0.1" + (
    f" (contact: {CONTACT_EMAIL})" if CONTACT_EMAIL else " (personal, low-volume use)"
)


def get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Metadata provider returned an invalid JSON document.")
    return payload


def clean(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip() or None
    return value
