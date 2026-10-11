from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
import umap
from dotenv import load_dotenv

from .embedding import load_embedder

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
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


def collection_paths(media_type: str) -> tuple[Path, Path]:
    return ROOT / f"{media_type}.sqlite3", ROOT / f"{media_type}_map_model.joblib"


def connect(media_type: str) -> sqlite3.Connection:
    db_path, _ = collection_paths(media_type)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS albums (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        artists TEXT NOT NULL DEFAULT '',
        year TEXT NOT NULL DEFAULT '',
        label TEXT NOT NULL DEFAULT '',
        release_format TEXT NOT NULL DEFAULT '',
        country TEXT NOT NULL DEFAULT '',
        barcode TEXT NOT NULL DEFAULT '',
        source_id TEXT NOT NULL UNIQUE,
        semantic_text TEXT NOT NULL DEFAULT '',
        x REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    db.commit()
    return db


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


def _embedding(texts: list[str], progress: bool = False) -> np.ndarray:
    model = load_embedder()
    return np.asarray(model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=progress,
    ))


def _save_record(db: sqlite3.Connection, media_type: str,
                 record: dict[str, Any]) -> None:
    duplicate = db.execute(
        "SELECT id,title,artists,x FROM albums WHERE source_id=?",
        (record["source_id"],),
    ).fetchone()
    if duplicate:
        location = (
            f"{duplicate['x']:.3f}" if duplicate["x"] is not None else "not mapped"
        )
        print(
            f"Already in this {MEDIA_NAMES[media_type]} collection: "
            f"{duplicate['title']} — {duplicate['artists']} (position {location})."
        )
        return

    text = _semantic_text(record, media_type)
    db.execute(
        """INSERT INTO albums
           (title,artists,year,label,release_format,country,barcode,source_id,semantic_text)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (
            record["title"], ", ".join(record["artists"]), record["year"],
            record["label"], record["release_format"], record["country"],
            record["barcode"], record["source_id"], text,
        ),
    )
    db.commit()
    album_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    print(f"Added {MEDIA_NAMES[media_type]} #{album_id}: {record['title']}")

    _, model_path = collection_paths(media_type)
    if model_path.exists():
        _project_one(db, album_id, text, model_path)
    else:
        print(f"No {MEDIA_NAMES[media_type]} map exists yet; add a collection and build it.")


def add_album(db: sqlite3.Connection, media_type: str) -> None:
    title = input("Album title: ").strip()
    if not title:
        print("Cancelled: title is required.")
        return
    artist = input("Artist (optional): ").strip()
    record = choose_album(media_type, title, artist)
    if record:
        _save_record(db, media_type, record)


def _project_one(db: sqlite3.Connection, album_id: int, text: str,
                 model_path: Path) -> None:
    reducer = joblib.load(model_path)
    coordinate = float(reducer.transform(_embedding([text]))[0][0])
    db.execute("UPDATE albums SET x=? WHERE id=?", (coordinate, album_id))
    db.commit()
    print(f"Projected onto the 1D map at {coordinate:.3f}.")

    nearby = db.execute(
        """SELECT title,artists,x FROM albums
           WHERE id<>? AND x IS NOT NULL ORDER BY ABS(x-?) LIMIT 5""",
        (album_id, coordinate),
    ).fetchall()
    if nearby:
        print("Nearby albums:")
        for album in nearby:
            print(f"  {album['x']:.3f} | {album['title']} — {album['artists']}")


def build_map(db: sqlite3.Connection, media_type: str, neighbors: int) -> None:
    rows = db.execute(
        "SELECT id,semantic_text FROM albums ORDER BY id"
    ).fetchall()
    if len(rows) < 3:
        print(f"Add at least 3 {MEDIA_NAMES[media_type]} before fitting a 1D map.")
        return

    print(f"Embedding {len(rows)} {MEDIA_NAMES[media_type]} albums …")
    vectors = _embedding(
        [row["semantic_text"] for row in rows],
        progress=True,
    )
    reducer = umap.UMAP(
        n_components=1,
        n_neighbors=max(2, min(neighbors, len(rows) - 1)),
        min_dist=0.15,
        metric="cosine",
        random_state=42,
        transform_seed=42,
    )
    coordinates = reducer.fit_transform(vectors).reshape(-1)
    _, model_path = collection_paths(media_type)
    joblib.dump(reducer, model_path)
    db.executemany(
        "UPDATE albums SET x=? WHERE id=?",
        [
            (float(coordinate), row["id"])
            for row, coordinate in zip(rows, coordinates, strict=True)
        ],
    )
    db.commit()
    print(f"Saved the 1D {MEDIA_NAMES[media_type]} map to {model_path.name}.")
    print("Rebuilding it can move every album on this collection's axis.")


def project_pending(db: sqlite3.Connection, media_type: str) -> None:
    _, model_path = collection_paths(media_type)
    if not model_path.exists():
        print(f"No {MEDIA_NAMES[media_type]} map found. Build one first.")
        return
    rows = db.execute(
        "SELECT id,semantic_text FROM albums WHERE x IS NULL ORDER BY id"
    ).fetchall()
    for row in rows:
        _project_one(db, row["id"], row["semantic_text"], model_path)


def list_albums(db: sqlite3.Connection, media_type: str) -> None:
    rows = db.execute(
        "SELECT * FROM albums ORDER BY x IS NULL,x,title"
    ).fetchall()
    for row in rows:
        position = "unmapped" if row["x"] is None else f"{row['x']:.3f}"
        print(f"{position:>10} | {row['title']} — {row['artists']} [{row['year']}]")
    print(f"Total: {len(rows)} {MEDIA_NAMES[media_type]} albums")


def search_albums(db: sqlite3.Connection, query: str) -> None:
    like = f"%{query}%"
    rows = db.execute(
        """SELECT * FROM albums WHERE title LIKE ? OR artists LIKE ?
           ORDER BY title LIMIT 50""",
        (like, like),
    ).fetchall()
    if not rows:
        print("No matching albums in this collection.")
        return
    for row in rows:
        position = "unmapped" if row["x"] is None else f"{row['x']:.3f}"
        print(f"{position:>10} | {row['title']} — {row['artists']} [{row['year']}]")


def main(media_type: str, argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=f"Manage the {media_type.upper()} collection.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="search MusicBrainz and add a release")
    build = commands.add_parser("build-map", help="fit this collection's frozen 1D map")
    build.add_argument("--neighbors", type=int, default=8)
    search = commands.add_parser("search", help="search this local collection")
    search.add_argument("query")
    commands.add_parser("list", help="list albums by 1D map position")
    commands.add_parser("project-pending", help="project saved albums into the map")
    args = parser.parse_args(argv)

    db = connect(media_type)
    try:
        if args.command == "add":
            add_album(db, media_type)
        elif args.command == "build-map":
            build_map(db, media_type, args.neighbors)
        elif args.command == "search":
            search_albums(db, args.query)
        elif args.command == "list":
            list_albums(db, media_type)
        elif args.command == "project-pending":
            project_pending(db, media_type)
    finally:
        db.close()
