#!/usr/bin/env python3
"""BookMap: a small German/English semantic book catalogue.

Workflow:
  python library.py add          # add books by typing title and author
  python library.py build-map    # fit the initial, frozen UMAP map
  python library.py add          # later books are projected into the frozen map
  python library.py search TEXT  # search local catalogue by title/author

First run downloads the multilingual embedding model. Metadata lookup needs internet.
The fitted UMAP model is saved in map_model.joblib; the catalogue is library.sqlite3.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from metadata import USER_AGENT, lookup_metadata

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "library.sqlite3"
MODEL_PATH = HERE / "map_model.joblib"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS books (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        authors TEXT NOT NULL DEFAULT '',
        year TEXT NOT NULL DEFAULT '',
        isbn TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL DEFAULT '',
        source_id TEXT NOT NULL DEFAULT '',
        description TEXT NOT NULL DEFAULT '',
        subjects TEXT NOT NULL DEFAULT '',
        language TEXT NOT NULL DEFAULT '',
        semantic_text TEXT NOT NULL DEFAULT '',
        x REAL,
        y REAL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(source, source_id)
    )""")
    db.commit()
    return db


def get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Fetch JSON. Return None on failure so another metadata provider can be tried."""
    try:
        r = requests.get(url, params=params, timeout=15, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        if r.status_code == 429:
            relevant_headers = {
                name: value
                for name, value in r.headers.items()
                if name.lower() in {"date", "content-type", "retry-after", "www-authenticate", "server"}
                or any(term in name.lower() for term in ("rate", "quota"))
                or name.lower().startswith(("x-goog-", "x-google-"))
            }
            header_details = ", ".join(
                f"{name}: {value}" for name, value in sorted(relevant_headers.items())
            )
            suffix = f" Relevant headers: {header_details}." if header_details else ""
            body = r.text.strip()
            body_suffix = f" Response body: {body[:2000]}" if body else ""
            if len(body) > 2000:
                body_suffix += "…"
            print(f"Metadata provider rate-limited this request (HTTP 429): {url}.{suffix}{body_suffix}", file=sys.stderr)
            return None  # Do not hammer the provider; caller can try the next source.
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as e:
        print(f"Metadata lookup failed for {url}: {e}", file=sys.stderr)
        return None


def enrich_openlibrary(book: dict[str, Any]) -> dict[str, Any]:
    """Try to obtain a work description from Open Library when available."""
    key = book.get("source_id", "")
    if book.get("source") != "openlibrary" or not key.startswith("/works/"):
        return book
    data = get_json("https://openlibrary.org" + key + ".json")
    if data:
        desc = data.get("description", "")
        if isinstance(desc, dict): desc = desc.get("value", "")
        book["description"] = str(desc or "")
        if data.get("subjects"):
            book["subjects"] = list(dict.fromkeys((book.get("subjects") or []) + data["subjects"]))[:40]
    return book


def make_semantic_text(b: dict[str, Any]) -> str:
    # Do not overweight author/title: content and subject terms carry the semantics.
    parts = []
    if b.get("description"): parts.append("Book description: " + b["description"])
    subjects = b.get("subjects") or []
    if subjects: parts.append("Subjects and categories: " + "; ".join(subjects))
    if b.get("title"): parts.append("Title: " + b["title"])
    if b.get("authors"): parts.append("Author: " + ", ".join(b["authors"]))
    return "\n".join(parts)[:12000]


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.casefold()).strip()


def choose_candidate(title: str, author: str) -> dict[str, Any] | None:
    metadata = lookup_metadata(title=title, author=author)
    if not metadata["sources"]:
        print("No metadata result found from Google Books or Open Library.")
        print("Check your connection, wait if a provider rate-limited requests, or try a shorter title.")
        print("The book has not been saved yet; you can retry later.")
        return None

    source_ids = metadata.get("source_ids", {})
    source = next(
        (provider for provider in metadata["sources"] if source_ids.get(provider)),
        metadata["sources"][0],
    )
    published_date = metadata.get("published_date") or ""
    candidate = {
        "source": source,
        "source_id": source_ids.get(source, ""),
        "display_source": " + ".join(metadata["sources"]),
        "title": metadata.get("title") or title,
        "authors": metadata.get("authors") or ([author] if author else []),
        "year": published_date[:4],
        "isbn": metadata.get("isbn_13") or metadata.get("isbn_10") or "",
        "description": metadata.get("description") or "",
        "subjects": metadata.get("subjects") or [],
        "language": metadata.get("language") or "",
    }
    candidates = [candidate]

    print("\nMatching catalogue records:")
    for i, b in enumerate(candidates, 1):
        authors = ", ".join(b["authors"]) or "author unknown"
        print(f" {i:2}. {b['title']} — {authors} ({b['year'] or 'year unknown'}; {b.get('display_source', b['source'])})")
        if b.get("description"):
            snippet = re.sub(r"\s+", " ", b["description"])
            print("     " + (snippet[:210] + ("…" if len(snippet) > 210 else "")))
        elif b.get("subjects"):
            print("     Subjects: " + ", ".join(b["subjects"][:8]))
    print("  0. Don't use these results; cancel")
    while True:
        raw = input("Choose the matching edition/work number: ").strip()
        if raw == "0": return None
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return enrich_openlibrary(candidates[int(raw)-1])
        print("Please enter a number from the list.")


def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Missing dependencies. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(2)
    return SentenceTransformer(EMBEDDING_MODEL)


def encode_text(text: str):
    model = load_embedder()
    return model.encode([text], normalize_embeddings=True)[0]


def add_book(db: sqlite3.Connection) -> None:
    title = input("Title: ").strip()
    if not title: print("Cancelled: title is required."); return
    author = input("Author (optional): ").strip()
    candidates = choose_candidate(title, author)
    if not candidates: return
    b = candidates
    # Existing source record is a duplicate; return its current location.
    if b.get("source_id"):
        old = db.execute("SELECT * FROM books WHERE source=? AND source_id=?", (b["source"], b["source_id"])).fetchone()
        if old:
            print(f"Already catalogued: {old['title']} — {old['authors']}")
            print_location(old)
            return
    # Keep user-entered title if external metadata has an odd title variant, but store provider title too.
    semantic_text = make_semantic_text(b)
    if not b.get("description") and not b.get("subjects"):
        print("Warning: this record has no description or subjects; placement may be weak.")
    db.execute("""INSERT INTO books(title,authors,year,isbn,source,source_id,description,subjects,language,semantic_text)
        VALUES(?,?,?,?,?,?,?,?,?,?)""", (b.get("title") or title, ", ".join(b.get("authors") or ([author] if author else [])),
        b.get("year", ""), b.get("isbn", ""), b.get("source", ""), b.get("source_id", ""),
        b.get("description", ""), json.dumps(b.get("subjects", []), ensure_ascii=False), b.get("language", ""), semantic_text))
    db.commit()
    book_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    print(f"Added book #{book_id}: {b.get('title')}")
    if MODEL_PATH.exists():
        try:
            place_new_book(db, book_id, semantic_text)
        except Exception as e:
            print(f"Could not project into the existing map: {e}")
            print("Book saved without coordinates. Run `python library.py project-pending` after resolving the issue.")
    else:
        print("No map exists yet. Add a representative batch, then run `python library.py build-map`.")


def print_location(b: sqlite3.Row) -> None:
    if b["x"] is None or b["y"] is None:
        print("  Not mapped yet.")
    else:
        print(f"  Semantic coordinates: x={b['x']:.3f}, y={b['y']:.3f}")
        print("  These are relative map coordinates, not literal shelf measurements.")


def place_new_book(db: sqlite3.Connection, book_id: int, semantic_text: str) -> None:
    import joblib
    model = load_embedder()
    embedding = model.encode([semantic_text], normalize_embeddings=True)
    reducer = joblib.load(MODEL_PATH)
    xy = reducer.transform(embedding)[0]
    db.execute("UPDATE books SET x=?, y=? WHERE id=?", (float(xy[0]), float(xy[1]), book_id))
    db.commit()
    print(f"Projected into frozen map: x={xy[0]:.3f}, y={xy[1]:.3f}")
    nearby = db.execute("""SELECT title,authors,x,y FROM books WHERE id<>? AND x IS NOT NULL AND y IS NOT NULL
                          ORDER BY ((x-?)*(x-?)+(y-?)*(y-?)) LIMIT 5""",
                        (book_id, float(xy[0]), float(xy[0]), float(xy[1]), float(xy[1]))).fetchall()
    if nearby:
        print("Nearby mapped books:")
        for n in nearby: print(f"  • {n['title']} — {n['authors']} ({n['x']:.2f}, {n['y']:.2f})")


def build_map(db: sqlite3.Connection, neighbors: int) -> None:
    import joblib
    import numpy as np
    import umap
    rows = db.execute("SELECT id,semantic_text FROM books ORDER BY id").fetchall()
    if len(rows) < 5:
        print(f"Add at least 5 books before fitting UMAP (currently {len(rows)}). 20+ varied books is preferable.")
        return
    print(f"Embedding {len(rows)} books with {EMBEDDING_MODEL} …")
    model = load_embedder()
    texts = [r["semantic_text"] or "Book with no available description" for r in rows]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    n_neighbors = max(2, min(neighbors, len(rows)-1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.15,
                        metric="cosine", random_state=42, transform_seed=42)
    coords = reducer.fit_transform(np.asarray(vectors))
    joblib.dump(reducer, MODEL_PATH)
    db.executemany("UPDATE books SET x=?,y=? WHERE id=?", [(float(p[0]), float(p[1]), r["id"]) for r,p in zip(rows,coords)])
    db.commit()
    print(f"Saved frozen map to {MODEL_PATH.name}; coordinates stored in {DB_PATH.name}.")
    print("Important: do not rebuild this map if you want existing coordinates to remain stable.")
    print("A future `build-map` run intentionally refits and can move every book.")


def project_pending(db: sqlite3.Connection) -> None:
    if not MODEL_PATH.exists(): print("No map found. Run `python library.py build-map` first."); return
    rows = db.execute("SELECT id,semantic_text FROM books WHERE x IS NULL OR y IS NULL ORDER BY id").fetchall()
    for row in rows:
        place_new_book(db, row["id"], row["semantic_text"])


def search_books(db: sqlite3.Connection, query: str) -> None:
    like = f"%{query}%"
    rows = db.execute("SELECT * FROM books WHERE title LIKE ? OR authors LIKE ? ORDER BY title LIMIT 50", (like,like)).fetchall()
    if not rows: print("No matching local books."); return
    for b in rows:
        print(f"#{b['id']} {b['title']} — {b['authors']} [{b['year']}]")
        print_location(b)


def list_books(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT * FROM books ORDER BY x IS NULL, y, x, title").fetchall()
    for b in rows:
        xy = "unmapped" if b["x"] is None else f"x={b['x']:.3f}, y={b['y']:.3f}"
        print(f"#{b['id']:4} | {xy:23} | {b['title']} — {b['authors']}")
    print(f"Total: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen multilingual semantic map of your books.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add", help="manually enter a title/author and select external metadata")
    bm = sub.add_parser("build-map", help="fit UMAP on the current collection (refits all coordinates)")
    bm.add_argument("--neighbors", type=int, default=8, help="UMAP neighborhood size (default: 8)")
    s = sub.add_parser("search", help="search local title/author catalogue"); s.add_argument("query")
    sub.add_parser("list", help="list books sorted by semantic coordinates")
    sub.add_parser("project-pending", help="project saved but unmapped books into the frozen map")
    args = parser.parse_args()
    db = connect()
    try:
        if args.cmd == "add": add_book(db)
        elif args.cmd == "build-map": build_map(db, args.neighbors)
        elif args.cmd == "search": search_books(db, args.query)
        elif args.cmd == "list": list_books(db)
        elif args.cmd == "project-pending": project_pending(db)
    finally: db.close()

if __name__ == "__main__": main()
