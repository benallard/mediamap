#!/usr/bin/env python3
"""
A first-pass semantic book mapper.

- Manual input: title + author (+ optional notes/description)
- Multilingual sentence embeddings, suitable for German and English books
- SQLite persistence: previously assigned coordinates never change
- Incremental 2D map: new books are placed near their semantic neighbours
- Re-entering a known title/author returns its saved coordinates

This creates a *semantic map*, not physical shelf coordinates yet. Once the
actual bookcase/shelf layout is known, map regions can be translated to shelves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).with_name("library.sqlite3")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def canonical_key(title: str, author: str) -> str:
    raw = f"{title.strip().casefold()}|{author.strip().casefold()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stable_key TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            author TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            embedding TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    return con


def get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "Missing dependency. Install with:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(f"Loading multilingual embedding model: {MODEL_NAME}", file=sys.stderr)
    return SentenceTransformer(MODEL_NAME)


def embed(model, title: str, author: str, notes: str):
    text = f"Title: {title}. Author: {author}."
    if notes.strip():
        text += f" Subject, description, or keywords: {notes.strip()}."
    return model.encode(text, normalize_embeddings=True).tolist()


def cosine(a, b) -> float:
    # Vectors are normalized by the model, so cosine similarity is their dot product.
    return sum(x * y for x, y in zip(a, b))


def stable_jitter(key: str, scale: float = 0.10):
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Deterministic values in [-scale, scale].
    a = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    b = int.from_bytes(digest[8:16], "big") / (2**64 - 1)
    return (2 * a - 1) * scale, (2 * b - 1) * scale


def place_incrementally(con, key: str, vector):
    rows = con.execute("SELECT id, title, author, embedding, x, y FROM books").fetchall()
    if not rows:
        # Deterministic starting point. Later additions are positioned relative to it.
        jx, jy = stable_jitter(key, 0.25)
        return 5.0 + jx, 5.0 + jy, []

    scored = []
    for row in rows:
        old_vec = json.loads(row["embedding"])
        scored.append((cosine(vector, old_vec), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    neighbours = scored[:min(5, len(scored))]

    # Similarity-weighted centroid of the closest existing books. The tiny
    # deterministic jitter keeps exact duplicates from receiving identical points.
    weights = [max(0.01, score - min(s for s, _ in neighbours) + 0.05)
               for score, _ in neighbours]
    total = sum(weights)
    x = sum(w * row["x"] for w, (_, row) in zip(weights, neighbours)) / total
    y = sum(w * row["y"] for w, (_, row) in zip(weights, neighbours)) / total
    jx, jy = stable_jitter(key, 0.16)
    x += jx
    y += jy

    # Keep a modest gap from existing points. Existing coordinates never move.
    # This is a simple first prototype, not a mathematically optimal layout.
    for _ in range(30):
        too_close = []
        for _, row in scored:
            dx, dy = x - row["x"], y - row["y"]
            d2 = dx * dx + dy * dy
            if d2 < 0.16:
                too_close.append((dx, dy, d2))
        if not too_close:
            break
        # Deterministic repulsion direction when points overlap exactly.
        rx = ry = 0.0
        for dx, dy, d2 in too_close:
            if d2 < 1e-10:
                a, b = stable_jitter(key + str(len(too_close)), 1.0)
                dx, dy = a or 0.01, b or 0.01
                d2 = dx * dx + dy * dy
            factor = (0.40 - d2 ** 0.5) / max(d2 ** 0.5, 1e-6)
            rx += dx * factor
            ry += dy * factor
        x += rx * 0.08
        y += ry * 0.08

    return x, y, [(score, row) for score, row in neighbours]


def add_book(title: str, author: str, notes: str):
    title, author = title.strip(), author.strip()
    if not title or not author:
        raise SystemExit("Both title and author are required.")
    key = canonical_key(title, author)
    con = connect()
    existing = con.execute("SELECT * FROM books WHERE stable_key = ?", (key,)).fetchone()
    if existing:
        print(f"Already catalogued: {existing['title']} — {existing['author']}")
        print(f"Saved semantic coordinates: x={existing['x']:.3f}, y={existing['y']:.3f}")
        print("Existing coordinates were not recalculated.")
        con.close()
        return

    model = get_model()
    vector = embed(model, title, author, notes)
    x, y, neighbours = place_incrementally(con, key, vector)
    con.execute(
        "INSERT INTO books (stable_key, title, author, notes, embedding, x, y) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (key, title, author, notes, json.dumps(vector), x, y),
    )
    con.commit()
    print(f"\nAdded: {title} — {author}")
    print(f"Semantic map coordinate: x={x:.3f}, y={y:.3f}")
    print("\nClosest existing books:")
    if not neighbours:
        print("  (This is the first book in the map.)")
    for score, row in neighbours:
        print(f"  {score:.3f}  {row['title']} — {row['author']} "
              f"(x={row['x']:.3f}, y={row['y']:.3f})")
    print("\nNote: these are semantic-map coordinates, not shelf labels yet.")
    con.close()


def list_books():
    con = connect()
    rows = con.execute("SELECT title, author, x, y FROM books ORDER BY x, y").fetchall()
    if not rows:
        print("No books yet. Add one with: python library.py add")
    else:
        for r in rows:
            print(f"({r['x']:8.3f}, {r['y']:8.3f})  {r['title']} — {r['author']}")
    con.close()


def find_book(query: str):
    con = connect()
    rows = con.execute(
        "SELECT title, author, x, y FROM books "
        "WHERE title LIKE ? OR author LIKE ? ORDER BY title",
        (f"%{query}%", f"%{query}%"),
    ).fetchall()
    if not rows:
        print("No matching books found.")
    for r in rows:
        print(f"{r['title']} — {r['author']}: x={r['x']:.3f}, y={r['y']:.3f}")
    con.close()


def main():
    parser = argparse.ArgumentParser(description="Incremental semantic book mapper")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="Add a book manually")
    add.add_argument("--title", help="Book title")
    add.add_argument("--author", help="Author")
    add.add_argument("--notes", default="", help="Optional subject/description/keywords")

    sub.add_parser("list", help="List saved books")
    find = sub.add_parser("find", help="Find an existing book")
    find.add_argument("query", help="Text to search in title or author")

    args = parser.parse_args()
    if args.command == "add":
        title = args.title or input("Title: ").strip()
        author = args.author or input("Author: ").strip()
        notes = args.notes
        if not notes:
            notes = input("Optional subject/description (Enter to skip): ").strip()
        add_book(title, author, notes)
    elif args.command == "list":
        list_books()
    elif args.command == "find":
        find_book(args.query)


if __name__ == "__main__":
    main()
