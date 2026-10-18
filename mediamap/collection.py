"""Media-agnostic catalogue + frozen semantic map.

Everything that is the same for books, CDs, LPs and films lives here: the SQLite
table, the commands and the CLI. How items get positions (fitting a map, placing
newcomers into it) is delegated to a Placement (see placement.py). A media type
only supplies a MediaSpec (see books.py, films.py and music.py).
"""
from __future__ import annotations

import argparse
import math
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .embedding import encode_texts
from .placement import Placement, UmapPlacement

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # providers read os.environ, so load this early


def _embed(*args: Any, **kwargs: Any) -> Any:
    """The embedder handed to the placement. `encode_texts` is looked up when called, so
    replacing it in this module (as the tests do) takes effect."""
    return encode_texts(*args, **kwargs)


class MediaSpec(ABC):
    """Everything that differs between media types."""

    label: str                      # used in messages: "book", "CD", "LP"
    table: str
    creators_col: str               # "authors" / "artists"
    extra_columns: tuple[str, ...]  # media-specific TEXT columns
    unique: str                     # column list of the UNIQUE constraint
    db_name: str
    model_name: str
    dims: int = 2                   # dimensions of the destination space (1 or 2)
    min_items: int = 5              # smallest collection build-map accepts
    add_help: str = "look up an item and add it"

    # -- derived ----------------------------------------------------------
    @property
    def coord_cols(self) -> tuple[str, ...]:
        return ("x", "y")[: self.dims]

    @property
    def db_path(self) -> Path:
        return ROOT / self.db_name

    @property
    def model_path(self) -> Path:
        return ROOT / self.model_name

    def schema_sql(self) -> str:
        columns = [
            "id INTEGER PRIMARY KEY",
            "title TEXT NOT NULL",
            f"{self.creators_col} TEXT NOT NULL DEFAULT ''",
            "year TEXT NOT NULL DEFAULT ''",
            *(f"{name} TEXT NOT NULL DEFAULT ''" for name in self.extra_columns),
            "semantic_text TEXT NOT NULL DEFAULT ''",
            *(f"{name} REAL" for name in self.coord_cols),
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            f"UNIQUE({self.unique})",
        ]
        return f"CREATE TABLE IF NOT EXISTS {self.table} ({', '.join(columns)})"

    # -- to implement -----------------------------------------------------
    @abstractmethod
    def prompt_and_choose(self) -> dict[str, Any] | None:
        """Ask the user what to add, query providers, return the chosen record
        (a dict with at least a 'title') or None if cancelled."""

    @abstractmethod
    def semantic_text(self, record: dict[str, Any]) -> str:
        """The text that gets embedded."""

    @abstractmethod
    def to_row(self, record: dict[str, Any]) -> dict[str, Any]:
        """Column -> value for the INSERT (without semantic_text and coordinates)."""

    # -- optional hooks ---------------------------------------------------
    def duplicate_key(self, record: dict[str, Any]) -> dict[str, str] | None:
        """Columns identifying an already-catalogued item, or None to skip the check."""
        return None

    def duplicate_updates(
        self, existing: sqlite3.Row, record: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Updates to apply when an existing item is added again."""
        return None

    def enrich(self, record: dict[str, Any]) -> None:
        """Fetch extra metadata (in place) once the item is known not to be a duplicate."""

    def warn(self, record: dict[str, Any]) -> None:
        """Print warnings about a record before it is saved."""

    def display_details(self, row: sqlite3.Row) -> str:
        """Optional media-specific details included in catalogue listings."""
        return ""

    def text_from_row(self, row: sqlite3.Row) -> str | None:
        """Rebuild the embedded text from stored columns, so the recipe can change
        without re-querying providers. None = keep the stored semantic_text."""
        return None

    supports_refresh: bool = False

    def refresh_row(self, row: sqlite3.Row) -> dict[str, Any] | None:
        """Fetch metadata missing from an existing row. Return column -> value
        updates, or None/{} if there is nothing to do or the lookup failed."""
        return None


class Collection:
    def __init__(self, spec: MediaSpec, placement: Placement | None = None):
        self.spec = spec
        self.placement = placement or UmapPlacement(
            spec.model_path, spec.dims, embed=_embed, label=spec.label,
        )
        self.db = sqlite3.connect(spec.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute(spec.schema_sql())
        self._add_missing_columns()
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def _add_missing_columns(self) -> None:
        """Databases created before a spec grew new columns get them added."""
        existing = {r["name"] for r in self.db.execute(f"PRAGMA table_info({self.spec.table})")}
        for name in self.spec.extra_columns:
            if name not in existing:
                self.db.execute(
                    f"ALTER TABLE {self.spec.table} ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                )

    # -- helpers ----------------------------------------------------------
    def describe(self, row: sqlite3.Row) -> str:
        description = f"{row['title']} — {row[self.spec.creators_col]} [{row['year']}]"
        details = self.spec.display_details(row)
        return f"{description} ({details})" if details else description

    def position(self, row: sqlite3.Row) -> str:
        cols = self.spec.coord_cols
        if any(row[c] is None for c in cols):
            return "unmapped"
        return ", ".join(f"{c}={row[c]:.3f}" for c in cols)

    def _find_duplicate(self, record: dict[str, Any]) -> sqlite3.Row | None:
        key = self.spec.duplicate_key(record)
        if not key:
            return None
        where = " AND ".join(f"{col}=?" for col in key)
        return self.db.execute(
            f"SELECT * FROM {self.spec.table} WHERE {where}", tuple(key.values()),
        ).fetchone()

    # -- commands ---------------------------------------------------------
    def add(self) -> None:
        spec = self.spec
        record = spec.prompt_and_choose()
        if not record:
            return
        self.add_record(record)

    def add_record(self, record: dict[str, Any], *, enrich: bool = True) -> int | None:
        """Save a selected record, optionally running its interactive enrichment."""
        spec = self.spec
        old = self._find_duplicate(record)
        if old:
            updates = spec.duplicate_updates(old, record)
            if updates:
                assignments = ", ".join(f"{column}=?" for column in updates)
                self.db.execute(
                    f"UPDATE {spec.table} SET {assignments} WHERE id=?",
                    (*updates.values(), old["id"]),
                )
                self.db.commit()
                updated = self.db.execute(
                    f"SELECT * FROM {spec.table} WHERE id=?", (old["id"],)
                ).fetchone()
                print(f"Updated {spec.label}: {self.describe(updated)}")
                print(f"  Position: {self.position(updated)}")
                return old["id"]
            print(f"Already in this collection: {self.describe(old)}")
            print(f"  Position: {self.position(old)}")
            return None

        if enrich:
            spec.enrich(record)
        spec.warn(record)
        text = spec.semantic_text(record)
        values = {**spec.to_row(record), "semantic_text": text}
        cursor = self.db.execute(
            f"INSERT INTO {spec.table} ({', '.join(values)}) "
            f"VALUES ({', '.join('?' * len(values))})",
            tuple(values.values()),
        )
        self.db.commit()
        item_id = cursor.lastrowid
        print(f"Added {spec.label} #{item_id}: {record['title']}")

        if not self.placement.exists():
            print("No map exists yet. Add a representative batch, then run `build-map`.")
            return item_id
        try:
            self._project(item_id)
        except Exception as exc:  # the item is already saved; keep it
            print(f"Could not project into the existing map: {exc}")
            print("Saved without coordinates. Run `project-pending` after resolving the issue.")
        return item_id

    def _project(self, item_id: int) -> None:
        spec = self.spec
        row = self.db.execute(f"SELECT * FROM {spec.table} WHERE id=?", (item_id,)).fetchone()
        coords = list(self.placement.place(row))
        assignments = ", ".join(f"{c}=?" for c in spec.coord_cols)
        self.db.execute(
            f"UPDATE {spec.table} SET {assignments} WHERE id=?", (*coords, item_id),
        )
        self.db.commit()
        shown = ", ".join(f"{c}={v:.3f}" for c, v in zip(spec.coord_cols, coords))
        print(f"Projected into the frozen map: {shown}")

        if spec.dims == 1:
            rows = self.db.execute(
                f"SELECT * FROM {spec.table} WHERE {spec.coord_cols[0]} IS NOT NULL "
                f"ORDER BY {spec.coord_cols[0]}, title, id"
            ).fetchall()
            index = next(i for i, row in enumerate(rows) if row["id"] == item_id)
            previous = rows[index - 1] if index > 0 else None
            following = rows[index + 1] if index + 1 < len(rows) else None
            if previous and following:
                print(
                    f"Suggested slot for this release: between "
                    f"{self.describe(previous)} and {self.describe(following)}."
                )
            elif previous:
                print(f"Suggested slot for this release: after {self.describe(previous)}.")
            elif following:
                print(f"Suggested slot for this release: before {self.describe(following)}.")
            else:
                print("No other mapped releases yet; this is the first placement.")
            return

        mapped = self.db.execute(
            f"SELECT * FROM {spec.table} WHERE id<>? AND "
            + " AND ".join(f"{c} IS NOT NULL" for c in spec.coord_cols),
            (item_id,),
            ).fetchall()
        mapped.sort(key=lambda r: math.dist(coords, [r[c] for c in spec.coord_cols]))
        if mapped:
            print(f"Nearby {spec.label}s:")
            for row in mapped[:5]:
                print(f"  • {self.position(row)} | {self.describe(row)}")

    def build_map(self, neighbors: int) -> None:
        spec = self.spec
        rows = self.db.execute(f"SELECT * FROM {spec.table} ORDER BY id").fetchall()
        if len(rows) < spec.min_items:
            print(f"Add at least {spec.min_items} {spec.label}s before fitting the map "
                  f"(currently {len(rows)}). 20+ varied items is preferable.")
            return

        texts = [spec.text_from_row(r) or r["semantic_text"] for r in rows]
        items = [{**dict(r), "semantic_text": t} for r, t in zip(rows, texts, strict=True)]
        coords = self.placement.fit(items, neighbors)
        assignments = ", ".join(f"{c}=?" for c in spec.coord_cols)
        self.db.executemany(
            f"UPDATE {spec.table} SET {assignments}, semantic_text=? WHERE id=?",
            [(*point, text, row["id"])
             for row, point, text in zip(rows, coords, texts, strict=True)],
        )
        self.db.commit()
        print(f"Saved the frozen {spec.dims}-D map to {self.placement.saved_to}; "
              f"coordinates stored in {spec.db_path.name}.")
        print("Do not rebuild it casually: a future `build-map` refits and can move every item.")

    def project_pending(self) -> None:
        spec = self.spec
        if not self.placement.exists():
            print("No map found. Run `build-map` first.")
            return
        rows = self.db.execute(
            f"SELECT id FROM {spec.table} WHERE "
            + " OR ".join(f"{c} IS NULL" for c in spec.coord_cols) + " ORDER BY id"
        ).fetchall()
        for row in rows:
            self._project(row["id"])

    def refresh(self) -> None:
        """Backfill metadata for existing rows (e.g. after a spec gained columns)."""
        spec = self.spec
        if not spec.supports_refresh:
            print(f"`refresh` is not supported for {spec.label}s.")
            return
        rows = self.db.execute(f"SELECT * FROM {spec.table} ORDER BY id").fetchall()
        done = 0
        for row in rows:
            updates = spec.refresh_row(row)
            if not updates:
                continue
            assignments = ", ".join(f"{c}=?" for c in updates)
            self.db.execute(
                f"UPDATE {spec.table} SET {assignments} WHERE id=?",
                (*updates.values(), row["id"]),
            )
            fresh = self.db.execute(
                f"SELECT * FROM {spec.table} WHERE id=?", (row["id"],)
            ).fetchone()
            text = spec.text_from_row(fresh)
            if text is not None:
                self.db.execute(
                    f"UPDATE {spec.table} SET semantic_text=? WHERE id=?", (text, row["id"])
                )
            self.db.commit()  # per row, so an interrupted run keeps its progress
            done += 1
            print(f"  refreshed {self.describe(fresh)}")
        print(f"Refreshed {done} of {len(rows)} {spec.label}s.")
        if done:
            print("Texts changed: run `build-map` to refit the map (this refits all coordinates).")

    def list_all(self) -> None:
        cols = self.spec.coord_cols
        order = ", ".join(reversed(cols))  # 2-D: row-major (y, then x); 1-D: x
        rows = self.db.execute(
            f"SELECT * FROM {self.spec.table} ORDER BY {cols[0]} IS NULL, {order}, title"
        ).fetchall()
        for row in rows:
            print(f"#{row['id']:4} | {self.position(row):<22} | {self.describe(row)}")
        print(f"Total: {len(rows)}")

    def search(self, query: str) -> None:
        like = f"%{query}%"
        rows = self.db.execute(
            f"SELECT * FROM {self.spec.table} WHERE title LIKE ? OR "
            f"{self.spec.creators_col} LIKE ? ORDER BY title LIMIT 50",
            (like, like),
        ).fetchall()
        if not rows:
            print(f"No matching {self.spec.label}s in the local collection.")
            return
        for row in rows:
            print(f"#{row['id']:4} | {self.position(row):<22} | {self.describe(row)}")


def add_commands(parser: argparse.ArgumentParser, spec: MediaSpec) -> None:
    """Attach the collection commands (add, build-map, ...) to a parser."""
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="<command>")
    sub.add_parser("add", help=spec.add_help)
    build = sub.add_parser("build-map", help="fit the map on the current collection (refits all coordinates)")
    build.add_argument("--neighbors", type=int, default=8, help="UMAP neighborhood size (default: 8)")
    search = sub.add_parser("search", help="search the local collection by title/creator")
    search.add_argument("query")
    sub.add_parser("list", help="list items sorted by map position")
    sub.add_parser("project-pending", help="project saved but unmapped items into the frozen map")
    sub.add_parser("refresh", help="fetch metadata that existing items are missing")
    if getattr(spec, "media_type", None) == "lps" or spec.table == "books":
        medium = "LP" if getattr(spec, "media_type", None) == "lps" else "book"
        web = sub.add_parser("web", help=f"open a mobile-friendly {medium} catalogue over HTTPS")
        web.add_argument("--host", default="0.0.0.0", help="interface to listen on")
        web.add_argument("--port", type=int, default=8765, help="HTTPS port")
        web.add_argument("--cert", required=True, help="TLS certificate file (must be trusted by the phone)")
        web.add_argument("--key", required=True, help="TLS private key file")


def run_command(spec: MediaSpec, args: argparse.Namespace) -> None:
    """Run the command parsed by a parser built with add_commands."""
    collection = Collection(spec)
    try:
        actions = {
            "add": collection.add,
            "build-map": lambda: collection.build_map(args.neighbors),
            "search": lambda: collection.search(args.query),
            "list": collection.list_all,
            "project-pending": collection.project_pending,
            "refresh": collection.refresh,
        }
        actions[args.cmd]()
    finally:
        collection.close()