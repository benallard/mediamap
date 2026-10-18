"""SQLite persistence for media collections."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from .collection import MediaSpec


class SQLitePersistence:
    """Own the SQLite connection, schema, and catalogue queries for one media type."""

    def __init__(self, spec: MediaSpec, db_path: Path):
        self.spec = spec
        self.connection = sqlite3.connect(db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute(self._schema_sql())
        self._add_missing_columns()
        self.connection.commit()

    def _schema_sql(self) -> str:
        spec = self.spec
        columns = [
            "id INTEGER PRIMARY KEY",
            "title TEXT NOT NULL",
            f"{spec.creators_col} TEXT NOT NULL DEFAULT ''",
            "year TEXT NOT NULL DEFAULT ''",
            *(f"{name} TEXT NOT NULL DEFAULT ''" for name in spec.extra_columns),
            "semantic_text TEXT NOT NULL DEFAULT ''",
            *(f"{name} REAL" for name in spec.coord_cols),
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            f"UNIQUE({spec.unique})",
        ]
        return f"CREATE TABLE IF NOT EXISTS {spec.table} ({', '.join(columns)})"

    def _add_missing_columns(self) -> None:
        spec = self.spec
        existing = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({spec.table})")
        }
        for name in spec.extra_columns:
            if name not in existing:
                self.connection.execute(
                    f"ALTER TABLE {spec.table} ADD COLUMN {name} TEXT NOT NULL DEFAULT ''"
                )

    def close(self) -> None:
        self.connection.close()

    def find_by_key(self, key: Mapping[str, object]) -> sqlite3.Row | None:
        where = " AND ".join(f"{column}=?" for column in key)
        return self.connection.execute(
            f"SELECT * FROM {self.spec.table} WHERE {where}", tuple(key.values())
        ).fetchone()

    def get(self, item_id: int) -> sqlite3.Row:
        row = self.connection.execute(
            f"SELECT * FROM {self.spec.table} WHERE id=?", (item_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"No {self.spec.table} item found with id {item_id}.")
        return row

    def insert(self, values: Mapping[str, object]) -> int:
        cursor = self.connection.execute(
            f"INSERT INTO {self.spec.table} ({', '.join(values)}) "
            f"VALUES ({', '.join('?' * len(values))})",
            tuple(values.values()),
        )
        self.connection.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not return an ID for the inserted item.")
        return cursor.lastrowid

    def update(self, item_id: int, values: Mapping[str, object]) -> None:
        assignments = ", ".join(f"{column}=?" for column in values)
        self.connection.execute(
            f"UPDATE {self.spec.table} SET {assignments} WHERE id=?",
            (*values.values(), item_id),
        )
        self.connection.commit()

    def update_coordinates(self, item_id: int, coords: Sequence[float]) -> None:
        assignments = ", ".join(f"{column}=?" for column in self.spec.coord_cols)
        self.connection.execute(
            f"UPDATE {self.spec.table} SET {assignments} WHERE id=?",
            (*coords, item_id),
        )
        self.connection.commit()

    def save_map(self, entries: Iterable[tuple[int, Sequence[float], str]]) -> None:
        assignments = ", ".join(f"{column}=?" for column in self.spec.coord_cols)
        self.connection.executemany(
            f"UPDATE {self.spec.table} SET {assignments}, semantic_text=? WHERE id=?",
            [(*coords, text, item_id) for item_id, coords, text in entries],
        )
        self.connection.commit()

    def all_by_id(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            f"SELECT * FROM {self.spec.table} ORDER BY id"
        ).fetchall()

    def all_by_position(self) -> list[sqlite3.Row]:
        cols = self.spec.coord_cols
        order = ", ".join(reversed(cols))
        return self.connection.execute(
            f"SELECT * FROM {self.spec.table} "
            f"ORDER BY {cols[0]} IS NULL, {order}, title"
        ).fetchall()

    def mapped_by_primary_coordinate(self) -> list[sqlite3.Row]:
        coordinate = self.spec.coord_cols[0]
        return self.connection.execute(
            f"SELECT * FROM {self.spec.table} WHERE {coordinate} IS NOT NULL "
            f"ORDER BY {coordinate}, title, id"
        ).fetchall()

    def count(self) -> int:
        return self.connection.execute(
            f"SELECT COUNT(*) FROM {self.spec.table}"
        ).fetchone()[0]

    def pending_ids(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            f"SELECT id FROM {self.spec.table} WHERE "
            + " OR ".join(f"{column} IS NULL" for column in self.spec.coord_cols)
            + " ORDER BY id"
        ).fetchall()

    def mapped_except(self, item_id: int) -> list[sqlite3.Row]:
        return self.connection.execute(
            f"SELECT * FROM {self.spec.table} WHERE id<>? AND "
            + " AND ".join(f"{column} IS NOT NULL" for column in self.spec.coord_cols),
            (item_id,),
        ).fetchall()

    def search(self, query: str) -> list[sqlite3.Row]:
        like = f"%{query}%"
        return self.connection.execute(
            f"SELECT * FROM {self.spec.table} WHERE title LIKE ? OR "
            f"{self.spec.creators_col} LIKE ? ORDER BY title LIMIT 50",
            (like, like),
        ).fetchall()
