"""The generic core, tested with a minimal fake media type (no provider code)."""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

from mediamap.collection import MediaSpec

from support import CatalogueTestCase, FakeReducer, count, with_fake_models


class ThingSpec(MediaSpec):
    label = "thing"
    table = "things"
    creators_col = "makers"
    extra_columns = ("code", "note")  # code identifies an item; note is filled in by enrich/refresh
    unique = "code"
    db_name = "things.sqlite3"
    model_name = "things_map.joblib"
    dims = 1
    min_items = 3
    supports_refresh = True

    def __init__(self):
        self.next_record = None
        self.enriched = []
        self.refresh_fails = False

    def prompt_and_choose(self):
        return dict(self.next_record)

    def semantic_text(self, record):
        return f"{record['title']}: {record.get('note', '')}"

    def to_row(self, record):
        return {"title": record["title"], "makers": "Maker", "year": "2000",
                "code": record["code"], "note": record.get("note", "")}

    def duplicate_key(self, record):
        return {"code": record["code"]}

    def enrich(self, record):
        self.enriched.append(record["code"])
        record["note"] = "enriched"

    def text_from_row(self, row):
        return f"{row['title']}: {row['note']}"

    def refresh_row(self, row):
        if row["note"] or self.refresh_fails:
            return None
        return {"note": "filled"}


class PlaneSpec(ThingSpec):
    dims = 2
    db_name = "plane.sqlite3"
    model_name = "plane_map.joblib"


class CollectionTests(CatalogueTestCase):
    def add_thing(self, col, code="a", title="Title"):
        col.spec.next_record = {"code": code, "title": title}
        col.add()

    def insert_things(self, col, n, note=""):
        col.db.executemany(
            "INSERT INTO things(title,code,note,semantic_text) VALUES(?,?,?,?)",
            [(f"Thing {i}", f"t{i}", note, "stale") for i in range(n)],
        )
        col.db.commit()

    def x_by_code(self, col):
        return dict(col.db.execute("SELECT code, x FROM things").fetchall())

    # -- add --------------------------------------------------------------
    def test_add_enriches_and_stores_the_record(self):
        col = self.open_collection(ThingSpec())

        self.add_thing(col)

        row = col.db.execute("SELECT * FROM things").fetchone()
        self.assertEqual((row["code"], row["note"]), ("a", "enriched"))
        self.assertEqual(row["semantic_text"], "Title: enriched")

    def test_duplicates_are_neither_enriched_nor_saved_again(self):
        col = self.open_collection(ThingSpec())

        self.add_thing(col)
        self.add_thing(col)

        self.assertEqual(col.spec.enriched, ["a"])
        self.assertEqual(count(col), 1)
        self.assertIn("Already in this collection", self.stdout.getvalue())

    # -- map --------------------------------------------------------------
    @with_fake_models
    def test_map_has_the_dimensions_the_spec_asks_for(self):
        for spec in (ThingSpec(), PlaneSpec()):
            with self.subTest(dims=spec.dims):
                col = self.open_collection(spec)
                self.insert_things(col, 4)

                col.build_map(neighbors=3)

                self.assertEqual(FakeReducer.created[-1].n_components, spec.dims)
                mapped = " AND ".join(f"{c} IS NOT NULL" for c in spec.coord_cols)
                self.assertEqual(count(col, mapped), 4)

    @with_fake_models
    def test_build_map_rebuilds_text_from_stored_columns(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 3, note="x")

        col.build_map(neighbors=2)

        texts = {row[0] for row in col.db.execute("SELECT semantic_text FROM things")}
        self.assertEqual(texts, {"Thing 0: x", "Thing 1: x", "Thing 2: x"})

    @with_fake_models
    def test_new_items_are_projected_into_the_existing_map(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 3)
        col.build_map(neighbors=2)

        self.add_thing(col, code="new")

        self.assertEqual(self.x_by_code(col)["new"], 0.25)  # what FakeReducer.transform returns

    @with_fake_models
    def test_project_pending_only_touches_unmapped_items(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 3)
        col.build_map(neighbors=2)
        col.db.execute("INSERT INTO things(title,code) VALUES('Late','late')")
        col.db.commit()

        col.project_pending()

        self.assertEqual(self.x_by_code(col), {"t0": 0.0, "t1": 1.0, "t2": 2.0, "late": 0.25})

    # -- schema -----------------------------------------------------------
    def test_old_database_gains_new_columns_and_keeps_its_rows(self):
        spec = ThingSpec()
        old = sqlite3.connect(spec.db_path)
        old.execute("""CREATE TABLE things (
            id INTEGER PRIMARY KEY, title TEXT NOT NULL, makers TEXT NOT NULL DEFAULT '',
            year TEXT NOT NULL DEFAULT '', code TEXT NOT NULL,
            semantic_text TEXT NOT NULL DEFAULT '', x REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(code))""")
        old.execute("INSERT INTO things(title,code) VALUES('Old','old')")
        old.commit()
        old.close()

        col = self.open_collection(spec)

        columns = {row["name"] for row in col.db.execute("PRAGMA table_info(things)")}
        self.assertIn("note", columns)
        self.assertEqual(count(col), 1)

    # -- refresh ----------------------------------------------------------
    def test_refresh_fills_in_missing_metadata_and_rebuilds_the_text(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 2)
        col.db.execute("UPDATE things SET note='kept' WHERE code='t1'")
        col.db.commit()

        col.refresh()

        rows = {r["code"]: r for r in col.db.execute("SELECT * FROM things")}
        self.assertEqual((rows["t0"]["note"], rows["t1"]["note"]), ("filled", "kept"))
        self.assertEqual(rows["t0"]["semantic_text"], "Thing 0: filled")

    def test_failed_refresh_leaves_the_item_retryable(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 1)

        col.spec.refresh_fails = True
        col.refresh()
        self.assertEqual(col.db.execute("SELECT note FROM things").fetchone()[0], "")

        col.spec.refresh_fails = False
        col.refresh()
        self.assertEqual(col.db.execute("SELECT note FROM things").fetchone()[0], "filled")

    # -- behaviour pinned before the placement refactor --------------------
    @with_fake_models
    def test_one_dimensional_placement_suggests_a_slot_between_neighbours(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 3)
        col.build_map(neighbors=2)

        self.add_thing(col, code="new")

        out = self.stdout.getvalue()
        self.assertIn("Projected into the frozen map: x=0.250", out)
        self.assertIn("Suggested slot for this release: between Thing 0", out)
        self.assertIn("and Thing 1", out)

    @with_fake_models
    def test_two_dimensional_placement_lists_nearby_items(self):
        col = self.open_collection(PlaneSpec())
        self.insert_things(col, 4)
        col.build_map(neighbors=3)

        self.add_thing(col, code="new")

        out = self.stdout.getvalue()
        self.assertIn("Projected into the frozen map: x=0.250, y=0.250", out)
        self.assertIn("Nearby things:", out)

    def test_items_added_before_a_map_exists_stay_unmapped(self):
        col = self.open_collection(ThingSpec())

        self.add_thing(col)
        self.assertIn("No map exists yet", self.stdout.getvalue())
        self.assertIsNone(col.db.execute("SELECT x FROM things").fetchone()[0])

        col.project_pending()
        self.assertIn("No map found. Run `build-map` first.", self.stdout.getvalue())

    @with_fake_models
    def test_build_map_reports_where_the_map_was_saved(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 3)

        col.build_map(neighbors=2)

        self.assertIn("Saved the frozen 1-D map to things_map.joblib; "
                      "coordinates stored in things.sqlite3.", self.stdout.getvalue())

    def test_build_map_refuses_a_collection_that_is_too_small(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 2)

        col.build_map(neighbors=2)

        self.assertIn("Add at least 3 things", self.stdout.getvalue())
        self.assertFalse(col.spec.model_path.exists())

    @with_fake_models
    def test_failed_placement_keeps_the_item_for_project_pending(self):
        col = self.open_collection(ThingSpec())
        self.insert_things(col, 3)
        col.build_map(neighbors=2)

        with patch("mediamap.collection.encode_texts", side_effect=RuntimeError("boom")):
            self.add_thing(col, code="late")
        self.assertIn("Could not project into the existing map: boom", self.stdout.getvalue())
        self.assertIsNone(self.x_by_code(col)["late"])

        col.project_pending()
        self.assertEqual(self.x_by_code(col)["late"], 0.25)
