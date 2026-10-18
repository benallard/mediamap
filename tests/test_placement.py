"""The placement seam: the UMAP strategy on its own, and Collection driving any strategy."""
from __future__ import annotations

from mediamap.collection import Collection
from mediamap.placement import Placement, UmapPlacement

from support import CatalogueTestCase, fake_encode, with_fake_models
from test_collection import ThingSpec


class UmapPlacementTests(CatalogueTestCase):
    def placement(self, dims=1, embed=fake_encode):
        return UmapPlacement(self.root / "m.joblib", dims, embed=embed, label="thing")

    @with_fake_models
    def test_fit_gives_one_coordinate_tuple_per_row_and_saves_the_model(self):
        placement = self.placement(dims=2)
        self.assertFalse(placement.exists())

        coords = placement.fit([{"semantic_text": f"t{i}"} for i in range(4)], neighbors=3)

        self.assertEqual(coords, [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
        self.assertTrue(placement.exists())
        self.assertEqual(placement.saved_to, "m.joblib")

    @with_fake_models
    def test_place_projects_a_new_row_through_the_saved_model(self):
        placement = self.placement(dims=2)
        placement.fit([{"semantic_text": f"t{i}"} for i in range(4)], neighbors=3)

        self.assertEqual(placement.place({"semantic_text": "new"}), (0.25, 0.25))

    @with_fake_models
    def test_rows_without_text_get_a_placeholder_so_they_can_still_be_embedded(self):
        seen = []

        def recording(texts, progress=False):
            seen.append(list(texts))
            return fake_encode(texts, progress)

        rows = [{"semantic_text": ""}, {"semantic_text": "x"}, {"semantic_text": "y"}]
        self.placement(embed=recording).fit(rows, neighbors=2)

        self.assertEqual(seen[0], ["thing with no available description", "x", "y"])


class MergingSpec(ThingSpec):
    """A spec whose duplicates are merged into the existing item (like films and their formats)."""

    def duplicate_updates(self, existing, record):
        return {"note": "merged"}


class RecordingPlacement(Placement):
    """A non-UMAP strategy: remembers what Collection hands it."""

    dims = 1
    saved_to = "recording"

    def __init__(self):
        self.ready, self.fitted, self.placed = False, None, []

    def exists(self):
        return self.ready

    def fit(self, rows, neighbors):
        self.ready, self.fitted = True, (list(rows), neighbors)
        return [(float(i),) for i in range(len(rows))]

    def place(self, row):
        self.placed.append(row["code"])
        return (9.0,)


class CollectionPlacementTests(CatalogueTestCase):
    def setUp(self):
        super().setUp()
        self.placement = RecordingPlacement()
        self.col = Collection(ThingSpec(), placement=self.placement)
        self.addCleanup(self.col.close)

    def insert_things(self, n):
        self.col.db.executemany(
            "INSERT INTO things(title,code,note) VALUES(?,?,?)",
            [(f"Thing {i}", f"t{i}", "x") for i in range(n)],
        )
        self.col.db.commit()

    def x_by_code(self):
        return dict(self.col.db.execute("SELECT code, x FROM things").fetchall())

    def add_thing(self, code):
        self.col.spec.next_record = {"code": code, "title": "T"}
        self.col.add()

    def test_collections_default_to_umap_on_the_specs_model_file(self):
        col = self.open_collection(ThingSpec())

        self.assertIsInstance(col.placement, UmapPlacement)
        self.assertEqual(col.placement.model_path, col.spec.model_path)
        self.assertEqual((col.placement.dims, col.placement.label), (1, "thing"))

    def test_build_map_fits_the_strategy_on_rebuilt_rows_and_stores_its_coordinates(self):
        self.insert_things(3)

        self.col.build_map(neighbors=5)

        rows, neighbors = self.placement.fitted
        self.assertEqual(neighbors, 5)
        self.assertEqual([r["semantic_text"] for r in rows], ["Thing 0: x", "Thing 1: x", "Thing 2: x"])
        self.assertEqual([r["code"] for r in rows], ["t0", "t1", "t2"])  # whole rows, not just text
        self.assertEqual(self.x_by_code(), {"t0": 0.0, "t1": 1.0, "t2": 2.0})
        self.assertIn("Saved the frozen 1-D map to recording", self.stdout.getvalue())

    def test_new_items_are_placed_only_once_the_strategy_has_been_fitted(self):
        self.add_thing("early")
        self.assertEqual(self.placement.placed, [])
        self.assertIn("No map exists yet", self.stdout.getvalue())

        self.insert_things(3)
        self.col.build_map(neighbors=2)
        self.add_thing("late")

        self.assertEqual(self.placement.placed, ["late"])  # it received the stored row
        self.assertEqual(self.x_by_code()["late"], 9.0)

    def test_project_pending_places_only_unmapped_items(self):
        self.insert_things(3)
        self.col.build_map(neighbors=2)
        self.col.db.execute("INSERT INTO things(title,code) VALUES('Late','late')")
        self.col.db.commit()

        self.col.project_pending()

        self.assertEqual(self.placement.placed, ["late"])
        self.assertEqual(self.x_by_code()["late"], 9.0)

    def test_merging_a_duplicate_neither_places_nor_moves_the_existing_item(self):
        col = Collection(MergingSpec(), placement=self.placement)
        self.addCleanup(col.close)
        col.db.executemany(
            "INSERT INTO things(title,code,note) VALUES(?,?,?)",
            [(f"Thing {i}", f"t{i}", "x") for i in range(3)],
        )
        col.db.commit()
        col.build_map(neighbors=2)
        before = dict(col.db.execute("SELECT code, x FROM things").fetchall())
        existing_id = col.db.execute("SELECT id FROM things WHERE code='t1'").fetchone()[0]

        merged_id = col.add_record({"code": "t1", "title": "Thing 1"}, enrich=False)

        self.assertEqual(merged_id, existing_id)
        self.assertEqual(self.placement.placed, [])  # nothing was placed again
        self.assertEqual(dict(col.db.execute("SELECT code, x FROM things").fetchall()), before)
        self.assertEqual(col.db.execute("SELECT note FROM things WHERE code='t1'").fetchone()[0], "merged")
        self.assertIn("Updated thing:", self.stdout.getvalue())