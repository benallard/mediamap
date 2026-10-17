from __future__ import annotations

import unittest

import numpy as np

from research import distances
from research.distances import Block


class DistanceTests(unittest.TestCase):
    def setUp(self):
        years = np.array([1970, 1971, 2000])
        self.blocks = [
            Block("era", distances.era_block(years, sigma=10), 0.25),
            Block("creator", distances.creator_block(["a", "a", "b"]), 0.75),
        ]

    def test_combined_distance_is_symmetric_with_a_zero_diagonal(self):
        D = distances.combine(self.blocks)
        self.assertTrue(np.allclose(D, D.T))
        self.assertTrue(np.allclose(np.diag(D), 0))

    def test_explanation_adds_up_to_the_distance(self):
        D = distances.combine(self.blocks)
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            self.assertAlmostEqual(sum(distances.explain(self.blocks, i, j).values()), D[i, j])

    def test_era_distance_grows_with_the_gap_and_creator_distance_is_binary(self):
        era = distances.era_block(np.array([1970, 1971, 2000]), sigma=10)
        self.assertLess(era[0, 1], era[0, 2])
        self.assertEqual(era[0, 0], 0.0)
        self.assertEqual(distances.creator_block(["a", "a", "b"]).tolist(),
                         [[0, 0, 1], [0, 0, 1], [1, 1, 0]])

    def test_first_tag_counts_most(self):
        vectors = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0])}
        item = distances.weighted_tag_vectors([["x", "y"]], vectors)[0]
        self.assertGreater(item[0], item[1])
