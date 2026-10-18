from __future__ import annotations

import unittest

import numpy as np

from research import ordering


class OrderingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        points = rng.permutation(np.sort(rng.uniform(0, 1, 40)))
        self.D = np.abs(points[:, None] - points[None, :])
        self.truth = np.argsort(points)

    def test_algorithms_recover_a_shuffled_line(self):
        for algorithm in (ordering.pcoa1d, ordering.spectral, ordering.leaf, ordering.tsp):
            with self.subTest(algorithm.__name__):
                order = algorithm(self.D)
                self.assertTrue(np.array_equal(order, self.truth)
                                or np.array_equal(order[::-1], self.truth))

    def test_every_algorithm_returns_a_permutation(self):
        for algorithm in (ordering.random_order, ordering.pcoa1d, ordering.spectral,
                          ordering.leaf, ordering.tsp):
            with self.subTest(algorithm.__name__):
                self.assertEqual(sorted(algorithm(self.D)), list(range(len(self.D))))

    def test_refine_never_lengthens_a_path(self):
        for seed in range(5):
            start = ordering.random_order(self.D, seed)
            refined = ordering.refine(self.D, start)
            self.assertLessEqual(ordering.path_length(self.D, refined),
                                 ordering.path_length(self.D, start) + 1e-9)

    def test_path_length_sums_consecutive_distances(self):
        D = np.array([[0, 1, 5], [1, 0, 2], [5, 2, 0]], dtype=float)
        self.assertEqual(ordering.path_length(D, [0, 1, 2]), 3.0)
        self.assertEqual(ordering.path_length(D, [0, 2, 1]), 7.0)

    def test_insert_puts_an_item_back_where_it_belongs_without_moving_the_others(self):
        truth = self.truth
        for removed in (0, 17, len(truth) - 1):
            remaining = np.delete(truth, removed)
            restored = ordering.insert(self.D, remaining, int(truth[removed]))
            self.assertEqual(restored.tolist(), truth.tolist())
            self.assertEqual(restored[restored != truth[removed]].tolist(), remaining.tolist())
