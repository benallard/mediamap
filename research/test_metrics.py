from __future__ import annotations

import unittest

import numpy as np

from research import metrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        points = np.arange(30, dtype=float)
        self.D = np.abs(points[:, None] - points[None, :])
        self.line = np.arange(30)

    def test_mean_step_of_a_line_is_the_spacing(self):
        self.assertEqual(metrics.mean_step(self.D, self.line), 1.0)

    def test_neighbour_recall_is_perfect_on_a_line_and_poor_when_shuffled(self):
        shuffled = np.random.default_rng(0).permutation(30)
        self.assertEqual(metrics.neighbour_recall(self.D, self.line), 1.0)
        self.assertLess(metrics.neighbour_recall(self.D, shuffled), 0.6)

    def test_label_breaks_counts_changes_between_neighbours(self):
        labels = ["a", "a", "b", "b", "a"]
        self.assertEqual(metrics.label_breaks(labels, [0, 1, 2, 3, 4]), 2)
        self.assertEqual(metrics.label_breaks(labels, [0, 1, 4, 2, 3]), 1)

    def test_worst_joins_are_the_largest_steps_first(self):
        D = np.abs(np.array([0.0, 1.0, 4.0, 5.0])[:, None] - np.array([0.0, 1.0, 4.0, 5.0])[None, :])
        self.assertEqual(metrics.worst_joins(D, [0, 1, 2, 3], top=1), [(1, 2, 3.0)])
