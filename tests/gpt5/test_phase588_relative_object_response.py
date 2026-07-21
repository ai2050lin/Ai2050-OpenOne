#!/usr/bin/env python3
"""Unit checks for the Phase588 pairwise object-response diagnostic."""

from __future__ import annotations

import unittest

import phase588_relative_object_response_analysis as analysis


class PairwiseAucTest(unittest.TestCase):
    def test_perfect_order(self) -> None:
        self.assertEqual(analysis.pairwise_auc([3.0, 4.0], [1.0, 2.0]), 1.0)

    def test_reversed_order(self) -> None:
        self.assertEqual(analysis.pairwise_auc([1.0, 2.0], [3.0, 4.0]), 0.0)

    def test_ties_count_half(self) -> None:
        self.assertEqual(analysis.pairwise_auc([1.0], [1.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
