#!/usr/bin/env python3
"""Contract checks for the prospective Phase589 protocol."""

from __future__ import annotations

import unittest

import phase589_food_attribute_observer as observer
import phase589_food_attribute_protocol as protocol


class Phase589ContractTest(unittest.TestCase):
    def test_object_panels_are_disjoint(self) -> None:
        ids = {
            split: {item["object_id"] for item in protocol.objects_for(split)}
            for split in protocol.SPLITS
        }
        self.assertFalse(ids["prospective_confirmation"] & ids["prospective_heldout"])
        self.assertFalse(ids["prospective_confirmation"] & ids["sealed"])
        self.assertFalse(ids["prospective_heldout"] & ids["sealed"])

    def test_panel_sizes(self) -> None:
        self.assertEqual(len(protocol.objects_for("prospective_confirmation")), 60)
        self.assertEqual(len(protocol.objects_for("prospective_heldout")), 30)
        self.assertEqual(len(protocol.objects_for("sealed")), 30)

    def test_pairwise_auc(self) -> None:
        self.assertEqual(observer.pairwise_auc([2.0, 3.0], [0.0, 1.0]), 1.0)
        self.assertEqual(observer.pairwise_auc([1.0], [1.0]), 0.5)


if __name__ == "__main__":
    unittest.main()
