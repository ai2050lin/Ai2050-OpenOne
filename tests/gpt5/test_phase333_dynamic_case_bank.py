from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase333_dynamic_path_atlas/dynamic_path_atlas"


class Phase333CaseBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line) for line in (RESULT / "phase333_registered_cases.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def test_frozen_denominator(self) -> None:
        self.assertEqual(len(self.rows), 648)
        self.assertEqual(len({row["case_id"] for row in self.rows}), 648)
        self.assertEqual({row["interface"] for row in self.rows}, {
            "raw_completion", "native_chat", "answer_aligned_chat",
        })

    def test_object_splits_are_disjoint(self) -> None:
        by_split = {
            split: {row["item_index"] for row in self.rows if row["split"] == split}
            for split in ("discovery", "calibration", "heldout")
        }
        self.assertEqual(by_split["discovery"], set(range(6)))
        self.assertEqual(by_split["calibration"], {6, 7, 8})
        self.assertEqual(by_split["heldout"], {9, 10, 11})
        self.assertFalse(by_split["discovery"] & by_split["heldout"])

    def test_mechanism_targets_are_distinct(self) -> None:
        targets = {
            mechanism: {row["target"] for row in self.rows if row["mechanism_id"] == mechanism}
            for mechanism in ("missing_condition_control", "two_hop_blocked")
        }
        self.assertEqual(targets["missing_condition_control"], {"unknown"})
        self.assertEqual(targets["two_hop_blocked"], {"no"})


if __name__ == "__main__":
    unittest.main()
