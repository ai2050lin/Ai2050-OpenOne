from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase332_interface_branch_case_bank as phase332


ROUND = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas/interface_branch_atlas"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase332CaseBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(ROUND / "phase332_registered_cases.jsonl")

    def test_denominator_is_exact_and_balanced(self):
        self.assertEqual(len(self.rows), 1152)
        self.assertEqual(len({row["semantic_case_id"] for row in self.rows}), 96)
        self.assertEqual({row["interface"] for row in self.rows}, set(phase332.INTERFACES))
        self.assertEqual({row["item_index"] for row in self.rows}, set(range(8)))
        self.assertEqual(sum(row["split"] == "discovery" for row in self.rows), 576)
        self.assertEqual(sum(row["split"] == "heldout" for row in self.rows), 576)

    def test_discovery_and_heldout_items_do_not_overlap(self):
        discovery = {row["item_id"] for row in self.rows if row["split"] == "discovery"}
        heldout = {row["item_id"] for row in self.rows if row["split"] == "heldout"}
        self.assertFalse(discovery & heldout)
        self.assertTrue(all(not row["selection_updates_allowed"] for row in self.rows))

    def test_glm_native_and_no_think_equivalence_is_explicit(self):
        equivalent = [row for row in self.rows if row["interface_equivalent_to"]]
        self.assertEqual(len(equivalent), 96)
        self.assertTrue(all(row["model"] == "glm4" for row in equivalent))
        self.assertTrue(all(row["interface"] == "chat_no_think" for row in equivalent))


if __name__ == "__main__":
    unittest.main()
