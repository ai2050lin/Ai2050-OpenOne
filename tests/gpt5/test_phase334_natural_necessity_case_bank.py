from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas/natural_necessity_atlas"


class Phase334CaseBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            json.loads(line)
            for line in (RESULT / "phase334_registered_cases.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def test_frozen_denominator(self) -> None:
        self.assertEqual(len(self.rows), 1944)
        self.assertEqual(len({row["case_id"] for row in self.rows}), 1944)
        self.assertEqual(len({row["mechanism_id"] for row in self.rows}), 6)
        self.assertEqual(len({row["family_id"] for row in self.rows}), 3)

    def test_balanced_axes(self) -> None:
        for model in ("qwen3", "glm4", "deepseek7b"):
            self.assertEqual(sum(row["model"] == model for row in self.rows), 648)
        for split, expected in (("discovery", 972), ("calibration", 486), ("heldout", 486)):
            self.assertEqual(sum(row["split"] == split for row in self.rows), expected)

    def test_claim_boundaries(self) -> None:
        protocol = json.loads((RESULT / "phase334_registered_protocol.json").read_text())
        self.assertIn("explicit relation binding", " ".join(protocol["claim_boundaries"]))
        self.assertFalse(any(row["single_unit_intervention_gate_open"] for row in self.rows))
        self.assertFalse(any(row["selection_updates_allowed"] for row in self.rows))


if __name__ == "__main__":
    unittest.main()
