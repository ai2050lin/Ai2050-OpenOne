from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "tests/gpt5/result/phase360_denominator_freeze/phase360_denominator_summary.json"


class Phase360DenominatorFreezeTests(unittest.TestCase):
    def test_frozen_denominator_and_claim_boundary(self) -> None:
        if not SUMMARY.exists():
            self.skipTest("Phase360 has not run")
        payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
        self.assertEqual(payload["denominator"]["family_count"], 9)
        self.assertEqual(payload["denominator"]["mechanism_count"], 18)
        self.assertEqual(payload["denominator"]["blind_discovery_admitted_count"], 3)
        self.assertFalse(payload["evidence_boundary"]["nine_family_blind_discovery_ready"])
        self.assertFalse(payload["evidence_boundary"]["single_global_progress_percentage_valid"])
        self.assertEqual(payload["decision"], "do_not_expand_r0_r1_to_nine_families")


if __name__ == "__main__":
    unittest.main()
