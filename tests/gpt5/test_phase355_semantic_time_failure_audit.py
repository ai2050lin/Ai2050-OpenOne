from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase355_semantic_time_failure_audit/strict_failure_decomposition"


class Phase355FailureAuditTests(unittest.TestCase):
    def test_strict_boundary_and_export(self) -> None:
        summary = json.loads((RESULT / "phase355_global_summary.json").read_text())
        self.assertEqual(summary["denominator"]["fixed_node_count"], 2592)
        self.assertEqual(summary["results"]["strict_dynamic_candidate_count"], 0)
        self.assertGreater(summary["results"]["replicated_direction_near_candidate_count"], 0)
        self.assertEqual(summary["results"]["physical_heldout_entry_contract_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)

    def test_near_candidates_are_not_promoted(self) -> None:
        rows = [json.loads(line) for line in (RESULT / "phase355_near_candidates.jsonl").read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(not row["heldout_eligible"] for row in rows))
        self.assertTrue(all(not row["causal_eligible"] for row in rows))


if __name__ == "__main__":
    unittest.main()
