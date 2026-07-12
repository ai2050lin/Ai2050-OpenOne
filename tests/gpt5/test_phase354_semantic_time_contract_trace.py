from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase354_semantic_time_contract_trace/qualified_contract_semantic_time"


class Phase354SemanticTimeContractTests(unittest.TestCase):
    def test_registered_denominator_and_seals(self) -> None:
        rows = [json.loads(line) for line in (RESULT / "phase354_registered_cases.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 576)
        self.assertEqual(len({(row["family_id"], row["mechanism_id"]) for row in rows}), 3)
        self.assertEqual({row["split"] for row in rows}, {"physical_discovery", "physical_calibration"})
        self.assertFalse(any(row["internal_intervention_allowed"] for row in rows))

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase354_global_summary.json"
        if not path.exists():
            self.skipTest("Phase354 model execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["case_row_count"], 576)
        self.assertEqual(summary["results"]["physical_heldout_trace_revealed"], False)
        self.assertEqual(summary["results"]["causal_sealed_trace_revealed"], False)
        self.assertEqual(summary["results"]["internal_intervention_executed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
