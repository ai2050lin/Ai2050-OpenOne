from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase351_signed_paired_trace/signed_paired_physical_trace"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase351SignedPairedTraceTests(unittest.TestCase):
    def test_denominator_seals(self) -> None:
        rows = read_jsonl(RESULT / "phase351_registered_cases.jsonl")
        self.assertEqual(len(rows), 576)
        self.assertEqual(len({row["case_id"] for row in rows}), 576)
        self.assertEqual({row["family_id"] for row in rows}, {"closure", "language_action", "state_drift"})
        self.assertFalse(any(row["physical_heldout_trace_allowed"] for row in rows))
        self.assertFalse(any(row["causal_sealed_trace_allowed"] for row in rows))

    def test_trace_claim_boundary(self) -> None:
        path = RESULT / "phase351_global_summary.json"
        if not path.exists():
            self.skipTest("Phase351 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["registered_case_count"], 576)
        self.assertEqual(summary["denominator"]["fixed_signed_node_count"], 243)
        self.assertEqual(summary["denominator"]["incomplete_pair_count"], 0)
        self.assertFalse(summary["results"]["generated_time_tested"])
        self.assertFalse(summary["results"]["physical_heldout_trace_revealed"])
        self.assertFalse(summary["claim_boundary"]["signed_margin_delta_is_causal"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
