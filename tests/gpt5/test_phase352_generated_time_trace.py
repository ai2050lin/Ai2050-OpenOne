from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase352_generated_time_trace/generated_time_signed_trace"


class Phase352GeneratedTimeTraceTests(unittest.TestCase):
    def test_execution_and_seals(self) -> None:
        path = RESULT / "phase352_global_summary.json"
        if not path.exists():
            self.skipTest("Phase352 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["registered_case_count"], 576)
        self.assertEqual(summary["denominator"]["fixed_dynamic_node_count"], 1296)
        self.assertGreater(summary["denominator"]["paired_event_count"], 0)
        self.assertGreater(summary["denominator"]["incomplete_pair_count"], 0)
        self.assertFalse(summary["results"]["physical_heldout_trace_revealed"])
        self.assertFalse(summary["results"]["causal_sealed_trace_revealed"])
        self.assertFalse(summary["claim_boundary"]["teacher_forced_time_is_free_generation_path"])
        self.assertFalse(summary["claim_boundary"]["cross_lexical_generation_phase_alignment_complete"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
