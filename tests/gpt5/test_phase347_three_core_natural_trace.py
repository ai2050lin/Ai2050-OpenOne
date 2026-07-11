from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase347_three_core_natural_trace/three_core_natural_physical_trace"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase347NaturalTraceTests(unittest.TestCase):
    def test_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase347_registered_cases.jsonl")
        self.assertEqual(len(cases), 720)
        self.assertEqual(len({row["case_id"] for row in cases}), 720)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 10)
        self.assertEqual({row["model"] for row in cases}, {"qwen3", "glm4", "deepseek7b"})
        self.assertTrue(all(row["natural_trace_only"] for row in cases))
        self.assertFalse(any(row["internal_intervention_allowed"] for row in cases))

    def test_atlas_boundary(self) -> None:
        path = RESULT / "phase347_global_summary.json"
        if not path.exists():
            self.skipTest("Phase347 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["atlas_valid"])
        self.assertEqual(summary["denominator"]["registered_case_count"], 720)
        self.assertEqual(summary["denominator"]["case_row_count"], 720)
        self.assertEqual(summary["denominator"]["fixed_physical_node_count"], 810)
        self.assertEqual(len(read_jsonl(RESULT / "phase347_dominant_natural_regions.jsonl")), 30)
        self.assertEqual(summary["results"]["internal_intervention_executed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertFalse(summary["claim_boundary"]["natural_trace_is_causal"])


if __name__ == "__main__":
    unittest.main()
