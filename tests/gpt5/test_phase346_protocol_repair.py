from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase346_protocol_repair/three_core_protocol_repair"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase346ProtocolRepairTests(unittest.TestCase):
    def test_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase346_registered_cases.jsonl")
        self.assertEqual(len(cases), 432)
        self.assertEqual(len({row["case_id"] for row in cases}), 432)
        self.assertEqual(len({row["mechanism_id"] for row in cases}), 2)
        self.assertFalse(any(row["internal_intervention_allowed"] for row in cases))

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase346_global_summary.json"
        if not path.exists():
            self.skipTest("Phase346 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["phrase_row_count"], 432)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 432)
        self.assertEqual(summary["results"]["internal_intervention_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(len(read_jsonl(RESULT / "phase346_protocol_nodes.jsonl")), 6)


if __name__ == "__main__":
    unittest.main()
