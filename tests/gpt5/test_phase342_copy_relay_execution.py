from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase342_copy_relay_execution/copy_relay_execution_invariance"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase342CopyRelayExecutionTests(unittest.TestCase):
    def test_registered_denominator(self) -> None:
        cases = read_jsonl(RESULT / "phase342_registered_cases.jsonl")
        self.assertEqual(len(cases), 216)
        self.assertEqual(len({row["case_id"] for row in cases}), 216)
        self.assertEqual(
            {model: sum(row["model"] == model for row in cases)
             for model in ("qwen3", "glm4", "deepseek7b")},
            {"qwen3": 72, "glm4": 72, "deepseek7b": 72},
        )
        self.assertFalse(any(row["internal_intervention_allowed"] for row in cases))

    def test_modes(self) -> None:
        protocol = json.loads((RESULT / "phase342_registered_protocol.json").read_text())
        self.assertEqual(len(protocol["execution_modes"]), 11)
        self.assertEqual(protocol["reference_mode"], "b1_left_cache0")
        self.assertTrue(any(mode["padding_side"] == "right" for mode in protocol["execution_modes"]))
        self.assertTrue(any(mode["use_cache"] for mode in protocol["execution_modes"]))

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase342_global_summary.json"
        if not path.exists():
            self.skipTest("Phase342 execution has not completed")
        summary = json.loads(path.read_text())
        self.assertTrue(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["result_row_count"], 2376)
        self.assertEqual(summary["results"]["internal_intervention_executed_count"], 0)
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)
        self.assertEqual(len(read_jsonl(RESULT / "phase342_execution_nodes.jsonl")), 33)


if __name__ == "__main__":
    unittest.main()
