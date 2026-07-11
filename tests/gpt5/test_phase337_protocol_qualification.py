from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase337_protocol_qualification/material_relation_binding"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase337ProtocolQualificationTests(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        rows = read_jsonl(RESULT / "phase337_registered_cases.jsonl")
        self.assertEqual(len(rows), 108)
        self.assertEqual(len({row["case_id"] for row in rows}), 108)
        self.assertEqual({row["mechanism_id"] for row in rows}, {"material_relation_binding"})
        self.assertEqual({row["template_id"] for row in rows}, {"template_a"})
        self.assertFalse(any(row["internal_intervention_allowed"] for row in rows))

    def test_balanced_axes(self) -> None:
        rows = read_jsonl(RESULT / "phase337_registered_cases.jsonl")
        for model in ("qwen3", "glm4", "deepseek7b"):
            self.assertEqual(sum(row["model"] == model for row in rows), 36)
        for interface in ("raw_completion", "native_chat", "answer_aligned_chat"):
            self.assertEqual(sum(row["interface"] == interface for row in rows), 36)

    def test_rule_contract_boundary(self) -> None:
        contract = json.loads((RESULT / "phase337_rule_contract.json").read_text())
        self.assertIn("not parametric knowledge", contract["claim_scope"])
        self.assertEqual(contract["causal_status"], "not_tested_in_phase337")

    def test_execution_and_claim_boundary(self) -> None:
        summary_path = RESULT / "phase337_global_summary.json"
        if not summary_path.exists():
            self.skipTest("Phase337 model execution has not completed")
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["denominator"]["registered_case_count"], 108)
        self.assertEqual(summary["denominator"]["executed_case_count"], 108)
        self.assertTrue(summary["denominator"]["all_models_complete"])
        qualified = read_jsonl(RESULT / "phase337_qualified_rows.jsonl")
        self.assertEqual(len(qualified), 108)
        self.assertTrue(all("answer_head_semantic_correct" in row for row in qualified))
        self.assertEqual(summary["results"]["capable_cell_count"], 7)
        self.assertEqual(
            summary["results"]["passing_interfaces"],
            ["raw_completion", "answer_aligned_chat"],
        )
        self.assertEqual(
            summary["results"]["preferred_interface_for_next_stage"],
            "answer_aligned_chat",
        )
        self.assertEqual(summary["results"]["mechanism_causal_count"], 0)
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)


if __name__ == "__main__":
    unittest.main()
