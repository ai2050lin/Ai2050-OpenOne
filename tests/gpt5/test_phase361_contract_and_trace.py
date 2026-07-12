from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/gpt5/result/phase361_contract_repair/seven_contract_repair"
TRACE = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace"
BANNED = {"model", "family_id", "mechanism_id", "target", "case_id", "source_case_id"}


class Phase361ContractAndTraceTests(unittest.TestCase):
    def test_repaired_contract_and_behavior_denominators(self) -> None:
        contracts = [json.loads(line) for line in (CONTRACT / "phase361_repaired_contract_registry.jsonl").read_text().splitlines()]
        summary = json.loads((CONTRACT / "phase361_behavior_summary.json").read_text())
        self.assertEqual(len(contracts), 7)
        self.assertTrue(all(row["strict_contract_gate_pass"] for row in contracts))
        self.assertEqual(summary["denominator"]["registered_case_count"], 2016)
        self.assertEqual(summary["results"]["total_blind_discovery_admitted_count"], 4)
        self.assertFalse(summary["evidence_boundary"]["physical_heldout_trace_revealed"])

    def test_blind_ledger_and_role_mapping(self) -> None:
        summary = json.loads((TRACE / "phase361_r0_r1_summary.json").read_text())
        self.assertEqual(summary["denominator"]["case_count"], 96)
        self.assertEqual(summary["denominator"]["ledger_row_count"], 3328)
        self.assertTrue(summary["quality"]["all_component_gates_pass"])
        self.assertEqual(summary["quality"]["blind_ledger_label_leak_row_count"], 0)
        self.assertTrue(all(value == 3328 for value in summary["quality"]["role_exact_counts"].values()))
        self.assertTrue(all(value == 6 for value in summary["quality"]["r1_shard_counts"].values()))
        for model in ("qwen3", "glm4", "deepseek7b"):
            path = TRACE / "models" / model / "phase361_r0_r1_ledger_rows.jsonl"
            for line in path.read_text().splitlines()[:10]:
                self.assertFalse(BANNED & set(json.loads(line)))

    def test_prediction_claim_boundary(self) -> None:
        prediction = json.loads((TRACE / "phase361_blind_prediction_summary.json").read_text())
        posthoc = json.loads((TRACE / "phase361_posthoc_predictive_summary.json").read_text())
        self.assertEqual(prediction["denominator"]["shared_positive_candidate_count"], 93)
        self.assertEqual(posthoc["results"]["universally_positive_candidate_count"], 77)
        self.assertEqual(posthoc["results"]["selective_association_candidate_count"], 16)
        self.assertEqual(posthoc["results"]["operation_specific_mechanism_count"], 0)
        self.assertFalse(posthoc["claim_boundary"]["physical_heldout_revealed"])


if __name__ == "__main__":
    unittest.main()
