from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase353_family_contracts/family_specific_contract_compiler"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase353FamilyContractTests(unittest.TestCase):
    def test_denominator(self) -> None:
        rows = read_jsonl(RESULT / "phase353_registered_cases.jsonl")
        self.assertEqual(len(rows), 5184)
        self.assertEqual(len({row["case_id"] for row in rows}), 5184)
        self.assertEqual(len({(row["family_id"], row["mechanism_id"]) for row in rows}), 18)
        self.assertFalse(any(row["internal_intervention_allowed"] for row in rows))

    def test_contract_entry_boundary(self) -> None:
        contracts = read_jsonl(RESULT / "phase353_contract_registry.jsonl")
        summary = json.loads((RESULT / "phase353_contract_summary.json").read_text())
        self.assertEqual(len(contracts), 18)
        self.assertEqual(summary["results"]["strict_contract_count"], sum(row["strict_contract_gate_pass"] for row in contracts))
        self.assertFalse(summary["results"]["model_execution_started"])
        self.assertEqual(summary["results"]["single_unit_causal_count"], 0)

    def test_execution_boundary(self) -> None:
        path = RESULT / "phase353_global_summary.json"
        if not path.exists():
            self.skipTest("Phase353 model execution has not completed")
        summary = json.loads(path.read_text())
        self.assertFalse(summary["denominator"]["all_model_completions_valid"])
        self.assertEqual(summary["denominator"]["invalid_phrase_row_count"], 1)
        self.assertEqual(summary["denominator"]["executed_case_count"], 3168)
        self.assertEqual(summary["denominator"]["phrase_row_count"], 3168)
        self.assertEqual(summary["denominator"]["rollout_row_count"], 3168)
        self.assertFalse(summary["results"]["physical_heldout_trace_revealed"])
        self.assertFalse(summary["results"]["causal_sealed_trace_revealed"])


if __name__ == "__main__":
    unittest.main()
