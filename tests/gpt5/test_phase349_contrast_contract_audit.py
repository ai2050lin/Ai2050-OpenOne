from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase349_contrast_contract_audit/orthogonal_contrast_contract_audit"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Phase349ContrastContractAuditTests(unittest.TestCase):
    def test_registry_denominator(self) -> None:
        rows = read_jsonl(RESULT / "phase349_contrast_contract_registry.jsonl")
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({row["contract_id"] for row in rows}), 72)
        self.assertEqual(len({row["family_id"] for row in rows}), 9)
        self.assertTrue(all(not row["selection_used_model_effects"] for row in rows))

    def test_claim_boundary(self) -> None:
        summary = json.loads((RESULT / "phase349_global_summary.json").read_text())
        self.assertEqual(summary["denominator"]["mechanism_count"], 72)
        self.assertEqual(summary["denominator"]["proposed_prompt_model_case_count"], 20736)
        self.assertFalse(summary["results"]["model_execution_started"])
        self.assertEqual(summary["results"]["behavior_mechanism_closed_count"], 0)
        self.assertFalse(summary["claim_boundaries"]["candidate_operation_labels_are_validated_theory"])


if __name__ == "__main__":
    unittest.main()
