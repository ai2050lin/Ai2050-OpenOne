from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase601_foodon_public_ontology"
MODELS = ("qwen3", "glm4", "deepseek7b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Phase601FoodOnResultTest(unittest.TestCase):
    def test_three_model_runs_are_complete_and_bound(self) -> None:
        for model in MODELS:
            stem = OUT / f"phase601_{model}_foodon_behavior"
            summary_path = stem.with_name(stem.name + "_summary.json")
            rows_path = stem.with_name(stem.name + "_rows.jsonl.gz")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "complete", model)
            self.assertEqual(summary["overall"]["case_count"], 1_920, model)
            self.assertEqual(summary["overall"]["concept_count"], 480, model)
            self.assertEqual(summary["rows_sha256"], sha256_file(rows_path), model)
            self.assertFalse(summary["internal_state_collected"], model)
            self.assertFalse(summary["causal_intervention_authorized"], model)

    def test_frozen_gates_block_all_internal_followup(self) -> None:
        analysis = json.loads(
            (OUT / "phase601_cross_model_analysis.json").read_text(encoding="utf-8")
        )
        self.assertEqual(analysis["case_count"], 1_920)
        self.assertEqual(analysis["qualified_models"], [])
        self.assertEqual(
            analysis["model_qualification"],
            {"qwen3": False, "glm4": False, "deepseek7b": False},
        )
        self.assertFalse(analysis["cross_model_internal_observer_followup_authorized"])
        self.assertFalse(analysis["causal_intervention_authorized"])
        self.assertFalse(analysis["mechanism_claim_authorized"])
        self.assertTrue(analysis["ontology_semantic_calibration_required"])
        self.assertFalse(analysis["posthoc_case_removal_authorized"])

    def test_cross_model_denominator_is_complete(self) -> None:
        analysis = json.loads(
            (OUT / "phase601_cross_model_analysis.json").read_text(encoding="utf-8")
        )
        self.assertEqual(analysis["all_models_wrong_case_count"], 31)
        self.assertEqual(analysis["all_models_wrong_concept_count"], 25)
        self.assertEqual(len(analysis["true_false_pair_metrics"]), 20)
        self.assertGreater(len(analysis["hardest_concepts"]), 0)


if __name__ == "__main__":
    unittest.main()
