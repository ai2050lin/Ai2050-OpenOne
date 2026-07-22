from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase602_three_track_semantics"
MODELS = ("qwen3", "glm4", "deepseek7b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Phase602ThreeTrackResultTest(unittest.TestCase):
    def test_runs_are_complete_and_observer_only(self) -> None:
        for model in MODELS:
            stem = OUT / f"phase602_{model}_three_track_behavior"
            summary = json.loads(stem.with_name(stem.name + "_summary.json").read_text())
            rows = stem.with_name(stem.name + "_rows.jsonl.gz")
            self.assertEqual(summary["status"], "complete", model)
            self.assertEqual(summary["case_count"], 1_440, model)
            self.assertEqual(summary["concept_count"], 120, model)
            self.assertEqual(summary["rows_sha256"], sha256_file(rows), model)
            self.assertFalse(summary["internal_state_collected"], model)
            self.assertFalse(summary["causal_intervention_authorized"], model)

    def test_analysis_keeps_evidence_boundary(self) -> None:
        analysis = json.loads((OUT / "phase602_cross_model_analysis.json").read_text())
        self.assertEqual(analysis["case_count"], 1_440)
        self.assertFalse(analysis["causal_intervention_authorized"])
        self.assertFalse(analysis["mechanism_claim_authorized"])
        self.assertFalse(analysis["theory_or_formula_update_authorized"])
        self.assertFalse(analysis["full_five_family_completion_claim_authorized"])


if __name__ == "__main__":
    unittest.main()
