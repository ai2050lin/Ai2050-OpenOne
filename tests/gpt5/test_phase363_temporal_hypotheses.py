from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / "tests/gpt5/result/phase363_temporal_hypotheses/strict_temporal_innovation_formulas"
P361_CANDIDATES = ROOT / "tests/gpt5/result/phase361_r0_r1_blind_trace/four_admitted_balanced_trace/phase361_frozen_predictive_candidates.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase363TemporalHypothesisTests(unittest.TestCase):
    def test_candidate_hash_and_denominator_are_frozen(self) -> None:
        summary = json.loads((ROUND / "phase363_hypothesis_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["frozen_inputs"]["phase361_candidate_sha256"], hashlib.sha256(P361_CANDIDATES.read_bytes()).hexdigest())
        self.assertEqual(summary["denominator"]["input_candidate_count"], 7)
        self.assertEqual(summary["denominator"]["tested_formula_count"], 20)
        self.assertEqual(summary["denominator"]["frozen_formula_count"], 0)
        self.assertFalse(summary["quality"]["physical_confirmation_read"])

    def test_group_split_uses_groups_as_units(self) -> None:
        rows = read_jsonl(ROUND / "phase363_formula_split_rows.jsonl")
        self.assertEqual(len(rows), 72)
        self.assertTrue(all(row["case_count"] == 4 for row in rows))
        counts = Counter((row["model"], row["mechanism"], row["split"]) for row in rows)
        for model in ("qwen3", "glm4", "deepseek7b"):
            for mechanism in {row["mechanism"] for row in rows if row["model"] == model}:
                self.assertEqual(counts[(model, mechanism, "formula_train")], 4)
                self.assertEqual(counts[(model, mechanism, "formula_validation")], 2)

    def test_strict_result_closes_route(self) -> None:
        formulas = read_jsonl(ROUND / "phase363_all_formula_rows.jsonl")
        self.assertEqual(sum(row["target_type"] == "time_innovation" for row in formulas), 14)
        self.assertEqual(sum(row["target_type"] == "competition_change" for row in formulas), 6)
        self.assertFalse(any(row["all_models_discovery_pass"] for row in formulas))
        self.assertEqual((ROUND / "phase363_frozen_formula_rows.jsonl").read_text(encoding="utf-8"), "")
        summary = json.loads((ROUND / "phase363_global_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["results"]["physical_confirmation_executed_case_count"], 0)
        self.assertTrue(summary["claim_boundary"]["phase361_phase362_candidate_route_closed"])

    def test_frontend_receives_summaries_not_tensors(self) -> None:
        public = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
        manifest = json.loads((public / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["phase363"]["status"], "temporal_candidate_route_closed_confirmation_remains_sealed")
        self.assertFalse(manifest["phase363"]["raw_tensors_frontend_exported"])
        self.assertFalse(any(path.suffix == ".pt" for path in public.rglob("*")))


if __name__ == "__main__":
    unittest.main()
