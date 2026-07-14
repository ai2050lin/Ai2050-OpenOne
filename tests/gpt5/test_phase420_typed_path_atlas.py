#!/usr/bin/env python3
"""Contract tests for the Phase420 typed natural-path atlas."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase420_typed_path_atlas"
PUBLIC = ROOT / "frontend/public/vis_data/phase420_typed_path_atlas"
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase420TypedPathAtlasTest(unittest.TestCase):
    def test_frozen_denominator(self) -> None:
        qualification = read_json(RESULT / "phase420_denominator_qualification.json")
        groups = read_jsonl(RESULT / "phase420_frozen_groups.jsonl")
        conditions = read_jsonl(RESULT / "phase420_registered_conditions.jsonl")
        self.assertTrue(qualification["valid"])
        self.assertEqual(len(groups), 33)
        self.assertEqual(len(conditions), 792)
        self.assertEqual(
            qualification["group_split_count"],
            {"discovery": 15, "calibration": 6, "behavior_holdout": 6, "physical_holdout": 6},
        )
        self.assertTrue(all(row["full_prompt_history_pair_token_count_exact"] for row in conditions))
        self.assertEqual(len({row["phase420_condition_id"] for row in conditions}), 792)

    def test_model_traces_and_sealed_holdout(self) -> None:
        for model in MODELS:
            complete = read_json(RESULT / "models" / model / "phase420_trace_complete.json")
            self.assertEqual(complete["behavior_condition_count"], 264)
            self.assertEqual(complete["development_physical_condition_count"], 216)
            self.assertEqual(complete["physical_holdout_condition_count"], 0)
            self.assertTrue(complete["physical_holdout_remains_sealed"])
            self.assertTrue(complete["all_development_rows_pass"])
            self.assertLessEqual(complete["max_physical_ledger_relative_error"], 0.01)

    def test_global_evidence_gates(self) -> None:
        summary = read_json(RESULT / "phase420_global_summary.json")
        self.assertTrue(summary["valid"])
        self.assertEqual(summary["behavior_condition_count"], 792)
        self.assertEqual(summary["development_physical_condition_count"], 648)
        self.assertTrue(summary["gates"]["source_write_replication"])
        self.assertFalse(summary["gates"]["history_current_source_separation"])
        self.assertFalse(summary["gates"]["unseen_behavior_prediction"])
        self.assertFalse(summary["gates"]["physical_holdout_authorized"])
        self.assertFalse(summary["gates"]["causal_intervention_authorized"])
        self.assertEqual(summary["strict_mechanism_closure_count"], 0)

    def test_prediction_does_not_beat_frozen_baselines(self) -> None:
        rows = read_jsonl(RESULT / "phase420_prediction_audit.jsonl")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(not row["strict_prediction_gate_pass"] for row in rows))
        self.assertTrue(all(not row["physical_holdout_used"] for row in rows))

    def test_client_manifest_and_graph_contract(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase420_typed_path_atlas_manifest.v1")
        self.assertEqual(len(manifest["items"]), 12)
        for item in manifest["items"]:
            payload = read_json(PUBLIC / item["filename"])
            self.assertEqual(payload["schema_version"], "atlas_graph_v1")
            self.assertTrue(payload["graph"]["nodes"])
            self.assertTrue(all(not node.get("causal", False) for node in payload["graph"]["nodes"]))
            self.assertTrue(all(not edge.get("causal", False) for edge in payload["graph"]["edges"]))

    def test_source_registry(self) -> None:
        registry = read_json(REGISTRY)
        sources = {row["id"]: row for row in registry["sources"]}
        self.assertIn("gpt5_phase420_typed_path_atlas", sources)
        self.assertEqual(sources["gpt5_phase420_typed_path_atlas"]["route_id"], "gpt5")


if __name__ == "__main__":
    unittest.main()
