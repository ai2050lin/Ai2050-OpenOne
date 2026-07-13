#!/usr/bin/env python3
"""Offline regression checks for Phase397 factor-separated binding evidence."""

from __future__ import annotations

import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"
MODELS = ("qwen3", "glm4", "deepseek7b")
SURFACES = (
    "field_extraction", "possession_relation", "spatial_relation",
    "role_filling", "coreference_resolution", "event_state_update",
)
CONDITIONS = set("ABCDEFGHIJ")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase397FactorSeparationTests(unittest.TestCase):
    def test_candidate_denominator_is_frozen_and_balanced(self):
        protocol = read_json(RESULT / "phase397_protocol.json")
        rows = read_jsonl(RESULT / "protocol/private/phase397_candidate_cases.jsonl")
        self.assertEqual(protocol["denominator"]["candidate_case_count"], 4320)
        self.assertEqual(len(rows), 4320)
        counts = Counter((row["private_execution_model"], row["task_surface_private"]) for row in rows)
        self.assertEqual(set(counts.values()), {240})
        grouped = defaultdict(set)
        for row in rows:
            grouped[(row["private_execution_model"], row["anonymous_parallel_group_id"])].add(row["anonymous_condition_slot"])
        self.assertEqual(len(grouped), 432)
        self.assertTrue(all(conditions == CONDITIONS for conditions in grouped.values()))

    def test_relation_pairs_fix_value_positions(self):
        rows = read_jsonl(RESULT / "protocol/private/phase397_candidate_cases.jsonl")
        grouped = defaultdict(dict)
        for row in rows:
            grouped[(row["private_execution_model"], row["anonymous_parallel_group_id"])][row["anonymous_condition_slot"]] = row
        for conditions in grouped.values():
            self.assertEqual(conditions["A"]["literal_value_positions_private"], conditions["B"]["literal_value_positions_private"])
            self.assertEqual(conditions["F"]["literal_value_positions_private"], conditions["G"]["literal_value_positions_private"])
            self.assertTrue(set(conditions["A"]["relation_diff_positions_private"]).issubset(conditions["A"]["source_entity_positions_private"]))

    def test_behavior_gate_keeps_failures_and_fixed_splits(self):
        freeze = read_json(RESULT / "phase397_behavior_freeze_summary.json")
        self.assertEqual(freeze["denominator"]["qualified_parallel_group_count"], 79)
        self.assertEqual(freeze["eligible_surfaces"], ["possession_relation", "role_filling", "coreference_resolution"])
        gates = {row["task_surface"]: row for row in freeze["surface_gates"]}
        self.assertEqual({surface: gates[surface]["qualified_group_count"] for surface in SURFACES}, {
            "field_extraction": 13,
            "possession_relation": 18,
            "spatial_relation": 0,
            "role_filling": 24,
            "coreference_resolution": 24,
            "event_state_update": 0,
        })
        for surface in freeze["eligible_surfaces"]:
            selected = freeze["selected_groups_private"][surface]
            self.assertEqual({key: len(value) for key, value in selected.items()}, {"discovery": 8, "calibration": 4, "physical_holdout": 4})
            self.assertEqual(len(set().union(*map(set, selected.values()))), 16)

    def test_three_split_observation_passes_but_causal_gate_closes(self):
        discovery = read_json(RESULT / "phase397_factor_discovery_analysis.json")
        calibration = read_json(RESULT / "phase397_factor_calibration_analysis.json")
        physical = read_json(RESULT / "phase397_factor_physical_analysis.json")
        causal = read_json(RESULT / "phase397_causal_analysis.json")
        self.assertEqual(discovery["results"]["passing_model_surface_cell_count"], 9)
        self.assertEqual(calibration["results"]["passing_model_surface_cell_count"], 9)
        self.assertEqual(physical["results"]["passing_model_surface_cell_count"], 9)
        self.assertEqual(causal["results"]["passing_model_surface_cell_count"], 0)
        self.assertFalse(causal["authorization"]["calibration_causal_intervention"])
        self.assertFalse(causal["authorization"]["single_neuron_scan"])
        self.assertEqual(sum(cell["relation_answer_switch_count"] for cell in causal["cells"]), 0)
        self.assertEqual(causal["denominator"]["direction_count"], 144)

    def test_query_source_invariance_and_patch_locality_are_exact(self):
        audit = read_json(RESULT / "phase397_factor_trace_instrument_audit.json")
        self.assertTrue(audit["results"]["all_three_model_instruments_valid"])
        self.assertTrue(audit["results"]["causal_source_invariance_pass"])
        self.assertTrue(audit["results"]["identity_patch_locality_pass"])
        causal = read_json(RESULT / "phase397_causal_analysis.json")
        self.assertEqual(causal["results"]["maximum_identity_effect"], 0)
        self.assertEqual(causal["results"]["maximum_query_source_control_effect"], 0)
        self.assertEqual(causal["results"]["maximum_patch_locality_error"], 0)

    def test_public_atlas_mirrors_match(self):
        names = (
            "phase397_factor_separated_binding_stage_summary.json",
            "phase397_behavior_freeze_summary.json",
            "phase397_factor_physical_analysis.json",
            "phase397_causal_analysis.json",
            "progress.json",
            "manifest.json",
        )
        for name in names:
            self.assertEqual((ATLAS / name).read_bytes(), (CLIENT / name).read_bytes(), name)
        self.assertEqual((NEURON / "manifest.json").read_bytes(), (NEURON_CLIENT / "manifest.json").read_bytes())

    def test_phase397_3d_anchors_are_aggregate_not_neurons(self):
        family_by_surface = {
            "possession_relation": "content_knowledge",
            "role_filling": "language_action",
            "coreference_resolution": "reasoning_constraint",
        }
        count = 0
        for surface, family in family_by_surface.items():
            for model in MODELS:
                partition = read_json(NEURON / f"partitions/{family}/{model}.json")
                nodes = [node for node in partition["nodes"] if node.get("phase397_tested")]
                self.assertEqual(len(nodes), 1)
                node = nodes[0]
                self.assertEqual(node["relation"], surface)
                self.assertEqual(node["node_type"], "aggregate_token_state_anchor")
                self.assertFalse(node["is_real_unit"])
                self.assertFalse(node["single_neuron_claim"])
                self.assertFalse(node["phase397_causal_gate_pass"])
                count += 1
        self.assertEqual(count, 9)

    def test_memo_preserves_phase397_before_phase398(self):
        memo = (ROOT / "research/gpt5/docs/AGI_GPT5_MEMO.md").read_text(encoding="utf-8")
        self.assertIn("## Phase 397: 多任务关系签名与因果载体分离 [2026-07-12 16:47]", memo)
        self.assertEqual(memo.count("## Phase 397: 多任务关系签名与因果载体分离"), 1)
        self.assertIn("## Phase 398: 顺序条件化联合轨迹与单查询位置因果审计 [2026-07-12 18:40]", memo)
        self.assertLess(memo.index("## Phase 397:"), memo.index("## Phase 398:"))


if __name__ == "__main__":
    unittest.main()
