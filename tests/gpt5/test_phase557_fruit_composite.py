#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase557_fruit_composite_protocol as protocol  # noqa: E402
import phase557_natural_color_source_intervention as source_intervention  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase557FruitCompositeTests(unittest.TestCase):
    def test_static_denominator_and_seal(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 29184)
        self.assertEqual(audit["open_case_count"], 24192)
        self.assertEqual(audit["sealed_case_count"], 4992)
        self.assertEqual(audit["controlled_rows_per_anchor"], [32])
        self.assertEqual(audit["target_dependency_error_count"], 0)
        self.assertEqual(commitment["sealed_case_count"], 4992)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_open_bank_contains_no_sealed_rows(self) -> None:
        rows = read_jsonl(protocol.OPEN_CASES_PATH)
        self.assertEqual(len(rows), 24192)
        self.assertFalse(any(row["sealed"] for row in rows))
        self.assertEqual({row["split"] for row in rows}, set(protocol.OPEN_SPLITS))

    def test_world_has_two_query_strata_and_four_independent_factors(self) -> None:
        rows = [
            row for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["anchor_id"] == "phase557_controlled_behavior_discovery_000"
        ]
        self.assertEqual(len(rows), 32)
        self.assertEqual({row["factorial_cell"] for row in rows}, set(protocol.CELLS))
        self.assertEqual({row["query_stratum"] for row in rows}, set(protocol.QUERY_STRATA))
        self.assertEqual(len({row["fact_token_multiset_key"] for row in rows}), 1)

    def test_attribute_content_and_binding_are_not_aliased(self) -> None:
        rows = {
            row["factorial_cell"]: row
            for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["anchor_id"] == "phase557_controlled_behavior_discovery_000"
        }
        targets = {
            rows[f"attribute__object0_category0_attribute{attribute}_binding{binding}"]["target"]
            for attribute in (0, 1) for binding in (0, 1)
        }
        self.assertEqual(len(targets), 4)
        category_targets = {
            rows[f"category__object0_category0_attribute{attribute}_binding{binding}"]["target"]
            for attribute in (0, 1) for binding in (0, 1)
        }
        self.assertEqual(len(category_targets), 1)

    def test_target_dependencies_match_query_stratum(self) -> None:
        for query in protocol.QUERY_STRATA:
            base = protocol.controlled_world(
                "behavior_discovery", 3,
                f"{query}__object0_category0_attribute0_binding0",
            )
            expected = set(base["target_factor_dependencies"])
            for factor in protocol.FACTORS:
                values = {name: 0 for name in protocol.FACTORS}
                values[factor] = 1
                cell = "_".join(f"{name}{values[name]}" for name in protocol.FACTORS)
                changed = protocol.controlled_world("behavior_discovery", 3, f"{query}__{cell}")
                self.assertEqual(base["target"] != changed["target"], factor in expected)

    def test_protocol_forbids_early_parameter_and_neuron_scans(self) -> None:
        frozen = read_json(protocol.PROTOCOL_PATH)
        policy = frozen["evidence_policy"]
        self.assertTrue(policy["complete_state_replacement_is_state_sufficiency_only"])
        self.assertTrue(policy["additive_parent_sum_is_state_reconstruction_only"])
        self.assertTrue(policy["compute_edge_requires_source_recompute_intervention"])
        self.assertTrue(policy["parameter_scan_requires_replicated_compute_edge"])
        self.assertFalse(policy["single_neuron_scan_before_compute_edge"])
        self.assertFalse(policy["sealed_split_read"])

    def test_behavior_gate_keeps_contextual_and_parametric_routes_separate(self) -> None:
        summary_path = protocol.OUT_DIR / "phase557_behavior_summary.json"
        if not summary_path.exists():
            self.skipTest("Phase557 behavior denominator has not run")
        summary = read_json(summary_path)
        self.assertEqual(summary["models_authorized_for_contextual_internal_collection"], [])
        self.assertEqual(summary["natural_authorizations"], {
            "qwen3": ["color"],
            "glm4": ["color"],
            "deepseek7b": [],
        })
        self.assertFalse(summary["sealed_split_read"])

    def test_natural_source_intervention_is_coarse_and_legally_recomputed(self) -> None:
        self.assertEqual(source_intervention.CONFIRMATION_SPLIT, "behavior_confirmation")
        self.assertEqual(source_intervention.UNSEEN_SPLIT, "unseen_recombination")
        self.assertIn("correct_donor_replace", source_intervention.CONDITIONS)
        self.assertIn("wrong_depth_donor_replace", source_intervention.CONDITIONS)
        self.assertIn("relation_position_donor_replace", source_intervention.CONDITIONS)
        self.assertIn("channel_roll_donor_replace", source_intervention.CONDITIONS)
        source = Path(source_intervention.__file__).read_text(encoding="utf-8")
        self.assertIn('"source_position": "object_source_end"', source)
        self.assertIn('"query_end_patch_executed": False', source)
        self.assertNotIn("self_attn.register_forward_hook", source)
        self.assertNotIn("sealed_cases", source)


if __name__ == "__main__":
    unittest.main()
