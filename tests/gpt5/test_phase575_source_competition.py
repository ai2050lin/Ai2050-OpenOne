#!/usr/bin/env python3
"""Artifact-level regression tests for the complete Phase575 evidence chain."""

from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P575 = ROOT / "tests/gpt5/result/phase575_source_competition"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def jsonl_count(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


class Phase575ProtocolTests(unittest.TestCase):
    def test_frozen_five_variant_denominator(self) -> None:
        protocol = read(P575 / "phase575_frozen_protocol.json")
        audit = read(P575 / "phase575_static_audit.json")
        self.assertEqual(protocol["phase_id"], "Phase575")
        self.assertTrue(audit["valid"])
        self.assertEqual(protocol["candidate_worlds_per_split"], 1024)
        self.assertEqual(protocol["open_case_count"], 25600)
        self.assertEqual(protocol["sealed_case_count"], 5120)
        self.assertEqual(audit["world_group_count"], 6144)
        self.assertEqual(audit["prior_open_object_overlap_count"], 0)
        self.assertEqual(audit["counterfactual_invariant_failure_count"], 0)
        self.assertEqual(
            audit["open_cases_sha256"],
            sha256(P575 / "phase575_open_cases.jsonl.gz"),
        )

    def test_each_open_world_has_all_five_counterfactuals(self) -> None:
        groups: dict[str, dict[str, dict]] = defaultdict(dict)
        rows = list(iter_jsonl(P575 / "phase575_open_cases.jsonl.gz"))
        self.assertEqual(len(rows), 25600)
        for row in rows:
            self.assertFalse(row["sealed"])
            groups[row["base_case_id"]][row["variant"]] = row
        self.assertEqual(len(groups), 5120)
        expected = {
            "base",
            "relation_swap",
            "object_swap",
            "relation_object_swap",
            "order_swap",
        }
        for group in groups.values():
            self.assertEqual(set(group), expected)
            self.assertEqual(
                len(
                    {
                        group[name]["target"]
                        for name in (
                            "base",
                            "relation_swap",
                            "object_swap",
                            "relation_object_swap",
                        )
                    }
                ),
                4,
            )
            self.assertEqual(group["base"]["target"], group["order_swap"]["target"])


class Phase575BehaviorTests(unittest.TestCase):
    def test_only_qwen3_enters_the_internal_ledger(self) -> None:
        qwen = read(P575 / "phase575_qwen3_behavior_summary.json")
        glm = read(P575 / "phase575_glm4_behavior_summary.json")
        ds = read(P575 / "phase575_deepseek7b_behavior_summary.json")
        self.assertTrue(qwen["authorized_for_natural_ledger"])
        self.assertEqual(qwen["selected_count_per_split"], {
            "heldout_recombination": 128,
            "structure_confirmation": 128,
            "structure_discovery": 128,
        })
        self.assertFalse(glm["authorized_for_natural_ledger"])
        self.assertTrue(glm["structure_behavior_gate_pass"])
        self.assertFalse(glm["heldout_behavior_gate_pass"])
        self.assertFalse(ds["authorized_for_natural_ledger"])
        self.assertFalse(ds["structure_behavior_gate_pass"])
        for summary in (qwen, glm, ds):
            self.assertEqual(summary["noop_exact_text_mismatch_count"], 0)
            self.assertEqual(summary["noop_semantic_event_mismatch_count"], 0)
            self.assertFalse(summary["causal_splits_read"])
            self.assertFalse(summary["sealed_split_read"])


class Phase575NaturalStructureTests(unittest.TestCase):
    def test_full_depth_ledger_is_exact_and_data_first(self) -> None:
        summary = read(P575 / "phase575_qwen3_natural_ledger_summary.json")
        analysis = read(P575 / "phase575_qwen3_natural_structure_analysis.json")
        decision = read(P575 / "phase575_natural_structure_decision.json")
        self.assertEqual(summary["world_count"], 384)
        self.assertEqual(summary["layer_count"], 36)
        self.assertEqual(summary["ledger_row_count"], 13824)
        self.assertEqual(summary["attention_weight_reconstruction_max_abs_error"], 0.0)
        self.assertEqual(summary["causal_prefix_max_relative_delta"], 0.0)
        self.assertEqual(summary["duplicate_trace_max_abs_delta"], 0.0)
        self.assertTrue(all(summary["quality_gates"].values()))
        self.assertIn(
            {"start": 24, "end": 24},
            analysis["replicated_event_bands"]["score"]["answer_boundary"],
        )
        self.assertIn(
            {"start": 23, "end": 34},
            analysis["replicated_event_bands"]["weight"]["answer_boundary"],
        )
        self.assertEqual(
            analysis["replicated_event_bands"]["score"]["query_terminal"], []
        )
        self.assertEqual(decision["replicated_routing_coordinate_count"], 24)
        self.assertTrue(decision["causal_protocol_authorized"])
        self.assertFalse(decision["theory_formula_frozen_before_discovery"])

    def test_layer24_natural_event_has_three_split_floors(self) -> None:
        analysis = read(P575 / "phase575_qwen3_natural_structure_analysis.json")
        coordinate = next(
            row
            for row in analysis["coordinate_rows"]
            if row["layer"] == 24 and row["receiver"] == "answer_boundary"
        )
        score = coordinate["replicated_channels"]["score"]
        weight = coordinate["replicated_channels"]["weight"]
        self.assertTrue(score["replicated_routing_event"])
        self.assertEqual(score["direction_rate_floor"], 1.0)
        self.assertGreaterEqual(score["semantic_rate_floor"], 0.80)
        self.assertTrue(weight["replicated_routing_event"])
        self.assertGreaterEqual(weight["direction_rate_floor"], 0.99)
        self.assertGreaterEqual(weight["semantic_rate_floor"], 0.96)
        self.assertGreaterEqual(weight["order_rate_floor"], 0.97)


class Phase575OpenCausalTests(unittest.TestCase):
    def test_discovery_selects_score_after_maximum_branch_resampling(self) -> None:
        summary = read(P575 / "phase575_qwen3_routing_causal_discovery_summary.json")
        decision = read(P575 / "phase575_routing_causal_discovery_decision.json")
        score = summary["branch_results"]["score"]
        self.assertEqual(summary["world_count"], 128)
        self.assertEqual(summary["row_count"], 2432)
        self.assertEqual(summary["selected_branch"], "score")
        self.assertTrue(score["physical_routing_gate_pass"])
        self.assertTrue(score["behavior_gate_pass"])
        self.assertEqual(score["relation_route_effect_positive_rate"], 1.0)
        self.assertGreater(score["relation_route_effect_mean"], 0.11)
        self.assertGreater(score["relation_logit_effect_mean"], 18.0)
        self.assertGreater(score["relation_vs_object_gap"], 0.07)
        self.assertGreater(score["relation_vs_order_gap"], 0.06)
        audit = summary["maximum_branch_pipeline_resample"]
        self.assertEqual(audit["resample_count"], 1024)
        self.assertEqual(audit["count_at_least_observed"], 0)
        self.assertLess(audit["smoothed_tail_fraction"], 0.01)
        self.assertTrue(decision["confirmation_internal_state_authorized"])
        self.assertFalse(decision["sealed_split_authorized"])

    def test_independent_confirmation_replicates_selected_score_branch(self) -> None:
        summary = read(P575 / "phase575_qwen3_routing_causal_confirmation_summary.json")
        relation = summary["condition_metrics"]["score_relation_replace"]
        restore = summary["condition_metrics"]["score_relation_weight_restore"]
        self.assertEqual(summary["selected_branch"], "score")
        self.assertEqual(summary["world_count"], 128)
        self.assertEqual(summary["row_count"], 1024)
        self.assertTrue(summary["open_confirmation_pass"])
        self.assertEqual(relation["relation_route_effect_positive_rate"], 1.0)
        self.assertGreater(relation["relation_route_effect_mean"], 0.11)
        self.assertGreater(relation["relation_logit_effect_mean"], 18.0)
        self.assertEqual(restore["relation_route_effect_mean"], 0.0)
        self.assertEqual(restore["maximum_candidate_logit_delta"], 0.0)
        self.assertLess(
            summary["paired_pipeline_resample"]["smoothed_tail_fraction"], 0.01
        )

    def test_open_full_generation_changes_complete_output(self) -> None:
        summary = read(P575 / "phase575_qwen3_full_generation_summary.json")
        metrics = summary["condition_metrics"]
        self.assertTrue(summary["full_generation_gate_pass"])
        self.assertEqual(summary["world_count"], 128)
        self.assertEqual(summary["row_count"], 1280)
        self.assertGreaterEqual(metrics["natural_baseline"]["base_target_rate"], 0.98)
        self.assertGreaterEqual(
            metrics["score_relation_replace"]["relation_target_rate"], 0.53
        )
        self.assertGreater(summary["relation_target_rate_gain_over_natural"], 0.53)
        self.assertGreater(summary["relation_vs_object_target_rate_gap"], 0.40)
        self.assertGreater(summary["relation_vs_order_target_rate_gap"], 0.41)
        self.assertEqual(summary["restore_exact_text_mismatch_count"], 0)
        self.assertEqual(summary["restore_semantic_event_mismatch_count"], 0)


class Phase575SealedTests(unittest.TestCase):
    def test_one_shot_seal_replicates_without_claiming_closure(self) -> None:
        receipt = read(P575 / "phase575_sealed_execution_receipt.json")
        summary = read(P575 / "phase575_qwen3_sealed_validation_summary.json")
        decision = read(P575 / "phase575_sealed_validation_decision.json")
        self.assertTrue(receipt["one_shot"])
        self.assertTrue(receipt["sealed_split_read"])
        self.assertEqual(summary["sealed_candidate_world_count"], 1024)
        self.assertEqual(summary["selected_world_count"], 128)
        self.assertEqual(
            summary["behavior_analysis"]["relation_qualified_world_count"], 763
        )
        self.assertEqual(
            summary["behavior_analysis"]["five_variant_qualified_world_count"], 215
        )
        self.assertTrue(summary["causal_analysis"]["causal_gate_pass"])
        self.assertTrue(
            summary["generation_analysis"]["full_generation_gate_pass"]
        )
        self.assertTrue(summary["sealed_validation_pass"])
        self.assertTrue(decision["sealed_validation_pass"])
        self.assertEqual(decision["candidate_status"], "sealed_local_causal_replication")
        self.assertFalse(decision["strict_mechanism_closure_claimed"])
        self.assertFalse(decision["cross_model_mechanism_claimed"])
        self.assertFalse(decision["broad_language_encoding_claimed"])
        self.assertFalse(decision["phase575_seal_may_be_reopened"])

    def test_sealed_causal_and_generation_effects_are_specific(self) -> None:
        summary = read(P575 / "phase575_qwen3_sealed_validation_summary.json")
        causal = summary["causal_analysis"]
        relation = causal["condition_metrics"]["score_relation_replace"]
        generation = summary["generation_analysis"]
        generated = generation["condition_metrics"]
        self.assertEqual(relation["relation_route_effect_positive_rate"], 1.0)
        self.assertGreater(relation["relation_route_effect_mean"], 0.113)
        self.assertGreater(relation["relation_logit_effect_mean"], 17.9)
        self.assertGreater(causal["relation_vs_object_gap"], 0.071)
        self.assertGreater(causal["relation_vs_order_gap"], 0.073)
        self.assertGreaterEqual(
            generated["score_relation_replace"]["relation_target_rate"], 0.44
        )
        self.assertLessEqual(
            generated["score_object_replace"]["relation_target_rate"], 0.055
        )
        self.assertLessEqual(
            generated["score_order_replace"]["relation_target_rate"], 0.102
        )
        self.assertEqual(generation["restore_exact_text_mismatch_count"], 0)
        self.assertEqual(generation["restore_semantic_event_mismatch_count"], 0)

    def test_sealed_artifact_counts_and_hashes(self) -> None:
        summary = read(P575 / "phase575_qwen3_sealed_validation_summary.json")
        paths = {
            "behavior": P575 / "phase575_qwen3_sealed_behavior_rows.jsonl.gz",
            "causal": P575 / "phase575_qwen3_sealed_causal_rows.jsonl.gz",
            "generation": P575 / "phase575_qwen3_sealed_generation_rows.jsonl.gz",
        }
        expected_counts = {"behavior": 6400, "causal": 640, "generation": 1280}
        for kind, path in paths.items():
            self.assertEqual(jsonl_count(path), expected_counts[kind])
            self.assertEqual(summary[f"{kind}_rows_sha256"], sha256(path))


if __name__ == "__main__":
    unittest.main()
