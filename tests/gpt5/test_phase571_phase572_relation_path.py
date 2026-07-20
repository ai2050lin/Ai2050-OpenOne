#!/usr/bin/env python3
"""Artifact-level regression tests for the Phase571-572 relation path."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase569_role_position_utils import span_in_parent  # noqa: E402


P571 = ROOT / "tests/gpt5/result/phase571_relation_block"
P572 = ROOT / "tests/gpt5/result/phase572_relation_joint"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_count(path: Path) -> int:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


class Phase571ArtifactTests(unittest.TestCase):
    def test_static_protocol_and_exact_matching(self) -> None:
        protocol = read(P571 / "phase571_frozen_protocol.json")
        audit = read(P571 / "phase571_static_audit.json")
        self.assertTrue(audit["valid"])
        self.assertEqual(protocol["models"], ["qwen3", "glm4", "deepseek7b"])
        self.assertEqual(protocol["candidate_cases_per_pool_model"], 1024)
        self.assertEqual(protocol["open_case_count"], 9216)
        self.assertEqual(protocol["sealed_case_count"], 3072)
        self.assertEqual(audit["pool_object_overlap_count"], 0)
        self.assertEqual(
            audit["open_cases_sha256"], sha256(P571 / "phase571_open_cases.jsonl.gz")
        )
        for model in protocol["models"]:
            matched = read(P571 / f"phase571_{model}_matched_behavior_summary.json")
            self.assertTrue(matched["qualified_for_signed_write_trace"])
            self.assertTrue(matched["matched_stratum_distributions_exactly_equal"]["block_discovery"])
            self.assertTrue(matched["matched_stratum_distributions_exactly_equal"]["block_confirmation"])
            self.assertGreaterEqual(
                matched["matched_case_count_per_phenotype_by_pool"]["block_discovery"], 128
            )
            self.assertGreaterEqual(
                matched["matched_case_count_per_phenotype_by_pool"]["block_confirmation"], 128
            )

    def test_permutation_and_coarse_causal_gates(self) -> None:
        permutation = read(P571 / "phase571_max_block_permutation_audit.json")
        self.assertEqual(permutation["rounds"], 1024)
        self.assertEqual(permutation["passed_models"], ["qwen3", "deepseek7b"])
        by_model = {row["model"]: row for row in permutation["model_reports"]}
        self.assertLessEqual(by_model["qwen3"]["selected_block_familywise_discovery_p"], 0.05)
        self.assertLessEqual(by_model["deepseek7b"]["selected_block_familywise_discovery_p"], 0.05)
        self.assertIsNone(by_model["glm4"]["selected_block"])
        causal = read(P571 / "phase571_coarse_block_causal_analysis.json")
        self.assertEqual(causal["passed_models"], ["qwen3"])
        reports = {row["model"]: row for row in causal["model_reports"]}
        self.assertAlmostEqual(
            reports["qwen3"]["derived_behavior_effects"]["confusion_repair"], 0.3125
        )
        self.assertAlmostEqual(
            reports["deepseek7b"]["derived_behavior_effects"]["confusion_repair"],
            0.0703125,
        )

    def test_donor_denominator_and_negative_relation_gate(self) -> None:
        execution = read(P571 / "phase571_qwen3_relation_donor_execution_summary.json")
        rows_path = P571 / "phase571_qwen3_relation_donor_rows.jsonl.gz"
        self.assertEqual(execution["rows_sha256"], sha256(rows_path))
        self.assertEqual(execution["final_pair_count"], 128)
        self.assertEqual(jsonl_count(rows_path), 2048)
        analysis = read(P571 / "phase571_relation_donor_analysis.json")
        self.assertFalse(analysis["relation_selection_donor_gate_pass"])
        self.assertTrue(analysis["terminal_answer_content_transport_observed"])
        effects = analysis["derived_effects"]
        self.assertAlmostEqual(effects["matched_answer_exit_confusion_repair"], 0.796875)
        self.assertAlmostEqual(effects["matched_query_entry_confusion_repair"], 0.0)
        self.assertAlmostEqual(effects["matched_target_fact_entry_confusion_repair"], 0.0390625)
        self.assertLess(effects["answer_entry_specificity_over_random"], 0.0)
        self.assertFalse(analysis["sealed_split_read"])
        self.assertFalse(analysis["head_channel_parameter_neuron_scan_executed"])

    def test_value_span_uses_final_occurrence(self) -> None:
        prompt = "For entity alphawood, the tag label is alpha."
        parent = prompt
        first = span_in_parent(prompt, parent, "alpha")
        last = span_in_parent(prompt, parent, "alpha", last_child=True)
        self.assertLess(first[0], last[0])
        self.assertEqual(prompt[last[0]:last[1]], "alpha")


class Phase572ArtifactTests(unittest.TestCase):
    def test_fresh_behavior_denominator(self) -> None:
        protocol = read(P572 / "phase572_frozen_protocol.json")
        audit = read(P572 / "phase572_static_audit.json")
        behavior = read(P572 / "phase572_qwen3_behavior_summary.json")
        self.assertTrue(audit["valid"])
        self.assertEqual(protocol["candidate_case_count"], 1024)
        self.assertEqual(audit["phase571_open_object_overlap_count"], 0)
        self.assertEqual(behavior["noop_semantic_event_mismatch_count"], 0)
        self.assertEqual(behavior["exact_matched_pair_count"], 252)
        self.assertGreaterEqual(behavior["matched_target_other_pair_count"], 8)
        self.assertTrue(behavior["qualified_for_joint_causal"])

    def test_joint_gate_closes_late_distributed_state(self) -> None:
        summary = read(P572 / "phase572_qwen3_joint_causal_summary.json")
        rows_path = P572 / "phase572_qwen3_joint_causal_rows.jsonl.gz"
        self.assertEqual(summary["rows_sha256"], sha256(rows_path))
        self.assertEqual(summary["final_pair_count"], 128)
        self.assertEqual(summary["condition_count"], 11)
        self.assertEqual(jsonl_count(rows_path), 2816)
        analysis = read(P572 / "phase572_joint_causal_analysis.json")
        self.assertFalse(analysis["joint_relation_state_gate_pass"])
        self.assertEqual(
            analysis["observed_scope"],
            "late_answer_content_transport_not_joint_relation_state",
        )
        repairs = analysis["confusion_repair_by_role_set"]
        self.assertAlmostEqual(repairs["answer"], 0.3203125)
        self.assertAlmostEqual(repairs["query_fact_answer"], 0.203125)
        self.assertLess(analysis["joint_gain_over_best_single"], 0.0)
        contributions = analysis["leave_one_out_contributions"]
        self.assertEqual(contributions["query_contribution"], 0.0)
        self.assertLess(contributions["fact_contribution"], 0.0)
        self.assertGreater(contributions["answer_contribution"], 0.0)
        decision = read(P572 / "phase572_stage_decision.json")
        self.assertTrue(decision["late_static_joint_role_route_closed"])
        self.assertFalse(decision["late_block_head_channel_parameter_neuron_scan_allowed"])
        self.assertFalse(decision["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
