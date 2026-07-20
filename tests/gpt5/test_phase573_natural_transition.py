#!/usr/bin/env python3
"""Artifact-level regression tests for the complete Phase573 evidence chain."""

from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P573 = ROOT / "tests/gpt5/result/phase573_natural_transition"


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


class Phase573ProtocolAndBehaviorTests(unittest.TestCase):
    def test_frozen_worlds_and_split_isolation(self) -> None:
        protocol = read(P573 / "phase573_frozen_protocol.json")
        audit = read(P573 / "phase573_static_audit.json")
        self.assertTrue(audit["valid"])
        self.assertEqual(protocol["candidate_worlds_per_split"], 1024)
        self.assertEqual(protocol["open_case_count"], 20480)
        self.assertEqual(protocol["sealed_case_count"], 4096)
        self.assertEqual(audit["world_group_count"], 6144)
        self.assertEqual(audit["counterfactual_invariant_failure_count"], 0)
        self.assertEqual(audit["prior_open_object_overlap_count"], 0)
        self.assertEqual(
            audit["open_cases_sha256"],
            sha256(P573 / "phase573_open_cases.jsonl.gz"),
        )

    def test_three_model_behavior_qualification_is_not_collapsed(self) -> None:
        qwen = read(P573 / "phase573_qwen3_behavior_summary.json")
        glm = read(P573 / "phase573_glm4_behavior_summary.json")
        ds = read(P573 / "phase573_deepseek7b_behavior_summary.json")
        self.assertTrue(qwen["authorized_for_natural_trace"])
        self.assertEqual(qwen["selected_count_per_split"], {
            "structure_discovery": 128,
            "structure_confirmation": 128,
            "heldout_recombination": 128,
        })
        self.assertFalse(glm["authorized_for_natural_trace"])
        self.assertEqual(glm["all_axis_qualified_counts_by_split"]["heldout_recombination"], 103)
        self.assertFalse(ds["structure_behavior_gate_pass"])
        self.assertEqual(ds["relation_qualified_counts_by_split"], {
            "structure_discovery": 141,
            "structure_confirmation": 188,
        })
        for report in (qwen, glm, ds):
            self.assertEqual(report["noop_exact_text_mismatch_count"], 0)
            self.assertEqual(report["noop_semantic_event_mismatch_count"], 0)
            self.assertFalse(report["causal_splits_read"])
            self.assertFalse(report["sealed_split_read"])


class Phase573TraceTests(unittest.TestCase):
    def test_causal_prefix_audit_and_replicated_events(self) -> None:
        summary = read(P573 / "phase573_qwen3_natural_trace_summary.json")
        self.assertEqual(summary["world_count"], 384)
        self.assertEqual(summary["world_count_per_split"], 128)
        self.assertEqual(summary["layer_count"], 36)
        self.assertTrue(summary["causal_mask_prefix_audit_pass"])
        self.assertEqual(summary["maximum_fixed_prefix_relative_delta"], 0.0)
        self.assertEqual(summary["mean_fixed_prefix_relative_delta"], 0.0)
        self.assertEqual(summary["earliest_state_event"]["layer"], 5)
        self.assertEqual(
            summary["earliest_state_event"]["receiver_role"], "query_terminal"
        )
        route = summary["earliest_routing_event"]
        self.assertEqual(route["layer"], 24)
        self.assertEqual(route["receiver_role"], "answer_boundary")
        self.assertTrue(summary["coarse_message_causal_authorized"])
        self.assertFalse(summary["output_embedding_direction_used"])
        self.assertFalse(summary["causal_splits_read"])
        self.assertFalse(summary["sealed_split_read"])
        self.assertEqual(
            summary["trace_rows_sha256"],
            sha256(P573 / "phase573_qwen3_natural_trace_rows.jsonl.gz"),
        )
        self.assertEqual(
            summary["routing_rows_sha256"],
            sha256(P573 / "phase573_qwen3_natural_routing_rows.jsonl.gz"),
        )


class Phase573CausalTests(unittest.TestCase):
    def test_two_open_causal_splits_pass_with_controls(self) -> None:
        summary = read(P573 / "phase573_qwen3_coarse_message_causal_summary.json")
        self.assertTrue(summary["coarse_message_causal_gate_pass"])
        self.assertEqual(summary["selected_world_count_by_split"], {
            "causal_discovery": 128,
            "causal_confirmation": 128,
        })
        self.assertEqual(summary["behavior_row_count"], 11264)
        self.assertEqual(summary["causal_row_count"], 3584)
        self.assertEqual(summary["same_shape_baseline_mismatch_count"], 0)
        self.assertLess(summary["maximum_reconstruction_relative_error"], 0.02)
        for split in ("causal_discovery", "causal_confirmation"):
            report = summary["metrics_by_split"][split]
            self.assertTrue(report["causal_gate_pass"])
            conditions = report["conditions"]
            removal = conditions["selected_edge_remove"]
            replacement = conditions["paired_relation_selected_replace"]
            self.assertGreaterEqual(removal["positive_effect_rate"], 0.65)
            self.assertEqual(replacement["positive_effect_rate"], 1.0)
            self.assertGreaterEqual(replacement["donor_candidate_win_rate"], 0.70)
            self.assertGreater(report["selected_vs_nonselected_removal_mean_gap"], 9.0)
            self.assertGreater(report["paired_replace_vs_strongest_control_mean_gap"], 14.0)
        self.assertEqual(
            summary["behavior_rows_sha256"],
            sha256(P573 / "phase573_qwen3_causal_split_behavior_rows.jsonl.gz"),
        )
        self.assertEqual(
            summary["causal_rows_sha256"],
            sha256(P573 / "phase573_qwen3_coarse_message_causal_rows.jsonl.gz"),
        )

    def test_sealed_result_preserves_claim_boundary(self) -> None:
        commitment = read(P573 / "phase573_sealed_commitment.json")
        receipt = read(P573 / "phase573_sealed_execution_receipt.json")
        summary = read(P573 / "phase573_qwen3_sealed_validation_summary.json")
        decision = read(P573 / "phase573_sealed_validation_decision.json")
        sealed_cases = P573 / "protocol/private/phase573_sealed_cases.jsonl.gz"
        self.assertEqual(commitment["sealed_cases_sha256"], sha256(sealed_cases))
        self.assertEqual(receipt["sealed_cases_sha256_verified"], sha256(sealed_cases))
        self.assertEqual(receipt["sealed_case_count_read"], 4096)
        self.assertTrue(summary["sealed_causal_gate_pass"])
        self.assertEqual(summary["selected_world_count"], 128)
        self.assertEqual(summary["behavior_row_count"], 5632)
        self.assertEqual(summary["causal_row_count"], 1792)
        self.assertEqual(summary["same_shape_baseline_mismatch_count"], 0)
        replacement = summary["condition_metrics"][
            "paired_relation_selected_replace"
        ]
        self.assertEqual(replacement["positive_effect_rate"], 1.0)
        self.assertGreater(replacement["donor_candidate_win_rate"], 0.60)
        self.assertTrue(decision["sealed_causal_gate_pass"])
        self.assertIn("relation selection rule closure", decision["claim_not_allowed"])
        self.assertIn("cross-model portability", decision["claim_not_allowed"])
        self.assertIn("72-mechanism closure", decision["claim_not_allowed"])

    def test_artifact_row_counts(self) -> None:
        self.assertEqual(
            jsonl_count(P573 / "phase573_qwen3_coarse_message_causal_rows.jsonl.gz"),
            3584,
        )
        self.assertEqual(
            jsonl_count(P573 / "phase573_qwen3_sealed_causal_rows.jsonl.gz"),
            1792,
        )


if __name__ == "__main__":
    unittest.main()
