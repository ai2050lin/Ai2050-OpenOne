#!/usr/bin/env python3
"""Artifact-level regression tests for the complete Phase574 evidence chain."""

from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P574 = ROOT / "tests/gpt5/result/phase574_query_source"


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


class Phase574ProtocolTests(unittest.TestCase):
    def test_frozen_denominator_and_sealed_commitment(self) -> None:
        protocol = read(P574 / "phase574_frozen_protocol.json")
        audit = read(P574 / "phase574_static_audit.json")
        commitment = read(P574 / "phase574_sealed_commitment.json")

        self.assertEqual(protocol["phase_id"], "Phase574")
        self.assertTrue(audit["valid"])
        self.assertEqual(protocol["candidate_worlds_per_split"], 1024)
        self.assertEqual(protocol["open_case_count"], 20480)
        self.assertEqual(protocol["sealed_case_count"], 4096)
        self.assertEqual(audit["world_group_count"], 6144)
        self.assertEqual(audit["counterfactual_invariant_failure_count"], 0)
        self.assertEqual(audit["prior_object_overlap_count"], 0)
        self.assertEqual(
            audit["open_cases_sha256"],
            sha256(P574 / "phase574_open_cases.jsonl.gz"),
        )
        self.assertEqual(
            commitment["sealed_cases_sha256"], audit["sealed_cases_sha256"]
        )
        self.assertFalse(protocol["sealed_split_read_for_analysis"])
        self.assertFalse(commitment["sealed_split_read_for_analysis"])
        self.assertFalse(commitment["sealed_behavior_executed"])

    def test_open_worlds_have_four_frozen_counterfactual_axes(self) -> None:
        groups: dict[str, dict[str, dict]] = defaultdict(dict)
        splits = set()
        rows = list(iter_jsonl(P574 / "phase574_open_cases.jsonl.gz"))
        self.assertEqual(len(rows), 20480)
        for row in rows:
            splits.add(row["split"])
            groups[row["base_case_id"]][row["variant"]] = row

        self.assertEqual(len(groups), 5120)
        self.assertEqual(splits, {
            "structure_discovery",
            "structure_confirmation",
            "causal_discovery",
            "causal_confirmation",
            "heldout_recombination",
        })
        expected_variants = {"base", "relation_swap", "object_swap", "order_swap"}
        for group in groups.values():
            self.assertEqual(set(group), expected_variants)
            base = group["base"]
            relation = group["relation_swap"]
            obj = group["object_swap"]
            order = group["order_swap"]
            self.assertEqual(base["context"], relation["context"])
            self.assertEqual(base["context"], obj["context"])
            self.assertEqual(base["target"], relation["other_relation_target"])
            self.assertEqual(base["other_relation_target"], relation["target"])
            self.assertEqual(base["query_relation"], obj["query_relation"])
            self.assertNotEqual(base["query_object"], obj["query_object"])
            self.assertNotEqual(base["target"], obj["target"])
            self.assertEqual(base["target"], order["target"])
            self.assertEqual(base["other_relation_target"], order["other_relation_target"])


class Phase574BehaviorTests(unittest.TestCase):
    def test_three_models_remain_separate_at_the_behavior_gate(self) -> None:
        qwen = read(P574 / "phase574_qwen3_behavior_summary.json")
        glm = read(P574 / "phase574_glm4_behavior_summary.json")
        ds = read(P574 / "phase574_deepseek7b_behavior_summary.json")

        self.assertTrue(qwen["authorized_for_query_source_trace"])
        self.assertEqual(qwen["selected_count_per_split"], {
            "heldout_recombination": 128,
            "structure_confirmation": 128,
            "structure_discovery": 128,
        })
        self.assertEqual(qwen["all_axis_qualified_counts_by_split"], {
            "heldout_recombination": 285,
            "structure_confirmation": 291,
            "structure_discovery": 298,
        })

        self.assertFalse(glm["authorized_for_query_source_trace"])
        self.assertTrue(glm["structure_behavior_gate_pass"])
        self.assertFalse(glm["heldout_behavior_gate_pass"])
        self.assertEqual(glm["all_axis_qualified_counts_by_split"]["heldout_recombination"], 98)

        self.assertFalse(ds["authorized_for_query_source_trace"])
        self.assertFalse(ds["structure_behavior_gate_pass"])
        self.assertEqual(ds["relation_qualified_counts_by_split"], {
            "structure_confirmation": 139,
            "structure_discovery": 144,
        })

        for model, summary in (("qwen3", qwen), ("glm4", glm), ("deepseek7b", ds)):
            rows = P574 / f"phase574_{model}_behavior_rows.jsonl.gz"
            self.assertEqual(summary["executed_behavior_row_count"], jsonl_count(rows))
            self.assertEqual(summary["rows_sha256"], sha256(rows))
            self.assertEqual(summary["noop_exact_text_mismatch_count"], 0)
            self.assertEqual(summary["noop_semantic_event_mismatch_count"], 0)
            self.assertFalse(summary["causal_splits_read"])
            self.assertFalse(summary["sealed_split_read"])


class Phase574NaturalTraceTests(unittest.TestCase):
    def test_fresh_natural_endpoints_replicate_without_prefix_leakage(self) -> None:
        protocol = read(P574 / "phase574_query_source_trace_protocol.json")
        summary = read(P574 / "phase574_qwen3_query_source_trace_summary.json")
        decision = read(P574 / "phase574_query_source_trace_decision.json")

        self.assertEqual(len(protocol["trace_layers"]), 20)
        self.assertEqual(len(protocol["causal_candidates"]), 8)
        self.assertEqual(summary["world_count"], 384)
        self.assertEqual(summary["trace_row_count"], 7680)
        self.assertEqual(summary["full_vector_snapshot_world_count"], 32)
        self.assertTrue(summary["causal_prefix_audit_pass"])
        self.assertEqual(summary["causal_prefix_maximum_relative_delta"], 0.0)
        self.assertEqual(summary["causal_prefix_mean_relative_delta"], 0.0)
        self.assertLess(summary["maximum_attention_reconstruction_relative_error"], 0.01)

        for split, report in summary["metrics_by_split"].items():
            self.assertEqual(report["world_count"], 128, split)
            self.assertGreaterEqual(report["layer5_relation_full_attention_event_rate"], 0.75)
            self.assertGreaterEqual(report["layer5_relation_full_attention_mean_relative_delta"], 0.05)
            self.assertGreaterEqual(report["layer24_relation_semantic_selection_pair_rate"], 0.75)
            self.assertGreaterEqual(report["layer24_object_semantic_selection_pair_rate"], 0.75)
            self.assertGreaterEqual(report["layer24_order_semantic_selection_pair_rate"], 0.75)

        rows = P574 / "phase574_qwen3_query_source_trace_rows.jsonl.gz"
        snapshots = P574 / "phase574_qwen3_discovery_vector_snapshots.pt"
        self.assertEqual(summary["rows_sha256"], sha256(rows))
        self.assertEqual(summary["snapshots_sha256"], sha256(snapshots))
        self.assertTrue(decision["coarse_query_source_causal_authorized"])
        self.assertFalse(decision["causal_splits_read"])
        self.assertFalse(decision["sealed_split_read"])


class Phase574CausalTests(unittest.TestCase):
    def test_coarse_query_source_candidates_fail_the_frozen_gate(self) -> None:
        summary = read(P574 / "phase574_qwen3_query_source_causal_summary.json")
        decision = read(P574 / "phase574_query_source_causal_decision.json")
        causal_protocol = read(P574 / "phase574_query_source_causal_protocol.json")

        self.assertEqual(summary["selected_world_count_by_split"], {
            "causal_confirmation": 128,
            "causal_discovery": 128,
        })
        self.assertEqual(summary["behavior_row_count"], 11264)
        self.assertEqual(summary["causal_row_count"], 9216)
        self.assertEqual(summary["generation_row_count"], 0)
        self.assertLess(summary["maximum_attention_reconstruction_relative_error"], 0.01)
        self.assertEqual(summary["eligible_discovery_candidate_ids"], [])
        self.assertFalse(summary["discovery_gate_pass"])
        self.assertFalse(summary["confirmation_gate_pass"])
        self.assertFalse(summary["full_generation_gate_pass"])
        self.assertFalse(summary["open_query_source_causal_gate_pass"])
        self.assertIsNone(summary["selected_candidate"])
        self.assertFalse(summary["head_channel_parameter_neuron_scan_executed"])
        self.assertFalse(summary["sealed_split_read"])

        audit = summary["discovery_familywise_pipeline_audit"]
        self.assertEqual(audit["candidate_count"], 8)
        self.assertEqual(audit["world_count"], 128)
        self.assertEqual(audit["permutation_count"], 1024)
        self.assertEqual(audit["count_at_least_observed"], 1024)
        self.assertEqual(audit["smoothed_tail_fraction"], 1.0)
        self.assertLess(max(audit["observed_mean_by_candidate"].values()), 0.0)
        self.assertTrue(all(
            not report["eligible"]
            for report in summary["discovery_candidate_metrics"].values()
        ))

        strongest = summary["discovery_candidate_metrics"][
            "query_relation_value_message__L13_L18"
        ]
        self.assertGreater(strongest["relation_route_switch_effect_positive_rate"], 0.80)
        self.assertGreater(strongest["relation_logit_switch_effect_mean"], 0.90)
        self.assertLess(strongest["relation_route_switch_effect_mean"], 0.01)
        self.assertGreater(strongest["restore_maximum_candidate_logit_delta"], 1.0)

        exact_restore = [
            report for candidate, report in summary["discovery_candidate_metrics"].items()
            if candidate.startswith("query_terminal_attention_output")
        ]
        self.assertTrue(all(report["restore_maximum_candidate_logit_delta"] == 0.0 for report in exact_restore))
        self.assertTrue(all(not report["eligible"] for report in exact_restore))

        self.assertEqual(causal_protocol["recipient_variant"], "base")
        self.assertTrue(causal_protocol["strict_object_control"][
            "recipient_and_object_donor_have_same_relation"
        ])
        self.assertTrue(causal_protocol["strict_object_control"][
            "recipient_and_object_donor_have_different_object"
        ])
        self.assertFalse(causal_protocol["head_channel_parameter_neuron_scan_allowed"])
        self.assertFalse(causal_protocol["sealed_split_read"])

        self.assertFalse(decision["discovery_gate_pass"])
        self.assertFalse(decision["confirmation_gate_pass"])
        self.assertFalse(decision["full_generation_gate_pass"])
        self.assertFalse(decision["new_sealed_validation_authorized"])
        self.assertFalse(decision["sealed_split_read"])
        self.assertIsNone(decision["selected_candidate_id"])

    def test_causal_artifact_counts_and_hash_chain(self) -> None:
        summary = read(P574 / "phase574_qwen3_query_source_causal_summary.json")
        paths = {
            "behavior": P574 / "phase574_qwen3_causal_behavior_rows.jsonl.gz",
            "causal": P574 / "phase574_qwen3_query_source_causal_rows.jsonl.gz",
            "generation": P574 / "phase574_qwen3_query_source_generation_rows.jsonl.gz",
        }
        self.assertEqual(jsonl_count(paths["behavior"]), 11264)
        self.assertEqual(jsonl_count(paths["causal"]), 9216)
        self.assertEqual(jsonl_count(paths["generation"]), 0)
        for kind, path in paths.items():
            self.assertEqual(summary[f"{kind}_rows_sha256"], sha256(path))


if __name__ == "__main__":
    unittest.main()
