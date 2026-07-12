from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P386 = ROOT / "tests/gpt5/result/phase386_multitime_relation_atlas"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase386MultitimeRelationTests(unittest.TestCase):
    def test_fresh_behavior_denominator_is_not_backfilled(self) -> None:
        summary = read_json(P386 / "phase386_behavior_freeze_summary.json")
        self.assertEqual(summary["denominator"]["candidate_case_count"], 2880)
        self.assertEqual(summary["eligible_mechanisms"], [
            "relation_binding", "entity_recency", "field_extraction"
        ])
        self.assertEqual(summary["denominator"]["selected_case_count"], 576)
        self.assertFalse(summary["claim_boundary"]["excluded_mechanism_may_be_replaced"])
        gates = {row["mechanism_id"]: row for row in summary["gates"]}
        self.assertFalse(gates["number_agreement"]["eligible"])
        self.assertFalse(gates["target_vs_wrong"]["eligible"])
        self.assertFalse(gates["missing_condition_control"]["eligible"])

    def test_teacher_forced_path_is_retired_for_incremental_cache(self) -> None:
        amendment = read_json(P386 / "phase386_incremental_contract_amendment.json")
        self.assertEqual(
            amendment["trigger"]["deepseek7b_required_transition_pass_count"], 94
        )
        self.assertEqual(amendment["probe"]["case_count"], 2)
        self.assertTrue(amendment["probe"]["all_actual_incremental_transitions_match"])
        self.assertEqual(
            amendment["retired_contract"]["status"], "engineering_pilot_only"
        )
        self.assertEqual(
            amendment["replacement_contract"]["generation_path"],
            "actual_incremental_kv_cache",
        )

    def test_incremental_event_ledgers_are_complete_and_conserved(self) -> None:
        instrument = read_json(P386 / "phase386_instrument_audit_summary.json")
        discovery = read_json(P386 / "phase386_discovery_collection_summary.json")
        calibration = read_json(P386 / "phase386_calibration_collection_summary.json")
        physical = read_json(P386 / "phase386_physical_collection_summary.json")
        self.assertTrue(instrument["results"]["all_three_model_instruments_valid"])
        self.assertEqual(instrument["denominator"]["model_call_count"], 192)
        self.assertTrue(discovery["results"]["all_discovery_artifacts_valid"])
        self.assertEqual(discovery["denominator"]["case_count"], 288)
        self.assertEqual(discovery["denominator"]["model_call_count"], 1850)
        self.assertEqual(discovery["denominator"]["layer_file_count"], 9984)
        self.assertTrue(calibration["results"]["all_calibration_artifacts_valid"])
        self.assertTrue(physical["results"]["all_physical_holdout_artifacts_valid"])
        self.assertTrue(
            all(
                max(row["gate_maxima"].values()) <= 0.01
                for stage in (instrument, discovery, calibration, physical)
                for row in stage["models"]
            )
        )

    def test_prediction_funnel_preserves_strict_boundaries(self) -> None:
        discovery = read_json(P386 / "phase386_discovery_relation_summary.json")
        calibration = read_json(P386 / "phase386_calibration_summary.json")
        physical = read_json(P386 / "phase386_physical_summary.json")
        self.assertEqual(
            discovery["denominator"]["crossmodel_frozen_candidate_count"], 135
        )
        self.assertEqual(
            discovery["candidate_counts"]["neuron_channel_relation_candidate_count"], 5
        )
        self.assertEqual(
            calibration["results"]["crossmodel_relation_replication_count"], 117
        )
        self.assertEqual(
            calibration["results"]["crossmodel_predictive_relation_path_count"], 12
        )
        self.assertEqual(physical["results"]["physical_relation_replication_count"], 11)
        self.assertEqual(
            physical["results"]["physical_predictive_relation_path_count"], 10
        )
        self.assertFalse(physical["results"]["causal_relation_established"])
        self.assertFalse(physical["results"]["language_encoding_closed"])
        self.assertFalse(physical["authorization"]["additional_holdout_reuse"])

    def test_only_one_survivor_is_upstream_and_none_is_mlp_channel(self) -> None:
        rows = [
            row
            for row in read_jsonl(P386 / "phase386_physical_candidate_rows.jsonl")
            if row["physical_predictive_relation_path_gate_pass"]
        ]
        upstream = [
            row
            for row in rows
            if row["source_coordinate"] == "source_encoded"
            and row["target_coordinate"] == "query_integrated"
        ]
        terminal = [
            row
            for row in rows
            if row["source_coordinate"] == "target_encoded"
            and row["target_coordinate"] == "post_decision_next_token"
        ]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["physical_holdout_used"] for row in rows))
        self.assertEqual(len(upstream), 1)
        self.assertEqual(len(terminal), 9)
        self.assertEqual(upstream[0]["mechanism_id"], "relation_binding")
        self.assertEqual(upstream[0]["vector_family"], "attention_head_state")
        self.assertFalse(any(row["vector_family"] == "mlp_channel_product" for row in rows))
        self.assertFalse(any(row["causal_path_claim"] for row in rows))

    def test_atlas_and_client_publish_identical_phase386_boundary(self) -> None:
        names = (
            "manifest.json",
            "progress.json",
            "client_index.json",
            "phase386_relation_stage_summary.json",
            "phase386_evidence_nodes.jsonl",
            "phase386_evidence_edges.jsonl",
            "phase386_physical_summary.json",
            "phase386_physical_candidate_rows.jsonl",
        )
        for name in names:
            self.assertEqual((ATLAS / name).read_bytes(), (CLIENT / name).read_bytes(), name)
        manifest = read_json(ATLAS / "manifest.json")
        progress = read_json(ATLAS / "progress.json")
        latest_phase = int(manifest["last_phase"].split("Phase", 1)[1].split("-", 1)[0])
        self.assertGreaterEqual(latest_phase, 389)
        self.assertEqual(manifest["phase386"]["physical_predictive_relation_count"], 10)
        self.assertEqual(manifest["phase386"]["language_path_count"], 0)
        self.assertEqual(
            progress["multitime_relation_stage"]["physical_predictive_relations"],
            {"numerator": 10, "denominator": 12},
        )
        self.assertEqual(
            progress["multitime_relation_stage"]["single_neuron_causal_paths"]["numerator"],
            0,
        )
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        for root in (NEURON, NEURON_CLIENT):
            neuron = read_json(root / "manifest.json")
            self.assertGreaterEqual(neuron["phase"], 389)
            self.assertEqual(
                neuron["phase386_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertEqual(
                neuron["phase389_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertFalse(neuron["evidence_boundary"]["single_unit_causal_closure"])


if __name__ == "__main__":
    unittest.main()
