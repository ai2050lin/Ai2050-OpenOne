#!/usr/bin/env python3
"""Contract tests for the Phase414 observer/event preflight."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase414_observer_event_preflight import OUT  # noqa: E402


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase414ObserverEventPreflightTest(unittest.TestCase):
    def test_complete_natural_replay_is_identity_not_layer_curve(self) -> None:
        audit = read_json("phase414_natural_replay_identity_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_natural_replay_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["synthetic_case_count"], 12)
        self.assertEqual(audit["complete_natural_replay_cell_count"], 60)
        self.assertEqual(audit["complete_natural_replay_exact_count"], 60)
        self.assertEqual(audit["complete_natural_replay_failure_count"], 0)
        self.assertEqual(
            audit["case_with_layerwise_terminal_kernel_variation_count"], 0
        )
        self.assertFalse(
            audit["natural_complete_replay_is_layerwise_candidate_curve"]
        )
        self.assertGreater(
            audit["incomplete_local_state_replay_failure_count"], 0
        )
        self.assertEqual(len(rows), 60)
        self.assertTrue(all(row["complete_natural_replay_exact"] for row in rows))

    def test_trajectory_objects_remain_typed(self) -> None:
        ontology = read_json("phase414_trajectory_ontology.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_trajectory_object_rows.jsonl"
        )
        self.assertTrue(ontology["valid"])
        self.assertEqual(ontology["registered_trajectory_object_count"], 5)
        self.assertEqual(
            ontology["native_generation_time_probability_trajectory_count"], 1
        )
        self.assertEqual(ontology["observer_indexed_layer_trajectory_count"], 1)
        self.assertEqual(
            ontology["natural_descriptive_physical_trajectory_count"], 1
        )
        self.assertEqual(ontology["causal_effect_trajectory_count"], 1)
        self.assertEqual(ontology["instrument_identity_object_count"], 1)
        self.assertEqual(
            ontology[
                "unqualified_generic_intermediate_candidate_trajectory_count"
            ],
            0,
        )
        self.assertFalse(ontology["mixing_objects_without_type_label_authorized"])
        self.assertEqual(len(rows), 5)

    def test_observer_readability_is_indexed_and_not_native_probability(self) -> None:
        audit = read_json("phase414_observer_readability_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_observer_readability_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["observer_index_required"])
        self.assertEqual(audit["observer_count"], 2)
        self.assertEqual(audit["observer_cell_count"], 120)
        self.assertEqual(audit["case_observer_trajectory_count"], 24)
        self.assertEqual(audit["varying_case_observer_trajectory_count"], 24)
        self.assertGreater(audit["same_state_observer_disagreement_cell_count"], 0)
        self.assertEqual(audit["native_intermediate_probability_count"], 0)
        self.assertEqual(len(rows), 120)
        self.assertTrue(all(row["observer_indexed"] for row in rows))
        self.assertTrue(
            all(not row["native_intermediate_probability"] for row in rows)
        )

    def test_variable_length_events_are_disjoint_and_report_outside_mass(self) -> None:
        contract = read_json("phase414_variable_length_event_contract.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_candidate_event_rows.jsonl"
        )
        invalid = read_jsonl(
            OUT / "protocol/private/phase414_invalid_event_panel_rows.jsonl"
        )
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["registered_event_count"], 4)
        self.assertEqual(contract["event_length_set"], [1, 2, 3])
        self.assertFalse(contract["equal_length_required"])
        self.assertTrue(contract["pairwise_prefix_free"])
        self.assertTrue(contract["all_events_eos_closed"])
        self.assertAlmostEqual(contract["panel_mass"], 0.51)
        self.assertAlmostEqual(contract["outside_mass"], 0.49)
        self.assertEqual(contract["invalid_prefix_conflict_panel_count"], 1)
        self.assertEqual(
            contract["invalid_prefix_conflict_panel_rejected_count"], 1
        )
        self.assertFalse(contract["model_token_ids_registered"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(invalid), 2)
        self.assertTrue(all(row["ends_with_eos"] for row in rows))

    def test_cross_tokenizer_comparison_uses_semantic_events(self) -> None:
        audit = read_json("phase414_cross_tokenizer_semantic_alignment.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_cross_tokenizer_event_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(
            audit["comparison_unit"],
            "registered_semantic_event_not_token_id_sequence",
        )
        self.assertEqual(audit["tokenizer_count"], 2)
        self.assertEqual(audit["semantic_event_count"], 3)
        self.assertEqual(
            audit["cross_tokenizer_semantic_event_alignment_count"], 3
        )
        self.assertEqual(
            audit["cross_tokenizer_identical_token_id_sequence_count"], 0
        )
        self.assertEqual(len(rows), 6)

    def test_no_intermediate_observer_is_prequalified(self) -> None:
        contract = read_json("phase414_observer_qualification_contract.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_observer_method_rows.jsonl"
        )
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["observer_method_count"], 3)
        self.assertEqual(contract["diagnostic_observer_count"], 2)
        self.assertEqual(contract["learned_observer_pending_count"], 1)
        self.assertEqual(contract["qualified_observer_count"], 0)
        self.assertEqual(contract["required_split_count"], 8)
        self.assertEqual(contract["required_control_count"], 6)
        self.assertFalse(contract["low_calibration_error_proves_natural_model_use"])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row["qualified"] for row in rows))

    def test_ninety_six_items_are_mixed_evidence_not_progress_units(self) -> None:
        audit = read_json("phase414_catalog_qualification.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase414_catalog_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["catalog_item_count"], 96)
        self.assertEqual(audit["category_count"], 12)
        self.assertEqual(audit["items_per_category"], [8] * 12)
        self.assertEqual(len(audit["evidence_status_counts"]), 5)
        self.assertEqual(audit["strict_model_mechanism_closed_item_count"], 0)
        self.assertEqual(audit["global_progress_unit_count"], 0)
        self.assertFalse(audit["stable_homogeneous_knowledge_directory"])
        self.assertFalse(audit["catalog_as_completion_percentage_denominator_valid"])
        self.assertEqual(len(rows), 96)
        self.assertTrue(
            all(not row["strict_model_mechanism_closed"] for row in rows)
        )
        self.assertTrue(
            all(not row["counts_as_global_progress_unit"] for row in rows)
        )

    def test_stage_keeps_external_model_physical_causal_and_neuron_gates_closed(self) -> None:
        stage = read_json("phase414_stage_summary.json")
        qualification = read_json("phase414_execution_qualification.json")
        self.assertTrue(stage["assessment"]["machine_preflight_pass"])
        self.assertFalse(stage["assessment"]["cuda_execution_performed"])
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["results"]["completed_external_reviewer_count"], 0)
        self.assertEqual(
            stage["results"]["sealed_model_collector_equivalence_case_count"], 0
        )
        self.assertEqual(stage["results"]["qualified_observer_count"], 0)
        self.assertFalse(qualification["model_qualification_authorized"])
        self.assertFalse(stage["authorization"]["run_qwen3_model_qualification_next"])
        self.assertFalse(stage["authorization"]["run_descriptive_physical_mapping_next"])
        self.assertFalse(stage["authorization"]["run_causal_intervention_next"])
        self.assertFalse(stage["authorization"]["run_neuron_scan_next"])
        self.assertFalse(stage["next_stage"]["automatic_execution_now"])
        self.assertFalse(stage["single_global_progress_percentage_valid"])

    def test_atlas_mirrors_protocol_without_physical_promotion(self) -> None:
        stage = read_json("phase414_stage_summary.json")
        for root in (
            ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
            ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
        ):
            mirror = json.loads(
                (root / "phase414_stage_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            nodes = read_jsonl(root / "phase414_evidence_nodes.jsonl")
            self.assertEqual(mirror, stage)
            self.assertEqual(
                manifest["last_phase"],
                "Phase414-ObserverIndexedEventPreflightStage",
            )
            self.assertEqual(manifest["phase414"]["model_case_count"], 0)
            self.assertEqual(manifest["phase414"]["qualified_observer_count"], 0)
            self.assertFalse(progress["single_global_progress_percentage_valid"])
            self.assertTrue(all(not node["physical"] for node in nodes))
            self.assertTrue(all(not node["causal"] for node in nodes))
            self.assertTrue(all(not node["language_path"] for node in nodes))
        for root in (
            ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1",
            ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1",
        ):
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["phase"], 414)
            self.assertEqual(
                manifest["phase414_audit"]["new_neuron_path_nodes_promoted"], 0
            )


if __name__ == "__main__":
    unittest.main()
