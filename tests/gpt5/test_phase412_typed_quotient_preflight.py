#!/usr/bin/env python3
"""Contract tests for the Phase412 typed finite-quotient preflight."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_protocol import STATE_IDS  # noqa: E402
from phase411_finite_operation_preflight import operation_registry  # noqa: E402
from phase412_typed_quotient_preflight import (  # noqa: E402
    OUT,
    all_partitions,
    observer_transform,
)


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase412TypedQuotientPreflightTest(unittest.TestCase):
    def test_partition_enumerator_exhausts_each_finite_universe(self) -> None:
        expected = {
            "knowledge_binding": 203,
            "rule_reasoning": 5,
            "grammar_constraint": 15,
        }
        for family, count in expected.items():
            partitions = all_partitions(tuple(STATE_IDS[family]))
            self.assertEqual(len(partitions), count)
            self.assertEqual(len(set(partitions)), count)

    def test_entity_relabeling_transports_the_query_role(self) -> None:
        operation = next(
            row
            for row in operation_registry()["knowledge_binding"]
            if tuple(row["entity_permutation"]) == (1, 0, 2)
            and tuple(row["value_permutation"]) == (0, 1, 2)
        )
        self.assertEqual(
            observer_transform(
                "knowledge_binding",
                "single_entity_value:entity_0",
                operation,
            ),
            "single_entity_value:entity_1",
        )
        self.assertEqual(
            observer_transform(
                "knowledge_binding",
                "entity_value_order:default",
                operation,
            ),
            "entity_value_order:default",
        )

    def test_typed_covariance_explains_all_fixed_observer_failures(self) -> None:
        audit = read_json("phase412_typed_observer_covariance_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase412_observer_operation_covariance.jsonl"
        )
        action = read_jsonl(
            OUT / "protocol/private/phase412_observer_action_composition.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_query_observer_operation_cell_count"], 210)
        self.assertEqual(audit["fixed_observer_unstable_cell_count"], 72)
        self.assertEqual(audit["role_moved_cell_count"], 72)
        self.assertEqual(
            audit["fixed_instability_explained_by_role_transport_count"], 72
        )
        self.assertEqual(audit["typed_observer_unstable_cell_count"], 0)
        self.assertEqual(audit["typed_response_class_map_failure_cell_count"], 0)
        self.assertEqual(audit["observer_action_composition_case_count"], 7984)
        self.assertEqual(audit["observer_action_composition_failure_count"], 0)
        self.assertEqual(len(rows), 256)
        self.assertEqual(len(action), 7984)

    def test_exhaustive_quotient_audit_separates_global_and_role_conditioned(self) -> None:
        audit = read_json("phase412_nontrivial_quotient_audit.json")
        partitions = read_jsonl(
            OUT / "protocol/private/phase412_partition_catalog.jsonl"
        )
        evaluations = read_jsonl(
            OUT
            / "protocol/private/phase412_partition_observer_evaluations.jsonl"
        )
        induced = read_jsonl(
            OUT / "protocol/private/phase412_induced_quotients_and_bundles.jsonl"
        )
        self.assertEqual(audit["partition_count"], 223)
        self.assertEqual(audit["nontrivial_partition_count"], 217)
        self.assertEqual(
            audit["full_operation_congruent_nontrivial_partition_count"], 4
        )
        self.assertEqual(
            audit["joint_observation_sufficient_nontrivial_partition_count"], 0
        )
        self.assertEqual(audit["global_nontrivial_qualifying_partition_count"], 0)
        self.assertEqual(
            audit["fixed_observer_nontrivial_qualifying_partition_count"], 0
        )
        self.assertEqual(
            audit["role_stabilizer_nontrivial_qualifying_partition_count"], 3
        )
        self.assertEqual(audit["external_role_conditioned_quotient_count"], 3)
        self.assertEqual(
            audit["external_role_indexed_partition_bundle_count"], 1
        )
        self.assertEqual(
            audit["model_derived_nontrivial_predictive_quotient_count"], 0
        )
        self.assertEqual(len(partitions), 223)
        self.assertEqual(len(evaluations), 1075)
        bundles = [row for row in induced if "bundle_id" in row]
        self.assertEqual(len(bundles), 1)
        self.assertFalse(bundles[0]["global_state_quotient"])
        self.assertFalse(bundles[0]["model_derived"])

    def test_irreversible_and_cross_family_operations_remain_unregistered(self) -> None:
        irreversible = read_json("phase412_irreversible_operation_readiness.json")
        composition = read_json("phase412_typed_composition_readiness.json")
        irreversible_rows = read_jsonl(
            OUT
            / "protocol/private/phase412_irreversible_operation_proposals.jsonl"
        )
        bridge_rows = read_jsonl(
            OUT / "protocol/private/phase412_cross_family_bridge_proposals.jsonl"
        )
        self.assertEqual(irreversible["proposed_irreversible_operation_count"], 7)
        self.assertEqual(
            irreversible["registered_executable_irreversible_operation_count"], 0
        )
        self.assertFalse(irreversible["registration_authorized"])
        self.assertEqual(len(irreversible_rows), 7)
        self.assertTrue(
            all(not row["closed_on_current_finite_universe"] for row in irreversible_rows)
        )
        self.assertEqual(composition["proposed_cross_family_bridge_count"], 4)
        self.assertEqual(
            composition["registered_executable_cross_family_bridge_count"], 0
        )
        self.assertEqual(composition["valid_cross_family_composition_count"], 0)
        self.assertFalse(composition["cross_family_operation_system_established"])
        self.assertEqual(len(bridge_rows), 4)

    def test_stage_keeps_model_physical_causal_and_neuron_gates_closed(self) -> None:
        stage = read_json("phase412_stage_summary.json")
        qualification = read_json("phase412_qualification.json")
        self.assertTrue(stage["assessment"]["machine_preflight_pass"])
        self.assertFalse(stage["assessment"]["cuda_execution_performed"])
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["denominators"]["physical_case_count_consumed"], 0)
        self.assertFalse(qualification["model_qualification_authorized"])
        self.assertFalse(stage["authorization"]["run_qwen3_model_qualification_next"])
        self.assertFalse(stage["authorization"]["run_descriptive_physical_mapping_next"])
        self.assertFalse(stage["authorization"]["run_causal_intervention_next"])
        self.assertFalse(stage["authorization"]["run_neuron_scan_next"])
        self.assertFalse(stage["next_stage"]["automatic_execution_now"])
        self.assertFalse(stage["single_global_progress_percentage_valid"])

    def test_atlas_mirrors_protocol_without_physical_promotion(self) -> None:
        stage = read_json("phase412_stage_summary.json")
        for root in (
            ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
            ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
        ):
            mirror = json.loads(
                (root / "phase412_stage_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            nodes = read_jsonl(root / "phase412_evidence_nodes.jsonl")
            self.assertEqual(mirror, stage)
            self.assertIn(
                manifest["last_phase"],
                {
                    "Phase412-TypedObserverQuotientPreflightStage",
                    "Phase413-PredictionKernelMeasurementPreflightStage",
                    "Phase414-ObserverIndexedEventPreflightStage",
                },
            )
            self.assertEqual(manifest["phase412"]["model_case_count"], 0)
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
            self.assertGreaterEqual(manifest["phase"], 412)
            self.assertEqual(
                manifest["phase412_audit"]["new_neuron_path_nodes_promoted"], 0
            )


if __name__ == "__main__":
    unittest.main()
