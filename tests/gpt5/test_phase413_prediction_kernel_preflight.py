#!/usr/bin/env python3
"""Contract tests for the Phase413 prediction-kernel preflight."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase413_prediction_kernel_preflight import OUT  # noqa: E402


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase413PredictionKernelPreflightTest(unittest.TestCase):
    def test_candidate_panel_is_disjoint_equal_horizon_and_multiaxis(self) -> None:
        contract = read_json("phase413_candidate_panel_contract.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase413_candidate_panel_rows.jsonl"
        )
        self.assertTrue(contract["valid"])
        self.assertEqual(contract["candidate_count"], 4)
        self.assertEqual(contract["horizon_tokens"], 2)
        self.assertEqual(contract["axis_count"], 5)
        self.assertTrue(contract["events_pairwise_disjoint"])
        self.assertTrue(contract["all_candidates_equal_horizon"])
        self.assertTrue(
            contract["axes_are_independent_labels_not_exclusive_buckets"]
        )
        self.assertFalse(contract["model_token_ids_registered"])
        self.assertFalse(contract["real_model_panel_exhaustive"])
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({tuple(row["abstract_tokens"]) for row in rows}), 4)
        self.assertEqual(sum(row["fully_valid"] for row in rows), 1)

    def test_terminal_distribution_does_not_identify_internal_trajectory(self) -> None:
        audit = read_json("phase413_terminal_nonidentifiability_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase413_trajectory_rows.jsonl"
        )
        steps = read_jsonl(
            OUT / "protocol/private/phase413_trajectory_step_rows.jsonl"
        )
        path_rows = [row for row in rows if "path_id" in row and "event_count" in row]
        pair_rows = [row for row in rows if "left_path_id" in row]
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["synthetic_path_count"], 4)
        self.assertEqual(audit["same_terminal_distribution_path_count"], 4)
        self.assertEqual(audit["path_pair_count"], 6)
        self.assertEqual(audit["same_endpoint_different_internal_pair_count"], 6)
        self.assertEqual(audit["valid_mass_nonmonotonic_path_count"], 2)
        self.assertGreaterEqual(audit["entropy_expansion_path_count"], 1)
        self.assertFalse(audit["terminal_distribution_identifies_internal_trajectory"])
        self.assertEqual(len(path_rows), 4)
        self.assertEqual(len(pair_rows), 6)
        self.assertEqual(len(steps), 16)
        self.assertTrue(all(abs(row["distribution_sum"] - 1.0) < 1e-12 for row in steps))

    def test_one_step_equality_is_not_full_future_equivalence(self) -> None:
        audit = read_json("phase413_future_equivalence_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase413_future_equivalence_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["state_pair_count"], 1)
        self.assertEqual(audit["one_step_equal_state_pair_count"], 1)
        self.assertEqual(audit["horizon_two_equal_state_pair_count"], 0)
        self.assertEqual(audit["one_step_equal_but_future_different_pair_count"], 1)
        self.assertGreater(audit["horizon_two_js_divergence"], 0)
        self.assertFalse(audit["one_step_equality_implies_full_future_equivalence"])
        self.assertEqual(len(rows), 2)

    def test_channel_permutation_preserves_output_not_fixed_coordinates(self) -> None:
        audit = read_json("phase413_channel_permutation_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase413_channel_permutation_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["case_count"], 3)
        self.assertEqual(audit["native_output_invariant_case_count"], 3)
        self.assertEqual(audit["fixed_coordinate_probe_invariant_case_count"], 0)
        self.assertEqual(audit["fixed_coordinate_probe_failure_count"], 3)
        self.assertEqual(audit["transported_probe_invariant_case_count"], 3)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["native_output_invariant"] for row in rows))
        self.assertTrue(all(not row["fixed_coordinate_probe_invariant"] for row in rows))

    def test_no_direct_intermediate_probability_readout_is_prequalified(self) -> None:
        qualification = read_json("phase413_readout_qualification.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase413_readout_registry.jsonl"
        )
        self.assertTrue(qualification["valid"])
        self.assertEqual(qualification["readout_mode_count"], 7)
        self.assertEqual(qualification["native_terminal_method_count"], 2)
        self.assertEqual(qualification["direct_layer_local_readout_mode_count"], 3)
        self.assertEqual(
            qualification["qualified_direct_layer_local_probability_readout_count"],
            0,
        )
        self.assertEqual(
            qualification["diagnostic_direct_layer_local_readout_count"], 2
        )
        self.assertEqual(qualification["model_executed_readout_count"], 0)
        self.assertFalse(qualification["intermediate_mu_te_native_without_decoder"])
        self.assertEqual(len(rows), 7)

    def test_source_claims_are_split_instead_of_wholesale_accepted(self) -> None:
        audit = read_json("phase413_source_claim_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase413_source_claim_rows.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["claim_count"], 18)
        self.assertEqual(audit["status_counts"]["supported_by_current_evidence"], 6)
        self.assertEqual(audit["status_counts"]["methodologically_valid_proposal"], 3)
        self.assertEqual(audit["status_counts"]["requires_qualification"], 3)
        self.assertEqual(audit["status_counts"]["incorrect_as_stated"], 6)
        self.assertEqual(len(rows), 18)

    def test_stage_keeps_model_physical_causal_and_neuron_gates_closed(self) -> None:
        stage = read_json("phase413_stage_summary.json")
        qualification = read_json("phase413_execution_qualification.json")
        self.assertTrue(stage["assessment"]["machine_preflight_pass"])
        self.assertFalse(stage["assessment"]["cuda_execution_performed"])
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["denominators"]["physical_case_count_consumed"], 0)
        self.assertEqual(stage["results"]["completed_external_reviewer_count"], 0)
        self.assertEqual(
            stage["results"]["sealed_model_collector_equivalence_case_count"], 0
        )
        self.assertFalse(qualification["model_qualification_authorized"])
        self.assertFalse(stage["authorization"]["run_qwen3_model_qualification_next"])
        self.assertFalse(stage["authorization"]["run_descriptive_physical_mapping_next"])
        self.assertFalse(stage["authorization"]["run_causal_intervention_next"])
        self.assertFalse(stage["authorization"]["run_neuron_scan_next"])
        self.assertFalse(stage["next_stage"]["automatic_execution_now"])
        self.assertFalse(stage["single_global_progress_percentage_valid"])

    def test_atlas_mirrors_measurement_protocol_without_physical_promotion(self) -> None:
        stage = read_json("phase413_stage_summary.json")
        for root in (
            ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
            ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
        ):
            mirror = json.loads(
                (root / "phase413_stage_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            nodes = read_jsonl(root / "phase413_evidence_nodes.jsonl")
            self.assertEqual(mirror, stage)
            self.assertIn(
                manifest["last_phase"],
                {
                    "Phase413-PredictionKernelMeasurementPreflightStage",
                    "Phase414-ObserverIndexedEventPreflightStage",
                },
            )
            self.assertEqual(manifest["phase413"]["model_case_count"], 0)
            self.assertEqual(
                manifest["phase413"][
                    "qualified_direct_layer_local_probability_readout_count"
                ],
                0,
            )
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
            self.assertGreaterEqual(manifest["phase"], 413)
            self.assertEqual(
                manifest["phase413_audit"]["new_neuron_path_nodes_promoted"], 0
            )


if __name__ == "__main__":
    unittest.main()
