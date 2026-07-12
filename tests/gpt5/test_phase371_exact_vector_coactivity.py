from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Phase371ExactVectorTests(unittest.TestCase):
    def test_phase371a_identifies_exact_tree_and_missing_state(self) -> None:
        summary = read_json(
            PHASE371 / "engineering_feasibility/phase371a_existing_ledger_tree_feasibility_summary.json"
        )
        self.assertTrue(summary["results"]["role_limited_exact_tree_numeric_gate_pass"])
        self.assertFalse(summary["results"]["all_token_receiver_states_available"])
        self.assertFalse(summary["results"]["query_key_states_available"])
        self.assertEqual(summary["denominator"]["audit_row_count"], 27)

    def test_sufficient_state_repair_is_lossless_and_within_budget(self) -> None:
        initial = read_json(PHASE371 / "phase371b_engineering_summary.json")
        repaired = read_json(PHASE371 / "phase371b_sufficient_state_summary.json")
        self.assertFalse(initial["results"]["materialized_derivative_storage_gate_pass"])
        self.assertTrue(repaired["results"]["all_on_demand_replay_gates_pass"])
        self.assertTrue(repaired["results"]["actual_compact_storage_projection_pass"])
        self.assertLess(repaired["storage"]["projected_discovery_bytes"], repaired["storage"]["budget_bytes"])

    def test_behavior_and_internal_denominators_are_frozen(self) -> None:
        behavior = read_json(PHASE371 / "phase371c_behavior_analysis/phase371c_behavior_analysis_summary.json")
        ledger = read_json(PHASE371 / "phase371c_internal_collection_audit.json")
        self.assertEqual(behavior["behavior"]["nonphysical_case_count"], 864)
        self.assertEqual(behavior["results"]["eligible_mechanisms"], ["entity_recency", "relation_binding"])
        self.assertFalse(behavior["results"]["four_mechanism_behavior_gate_pass"])
        self.assertTrue(ledger["valid"])
        self.assertEqual(ledger["denominator"]["case_count"], 264)
        self.assertEqual(ledger["denominator"]["file_count"], 3168)

    def test_adjacent_continuity_and_lazy_path_counts(self) -> None:
        adjacent = read_json(PHASE371 / "phase371c_adjacent_extension_audit.json")
        paths = read_json(PHASE371 / "phase371c_lazy_exact_paths/phase371c_lazy_exact_path_summary.json")
        self.assertTrue(adjacent["valid"])
        self.assertEqual(adjacent["denominator"]["continuity_row_count"], 2376)
        self.assertTrue(all(row["max_layer_continuity_relative_error"] == 0.0 for row in adjacent["models"]))
        self.assertTrue(paths["valid"])
        self.assertEqual(paths["denominator"]["explicit_node_count"], 38808)
        self.assertEqual(paths["results"]["candidate_language_path_count"], 0)

    def test_blind_rows_are_complete_before_mapping(self) -> None:
        audit = read_json(
            PHASE371 / "phase371c_blind_vector_contrast/phase371c_blind_contrast_audit.json"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["denominator"]["route_row_count"], 299376)
        self.assertEqual(audit["denominator"]["vocab_row_count"], 1188)
        self.assertEqual(audit["quality"]["candidate_selected_count"], 0)
        self.assertTrue(audit["quality"]["all_indices_finite"])

    def test_history_gate_stops_before_calibration(self) -> None:
        mapping = read_json(
            PHASE371 / "phase371c_discovery_mapping/phase371c_discovery_mapping_summary.json"
        )
        history = read_json(
            PHASE371 / "phase371c_exact_history_residual/phase371c_exact_history_residual_summary.json"
        )
        self.assertEqual(mapping["results"]["provisional_heterogeneous_level2_count"], 39)
        self.assertEqual(mapping["results"]["provisional_level3_count"], 3)
        self.assertEqual(history["results"]["history_heterogeneous_level2_count"], 0)
        self.assertEqual(history["results"]["history_level3_count"], 0)
        self.assertFalse(history["results"]["calibration_authorized"])
        self.assertEqual(history["results"]["full_candidate_language_path_count"], 0)

    def test_pattern_atlas_and_client_are_mirrored(self) -> None:
        files = [
            "phase371_exact_vector_stage_summary.json",
            "phase371_measured_layer_pairs.jsonl",
            "phase371_evidence_nodes.jsonl",
            "phase371_evidence_edges.jsonl",
            "manifest.json",
            "progress.json",
        ]
        for name in files:
            self.assertEqual((ATLAS / name).read_bytes(), (CLIENT / name).read_bytes(), name)
        stage = read_json(ATLAS / "phase371_exact_vector_stage_summary.json")
        progress = read_json(ATLAS / "progress.json")
        self.assertFalse(stage["authorization"]["show_phase371_objects_as_language_family_paths"])
        self.assertFalse(stage["single_global_progress_percentage_valid"])
        self.assertEqual(progress["last_phase"], "Phase378-PhysicalMerge")
        self.assertIn("phase371_decision", progress)
        self.assertFalse(progress["single_global_progress_percentage_valid"])

    def test_neuron_atlas_mirror_and_checksums(self) -> None:
        for name in (
            "phase371_exact_vector_stage_summary.json",
            "phase371_evidence_nodes.jsonl",
            "phase371_evidence_edges.jsonl",
            "manifest.json",
        ):
            self.assertEqual((NEURON / name).read_bytes(), (NEURON_CLIENT / name).read_bytes(), name)
        for root in (NEURON, NEURON_CLIENT):
            checksum = read_json(root / "checksums.json")
            entries = {row["path"]: row["sha256"] for row in checksum["files"]}
            for name in (
                "phase371_exact_vector_stage_summary.json",
                "phase371_evidence_nodes.jsonl",
                "phase371_evidence_edges.jsonl",
                "manifest.json",
            ):
                self.assertEqual(entries[name], sha256(root / name))
            manifest = read_json(root / "manifest.json")
            self.assertEqual(manifest["phase371_audit"]["new_neuron_path_nodes_promoted"], 0)
            self.assertEqual(
                manifest["evidence_boundary"]["latest_phase"],
                "Phase378-PhysicalMerge",
            )
            self.assertEqual(manifest["phase378_audit"]["new_neuron_path_nodes_promoted"], 0)


if __name__ == "__main__":
    unittest.main()
