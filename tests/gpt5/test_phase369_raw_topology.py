from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUND = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_topology_preregister"


class Phase369RawTopologyTests(unittest.TestCase):
    def test_protocol_prevents_label_and_mapping_leakage(self) -> None:
        protocol = json.loads((ROUND / "phase369_protocol.json").read_text(encoding="utf-8"))
        self.assertFalse(protocol["evidence_denominators"]["single_global_progress_percentage_valid"])
        self.assertIn("target_rank", protocol["blind_event_contract"]["forbidden"])
        self.assertFalse(protocol["raw_relation_signature"]["unrestricted_learned_cross_model_rotation_allowed"])
        self.assertFalse(protocol["topology_gate_vector"]["weighted_scalar_distance_used"])
        self.assertEqual(protocol["cross_model_evidence_levels"]["unified_theory_entry_minimum"], "level_3")
        self.assertFalse(protocol["authorization"]["physical_holdout_execution"])
        self.assertEqual(
            protocol["dataset_contract"]["minimum_cross_model_qualified_discovery_groups_per_mechanism"],
            4,
        )
        self.assertTrue(
            protocol["dataset_contract"]["qualification_requires_all_four_conditions_and_all_three_models"]
        )

    def test_fresh_case_bank_denominator_and_seals(self) -> None:
        summary = json.loads((ROUND / "phase369_case_bank_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["case_count"], 576)
        self.assertEqual(summary["denominator"]["fresh_discovery_case_count"], 288)
        self.assertEqual(summary["denominator"]["fresh_calibration_case_count"], 144)
        self.assertEqual(summary["denominator"]["physical_holdout_case_count"], 144)
        self.assertEqual(summary["quality"]["prior_prompt_overlap_count"], 0)
        self.assertEqual(summary["quality"]["unique_rendered_prompt_count"], 576)
        self.assertTrue(summary["quality"]["every_group_has_four_conditions"])
        self.assertFalse(summary["quality"]["phase368_calibration_reused"])
        self.assertFalse(summary["authorization"]["run_physical_holdout"])

    def test_blind_registry_contains_no_semantic_fields(self) -> None:
        rows = [json.loads(line) for line in (ROUND / "phase369_blind_case_registry.jsonl").read_text(encoding="utf-8").splitlines() if line]
        forbidden = {"family_id", "mechanism_id", "target", "target_aliases", "distractors", "contrast_condition"}
        self.assertEqual(len(rows), 576)
        self.assertTrue(all(not (set(row) & forbidden) for row in rows))
        self.assertEqual(len({row["anonymous_parallel_group_id"] for row in rows}), 48)

    def test_final_behavior_denominator_and_collection_seal(self) -> None:
        final_root = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/behavior_qualification_final_v2"
        summary = json.loads(
            (final_root / "phase369_behavior_qualification_final_v2_summary.json").read_text(encoding="utf-8")
        )
        self.assertTrue(summary["qualification"]["all_gates_passed"])
        self.assertEqual(summary["qualification"]["fresh_discovery_blind_case_count"], 336)
        self.assertEqual(summary["qualification"]["fresh_calibration_blind_case_count"], 180)
        self.assertFalse(summary["authorization"]["physical_holdout_execution"])
        collection = ROOT / "tests/gpt5/result/phase369_raw_topology_flow/raw_collection_freeze"
        if (collection / "phase369_raw_collection_freeze_summary.json").is_file():
            frozen = json.loads(
                (collection / "phase369_raw_collection_freeze_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frozen["case_count"], 336)
            self.assertEqual(frozen["physical_holdout_case_count"], 0)

    def test_raw_ledger_bundles_and_relations_are_complete(self) -> None:
        phase_root = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
        bundle = json.loads(
            (phase_root / "dynamic_bundle_extraction/phase365_dynamic_bundle_summary.json").read_text(encoding="utf-8")
        )
        relation = json.loads(
            (phase_root / "raw_relation_features/phase369_raw_relation_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(bundle["results"]["valid_bundle_count"], 336)
        self.assertEqual(bundle["denominator"]["event_count"], 1247344)
        self.assertEqual(relation["denominator"]["route_row_count"], 605696)
        self.assertTrue(relation["results"]["raw_and_low_features_share_identical_route_rows"])
        self.assertFalse(relation["results"]["target_rank_or_margin_used"])

    def test_strict_negative_keeps_calibration_and_physical_sealed(self) -> None:
        phase_root = ROOT / "tests/gpt5/result/phase369_raw_topology_flow"
        discovery = json.loads(
            (phase_root / "blind_future_and_crossmodel/phase369_blind_future_and_crossmodel_summary.json").read_text(encoding="utf-8")
        )
        diagnostic = json.loads(
            (phase_root / "head_neuron_topology_diagnostic_evaluation/phase369_head_neuron_diagnostic_evaluation_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(discovery["evidence"]["level_1_model_count"], 0)
        self.assertEqual(discovery["evidence"]["level_2_heterogeneous_pair_count"], 0)
        self.assertFalse(discovery["authorization"]["fresh_calibration_raw_collection"])
        self.assertFalse(discovery["authorization"]["physical_holdout"])
        self.assertFalse(diagnostic["authorization"]["new_independent_topology_cycle"])
        self.assertFalse(diagnostic["claim_boundary"]["physical_holdout_opened"])

    def test_atlas_and_client_share_phase370_boundary(self) -> None:
        atlas = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
        client = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
        for name in ("manifest.json", "progress.json", "phase369_raw_topology_stage_summary.json"):
            self.assertEqual(
                json.loads((atlas / name).read_text(encoding="utf-8")),
                json.loads((client / name).read_text(encoding="utf-8")),
            )
        stage = json.loads((atlas / "phase369_raw_topology_stage_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(stage["authorization"]["show_phase369_or_phase370_as_language_family_paths"])
        self.assertFalse(stage["single_global_progress_percentage_valid"])


if __name__ == "__main__":
    unittest.main()
