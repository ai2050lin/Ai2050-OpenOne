from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P383 = ROOT / "tests/gpt5/result/phase383_exact_component_event_map"
P384 = ROOT / "tests/gpt5/result/phase384_exact_subunit_mass_map"
P385 = ROOT / "tests/gpt5/result/phase385_opposing_mass_specificity"
ATLAS = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
CLIENT = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
NEURON = ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1"
NEURON_CLIENT = ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase383385ExactEventMapTests(unittest.TestCase):
    def test_single_sample_runtime_requalifies_the_denominator(self) -> None:
        protocol = read_json(P383 / "phase383_protocol.json")
        source = protocol["source_denominator"]
        self.assertFalse(source["source_batch_contract_reused"])
        self.assertTrue(source["single_sample_requalification_completed"])
        self.assertEqual(
            source["single_path_qualified_case_count_by_model"],
            {"qwen3": 260, "glm4": 260, "deepseek7b": 239},
        )
        self.assertEqual(
            source["single_path_qualified_groups_by_mechanism"],
            {
                "relation_binding": 13,
                "entity_recency": 18,
                "number_agreement": 7,
                "target_vs_wrong": 11,
            },
        )
        self.assertEqual(
            protocol["frozen_denominator"]["balanced_group_count_per_mechanism"], 7
        )
        self.assertEqual(protocol["frozen_denominator"]["case_count"], 336)
        self.assertEqual(
            protocol["instrument_contract"]["runtime_dtype_by_model"],
            {"qwen3": "float16", "glm4": "float16", "deepseek7b": "bfloat16"},
        )

    def test_exact_component_instrument_passes_all_three_models(self) -> None:
        audit = read_json(P383 / "phase383_instrument_audit_summary.json")
        self.assertEqual(audit["denominator"]["case_count"], 48)
        self.assertEqual(audit["results"]["baseline_replay_match_count"], 48)
        self.assertTrue(audit["results"]["all_three_model_instruments_valid"])
        self.assertTrue(
            audit["results"]["attention_head_source_events_exactly_replayable"]
        )
        self.assertTrue(audit["results"]["mlp_channel_events_exactly_replayable"])
        self.assertFalse(audit["results"]["top_k_used"])
        self.assertTrue(
            all(max(row["gate_maxima"].values()) <= 0.01 for row in audit["models"])
        )

    def test_signed_parent_event_map_is_late_and_not_upstream(self) -> None:
        discovery = read_json(P383 / "phase383_signed_event_map_summary.json")
        calibration = read_json(P383 / "phase383_calibration_summary.json")
        self.assertEqual(discovery["denominator"]["event_row_count"], 149760)
        self.assertEqual(discovery["results"]["heterogeneous_level2_candidate_count"], 32)
        self.assertEqual(discovery["results"]["upstream_level2_candidate_count"], 0)
        self.assertEqual(discovery["results"]["terminal_interface_level2_candidate_count"], 23)
        self.assertEqual(calibration["denominator"]["frozen_candidate_count"], 32)
        self.assertEqual(calibration["results"]["calibration_level2_replication_count"], 24)
        self.assertEqual(calibration["results"]["upstream_level2_replication_count"], 0)
        self.assertEqual(calibration["results"]["attention_source_write_replication_count"], 0)
        self.assertFalse(calibration["authorization"]["physical_holdout_collection"])
        self.assertFalse(calibration["results"]["language_path_discovered"])

    def test_all_subunit_mass_is_conserved_without_top_k(self) -> None:
        discovery = read_json(P384 / "phase384_discovery_summary.json")
        calibration = read_json(P384 / "phase384_calibration_summary.json")
        self.assertEqual(
            discovery["denominator"]["exact_attention_head_event_count"], 2315520
        )
        self.assertEqual(
            discovery["denominator"]["exact_mlp_channel_event_count"], 205701120
        )
        self.assertTrue(discovery["results"]["parent_projection_conservation_pass"])
        self.assertEqual(discovery["results"]["coherent_pattern_count"], 0)
        self.assertEqual(discovery["results"]["upstream_opposing_pattern_count"], 2)
        self.assertTrue(calibration["results"]["parent_projection_conservation_pass"])
        self.assertEqual(calibration["results"]["upstream_level2_replication_count"], 2)
        self.assertEqual(calibration["results"]["upstream_coherent_replication_count"], 0)
        self.assertEqual(calibration["results"]["upstream_opposing_replication_count"], 2)
        self.assertFalse(calibration["results"]["top_k_used"])
        self.assertFalse(calibration["authorization"]["physical_holdout"])

    def test_matched_controls_reject_function_specificity(self) -> None:
        summary = read_json(P385 / "phase385_specificity_summary.json")
        self.assertEqual(
            summary["denominator"]["replicated_upstream_opposing_candidate_count"], 2
        )
        self.assertEqual(
            summary["results"]["replicated_function_specific_candidate_count"], 0
        )
        self.assertEqual(summary["results"]["failed_control_counts"]["wrong_depth"], 12)
        self.assertFalse(
            summary["results"]["opposing_mass_function_specificity_established"]
        )
        self.assertFalse(summary["authorization"]["physical_holdout"])
        self.assertFalse(summary["authorization"]["causal_intervention"])

    def test_atlas_and_client_publish_identical_strict_boundaries(self) -> None:
        names = (
            "manifest.json",
            "progress.json",
            "client_index.json",
            "phase385_exact_event_stage_summary.json",
            "phase385_evidence_nodes.jsonl",
            "phase385_evidence_edges.jsonl",
            "phase385_specificity_summary.json",
            "phase384_calibration_summary.json",
            "phase383_calibration_summary.json",
        )
        for name in names:
            self.assertEqual((ATLAS / name).read_bytes(), (CLIENT / name).read_bytes(), name)
        manifest = read_json(ATLAS / "manifest.json")
        progress = read_json(ATLAS / "progress.json")
        latest_phase = int(manifest["last_phase"].split("Phase", 1)[1].split("-", 1)[0])
        self.assertGreaterEqual(latest_phase, 389)
        self.assertEqual(manifest["phase383_385"]["language_path_count"], 0)
        self.assertEqual(
            progress["exact_event_stage"]["audited_families"],
            {"numerator": 4, "denominator": 9},
        )
        self.assertEqual(
            progress["exact_event_stage"]["audited_registered_mechanisms"],
            {"numerator": 4, "denominator": 72},
        )
        self.assertEqual(
            progress["exact_event_stage"]["function_specific_upstream_patterns"][
                "numerator"
            ],
            0,
        )
        self.assertFalse(progress["single_global_progress_percentage_valid"])
        for root in (NEURON, NEURON_CLIENT):
            neuron = read_json(root / "manifest.json")
            self.assertGreaterEqual(neuron["phase"], 389)
            self.assertEqual(
                neuron["phase383_385_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertFalse(
                neuron["evidence_boundary"][
                    "function_specific_upstream_subunit_pattern_available"
                ]
            )
            self.assertEqual(
                neuron["evidence_boundary"]["exact_event_audited_families"],
                [
                    "content_knowledge",
                    "readout_competition",
                    "state_drift",
                    "syntax_structure",
                ],
            )

    def test_client_canvas_and_strict_dashboard_are_nonblank_and_reachable(self) -> None:
        validation = P385 / "client_validation"
        for name in (
            "phase385_desktop_compositor_check.json",
            "phase385_mobile_compositor_check.json",
        ):
            payload = read_json(validation / name)
            self.assertTrue(payload["compositorPixels"]["nonBlank"], name)
            self.assertEqual(payload["after"]["panelOverlaps"], [], name)
        for name in (
            "phase385_atlas_dashboard_check.json",
            "phase385_atlas_dashboard_mobile_check.json",
        ):
            payload = read_json(validation / name)
            self.assertTrue(payload["after"]["bodyTextHasStrictPathCount"], name)
            self.assertTrue(payload["after"]["bodyTextHasExactEventScope"], name)


if __name__ == "__main__":
    unittest.main()
