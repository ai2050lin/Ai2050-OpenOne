from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch
from transformers.models.glm.configuration_glm import GlmConfig
from transformers.models.glm.modeling_glm import GlmMLP
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from phase365_dynamic_flow_instrumentation import (
    align_bundle_events, decompose_mlp_input, direct_mlp_output,
    relative_error, replay_mlp_from_neuron_writes, schema_payload,
    validate_blind_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def vector_ref(name: str) -> dict:
    return {"relative_path": f"private/{name}.pt", "sha256": "0" * 64, "dtype": "float16", "shape": [1, 1, 8], "slice": [0, 1]}


def event(event_id: str, event_type: str, layer: int, **extra) -> dict:
    return {
        "event_id": event_id, "event_type": event_type, "generation_time": 0,
        "layer_index": layer, "receiver_role": "query", "vector_ref": vector_ref(event_id),
        "raw_event_retained": True, **extra,
    }


def bundle(events: list[dict]) -> dict:
    return {
        "schema_version": "42.0.0", "bundle_id": "b0", "anonymous_case_id": "c0",
        "anonymous_model_id": "m0", "anonymous_group_id": "g0", "anonymous_condition_slot": "slot_0",
        "split": "blind_discovery", "events": events, "edges": [],
    }


class Phase365DynamicFlowInstrumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(365)
        self.hidden = torch.randn(2, 3, 8)

    def test_three_real_mlp_layouts_replay_from_neuron_writes(self) -> None:
        modules = {
            "qwen3": Qwen3MLP(Qwen3Config(hidden_size=8, intermediate_size=12, hidden_act="silu")),
            "glm4": GlmMLP(GlmConfig(hidden_size=8, intermediate_size=12, hidden_act="silu")),
            "deepseek7b": Qwen2MLP(Qwen2Config(hidden_size=8, intermediate_size=12, hidden_act="silu")),
        }
        for model, mlp in modules.items():
            with self.subTest(model=model):
                parts = decompose_mlp_input(model, mlp, self.hidden)
                direct = direct_mlp_output(parts)
                actual = mlp(self.hidden)
                replayed = replay_mlp_from_neuron_writes(parts, chunk_size=5)
                self.assertTrue(torch.allclose(direct, actual, atol=1e-6, rtol=1e-6))
                self.assertLess(relative_error(actual, replayed), 1e-6)

    def test_blind_schema_rejects_answer_and_semantic_leakage(self) -> None:
        clean = bundle([event("e0", "mlp_neuron_write", 1, channel_id=3)])
        self.assertEqual(validate_blind_bundle(clean), [])
        leaked = dict(clean)
        leaked["target"] = "Paris"
        self.assertTrue(any(error.startswith("forbidden_blind_keys") for error in validate_blind_bundle(leaked)))

    def test_graph_contrast_requires_typed_alignment(self) -> None:
        left = bundle([
            event("left_route", "attention_source_write", 1, source_role="source", head_index=0),
            event("left_write", "mlp_neuron_write", 2, channel_id=4),
        ])
        right = bundle([
            event("right_route", "attention_source_write", 1, source_role="source", head_index=0),
            event("right_only", "mlp_neuron_write", 3, channel_id=7),
        ])
        aligned = align_bundle_events(left, right)
        self.assertEqual(len(aligned), 3)
        self.assertEqual(sum(row["vector_subtraction_authorized"] for row in aligned), 1)
        self.assertEqual(sum(row["left_present"] != row["right_present"] for row in aligned), 2)

    def test_schema_freezes_raw_retention_and_repeat_noise_gate(self) -> None:
        payload = schema_payload()
        self.assertTrue(payload["public_backbone"]["raw_events_must_be_retained"])
        self.assertFalse(payload["public_backbone"]["residual_view_may_replace_raw"])
        self.assertTrue(payload["thresholds"]["repeat_noise_floor_required"])
        self.assertFalse(payload["thresholds"]["multiscale_persistence_is_independent_replication"])
        self.assertFalse(payload["blind_discovery"]["target_specific_competition_in_event"])

    def test_readiness_only_authorizes_six_repeat_noise_runs(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/schema_and_adapter_gate/phase365_instrumentation_readiness_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["results"]["synthetic_adapter_gate_model_count"], 3)
        self.assertEqual(summary["results"]["real_checkpoint_down_reference_gate_model_count"], 3)
        self.assertFalse(summary["authorization"]["full_96_case_engineering_run_authorized"])
        self.assertTrue(summary["authorization"]["six_run_repeat_noise_format_gate_authorized"])
        self.assertEqual(summary["authorization"]["repeat_noise_format_gate"]["total_forward_run_count"], 6)

    def test_real_repeat_noise_gate_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/repeat_noise_format_gate/phase365_repeat_noise_summary.json"
        if not path.exists():
            self.skipTest("Sequential real-model format gate has not run yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["model_order"], ["qwen3", "glm4", "deepseek7b"])
        self.assertEqual(summary["denominator"]["total_forward_run_count"], 6)
        self.assertEqual(summary["results"]["valid_model_count"], 3)
        self.assertEqual(summary["results"]["repeat_exact_model_count"], 3)
        self.assertTrue(summary["authorization"]["phase365_96_case_engineering_run_authorized"])
        self.assertFalse(summary["authorization"]["physical_confirmation_authorized"])

    def test_engineering_collection_freeze_when_available(self) -> None:
        root = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection_freeze"
        path = root / "phase365_collection_freeze_summary.json"
        if not path.exists():
            self.skipTest("Engineering collection has not been frozen yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["selected_case_count"], 96)
        self.assertEqual(summary["denominator"]["selected_group_count"], 24)
        self.assertEqual(summary["denominator"]["physical_confirmation_overlap_count"], 0)
        self.assertTrue(summary["storage"]["fits_with_reserve"])
        self.assertTrue(summary["quality"]["single_neuron_writes_offline_recoverable"])
        self.assertFalse(summary["quality"]["explicit_all_neuron_write_tensor_saved"])
        self.assertFalse(summary["authorization"]["physical_confirmation_authorized"])
        schema = json.loads((root / "phase365_dynamic_collection_schema.json").read_text(encoding="utf-8"))
        self.assertFalse(schema["scope"]["scope_is_all_token_positions"])
        self.assertTrue(schema["threshold_policy"]["template_and_condition_floor_still_missing"])

    def test_engineering_collection_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection/phase365_engineering_collection_summary.json"
        if not path.exists():
            self.skipTest("Engineering collection has not run yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["case_count"], 96)
        self.assertEqual(summary["results"]["valid_model_count"], 3)
        self.assertTrue(summary["results"]["single_neuron_writes_offline_recoverable"])
        self.assertFalse(summary["results"]["dynamic_bundle_extraction_executed"])
        self.assertFalse(summary["authorization"]["physical_confirmation_authorized"])

    def test_dynamic_bundles_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/dynamic_bundle_extraction/phase365_dynamic_bundle_summary.json"
        if not path.exists():
            self.skipTest("Dynamic bundles have not been extracted yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["bundle_count"], 288)
        self.assertEqual(summary["results"]["valid_bundle_count"], 288)
        self.assertTrue(summary["results"]["receiver_source_aliasing_explicit"])
        self.assertTrue(summary["results"]["other_source_bucket_retained"])
        self.assertEqual(summary["results"]["path_candidate_count"], 0)
        self.assertFalse(summary["claim_boundary"]["all_token_position_bundle_complete"])
        self.assertFalse(summary["authorization"]["target_or_condition_label_reveal_authorized"])

    def test_phase366_protocol_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/blind_motif_protocol/phase366_blind_motif_protocol.json"
        if not path.exists():
            self.skipTest("Blind motif protocol has not been frozen yet")
        protocol = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["denominator"]["case_count"], 288)
        self.assertFalse(protocol["condition_handling"]["average_four_conditions_before_discovery"])
        self.assertFalse(protocol["condition_handling"]["direct_graph_subtraction"])
        self.assertFalse(protocol["motif_enumeration"]["fixed_mad_only_threshold"])
        self.assertFalse(protocol["motif_enumeration"]["path_length_persistence_counts_as_independent_replication"])
        self.assertTrue(protocol["threshold_custodian"]["same_condition_template_floor_required"])
        self.assertFalse(protocol["threshold_custodian"]["condition_effect_may_be_counted_as_noise"])
        self.assertFalse(protocol["authorization"]["scientific_motif_scoring_now"])

    def test_phase366_full_collection_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/engineering_collection/phase366_full_collection_summary.json"
        if not path.exists():
            self.skipTest("The remaining Phase366 cases have not run yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["case_count"], 288)
        self.assertEqual(summary["results"]["valid_model_count"], 3)
        self.assertFalse(summary["quality"]["physical_confirmation_opened"])

    def test_phase366_bundle_split_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/dynamic_bundle_extraction/phase366_bundle_split_summary.json"
        if not path.exists():
            self.skipTest("Blind bundle split has not been frozen yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["bundle_count"], 288)
        self.assertEqual(summary["denominator"]["independent_group_count"], 72)
        self.assertEqual(summary["denominator"]["blind_discovery_group_count"], 48)
        self.assertEqual(summary["denominator"]["blind_calibration_group_count"], 24)
        self.assertEqual(summary["results"]["valid_bundle_count"], 288)
        self.assertEqual(summary["results"]["physical_confirmation_overlap_count"], 0)
        self.assertTrue(summary["results"]["all_groups_have_four_condition_slots"])
        self.assertFalse(summary["authorization"]["semantic_label_reveal_authorized"])

    def test_phase366_descriptors_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/label_blind_flow_descriptors/phase366_descriptor_summary.json"
        if not path.exists():
            self.skipTest("Label-blind directed descriptors have not been extracted yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["bundle_count"], 288)
        self.assertGreater(summary["denominator"]["directed_path_descriptor_count"], 0)
        self.assertTrue(summary["results"]["all_source_routes_retained"])
        self.assertFalse(summary["results"]["top_k_selection_used"])
        self.assertFalse(summary["results"]["condition_average_used"])
        self.assertFalse(summary["results"]["semantic_or_target_label_used"])
        self.assertEqual(summary["results"]["path_candidate_count"], 0)
        self.assertFalse(summary["authorization"]["blind_motif_scoring_authorized"])

    def test_phase366_threshold_custodian_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/blind_threshold_custodian/phase366_threshold_custodian_summary.json"
        if not path.exists():
            self.skipTest("Blind descriptor floors have not been frozen yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["discovery_group_count"], 48)
        self.assertEqual(summary["denominator"]["same_operation_case_pair_count"], 96)
        self.assertGreater(summary["denominator"]["paired_route_observation_count"], 0)
        self.assertFalse(summary["results"]["calibration_rows_used_for_thresholds"])
        self.assertFalse(summary["results"]["condition_effect_counted_as_noise"])
        self.assertTrue(summary["results"]["same_operation_cross_lexical_template_floor_used"])
        self.assertTrue(summary["results"]["thresholds_frozen_before_scoring"])
        self.assertFalse(summary["authorization"]["blind_calibration_scoring_authorized_before_candidate_freeze"])

    def test_phase367_blind_discovery_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/blind_motif_discovery/phase367_blind_motif_discovery_summary.json"
        if not path.exists():
            self.skipTest("Blind motif discovery has not run yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["discovery_case_count"], 192)
        self.assertEqual(summary["denominator"]["discovery_independent_group_count"], 48)
        self.assertEqual(summary["denominator"]["path_lengths"], [2, 3, 4, 6, 8])
        self.assertTrue(summary["frozen_gates"]["independent_group_is_analysis_unit"])
        self.assertFalse(summary["results"]["calibration_rows_used"])
        self.assertFalse(summary["results"]["semantic_or_target_labels_used"])
        self.assertFalse(summary["results"]["condition_average_used"])
        self.assertFalse(summary["results"]["top_k_selection_used"])
        self.assertFalse(summary["authorization"]["semantic_label_reveal_authorized"])
        self.assertFalse(summary["authorization"]["physical_confirmation_authorized"])

    def test_phase368_blind_calibration_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/blind_motif_calibration/phase368_blind_motif_calibration_summary.json"
        if not path.exists():
            self.skipTest("Frozen motifs have not run on blind calibration groups yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["calibration_case_count"], 96)
        self.assertEqual(summary["denominator"]["calibration_independent_group_count"], 24)
        self.assertFalse(summary["frozen_gates"]["candidate_shape_refit"])
        self.assertFalse(summary["frozen_gates"]["threshold_refit"])
        self.assertFalse(summary["frozen_gates"]["prediction_refit"])
        self.assertFalse(summary["results"]["semantic_or_target_labels_used"])
        self.assertFalse(summary["results"]["condition_average_used"])
        self.assertFalse(summary["results"]["physical_confirmation_opened"])
        self.assertFalse(summary["authorization"]["physical_confirmation_authorized"])

    def test_phase366_derived_integrity_when_available(self) -> None:
        path = ROOT / "tests/gpt5/result/phase365_dynamic_flow_instrumentation/dynamic_bundle_extraction/phase366_derived_artifact_integrity_summary.json"
        if not path.exists():
            self.skipTest("Derived dynamic bundle artifacts have not been audited yet")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["denominator"]["bundle_count"], 288)
        self.assertEqual(summary["denominator"]["unique_derived_file_reference_count"], 29952)
        self.assertEqual(summary["results"]["bad_file_count"], 0)
        self.assertTrue(summary["results"]["all_referenced_derived_files_valid"])


if __name__ == "__main__":
    unittest.main()
