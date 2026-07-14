#!/usr/bin/env python3
"""Contract tests for the Phase410 model-free protocol preflight."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_protocol import package_for  # noqa: E402
from phase409_dynamic_response_protocol import (  # noqa: E402
    prior_state_for,
    response_contract,
)
from phase410_orthogonal_preflight import (  # noqa: E402
    OUT,
    exact_response_parse,
    orthogonal_prefix_state,
    scan_orthogonal_event_process,
)


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase410OrthogonalPreflightTest(unittest.TestCase):
    def grammar_contract(self) -> tuple[dict, str, str, str]:
        family = "grammar_constraint"
        package = package_for(family, 0, 0)
        current = "singular_present"
        prior = prior_state_for(family, current)
        contract = response_contract(family, package, "sentence_completion")
        return contract, package["modifier"], current, prior

    def test_exact_parser_rejects_interior_substring_and_extra_words(self) -> None:
        contract, modifier, current, prior = self.grammar_contract()
        valid = exact_response_parse(
            f"is {modifier}", contract, current, prior, "h0_current_only"
        )
        leading = exact_response_parse(
            f"answer is {modifier}", contract, current, prior, "h0_current_only"
        )
        trailing = exact_response_parse(
            f"is {modifier} extra", contract, current, prior, "h0_current_only"
        )
        self.assertEqual(valid["semantic_class"], "allowed_response")
        self.assertEqual(leading["semantic_class"], "no_registered_response")
        self.assertEqual(trailing["semantic_class"], "no_registered_response")

    def test_orthogonal_axes_can_coexist(self) -> None:
        contract, modifier, current, prior = self.grammar_contract()
        state = orthogonal_prefix_state(
            f"is {modifier}.",
            contract,
            current,
            prior,
            "h0_current_only",
            numeric_validity="finite",
            stopped=True,
        )
        self.assertEqual(state["semantic_class"], "allowed_response")
        self.assertEqual(state["format_class"], "complete")
        self.assertTrue(state["boundary_reached"])
        self.assertTrue(state["model_stopped"])
        self.assertEqual(state["numeric_validity"], "finite")
        self.assertEqual(state["response_role"], "current")

    def test_axis_transitions_are_independent_and_revision_is_preserved(self) -> None:
        contract, modifier, current, prior = self.grammar_contract()
        process = scan_orthogonal_event_process(
            ["is", f"is {modifier}", f"is {modifier}.", f"is {modifier}. extra"],
            contract,
            current,
            prior,
            "h0_current_only",
            numeric_validity_by_step=["finite"] * 4,
            stopped=True,
        )
        self.assertIn(2, process["coincident_transition_steps"])
        self.assertEqual(
            process["step_states"][1]["state"]["semantic_class"],
            "allowed_response",
        )
        self.assertTrue(process["step_states"][2]["state"]["boundary_reached"])
        self.assertEqual(
            process["step_states"][3]["state"]["semantic_class"],
            "no_registered_response",
        )
        self.assertTrue(process["final_state"]["model_stopped"])

    def test_h3_contract_is_mirrored_for_every_registered_order_pair(self) -> None:
        audit = read_json("phase410_h3_order_symmetry_audit.json")
        rows = read_jsonl(OUT / "protocol/private/phase410_h3_order_cases.jsonl")
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["unordered_contract_pair_count"], 408)
        self.assertEqual(audit["order_variant_count"], 816)
        self.assertEqual(audit["pair_failure_count"], 0)
        self.assertEqual(audit["forbidden_priority_cue_hit_count"], 0)
        self.assertEqual(len(rows), 816)

    def test_finite_grammar_universe_has_no_contract_failure(self) -> None:
        audit = read_json("phase410_grammar_finite_universe_audit.json")
        rows = read_jsonl(
            OUT / "protocol/private/phase410_grammar_finite_universe.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["finite_response_case_count"], 4560)
        self.assertEqual(audit["failure_count"], 0)
        self.assertFalse(audit["substring_acceptance_allowed"])
        self.assertFalse(
            audit["bare_be_form_registered_for_sentence_completion"]
        )
        self.assertEqual(len(rows), 4560)
        self.assertTrue(all(row["valid"] for row in rows))

    def test_external_review_packets_are_blind_and_differently_ordered(self) -> None:
        packet_a = read_jsonl(OUT / "external_review/reviewer_a_packet.jsonl")
        packet_b = read_jsonl(OUT / "external_review/reviewer_b_packet.jsonl")
        answer_key = read_jsonl(
            OUT / "protocol/private/phase410_review_answer_key.jsonl"
        )
        self.assertEqual(len(packet_a), 65)
        self.assertEqual(len(packet_b), 65)
        self.assertEqual(len(answer_key), 65)
        self.assertEqual(
            {row["review_item_id"] for row in packet_a},
            {row["review_item_id"] for row in packet_b},
        )
        self.assertNotEqual(
            [row["review_item_id"] for row in packet_a],
            [row["review_item_id"] for row in packet_b],
        )
        self.assertTrue(
            all("admissible_state_set" not in row for row in packet_a + packet_b)
        )

    def test_machine_checks_do_not_impersonate_external_review(self) -> None:
        status = read_json("phase410_external_review_status.json")
        qualification = read_json("phase410_preflight_qualification.json")
        collector = read_json("phase410_collector_reducer_equivalence.json")
        self.assertFalse(status["independent_human_rule_review_completed"])
        self.assertFalse(status["machine_generated_review_is_acceptable"])
        self.assertTrue(qualification["machine_preflight_pass"])
        self.assertFalse(qualification["model_qualification_authorized"])
        self.assertTrue(collector["synthetic_reducer_equivalence_pass"])
        self.assertFalse(
            collector["incremental_collector_model_token_equivalence_completed"]
        )

    def test_stage_keeps_model_physical_causal_and_neuron_gates_closed(self) -> None:
        stage = read_json("phase410_preflight_stage_summary.json")
        self.assertTrue(stage["assessment"]["machine_preflight_pass"])
        self.assertFalse(stage["assessment"]["cuda_execution_performed"])
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["denominators"]["physical_case_count_consumed"], 0)
        self.assertFalse(stage["authorization"]["run_qwen3_model_qualification_next"])
        self.assertFalse(stage["authorization"]["run_descriptive_physical_mapping_next"])
        self.assertFalse(stage["authorization"]["run_causal_intervention_next"])
        self.assertFalse(stage["authorization"]["run_neuron_scan_next"])
        self.assertFalse(stage["single_global_progress_percentage_valid"])

    def test_atlas_mirrors_publish_preflight_without_physical_promotion(self) -> None:
        stage = read_json("phase410_preflight_stage_summary.json")
        atlas_roots = (
            ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
            ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
        )
        for root in atlas_roots:
            mirror = json.loads(
                (root / "phase410_preflight_stage_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            nodes = read_jsonl(root / "phase410_evidence_nodes.jsonl")
            self.assertEqual(mirror, stage)
            self.assertIn(
                manifest["last_phase"],
                {
                    "Phase410-OrthogonalDynamicPreflightStage",
                    "Phase411-FiniteSemanticOperationPreflightStage",
                    "Phase412-TypedObserverQuotientPreflightStage",
                    "Phase413-PredictionKernelMeasurementPreflightStage",
                    "Phase414-ObserverIndexedEventPreflightStage",
                },
            )
            self.assertEqual(manifest["phase410"]["model_case_count"], 0)
            self.assertFalse(progress["single_global_progress_percentage_valid"])
            self.assertEqual(len(nodes), 5)
            self.assertTrue(all(not node["physical"] for node in nodes))
            self.assertTrue(all(not node["causal"] for node in nodes))
            self.assertTrue(all(not node["language_path"] for node in nodes))
        neuron_roots = (
            ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1",
            ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1",
        )
        for root in neuron_roots:
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(manifest["phase"], 410)
            self.assertEqual(
                manifest["phase410_audit"]["new_neuron_path_nodes_promoted"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
