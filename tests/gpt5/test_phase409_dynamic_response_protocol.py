#!/usr/bin/env python3
"""Contract tests for Phase409 dynamic-response protocol engineering."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_protocol import (  # noqa: E402
    STATE_IDS,
    STRUCTURAL_SURFACES,
    package_for,
)
from phase409_dynamic_response_protocol import (  # noqa: E402
    CONFLICT_DIAGNOSTIC_HISTORY_MODE,
    FAMILIES,
    GATE_ELIGIBLE_HISTORY_MODES,
    HISTORY_MODES,
    INTERFACES,
    OUT,
    direct_solver,
    enumerative_solver,
    history_messages,
    interface_state_signatures,
    joint_query_signature_is_injective,
    parse_response_prefix,
    prior_state_for,
    query_roles,
    raw_response_aliases,
    response_contract,
    scan_event_process,
)


class Phase409DynamicResponseProtocolTest(unittest.TestCase):
    def test_dual_solvers_agree_for_every_finite_scenario(self) -> None:
        scenario_count = 0
        for family in FAMILIES:
            for current in STATE_IDS[family]:
                for history_mode in HISTORY_MODES:
                    prior = (
                        current
                        if history_mode == "h1_prior_equivalent_state"
                        else prior_state_for(family, current)
                    )
                    direct = direct_solver(family, current, history_mode, prior)
                    enumerated = enumerative_solver(
                        family, current, history_mode, prior
                    )
                    self.assertEqual(direct, enumerated)
                    if history_mode in GATE_ELIGIBLE_HISTORY_MODES:
                        self.assertEqual(direct, (current,))
                    else:
                        self.assertEqual(set(direct), {current, prior})
                    scenario_count += 1
        self.assertEqual(scenario_count, 65)

    def test_conflict_is_diagnostic_and_explicit_override_is_unique(self) -> None:
        family = "knowledge_binding"
        current = STATE_IDS[family][0]
        prior = prior_state_for(family, current)
        self.assertEqual(
            set(
                direct_solver(
                    family,
                    current,
                    CONFLICT_DIAGNOSTIC_HISTORY_MODE,
                    prior,
                )
            ),
            {current, prior},
        )
        self.assertEqual(
            direct_solver(
                family,
                current,
                "h4_prior_conflict_then_current_explicit_override",
                prior,
            ),
            (current,),
        )

    def test_single_entity_queries_are_jointly_not_individually_identifiable(self) -> None:
        family = "knowledge_binding"
        interface = "single_entity_value"
        package = package_for(family, 0, 0)
        for role in query_roles(family, interface):
            contract = response_contract(family, package, interface, role)
            self.assertFalse(contract["single_query_injective"])
            self.assertEqual(len(contract["raw_class_to_states"]), 3)
            self.assertTrue(
                all(len(states) == 2 for states in contract["raw_class_to_states"].values())
            )
        signatures = interface_state_signatures(family, interface)
        self.assertEqual(len(set(signatures.values())), 6)
        self.assertTrue(joint_query_signature_is_injective(family, interface))

    def test_every_interface_has_an_identifiable_registered_gate_unit(self) -> None:
        for family in FAMILIES:
            for interface in INTERFACES[family]:
                self.assertTrue(joint_query_signature_is_injective(family, interface))

    def test_sentence_completion_excludes_bare_be_forms(self) -> None:
        package = package_for("grammar_constraint", 0, 0)
        aliases = raw_response_aliases(
            "grammar_constraint", package, "sentence_completion"
        )
        flattened = {alias for values in aliases.values() for alias in values}
        self.assertFalse({"is", "are", "was", "were"} & flattened)
        self.assertTrue(all(package["modifier"] in alias for alias in flattened))

    def test_sentence_parser_marks_bare_form_incomplete(self) -> None:
        family = "grammar_constraint"
        package = package_for(family, 0, 0)
        current = "singular_present"
        prior = prior_state_for(family, current)
        contract = response_contract(
            family, package, "sentence_completion"
        )
        bare = parse_response_prefix(
            " is", contract, current, prior, "h0_current_only"
        )
        complete = parse_response_prefix(
            f" is {package['modifier']}",
            contract,
            current,
            prior,
            "h0_current_only",
        )
        wrong = parse_response_prefix(
            f" are {package['modifier']}",
            contract,
            current,
            prior,
            "h0_current_only",
        )
        self.assertEqual(bare["automaton_state"], "format_incomplete")
        self.assertEqual(complete["automaton_state"], "allowed_response")
        self.assertEqual(wrong["automaton_state"], "rejected_response")

    def test_event_process_preserves_revision_after_allowed_response(self) -> None:
        family = "grammar_constraint"
        package = package_for(family, 0, 0)
        current = "singular_present"
        prior = prior_state_for(family, current)
        modifier = package["modifier"]
        contract = response_contract(
            family, package, "sentence_completion"
        )
        result = scan_event_process(
            [
                " is",
                f" is {modifier}",
                f" is {modifier}.",
                f" is {modifier}. are {modifier}",
            ],
            contract,
            current,
            prior,
            "h0_current_only",
            stopped=True,
        )
        self.assertEqual(result["first_registered_event"], 2)
        self.assertEqual(result["first_allowed_event"], 2)
        self.assertEqual(result["boundary_event"], 3)
        self.assertEqual(result["allowed_exit_event"], 4)
        self.assertEqual(result["stop_event"], 5)
        events = [row["event"] for row in result["event_transition_sequence"]]
        self.assertEqual(events[0], "format_incomplete")
        self.assertIn("allowed_response", events)
        self.assertIn("ambiguous_response", events)
        self.assertIn("boundary_reached", events)
        self.assertEqual(events[-1], "model_stopped")

    def test_history_messages_distinguish_unresolved_and_override_conditions(self) -> None:
        family = "rule_reasoning"
        package = package_for(family, 0, 0)
        irrelevant = package_for(family, 140, 1)
        current = STATE_IDS[family][0]
        prior = prior_state_for(family, current)
        common = (
            family,
            package,
            current,
            prior,
            irrelevant,
            STRUCTURAL_SURFACES[0],
            "holder_ordinal",
            "default",
        )
        conflict_messages, _ = history_messages(
            *common, CONFLICT_DIAGNOSTIC_HISTORY_MODE
        )
        override_messages, _ = history_messages(
            *common, "h4_prior_conflict_then_current_explicit_override"
        )
        self.assertIn("no priority is specified", conflict_messages[-1]["content"])
        self.assertIn("explicitly supersedes", override_messages[-1]["content"])

    def test_frozen_registry_counts_and_execution_boundary(self) -> None:
        protocol = json.loads(
            (OUT / "phase409_dynamic_response_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        denominator = protocol["denominator"]
        self.assertEqual(protocol["schema_version"], "83.0.0")
        self.assertEqual(
            denominator["abstract_case_count_all_registered_splits"], 65280
        )
        self.assertEqual(
            denominator["future_model_rendered_case_count_all_models"], 195840
        )
        self.assertEqual(denominator["discovery_abstract_case_count"], 24480)
        self.assertEqual(
            denominator["future_discovery_case_count_all_models"], 73440
        )
        self.assertEqual(
            denominator["sealed_qualification_case_count_per_future_model"], 55
        )
        self.assertFalse(protocol["authorization"]["model_qualification"])
        self.assertFalse(protocol["authorization"]["physical_mapping"])
        self.assertFalse(protocol["authorization"]["neuron_scan"])
        self.assertFalse(
            protocol["claim_boundary"]["single_global_progress_percentage_valid"]
        )

    def test_machine_qualification_does_not_impersonate_external_review(self) -> None:
        qualification = json.loads(
            (OUT / "phase409_protocol_qualification.json").read_text(
                encoding="utf-8"
            )
        )
        agreement = json.loads(
            (OUT / "phase409_rule_engine_agreement.json").read_text(
                encoding="utf-8"
            )
        )
        prompt_audit = json.loads(
            (OUT / "phase409_prompt_hash_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(qualification["machine_protocol_gate_pass"])
        self.assertFalse(qualification["independent_human_rule_review_completed"])
        self.assertFalse(qualification["model_execution_authorized"])
        self.assertTrue(agreement["valid"])
        self.assertEqual(agreement["scenario_count"], 65)
        self.assertEqual(agreement["expanded_abstract_case_count"], 65280)
        self.assertEqual(agreement["disagreement_count"], 0)
        self.assertTrue(prompt_audit["valid"])
        self.assertEqual(prompt_audit["previous_phase_prompt_overlap_count"], 0)
        self.assertEqual(
            prompt_audit["within_phase_model_prompt_duplicate_count"], 0
        )
        self.assertEqual(prompt_audit["model_rendered_prompt_count"], 195840)

    def test_protocol_stage_summary_preserves_execution_boundary(self) -> None:
        stage = json.loads(
            (OUT / "phase409_protocol_stage_summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(stage["assessment"]["machine_protocol_gate_pass"])
        self.assertFalse(stage["assessment"]["model_execution_performed"])
        self.assertEqual(stage["denominators"]["model_case_count_consumed"], 0)
        self.assertEqual(stage["denominators"]["physical_case_count_consumed"], 0)
        self.assertEqual(stage["results"]["new_physical_path_count"], 0)
        self.assertEqual(stage["results"]["new_head_channel_or_neuron_count"], 0)
        self.assertFalse(stage["authorization"]["run_model_qualification_next"])
        self.assertFalse(stage["authorization"]["run_neuron_scan_next"])

    def test_atlas_and_neuron_mirrors_publish_protocol_only(self) -> None:
        roots = (
            ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
            ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
        )
        source = json.loads(
            (OUT / "phase409_protocol_stage_summary.json").read_text(
                encoding="utf-8"
            )
        )
        for root in roots:
            mirror = json.loads(
                (root / "phase409_protocol_stage_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(mirror, source)
            self.assertEqual(
                manifest["last_phase"], "Phase409-DynamicResponseProtocolStage"
            )
            self.assertEqual(manifest["phase409"]["model_case_count"], 0)
            self.assertFalse(progress["single_global_progress_percentage_valid"])
            self.assertEqual(
                progress["dynamic_response_protocol_stage"]
                ["model_qualification_cases_consumed"]["numerator"],
                0,
            )
        neuron_roots = (
            ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1",
            ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1",
        )
        for root in neuron_roots:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["phase"], 409)
            self.assertEqual(
                manifest["phase409_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            nodes = [
                json.loads(line)
                for line in (root / "phase409_evidence_nodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(nodes), 4)
            self.assertTrue(all(not node["causal"] for node in nodes))
            self.assertTrue(all(not node["physical"] for node in nodes))
            self.assertTrue(all(not node["language_path"] for node in nodes))

    def test_phase408_data_remains_a_closed_prior_stage(self) -> None:
        summary_path = (
            ROOT
            / "tests/gpt5/result/phase408_partition_interface/"
            "phase408_partition_interface_stage_summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["results"]["functional_group_pass_count"], 0)
        self.assertFalse(summary["authorization"]["run_physical_mapping_next"])


if __name__ == "__main__":
    unittest.main()
