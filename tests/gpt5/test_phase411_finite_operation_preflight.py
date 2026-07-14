#!/usr/bin/env python3
"""Contract tests for the Phase411 model-free finite-operation preflight."""

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
from phase410_orthogonal_preflight import exact_response_parse  # noqa: E402
from phase411_finite_operation_preflight import (  # noqa: E402
    OUT,
    registered_semantic_parse,
    semantic_contract_index,
)


def read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase411FiniteOperationPreflightTest(unittest.TestCase):
    def grammar_contract(self) -> tuple[dict, str, str, str]:
        family = "grammar_constraint"
        package = package_for(family, 0, 0)
        current = "singular_present"
        prior = prior_state_for(family, current)
        contract = response_contract(family, package, "sentence_completion")
        return contract, package["modifier"], current, prior

    def test_dual_channel_preserves_strict_exact_and_adds_registered_wrapper(self) -> None:
        contract, modifier, current, prior = self.grammar_contract()
        text = f"the answer is is {modifier}"
        strict = exact_response_parse(
            text, contract, current, prior, "h0_current_only"
        )
        semantic = registered_semantic_parse(
            text,
            contract,
            current,
            prior,
            "h0_current_only",
            index=semantic_contract_index(contract),
        )
        self.assertEqual(strict["semantic_class"], "no_registered_response")
        self.assertEqual(semantic["channel_status"], "resolved_registered_response")
        self.assertEqual(semantic["semantic_class"], "allowed_response")

    def test_negation_scope_and_hedging_are_not_forced_into_states(self) -> None:
        contract, modifier, current, prior = self.grammar_contract()
        for text, status in (
            (f"not is {modifier}", "explicitly_negated_candidate"),
            (f"not only is {modifier}", "scope_unresolved"),
            (f"maybe is {modifier}", "hedged_candidate"),
        ):
            observed = registered_semantic_parse(
                text, contract, current, prior, "h0_current_only"
            )
            self.assertEqual(observed["channel_status"], status)
            self.assertIsNone(observed["semantic_class"])
            self.assertIsNone(observed["raw_response_class"])

    def test_explicit_revision_uses_only_the_frozen_final_clause(self) -> None:
        contract, modifier, current, prior = self.grammar_contract()
        observed = registered_semantic_parse(
            f"I first answered are {modifier}, but my final answer is is {modifier}",
            contract,
            current,
            prior,
            "h0_current_only",
        )
        self.assertTrue(observed["revision_detected"])
        self.assertEqual(observed["semantic_class"], "allowed_response")
        self.assertEqual(len(observed["candidate_raw_classes"]), 2)

    def test_semantic_finite_universe_is_large_exhaustive_and_failure_free(self) -> None:
        audit = read_json("phase411_registered_semantic_dual_channel_audit.json")
        contexts = read_jsonl(
            OUT / "protocol/private/phase411_semantic_context_index.jsonl"
        )
        failures = read_jsonl(
            OUT / "protocol/private/phase411_semantic_failures.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["contract_context_count"], 16320)
        self.assertEqual(audit["finite_response_case_count"], 1208640)
        self.assertEqual(audit["failure_count"], 0)
        self.assertEqual(len(contexts), 16320)
        self.assertEqual(failures, [])
        self.assertGreater(
            audit["registered_semantic_resolved_case_count"],
            audit["strict_resolved_case_count"],
        )
        self.assertFalse(audit["open_language_semantics_tested"])

    def test_operation_algebra_closes_but_coarse_partitions_can_fail(self) -> None:
        audit = read_json("phase411_finite_operation_closure_audit.json")
        operations = read_jsonl(
            OUT / "protocol/private/phase411_operation_registry.jsonl"
        )
        transitions = read_jsonl(
            OUT / "protocol/private/phase411_state_transitions.jsonl"
        )
        compositions = read_jsonl(
            OUT / "protocol/private/phase411_operation_composition.jsonl"
        )
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["operation_count"], 46)
        self.assertEqual(audit["two_sided_inverse_count"], 46)
        self.assertEqual(audit["state_transition_count"], 250)
        self.assertEqual(audit["composition_case_count"], 1348)
        self.assertEqual(audit["composition_failure_count"], 0)
        self.assertEqual(audit["history_rule_covariance_case_count"], 1250)
        self.assertEqual(audit["history_rule_covariance_failure_count"], 0)
        self.assertGreater(audit["coarse_observer_unstable_operation_cell_count"], 0)
        self.assertEqual(audit["joint_observer_unstable_operation_cell_count"], 0)
        self.assertFalse(audit["model_functional_bisimulation_established"])
        self.assertEqual(len(operations), 46)
        self.assertEqual(len(transitions), 250)
        self.assertEqual(len(compositions), 1348)

    def test_review_workflow_requires_confidence_reason_and_two_people(self) -> None:
        status = read_json("phase411_external_review_v2_status.json")
        template_a = read_jsonl(
            OUT / "external_review/reviewer_a_response_template_v2.jsonl"
        )
        template_b = read_jsonl(
            OUT / "external_review/reviewer_b_response_template_v2.jsonl"
        )
        self.assertEqual(len(template_a), 65)
        self.assertEqual(len(template_b), 65)
        self.assertTrue(
            all(
                "confidence_1_to_5" in row and "reason" in row
                for row in template_a + template_b
            )
        )
        self.assertFalse(status["independent_human_rule_review_completed"])
        self.assertFalse(status["machine_registry_is_privileged_during_disagreement"])
        self.assertEqual(status["pair_status_counts"]["pending_independent_pair"], 65)

    def test_stage_keeps_every_scientific_execution_gate_closed(self) -> None:
        stage = read_json("phase411_stage_summary.json")
        qualification = read_json("phase411_qualification.json")
        self.assertTrue(stage["assessment"]["machine_preflight_pass"])
        self.assertFalse(stage["assessment"]["cuda_execution_performed"])
        self.assertFalse(stage["assessment"]["model_functional_bisimulation_established"])
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
        stage = read_json("phase411_stage_summary.json")
        for root in (
            ROOT / "tests/gpt5/result/pattern_family_atlas/v2",
            ROOT / "frontend/public/vis_data/pattern_family_atlas/v2",
        ):
            mirror = json.loads(
                (root / "phase411_stage_summary.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (root / "progress.json").read_text(encoding="utf-8")
            )
            nodes = read_jsonl(root / "phase411_evidence_nodes.jsonl")
            self.assertEqual(mirror, stage)
            self.assertIn(
                manifest["last_phase"],
                {
                    "Phase411-FiniteSemanticOperationPreflightStage",
                    "Phase412-TypedObserverQuotientPreflightStage",
                    "Phase413-PredictionKernelMeasurementPreflightStage",
                    "Phase414-ObserverIndexedEventPreflightStage",
                },
            )
            self.assertEqual(manifest["phase411"]["model_case_count"], 0)
            self.assertFalse(progress["single_global_progress_percentage_valid"])
            self.assertEqual(len(nodes), 5)
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
            self.assertGreaterEqual(manifest["phase"], 411)
            self.assertEqual(
                manifest["phase411_audit"]["new_neuron_path_nodes_promoted"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
