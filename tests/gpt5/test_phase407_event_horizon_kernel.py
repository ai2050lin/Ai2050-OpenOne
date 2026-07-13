#!/usr/bin/env python3
"""Contract tests for Phase407 condition-response event ledgers."""

from __future__ import annotations

import json
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase407_event_horizon_analysis import (  # noqa: E402
    enrich_row,
    group_audit,
    parse_semantic_state,
)
from phase407_event_horizon_protocol import (  # noqa: E402
    FAMILIES,
    HISTORY_MODES,
    INTERFACES,
    MODELS,
    OUT,
    STATE_IDS,
    SURFACE_REPLICAS,
    semantic_transition_table,
)
from phase407_response_partition_diagnostic import classify_mapping  # noqa: E402


class Phase407EventHorizonKernelTest(unittest.TestCase):
    @staticmethod
    def base_row(
        family: str = "knowledge_binding",
        interface: str | None = None,
        target: str | None = None,
    ) -> dict:
        interface = interface or INTERFACES[family][0]
        target = target or STATE_IDS[family][0]
        aliases = {
            "knowledge_binding": {
                "green_yellow": ["green, yellow"],
                "yellow_green": ["yellow, green"],
                "green_green": ["both green"],
                "yellow_yellow": ["both yellow"],
            },
            "rule_reasoning": (
                {"holder_a": ["yes", "true"], "holder_b": ["no", "false"]}
                if interface == "truth_condition"
                else {
                    "holder_a": ["A", "person A", "Arlo"],
                    "holder_b": ["B", "person B", "Galen"],
                }
            ),
            "grammar_constraint": {
                "singular_present": ["is"],
                "plural_present": ["are"],
                "singular_past": ["was"],
                "plural_past": ["were"],
            },
        }[family]
        return {
            "family_id": family,
            "interface_private": interface,
            "semantic_aliases_by_state_private": aliases,
            "target_semantic_state_private": target,
            "all_generated_step_logits_valid": True,
            "step_ledger_private": [],
            "eos_observed": False,
            "eos_step_private": None,
            "H48_right_edge_reached": False,
            "generated_token_count": 0,
            "canonical_target_preferred_to_foil": True,
        }

    @staticmethod
    def with_prefixes(row: dict, prefixes: list[str], eos_step: int | None = None) -> dict:
        row = dict(row)
        row["step_ledger_private"] = [
            {"step": index, "decoded_prefix_private": prefix}
            for index, prefix in enumerate(prefixes, 1)
        ]
        row["generated_token_count"] = len(prefixes)
        row["eos_step_private"] = eos_step
        row["eos_observed"] = eos_step is not None
        return row

    def test_protocol_denominator_and_qualification_are_frozen(self) -> None:
        path = OUT / "protocol/private/phase407_all_cases.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 14_400)
        self.assertEqual(
            sum(row["candidate_split_private"] == "discovery" for row in rows),
            5_760,
        )
        self.assertEqual(len({row["blind_case_id"] for row in rows}), len(rows))
        for model in MODELS:
            for family in FAMILIES:
                selected = [
                    row
                    for row in rows
                    if row["execution_qualification_case"]
                    and row["private_execution_model"] == model
                    and row["family_id"] == family
                ]
                self.assertEqual(len(selected), 8)

    def test_conservative_interface_parsers(self) -> None:
        knowledge = self.base_row()
        parsed = parse_semantic_state(" green, yellow.", knowledge)
        self.assertEqual(parsed["semantic_state_private"], "green_yellow")
        self.assertFalse(parsed["semantic_parse_ambiguous"])
        parsed = parse_semantic_state(" Both green.", knowledge)
        self.assertEqual(parsed["semantic_state_private"], "green_green")

        truth = self.base_row("rule_reasoning", "truth_condition", "holder_a")
        self.assertEqual(
            parse_semantic_state(" True.", truth)["semantic_state_private"],
            "holder_a",
        )
        self.assertIsNone(
            parse_semantic_state("This might not be true.", truth)[
                "semantic_state_private"
            ]
        )

        label = self.base_row(
            "rule_reasoning", "conclusion_completion", "holder_b"
        )
        self.assertEqual(
            parse_semantic_state(" Person B.", label)["semantic_state_private"],
            "holder_b",
        )
        self.assertEqual(
            parse_semantic_state(" Galen.", label)["semantic_state_private"],
            "holder_b",
        )

        grammar = self.base_row(
            "grammar_constraint", "minimal_contrast", "plural_present"
        )
        self.assertEqual(
            parse_semantic_state(" 'are'.", grammar)["semantic_state_private"],
            "plural_present",
        )

    def test_semantic_boundary_and_stop_are_separate_events(self) -> None:
        row = self.with_prefixes(
            self.base_row(),
            [" green", " green, yellow", " green, yellow."],
        )
        enriched = enrich_row(row)
        self.assertEqual(enriched["tau_semantic_private"], 2)
        self.assertEqual(enriched["tau_boundary_private"], 3)
        self.assertIsNone(enriched["tau_stop_private"])
        self.assertTrue(enriched["semantic_correct"])
        self.assertTrue(enriched["complete_response"])

        stopped = enrich_row(self.with_prefixes(self.base_row(), [" green"], 1))
        self.assertIsNone(stopped["tau_semantic_private"])
        self.assertIsNone(stopped["tau_boundary_private"])
        self.assertEqual(stopped["tau_stop_private"], 1)
        self.assertFalse(stopped["complete_response"])

    def test_later_correction_invalidates_first_correct_state(self) -> None:
        row = self.with_prefixes(
            self.base_row(),
            [
                " green, yellow.",
                " green, yellow. Correction: yellow, green.",
            ],
        )
        enriched = enrich_row(row)
        self.assertEqual(
            enriched["normalized_semantic_state_private"], "green_yellow"
        )
        self.assertTrue(enriched["semantic_parse_ambiguous"])
        self.assertTrue(enriched["semantic_reversal"])
        self.assertFalse(enriched["semantic_correct"])

    def test_H48_events_are_independently_right_censored(self) -> None:
        row = self.with_prefixes(self.base_row(), [" undecided"] * 48)
        row["H48_right_edge_reached"] = True
        enriched = enrich_row(row)
        self.assertTrue(enriched["semantic_right_censored_at_H48"])
        self.assertTrue(enriched["boundary_right_censored_at_H48"])
        self.assertTrue(enriched["stop_right_censored_at_H48"])

    @staticmethod
    def perfect_group(family: str) -> list[dict]:
        rows = []
        for state in STATE_IDS[family]:
            for surface in SURFACE_REPLICAS:
                for interface in INTERFACES[family]:
                    for history in HISTORY_MODES:
                        rows.append(
                            {
                                "state_id_private": state,
                                "surface_id_private": surface["surface_id"],
                                "interface_private": interface,
                                "history_mode_private": history,
                                "semantic_correct": True,
                                "complete_response": True,
                                "eos_observed": True,
                                "tau_boundary_private": 3,
                                "semantic_reversal": False,
                                "canonical_target_preferred_to_foil": True,
                            }
                        )
        return rows

    def test_perfect_groups_pass_all_independent_gates(self) -> None:
        for family in FAMILIES:
            audit = group_audit(self.perfect_group(family), family)
            self.assertTrue(audit["surface_group_pass"])
            self.assertTrue(audit["interface_group_pass"])
            self.assertTrue(audit["history_group_pass"])
            self.assertTrue(audit["sequence_group_pass"])
            self.assertTrue(audit["completion_group_pass"])
            self.assertTrue(audit["direct_endpoint_operator_group_pass"])

    def test_surface_gate_does_not_imply_interface_transfer(self) -> None:
        family = "knowledge_binding"
        rows = self.perfect_group(family)
        state = STATE_IDS[family][0]
        history = HISTORY_MODES[0]
        failures = {
            (INTERFACES[family][0], SURFACE_REPLICAS[0]["surface_id"]),
            (INTERFACES[family][1], SURFACE_REPLICAS[1]["surface_id"]),
        }
        for row in rows:
            if (
                row["state_id_private"] == state
                and row["history_mode_private"] == history
                and (row["interface_private"], row["surface_id_private"])
                in failures
            ):
                row["semantic_correct"] = False
        audit = group_audit(rows, family)
        self.assertTrue(audit["surface_group_pass"])
        self.assertFalse(audit["interface_group_pass"])

    def test_surface_gate_does_not_imply_history_transfer(self) -> None:
        family = "grammar_constraint"
        rows = self.perfect_group(family)
        state = STATE_IDS[family][0]
        interface = INTERFACES[family][0]
        failures = {
            (HISTORY_MODES[0], SURFACE_REPLICAS[0]["surface_id"]),
            (HISTORY_MODES[1], SURFACE_REPLICAS[1]["surface_id"]),
        }
        for row in rows:
            if (
                row["state_id_private"] == state
                and row["interface_private"] == interface
                and (row["history_mode_private"], row["surface_id_private"])
                in failures
            ):
                row["semantic_correct"] = False
        audit = group_audit(rows, family)
        self.assertTrue(audit["surface_group_pass"])
        self.assertFalse(audit["history_group_pass"])

    def test_endpoint_graph_is_explicit_but_not_an_instruction(self) -> None:
        graph = semantic_transition_table()
        self.assertEqual(set(graph), set(FAMILIES))
        for family, edges in graph.items():
            for edge in edges:
                self.assertIn(edge["source"], STATE_IDS[family])
                self.assertIn(edge["target"], STATE_IDS[family])
                self.assertNotEqual(edge["source"], edge["target"])

    def test_response_mapping_classes_do_not_credit_wrong_permutations(self) -> None:
        states = ("s0", "s1", "s2")
        self.assertEqual(
            classify_mapping(states, {"s0": "s0", "s1": "s1", "s2": "s2"}),
            "registered_identity_mapping",
        )
        self.assertEqual(
            classify_mapping(states, {"s0": "s1", "s1": "s2", "s2": "s0"}),
            "registered_bijective_nonidentity_mapping",
        )
        self.assertEqual(
            classify_mapping(states, {"s0": "s0", "s1": "s0", "s2": "s2"}),
            "registered_state_collapse",
        )
        self.assertEqual(
            classify_mapping(states, {"s0": "s0", "s1": None, "s2": "s2"}),
            "incomplete_registered_mapping",
        )

    def test_atlas_mirrors_and_evidence_boundary_match(self) -> None:
        research = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
        client = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
        published = (
            "phase407_event_horizon_protocol.json",
            "phase407_discovery_analysis.json",
            "phase407_failure_diagnostic.json",
            "phase407_response_partition_diagnostic.json",
            "phase407_event_horizon_stage_summary.json",
            "phase407_failure_axes.jsonl",
            "phase407_evidence_nodes.jsonl",
            "phase407_evidence_edges.jsonl",
        )
        for name in published:
            self.assertEqual((research / name).read_bytes(), (client / name).read_bytes())
        manifest = json.loads((client / "manifest.json").read_text(encoding="utf-8"))
        latest_phase = int(manifest["last_phase"].split("-", 1)[0].removeprefix("Phase"))
        self.assertGreaterEqual(latest_phase, 407)
        self.assertEqual(
            manifest["phase407"]["strict_crossmodel_candidate_family_count"], 0
        )

    def test_neuron_atlas_keeps_phase407_observational(self) -> None:
        roots = (
            ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1",
            ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1",
        )
        for root in roots:
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(manifest["phase"], 407)
            self.assertEqual(
                manifest["phase407_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            nodes = [
                json.loads(line)
                for line in (root / "phase407_evidence_nodes.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertTrue(nodes)
            self.assertTrue(all(not row["causal"] for row in nodes))
            self.assertTrue(all(not row["language_path"] for row in nodes))
            checksums = json.loads(
                (root / "checksums.json").read_text(encoding="utf-8")
            )
            entries = {row["path"]: row["sha256"] for row in checksums["files"]}
            for name in (
                "phase407_event_horizon_stage_summary.json",
                "phase407_evidence_nodes.jsonl",
                "phase407_evidence_edges.jsonl",
            ):
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                self.assertEqual(entries[name], actual)


if __name__ == "__main__":
    unittest.main()
