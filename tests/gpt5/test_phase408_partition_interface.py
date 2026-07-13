#!/usr/bin/env python3
"""Contract tests for Phase408 response partitions and interface coordinates."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase408_partition_interface_analysis import (  # noqa: E402
    enrich_row,
    group_audit,
    parse_response,
)
from phase408_partition_interface_protocol import (  # noqa: E402
    FAMILIES,
    INTERFACES,
    MODELS,
    OUT,
    STATE_IDS,
    STRUCTURAL_SURFACES,
    encode_raw_class,
    interface_coordinate_map,
    package_for,
    raw_class_aliases,
    raw_class_to_state,
)
from phase408_atlas_sync import evidence_graph  # noqa: E402


class Phase408PartitionInterfaceTest(unittest.TestCase):
    def parser_row(
        self,
        family: str = "knowledge_binding",
        interface: str | None = None,
        state_id: str | None = None,
    ) -> dict:
        interface = interface or INTERFACES[family][0]
        state_id = state_id or STATE_IDS[family][0]
        package = package_for(family, 0, 0)
        return {
            "family_id": family,
            "target_semantic_state_private": state_id,
            "target_raw_response_class_private": encode_raw_class(
                family, interface, state_id
            ),
            "raw_response_aliases_private": raw_class_aliases(
                family, package, interface
            ),
            "raw_class_to_semantic_state_private": raw_class_to_state(
                family, interface
            ),
            "ambiguous_aliases_private": [
                "cannot determine",
                "not enough information",
            ],
        }

    @staticmethod
    def synthetic_group(
        family: str,
        state_permutation: dict[str, str] | None = None,
        unstable_surface: bool = False,
    ) -> list[dict]:
        states = STATE_IDS[family]
        state_permutation = state_permutation or {state: state for state in states}
        rows = []
        for lexical in (0, 1):
            for surface in STRUCTURAL_SURFACES:
                for interface in INTERFACES[family]:
                    for state in states:
                        mapped_state = state_permutation[state]
                        if unstable_surface and surface["surface_id"] == "r003":
                            mapped_state = states[(states.index(mapped_state) + 1) % len(states)]
                        raw_class = encode_raw_class(family, interface, mapped_state)
                        rows.append(
                            {
                                "model": "qwen3",
                                "family_id": family,
                                "split": "discovery",
                                "anonymous_parallel_group_id": "synthetic",
                                "group_priority": 0,
                                "state_id_private": state,
                                "lexical_replica_private": lexical,
                                "surface_id_private": surface["surface_id"],
                                "interface_private": interface,
                                "runtime_numeric_status": "valid",
                                "raw_response_class": raw_class,
                                "semantic_class": (
                                    "allowed" if mapped_state == state else "rejected"
                                ),
                            }
                        )
        return rows

    def test_protocol_denominator_and_registry_are_frozen(self) -> None:
        protocol = json.loads(
            (OUT / "phase408_partition_interface_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        denominator = protocol["denominator"]
        self.assertEqual(denominator["case_count_all_models_all_registered_splits"], 29952)
        self.assertEqual(denominator["discovery_case_count_per_model"], 3744)
        self.assertEqual(denominator["discovery_case_count_all_models"], 11232)
        self.assertEqual(denominator["execution_qualification_case_count_per_model"], 32)
        self.assertEqual(denominator["interfaces_per_family"], 3)
        self.assertEqual(denominator["lexical_replicas"], 2)
        self.assertEqual(denominator["history_mode"], "fixed_empty")
        qualification = json.loads(
            (OUT / "phase408_registry_qualification.json").read_text(encoding="utf-8")
        )
        self.assertTrue(qualification["valid"])
        self.assertEqual(qualification["alias_collision_count"], 0)

    def test_stepwise_greedy_is_not_registered_as_global_map(self) -> None:
        protocol = json.loads(
            (OUT / "phase408_partition_interface_protocol.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            protocol["execution_contract"]["generation"],
            "stepwise_deterministic_greedy_not_global_sequence_map",
        )
        self.assertFalse(
            protocol["claim_boundary"]["greedy_trace_is_global_sequence_map"]
        )

    def test_semantic_runtime_and_event_axes_are_orthogonal(self) -> None:
        registry = [
            json.loads(line)
            for line in (OUT / "phase408_query_contract_registry.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(registry), 9)
        for row in registry:
            self.assertEqual(
                row["semantic_axis"],
                ["allowed", "rejected", "ambiguous", "unparsed"],
            )
            self.assertTrue(row["axes_are_not_a_single_six_class_partition"])

    def test_grammar_interface_alias_overlap_is_explicitly_detectable(self) -> None:
        package = package_for("grammar_constraint", 0, 0)
        be_aliases = {
            alias
            for values in raw_class_aliases(
                "grammar_constraint", package, "be_form"
            ).values()
            for alias in values
        }
        sentence_aliases = {
            alias
            for values in raw_class_aliases(
                "grammar_constraint", package, "sentence_completion"
            ).values()
            for alias in values
        }
        self.assertEqual(be_aliases, {"is", "are", "was", "were"})
        self.assertTrue(be_aliases.issubset(sentence_aliases))

    def test_three_slot_coordinate_maps_compose(self) -> None:
        self.assertEqual(len(STATE_IDS["knowledge_binding"]), 6)
        for family in FAMILIES:
            for source in INTERFACES[family]:
                for middle in INTERFACES[family]:
                    first = interface_coordinate_map(family, source, middle)
                    for target in INTERFACES[family]:
                        second = interface_coordinate_map(family, middle, target)
                        direct = interface_coordinate_map(family, source, target)
                        self.assertEqual(
                            {key: second[value] for key, value in first.items()},
                            direct,
                        )

    def test_parser_separates_allowed_rejected_ambiguous_unparsed(self) -> None:
        row = self.parser_row()
        aliases = row["raw_response_aliases_private"]
        target = row["target_raw_response_class_private"]
        rejected = next(key for key in aliases if key != target)
        allowed = parse_response(aliases[target][0] + ".", row)
        wrong = parse_response(aliases[rejected][0] + ".", row)
        ambiguous = parse_response(
            aliases[target][0] + ", but cannot determine.", row
        )
        missing = parse_response("No registered words are present.", row)
        self.assertEqual(allowed["semantic_class"], "allowed")
        self.assertEqual(wrong["semantic_class"], "rejected")
        self.assertEqual(ambiguous["semantic_class"], "ambiguous")
        self.assertEqual(missing["semantic_class"], "unparsed")

    def test_allowed_semantics_can_coexist_with_stop_censoring(self) -> None:
        row = self.parser_row("grammar_constraint", "be_form", "singular_present")
        row.update(
            {
                "generated_text_clean_private": " is.",
                "generated_text_raw_private": " is.",
                "all_generated_raw_logits_valid": True,
                "all_generated_processed_scores_valid": True,
                "eos_observed": False,
                "H48_right_edge_reached": True,
                "step_ledger_private": [
                    {"step": 1, "decoded_prefix_private": " is"},
                    {"step": 2, "decoded_prefix_private": " is."},
                ],
            }
        )
        enriched = enrich_row(row)
        self.assertEqual(enriched["semantic_class"], "allowed")
        self.assertEqual(enriched["runtime_numeric_status"], "valid")
        self.assertTrue(enriched["boundary_observed"])
        self.assertTrue(enriched["stop_right_censored"])

    def test_perfect_group_passes_partition_and_covariance(self) -> None:
        audit = group_audit(
            self.synthetic_group("knowledge_binding"), "knowledge_binding"
        )
        self.assertEqual(audit["condition_separation_count"], 24)
        self.assertEqual(audit["stable_interface_response_map_count"], 3)
        self.assertTrue(audit["task_coordinate_covariance_pass"])
        self.assertTrue(audit["functional_partition_interface_pass"])

    def test_common_label_permutation_preserves_functional_partition_only(self) -> None:
        states = STATE_IDS["knowledge_binding"]
        permutation = {
            state: states[(index + 1) % len(states)]
            for index, state in enumerate(states)
        }
        audit = group_audit(
            self.synthetic_group("knowledge_binding", permutation),
            "knowledge_binding",
        )
        self.assertTrue(audit["functional_partition_interface_pass"])
        self.assertEqual(audit["condition_label_alignment_count"], 0)

    def test_surface_specific_permutation_fails_stability(self) -> None:
        audit = group_audit(
            self.synthetic_group("knowledge_binding", unstable_surface=True),
            "knowledge_binding",
        )
        self.assertEqual(audit["condition_separation_count"], 24)
        self.assertEqual(audit["stable_interface_response_map_count"], 0)
        self.assertFalse(audit["functional_partition_interface_pass"])

    def test_qualification_denominator_is_exact_for_each_model(self) -> None:
        rows = [
            json.loads(line)
            for line in (OUT / "protocol/private/phase408_all_cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        for model in MODELS:
            selected = [
                row
                for row in rows
                if row["private_execution_model"] == model
                and row["execution_qualification_case"]
            ]
            self.assertEqual(len(selected), 32)
            self.assertEqual({row["candidate_split_private"] for row in selected}, {"qualification"})

    def test_behavioral_partition_does_not_authorize_physical_mapping(self) -> None:
        stage = {
            "denominators": {
                "discovery_case_count": 11232,
                "discovery_group_count": 108,
            },
            "results": {
                "semantic_class_counts": {"allowed": 11232},
                "runtime_numeric_status_counts": {"valid": 11232},
                "condition_separation_pass_group_count": 108,
                "surface_lexical_stability_pass_group_count": 108,
                "task_coordinate_covariance_pass_group_count": 108,
                "functional_group_pass_count": 108,
                "discovery_crossmodel_candidate_families": [
                    "knowledge_binding"
                ],
                "calibration_crossmodel_candidate_families": [
                    "knowledge_binding"
                ],
                "behavioral_crossmodel_candidate_families": [
                    "knowledge_binding"
                ],
            },
        }
        nodes, _ = evidence_graph(stage)
        gate = next(
            node for node in nodes if node["node_id"] == "p408_crossmodel_partition_gate"
        )
        self.assertTrue(gate["history_replication_protocol_authorized"])
        self.assertFalse(gate["physical_protocol_authorized"])

    def test_phase408_atlas_mirrors_match(self) -> None:
        research = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
        client = ROOT / "frontend/public/vis_data/pattern_family_atlas/v2"
        published = (
            "phase408_partition_interface_protocol.json",
            "phase408_discovery_analysis.json",
            "phase408_calibration_analysis.json",
            "phase408_behavioral_holdout_analysis.json",
            "phase408_failure_diagnostic.json",
            "phase408_execution_recovery_audit.json",
            "phase408_partition_interface_stage_summary.json",
            "phase408_failure_axes.jsonl",
            "phase408_interface_failure_axes.jsonl",
            "phase408_evidence_nodes.jsonl",
            "phase408_evidence_edges.jsonl",
        )
        for name in published:
            self.assertEqual((research / name).read_bytes(), (client / name).read_bytes())
        manifest = json.loads((client / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["last_phase"].startswith("Phase"))
        self.assertIn("phase408", manifest)
        self.assertEqual(
            manifest["phase408"]["behavioral_crossmodel_candidate_families"], []
        )

    def test_phase408_neuron_atlas_promotes_no_internal_nodes(self) -> None:
        roots = (
            ROOT / "tests/gpt5/result/pattern_family_neuron_atlas/v1",
            ROOT / "frontend/public/vis_data/pattern_family_neuron_atlas/v1",
        )
        for root in roots:
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertGreaterEqual(manifest["phase"], 408)
            self.assertEqual(
                manifest["phase408_audit"]["new_neuron_path_nodes_promoted"], 0
            )
            self.assertFalse(
                manifest["evidence_boundary"]["validated_internal_operator_available"]
            )
            nodes = [
                json.loads(line)
                for line in (root / "phase408_evidence_nodes.jsonl")
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
                "phase408_partition_interface_stage_summary.json",
                "phase408_evidence_nodes.jsonl",
                "phase408_evidence_edges.jsonl",
            ):
                actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
                self.assertEqual(entries[name], actual)


if __name__ == "__main__":
    unittest.main()
