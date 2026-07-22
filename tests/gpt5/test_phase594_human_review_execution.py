#!/usr/bin/env python3
"""Tests for the Phase594 final pre-human review amendment."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase594_human_review_execution_analysis as analysis  # noqa: E402
import phase594_human_review_execution_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def completed_row(
    polarity: str = "negative",
    factuality: str = "supported",
    start: int = 4,
    end: int = 9,
) -> dict:
    return {
        "semantic_polarity": polarity,
        "factuality": factuality,
        "condition_types": ["none"],
        "response_complete": True,
        "later_text_changes_final_semantics": False,
        "decisive_span_start": start,
        "decisive_span_end": end,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def synthetic_fact_registry(review_rows: list[dict]) -> dict:
    template = read_json(protocol.FACT_REGISTRY_TEMPLATE_PATH)
    object_names = sorted(
        (row["object"] for row in template["records"]), key=len, reverse=True
    )
    claim_id_by_object = {
        object_name: f"synthetic_claim_{index:03d}"
        for index, object_name in enumerate(object_names)
    }
    records = []
    for object_name in sorted(object_names):
        claim_id = claim_id_by_object[object_name]
        records.append(
            {
                "object": object_name,
                "audit_status": "completed_with_claims",
                "claims": [
                    {
                        "claim_id": claim_id,
                        "relation": "ordinary_human_food_use",
                        "value_status": "synthetic_fixture_only",
                        "polarity": "uncertain",
                        "subject_part": "whole_or_unspecified",
                        "processing_condition": None,
                        "dose_or_quantity_condition": None,
                        "variety_condition": None,
                        "ripeness_or_life_stage": None,
                        "culture_or_use_context": None,
                        "temporal_scope": None,
                        "geographic_scope": None,
                        "risk_or_exception": None,
                        "dispute_status": "synthetic_fixture",
                        "sources": [
                            {
                                "source_tier": "tier5_unconfirmed",
                                "source_title": "Synthetic fixture",
                                "source_locator": f"synthetic://{claim_id}",
                                "source_version_or_date": "2026-07-21",
                                "source_accessed_at": "2026-07-21T12:00:00+00:00",
                                "evidence_relation": "qualifies",
                                "source_independence": "unknown",
                            }
                        ],
                        "confidence_1_to_5": 1,
                        "supersedes_claim_ids": [],
                        "auditor_id": "synthetic_fact_auditor",
                        "audit_rationale": (
                            "Synthetic integration fixture; not external evidence."
                        ),
                        "audited_at": "2026-07-21T12:00:00+00:00",
                    }
                ],
            }
        )

    review_dispositions = []
    for row in review_rows:
        matches = [
            object_name
            for object_name in object_names
            if object_name.casefold() in row["prompt"].casefold()
        ]
        object_name = matches[0]
        review_dispositions.append(
            {
                "review_id": row["review_id"],
                "disposition": "claims_sufficient",
                "claim_ids": [claim_id_by_object[object_name]],
                "propositions": [
                    {
                        "proposition_id": f"synthetic_prop_{row['review_id']}",
                        "text_span_start": 0,
                        "text_span_end": len(row["response"]),
                        "proposition_text": row["response"],
                        "fact_checkability": "fact_checkable",
                        "claim_links": [
                            {
                                "claim_id": claim_id_by_object[object_name],
                                "coverage_relation": "qualifies",
                            }
                        ],
                    }
                ],
                "auditor_id": "synthetic_fact_auditor",
                "audit_rationale": (
                    "Synthetic integration fixture; not an external fact judgment."
                ),
                "audited_at": "2026-07-21T12:00:00+00:00",
            }
        )
    return {
        "schema_version": "phase594_authoritative_fact_registry_completed.v1",
        "phase_id": protocol.PHASE,
        "source_template_sha256": analysis.io_helpers.sha256_file(
            protocol.FACT_REGISTRY_TEMPLATE_PATH
        ),
        "contains_expected_polarity_or_group": False,
        "registry_complete": True,
        "records": records,
        "review_dispositions": review_dispositions,
    }


class Phase594HumanReviewExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol.register()
        analysis.analyze()

    def test_amendment_precedes_every_human_submission(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["pre_review_amendment"])
        self.assertEqual(audit["prior_review_or_lock_artifact_count"], 0)
        self.assertEqual(audit["non_nfc_prompt_or_response_field_count"], 0)
        self.assertEqual(audit["packet_digest_mismatch_count"], 0)
        self.assertFalse(audit["private_answer_key_read"])
        self.assertFalse(audit["phase590_sealed_cases_read"])

    def test_fact_registry_is_empty_template_not_prefilled_truth(self) -> None:
        registry = read_json(protocol.FACT_REGISTRY_TEMPLATE_PATH)
        self.assertFalse(registry["registry_complete"])
        self.assertFalse(registry["contains_expected_polarity_or_group"])
        self.assertTrue(registry["one_object_can_have_multiple_fact_claims"])
        self.assertEqual(len(registry["records"]), 96)
        self.assertEqual(len({row["object"] for row in registry["records"]}), 96)
        self.assertTrue(all(row["claims"] == [] for row in registry["records"]))
        self.assertEqual(registry["review_dispositions"], [])
        self.assertTrue(registry["completed_registry_is_separate_from_template"])
        self.assertIn(
            "source_independence",
            registry["claim_record_schema"]["required_source_fields"],
        )
        self.assertIn(
            "temporal_scope",
            registry["claim_record_schema"]["required_claim_fields"],
        )
        self.assertTrue(
            registry["supersedes_policy"]["directed_cycles_forbidden"]
        )
        self.assertTrue(
            registry["proposition_schema"][
                "fact_checkable_requires_evidentiary_claim_link"
            ]
        )
        self.assertTrue(
            registry["claim_record_schema"]["claims_must_be_independently_sourced"]
        )

    def test_anchor_rejects_whitespace_and_punctuation_only(self) -> None:
        response = "No, ... edible"
        self.assertFalse(analysis.anchor_has_semantic_content(response, [2, 7]))
        self.assertTrue(analysis.anchor_has_semantic_content(response, [8, 14]))
        self.assertFalse(analysis.anchor_has_semantic_content(response, None))

    def test_repeat_semantics_and_anchor_are_separate_ledgers(self) -> None:
        accepted = {
            "main": completed_row(start=4, end=9),
            "repeat": completed_row(start=5, end=9),
        }
        private_rows = [
            {
                "reviewer_slot": "reviewer_a",
                "submission_id": "main",
                "canonical_review_id": "item-1",
                "item_role": "main",
            },
            {
                "reviewer_slot": "reviewer_a",
                "submission_id": "repeat",
                "canonical_review_id": "item-1",
                "item_role": "repeat_control",
            },
        ]
        _, audit = analysis.split_repeat_ledgers(
            "reviewer_a", accepted, private_rows
        )
        self.assertEqual(audit["substantive_repeat_mismatch_count"], 0)
        self.assertEqual(audit["anchor_repeat_mismatch_count"], 1)
        self.assertTrue(audit["reviewer_quality_pass"])

    def test_three_substantive_repeat_mismatches_fail_reviewer_gate(self) -> None:
        accepted = {}
        private_rows = []
        for index in range(3):
            canonical = f"item-{index}"
            main_id = f"main-{index}"
            repeat_id = f"repeat-{index}"
            accepted[main_id] = completed_row(polarity="negative")
            accepted[repeat_id] = completed_row(polarity="conditional")
            accepted[repeat_id]["condition_types"] = ["processing_required"]
            private_rows.extend(
                [
                    {
                        "reviewer_slot": "reviewer_a",
                        "submission_id": main_id,
                        "canonical_review_id": canonical,
                        "item_role": "main",
                    },
                    {
                        "reviewer_slot": "reviewer_a",
                        "submission_id": repeat_id,
                        "canonical_review_id": canonical,
                        "item_role": "repeat_control",
                    },
                ]
            )
        _, audit = analysis.split_repeat_ledgers(
            "reviewer_a", accepted, private_rows
        )
        self.assertEqual(audit["substantive_repeat_mismatch_count"], 3)
        self.assertFalse(audit["reviewer_quality_pass"])

    def test_time_parser_requires_timezone(self) -> None:
        self.assertIsNone(analysis.parse_aware_time("2026-07-21T12:00:00"))
        self.assertIsNotNone(analysis.parse_aware_time("2026-07-21T12:00:00Z"))
        self.assertIsNotNone(
            analysis.parse_aware_time("2026-07-21T12:00:00-09:00")
        )

    def test_no_humans_keep_every_scientific_gate_closed(self) -> None:
        status = read_json(analysis.STATUS_PATH)
        stage = read_json(analysis.STAGE_PATH)
        self.assertEqual(status["quality_valid_locked_reviewer_count"], 0)
        self.assertEqual(status["completed_quality_valid_locked_main_label_count"], 0)
        self.assertEqual(status["workflow_unresolved_item_count"], 288)
        self.assertEqual(status["event_anchor_qualified_item_count"], 0)
        self.assertFalse(status["semantic_gold_complete"])
        self.assertFalse(status["reviewed_factuality_gold_complete"])
        self.assertFalse(status["authoritative_fact_registry_complete"])
        self.assertFalse(status["separate_gold_artifacts_written"])
        self.assertEqual(
            stage["status"], "blocked_pending_three_external_human_reviewers"
        )
        self.assertFalse(stage["automatic_model_execution_now"])
        self.assertTrue(all(not value for value in stage["authorization"].values()))

    def test_fact_registry_rejects_cycles_and_partial_proposition_coverage(
        self,
    ) -> None:
        source_rows = analysis.io_helpers.read_jsonl_gz(analysis.phase593.SOURCE_QUEUE_PATH)

        def self_reference(registry: dict) -> None:
            claim = registry["records"][0]["claims"][0]
            claim["supersedes_claim_ids"] = [claim["claim_id"]]

        def two_claim_cycle(registry: dict) -> None:
            first = registry["records"][0]["claims"][0]
            second = registry["records"][1]["claims"][0]
            first["supersedes_claim_ids"] = [second["claim_id"]]
            second["supersedes_claim_ids"] = [first["claim_id"]]

        def partial_proposition_coverage(registry: dict) -> None:
            registry["review_dispositions"][0]["propositions"][0][
                "claim_links"
            ] = []

        cases = (
            ("self_superseded_claim", self_reference),
            ("supersedes_cycle", two_claim_cycle),
            ("fact_checkable_without_evidence", partial_proposition_coverage),
        )
        for expected_error, mutate in cases:
            with self.subTest(expected_error=expected_error):
                registry = synthetic_fact_registry(source_rows)
                mutate(registry)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    temporary_root = Path(temporary_directory)
                    completed_path = temporary_root / "fact_registry.json"
                    lock_path = temporary_root / "fact_registry_lock.json"
                    analysis.io_helpers.write_json(completed_path, registry)
                    with (
                        patch.object(
                            analysis.protocol,
                            "FACT_REGISTRY_COMPLETED_PATH",
                            completed_path,
                        ),
                        patch.object(
                            analysis.protocol,
                            "FACT_REGISTRY_LOCK_PATH",
                            lock_path,
                        ),
                    ):
                        result = analysis.validate_fact_registry()
                    self.assertFalse(result["registry_complete"])
                    self.assertGreater(
                        result["structural_error_counts"].get(expected_error, 0),
                        0,
                    )
                    self.assertFalse(lock_path.exists())

    def test_isolated_full_workflow_reaches_adjudication_and_gold(self) -> None:
        source_rows = analysis.io_helpers.read_jsonl_gz(analysis.phase593.SOURCE_QUEUE_PATH)
        conflict_review_id = source_rows[0]["review_id"]
        private_rows = read_json(analysis.phase593.PRIVATE_MAP_PATH)["rows"]
        mapping_by_slot_and_submission = {
            (row["reviewer_slot"], row["submission_id"]): row
            for row in private_rows
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            completed_paths = {
                slot: temporary_root / f"{slot}_completed.jsonl"
                for slot in protocol.REVIEWER_SLOTS
            }
            lock_paths = {
                slot: temporary_root / f"{slot}_lock.json"
                for slot in protocol.REVIEWER_SLOTS
            }

            for slot_index, slot in enumerate(protocol.REVIEWER_SLOTS):
                templates = analysis.read_jsonl(
                    analysis.phase593.response_template_path(slot)
                )
                packet_by_submission = {
                    row["submission_id"]: row
                    for row in analysis.read_jsonl_gz(
                        analysis.phase593.packet_path(slot)
                    )
                }
                completed_rows = []
                for template in templates:
                    row = dict(template)
                    mapping = mapping_by_slot_and_submission[
                        (slot, row["submission_id"])
                    ]
                    batch_index = int(row["batch_id"].split("_")[1])
                    batch_start = datetime(
                        2026, 7, 21, batch_index, tzinfo=timezone.utc
                    )
                    response = packet_by_submission[row["submission_id"]]["response"]
                    row.update(
                        {
                            "reviewer_id": f"synthetic_reviewer_{slot_index}",
                            "semantic_polarity": (
                                "positive"
                                if slot == "reviewer_c"
                                and mapping["canonical_review_id"]
                                == conflict_review_id
                                else "negative"
                            ),
                            "condition_types": ["none"],
                            "factuality": "supported",
                            "factuality_source_tier": (
                                "tier4_stable_ordinary_knowledge"
                            ),
                            "factuality_evidence": (
                                "Synthetic integration fixture; not scientific evidence."
                            ),
                            "negation_scope_text": [],
                            "condition_scope_text": [],
                            "has_contrast": False,
                            "response_complete": True,
                            "later_text_changes_final_semantics": False,
                            "decisive_span_start": 0,
                            "decisive_span_end": min(1, len(response)),
                            "confidence_1_to_5": 4,
                            "rationale": (
                                "Synthetic integration fixture; not a human judgment."
                            ),
                            "reviewed_at": (
                                batch_start + timedelta(minutes=15)
                            ).isoformat(),
                            "batch_started_at": batch_start.isoformat(),
                            "batch_completed_at": (
                                batch_start + timedelta(minutes=30)
                            ).isoformat(),
                        }
                    )
                    completed_rows.append(row)
                write_jsonl(completed_paths[slot], completed_rows)

            adjudicator_completed = temporary_root / "adjudicator_completed.jsonl"
            adjudicator_lock = temporary_root / "adjudicator_lock.json"
            fact_registry_completed = temporary_root / "fact_registry_completed.json"
            fact_registry_lock = temporary_root / "fact_registry_lock.json"
            output_paths = {
                "STATUS_PATH": temporary_root / "status.json",
                "STAGE_PATH": temporary_root / "stage.json",
                "ADJUDICATION_PACKET_PATH": temporary_root
                / "adjudicator_packet.jsonl.gz",
                "ADJUDICATION_TEMPLATE_PATH": temporary_root
                / "adjudicator_template.jsonl",
                "SEMANTIC_GOLD_PATH": temporary_root / "semantic_gold.jsonl.gz",
                "FACTUALITY_GOLD_PATH": temporary_root / "factuality_gold.jsonl.gz",
                "ANCHOR_GOLD_PATH": temporary_root / "anchor_gold.jsonl.gz",
                "PROVENANCE_PATH": temporary_root / "provenance.jsonl.gz",
            }

            with (
                patch.object(
                    analysis.phase593,
                    "completed_response_path",
                    side_effect=lambda slot: completed_paths[slot],
                ),
                patch.object(
                    analysis.phase593,
                    "submission_lock_path",
                    side_effect=lambda slot: lock_paths[slot],
                ),
                patch.object(
                    analysis.protocol,
                    "adjudicator_completed_path",
                    return_value=adjudicator_completed,
                ),
                patch.object(
                    analysis.protocol,
                    "adjudicator_lock_path",
                    return_value=adjudicator_lock,
                ),
                patch.object(
                    analysis.protocol,
                    "FACT_REGISTRY_COMPLETED_PATH",
                    fact_registry_completed,
                ),
                patch.object(
                    analysis.protocol,
                    "FACT_REGISTRY_LOCK_PATH",
                    fact_registry_lock,
                ),
                patch.multiple(analysis, **output_paths),
            ):
                pending_stage = analysis.analyze()
                pending_status = read_json(output_paths["STATUS_PATH"])
                self.assertEqual(
                    pending_stage["status"],
                    "blocked_pending_independent_adjudicator",
                )
                self.assertEqual(
                    pending_stage["next_required_external_action"],
                    "independent_fourth_person_complete_adjudication_packet",
                )
                self.assertEqual(
                    pending_status["quality_valid_locked_reviewer_count"], 3
                )
                self.assertEqual(
                    pending_status["adjudication_required_item_count"], 1
                )
                self.assertEqual(pending_status["directly_resolved_item_count"], 287)

                adjudication_rows = analysis.read_jsonl(
                    output_paths["ADJUDICATION_TEMPLATE_PATH"]
                )
                self.assertEqual(len(adjudication_rows), 1)
                for row in adjudication_rows:
                    row.update(
                        {
                            "adjudicator_id": "synthetic_adjudicator_3",
                            "semantic_polarity": "negative",
                            "condition_types": ["none"],
                            "factuality": "supported",
                            "factuality_source_tier": (
                                "tier4_stable_ordinary_knowledge"
                            ),
                            "factuality_evidence": (
                                "Synthetic integration fixture; not scientific evidence."
                            ),
                            "confidence_1_to_5": 4,
                            "rationale": (
                                "Synthetic integration fixture; not a human judgment."
                            ),
                            "adjudicated_at": "2026-07-21T12:00:00+00:00",
                        }
                    )
                write_jsonl(adjudicator_completed, adjudication_rows)

                completed_stage = analysis.analyze()
                completed_status = read_json(output_paths["STATUS_PATH"])
                self.assertEqual(
                    completed_stage["status"], "blocked_pending_external_fact_audit"
                )
                self.assertTrue(completed_status["semantic_gold_complete"])
                self.assertFalse(
                    completed_status["authoritative_fact_registry_complete"]
                )
                self.assertFalse(
                    completed_stage["authorization"]["evaluate_behavior_truth"]
                )
                self.assertTrue(
                    completed_stage["authorization"][
                        "develop_observer_on_discovery_gold"
                    ]
                )
                self.assertEqual(completed_status["workflow_unresolved_item_count"], 0)
                self.assertEqual(
                    completed_status["independently_adjudicated_item_count"], 1
                )
                self.assertEqual(
                    len(analysis.read_jsonl_gz(output_paths["SEMANTIC_GOLD_PATH"])),
                    288,
                )
                self.assertEqual(
                    len(analysis.read_jsonl_gz(output_paths["FACTUALITY_GOLD_PATH"])),
                    288,
                )
                self.assertTrue(output_paths["ANCHOR_GOLD_PATH"].exists())
                self.assertTrue(output_paths["PROVENANCE_PATH"].exists())

                fact_registry = synthetic_fact_registry(source_rows)
                fact_registry["review_dispositions"] = fact_registry[
                    "review_dispositions"
                ][:-1]
                analysis.io_helpers.write_json(fact_registry_completed, fact_registry)
                incomplete_fact_stage = analysis.analyze()
                incomplete_fact_status = read_json(output_paths["STATUS_PATH"])
                self.assertEqual(
                    incomplete_fact_stage["status"],
                    "blocked_pending_external_fact_audit",
                )
                self.assertEqual(
                    incomplete_fact_status["authoritative_fact_registry_result"][
                        "missing_review_disposition_count"
                    ],
                    1,
                )
                self.assertFalse(fact_registry_lock.exists())

                fact_registry = synthetic_fact_registry(source_rows)
                analysis.io_helpers.write_json(fact_registry_completed, fact_registry)
                final_stage = analysis.analyze()
                final_status = read_json(output_paths["STATUS_PATH"])
                self.assertEqual(
                    final_stage["status"],
                    "external_semantic_and_fact_artifacts_ready_pending_"
                    "evaluable_denominator_contract",
                )
                self.assertTrue(
                    final_status["authoritative_fact_registry_complete"]
                )
                self.assertTrue(fact_registry_lock.exists())
                self.assertTrue(
                    final_stage["authorization"][
                        "develop_observer_on_discovery_gold"
                    ]
                )
                self.assertTrue(
                    final_stage["readiness"]["behavior_truth_artifacts_ready"]
                )
                self.assertFalse(
                    final_stage["readiness"][
                        "evaluable_denominator_contract_frozen"
                    ]
                )
                self.assertFalse(
                    final_stage["authorization"]["evaluate_behavior_truth"]
                )


if __name__ == "__main__":
    unittest.main()
