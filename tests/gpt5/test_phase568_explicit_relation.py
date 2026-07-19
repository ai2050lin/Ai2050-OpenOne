#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase567_multi_relation_binding_protocol as phase567  # noqa: E402
import phase568_explicit_relation_protocol as protocol  # noqa: E402
from phase548_shared_attention_compute_protocol import tokenizer_for  # noqa: E402
from phase568_role_position_utils import ROLE_GROUPS, role_positions, typed_union  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase568ExplicitRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.open_rows = read_jsonl(protocol.OPEN_CASES_PATH)

    def test_static_denominator_and_seal(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 82944)
        self.assertEqual(audit["open_case_count"], 67392)
        self.assertEqual(audit["sealed_case_count"], 15552)
        self.assertEqual(audit["rows_per_world"], [108])
        self.assertEqual(audit["rows_per_triplet"], [3])
        self.assertEqual(audit["phase567_object_overlap_count"], 0)
        self.assertEqual(commitment["sealed_case_count"], 15552)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_open_bank_contains_no_sealed_rows(self) -> None:
        self.assertEqual(len(self.open_rows), 67392)
        self.assertFalse(any(row["sealed"] for row in self.open_rows))
        self.assertEqual({row["split"] for row in self.open_rows}, set(protocol.OPEN_SPLITS))

    def test_triplet_changes_only_queried_relation_binding(self) -> None:
        triplet_id = "phase568_gate_discovery_000_query0_relationbody_surface0_order0"
        rows = [
            row for row in self.open_rows
            if row["model"] == "qwen3" and row["triplet_id"] == triplet_id
        ]
        self.assertEqual(len(rows), 3)
        ordered = sorted(rows, key=lambda row: row["binding"])
        for key in (
            "objects", "values", "query_object", "query_relation", "surface_id",
            "fact_order", "fact_token_multiset_key", "prompt_token_multiset_key",
        ):
            self.assertEqual([row[key] for row in ordered], [ordered[0][key]] * 3)
        self.assertEqual({row["target"] for row in ordered}, set(ordered[0]["values"]))
        self.assertEqual(len({tuple(row["relation_maps"]["body"]) for row in ordered}), 3)
        self.assertEqual(len({tuple(row["relation_maps"]["tag"]) for row in ordered}), 1)

    def test_explicit_relation_keys_are_present(self) -> None:
        sampled = {
            row["surface_id"]: row
            for row in self.open_rows
            if row["model"] == "qwen3"
            and row["split"] == "gate_discovery"
            and row["query_relation"] == "body"
        }
        self.assertEqual(set(sampled), set(protocol.SURFACES))
        for row in sampled.values():
            prompt = row["raw_prompt"].casefold()
            self.assertIn("body", prompt)
            self.assertIn("color", prompt)
            self.assertIn("tag", prompt)
            self.assertNotIn("surface", prompt)
            self.assertNotIn("marker", prompt)

    def test_new_object_lexicon_is_disjoint_from_phase567(self) -> None:
        prior = {
            value
            for row in read_jsonl(phase567.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            for value in row["objects"]
        }
        current = {
            value
            for row in self.open_rows
            if row["model"] == "qwen3"
            for value in row["objects"]
        }
        self.assertFalse(prior & current)

    def test_world_covers_full_factorial(self) -> None:
        rows = [
            row for row in self.open_rows
            if row["model"] == "qwen3"
            and row["anchor_id"] == "phase568_gate_discovery_000"
        ]
        self.assertEqual(len(rows), 108)
        self.assertEqual({row["factorial_cell"] for row in rows}, set(protocol.CELLS))

    def test_eight_role_groups_are_disjoint(self) -> None:
        for model in protocol.MODELS:
            tokenizer = tokenizer_for(model)
            sampled = {}
            for row in self.open_rows:
                if row["model"] != model or row["split"] != "role_discovery":
                    continue
                key = (row["surface_id"], row["fact_order"], row["query_relation"])
                sampled.setdefault(key, row)
            self.assertEqual(len(sampled), 12)
            for row in sampled.values():
                ids, groups = role_positions(tokenizer, row)
                self.assertEqual(tuple(groups), ROLE_GROUPS)
                physical = typed_union(groups)
                self.assertEqual(len(physical), len(set(physical)))
                self.assertGreaterEqual(len(physical), 11)
                self.assertLess(max(physical), len(ids))

    def test_behavior_script_cannot_read_sealed_rows(self) -> None:
        source = (ROOT / "tests/gpt5/phase568_explicit_relation_behavior.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase568_sealed_cases", source)
        self.assertIn('runner.PHASE_ID = "Phase568"', source)

    def test_frozen_behavior_gate_authorizes_no_model(self) -> None:
        summary = read_json(
            protocol.OUT_DIR / "phase568_behavior_summary.json"
        )
        self.assertEqual(summary["authorized_models"], [])
        self.assertFalse(summary["sealed_split_read"])
        for report in summary["model_reports"]:
            self.assertFalse(report["gate_discovery_pass"])
            self.assertFalse(report["gate_confirmation_pass"])
            self.assertFalse(report["authorized_for_internal_collection"])

    def test_qwen_failure_signature_is_relation_specific(self) -> None:
        diagnostics = read_json(
            protocol.OUT_DIR / "phase568_behavior_failure_diagnostics.json"
        )
        qwen = next(
            report for report in diagnostics["models"] if report["model"] == "qwen3"
        )
        self.assertEqual(qwen["error_count"], 704)
        self.assertEqual(qwen["signature_counts"]["same_object_other_relation"], 701)
        self.assertEqual(qwen["top_10_worst_cell_overlap_count"], 9)
        self.assertFalse(diagnostics["phase568_result_reclassified"])


if __name__ == "__main__":
    unittest.main()
