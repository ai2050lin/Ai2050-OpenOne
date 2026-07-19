#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase558_fixed_identity_color_protocol as protocol  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase558FixedIdentityColorTests(unittest.TestCase):
    def test_static_denominator_and_seal(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 33792)
        self.assertEqual(audit["open_case_count"], 27648)
        self.assertEqual(audit["sealed_case_count"], 6144)
        self.assertEqual(audit["rows_per_world"], [32])
        self.assertEqual(audit["rows_per_counterfactual_pair"], [2])
        self.assertEqual(commitment["sealed_case_count"], 6144)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_open_bank_contains_no_sealed_rows(self) -> None:
        rows = read_jsonl(protocol.OPEN_CASES_PATH)
        self.assertEqual(len(rows), 27648)
        self.assertFalse(any(row["sealed"] for row in rows))
        self.assertEqual({row["split"] for row in rows}, set(protocol.OPEN_SPLITS))

    def test_counterfactual_pair_fixes_identity_and_swaps_only_binding(self) -> None:
        rows = [
            row for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["pair_id"] == "phase558_behavior_discovery_000_query0_surface0_order0"
        ]
        self.assertEqual(len(rows), 2)
        left, right = sorted(rows, key=lambda row: row["binding"])
        for key in (
            "object_a", "object_b", "color_a", "color_b", "query_object",
            "surface_id", "fact_order", "fact_token_multiset_key", "prompt_token_multiset_key",
        ):
            self.assertEqual(left[key], right[key])
        self.assertEqual(left["target"], right["nontarget_color"])
        self.assertEqual(right["target"], left["nontarget_color"])
        self.assertNotEqual(left["target"], right["target"])

    def test_world_covers_binding_query_surface_and_order(self) -> None:
        rows = [
            row for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["anchor_id"] == "phase558_behavior_discovery_000"
        ]
        self.assertEqual(len(rows), 32)
        self.assertEqual({row["factorial_cell"] for row in rows}, set(protocol.CELLS))
        self.assertEqual({row["binding"] for row in rows}, {0, 1})
        self.assertEqual({row["query_object_index"] for row in rows}, {0, 1})
        self.assertEqual({row["surface_id"] for row in rows}, set(protocol.SURFACES))
        self.assertEqual({row["fact_order"] for row in rows}, set(protocol.FACT_ORDERS))

    def test_splits_have_disjoint_object_lexicons(self) -> None:
        rows = read_jsonl(protocol.OPEN_CASES_PATH)
        objects = {
            split: {
                value
                for row in rows if row["model"] == "qwen3" and row["split"] == split
                for value in (row["object_a"], row["object_b"])
            }
            for split in protocol.OPEN_SPLITS
        }
        for index, left in enumerate(protocol.OPEN_SPLITS):
            for right in protocol.OPEN_SPLITS[index + 1:]:
                self.assertFalse(objects[left] & objects[right])

    def test_protocol_forbids_premature_fine_scan(self) -> None:
        frozen = read_json(protocol.PROTOCOL_PATH)
        policy = frozen["evidence_policy"]
        self.assertTrue(policy["object_identity_fixed_within_counterfactual_pair"])
        self.assertTrue(policy["color_set_fixed_within_counterfactual_pair"])
        self.assertTrue(policy["compute_edge_requires_source_delete_restore_and_exclusion"])
        self.assertTrue(policy["cross_position_parent_before_head_scan"])
        self.assertTrue(policy["parameter_scan_requires_replicated_compute_edge"])
        self.assertFalse(policy["single_neuron_scan_before_compute_edge"])
        self.assertFalse(policy["sealed_split_read"])

    def test_behavior_script_never_reads_private_seal(self) -> None:
        source = (ROOT / "tests/gpt5/phase558_fixed_identity_color_behavior.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase558_sealed_cases", source)
        self.assertIn("Phase558 behavior requires CUDA", source)
        self.assertIn("torch.bfloat16", source)

    def test_behavior_gate_uses_independent_discovery_and_confirmation(self) -> None:
        summary_path = protocol.OUT_DIR / "phase558_behavior_summary.json"
        if not summary_path.exists():
            self.skipTest("Phase558 behavior denominator has not run")
        summary = read_json(summary_path)
        for report in summary["model_reports"]:
            expected = bool(
                report["behavior_discovery_pass"]
                and report["behavior_confirmation_pass"]
                and report["path_all_correct_count_gate_pass"]
            )
            self.assertEqual(report["authorized_for_internal_collection"], expected)
        self.assertFalse(summary["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
