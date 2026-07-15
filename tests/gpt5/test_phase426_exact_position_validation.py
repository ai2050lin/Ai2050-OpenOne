#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase426_exact_position_role_validation"
VIS = ROOT / "frontend/public/vis_data/phase426_exact_position_role_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase426ExactPositionContractTest(unittest.TestCase):
    def test_frozen_denominator_and_implementation(self) -> None:
        protocol = read_json(OUT / "phase426_protocol.json")
        validation = protocol["validation"]
        self.assertEqual(
            protocol["schema_version"], "phase426_exact_position_role.v1"
        )
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["formal_replica_group_count"], 512)
        self.assertEqual(validation["formal_condition_count"], 49152)
        self.assertEqual(validation["open_condition_count"], 36864)
        self.assertEqual(validation["sealed_condition_count"], 12288)
        self.assertEqual(validation["instrument_replica_group_count"], 8)
        self.assertEqual(validation["instrument_condition_count"], 768)
        self.assertEqual(validation["position_counterfactual_mismatch_count"], 0)
        self.assertEqual(validation["replica_group_split_leak_count"], 0)
        self.assertFalse(protocol["evidence_contract"]["strict_human_double_blind"])
        self.assertFalse(
            protocol["evidence_contract"]["causal_claim_allowed_before_sealed_pass"]
        )
        self.assertTrue(protocol["geometry_contract"]["identity_map_is_primary"])
        self.assertFalse(
            protocol["geometry_contract"]["learned_three_dimensional_map_pre_registered"]
        )
        for filename, expected in protocol["implementation_commitments"].items():
            actual = hashlib.sha256(
                (ROOT / "tests/gpt5" / filename).read_bytes()
            ).hexdigest()
            self.assertEqual(actual, expected, filename)

    def test_exact_position_counterfactual(self) -> None:
        rows = read_jsonl(OUT / "phase426_registered_conditions_open.jsonl")
        by_axis = defaultdict(list)
        for row in rows:
            by_axis[
                (
                    row["pair_id"],
                    row["model"],
                    row["role"],
                    row["interface"],
                    row["history"],
                )
            ].append(row)
            self.assertTrue(row["target_sequence_token_ids"])
            self.assertTrue(row["opposite_sequence_token_ids"])
            self.assertNotEqual(
                row["target_branch_token_id"], row["opposite_branch_token_id"]
            )
        self.assertTrue(by_axis)
        for pair in by_axis.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual({row["timing"] for row in pair}, {"early_role", "late_role"})
            left, right = pair
            self.assertEqual(left["source_positions"], right["source_positions"])
            self.assertEqual(left["query_positions"], right["query_positions"])
            self.assertEqual(left["prediction_position"], right["prediction_position"])
            self.assertEqual(left["executed_token_count"], right["executed_token_count"])
            self.assertNotEqual(left["active_role_positions"], right["active_role_positions"])

    def test_independent_group_is_split_unit(self) -> None:
        pairs = read_jsonl(OUT / "phase426_registered_pairs.jsonl")
        groups = defaultdict(list)
        for row in pairs:
            groups[row["replica_group_id"]].append(row)
        self.assertEqual(len(groups), 512)
        by_block_split = Counter()
        for rows in groups.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["lexical_replica"] for row in rows}, {0, 1})
            self.assertEqual(len({row["split"] for row in rows}), 1)
            by_block_split[(rows[0]["block_id"], rows[0]["split"])] += 1
        self.assertTrue(all(value == 32 for value in by_block_split.values()))

    def test_instrument_and_completed_outputs(self) -> None:
        audit_path = OUT / "phase426_instrument_audit.json"
        if not audit_path.exists():
            self.skipTest("Phase426 instrument collection has not completed")
        instrument = read_json(audit_path)
        self.assertTrue(instrument["instrument_gate_pass"])
        self.assertFalse(instrument["thresholds_or_theory_updated"])
        for model in MODELS:
            self.assertTrue(instrument["model_results"][model]["gate_pass"])
            behavior = read_jsonl(
                OUT / "models" / model / "instrument" / "phase426_behavior_rows.jsonl"
            )
            self.assertEqual(len(behavior), 256)
            self.assertTrue(
                all(
                    math.isfinite(float(row["teacher_sequence_logprob_margin"]))
                    for row in behavior
                )
            )

        summary_path = OUT / "phase426_global_summary.json"
        if not summary_path.exists():
            return
        summary = read_json(summary_path)
        self.assertEqual(summary["strict_mechanism_closure"], "0/72")
        self.assertFalse(summary["causal_tested"])
        gate = read_json(OUT / "phase426_gate_freeze.json")
        self.assertFalse(gate["causal_unlock"])
        for model in MODELS:
            complete = read_json(
                OUT / "models" / model / "open" / "phase426_collection_complete.json"
            )
            self.assertTrue(complete["all_rows_complete"])
            self.assertTrue(complete["component_ledger_gate_pass"])
            self.assertEqual(complete["condition_count"], 12288)
        manifest = read_json(VIS / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase426_exact_position_role_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            graph = read_json(VIS / item["filename"])
            self.assertFalse(graph["graph"]["meta"]["causal"])
            self.assertTrue(
                all(not edge["compute_edge"] for edge in graph["graph"]["edges"])
            )


if __name__ == "__main__":
    unittest.main()
