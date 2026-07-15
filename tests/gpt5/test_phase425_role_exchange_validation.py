#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase425_role_exchange_validation"
VIS = ROOT / "frontend/public/vis_data/phase425_role_exchange_validation"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase425RoleExchangeContractTest(unittest.TestCase):
    def test_frozen_denominator_and_implementation(self) -> None:
        protocol = read_json(OUT / "phase425_protocol.json")
        validation = protocol["validation"]
        self.assertEqual(protocol["schema_version"], "phase425_role_exchange.v2")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["pair_count"], 384)
        self.assertEqual(validation["condition_count"], 9216)
        self.assertEqual(validation["open_condition_count"], 6912)
        self.assertEqual(validation["sealed_condition_count"], 2304)
        self.assertEqual(validation["replica_group_count"], 192)
        self.assertEqual(validation["replica_group_split_leak_count"], 0)
        self.assertFalse(protocol["evidence_contract"]["strict_human_double_blind"])
        self.assertFalse(protocol["evidence_contract"]["causal_claim_allowed_before_sealed_pass"])
        for filename, expected in protocol["implementation_commitments"].items():
            actual = hashlib.sha256((ROOT / "tests/gpt5" / filename).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, filename)

    def test_role_exchange_and_negative_control_contract(self) -> None:
        rows = read_jsonl(OUT / "phase425_registered_conditions_open.jsonl")
        by_pair_model = defaultdict(list)
        for row in rows:
            by_pair_model[(row["pair_id"], row["model"])].append(row)
            self.assertNotEqual(row["target_branch_token_id"], row["opposite_branch_token_id"])
            self.assertTrue(row["source_positions"])
            self.assertTrue(row["query_positions"])
            self.assertTrue(row["instruction_control_positions"])
        self.assertTrue(by_pair_model)
        for pair_rows in by_pair_model.values():
            self.assertEqual(len(pair_rows), 8)
            self.assertEqual(Counter(row["role"] for row in pair_rows), {"a": 4, "b": 4})
            self.assertEqual(Counter(row["interface"] for row in pair_rows), {"direct": 4, "result_field": 4})
            self.assertEqual(Counter(row["history"] for row in pair_rows), {"bare": 4, "worked_example": 4})
            targets = {row["target"] for row in pair_rows}
            if pair_rows[0]["candidate"]:
                self.assertEqual(len(targets), 2)
                self.assertTrue(all(row["role_changes_correct_event"] for row in pair_rows))
            else:
                self.assertEqual(len(targets), 1)
                self.assertFalse(any(row["role_changes_correct_event"] for row in pair_rows))

    def test_lexical_replica_is_the_split_unit(self) -> None:
        pairs = read_jsonl(OUT / "phase425_registered_pairs.jsonl")
        groups = defaultdict(list)
        for row in pairs:
            groups[row["replica_group_id"]].append(row)
        self.assertEqual(len(groups), 192)
        for rows in groups.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["lexical_replica"] for row in rows}, {0, 1})
            self.assertEqual(len({row["split"] for row in rows}), 1)

    def test_completed_outputs_remain_noncausal(self) -> None:
        summary_path = OUT / "phase425_global_summary.json"
        if not summary_path.exists():
            self.skipTest("Phase425 model collection has not completed")
        summary = read_json(summary_path)
        self.assertEqual(summary["strict_mechanism_closure"], "0/72")
        self.assertFalse(summary["causal_tested"])
        self.assertFalse(summary["sealed_unlock"])
        gate_freeze = read_json(OUT / "phase425_gate_freeze.json")
        self.assertFalse(gate_freeze["sealed_unlock"])
        self.assertFalse(gate_freeze["causal_unlock"])
        for model in MODELS:
            complete = read_json(
                OUT
                / "models"
                / model
                / "open"
                / "phase425_collection_complete.json"
            )
            self.assertTrue(complete["all_rows_complete"])
            self.assertTrue(complete["component_ledger_gate_pass"])
            self.assertEqual(complete["condition_count"], 2304)
            self.assertFalse(
                (
                    OUT
                    / "models"
                    / model
                    / "sealed"
                    / "phase425_collection_complete.json"
                ).exists()
            )
        manifest = read_json(VIS / "manifest.json")
        self.assertEqual(manifest["schema_version"], "phase425_role_exchange_manifest.v1")
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            graph = read_json(VIS / item["filename"])
            self.assertEqual(graph["schema_version"], "atlas_graph_v1")
            self.assertFalse(graph["graph"]["meta"]["causal"])
            self.assertTrue(all(not edge["compute_edge"] for edge in graph["graph"]["edges"]))


if __name__ == "__main__":
    unittest.main()
