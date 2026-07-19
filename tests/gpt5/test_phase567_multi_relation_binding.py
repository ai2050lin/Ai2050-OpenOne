#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase567_multi_relation_binding_protocol as protocol  # noqa: E402
from phase548_shared_attention_compute_protocol import tokenizer_for  # noqa: E402
from phase557_natural_color_source_intervention import word_token_ids  # noqa: E402
from phase568_role_position_utils import ROLE_GROUPS, role_positions, typed_union  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase567MultiRelationBindingTests(unittest.TestCase):
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
        self.assertEqual(audit["rows_per_counterfactual_triplet"], [3])
        self.assertEqual(commitment["sealed_case_count"], 15552)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_open_bank_contains_no_sealed_rows(self) -> None:
        self.assertEqual(len(self.open_rows), 67392)
        self.assertFalse(any(row["sealed"] for row in self.open_rows))
        self.assertEqual({row["split"] for row in self.open_rows}, set(protocol.OPEN_SPLITS))

    def test_triplet_changes_only_queried_relation_binding(self) -> None:
        triplet_id = (
            "phase567_behavior_discovery_000_query0_relationsurface_surface0_order0"
        )
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
        self.assertEqual(
            len({tuple(row["relation_maps"]["marker"]) for row in ordered}), 1
        )
        self.assertEqual(
            len({tuple(row["relation_maps"]["surface"]) for row in ordered}), 3
        )

    def test_world_covers_full_factorial(self) -> None:
        rows = [
            row for row in self.open_rows
            if row["model"] == "qwen3"
            and row["anchor_id"] == "phase567_behavior_discovery_000"
        ]
        self.assertEqual(len(rows), 108)
        self.assertEqual({row["factorial_cell"] for row in rows}, set(protocol.CELLS))
        self.assertEqual({row["binding"] for row in rows}, set(protocol.BINDINGS))
        self.assertEqual({row["query_object_index"] for row in rows}, set(protocol.QUERY_OBJECTS))
        self.assertEqual({row["query_relation"] for row in rows}, set(protocol.QUERY_RELATIONS))
        self.assertEqual({row["surface_id"] for row in rows}, set(protocol.SURFACES))
        self.assertEqual({row["fact_order"] for row in rows}, set(protocol.FACT_ORDERS))

    def test_splits_have_disjoint_object_lexicons(self) -> None:
        objects = {
            split: {
                value
                for row in self.open_rows if row["model"] == "qwen3" and row["split"] == split
                for value in row["objects"]
            }
            for split in protocol.OPEN_SPLITS
        }
        for index, left in enumerate(protocol.OPEN_SPLITS):
            for right in protocol.OPEN_SPLITS[index + 1:]:
                self.assertFalse(objects[left] & objects[right])

    def test_protocol_corrects_phase565_causal_labels(self) -> None:
        frozen = read_json(protocol.PROTOCOL_PATH)
        policy = frozen["evidence_policy"]
        self.assertTrue(policy["matched_wrong_state_is_counterfactual_sensitivity_not_natural_necessity"])
        self.assertTrue(policy["same_layer_identity_write_is_not_delete_restore"])
        self.assertTrue(policy["true_restore_requires_upstream_damage_and_later_layer_restore"])
        self.assertFalse(policy["fine_scan_before_replicated_coarse_role_edge"])
        self.assertFalse(policy["single_neuron_scan_before_compute_edge"])
        self.assertFalse(policy["sealed_split_read"])

    def test_eight_role_groups_map_to_disjoint_coordinates(self) -> None:
        for model in protocol.MODELS:
            tokenizer = tokenizer_for(model)
            sampled = {}
            for row in self.open_rows:
                if row["model"] != model or row["split"] != "behavior_discovery":
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

    def test_internal_single_token_worlds_have_disjoint_score_tokens(self) -> None:
        for model in protocol.MODELS:
            tokenizer = tokenizer_for(model)
            triples = {
                tuple(row["values"])
                for row in self.open_rows
                if row["model"] == model
                and row["split"] in {"role_discovery", "role_confirmation"}
                and all(
                    len(tokenizer(value, add_special_tokens=False)["input_ids"]) == 1
                    for value in row["values"]
                )
            }
            self.assertTrue(triples)
            for values in triples:
                token_sets = [set(word_token_ids(tokenizer, value)) for value in values]
                for index, left in enumerate(token_sets):
                    for right in token_sets[index + 1:]:
                        self.assertFalse(left & right)

    def test_behavior_script_never_reads_private_seal(self) -> None:
        source = (ROOT / "tests/gpt5/phase567_multi_relation_binding_behavior.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase567_sealed_cases", source)
        self.assertIn("Phase567 behavior requires CUDA", source)
        self.assertIn("torch.bfloat16", source)


if __name__ == "__main__":
    unittest.main()
