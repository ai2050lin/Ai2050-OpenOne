#!/usr/bin/env python3

from __future__ import annotations

import gzip
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from typing import Iterator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase569_relation_competition_protocol as protocol  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat, tokenizer_for  # noqa: E402
from phase569_role_position_utils import ROLE_GROUPS, role_positions, typed_union  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_open_rows() -> Iterator[dict]:
    with gzip.open(protocol.OPEN_CASES_PATH, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class Phase569RelationCompetitionTests(unittest.TestCase):
    def test_static_denominator_and_seal(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_semantic_case_count"], 58752)
        self.assertEqual(audit["open_semantic_case_count"], 48384)
        self.assertEqual(audit["sealed_semantic_case_count"], 10368)
        self.assertEqual(audit["registered_model_evaluation_count"], 176256)
        self.assertEqual(audit["target_other_relation_collision_count"], 0)
        self.assertEqual(audit["triplet_invariant_failure_count"], 0)
        self.assertEqual(audit["phase567_568_object_overlap_count"], 0)
        self.assertEqual(commitment["sealed_semantic_case_count"], 10368)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_open_bank_has_fixed_noncolliding_semantics(self) -> None:
        count = 0
        split_counts: dict[str, int] = defaultdict(int)
        for row in iter_open_rows():
            count += 1
            split_counts[row["split"]] += 1
            self.assertFalse(row["sealed"])
            self.assertNotEqual(row["target"], row["other_relation_target"])
            self.assertEqual(row["query_object_count"], 3)
            self.assertEqual(row["balancing_object_index"], 3)
            self.assertLess(row["query_object_index"], row["balancing_object_index"])
            self.assertEqual(len(row["objects"]), 4)
            self.assertEqual(len(set(row["values"])), 4)
            for model_ids in row["candidate_token_ids_by_model"].values():
                self.assertEqual({len(ids) for ids in model_ids.values()}, {1})
                self.assertEqual(len({tuple(ids) for ids in model_ids.values()}), 4)
        self.assertEqual(count, 48384)
        self.assertEqual(
            split_counts,
            {
                "phenotype_discovery": 13824,
                "phenotype_confirmation": 13824,
                "path_discovery": 10368,
                "path_confirmation": 10368,
            },
        )

    def test_first_world_triplets_keep_other_relation_fixed(self) -> None:
        rows = []
        for row in iter_open_rows():
            if row["anchor_id"] == "phase569_phenotype_discovery_000":
                rows.append(row)
            elif rows:
                break
        self.assertEqual(len(rows), 108)
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row["triplet_id"]].append(row)
        self.assertEqual(len(groups), 36)
        for group in groups.values():
            ordered = sorted(group, key=lambda row: row["binding"])
            self.assertEqual(len(ordered), 3)
            self.assertEqual(len({row["target"] for row in ordered}), 3)
            self.assertEqual(len({row["other_relation_target"] for row in ordered}), 1)
            self.assertNotIn(
                ordered[0]["other_relation_target"],
                {row["target"] for row in ordered},
            )
            self.assertEqual(len({row["fact_token_multiset_key"] for row in ordered}), 1)
            self.assertEqual(len({row["prompt_token_multiset_key"] for row in ordered}), 1)

    def test_ten_role_groups_are_disjoint(self) -> None:
        sampled: dict[tuple[int, int, str], dict] = {}
        for row in iter_open_rows():
            if row["split"] != "phenotype_discovery":
                continue
            key = (row["surface_id"], row["fact_order"], row["query_relation"])
            sampled.setdefault(key, row)
            if len(sampled) == 12:
                break
        self.assertEqual(len(sampled), 12)
        for model in protocol.MODELS:
            tokenizer = tokenizer_for(model)
            for row in sampled.values():
                prompt = render_chat(tokenizer, model, row["raw_prompt"])
                ids, groups = role_positions(tokenizer, prompt, row)
                self.assertEqual(tuple(groups), ROLE_GROUPS)
                physical = typed_union(groups)
                self.assertEqual(len(physical), len(set(physical)))
                self.assertGreaterEqual(len(physical), 10)
                self.assertLess(max(physical), len(ids))

    def test_behavior_runner_cannot_read_sealed_rows(self) -> None:
        source = (ROOT / "tests/gpt5/phase569_relation_competition_behavior.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("phase569_sealed_cases", source)
        self.assertIn("phase569_open_cases.jsonl.gz", source)
        self.assertIn('PHASE_ID = "Phase569"', source)

    def test_completed_behavior_and_trace_results(self) -> None:
        behavior = read_json(protocol.OUT_DIR / "phase569_behavior_summary.json")
        self.assertEqual(
            behavior["authorized_models_for_coarse_internal_trace"],
            ["qwen3", "glm4", "deepseek7b"],
        )
        expected_cells = {
            "qwen3": (60, 17),
            "glm4": (19, 14),
            "deepseek7b": (8, 32),
        }
        for report in behavior["model_reports"]:
            self.assertEqual(
                (
                    report["stable_correct_cell_count"],
                    report["stable_relation_confusion_cell_count"],
                ),
                expected_cells[report["model"]],
            )
            self.assertTrue(report["authorized_for_coarse_internal_trace"])
            self.assertFalse(report["sealed_split_read"])
        trace = read_json(protocol.OUT_DIR / "phase569_coarse_trace_analysis.json")
        self.assertEqual(trace["cross_model_shared_topology_count"], 13)
        shared = trace["cross_model_shared_topology"][0]
        self.assertEqual(shared["component"], "attention_output")
        self.assertEqual(shared["semantic_role"], "answer_boundary")
        self.assertEqual(shared["depth_band_8"], 6)
        self.assertEqual(shared["model_count"], 3)
        self.assertFalse(trace["causal_intervention_executed"])
        self.assertFalse(trace["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
