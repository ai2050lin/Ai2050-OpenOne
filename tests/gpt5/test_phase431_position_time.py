#!/usr/bin/env python3
"""Protocol and evidence-boundary tests for the Phase431 position-time atlas."""

from __future__ import annotations

import gzip
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase431_position_time_analysis import sanitize_blind_row  # noqa: E402
from phase431_position_time_protocol import (  # noqa: E402
    BLOCKS,
    GROUPS_PER_BLOCK_SPLIT,
    OPEN_SPLITS,
    OUT,
    ROUTE_MODES,
    ROLES,
    SEALED_SPLIT,
    build_groups,
    digest_rows,
    freeze,
    validate_groups,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase431ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = freeze()
        cls.open_rows, cls.sealed_rows = build_groups()

    def test_frozen_denominator_is_complete_and_disjoint(self) -> None:
        audit = validate_groups(self.open_rows, self.sealed_rows)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["open_group_count"], 576)
        self.assertEqual(audit["sealed_group_count"], 192)
        self.assertEqual(audit["open_condition_count"], 5760)
        self.assertEqual(audit["sealed_condition_count"], 1920)
        self.assertEqual(
            set(row["semantic_group_id"] for row in self.open_rows).intersection(
                row["semantic_group_id"] for row in self.sealed_rows
            ),
            set(),
        )
        self.assertEqual(
            digest_rows(self.open_rows), self.protocol["open_group_rows_sha256"]
        )
        self.assertEqual(
            digest_rows(self.sealed_rows),
            self.protocol["sealed_commitment"]["sealed_group_rows_sha256"],
        )

    def test_each_block_and_split_has_same_independent_group_count(self) -> None:
        counts = Counter(
            (row["block_id"], row["split"])
            for row in (*self.open_rows, *self.sealed_rows)
        )
        self.assertEqual(len(counts), len(BLOCKS) * (len(OPEN_SPLITS) + 1))
        self.assertEqual(set(counts.values()), {GROUPS_PER_BLOCK_SPLIT})
        self.assertEqual({row["split"] for row in self.sealed_rows}, {SEALED_SPLIT})

    def test_candidate_source_choice_is_balanced_without_metadata_oracle(self) -> None:
        for split in (*OPEN_SPLITS, SEALED_SPLIT):
            groups = [
                row
                for row in (*self.open_rows, *self.sealed_rows)
                if row["split"] == split and row["candidate"]
            ]
            targets = Counter(
                "source_1"
                if group["role_targets"][role] == group["source_1"]
                else "source_2"
                for group in groups
                for role in ROLES
                for route in ROUTE_MODES
                if route in {"source_only", "query_only", "consistent"}
            )
            self.assertEqual(targets["source_1"], targets["source_2"])
        forbidden = set(self.protocol["baseline_contract"]["forbidden_oracle_fields"])
        self.assertTrue(
            {"role", "active_selector_identity", "role_mapping_variant"}.issubset(
                forbidden
            )
        )
        self.assertEqual(
            self.protocol["baseline_contract"]["primary"],
            "balanced 50/50 source-choice prior on eligible conditions",
        )

    def test_blind_projection_removes_behavior_and_signed_answer_fields(self) -> None:
        row = {
            "condition_id": "secret-condition",
            "anonymous_context_id": "context",
            "position_metrics": {
                "question_end": {
                    "residual_post_rms": 1.0,
                    "source_1_minus_source_2_margin": 3.0,
                    "attention_source_margin_write": 2.0,
                    "mlp_source_margin_write": 1.0,
                }
            },
            "receiver_metrics": {
                "question_end": {
                    "attention_replay_relative_error": 0.0,
                    "source_partition": {
                        "source_1": {"write_norm": 2.0, "source_margin_write": 4.0},
                        "source_2": {"write_norm": 1.0, "source_margin_write": -1.0},
                    },
                }
            },
            "layer": 3,
            "relative_depth": 0.25,
            "actual_choice": "source_1",
            "normative_target": True,
            "candidate": True,
            "role": "a",
        }
        sanitized = sanitize_blind_row(row)
        serialized = json.dumps(sanitized, sort_keys=True)
        for forbidden in (
            "actual_choice",
            "normative_target",
            "candidate\"",
            "source_1_minus_source_2_margin",
            "source_margin_write",
            "attention_source_margin_write",
            "mlp_source_margin_write",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(sanitized["behavior_labels_visible"])
        self.assertFalse(sanitized["candidate_label_visible"])

    def test_protocol_never_claims_causal_or_neuron_evidence(self) -> None:
        evidence = self.protocol["evidence_contract"]
        record = self.protocol["record_contract"]
        self.assertFalse(evidence["causal"])
        self.assertFalse(evidence["single_neuron"])
        self.assertFalse(record["head_channel_neuron_scan"])
        self.assertFalse(record["intervention"])
        self.assertFalse(self.protocol["language_candidate"]["cross_model"])
        self.assertTrue(
            self.protocol["sealed_commitment"]["open_pipeline_must_not_import_sealed_file"]
        )
        self.assertEqual(
            self.protocol["execution_dtypes"],
            {"qwen3": "float16", "glm4": "bfloat16", "deepseek7b": "bfloat16"},
        )

    def test_generated_outputs_keep_evidence_boundary_when_present(self) -> None:
        open_gate_path = OUT / "phase431_open_gate.json"
        if open_gate_path.exists():
            gate = read_json(open_gate_path)
            self.assertFalse(gate["old_phase429_sealed_used"])
            self.assertFalse(gate["causal"])
            self.assertFalse(gate["single_neuron"])
        visual_path = ROOT / "frontend/public/vis_data/phase431_position_time/manifest.json"
        if visual_path.exists():
            manifest = read_json(visual_path)
            payload = read_json(visual_path.parent / manifest["items"][0]["filename"])
            self.assertTrue(all(not row["causal"] for row in payload["graph"]["nodes"]))
            self.assertTrue(
                all(not row["single_neuron"] for row in payload["graph"]["nodes"])
            )


if __name__ == "__main__":
    unittest.main()
