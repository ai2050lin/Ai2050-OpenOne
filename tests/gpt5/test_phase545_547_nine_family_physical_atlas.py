#!/usr/bin/env python3
"""Cross-stage contracts for Phase545-547 physical prediction and publishing."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
P545 = ROOT / "tests/gpt5/result/phase545_natural_entry_physical_path"
P546 = ROOT / "tests/gpt5/result/phase546_upstream_physical_prediction"
P547 = ROOT / "tests/gpt5/result/phase547_nine_family_natural_atlas"
ATLAS = ROOT / "frontend/public/vis_data/phase546_nine_family_natural_atlas"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Phase545TerminalAuditTest(unittest.TestCase):
    def test_nominal_predictions_are_all_terminal_identity_events(self) -> None:
        summary = read_json(P545 / "phase545_global_summary.json")
        self.assertEqual(summary["results"]["physical_prediction_pass_cells"], 7)
        self.assertEqual(summary["results"]["terminal_identity_prediction_cells"], 7)
        self.assertEqual(summary["results"]["upstream_route_eligible_cells"], 0)
        events = read_jsonl(P545 / "phase545_model_mechanism_events.jsonl")
        self.assertTrue(all(
            not row["physical_prediction_pass"] or row["terminal_identity_event"]
            for row in events
        ))


class Phase546FreshPredictionTest(unittest.TestCase):
    def test_fresh_physical_holdout_is_disjoint_and_complete(self) -> None:
        audit = read_json(P546 / "phase546_static_audit.json")
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_confirmation_pair_count"], 441)
        self.assertEqual(audit["pair_count_per_model_mechanism"], [49])
        self.assertEqual(audit["confirmation_pair_index_range"], [24, 72])
        self.assertEqual(audit["phase545_physical_pair_overlap_count"], 0)
        self.assertEqual(audit["generated_stage_event_count"], 0)
        self.assertEqual(audit["layer_zero_input_event_count"], 0)
        self.assertEqual(audit["sealed_pair_count"], 0)

    def test_protocol_and_collection_hashes_are_frozen(self) -> None:
        protocol = read_json(P546 / "phase546_upstream_protocol.json")
        self.assertEqual(protocol["models_in_required_order"], ["qwen3", "glm4", "deepseek7b"])
        self.assertEqual(protocol["registered_pairs_sha256"], sha256_file(
            P546 / "phase546_registered_confirmation_pairs.jsonl"
        ))
        self.assertEqual(protocol["frozen_events_sha256"], sha256_file(
            P546 / "phase546_frozen_upstream_events.jsonl"
        ))
        self.assertFalse(protocol["claim_boundaries"]["upstream_observer_is_compute_edge"])
        self.assertFalse(protocol["claim_boundaries"]["sealed_split_read"])

        expected = {
            "qwen3": (196, 7056, True),
            "glm4": (245, 9800, True),
            "deepseek7b": (0, 0, False),
        }
        for model, (pairs, rows, cuda_loaded) in expected.items():
            summary = read_json(P546 / f"phase546_{model}_collection_summary.json")
            self.assertEqual(summary["registered_pair_count"], pairs)
            self.assertEqual(summary["pair_layer_row_count"], rows)
            self.assertEqual(summary["cuda_loaded"], cuda_loaded)
            self.assertFalse(summary["sealed_split_read"])
            if rows:
                self.assertEqual(summary["output_sha256"], sha256_file(
                    P546 / f"phase546_{model}_pair_layer_rows.jsonl"
                ))

    def test_upstream_prediction_boundary(self) -> None:
        summary = read_json(P546 / "phase546_global_summary.json")
        self.assertEqual(summary["results"]["upstream_prediction_pass_cells"], 7)
        self.assertEqual(summary["results"]["cross_model_shared_upstream_topologies"], 2)
        self.assertEqual(summary["results"]["compute_edges"], 0)
        self.assertEqual(summary["results"]["causal_paths"], 0)
        self.assertEqual(summary["results"]["strict_closed_mechanisms"], 0)
        rows = read_jsonl(P546 / "phase546_upstream_prediction_results.jsonl")
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row["fresh_confirmation_pair_count"] == 49 for row in rows))
        self.assertTrue(all(row["frozen_discovery_event"]["stage"] == "prompt_end" for row in rows))
        self.assertTrue(all(not row["compute_edge"] and not row["causal"] for row in rows))


class Phase547AtlasTest(unittest.TestCase):
    def test_stage_success_does_not_promote_causality(self) -> None:
        audit = read_json(P547 / "phase547_stage_audit.json")
        self.assertTrue(all(audit["stage_success"].values()))
        self.assertFalse(audit["evidence_boundary"]["upstream_predictions_are_compute_edges"])
        self.assertFalse(audit["evidence_boundary"]["upstream_predictions_are_causal"])
        self.assertEqual(audit["progress"]["strict_closed_mechanisms"], 0)
        self.assertEqual(audit["progress"]["global_physical_atlas_percent"], 32.0)

    def test_client_source_and_three_model_graphs(self) -> None:
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        source = next(
            row for row in registry["sources"]
            if row["id"] == "gpt5_phase546_nine_family_natural_atlas"
        )
        self.assertEqual(source["models"], ["qwen3", "glm4", "deepseek7b"])
        manifest = read_json(ATLAS / "manifest.json")
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(ATLAS / item["path"])
            nodes = payload["graph"]["nodes"]
            edges = payload["graph"]["edges"]
            node_ids = {node["id"] for node in nodes}
            self.assertEqual(sum(node["type"].startswith("natural_behavior") for node in nodes), 18)
            self.assertTrue(all(
                -12 <= node["position"][0] <= 12
                and 0 <= node["position"][1] <= 58
                and -12 <= node["position"][2] <= 12
                for node in nodes
            ))
            self.assertTrue(all(edge["source"] in node_ids and edge["target"] in node_ids for edge in edges))
            self.assertTrue(all(not edge["causal"] and not edge["compute_edge"] for edge in edges))
            self.assertTrue(all(not node["causal"] and not node["compute_edge"] for node in nodes))


if __name__ == "__main__":
    unittest.main()
