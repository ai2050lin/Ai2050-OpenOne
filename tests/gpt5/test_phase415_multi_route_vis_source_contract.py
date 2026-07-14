#!/usr/bin/env python3
"""Contract tests for the Phase415 multi-route visualization data sources."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "frontend/public/vis_data/source_registry.json"
RESULT = (
    ROOT
    / "tests/gpt5/result/phase415_multi_route_vis_sources"
    / "phase415_multi_route_vis_source_contract.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase415MultiRouteVisSourceContractTest(unittest.TestCase):
    def test_registry_exposes_gpt5_and_glm5_routes(self) -> None:
        registry = read_json(REGISTRY)
        sources = registry["sources"]
        self.assertEqual(registry["schema_version"], "vis_data_source_registry.v1")
        self.assertEqual({source["route_id"] for source in sources}, {"gpt5", "glm5"})
        self.assertGreaterEqual(len(sources), 4)
        self.assertIn(
            registry["default_source_id"],
            {source["id"] for source in sources},
        )

    def test_every_registered_manifest_exists(self) -> None:
        registry = read_json(REGISTRY)
        for source in registry["sources"]:
            manifest = ROOT / "frontend/public" / source["manifest_path"].lstrip("/")
            self.assertTrue(manifest.is_file(), source["id"])
            self.assertEqual(
                read_json(manifest)["schema_version"],
                source["manifest_schema"],
            )

    def test_full_payload_contract_passed_without_causal_promotion(self) -> None:
        result = read_json(RESULT)
        registry = read_json(REGISTRY)
        self.assertTrue(result["valid"])
        self.assertEqual(result["route_count"], 2)
        self.assertEqual(result["source_count"], len(registry["sources"]))
        self.assertGreaterEqual(result["dataset_count"], 100)
        self.assertGreater(result["canonical_node_count"], 0)
        self.assertGreater(result["canonical_edge_count"], 0)
        self.assertTrue(
            all(row["all_payloads_renderable"] for row in result["source_results"])
        )
        dynamic_rows = [
            row
            for row in result["source_results"]
            if row["source_id"]
            in {"gpt5_real_component_trace", "gpt5_mechanism_trace"}
        ]
        self.assertEqual(len(dynamic_rows), 2)
        self.assertTrue(
            all(row["explicitly_noncausal_adapted_edge_count"] > 0 for row in dynamic_rows)
        )

    def test_client_no_longer_truncates_manifest_to_five_files(self) -> None:
        client = (ROOT / "frontend/src/neural_vis/index.jsx").read_text(
            encoding="utf-8"
        )
        main_client = (ROOT / "frontend/src/App.jsx").read_text(encoding="utf-8")
        hook = (ROOT / "frontend/src/neural_vis/hooks/useVisData.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('aria-label="测试路线数据源"', client)
        self.assertIn('data-testid="route-dataset-list"', client)
        self.assertNotIn("dataFiles.slice(0, 5)", client)
        self.assertIn("source_registry.json", hook)
        self.assertIn("selectDataSource", hook)
        self.assertIn('aria-label="主工作台测试路线数据源"', main_client)
        self.assertIn('aria-label="主工作台测试数据集"', main_client)


if __name__ == "__main__":
    unittest.main()
