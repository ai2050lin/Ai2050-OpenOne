from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase508_native_contract_stage_audit"
PUBLIC = ROOT / "frontend/public/vis_data/phase508_native_relation_contract_atlas"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase508NativeContractStageAuditTest(unittest.TestCase):
    def test_frozen_stage_decision(self) -> None:
        audit = read_json(OUT / "phase508_native_contract_stage_audit.json")
        self.assertEqual(audit["status"], "stage_complete_gate_stopped")
        self.assertEqual(audit["denominators"]["stage_a_model_rows"], 4_320)
        self.assertEqual(audit["denominators"]["stage_b_model_rows"], 5_184)
        self.assertEqual(audit["denominators"]["stage_c_model_rows"], 0)
        self.assertEqual(audit["denominators"]["physical_rows"], 0)
        self.assertEqual(len(audit["stage_a_shared_cells"]), 2)
        self.assertEqual(audit["stage_b_shared_contracts"], [])
        self.assertTrue(audit["gates"]["shared_stage_a_cell_found"])
        self.assertFalse(audit["gates"]["complete_native_contract_found"])
        self.assertFalse(audit["gates"]["independent_confirmation_authorized"])
        self.assertFalse(audit["gates"]["physical_authorized"])
        self.assertFalse(audit["evidence_boundary"]["sealed_split_read"])
        self.assertEqual(audit["denominators"]["strict_closed_mechanisms"], 0)
        self.assertEqual(audit["denominators"]["mechanism_denominator"], 72)

    def test_model_specific_stage_a_counts(self) -> None:
        audit = read_json(OUT / "phase508_native_contract_stage_audit.json")
        expected = {"qwen3": 2, "glm4": 4, "deepseek7b": 0}
        for model, count in expected.items():
            self.assertEqual(
                len(audit["stage_a"][model]["passed_function_polarity_cells"]),
                count,
            )
            self.assertEqual(audit["stage_b"][model]["passed_native_contracts"], [])

    def test_atlas_has_no_physical_or_causal_claims(self) -> None:
        manifest = read_json(PUBLIC / "manifest.json")
        self.assertEqual(
            manifest["schema_version"],
            "phase508_native_relation_contract_atlas_manifest.v1",
        )
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(PUBLIC / item["path"])
            nodes = payload["graph"]["nodes"]
            edges = payload["graph"]["edges"]
            self.assertGreater(len(nodes), 0)
            self.assertTrue(all(not node.get("physical", False) for node in nodes))
            self.assertTrue(all(not node.get("causal", False) for node in nodes))
            self.assertTrue(all(not node.get("single_neuron", False) for node in nodes))
            self.assertTrue(all(not edge.get("compute_edge", False) for edge in edges))
            self.assertTrue(all(not edge.get("causal", False) for edge in edges))

    def test_registry_entry(self) -> None:
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        sources = {source["id"]: source for source in registry["sources"]}
        source = sources["gpt5_phase508_native_relation_contract_atlas"]
        self.assertEqual(source["route_id"], "gpt5")
        self.assertEqual(source["models"], list(MODELS))
        self.assertEqual(
            source["manifest_path"],
            "/vis_data/phase508_native_relation_contract_atlas/manifest.json",
        )


if __name__ == "__main__":
    unittest.main()
