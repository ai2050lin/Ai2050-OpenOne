from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class Phase539To542Test(unittest.TestCase):
    def test_phase539_exploration_is_not_physical_authorization(self) -> None:
        root = ROOT / "tests/gpt5/result/phase539_pair_answer_logodds_observer"
        summaries = {model: read_json(root / f"phase539_{model}_summary.json") for model in ("qwen3", "glm4", "deepseek7b")}
        self.assertTrue(summaries["qwen3"]["exploratory_all_open_pass"])
        self.assertFalse(summaries["glm4"]["exploratory_all_open_pass"])
        self.assertFalse(summaries["deepseek7b"]["exploratory_all_open_pass"])
        self.assertTrue(all(not summary["physical_authorized"] for summary in summaries.values()))

    def test_phase540_fresh_protocol_is_balanced_and_disjoint(self) -> None:
        audit = read_json(ROOT / "tests/gpt5/result/phase540_pair_answer_logodds_fresh_protocol/phase540_static_audit.json")
        self.assertEqual(audit["status"], "static_pass_no_model_run")
        self.assertEqual(audit["open_row_count"], 4096)
        self.assertEqual(audit["sealed_row_count"], 2048)
        self.assertTrue(audit["fresh_entity_pool_disjoint_pass"])
        self.assertTrue(audit["fresh_relation_pool_disjoint_pass"])
        self.assertTrue(audit["phase535_open_vocabulary_disjoint_pass"])
        self.assertFalse(audit["phase535_sealed_split_read_by_current_protocol"])
        self.assertTrue(audit["historical_phase535_sealed_compromised_by_initial_phase540_audit"])
        for report in audit["splits"].values():
            self.assertTrue(report["pair_status_flip_pass"])
            self.assertTrue(report["slot_label_balance_pass"])
            self.assertTrue(report["matched_fact_token_bag_pass"])

    def test_phase541_refutes_portable_fixed_answer_observer(self) -> None:
        root = ROOT / "tests/gpt5/result/phase541_pair_answer_logodds_fresh_confirmation"
        qwen = read_json(root / "phase541_qwen3_summary.json")
        self.assertEqual(qwen["status"], "fresh_confirmation_failed")
        self.assertFalse(qwen["threshold_refit"])
        self.assertFalse(qwen["all_open_confirmation_pass"])
        self.assertFalse(qwen["split_reports"]["fresh_vocabulary_confirmation"]["gate_pass"])
        self.assertFalse(qwen["split_reports"]["fresh_relation_confirmation"]["gate_pass"])
        for model in ("glm4", "deepseek7b"):
            summary = read_json(root / f"phase541_{model}_summary.json")
            self.assertEqual(summary["status"], "not_phase539_candidate")
            self.assertFalse(summary["model_weights_loaded"])

    def test_no_physical_or_sealed_authorization(self) -> None:
        auth = read_json(ROOT / "tests/gpt5/result/phase541_pair_answer_logodds_fresh_confirmation/phase541_physical_collection_authorization.json")
        self.assertEqual(auth["physical_collection_authorized_models"], [])
        self.assertFalse(auth["causal_authorized"])
        self.assertFalse(auth["sealed_split_read"])

    def test_stage_audit_keeps_progress_and_evidence_boundary(self) -> None:
        audit = read_json(ROOT / "tests/gpt5/result/phase542_pair_answer_logodds_stage_audit/phase542_pair_answer_logodds_stage_audit.json")
        self.assertEqual(audit["status"], "complete_fresh_confirmation_failed_historical_seal_contamination_recorded")
        self.assertFalse(audit["stage_findings"]["fixed_answer_logodds_is_portable_pair_state"])
        self.assertEqual(audit["progress"]["strict_closed_mechanisms"], 0)
        self.assertEqual(audit["progress"]["global_physical_atlas_percent"], 31)
        self.assertFalse(audit["evidence_boundary"]["physical"])
        self.assertFalse(audit["evidence_boundary"]["causal"])
        self.assertTrue(audit["evidence_boundary"]["historical_phase535_sealed_read"])
        self.assertFalse(audit["evidence_boundary"]["current_phase540_sealed_read"])

    def test_historical_seal_incident_is_recorded_and_corrected(self) -> None:
        audit = read_json(ROOT / "tests/gpt5/result/phase543_seal_contamination_audit/phase543_seal_contamination_audit.json")
        self.assertEqual(audit["status"], "historical_contamination_recorded_and_current_allowlist_verified")
        self.assertEqual(audit["incident"]["contaminated_split"], "phase535.sealed")
        self.assertTrue(audit["evidence_boundary"]["global_any_sealed_split_read"])
        self.assertFalse(audit["evidence_boundary"]["current_phase540_sealed_read"])

    def test_atlas_graph_endpoints_and_registry(self) -> None:
        atlas = ROOT / "frontend/public/vis_data/phase541_pair_answer_logodds_atlas"
        manifest = read_json(atlas / "manifest.json")
        self.assertEqual(len(manifest["items"]), 3)
        for item in manifest["items"]:
            payload = read_json(atlas / item["path"])
            graph = payload["graph"]
            ids = {node["id"] for node in graph["nodes"]}
            self.assertTrue(ids)
            self.assertTrue(all(edge["source"] in ids and edge["target"] in ids for edge in graph["edges"]))
            self.assertTrue(all(not node["physical"] and not node["causal"] for node in graph["nodes"]))
        registry = read_json(ROOT / "frontend/public/vis_data/source_registry.json")
        self.assertIn("gpt5_phase541_pair_answer_logodds_atlas", {source["id"] for source in registry["sources"]})


if __name__ == "__main__":
    unittest.main()
