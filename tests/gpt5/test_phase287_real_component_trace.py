from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.research_kernel.store import ResearchEvidenceStore  # noqa: E402


class RealComponentTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ResearchEvidenceStore(ROOT)

    def test_three_models_have_full_component_traces(self) -> None:
        runs = [row for row in self.store.list_runs() if row.get("phase") == 287]
        self.assertEqual({row["model"] for row in runs}, {"qwen3", "glm4", "deepseek7b"})
        self.assertTrue(all(row["trace_event_count"] > 300 for row in runs))

    def test_required_component_events_exist(self) -> None:
        required = {"norm1", "q_projection", "k_projection", "v_projection", "attention_output", "norm2", "mlp_product", "mlp_down", "residual2", "unembedding_readout"}
        for model, run_id in {
            "qwen3": "phase287_qwen3_red_component_trace",
            "glm4": "phase287_glm4_red_component_trace",
            "deepseek7b": "phase287_deepseek7b_red_component_trace",
        }.items():
            rows = self.store.run_artifact(run_id, "trace_events")
            event_types = {row["event_type"] for row in rows}
            self.assertTrue(required.issubset(event_types), (model, sorted(required - event_types)))

    def test_full_vector_archive_is_registered(self) -> None:
        for run in [row for row in self.store.list_runs() if row.get("phase") == 287]:
            manifest = self.store.run_manifest(run["run_id"])
            archive = manifest["artifacts"]["full_vectors"]
            self.assertGreater(archive["rows"], 300)
            self.assertEqual(len(archive["sha256"]), 64)

    def test_deepseek_candidate_and_global_output_are_distinct(self) -> None:
        run = next(row for row in self.store.list_runs() if row.get("run_id") == "phase287_deepseek7b_red_component_trace")
        self.assertEqual(run["target_color_rank"], 1)
        self.assertNotEqual(run["next_token"]["token"].strip(), "red")


if __name__ == "__main__":
    unittest.main()
