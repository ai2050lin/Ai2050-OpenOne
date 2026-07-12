from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase359_storage_budget import model_budget  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase359_full_vector_anchor"


class Phase359FullVectorAnchorTests(unittest.TestCase):
    def test_storage_budget_rejects_naive_trace_and_accepts_multiresolution(self) -> None:
        config = {
            "num_hidden_layers": 36,
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "num_attention_heads": 32,
        }
        result = model_budget("qwen3", config)
        self.assertGreater(
            result["naive_full_trace"]["bytes_all_cases"],
            result["recommended_multiresolution"]["total_planned_bytes"],
        )
        self.assertGreater(result["recommended_multiresolution"]["r2_one_full_anchor_bytes"], 0)

    def test_replay_summary_if_present(self) -> None:
        path = OUT / "phase359_replay_summary.json"
        if not path.exists():
            self.skipTest("Phase359 has not run")
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["model_count"], 3)
        self.assertEqual(summary["anchor_count"], 3)
        self.assertTrue(summary["all_models_replay_pass"])
        self.assertFalse(summary["evidence_boundary"]["blind_motif_discovery_completed"])
        self.assertFalse(summary["evidence_boundary"]["language_encoding_closed"])


if __name__ == "__main__":
    unittest.main()
