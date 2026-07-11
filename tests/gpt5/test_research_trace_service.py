from __future__ import annotations

import unittest

from pydantic import ValidationError

from server.research_trace_service import ResearchTraceManager, TraceRunRequest


class ResearchTraceServiceTests(unittest.TestCase):
    def test_frozen_runs_are_validated_replay_sources(self) -> None:
        rows = ResearchTraceManager.frozen_runs()
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["source_mode"] == "replay" for row in rows))
        self.assertTrue(all(row["validated"] for row in rows))

    def test_request_rejects_unknown_target(self) -> None:
        with self.assertRaises(ValidationError):
            TraceRunRequest(model="qwen3", prompt="test", target_label="not-a-color")

    def test_command_uses_generated_run_id_and_no_shell(self) -> None:
        job = {
            "run_id": "live_qwen3_red_test",
            "model": "qwen3",
            "prompt": "A red cube is",
            "target_label": "red",
            "top_k": 16,
        }
        command = ResearchTraceManager.command_for(job)
        self.assertIn("live_qwen3_red_test", command)
        self.assertIn("A red cube is", command)
        self.assertIn("--skip-public-copy", command)
        self.assertNotIn("cmd", command)


if __name__ == "__main__":
    unittest.main()
