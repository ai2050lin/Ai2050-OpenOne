from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from server.research_orchestrator.runtime import ResearchRun, artifact_audit, validate_generated_code


class ResearchOrchestratorTests(unittest.TestCase):
    def test_accepts_plain_research_code(self) -> None:
        result = validate_generated_code("import json\nprint(json.dumps({'ok': True}))\n")
        self.assertTrue(result["ok"])

    def test_rejects_process_spawning(self) -> None:
        result = validate_generated_code("import subprocess\nsubprocess.run(['python', 'x.py'])\n")
        self.assertFalse(result["ok"])
        self.assertIn("blocked call: subprocess.run", result["errors"])

    def test_missing_artifacts_remain_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = ResearchRun("test_run", root, "test")
            run.experiment_path.write_text(
                '{"required_outputs":["cases.jsonl","manifest.json"]}',
                encoding="utf-8",
            )
            run.execution_path.write_text('{"status":"success"}', encoding="utf-8")
            audit = artifact_audit(run)
            self.assertEqual(audit["decision"], "inconclusive")
            self.assertEqual(audit["missing_required_artifacts"], ["cases.jsonl", "manifest.json"])


if __name__ == "__main__":
    unittest.main()
