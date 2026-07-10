from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from server.research_kernel.store import ResearchEvidenceStore  # noqa: E402


class ResearchKernelBundleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.store = ResearchEvidenceStore(ROOT)

    def test_manifest_has_three_real_model_runs(self) -> None:
        runs = self.store.list_runs()
        self.assertEqual({row["model"] for row in runs}, {"qwen3", "glm4", "deepseek7b"})
        self.assertTrue(all(row["unit_count"] > 0 for row in runs))

    def test_every_bundle_validates(self) -> None:
        for run in self.store.list_runs():
            result = self.store.validate_run(run["run_id"])
            self.assertTrue(result["valid"], result["issues"][:5])

    def test_real_units_are_not_promoted_to_single_unit_causality(self) -> None:
        for run in [row for row in self.store.list_runs() if row.get("phase") == 286]:
            rows = self.store.run_artifact(run["run_id"], "unit_evidence")
            self.assertTrue(rows)
            self.assertTrue(all(row["unit_kind"] == "mlp_product_neuron" for row in rows))
            self.assertTrue(all(row["evidence_level"] == "L4" for row in rows))
            self.assertFalse(any(row.get("causal_scope") == "single_unit" for row in rows))

    def test_progress_uses_explicit_denominators(self) -> None:
        dimensions = self.store.manifest()["progress"]["dimensions"]
        for item in dimensions.values():
            self.assertIn("valid", item)
            self.assertIn("required", item)
            self.assertIn("ratio", item)


if __name__ == "__main__":
    unittest.main()
