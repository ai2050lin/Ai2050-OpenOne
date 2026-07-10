import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase330_component_candidate_scan as scan


class Phase330ComponentScanTest(unittest.TestCase):
    def test_path_registry_complete(self):
        rows = scan.path_registry("nine_family_global_atlas")
        self.assertEqual(len(rows), 648)
        self.assertFalse(any(row["heldout_used"] for row in rows))
        self.assertEqual({row["component_type"] for row in rows}, {"attention", "mlp", "residual"})


if __name__ == "__main__":
    unittest.main()
