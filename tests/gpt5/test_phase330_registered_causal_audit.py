import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase330_registered_causal_audit as audit


class Phase330CausalAuditTest(unittest.TestCase):
    def test_registration(self):
        rows = audit.register_cases("nine_family_global_atlas")
        self.assertEqual(len(rows), 432)
        self.assertEqual({row["item_index"] for row in rows}, {18, 23})
        self.assertEqual(len(audit.CONDITIONS), 10)


if __name__ == "__main__":
    unittest.main()
