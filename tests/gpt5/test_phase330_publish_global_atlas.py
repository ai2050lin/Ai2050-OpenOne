import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase330_publish_global_atlas as publisher


class Phase330PublisherTest(unittest.TestCase):
    def test_publish_contract(self):
        result = publisher.build()
        validation = publisher.validate()
        self.assertEqual(result["phase"], 330)
        self.assertEqual(validation["partition_count"], 27)
        self.assertEqual(validation["phase330_component_members"], 864)
        self.assertEqual(validation["single_unit_causal"], 0)


if __name__ == "__main__":
    unittest.main()
