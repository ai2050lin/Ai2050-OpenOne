import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase329_publish_full_vocabulary_atlas as publisher


class Phase329PublisherTests(unittest.TestCase):
    def test_build_keeps_physical_nodes_and_adds_noncausal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as public_dir:
            output = Path(output_dir) / "atlas"
            public = Path(public_dir) / "atlas"
            manifest = publisher.build(output, public)
            self.assertEqual(manifest["phase"], 329)
            self.assertEqual(manifest["metrics"]["unique_unit_count"], 1121)
            self.assertEqual(manifest["metrics"]["full_vocabulary_mediation_path_count"], 9)
            self.assertEqual(manifest["metrics"]["cross_model_full_chain_candidate_count"], 0)
            self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)
            paths = [
                json.loads(line)
                for line in (output / "phase329_full_vocabulary_paths.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(paths), 9)
            self.assertTrue(all(not row["causal"] for row in paths))
            partition = json.loads(
                (output / "partitions/content_knowledge/qwen3.json").read_text()
            )
            self.assertEqual(len(partition["path"]["full_vocabulary_mediation_paths"]), 3)
            tested = [node for node in partition["nodes"] if node.get("phase329_tested")]
            self.assertEqual(len(tested), 36)
            self.assertTrue(all(not node["phase329_full_chain_candidate"] for node in tested))


if __name__ == "__main__":
    unittest.main()
