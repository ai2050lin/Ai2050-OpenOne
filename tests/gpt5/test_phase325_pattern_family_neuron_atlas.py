#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import phase325_pattern_family_neuron_atlas as atlas


class PatternFamilyNeuronAtlasTest(unittest.TestCase):
    def test_bundle_is_evidence_scoped_and_public_copy_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "result"
            public = root / "public"
            manifest = atlas.build_bundle(output, public)
            validation = atlas.validate_bundle(output)

            self.assertEqual(manifest["metrics"]["family_count"], 9)
            self.assertEqual(manifest["metrics"]["mapped_family_count"], 1)
            self.assertEqual(manifest["metrics"]["single_unit_causal_count"], 0)
            self.assertGreater(validation["nodes"], 0)
            self.assertEqual((output / "manifest.json").read_bytes(), (public / "manifest.json").read_bytes())

    def test_unmapped_families_have_no_synthetic_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "result"
            atlas.build_bundle(output, root / "public")
            manifest = atlas.read_json(output / "manifest.json")
            family_ids = {item["family_id"] for item in manifest["partitions"]}

            self.assertEqual(family_ids, {"content_knowledge"})
            self.assertIn("syntax_structure", manifest["evidence_boundary"]["unmapped_families"])


if __name__ == "__main__":
    unittest.main()
