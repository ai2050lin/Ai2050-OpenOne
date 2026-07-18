from __future__ import annotations

import json
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"


class Phase544ProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = json.loads((OUT / "phase544_static_audit.json").read_text(encoding="utf-8"))
        cls.protocol = json.loads((OUT / "phase544_frozen_protocol.json").read_text(encoding="utf-8"))
        cls.rows = [
            json.loads(line)
            for line in (OUT / "phase544_registered_cases.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_static_denominator_and_sample_bound(self) -> None:
        self.assertTrue(self.audit["valid"])
        self.assertEqual(self.audit["registered_case_count"], 31536)
        self.assertEqual(self.audit["family_count"], 9)
        self.assertEqual(self.audit["representative_mechanism_count"], 18)
        self.assertEqual(self.audit["semantic_units_per_mechanism_split"], [73])
        self.assertEqual(
            self.audit["minimum_independent_n"]["zero_unrecoverable_ucb95_le_0_05"], 73
        )

    def test_old_denominator_is_not_directly_reused(self) -> None:
        old = self.audit["old_denominator_audit"]
        self.assertEqual(old["phase330_mechanism_count"], 72)
        self.assertEqual(old["phase330_target_leak_count"], 1347)
        self.assertGreaterEqual(old["phase330_identical_semantic_contract_group_count"], 2)
        self.assertFalse(old["old_denominator_direct_reuse_authorized"])
        self.assertTrue(old["phase543_historical_phase535_sealed_read"])

    def test_counterfactual_pairs_flip_natural_targets(self) -> None:
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["surface_pair_id"]].append(row)
        self.assertEqual({len(group) for group in groups.values()}, {2})
        self.assertTrue(all(len({row["target"] for row in group}) == 2 for group in groups.values()))
        self.assertTrue(all(not row["arbitrary_label_output"] for row in self.rows))
        self.assertTrue(all(not row["sealed"] for row in self.rows))

    def test_confirmation_lexicon_is_disjoint(self) -> None:
        discovery = {row["lexical_key"] for row in self.rows if row["split"] == "discovery"}
        confirmation = {
            row["lexical_key"] for row in self.rows
            if row["split"] == "independent_confirmation"
        }
        self.assertFalse(discovery & confirmation)

    def test_required_model_order_and_evidence_boundaries(self) -> None:
        self.assertEqual(
            self.protocol["models_in_required_execution_order"],
            ["qwen3", "glm4", "deepseek7b"],
        )
        policy = self.protocol["evidence_policy"]
        self.assertFalse(policy["surface_rewrites_increase_independent_n"])
        self.assertFalse(policy["behavior_qualification_is_physical_mechanism"])
        self.assertFalse(policy["sealed_split_read"])


class Phase544ParserTest(unittest.TestCase):
    def test_plain_target_and_distractor(self) -> None:
        from tests.gpt5.phase544_nine_family_natural_behavior import classify_semantic

        row = {
            "mechanism_id": "category",
            "target_aliases": ["bird"],
            "distractors": ["tool", "unknown"],
            "strict_kind": "plain",
            "strict_expected": "bird",
            "target": "bird",
        }
        good = classify_semantic(row, "bird")
        bad = classify_semantic(row, "tool")
        self.assertTrue(good["semantic_correct"])
        self.assertTrue(good["strict_sequence_correct"])
        self.assertFalse(bad["semantic_correct"])
        self.assertEqual(bad["semantic_event"], "registered_distractor")

    def test_json_semantic_and_protocol_are_separate(self) -> None:
        from tests.gpt5.phase544_nine_family_natural_behavior import classify_semantic

        row = {
            "mechanism_id": "json",
            "target_aliases": ["17"],
            "distractors": ["16", "18"],
            "strict_kind": "json_answer",
            "strict_expected": '{"answer":"17"}',
            "target": "17",
        }
        parsed = classify_semantic(row, '{"answer": "17"}')
        self.assertTrue(parsed["semantic_correct"])
        self.assertTrue(parsed["protocol_valid"])
        self.assertFalse(parsed["strict_sequence_correct"])

    def test_transform_is_case_sensitive(self) -> None:
        from tests.gpt5.phase544_nine_family_natural_behavior import classify_semantic

        row = {
            "mechanism_id": "transform",
            "target_aliases": ["NARIN"],
            "distractors": ["narin", "Narin"],
            "strict_kind": "plain",
            "strict_expected": "NARIN",
            "target": "NARIN",
        }
        self.assertTrue(classify_semantic(row, "NARIN")["semantic_correct"])
        self.assertFalse(classify_semantic(row, "narin")["semantic_correct"])


if __name__ == "__main__":
    unittest.main()
