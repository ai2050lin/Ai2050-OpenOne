from __future__ import annotations

import json
import unittest

import phase585_object_swap_behavior as behavior
import phase585_object_swap_protocol as protocol


class Phase585ObjectSwapTests(unittest.TestCase):
    def test_static_protocol_is_valid_and_label_free(self) -> None:
        audit = json.loads(protocol.AUDIT_PATH.read_text(encoding="utf-8"))
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 1120)
        self.assertEqual(audit["category_label_in_prompt_count"], 0)
        self.assertEqual(audit["target_alias_in_prompt_count"], 0)
        self.assertFalse(
            frozen["evidence_policy"]["category_label_used_as_internal_coordinate"]
        )

    def test_fragment_matching_uses_word_start(self) -> None:
        self.assertTrue(protocol.fragment_present("manufactured", "manufactur"))
        self.assertTrue(protocol.fragment_present("used for cycling", "cycl"))
        self.assertFalse(protocol.fragment_present("a bicycle", "cycl"))
        self.assertFalse(protocol.fragment_present("answer briefly", "fly"))

    def test_object_echo_does_not_pass(self) -> None:
        row = {
            "target_aliases": ["nail", "strik", "pound"],
            "forbidden_aliases": [],
            "object_label": "hammer",
            "canonical_answer": "driving nails",
        }
        result = behavior.classify(row, "hammer")
        self.assertFalse(result["semantic_correct"])
        self.assertEqual(result["semantic_event"], "object_echo_without_answer")

    def test_target_with_forbidden_origin_is_not_accepted(self) -> None:
        row = {
            "target_aliases": ["plant", "grow"],
            "forbidden_aliases": ["manufactur", "factory"],
            "object_label": "apple",
            "canonical_answer": "grown by a plant",
        }
        result = behavior.classify(row, "grown on a plant, then manufactured")
        self.assertFalse(result["semantic_correct"])
        self.assertEqual(result["semantic_event"], "target_and_forbidden")


if __name__ == "__main__":
    unittest.main()
