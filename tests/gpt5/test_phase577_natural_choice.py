from __future__ import annotations

import unittest

import tests.gpt5.phase577_natural_choice_behavior as behavior
import tests.gpt5.phase577_natural_choice_protocol as protocol


class Phase577NaturalChoiceTests(unittest.TestCase):
    def test_foil_never_matches_an_accepted_color(self) -> None:
        for items in protocol.p576.OBJECT_GROUPS.values():
            for item in items:
                target, foil = protocol.target_and_foil(item, "outer_color")
                self.assertIn(target, item["outer_color_aliases"])
                self.assertNotIn(foil, item["outer_color_aliases"])

    def test_category_choice_is_fruit_vs_vegetable(self) -> None:
        for items in protocol.p576.OBJECT_GROUPS.values():
            for item in items:
                target, foil = protocol.target_and_foil(item, "category")
                self.assertEqual({target, foil}, {"fruit", "vegetable"})

    def test_option_swap_preserves_semantics(self) -> None:
        item = protocol.p576.OBJECT_GROUPS["A"][0]
        first = protocol.render_case(item, "category", 0, "target_first")
        second = protocol.render_case(item, "category", 0, "target_second")
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(first[3], second[4])
        self.assertEqual(first[4], second[3])

    def test_classifier_uses_semantic_word(self) -> None:
        row = {"all_candidates": ["fruit", "vegetable"], "target": "fruit", "foil": "vegetable"}
        self.assertTrue(behavior.classify(row, "fruit")["semantic_correct"])
        self.assertEqual(behavior.classify(row, "vegetable")["semantic_event"], "foil")
        self.assertEqual(behavior.classify(row, "unknown")["semantic_event"], "unrecoverable")


if __name__ == "__main__":
    unittest.main()
