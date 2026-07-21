from __future__ import annotations

import unittest

import tests.gpt5.phase576_natural_fruit_behavior as behavior
import tests.gpt5.phase576_natural_fruit_protocol as protocol


class Phase576NaturalFruitTests(unittest.TestCase):
    def test_object_groups_are_disjoint_and_balanced(self) -> None:
        seen: set[str] = set()
        for items in protocol.OBJECT_GROUPS.values():
            ids = {item["object_id"] for item in items}
            self.assertFalse(seen & ids)
            seen |= ids
            self.assertEqual(sum(item["is_fruit"] for item in items), 10)
            self.assertEqual(sum(not item["is_fruit"] for item in items), 4)

    def test_split_surface_families_are_fixed(self) -> None:
        for split in protocol.SPLITS:
            self.assertEqual(len(protocol.SPLIT_SURFACES[split]), 8)
            self.assertEqual(len(set(protocol.SPLIT_SURFACES[split])), 8)

    def test_relation_aliases_do_not_overlap_per_object(self) -> None:
        for items in protocol.OBJECT_GROUPS.values():
            for item in items:
                category = {value.casefold() for value in item["category_aliases"]}
                color = {value.casefold() for value in item["outer_color_aliases"]}
                self.assertFalse(category & color)

    def test_object_first_surfaces_place_object_before_field(self) -> None:
        item = protocol.OBJECT_GROUPS["A"][0]
        for split in protocol.SPLITS:
            for surface_id in protocol.SPLIT_SURFACES[split]:
                order, _ = protocol.surface_for(split, surface_id)
                prompt, _, field = protocol.prompt_for(item, "category", split, surface_id)
                if order == "object_first":
                    self.assertLess(prompt.index(item["label"]), prompt.index(field))

    def test_classifier_accepts_alias_and_rejects_other_relation(self) -> None:
        row = {
            "all_candidates": ["fruit", "vegetable", "green", "yellow"],
            "target_aliases": ["green", "yellow"],
            "other_relation_aliases": ["fruit"],
        }
        self.assertTrue(behavior.classify(row, "Green.")["semantic_correct"])
        wrong = behavior.classify(row, "fruit")
        self.assertFalse(wrong["semantic_correct"])
        self.assertEqual(wrong["semantic_event"], "same_object_other_relation")

    def test_candidate_matching_does_not_match_inside_word(self) -> None:
        self.assertIsNone(behavior.candidate_position("grapefruit", "fruit"))
        self.assertEqual(behavior.candidate_position("fruit", "fruit"), 0)


if __name__ == "__main__":
    unittest.main()
