from __future__ import annotations

from collections import Counter
import unittest

import phase580_open_category_behavior as behavior
import phase580_open_category_protocol as protocol


class Phase580OpenCategoryTests(unittest.TestCase):
    def test_object_groups_are_balanced_and_disjoint(self) -> None:
        seen: set[str] = set()
        for values in protocol.OBJECT_GROUPS.values():
            self.assertEqual(len(values), 30)
            self.assertEqual(
                Counter(item["category"] for item in values),
                {"fruit": 12, "vegetable": 6, "tool": 6, "vehicle": 6},
            )
            identifiers = {item["object_id"] for item in values}
            self.assertFalse(seen & identifiers)
            seen.update(identifiers)

    def test_surfaces_never_contain_registered_answer_words(self) -> None:
        for split in protocol.SPLITS:
            for item in protocol.OBJECT_GROUPS[protocol.SPLIT_GROUP[split]]:
                for surface_id in protocol.SPLIT_SURFACES[split]:
                    prompt = protocol.prompt_for(item, surface_id).casefold()
                    for aliases in protocol.CATEGORY_ALIASES.values():
                        for alias in aliases:
                            self.assertIsNone(
                                behavior.candidate_position(prompt, alias)
                            )

    def test_classifier_recovers_first_registered_category(self) -> None:
        row = {
            "target_category": "fruit",
            "target_aliases": ["fruit", "fruits"],
            "all_category_aliases": {
                key: list(value)
                for key, value in protocol.CATEGORY_ALIASES.items()
            },
        }
        result = behavior.classify(row, "Fruit. No explanation.")
        self.assertTrue(result["semantic_correct"])
        self.assertEqual(result["selected_category"], "fruit")
        self.assertEqual(result["semantic_event"], "target")

    def test_classifier_does_not_match_alias_inside_longer_word(self) -> None:
        row = {
            "target_category": "vehicle",
            "target_aliases": list(protocol.CATEGORY_ALIASES["vehicle"]),
            "all_category_aliases": {
                key: list(value)
                for key, value in protocol.CATEGORY_ALIASES.items()
            },
        }
        result = behavior.classify(row, "transportational")
        self.assertIsNone(result["selected_category"])
        self.assertEqual(result["semantic_event"], "unrecoverable")

    def test_stable_case_requires_correct_identical_repeats(self) -> None:
        base = {
            "semantic_correct": True,
            "selected_category": "fruit",
            "normalized_generated": "fruit",
        }
        rows = {("case", "noop1"): base, ("case", "noop2"): dict(base)}
        self.assertTrue(behavior.stable_case(rows, "case"))
        rows[("case", "noop2")]["normalized_generated"] = "Fruit."
        self.assertFalse(behavior.stable_case(rows, "case"))


if __name__ == "__main__":
    unittest.main()
