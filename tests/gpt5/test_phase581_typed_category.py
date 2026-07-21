from __future__ import annotations

import unittest

import phase580_open_category_behavior as base
import phase581_typed_category_protocol as protocol


class Phase581TypedCategoryTests(unittest.TestCase):
    def test_relation_partition_is_total_and_disjoint(self) -> None:
        categories = []
        for values in protocol.RELATION_CATEGORIES.values():
            categories.extend(values)
        self.assertEqual(sorted(categories), sorted(protocol.CATEGORY_ALIASES))
        self.assertEqual(len(categories), len(set(categories)))

    def test_all_typed_surfaces_exclude_answer_aliases(self) -> None:
        for split in protocol.SPLITS:
            for item in protocol.OBJECT_GROUPS[protocol.SPLIT_GROUP[split]]:
                for surface_id in protocol.SPLIT_SURFACES[split]:
                    prompt = protocol.prompt_for(item, surface_id)
                    for aliases in protocol.CATEGORY_ALIASES.values():
                        for alias in aliases:
                            self.assertIsNone(base.candidate_position(prompt, alias))

    def test_relation_matches_object_category(self) -> None:
        for values in protocol.OBJECT_GROUPS.values():
            for item in values:
                relation = protocol.relation_for(item["category"])
                self.assertIn(
                    item["category"], protocol.RELATION_CATEGORIES[relation]
                )

    def test_each_relation_has_twenty_four_surfaces(self) -> None:
        self.assertEqual(
            {key: len(value) for key, value in protocol.SURFACES_BY_RELATION.items()},
            {relation: 24 for relation in protocol.RELATIONS},
        )


if __name__ == "__main__":
    unittest.main()
