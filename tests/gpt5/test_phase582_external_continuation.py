from __future__ import annotations

import json
import unittest

import phase582_external_continuation_observer as observer
import phase582_external_continuation_protocol as protocol


class Phase582ExternalContinuationTests(unittest.TestCase):
    def test_each_relation_has_exactly_one_foil(self) -> None:
        for relation, categories in protocol.source.RELATION_CATEGORIES.items():
            for target in categories:
                row = {"relation": relation, "target_category": target}
                foil = observer.foil_for(row)
                self.assertIn(foil, categories)
                self.assertNotEqual(foil, target)

    def test_candidate_continuations_are_not_prompt_inputs(self) -> None:
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertFalse(
            frozen["score_definition"]["candidate_words_inserted_into_prompt"]
        )
        self.assertTrue(
            frozen["evidence_policy"]["external_observer_not_natural_generation"]
        )

    def test_gate_is_relation_specific_and_strict(self) -> None:
        self.assertEqual(protocol.MIN_TARGET_WIN_RATE, 0.90)
        self.assertGreater(protocol.MIN_MEAN_MARGIN, 0.0)
        self.assertEqual(
            set(protocol.MIN_QUALIFIED_BY_RELATION_CATEGORY),
            set(protocol.RELATIONS),
        )


if __name__ == "__main__":
    unittest.main()
