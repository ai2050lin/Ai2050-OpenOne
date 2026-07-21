from __future__ import annotations

import unittest

import tests.gpt5.phase578_choice_world_protocol as protocol


class Phase578ChoiceWorldTests(unittest.TestCase):
    def test_world_gate_is_principled_fraction(self) -> None:
        self.assertEqual(protocol.MIN_STABLE_WORLD_RATE, 0.75)
        self.assertEqual(protocol.MIN_STABLE_WORLDS_PER_SPLIT, int(224 * 0.75))
        self.assertEqual(protocol.MIN_STABLE_WORLDS_PER_RELATION, int(112 * 0.75))

    def test_selected_partition_is_exhaustive_and_disjoint(self) -> None:
        self.assertEqual(
            protocol.NATURAL_TRACE_WORLDS_PER_SPLIT
            + protocol.CAUSAL_HOLDOUT_WORLDS_PER_SPLIT,
            protocol.SELECTED_WORLDS_PER_SPLIT,
        )
        selected = list(range(protocol.SELECTED_WORLDS_PER_SPLIT))
        natural = set(selected[0::2])
        causal = set(selected[1::2])
        self.assertFalse(natural & causal)
        self.assertEqual(natural | causal, set(selected))


if __name__ == "__main__":
    unittest.main()
