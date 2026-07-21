from __future__ import annotations

import json
import unittest

import phase583_prompt_boundary_observer as observer
import phase583_prompt_boundary_protocol as protocol


class FakeTokenizer:
    mapping = {
        "fruit": [1],
        "vegetable": [2, 3],
        "tool": [4],
        "vehicle": [5],
    }

    def encode(self, value: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self.mapping[value])


class Phase583PromptBoundaryTests(unittest.TestCase):
    def test_observer_uses_only_first_distinct_tokens(self) -> None:
        labels = observer.label_token_ids(FakeTokenizer())
        self.assertEqual(labels["fruit"]["first_token_id"], 1)
        self.assertEqual(labels["vegetable"]["first_token_id"], 2)
        self.assertEqual(labels["vegetable"]["all_token_ids"], [2, 3])

    def test_protocol_forbids_candidate_input_and_teacher_forcing(self) -> None:
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
        score = frozen["score_definition"]
        self.assertFalse(score["candidate_words_inserted_into_model_input"])
        self.assertFalse(score["teacher_forced_continuation_used"])
        self.assertFalse(score["prior_or_length_calibration_used"])

    def test_gate_remains_frozen_at_ninety_percent(self) -> None:
        self.assertEqual(protocol.MIN_TARGET_WIN_RATE, 0.90)
        self.assertEqual(protocol.MAX_REPEAT_LOGIT_DELTA, 1e-6)


if __name__ == "__main__":
    unittest.main()
