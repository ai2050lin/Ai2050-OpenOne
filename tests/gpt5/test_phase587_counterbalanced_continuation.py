from __future__ import annotations

import json
import unittest

import phase587_counterbalanced_continuation_protocol as protocol


class Phase587CounterbalancedContinuationTests(unittest.TestCase):
    def test_static_protocol_is_valid(self) -> None:
        audit = json.loads(protocol.AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 1120)
        self.assertEqual(audit["continuation_fragment_in_prompt_count"], 0)
        self.assertEqual(audit["category_label_in_prompt_count"], 0)

    def test_same_continuations_are_reused_as_targets_and_foils(self) -> None:
        self.assertEqual(
            set(protocol.CONTINUATIONS["ordinary_origin"]),
            {"natural_growth", "human_manufacture"},
        )
        mapping = protocol.TARGET_CLASS_BY_GROUP["ordinary_origin"]
        self.assertEqual(mapping["fruit"], "natural_growth")
        self.assertEqual(mapping["tool"], "human_manufacture")

    def test_observer_does_not_authorize_causality(self) -> None:
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertTrue(frozen["evidence_policy"]["external_observer_not_causal_evidence"])
        self.assertFalse(frozen["evidence_policy"]["sealed_split_read"])


if __name__ == "__main__":
    unittest.main()
