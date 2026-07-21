#!/usr/bin/env python3
"""Unit tests for the frozen Phase590 observer protocol."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase590_natural_semantic_event_protocol as protocol  # noqa: E402


class Phase590ProtocolTest(unittest.TestCase):
    def test_positive_parser(self) -> None:
        for text in (
            "People commonly eat it as food.",
            "It is edible and widely consumed.",
            "Yes. People eat it.",
        ):
            self.assertEqual(protocol.classify_semantic_text(text)["semantic_polarity"], "positive")

    def test_negative_parser(self) -> None:
        for text in (
            "People do not commonly eat it as food.",
            "It is poisonous and should not be eaten.",
            "No. It is not suitable for human consumption.",
        ):
            self.assertEqual(protocol.classify_semantic_text(text)["semantic_polarity"], "negative")

    def test_ambiguous_and_unresolved_parser(self) -> None:
        self.assertEqual(
            protocol.classify_semantic_text("Some parts can be eaten only after processing.")["semantic_polarity"],
            "ambiguous",
        )
        self.assertEqual(
            protocol.classify_semantic_text("It has a long and interesting history.")["semantic_polarity"],
            "unresolved",
        )

    def test_registered_static_audit(self) -> None:
        if not protocol.AUDIT_PATH.exists():
            protocol.register()
        audit = json.loads(protocol.AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 1152)
        self.assertEqual(audit["open_case_count"], 1008)
        self.assertEqual(audit["sealed_case_count"], 144)
        self.assertEqual(audit["prior_object_overlap_count"], 0)
        self.assertEqual(audit["explicit_answer_label_in_prompt_count"], 0)


if __name__ == "__main__":
    unittest.main()
