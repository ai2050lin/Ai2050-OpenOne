#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase548_matched_observer_analysis as observer_analysis  # noqa: E402
import phase548_shared_attention_compute_protocol as protocol  # noqa: E402


class Phase548ProtocolTests(unittest.TestCase):
    def test_matched_target_relations(self) -> None:
        for mechanism in protocol.MECHANISMS:
            rows = {
                variant: protocol.case_spec(mechanism, "discovery", 7, variant)
                for variant in protocol.VARIANTS
            }
            base = rows["base_plus"]["target"]
            self.assertEqual(base, rows["identity_control"]["target"])
            self.assertEqual(base, rows["template_control"]["target"])
            self.assertEqual(
                len({
                    base,
                    rows["functional_minus"]["target"],
                    rows["answer_token_control"]["target"],
                }),
                3,
            )

    def test_split_entities_are_disjoint(self) -> None:
        discovery = {
            protocol.case_spec("category", "discovery", index, "base_plus")["entity_key"]
            for index in range(protocol.PAIR_UNITS_PER_SPLIT)
        }
        confirmation = {
            protocol.case_spec("category", "independent_confirmation", index, "base_plus")["entity_key"]
            for index in range(protocol.PAIR_UNITS_PER_SPLIT)
        }
        self.assertFalse(discovery & confirmation)

    def test_frozen_windows_exclude_deepseek(self) -> None:
        self.assertEqual(protocol.WINDOWS["qwen3"]["target_layers"], [28, 29, 30])
        self.assertEqual(protocol.WINDOWS["glm4"]["target_layers"], [34, 35, 36])
        self.assertEqual(protocol.WINDOWS["deepseek7b"]["target_layers"], [])

    def test_registered_static_audit(self) -> None:
        path = protocol.AUDIT_PATH
        if not path.exists():
            self.skipTest("Phase548 protocol has not been registered")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["registered_case_count"], 4380)
        self.assertEqual(payload["target_relation_error_count"], 0)
        self.assertEqual(payload["discovery_confirmation_entity_overlap_count"], 0)


class Phase548ObserverGateTests(unittest.TestCase):
    @staticmethod
    def rows(functional: float, identity: float, answer: float, template: float) -> list[dict[str, float]]:
        return [
            {
                "functional_delta": functional + index * 1e-5,
                "identity_delta": identity,
                "answer_token_delta": answer,
                "template_delta": template,
            }
            for index in range(73)
        ]

    def test_gate_passes_uniform_functional_dominance(self) -> None:
        report = observer_analysis.split_report(
            self.rows(1.0, 0.2, 0.3, 0.4), "synthetic:pass",
        )
        self.assertTrue(report["gate_pass"])
        self.assertTrue(all(item["one_sided_sign_flip_p"] <= 0.01 for item in report["controls"].values()))

    def test_gate_fails_answer_identity_dominance(self) -> None:
        report = observer_analysis.split_report(
            self.rows(0.4, 0.2, 0.8, 0.1), "synthetic:fail",
        )
        self.assertFalse(report["gate_pass"])
        self.assertFalse(report["controls"]["answer_token_delta"]["gate_pass"])


if __name__ == "__main__":
    unittest.main()
