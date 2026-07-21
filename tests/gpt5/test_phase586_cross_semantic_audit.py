from __future__ import annotations

import json
import unittest

import phase586_cross_semantic_audit as audit
import phase586_cross_semantic_audit_protocol as protocol
import phase586_cross_semantic_interface_analysis as interface_analysis


class Phase586CrossSemanticAuditTests(unittest.TestCase):
    def test_judgment_parser_is_strict(self) -> None:
        self.assertEqual(audit.parse_judgment("YES"), "YES")
        self.assertEqual(audit.parse_judgment("\nAMBIGUOUS"), "AMBIGUOUS")
        self.assertIsNone(audit.parse_judgment("YES or NO"))
        self.assertIsNone(audit.parse_judgment("certainly"))

    def test_contract_is_retrospective_and_blinded(self) -> None:
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.assertTrue(
            frozen["evidence_policy"]["retrospective_open_output_observer_calibration"]
        )
        self.assertTrue(frozen["judge_input_blinding"]["source_model_hidden"])
        self.assertTrue(frozen["judge_input_blinding"]["split_hidden"])
        self.assertTrue(
            frozen["evidence_policy"]["cannot_directly_authorize_internal_trace"]
        )

    def test_consensus_gate_is_high_precision(self) -> None:
        self.assertEqual(protocol.MIN_YES_VOTES, 2)
        self.assertEqual(protocol.MAX_NO_VOTES, 0)
        self.assertEqual(protocol.JUDGE_REPEATS, ("judge1", "judge2"))

    def test_interface_stop_preserves_evidence_boundary(self) -> None:
        self.assertTrue(interface_analysis.V1_DIR.name.startswith("v1_"))
        self.assertNotEqual(
            interface_analysis.OUTPUT,
            protocol.DECISION_PATH,
        )


if __name__ == "__main__":
    unittest.main()
