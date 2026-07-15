from __future__ import annotations

import json
import unittest

from tests.gpt5 import phase432_prechoice_terminal_protocol as protocol


class Phase432PrechoiceTerminalTests(unittest.TestCase):
    def test_denominator_is_large_balanced_and_disjoint(self) -> None:
        open_rows, sealed_rows = protocol.build_groups()
        audit = protocol.denominator_audit(open_rows, sealed_rows)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["groups_per_block_split"], 128)
        self.assertEqual(audit["conditions_per_model_open"], 2560)
        self.assertEqual(
            {row["semantic_group_id"] for row in open_rows}
            & {row["semantic_group_id"] for row in sealed_rows},
            set(),
        )

    def test_fresh_items_do_not_reuse_phase431_serials(self) -> None:
        open_rows, sealed_rows = protocol.build_groups()
        for row in [*open_rows, *sealed_rows]:
            self.assertTrue(row["first_item"].startswith("X432") or row["first_item"].startswith("X433"))
            self.assertTrue(row["second_item"].startswith("Y432") or row["second_item"].startswith("Y433"))

    def test_primary_window_is_fixed_and_prechoice(self) -> None:
        self.assertEqual(protocol.PRIMARY_WINDOW["model"], "qwen3")
        self.assertEqual(protocol.PRIMARY_WINDOW["layer"], 26)
        self.assertEqual(protocol.PRIMARY_WINDOW["position_role"], "prompt_terminal")
        self.assertNotIn("g1", json.dumps(protocol.PRIMARY_WINDOW))

    def test_candidate_source_targets_are_balanced(self) -> None:
        open_rows, _ = protocol.build_groups()
        candidate = [row for row in open_rows if row["candidate"]]
        first = sum(row["role_targets"]["a"] == row["source_1"] for row in candidate)
        second = sum(row["role_targets"]["a"] == row["source_2"] for row in candidate)
        self.assertEqual(first, second)

    def test_sealed_policy_requires_open_gate(self) -> None:
        frozen = protocol.freeze()
        self.assertTrue(
            frozen["sealed_commitment"]["read_requires_open_confirmation_pass"]
        )
        self.assertTrue(frozen["sealed_policy"]["no_read_before_open_gate"])

    def test_evidence_contract_forbids_mechanism_claim(self) -> None:
        frozen = protocol.freeze()
        evidence = frozen["evidence_contract"]
        self.assertFalse(evidence["causal"])
        self.assertFalse(evidence["single_neuron"])
        self.assertFalse(evidence["mechanism_closure"])


if __name__ == "__main__":
    unittest.main()
