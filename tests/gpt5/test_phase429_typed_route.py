#!/usr/bin/env python3
"""Contract tests for Phase429 observer and typed behavior denominators."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase429_typed_route_analysis import observer_gate, typed_gates, wilson  # noqa: E402
from phase429_typed_route_collect import parse_generation  # noqa: E402
from phase429_typed_route_protocol import (  # noqa: E402
    INTERFACES,
    MODELS,
    OUT,
    freeze,
    interface_payload,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Phase429ContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = freeze()

    def test_denominators_and_commitments(self) -> None:
        validation = self.protocol["validation"]
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["observer_formal_group_count"], 384)
        self.assertEqual(validation["observer_formal_condition_count"], 4608)
        self.assertEqual(validation["observer_instrument_condition_count"], 48)
        self.assertEqual(validation["behavior_formal_group_count"], 2304)
        self.assertEqual(validation["behavior_open_group_count"], 1536)
        self.assertEqual(validation["behavior_sealed_group_count"], 768)
        self.assertEqual(validation["behavior_instrument_group_count"], 16)
        self.assertEqual(validation["behavior_open_selected_condition_count_per_model"], 15360)
        self.assertEqual(validation["behavior_sealed_selected_condition_count_per_model"], 7680)
        self.assertFalse(self.protocol["physical_contract"]["head_channel_neuron_scan_allowed"])
        for filename, expected in self.protocol["implementation_commitments"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / "tests/gpt5" / filename).read_bytes()).hexdigest(),
                expected,
                filename,
            )

    def test_full_crossing_and_independent_contracts(self) -> None:
        groups = read_jsonl(OUT / "phase429_behavior_groups_open.jsonl")
        crossed = [row for row in groups if row["contract_variant"] == "fully_crossed_examples"]
        bare = [row for row in groups if row["contract_variant"] == "no_examples"]
        expected = {"a:source", "a:query", "b:source", "b:query"}
        self.assertTrue(crossed and bare)
        self.assertTrue(all(set(row["demonstration_cells"]) == expected for row in crossed))
        self.assertTrue(all(not row["demonstration_cells"] for row in bare))
        counts = Counter(
            (row["block_id"], row["contract_variant"], row["split"]) for row in groups
        )
        self.assertTrue(all(value == 96 for value in counts.values()))

    def test_syntax_control_balances_lexical_identity_and_position(self) -> None:
        groups = [
            row
            for row in read_jsonl(OUT / "phase429_behavior_groups_open.jsonl")
            if row["block_id"] == "syntax_marked_anchor_control"
        ]
        identities = Counter(
            "first" if row["stable_target"] == row["first_item"] else "second"
            for row in groups
        )
        self.assertEqual(identities["first"], identities["second"])
        self.assertTrue(all(row["stable_target"] != row["decoy"] for row in groups))

    def test_four_interfaces_do_not_change_semantic_target(self) -> None:
        outputs = {
            interface: interface_payload(interface, "X000001", "Y000001", "X000001")
            for interface in INTERFACES
        }
        self.assertEqual(outputs["direct_item"]["target"], "X000001")
        self.assertEqual(outputs["short_code"]["target"], "alpha")
        self.assertEqual(outputs["forced_choice"]["target"], "option1")
        self.assertEqual(json.loads(outputs["result_field"]["target"])["result"], "X000001")

    def test_wilson_and_typed_gates_are_frozen(self) -> None:
        interval = wilson(96, 96)
        self.assertGreater(interval["lcb"], 0.95)
        summary = {
            "independent_group_count": 96,
            "teacher_margin_median": 1.0,
            "metrics": {
                "teacher_all": wilson(96, 96),
                "target_first_all": wilson(96, 96),
                "opposite_first_any": wilson(0, 96),
                "event_coverage_all": wilson(96, 96),
                "interface_valid_all": wilson(96, 96),
                "revision_any": wilson(0, 96),
                "boundary_all": wilson(96, 96),
                "stop_all": wilson(96, 96),
                "censor_any": wilson(0, 96),
            },
        }
        gates = typed_gates(summary, self.protocol["typed_thresholds"])
        self.assertTrue(gates["complete_generation"]["gate_pass"])
        failed = json.loads(json.dumps(summary))
        failed["metrics"]["target_first_all"] = wilson(0, 96)
        self.assertFalse(
            typed_gates(failed, self.protocol["typed_thresholds"])["content"]["gate_pass"]
        )

    def test_parser_separates_content_interface_revision_and_stop(self) -> None:
        row = {
            "target": "alpha",
            "opposite_target": "beta",
            "semantic_target": "X000001",
            "semantic_opposite": "Y000001",
            "interface": "short_code",
            "natural_generation_max_new_tokens": 8,
        }
        parsed = parse_generation("alpha. beta", [1, 2, 3], row, {99})
        self.assertTrue(parsed["natural_target_first"])
        self.assertTrue(parsed["natural_revision"])
        self.assertFalse(parsed["natural_interface_valid"])
        self.assertTrue(parsed["natural_boundary"])
        self.assertTrue(parsed["natural_stop"])

    def test_completed_outputs_if_present(self) -> None:
        freeze_path = OUT / "phase429_interface_freeze.json"
        if not freeze_path.exists():
            return
        interface_freeze = read_json(freeze_path)
        self.assertEqual(set(interface_freeze["models"]), set(MODELS))
        for model in MODELS:
            complete = read_json(
                OUT / "models" / model / "observer" / "phase429_collection_complete.json"
            )
            self.assertTrue(complete["all_rows_complete"])
            self.assertEqual(complete["condition_count"], 1536)
        summary_path = OUT / "phase429_global_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            self.assertEqual(summary["strict_mechanism_closure"], "0/72")
            self.assertFalse(summary["causal_tested"])
            self.assertTrue(math.isfinite(float(summary["overall_scientific_progress_percent"])))


if __name__ == "__main__":
    unittest.main()
