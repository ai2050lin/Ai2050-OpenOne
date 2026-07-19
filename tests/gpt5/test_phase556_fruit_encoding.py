#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase556_fruit_encoding_protocol as protocol  # noqa: E402
import phase556_fruit_event_collect as event_collect  # noqa: E402
import phase556_fruit_causal_intervention as causal_intervention  # noqa: E402
import phase556_fruit_causal_analysis as causal_analysis  # noqa: E402
import phase556_fruit_layer_input_boundary as layer_boundary  # noqa: E402
import phase556_fruit_direct_parent_decomposition as parent_decomposition  # noqa: E402
import phase556_publish_fruit_encoding_atlas as atlas_publisher  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Phase556FruitEncodingTests(unittest.TestCase):
    def test_static_denominator_and_seal(self) -> None:
        audit = read_json(protocol.AUDIT_PATH)
        commitment = read_json(protocol.SEALED_COMMITMENT_PATH)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["registered_case_count"], 17424)
        self.assertEqual(audit["open_case_count"], 11616)
        self.assertEqual(audit["sealed_case_count"], 5808)
        self.assertEqual(audit["controlled_rows_per_anchor"], [16])
        self.assertEqual(commitment["sealed_case_count"], 5808)
        self.assertFalse(commitment["sealed_split_read_for_analysis"])

    def test_open_rows_never_include_sealed(self) -> None:
        rows = read_jsonl(protocol.OPEN_CASES_PATH)
        self.assertEqual(len(rows), 11616)
        self.assertFalse(any(row["sealed"] for row in rows))
        self.assertEqual({row["split"] for row in rows}, set(protocol.OPEN_SPLITS))

    def test_factorial_anchor_has_all_cells_and_relations(self) -> None:
        rows = [
            row for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["case_type"] == "controlled_factorial"
            and row["anchor_id"] == "phase556_controlled_discovery_000"
        ]
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["factorial_cell"] for row in rows}, set(protocol.CELLS))
        by_cell = {row["factorial_cell"]: row for row in rows}
        base = by_cell["entity0_category0_query0_binding0"]
        category_swap = by_cell["entity0_category1_query0_binding0"]
        binding_swap_category_query = by_cell["entity0_category0_query0_binding1"]
        attribute_query = by_cell["entity0_category0_query1_binding0"]
        attribute_binding_swap = by_cell["entity0_category0_query1_binding1"]
        self.assertNotEqual(base["target"], category_swap["target"])
        self.assertEqual(base["target"], binding_swap_category_query["target"])
        self.assertNotEqual(attribute_query["target"], attribute_binding_swap["target"])

    def test_natural_split_has_six_fruits_and_four_controls(self) -> None:
        for split in protocol.SPLITS:
            objects = [row for row in protocol.NATURAL_OBJECTS if row["split"] == split]
            self.assertEqual(len(objects), 10)
            self.assertEqual(sum(row["is_fruit"] for row in objects), 6)
            self.assertEqual(sum(not row["is_fruit"] for row in objects), 4)

    def test_protocol_forbids_early_neuron_scan(self) -> None:
        frozen = read_json(protocol.PROTOCOL_PATH)
        self.assertFalse(frozen["internal_gate"]["head_channel_neuron_scan_before_path_gate"])
        self.assertFalse(frozen["evidence_policy"]["sealed_split_read"])
        self.assertTrue(frozen["evidence_policy"]["similarity_is_not_causality"])

    def test_factor_effect_keeps_sign_and_factor_identity(self) -> None:
        rows = []
        vectors = []
        for cell in protocol.CELLS:
            factors = protocol.cell_factors(cell)
            rows.append({"factor_values": factors})
            vectors.append([
                float(factors["category"]),
                float(factors["binding"]),
                float(factors["entity"]),
                float(factors["query"]),
            ])
        import torch

        tensor = torch.tensor(vectors)
        category = event_collect.factor_effect(tensor, rows, ("category",))
        binding = event_collect.factor_effect(tensor, rows, ("binding",))
        self.assertGreater(float(category[0]), 0.0)
        self.assertEqual(float(category[1]), 0.0)
        self.assertGreater(float(binding[1]), 0.0)
        self.assertEqual(float(binding[0]), 0.0)

    def test_causal_pairs_change_one_factor_and_preserve_control_target(self) -> None:
        rows = [
            row for row in read_jsonl(protocol.OPEN_CASES_PATH)
            if row["model"] == "qwen3"
            and row["case_type"] == "controlled_factorial"
            and row["anchor_id"] == "phase556_controlled_independent_confirmation_048"
        ]
        rows.sort(key=lambda row: protocol.CELLS.index(row["factorial_cell"]))
        for mechanism in ("category_reuse", "attribute_binding"):
            pairs = causal_intervention.matched_pairs(rows, mechanism)
            self.assertEqual(len(pairs), 8)
            self.assertEqual(sum(pair["pair_role"] == "target" for pair in pairs), 4)
            self.assertEqual(sum(pair["pair_role"] == "specificity_control" for pair in pairs), 4)
            for pair in pairs:
                changed = [
                    factor for factor in protocol.FACTORS
                    if pair["recipient"]["factor_values"][factor]
                    != pair["donor"]["factor_values"][factor]
                ]
                self.assertEqual(changed, [pair["factor"]])
                if pair["pair_role"] == "target":
                    self.assertNotEqual(pair["recipient"]["target"], pair["donor"]["target"])
                else:
                    self.assertEqual(pair["recipient"]["target"], pair["donor"]["target"])

    def test_multimodel_causal_boundary_and_parent_holdouts_are_disjoint(self) -> None:
        qualified_by_model: dict[str, list[str]] = {}
        for model in ("qwen3", "glm4"):
            rows = read_jsonl(
                protocol.OUT_DIR / f"phase556_{model}_behavior_rows.jsonl"
            )
            grouped: dict[str, list[dict]] = {}
            for row in rows:
                if (
                    row["split"] == "independent_confirmation"
                    and row["case_type"] == "controlled_factorial"
                    and 48 <= int(row["world_index"]) < 96
                ):
                    grouped.setdefault(row["anchor_id"], []).append(row)
            qualified = sorted(
                anchor for anchor, anchor_rows in grouped.items()
                if len(anchor_rows) == 16
                and all(row["semantic_correct"] for row in anchor_rows)
            )
            qualified_by_model[model] = qualified
            causal = set(causal_intervention.authorized_holdout_anchors(model, 12))
            partitions = (
                causal,
                set(qualified[12:24]),
                set(qualified[24:36]),
                set(qualified[36:44]),
            )
            self.assertEqual([len(item) for item in partitions], [12, 12, 12, 8])
            for index, left in enumerate(partitions):
                for right in partitions[index + 1:]:
                    self.assertFalse(left & right)
            self.assertEqual(len(set().union(*partitions)), 44)
        self.assertEqual(
            set(layer_boundary.qualified_anchor_slice("boundary_discovery")),
            set(qualified_by_model["qwen3"][12:24]),
        )
        self.assertEqual(
            set(layer_boundary.qualified_anchor_slice("boundary_confirmation")),
            set(qualified_by_model["qwen3"][24:36]),
        )
        self.assertEqual(
            set(parent_decomposition.qualified_anchors()),
            set(qualified_by_model["qwen3"][36:44]),
        )

    def test_boundary_numerical_valid_rates_are_probabilities(self) -> None:
        for filename in (
            "phase556_layer_input_boundary_analysis.json",
            "phase556_glm4_layer_input_boundary_analysis.json",
        ):
            analysis_path = protocol.OUT_DIR / filename
            if not analysis_path.exists():
                self.skipTest("boundary analysis has not been generated")
            payload = read_json(analysis_path)
            rates = [
                scenario_report["numerical_valid_rate"]
                for mechanism_report in payload["mechanism_reports"].values()
                for split_report in mechanism_report["split_reports"].values()
                for layer_report in split_report["layer_reports"].values()
                for scenario_report in layer_report["scenario_reports"].values()
            ]
            self.assertTrue(rates)
            self.assertTrue(all(0.0 <= rate <= 1.0 for rate in rates))

    def test_all_executed_behavior_and_internal_runs_use_bfloat16(self) -> None:
        for model in protocol.MODELS:
            rows = read_jsonl(
                protocol.OUT_DIR / f"phase556_{model}_behavior_rows.jsonl"
            )
            self.assertEqual({row["torch_dtype"] for row in rows}, {"torch.bfloat16"})
            self.assertEqual({row["quantized_8bit"] for row in rows}, {False})
        audit = read_json(protocol.OUT_DIR / "phase556_final_audit.json")
        self.assertTrue(audit["checks"]["all_internal_runs_bfloat16"])

    def test_final_audit_preserves_open_evidence_boundary(self) -> None:
        audit = read_json(protocol.OUT_DIR / "phase556_final_audit.json")
        self.assertTrue(all(audit["checks"].values()))
        self.assertEqual(audit["denominators"]["registered_cases"], 17424)
        self.assertEqual(audit["denominators"]["open_cases_executed"], 11616)
        self.assertEqual(audit["denominators"]["sealed_cases_unread"], 5808)
        self.assertEqual(audit["evidence_boundary"]["strict_closed_mechanisms"], 0)
        self.assertFalse(audit["evidence_boundary"]["local_writer_recovered"])

    def test_parameter_scan_requires_independent_parent_writer_gate(self) -> None:
        gates = causal_analysis.parameter_parent_gates()
        self.assertFalse(gates.get(("qwen3", "category_reuse"), False))
        self.assertFalse(gates.get(("qwen3", "attribute_binding"), False))
        self.assertFalse(gates.get(("glm4", "category_reuse"), False))
        self.assertFalse(gates.get(("glm4", "attribute_binding"), False))

    def test_atlas_accepts_diagnostic_confirmation_contract(self) -> None:
        diagnostics = read_json(
            protocol.OUT_DIR / "phase556_natural_behavior_diagnostics.json"
        )
        behavior = {
            row["model"]: row
            for row in read_json(protocol.OUT_DIR / "phase556_behavior_summary.json")["model_reports"]
        }
        payload = atlas_publisher.graph_payload(
            "deepseek7b", behavior["deepseek7b"], [], {}, diagnostics, None, None
        )
        natural = next(
            row for row in payload["graph"]["nodes"]
            if row["id"] == "phase556:deepseek7b:natural_fruit_category"
        )
        self.assertAlmostEqual(natural["score"], 0.5625)

    def test_restricted_readout_excludes_shared_whitespace_token(self) -> None:
        tokenizer = protocol.tokenizer_for("qwen3")
        newline_id = tokenizer("\nfruit", add_special_tokens=False)["input_ids"][0]
        fruit_ids = causal_intervention.word_token_ids(tokenizer, "fruit")
        tool_ids = causal_intervention.word_token_ids(tokenizer, "tool")
        self.assertNotIn(newline_id, fruit_ids)
        self.assertNotIn(newline_id, tool_ids)
        self.assertFalse(set(fruit_ids) & set(tool_ids))


if __name__ == "__main__":
    unittest.main()
