from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch

import tests.gpt5.phase579_option_routing_causal as causal
import tests.gpt5.phase579_option_routing_causal_protocol as protocol


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase578_choice_world"


class Phase579OptionRoutingTests(unittest.TestCase):
    def test_natural_and_causal_worlds_are_disjoint(self) -> None:
        natural = json.loads(
            (OUT_DIR / "phase578_natural_trace_protocol.json").read_text()
        )
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text())
        for model in frozen["authorized_models"]:
            for split in ("causal_discovery", "causal_confirmation"):
                natural_ids = set(
                    natural["natural_trace_world_ids_by_model_and_split"][model][split]
                )
                causal_ids = set(
                    frozen["causal_holdout_world_ids_by_model_and_split"][model][split]
                )
                self.assertFalse(natural_ids & causal_ids)
                self.assertEqual(len(natural_ids), 72)
                self.assertEqual(len(causal_ids), 72)

    def test_frozen_coordinates_have_both_relations(self) -> None:
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text())
        for model in frozen["authorized_models"]:
            coordinates = frozen[
                "selected_coordinates_by_model_and_relation"
            ][model]
            self.assertEqual(set(coordinates), {"category", "outer_color"})
            self.assertTrue(all(item["joint_gate_score"] >= 0.70 for item in coordinates.values()))

    def test_score_group_swap_flips_group_means(self) -> None:
        tensor = torch.tensor([[1.0, 2.0, 7.0, 9.0]])
        swapped = causal.swap_group_means(tensor, [0, 1], [2, 3])
        self.assertAlmostEqual(float(swapped[:, [0, 1]].mean()), 8.0)
        self.assertAlmostEqual(float(swapped[:, [2, 3]].mean()), 1.5)
        self.assertAlmostEqual(float((swapped[0, 1] - swapped[0, 0])), 1.0)
        self.assertAlmostEqual(float((swapped[0, 3] - swapped[0, 2])), 2.0)

    def test_score_equalize_preserves_within_group_shape(self) -> None:
        tensor = torch.tensor([[1.0, 3.0, 7.0, 11.0]])
        equalized = causal.equalize_group_means(tensor, [0, 1], [2, 3])
        self.assertAlmostEqual(
            float(equalized[:, [0, 1]].mean()),
            float(equalized[:, [2, 3]].mean()),
        )
        self.assertAlmostEqual(float(equalized[0, 1] - equalized[0, 0]), 2.0)
        self.assertAlmostEqual(float(equalized[0, 3] - equalized[0, 2]), 4.0)

    def test_weight_swap_preserves_total_and_swaps_mass(self) -> None:
        weights = torch.tensor([[0.10, 0.20, 0.05, 0.15, 0.50]])
        swapped = causal.swap_group_weight_mass(weights, [0, 1], [2, 3])
        self.assertAlmostEqual(float(swapped.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(swapped[:, [0, 1]].sum()), 0.20, places=6)
        self.assertAlmostEqual(float(swapped[:, [2, 3]].sum()), 0.30, places=6)
        self.assertAlmostEqual(float(swapped[:, [4]].sum()), 0.50, places=6)

    def test_value_swap_exchanges_means_without_shape_change(self) -> None:
        values = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [8.0, 9.0], [10.0, 11.0]]]
        )
        swapped = causal.swap_group_value_means(values, [0, 1], [2, 3])
        self.assertEqual(swapped.shape, values.shape)
        self.assertTrue(
            torch.allclose(
                swapped[:, [0, 1]].mean(dim=1),
                values[:, [2, 3]].mean(dim=1),
            )
        )
        self.assertTrue(
            torch.allclose(
                swapped[:, [2, 3]].mean(dim=1),
                values[:, [0, 1]].mean(dim=1),
            )
        )


if __name__ == "__main__":
    unittest.main()
