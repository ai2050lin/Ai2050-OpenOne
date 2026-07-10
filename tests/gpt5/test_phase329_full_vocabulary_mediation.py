import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase329_full_vocabulary_mediation as phase329


class Phase329MediationTests(unittest.TestCase):
    def test_selection_aligns_observation_to_next_block_input(self) -> None:
        rows = []
        for layer in range(5):
            rows.extend([
                {
                    "model": "qwen3",
                    "mechanism_id": "category_retrieval",
                    "split": "registered_primary",
                    "position_role": "query",
                    "layer": layer,
                    "comparison_variant": "same_target_object",
                    "residual_rms_delta": 0.1,
                },
                {
                    "model": "qwen3",
                    "mechanism_id": "category_retrieval",
                    "split": "registered_primary",
                    "position_role": "query",
                    "layer": layer,
                    "comparison_variant": "same_semantic_wrong_target",
                    "residual_rms_delta": 0.2 + layer,
                },
            ])
        selection = phase329.freeze_residual_selection(
            "qwen3", "category_retrieval", rows
        )
        self.assertEqual(
            selection["intervention_input_layer"],
            selection["residual_observation_layer"] + 1,
        )
        self.assertFalse(selection["selection_updates_allowed"])

    def test_norm_matched_control_preserves_per_token_norm(self) -> None:
        reference = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        control = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        result = phase329.norm_matched(reference, control)
        self.assertTrue(torch.allclose(
            torch.linalg.vector_norm(result, dim=1),
            torch.linalg.vector_norm(reference, dim=1),
        ))

    def test_blocker_summaries_only_count_above_target(self) -> None:
        rows = [
            {
                "model": "qwen3", "mechanism_id": "category_retrieval",
                "condition": "recipient_baseline", "blocker_category": "semantic_content_other",
                "is_full_vocabulary_blocker": True, "case_id": "a", "logit_above_target": 1.0,
            },
            {
                "model": "qwen3", "mechanism_id": "category_retrieval",
                "condition": "recipient_baseline", "blocker_category": "semantic_content_other",
                "is_full_vocabulary_blocker": False, "case_id": "b", "logit_above_target": -1.0,
            },
        ]
        summaries = phase329.blocker_type_summaries(rows)
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["top50_blocker_observation_count"], 1)

    def test_rank_bands_are_fixed_and_exhaustive(self) -> None:
        self.assertEqual(phase329.rank_band(1), "rank_1")
        self.assertEqual(phase329.rank_band(5), "rank_2_5")
        self.assertEqual(phase329.rank_band(50), "rank_6_50")
        self.assertEqual(phase329.rank_band(100), "rank_51_100")
        self.assertEqual(phase329.rank_band(1000), "rank_101_1000")
        self.assertEqual(phase329.rank_band(1001), "rank_above_1000")


if __name__ == "__main__":
    unittest.main()
