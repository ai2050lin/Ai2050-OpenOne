from __future__ import annotations

import json
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]

from tests.gpt5.phase564_source_edge_intervention import (  # noqa: E402
    edge_contribution,
    reconstructed_target,
)
from tests.gpt5.phase564_source_edge_protocol import case_spec  # noqa: E402


class _Attention:
    def __init__(self, heads: int, head_dim: int, hidden: int) -> None:
        self.o_proj = torch.nn.Linear(heads * head_dim, hidden, bias=False)


class Phase564SourceEdgeTests(unittest.TestCase):
    def test_fixed_identity_pair_changes_only_binding_assignment(self) -> None:
        left = case_spec("edge_discovery", 0, 0, 0, 0, 0)
        right = case_spec("edge_discovery", 0, 1, 0, 0, 0)
        for key in (
            "object_a", "object_b", "color_a", "color_b", "query_object",
            "surface_id", "fact_order", "fact_token_multiset_key", "prompt_token_multiset_key",
        ):
            self.assertEqual(left[key], right[key])
        self.assertEqual(left["target"], right["nontarget_color"])
        self.assertEqual(right["target"], left["nontarget_color"])

    def test_source_contributions_sum_to_reconstructed_target(self) -> None:
        torch.manual_seed(564)
        batch, heads, targets, sources, head_dim, hidden = 2, 3, 4, 5, 7, 11
        module = _Attention(heads, head_dim, hidden)
        logits = torch.randn(batch, heads, targets, sources)
        weights = torch.softmax(logits, dim=-1)
        values = torch.randn(batch, heads, sources, head_dim)
        for batch_index in range(batch):
            for target in range(targets):
                parts = [
                    edge_contribution(module, weights, values, batch_index, target, [source])
                    for source in range(sources)
                ]
                reconstructed = reconstructed_target(
                    module, weights, values, batch_index, target
                )
                self.assertTrue(torch.allclose(sum(parts), reconstructed, atol=1e-5, rtol=1e-5))

    def test_final_audit_keeps_compute_closure_zero(self) -> None:
        audit_path = (
            ROOT / "tests/gpt5/result/phase565_residual_multiposition_operator/phase566_final_audit.json"
        )
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["objective_counts"]["qualified_source_compute_edges"], 0)
        self.assertEqual(payload["objective_counts"]["qualified_distributed_residual_operators"], 6)
        self.assertEqual(payload["objective_counts"]["strict_closed_mechanisms"], 0)


if __name__ == "__main__":
    unittest.main()
