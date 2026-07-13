#!/usr/bin/env python3
"""Contract tests for Phase403 finite predictive states."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase403_predictive_state_analysis import (  # noqa: E402
    base_single_group_audit,
    composition_group_audit,
)
from phase403_predictive_state_protocol import (  # noqa: E402
    CONTEXTS,
    FAMILIES,
    QUERIES,
    STATE_VARIANTS,
    SURFACE_REPLICAS,
    abstract_state,
    expected_answer,
    knowledge_state,
    package_for,
    reasoning_state,
)


class Phase403ContractTest(unittest.TestCase):
    def test_knowledge_composition_is_order_sensitive(self) -> None:
        for state_variant in STATE_VARIANTS:
            self.assertNotEqual(
                knowledge_state(state_variant, "swap_then_copy"),
                knowledge_state(state_variant, "copy_then_swap"),
            )

    def test_reasoning_composition_is_order_sensitive(self) -> None:
        for state_variant in STATE_VARIANTS:
            self.assertNotEqual(
                reasoning_state(state_variant, "swap_then_set_a"),
                reasoning_state(state_variant, "set_a_then_swap"),
            )

    def test_base_states_have_two_discriminating_queries(self) -> None:
        for family in FAMILIES:
            package = package_for(family, 0, 0)
            left = [
                expected_answer(family, package, 0, "base", query)[1]
                for query in QUERIES[family]
            ]
            right = [
                expected_answer(family, package, 1, "base", query)[1]
                for query in QUERIES[family]
            ]
            self.assertGreaterEqual(
                sum(a != b for a, b in zip(left, right, strict=True)), 2
            )

    def test_surface_axes_are_balanced(self) -> None:
        for axis in ("lexical", "syntax", "order"):
            self.assertEqual(
                [surface[axis] for surface in SURFACE_REPLICAS].count(0), 2
            )
            self.assertEqual(
                [surface[axis] for surface in SURFACE_REPLICAS].count(1), 2
            )

    def test_abstract_transition_table_is_deterministic(self) -> None:
        for family in FAMILIES:
            for state_variant in STATE_VARIANTS:
                for context, _kind in CONTEXTS[family]:
                    first = abstract_state(family, state_variant, context)
                    second = abstract_state(family, state_variant, context)
                    self.assertEqual(first, second)

    def synthetic_rows(self, family: str, contexts: list[str]) -> list[dict]:
        package = package_for(family, 0, 0)
        rows = []
        for state_variant in STATE_VARIANTS:
            for context in contexts:
                for surface in SURFACE_REPLICAS:
                    for query_index, query in enumerate(QUERIES[family]):
                        _target, canonical, _candidates = expected_answer(
                            family, package, state_variant, context, query
                        )
                        rows.append(
                            {
                                "state_variant_private": state_variant,
                                "operation_context_private": context,
                                "surface_id_private": surface["surface_id"],
                                "future_query_private": query,
                                "future_query_role_private": (
                                    "anchor"
                                    if query_index < 2
                                    else "pre_registered_unseen"
                                ),
                                "expected_canonical_private": canonical,
                                "predicted_canonical_private": canonical,
                                "semantic_correct": True,
                            }
                        )
        return rows

    def test_perfect_base_single_group_passes(self) -> None:
        for family in FAMILIES:
            contexts = [
                name
                for name, kind in CONTEXTS[family]
                if kind in {"base", "single"}
            ]
            audit = base_single_group_audit(
                self.synthetic_rows(family, contexts), family
            )
            self.assertTrue(audit["group_pass"])

    def test_perfect_composition_group_passes(self) -> None:
        for family in FAMILIES:
            contexts = [
                name for name, kind in CONTEXTS[family] if kind == "composition"
            ]
            audit = composition_group_audit(
                self.synthetic_rows(family, contexts), family
            )
            self.assertTrue(audit["group_pass"])

    def test_one_failed_surface_is_tolerated_but_two_are_not(self) -> None:
        family = "knowledge_binding"
        contexts = [
            name for name, kind in CONTEXTS[family] if kind in {"base", "single"}
        ]
        rows = self.synthetic_rows(family, contexts)
        for row in rows:
            if (
                row["state_variant_private"] == 0
                and row["operation_context_private"] == "base"
                and row["surface_id_private"] == SURFACE_REPLICAS[0]["surface_id"]
            ):
                row["semantic_correct"] = False
        self.assertTrue(base_single_group_audit(rows, family)["group_pass"])
        for row in rows:
            if (
                row["state_variant_private"] == 0
                and row["operation_context_private"] == "base"
                and row["surface_id_private"] == SURFACE_REPLICAS[1]["surface_id"]
            ):
                row["semantic_correct"] = False
        self.assertFalse(base_single_group_audit(rows, family)["group_pass"])


if __name__ == "__main__":
    unittest.main()
