#!/usr/bin/env python3
"""Contract tests for Phase405 natural future branches."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from phase405_natural_future_analysis import natural_group_audit  # noqa: E402
from phase405_natural_future_protocol import (  # noqa: E402
    FAMILIES,
    QUERIES,
    STATE_IDS,
    SURFACE_REPLICAS,
    expected_answer,
    natural_branch,
    package_for,
)


class Phase405NaturalFutureTest(unittest.TestCase):
    def test_branches_do_not_contain_explicit_choice_contract(self) -> None:
        forbidden = ("choose ", "return exactly", "listed answer")
        for family in FAMILIES:
            for state_id in STATE_IDS[family]:
                for surface in SURFACE_REPLICAS:
                    package = package_for(family, 0, surface["lexical"])
                    for query in QUERIES[family]:
                        branch = natural_branch(
                            family,
                            package,
                            state_id,
                            query,
                            surface["syntax"],
                        ).lower()
                        self.assertFalse(any(text in branch for text in forbidden))

    def perfect_rows(self, family: str) -> list[dict]:
        rows = []
        for state_id in STATE_IDS[family]:
            for surface in SURFACE_REPLICAS:
                for query in QUERIES[family]:
                    target = expected_answer(family, state_id, query)
                    rows.append(
                        {
                            "state_id_private": state_id,
                            "surface_id_private": surface["surface_id"],
                            "future_query_private": query,
                            "target_private": target,
                            "global_top_is_target_token": True,
                        }
                    )
        return rows

    def test_perfect_natural_group_passes(self) -> None:
        for family in FAMILIES:
            self.assertTrue(
                natural_group_audit(self.perfect_rows(family), family)[
                    "natural_group_pass"
                ]
            )

    def test_two_failed_surfaces_fail_natural_group(self) -> None:
        family = "grammar_constraint"
        rows = self.perfect_rows(family)
        failed_state = STATE_IDS[family][0]
        failed_surfaces = {
            SURFACE_REPLICAS[0]["surface_id"],
            SURFACE_REPLICAS[1]["surface_id"],
        }
        for row in rows:
            if (
                row["state_id_private"] == failed_state
                and row["surface_id_private"] in failed_surfaces
            ):
                row["global_top_is_target_token"] = False
        self.assertFalse(natural_group_audit(rows, family)["natural_group_pass"])


if __name__ == "__main__":
    unittest.main()
