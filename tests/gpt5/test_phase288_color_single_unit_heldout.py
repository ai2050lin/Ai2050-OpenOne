from __future__ import annotations

import unittest

from tests.gpt5.phase288_color_single_unit_heldout import COLORS, build_cases, select_preregistered_candidates


class Phase288DesignTests(unittest.TestCase):
    def test_full_case_design_has_declared_1440_cases(self) -> None:
        rows = build_cases(COLORS, 20, 6)
        self.assertEqual(len(rows), 1440)
        self.assertEqual(len({row["case_id"] for row in rows}), 1440)

    def test_smoke_case_design_is_small_and_deterministic(self) -> None:
        rows = build_cases(COLORS[:2], 2, 1)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["case_id"], "heldout:red:cube:t0")

    def test_missing_color_candidate_is_reported(self) -> None:
        selected, missing = select_preregistered_candidates("qwen3", ("red", "pink"))
        self.assertIn("red", selected)
        self.assertIn("pink", missing)


if __name__ == "__main__":
    unittest.main()
