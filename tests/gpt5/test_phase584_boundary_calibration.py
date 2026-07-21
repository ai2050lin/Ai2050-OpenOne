from __future__ import annotations

import json
import unittest

import phase584_boundary_calibration_analysis as analysis
import phase584_boundary_calibration_protocol as protocol


def row(target: str, target_logit: float, foil_logit: float) -> dict[str, object]:
    return {
        "target_category": target,
        "target_boundary_logit": target_logit,
        "foil_boundary_logit": foil_logit,
    }


class Phase584BoundaryCalibrationTests(unittest.TestCase):
    def test_fixed_axis_is_independent_of_target_order(self) -> None:
        categories = ("fruit", "vegetable")
        self.assertEqual(analysis.fixed_axis_value(row("fruit", 5.0, 2.0), categories), 3.0)
        self.assertEqual(
            analysis.fixed_axis_value(row("vegetable", 2.0, 5.0), categories),
            3.0,
        )

    def test_midpoint_boundary_uses_discovery_means(self) -> None:
        categories = ("fruit", "vegetable")
        rows = [
            row("fruit", 4.0, 0.0),
            row("fruit", 6.0, 0.0),
            row("vegetable", 5.0, 3.0),
            row("vegetable", 5.0, 1.0),
        ]
        boundary = analysis.fit_boundary(rows, categories)
        self.assertEqual(boundary["class_means"]["fruit"], 5.0)
        self.assertEqual(boundary["class_means"]["vegetable"], -3.0)
        self.assertEqual(boundary["threshold"], 1.0)

    def test_contract_marks_analysis_as_retrospective(self) -> None:
        frozen = json.loads(protocol.PROTOCOL_PATH.read_text(encoding="utf-8"))
        policy = frozen["evidence_policy"]
        self.assertTrue(policy["retrospective_open_data_diagnostic"])
        self.assertTrue(policy["registered_after_open_data_exploration"])
        self.assertFalse(policy["prompt_trace_authorized"])
        self.assertFalse(policy["causal_intervention_authorized"])


if __name__ == "__main__":
    unittest.main()
