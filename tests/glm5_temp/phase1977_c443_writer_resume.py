#!/usr/bin/env python3
"""Resume C443 after NumPy rejected a three-axis linalg.norm call."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))

import phase1968_c434_c445_guarded_response_graph_campaign as campaign


def main() -> None:
    out = campaign.OUTS["C443"]
    campaign.save(out / "audit/execution_recovery.json", {
        "error": "numpy.linalg.norm rejected axis=(-3,-2,-1)",
        "stage": "before metric reveal",
        "repair": "sqrt(sum(square(x), axis=(-3,-2,-1)))",
        "contract_changed": False,
        "seed_changed": False,
        "threshold_changed": False,
    })
    rng = np.random.default_rng(4431968)
    systems, checkpoints, roles, dim, operations = 96, 4, 6, campaign.DIM, 3
    base = rng.normal(0, 0.3, size=(systems, checkpoints, roles, dim)).astype(np.float32)
    templates = rng.normal(0, 0.04, size=(operations, checkpoints, roles, dim)).astype(np.float32)
    gates = rng.uniform(0.4, 1.3, size=(systems, operations, 1, 1, 1)).astype(np.float32)
    responses = gates * templates[None]
    damaged = base[:, None] - responses

    def norm3(value: np.ndarray) -> np.ndarray:
        return np.sqrt(np.sum(value * value, axis=(-3, -2, -1)))

    def recovery(patch: np.ndarray) -> float:
        restored = damaged + patch
        numerator = norm3(restored - damaged)
        error = norm3(restored - base[:, None])
        return float(np.mean(1.0 - error / (numerator + 1e-8)))

    correct = recovery(responses)
    controls = {
        "wrong_operation": recovery(np.roll(responses, 1, axis=1)),
        "wrong_role": recovery(np.roll(responses, 1, axis=3)),
        "wrong_checkpoint": recovery(np.roll(responses, 1, axis=2)),
        "coordinate_roll": recovery(np.roll(responses, 257, axis=4)),
        "matched_noise": recovery(rng.normal(0, np.std(responses), size=responses.shape).astype(np.float32)),
    }
    calibrated = correct >= 0.95 and max(controls.values()) <= 0.35
    campaign.save(out / "raw/calibration_seed.json", {"seed": 4431968})
    headline = {
        "status": "expanded_known_truth_writer_calibration_closed",
        "correct_recovery": correct, "control_recovery": controls,
        "writer_calibrated": calibrated, "execution_recovered": True,
        "strict_interpretation": "The instrument distinguishes registered synthetic truth from five mismatches; transfer to a natural model is not authorized by calibration alone.",
    }
    campaign.close("C443", headline, {"calibrated": calibrated, "finite": campaign.finite(headline)}, "C444_cross_model")


if __name__ == "__main__":
    main()
