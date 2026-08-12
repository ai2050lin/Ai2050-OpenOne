from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1189_quotient_formation_operator_calibration as p1189  # noqa: E402


SOURCE = p1171.OUT_ROOT / "runs/training/checkpoints"
STEPS = (25, 50, 75, 100, 150, 200, 350, 500, 750, 1000)
DEVICE = torch.device("cuda")


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(left, right)
        / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    )


def main() -> None:
    endpoints = sorted(SOURCE.glob("*step10000.pt"))[:8]
    trajectories = []
    for endpoint in endpoints:
        payload = p1189.load_payload(endpoint)
        panel = p1189.panel_from_payload(payload)
        states = {}
        for step in STEPS:
            path = endpoint.with_name(endpoint.name.replace("step10000", f"step{step:05d}"))
            current = p1189.load_model(p1189.load_payload(path), DEVICE)
            states[step] = {
                "calibration": p1189.response_unit_shape(current, panel, panel.train_mask, DEVICE),
                "evaluation": p1189.response_unit_shape(current, panel, panel.holdout_mask, DEVICE),
            }
            del current
        transitions = []
        for left, right in zip(STEPS[:-1], STEPS[1:]):
            transitions.append(
                {
                    "interval": f"{left}-{right}",
                    "calibration": states[right]["calibration"] - states[left]["calibration"],
                    "evaluation": states[right]["evaluation"] - states[left]["evaluation"],
                }
            )
        trajectories.append(transitions)
    rows = []
    for replicate, transitions in enumerate(trajectories):
        for index, transition in enumerate(transitions):
            replicate_null = trajectories[(replicate + 1) % len(trajectories)][index]["evaluation"]
            time_null = transitions[(index + 1) % len(transitions)]["evaluation"]
            rows.append(
                {
                    "replicate": replicate,
                    "interval": transition["interval"],
                    "norm": float(np.linalg.norm(transition["evaluation"])),
                    "true": cosine(transition["calibration"], transition["evaluation"]),
                    "replicate_null": cosine(transition["calibration"], replicate_null),
                    "time_null": cosine(transition["calibration"], time_null),
                }
            )
    print(
        json.dumps(
            {
                "count": len(rows),
                "means": {
                    key: float(np.mean([row[key] for row in rows]))
                    for key in ("norm", "true", "replicate_null", "time_null")
                },
                "true_advantage_replicate": float(
                    np.mean([row["true"] - row["replicate_null"] for row in rows])
                ),
                "true_advantage_time": float(
                    np.mean([row["true"] - row["time_null"] for row in rows])
                ),
                "positive_fraction_replicate": float(
                    np.mean([row["true"] > row["replicate_null"] for row in rows])
                ),
                "positive_fraction_time": float(
                    np.mean([row["true"] > row["time_null"] for row in rows])
                ),
                "by_interval": {
                    interval: {
                        key: float(np.mean([row[key] for row in rows if row["interval"] == interval]))
                        for key in ("norm", "true", "replicate_null", "time_null")
                    }
                    for interval in sorted({row["interval"] for row in rows})
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
