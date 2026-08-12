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
import phase1190_natural_sgd_quotient_transition as p1190  # noqa: E402


SOURCE = p1171.OUT_ROOT / "runs/training/checkpoints"
DEVICE = torch.device("cuda")
STEPS = (25, 150, 1000, 10000)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def main() -> None:
    endpoints = sorted(SOURCE.glob("*step10000.pt"))
    records = []
    for endpoint in endpoints:
        payload = p1189.load_payload(endpoint)
        panel = p1189.panel_from_payload(payload)
        state = {}
        for step in STEPS:
            current = p1189.load_model(p1189.load_payload(p1190.path_at(endpoint, step)), DEVICE)
            state[step] = {
                "cal": p1189.response_unit_shape(current, panel, panel.train_mask, DEVICE),
                "eval": p1189.response_unit_shape(current, panel, panel.holdout_mask, DEVICE),
            }
            del current
        records.append(
            {
                "task": payload["task_name"],
                "rep": payload["replicate"],
                "prefix": state[150]["cal"] - state[25]["cal"],
                "middle": state[1000]["eval"] - state[150]["eval"],
                "late": state[10000]["eval"] - state[1000]["eval"],
                "endpoint": state[10000]["eval"] - state[150]["eval"],
            }
        )
    rows = []
    for record in records:
        task_records = sorted([item for item in records if item["task"] == record["task"]], key=lambda x: x["rep"])
        null = task_records[(record["rep"] + 1) % len(task_records)]
        rows.append(
            {
                "task": record["task"],
                "rep": record["rep"],
                **{
                    key + "_true": cosine(record["prefix"], record[key])
                    for key in ("middle", "late", "endpoint")
                },
                **{
                    key + "_null": cosine(record["prefix"], null[key])
                    for key in ("middle", "late", "endpoint")
                },
            }
        )
    print(
        json.dumps(
            {
                key: {
                    "true": float(np.mean([row[key + "_true"] for row in rows])),
                    "null": float(np.mean([row[key + "_null"] for row in rows])),
                    "advantage": float(np.mean([row[key + "_true"] - row[key + "_null"] for row in rows])),
                    "positive_fraction": float(np.mean([row[key + "_true"] > row[key + "_null"] for row in rows])),
                }
                for key in ("middle", "late", "endpoint")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
