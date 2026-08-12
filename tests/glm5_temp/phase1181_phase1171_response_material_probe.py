#!/usr/bin/env python3
"""Development-only response-material probe over sealed Phase1171 endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))
sys.path.insert(0, str(ROOT / "tests/glm5_temp"))

import phase1171_fixed_dimension_formation_trajectory_tomography as p1171  # noqa: E402
import phase1181_natural_response_material_probe as probe  # noqa: E402


CHECKPOINT_ROOT = (
    ROOT
    / "tests/glm5/result/phase1171_fixed_dimension_formation_trajectory_tomography"
    / "runs/training/checkpoints"
)
OUTPUT = ROOT / "tests/glm5_temp/phase1181_phase1171_response_material_probe.json"


def pairwise_distances(vectors: list[list[float]]) -> list[float]:
    if len(vectors) < 2:
        return []
    array = np.asarray(vectors, dtype=np.float64)
    return [
        float(np.linalg.norm(array[left] - array[right]))
        for left in range(len(array))
        for right in range(left + 1, len(array))
    ]


def behavior_matched_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    if len(items) < 2:
        return {"matched_pair_count": 0}
    metric_names = (
        "train_accuracy",
        "holdout_accuracy",
        "train_loss",
        "holdout_loss",
        "train_mean_margin",
        "holdout_mean_margin",
        "parameter_norm",
    )
    behavior = np.asarray(
        [[item["behavior"][name] for name in metric_names] for item in items],
        dtype=np.float64,
    )
    behavior = (behavior - behavior.mean(axis=0)) / np.maximum(behavior.std(axis=0), 1e-12)
    response = np.asarray(
        [item["holdout_response"]["unit_shape"] for item in items], dtype=np.float64
    )
    all_behavior_distances: list[float] = []
    all_response_distances: list[float] = []
    nearest_pairs: set[tuple[int, int]] = set()
    for left in range(len(items)):
        candidates: list[tuple[float, int]] = []
        for right in range(len(items)):
            if left == right:
                continue
            distance = float(np.linalg.norm(behavior[left] - behavior[right]))
            candidates.append((distance, right))
            if right > left:
                all_behavior_distances.append(distance)
                all_response_distances.append(float(np.linalg.norm(response[left] - response[right])))
        nearest = min(candidates)[1]
        nearest_pairs.add(tuple(sorted((left, nearest))))
    matched_response = [
        float(np.linalg.norm(response[left] - response[right]))
        for left, right in sorted(nearest_pairs)
    ]
    correlation = float(np.corrcoef(all_behavior_distances, all_response_distances)[0, 1])
    return {
        "matched_pair_count": len(matched_response),
        "matched_response_distance_median": float(np.median(matched_response)),
        "matched_response_distance_minimum": min(matched_response),
        "pairwise_behavior_response_distance_correlation": correlation,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    checkpoints = sorted(CHECKPOINT_ROOT.glob("*step10000.pt"))
    if len(checkpoints) != 64:
        raise RuntimeError(f"expected 64 sealed endpoints, found {len(checkpoints)}")
    systems: list[dict[str, Any]] = []
    gauge_checks: list[dict[str, float]] = []
    for index, checkpoint_path in enumerate(checkpoints):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = p1171.RoleSquareConfig(**payload["config"])
        model = p1171.RoleSquareNetwork(config).to(device)
        model.load_state_dict(payload["state_dict"])
        data = p1171.make_data(tuple(payload["operation"]), int(payload["seed"]))
        x = torch.cat((data["train_x"], data["holdout_x"]), dim=0)
        y = torch.cat((data["train_y"], data["holdout_y"]), dim=0)
        train_mask = torch.zeros(len(x), dtype=torch.bool)
        train_mask[: len(data["train_x"])] = True
        panel = probe.DataPanel(
            x=x,
            y=y,
            train_mask=train_mask,
            holdout_mask=~train_mask,
        )
        behavior = probe.evaluate(model, panel, device)
        holdout_response = probe.response_spectrum(model, panel, panel.holdout_mask, device)
        replay_response = probe.response_spectrum(model, panel, panel.holdout_mask, device)
        replay_error = float(
            np.max(
                np.abs(
                    np.asarray(holdout_response["ordered"])
                    - np.asarray(replay_response["ordered"])
                )
            )
        )
        if index < 2:
            transformed = probe.gauge_transform(model, 11819100 + index, device)
            original_logits, _ = probe.logits_and_hidden(model, panel.x, device)
            transformed_logits, _ = probe.logits_and_hidden(transformed, panel.x, device)
            transformed_response = probe.response_spectrum(
                transformed, panel, panel.holdout_mask, device
            )
            gauge_checks.append(
                {
                    "maximum_logit_error": float(
                        (original_logits - transformed_logits).abs().max().item()
                    ),
                    "maximum_ordered_response_error": float(
                        np.max(
                            np.abs(
                                np.asarray(holdout_response["ordered"])
                                - np.asarray(transformed_response["ordered"])
                            )
                        )
                    ),
                }
            )
        systems.append(
            {
                "checkpoint": checkpoint_path.name,
                "task_index": int(payload["task_index"]),
                "replicate": int(payload["replicate"]),
                "operation": list(payload["operation"]),
                "behavior": behavior,
                "holdout_response": holdout_response,
                "replay_maximum_error": replay_error,
            }
        )
        del model
        torch.cuda.empty_cache()
        print(json.dumps({"completed": index + 1, "total": len(checkpoints)}), flush=True)

    qualified = [
        item
        for item in systems
        if item["behavior"]["train_accuracy"] >= 0.95
        and item["behavior"]["holdout_accuracy"] >= 0.90
        and item["holdout_response"]["centered_norm"] > 1e-8
    ]
    task_rows: dict[str, Any] = {}
    all_distances: list[float] = []
    for task_index in range(8):
        subset = [item for item in qualified if item["task_index"] == task_index]
        distances = pairwise_distances(
            [item["holdout_response"]["unit_shape"] for item in subset]
        )
        all_distances.extend(distances)
        task_rows[str(task_index)] = {
            "qualified_count": len(subset),
            "pair_count": len(distances),
            "median_unit_shape_distance": float(np.median(distances)) if distances else None,
            "minimum_unit_shape_distance": min(distances) if distances else None,
            "maximum_unit_shape_distance": max(distances) if distances else None,
            "behavior_matched": behavior_matched_summary(subset),
        }
    scales = [item["holdout_response"]["centered_norm"] for item in qualified]
    summary = {
        "system_count": len(systems),
        "qualified_count": len(qualified),
        "minimum_train_accuracy": min(item["behavior"]["train_accuracy"] for item in systems),
        "minimum_holdout_accuracy": min(item["behavior"]["holdout_accuracy"] for item in systems),
        "maximum_replay_error": max(item["replay_maximum_error"] for item in systems),
        "maximum_gauge_logit_error": max(item["maximum_logit_error"] for item in gauge_checks),
        "maximum_gauge_response_error": max(
            item["maximum_ordered_response_error"] for item in gauge_checks
        ),
        "global_median_within_task_unit_shape_distance": float(np.median(all_distances)),
        "global_minimum_within_task_unit_shape_distance": min(all_distances),
        "global_maximum_within_task_unit_shape_distance": max(all_distances),
        "response_scale_coefficient_of_variation": float(
            np.std(scales) / max(np.mean(scales), 1e-12)
        ),
        "by_task": task_rows,
    }
    result = {
        "status": "development_probe_over_preexisting_sealed_endpoints",
        "source_phase": 1171,
        "intervention": "single hidden channel pre-square ablation",
        "quotient": "ordered response spectrum under exact signed channel permutations",
        "summary": summary,
        "gauge_checks": gauge_checks,
        "systems": systems,
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
