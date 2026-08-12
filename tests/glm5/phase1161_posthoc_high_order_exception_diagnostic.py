#!/usr/bin/env python3
"""Post-hoc, non-upgrading diagnostic for Phase1161's largest error."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1161_ordered_intervention_response_prediction"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    protocol = read_json(OUT_ROOT / "protocol/preregistration.json")
    metadata = read_json(OUT_ROOT / "predictions/metadata.json")
    score = read_json(OUT_ROOT / "analysis/confirmation_score.json")
    selected = score["selected_algorithm"]
    with np.load(OUT_ROOT / "predictions/confirmation_predictions.npz") as pack:
        predicted = np.asarray(pack[selected], dtype=np.float64)
    with np.load(OUT_ROOT / "runs/confirmation/holdout_responses.npz") as pack:
        observed = np.asarray(pack["response"], dtype=np.float64)
    absolute_error = np.abs(predicted - observed)
    median_by_subset = np.median(absolute_error, axis=(0, 1))
    order = np.argsort(-median_by_subset, kind="stable")
    top_index = int(order[0])
    second_index = int(order[1])
    subset = tuple(protocol["confirmation_holdout_subsets"][top_index])
    site_rows = [protocol["sites"][index] for index in subset]
    per_unit_max = np.argmax(absolute_error, axis=2)
    top_is_unit_max_count = int(np.sum(per_unit_max == top_index))
    without_top = np.delete(absolute_error, top_index, axis=2)
    signed_remainder = observed[:, :, top_index] - predicted[:, :, top_index]
    diagnostic = {
        "phase": 1161,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_non_upgrading_diagnostic",
        "evidence_upgrade_forbidden": True,
        "selected_algorithm": selected,
        "top_exception_index": top_index,
        "top_exception_subset_id": metadata["holdout_subset_ids"][top_index],
        "top_exception_site_indices": list(subset),
        "top_exception_sites": site_rows,
        "top_exception_cardinality": len(subset),
        "top_exception_is_max_for_unit_count": top_is_unit_max_count,
        "unit_count": int(np.prod(absolute_error.shape[:2])),
        "median_observed_response": float(np.median(observed[:, :, top_index])),
        "median_predicted_response": float(np.median(predicted[:, :, top_index])),
        "median_absolute_error": float(median_by_subset[top_index]),
        "median_signed_higher_order_remainder": float(np.median(signed_remainder)),
        "second_largest_subset_id": metadata["holdout_subset_ids"][second_index],
        "second_largest_median_absolute_error": float(median_by_subset[second_index]),
        "median_unit_mae_with_exception": float(np.median(np.mean(absolute_error, axis=2))),
        "median_unit_mae_without_exception": float(np.median(np.mean(without_top, axis=2))),
        "absolute_error_mass_fraction": float(
            np.sum(absolute_error[:, :, top_index]) / np.sum(absolute_error)
        ),
        "interpretation": (
            "The frozen pairwise predictor succeeds globally but misses one reproducible four-site schedule in all units. "
            "This is evidence for a high-order remainder under the fixed patch schedule, not proof of a unique gate or hyperedge."
        ),
        "non_implications": [
            "The exception does not identify which component computes the interaction.",
            "The four sites are not thereby a symmetric physical hyperedge.",
            "The result does not show factor identity dissipation.",
            "Post-hoc localization cannot modify the frozen Phase1161 pass decision.",
        ],
        "protocol_digest": protocol["protocol_digest"],
        "score_digest": score["score_digest"],
    }
    diagnostic["diagnostic_digest"] = digest(diagnostic)
    write_json(OUT_ROOT / "analysis/posthoc_high_order_exception.json", diagnostic)
    print(canonical(diagnostic))


if __name__ == "__main__":
    main()
