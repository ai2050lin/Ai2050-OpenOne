#!/usr/bin/env python3
"""Post-hoc, non-authorizing diagnostic of Phase1169 training-only features."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/glm5"))

import phase1169_natural_training_trajectory_bifurcation as phase  # noqa: E402


FEATURES = (
    "embedding_circulant_gram",
    "output_circulant_gram",
    "embedding_fourier_top4_share",
    "output_fourier_top4_share",
    "parameter_l2_norm",
    "local_equivariance_cosine",
    "path_independence_cosine",
)


def main() -> None:
    root = phase.OUT_ROOT
    score = phase.read_json(root / "analysis/score.json")
    rows = phase.read_jsonl(root / "runs/holdout/holdout_metrics.jsonl")
    lookup = {(row["trajectory_id"], row["step"]): row for row in rows}
    feature_rows = []
    for feature in FEATURES:
        deltas = []
        pairs = []
        for trajectory in score["trajectories"]:
            if not trajectory["transition_present"]:
                continue
            memorizer = lookup[(trajectory["trajectory_id"], trajectory["memorizer_step"])]
            generalizer = lookup[(trajectory["trajectory_id"], trajectory["generalizer_step"])]
            before = float(memorizer["training_only_structure"][feature])
            after = float(generalizer["training_only_structure"][feature])
            delta = after - before
            deltas.append(delta)
            pairs.append({
                "trajectory_id": trajectory["trajectory_id"],
                "memorizer_step": trajectory["memorizer_step"],
                "generalizer_step": trajectory["generalizer_step"],
                "memorizer_value": before,
                "generalizer_value": after,
                "delta": delta,
            })
        feature_rows.append({
            "feature": feature,
            "pair_count": len(deltas),
            "positive_delta_count": sum(value > 0 for value in deltas),
            "negative_delta_count": sum(value < 0 for value in deltas),
            "median_delta": statistics.median(deltas),
            "minimum_delta": min(deltas),
            "maximum_delta": max(deltas),
            "pairs": pairs,
        })
    result = {
        "phase": phase.PHASE,
        "created_at_utc": phase.utc_now(),
        "status": "posthoc_non_authorizing",
        "source_score_digest": score["score_digest"],
        "successful_trajectory_count": sum(row["transition_present"] for row in score["trajectories"]),
        "features": feature_rows,
        "non_implications": [
            "Feature direction was inspected after held-out trajectory labels were known.",
            "Consistent paired change does not establish prediction, mediation, necessity, sufficiency, or language-model validity.",
            "The failed primary endpoint remains failed and Phase1170 is not authorized automatically.",
        ],
    }
    result["diagnostic_digest"] = phase.digest(result)
    phase.write_json(root / "analysis/posthoc_training_signature_diagnostic.json", result)
    print(json.dumps({
        "status": result["status"],
        "successful_trajectory_count": result["successful_trajectory_count"],
        "diagnostic_digest": result["diagnostic_digest"],
    }))


if __name__ == "__main__":
    main()
