#!/usr/bin/env python3
"""Post-hoc scope diagnostic for the frozen Phase1159 result."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "tests/glm5/result/phase1159_free_transformer_causal_use_external_validity"
FACTORS = ("row", "col", "context")
ROLES = ("bos", "row", "col", "context", "query")
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denominator <= 1e-12 else float(np.dot(a, b) / denominator)


def matrix(predicted: np.ndarray, actual: np.ndarray, indices: list[int]) -> list[list[float]]:
    return [
        [correlation(predicted[source, indices], actual[target, indices]) for target in range(len(FACTORS))]
        for source in range(len(FACTORS))
    ]


def retrieval(values: list[list[float]]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    predictions = np.argmax(array, axis=1)
    advantages = [float(array[index, index] - np.max(np.delete(array[index], index))) for index in range(len(FACTORS))]
    return {
        "correct": int(np.sum(predictions == np.arange(len(FACTORS)))),
        "total": len(FACTORS),
        "predicted_indices": predictions.tolist(),
        "diagonal_advantages": advantages,
        "minimum_diagonal_advantage": float(min(advantages)),
    }


def main() -> None:
    fit = json.loads((OUT_ROOT / "analysis/fit.json").read_text(encoding="utf-8"))
    with np.load(OUT_ROOT / "runs/confirmation/effect_pack.npz") as pack:
        actual = np.median(np.asarray(pack["matched_median"], dtype=np.float64), axis=0)
    predicted = np.asarray(fit["predicted_profile"], dtype=np.float64)
    all_indices = list(range(len(DEPTHS) * len(ROLES)))
    non_endpoint = [
        index
        for index, (depth, _role) in enumerate((value for value in __import__("itertools").product(DEPTHS, ROLES)))
        if depth not in (0.0, 1.0)
    ]
    interior_query = [
        index
        for index, (depth, role) in enumerate((value for value in __import__("itertools").product(DEPTHS, ROLES)))
        if depth not in (0.0, 1.0) and role == "query"
    ]
    panels = {}
    for name, indices in (
        ("all_sites", all_indices),
        ("non_endpoint_sites", non_endpoint),
        ("interior_query_only", interior_query),
    ):
        values = matrix(predicted, actual, indices)
        panels[name] = {"site_indices": indices, "correlation_matrix": values, "identity_retrieval": retrieval(values)}
    result = {
        "phase": 1159,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_scope_diagnostic_only",
        "factors": list(FACTORS),
        "panels": panels,
        "interpretation_rule": (
            "If all-site identity retrieval disappears after endpoint removal, the frozen pass is driven by "
            "factor-token injection plus a shared query-accumulation scaffold rather than an interior factor-identity map."
        ),
        "evidence_upgrade_forbidden": True,
    }
    result["diagnostic_digest"] = digest(result)
    path = OUT_ROOT / "analysis/posthoc_factor_specificity.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(canonical(result))


if __name__ == "__main__":
    main()
