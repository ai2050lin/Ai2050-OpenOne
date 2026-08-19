#!/usr/bin/env python3
"""Read-only posthoc decomposition of the frozen Phase1258 response tensor.

This analysis cannot change the preregistered verdict or select a template,
world, component or threshold. It localizes the held-out matched-null failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "tests/glm5/result/phase1258_c010_qwen3_natural_factorial_head_coalition"
DETAILS = BASE / "raw/head_coalition_result.json"
MATERIAL = BASE / "material/frozen_natural_factorial_worlds.jsonl"
FINAL = BASE / "analysis/final.json"
OUT = BASE / "analysis/null_geometry_posthoc.json"
PHASE1256 = ROOT / "tests/glm5/result/phase1256_c009_qwen3_typed_edge_coalition/raw/coalition_result.json"


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def vector_metrics(predicted: list[list[float]], target: list[list[float]]) -> dict[str, float]:
    p = [value for row in predicted for value in row]
    t = [value for row in target for value in row]
    dot = sum(left * right for left, right in zip(p, t))
    pn = math.sqrt(sum(value * value for value in p))
    tn = max(math.sqrt(sum(value * value for value in t)), 1.0e-12)
    return {
        "cosine": dot / max(pn * tn, 1.0e-12),
        "relative_error": math.sqrt(sum((left - right) ** 2 for left, right in zip(p, t))) / tn,
        "projection": dot / max(tn * tn, 1.0e-12),
    }


def null_decomposition(null: list[list[float]], target: list[list[float]]) -> dict[str, float]:
    n = [value for row in null for value in row]
    t = [value for row in target for value in row]
    t2 = max(sum(value * value for value in t), 1.0e-12)
    alpha = sum(left * right for left, right in zip(n, t)) / t2
    orthogonal = math.sqrt(sum((left - alpha * right) ** 2 for left, right in zip(n, t))) / math.sqrt(t2)
    total = math.sqrt(sum(value * value for value in n)) / math.sqrt(t2)
    return {"parallel_fraction": alpha, "orthogonal_fraction": orthogonal, "total_fraction": total}


def subset(matrix: list[list[float]], indices: list[int]) -> list[list[float]]:
    return [matrix[index] for index in indices]


def rowwise_summary(null: list[list[float]], target: list[list[float]], indices: list[int]) -> dict[str, float | int]:
    values = []
    for index in indices:
        item = null_decomposition([null[index]], [target[index]])
        values.append(item)
    orthogonal = sorted(item["orthogonal_fraction"] for item in values)
    return {
        "n": len(values),
        "parallel_mean": statistics.mean(item["parallel_fraction"] for item in values),
        "orthogonal_mean": statistics.mean(orthogonal),
        "orthogonal_median": statistics.median(orthogonal),
        "orthogonal_p90": orthogonal[max(0, math.ceil(0.9 * len(orthogonal)) - 1)],
        "total_mean": statistics.mean(item["total_fraction"] for item in values),
    }


def direct_ratio(response: list[list[float]], target: list[list[float]], rows: list[dict[str, Any]], indices: list[int]) -> float:
    values = ("red", "blue", "green", "black", "white", "yellow", "purple", "orange")
    numerator = 0.0
    denominator = 0.0
    for index in indices:
        target_slot = values.index(rows[index]["values"]["target"])
        wrong_slot = values.index(rows[index]["values"]["wrong"])
        numerator += response[index][target_slot] - response[index][wrong_slot]
        denominator += target[index][target_slot] - target[index][wrong_slot]
    return numerator / max(denominator, 1.0e-12)


def main() -> None:
    details = read(DETAILS)
    final = read(FINAL)
    material = [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lookup = {row["row_id"]: row for row in material}
    tensor = details["confirmation"]["response_tensor"]
    rows = [lookup[row_id] for row_id in tensor["row_ids"]]
    target = tensor["target"]
    null = tensor["null"]
    groups: dict[str, dict[str, list[int]]] = {
        "template_slot": defaultdict(list),
        "query_entity": defaultdict(list),
        "target_value": defaultdict(list),
        "unqueried_value": defaultdict(list),
    }
    for index, row in enumerate(rows):
        groups["template_slot"][str(row["template_slot"])].append(index)
        groups["query_entity"][row["query_entity"]].append(index)
        groups["target_value"][row["values"]["target"]].append(index)
        groups["unqueried_value"][row["values"]["null_alt"]].append(index)

    group_results = {}
    for group_name, values in groups.items():
        group_results[group_name] = {}
        for value, indices in values.items():
            group_results[group_name][value] = {
                "pooled_null": null_decomposition(subset(null, indices), subset(target, indices)),
                "rowwise_null": rowwise_summary(null, target, indices),
                "correct": vector_metrics(subset(tensor["correct"], indices), subset(target, indices)),
                "conditional": vector_metrics(subset(tensor["conditional"], indices), subset(tensor["conditional_target"], indices)),
                "direct_correct_ratio": direct_ratio(tensor["correct"], target, rows, indices),
                "direct_wrong_ratio": direct_ratio(tensor["wrong"], target, rows, indices),
            }

    old = read(PHASE1256)
    old_layers = {int(name[1:3]) for name in old["selected_components"]}
    new_layers = {int(name[1:3]) for name in details["selected_components"]}
    result = {
        "phase": 1258,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "posthoc_read_only_diagnostic",
        "cannot_change_formal_verdict": final["verdict"],
        "artifact_inputs": {"details_sha256": sha(DETAILS), "material_sha256": sha(MATERIAL), "final_sha256": sha(FINAL)},
        "formal_pooled_null": details["confirmation"]["null"],
        "all_rows_recomputed_null": null_decomposition(null, target),
        "all_rows_rowwise_null": rowwise_summary(null, target, list(range(len(rows)))),
        "groups": group_results,
        "late_band_replication": {
            "phase1256_layers": sorted(old_layers),
            "phase1258_layers": sorted(new_layers),
            "intersection": sorted(old_layers & new_layers),
            "layer_jaccard": len(old_layers & new_layers) / len(old_layers | new_layers),
            "interpretation": "Independent material and head-level selection repeat an output-proximal layer band; this is not coordinate-level or semantic-mechanism identity.",
        },
        "scope": "Exploratory localization only; no template, row, component, coalition size or threshold may be selected from this output.",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "formal_verdict": final["verdict"],
        "layer_jaccard": result["late_band_replication"]["layer_jaccard"],
        "template_null": {key: value["pooled_null"] for key, value in group_results["template_slot"].items()},
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
