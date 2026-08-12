"""Summarize singleton head-response morphology without adding causal claims."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1013_head_response_morphology"
)
PHASE1008_ROOT = (
    ROOT
    / "tests"
    / "glm5"
    / "result"
    / "phase1008_global_response_atlas"
    / "refinement_final"
)
MODELS = ("qwen3", "glm4", "deepseek7b")
SOURCE_OPERATION = {"F": "B", "Q": "Q"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def historical_selections() -> dict[tuple[str, str, int], set[int]]:
    result: dict[tuple[str, str, int], set[int]] = {}
    for model in ("qwen3", "glm4"):
        bundle = read_json(
            PHASE1008_ROOT / model / "causal_selection.json"
        )
        for row in bundle["selections"]:
            result[
                (
                    model,
                    str(row["operation"]),
                    int(row["layer"]),
                )
            ] = {int(head) for head in row["selected_heads"]}
    return result


def main() -> None:
    final_summary = read_json(RESULT_ROOT / "summary.json")
    selections = read_jsonl(
        RESULT_ROOT / "discovery_frozen_heads.jsonl"
    )
    historical = historical_selections()

    by_region: dict[tuple[str, str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    by_physical: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for row in selections:
        region = (
            str(row["model"]),
            str(row["operation"]),
            int(row["depth"]),
        )
        by_region[region].append(row)
        by_physical[
            (
                str(row["model"]),
                int(row["depth"]),
                int(row["head"]),
            )
        ].add(str(row["operation"]))

    regions = []
    historical_comparisons = []
    for (model, operation, depth), rows in sorted(by_region.items()):
        rows = sorted(rows, key=lambda row: int(row["head"]))
        regions.append(
            {
                "model": model,
                "operation": operation,
                "depth": depth,
                "selected_heads": [
                    int(row["head"]) for row in rows
                ],
                "head_count": len(rows),
                "confirming_heads": [
                    int(row["head"])
                    for row in rows
                    if int(row["confirmation_panel_count"]) > 0
                ],
                "maximum_discovery_panel_count": max(
                    int(row["discovery_panel_count"]) for row in rows
                ),
                "maximum_confirmation_panel_count": max(
                    int(row["confirmation_panel_count"]) for row in rows
                ),
                "maximum_discovery_all_axis_panel_count": max(
                    int(row["discovery_all_axis_panel_count"])
                    for row in rows
                ),
                "maximum_confirmation_all_axis_panel_count": max(
                    int(row["confirmation_all_axis_panel_count"])
                    for row in rows
                ),
            }
        )
        source_operation = SOURCE_OPERATION.get(operation)
        old_key = (
            model,
            source_operation,
            depth,
        )
        if source_operation is None or old_key not in historical:
            continue
        current_heads = {int(row["head"]) for row in rows}
        old_heads = historical[old_key]
        historical_comparisons.append(
            {
                "model": model,
                "phase1013_operation": operation,
                "phase1008_operation": source_operation,
                "depth": depth,
                "phase1013_heads": sorted(current_heads),
                "phase1008_heads": sorted(old_heads),
                "overlap_heads": sorted(current_heads & old_heads),
                "jaccard": (
                    len(current_heads & old_heads)
                    / len(current_heads | old_heads)
                    if current_heads | old_heads
                    else None
                ),
                "interpretation_limit": (
                    "same physical coordinate and response recurrence; "
                    "not causal replication"
                ),
            }
        )

    shared_operation_coordinates = [
        {
            "model": model,
            "depth": depth,
            "head": head,
            "operations": sorted(operations),
        }
        for (model, depth, head), operations in sorted(
            by_physical.items()
        )
        if len(operations) > 1
    ]

    model_summaries = {}
    for model in MODELS:
        rows = [row for row in selections if row["model"] == model]
        scan_summary = read_json(
            RESULT_ROOT / "scan" / model / "summary.json"
        )
        model_summaries[model] = {
            "model_forward_count": int(
                scan_summary["model_forward_count"]
            ),
            "event_count": int(scan_summary["event_count"]),
            "scalar_measurement_count": int(
                scan_summary["scalar_measurement_count"]
            ),
            "identity_maximum": float(
                scan_summary["identity_maximum"]
            ),
            "frozen_head_count": len(rows),
            "confirming_head_count": sum(
                int(row["confirmation_panel_count"]) > 0
                for row in rows
            ),
            "all_axis_discovery_head_count": sum(
                int(row["discovery_all_axis_panel_count"]) > 0
                for row in rows
            ),
            "all_axis_confirmation_head_count": sum(
                int(row["confirmation_all_axis_panel_count"]) > 0
                for row in rows
            ),
        }

    result = {
        "schema_version": "phase1013_head_response_analysis.v1",
        "phase": 1013,
        "method": (
            "post-registered descriptive analysis of discovery-frozen "
            "singleton head responses"
        ),
        "selection_used_confirmation": False,
        "model_summaries": model_summaries,
        "regions": regions,
        "historical_coordinate_comparisons": historical_comparisons,
        "shared_operation_coordinates": shared_operation_coordinates,
        "threshold_sensitivity": final_summary[
            "threshold_sensitivity"
        ],
        "claim_limits": [
            "a repeated head response is not necessity or sufficiency",
            "historical overlap is meaningful only at the same model, "
            "layer, and head coordinate",
            "head-number similarity across models is meaningless",
            "selection used discovery only; confirmation only updates "
            "descriptive evidence",
        ],
    }
    output = RESULT_ROOT / "analysis" / "summary.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
