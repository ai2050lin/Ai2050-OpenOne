#!/usr/bin/env python3
"""Select Phase556 causal probe coordinates from independent factor ledgers."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tests/gpt5/result/phase556_fruit_encoding"
MODELS = ("qwen3", "glm4")
SPLITS = ("discovery", "independent_confirmation")
COMPONENTS = ("layer_input", "attention_output", "mlp_output")
CONFIRMATION_SELECTION_STOP = 48
CANDIDATES_PATH = OUT_DIR / "phase556_causal_candidate_registry.json"
SUMMARY_PATH = OUT_DIR / "phase556_event_analysis_summary.json"

MECHANISMS = {
    "category_reuse": {
        "target_effect": "category|query=0",
        "control_effect": "category|query=1",
    },
    "attribute_binding": {
        "target_effect": "binding|query=1",
        "control_effect": "binding|query=0",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def rows_for(model: str, split: str) -> list[dict[str, Any]]:
    path = OUT_DIR / "event_collection" / model / split / "phase556_event_rows.jsonl"
    return read_jsonl(path)


def aggregate(rows: list[dict[str, Any]], effect: str) -> dict[tuple[str, int], dict[str, float]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["component"] in COMPONENTS:
            grouped[(row["component"], int(row["layer"]))].append(row)
    report: dict[tuple[str, int], dict[str, float]] = {}
    for key, values in grouped.items():
        target = [row["conditional_effects"][effect]["relative_effect_norm"] for row in values]
        report[key] = {
            "n": len(values),
            "median": statistics.median(target),
            "minimum": min(target),
            "maximum": max(target),
        }
    return report


def analyze() -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    model_reports: list[dict[str, Any]] = []
    for model in MODELS:
        raw_split_rows = {split: rows_for(model, split) for split in SPLITS}
        event_dtypes = sorted({
            str(row.get("torch_dtype", "torch.float16_default_before_dtype_ledger"))
            for rows in raw_split_rows.values() for row in rows
        })
        if event_dtypes != ["torch.bfloat16"]:
            raise RuntimeError(f"Phase556 event dtype drift for {model}: {event_dtypes}")
        semantic_positions = sorted({
            str(row.get(
                "semantic_position",
                "query_end_inferred_from_pre_coordinate_ledger_run"
                if model == "qwen3" else "unspecified",
            ))
            for rows in raw_split_rows.values() for row in rows
        })
        split_rows = {
            "discovery": raw_split_rows["discovery"],
            "independent_confirmation": [
                row for row in raw_split_rows["independent_confirmation"]
                if int(row["world_index"]) < CONFIRMATION_SELECTION_STOP
            ],
        }
        causal_holdout_rows = [
            row for row in raw_split_rows["independent_confirmation"]
            if int(row["world_index"]) >= CONFIRMATION_SELECTION_STOP
        ]
        layer_count = max(row["layer_count"] for row in split_rows["discovery"])
        model_candidate_count = 0
        for mechanism, spec in MECHANISMS.items():
            target = {
                split: aggregate(split_rows[split], spec["target_effect"])
                for split in SPLITS
            }
            control = {
                split: aggregate(split_rows[split], spec["control_effect"])
                for split in SPLITS
            }
            for component in COMPONENTS:
                ranked = []
                for layer in range(layer_count):
                    key = (component, layer)
                    discovery_target = target["discovery"][key]["median"]
                    confirmation_target = target["independent_confirmation"][key]["median"]
                    discovery_control = control["discovery"][key]["median"]
                    confirmation_control = control["independent_confirmation"][key]["median"]
                    replicated_target = min(discovery_target, confirmation_target)
                    replicated_control = max(discovery_control, confirmation_control)
                    ranked.append({
                        "layer": layer,
                        "replicated_target": replicated_target,
                        "replicated_control": replicated_control,
                        "replicated_specificity_margin": replicated_target - replicated_control,
                        "discovery_target": discovery_target,
                        "confirmation_target": confirmation_target,
                        "discovery_control": discovery_control,
                        "confirmation_control": confirmation_control,
                    })
                ranked.sort(
                    key=lambda row: (row["replicated_specificity_margin"], row["replicated_target"]),
                    reverse=True,
                )
                for component_rank, item in enumerate(ranked[:1], 1):
                    candidates.append({
                        "schema_version": "phase556_causal_candidate.v1",
                        "phase_id": "Phase556",
                        "created_at": now(),
                        "model": model,
                        "event_collection_dtypes": event_dtypes,
                        "event_semantic_positions": semantic_positions,
                        "mechanism": mechanism,
                        "component": component,
                        "layer": item["layer"],
                        "layer_count": layer_count,
                        "relative_depth": item["layer"] / max(1, layer_count - 1),
                        "component_rank": component_rank,
                        **item,
                        "selected_for_intervention_only": True,
                        "observer_claim": False,
                        "compute_edge": False,
                        "causal": False,
                        "sealed": False,
                    })
                    model_candidate_count += 1
        model_reports.append({
            "model": model,
            "event_collection_dtypes": event_dtypes,
            "event_semantic_positions": semantic_positions,
            "layer_count": layer_count,
            "discovery_anchor_count": len({row["anchor_id"] for row in split_rows["discovery"]}),
            "confirmation_anchor_count": len({row["anchor_id"] for row in split_rows["independent_confirmation"]}),
            "causal_holdout_anchor_count": len({row["anchor_id"] for row in causal_holdout_rows}),
            "candidate_count": model_candidate_count,
            "max_component_ledger_relative_error": max(
                row["component_ledger_relative_error"]
                for rows in split_rows.values() for row in rows
            ),
        })
    registry = {
        "schema_version": "phase556_causal_candidate_registry.v1",
        "phase_id": "Phase556",
        "created_at": now(),
        "candidate_count": len(candidates),
        "selection_rule": (
            "top_one_per_model_mechanism_component_by_min_discovery_and_confirmation_world_0_47_"
            "target_minus_max_control"
        ),
        "confirmation_selection_world_indices": list(range(CONFIRMATION_SELECTION_STOP)),
        "causal_holdout_world_indices": list(range(CONFIRMATION_SELECTION_STOP, 96)),
        "selection_uses_sealed": False,
        "candidates_are_mechanisms": False,
        "candidates": candidates,
    }
    write_json(CANDIDATES_PATH, registry)
    summary = {
        "schema_version": "phase556_event_analysis_summary.v1",
        "phase_id": "Phase556",
        "created_at": now(),
        "model_reports": model_reports,
        "candidate_count": len(candidates),
        "causal_intervention_authorized": bool(candidates),
        "candidate_registry_path": str(CANDIDATES_PATH.relative_to(ROOT)),
        "sealed_split_read": False,
        "observer_only": True,
        "mechanism_claim_count": 0,
    }
    write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    analyze()
