#!/usr/bin/env python3
"""Phase489 cross-model analysis and physical authorization freeze.

Analysis only. Recomputes the key ledgers from Phase488 rows, corrects the
recoverability-only gate name, and freezes which models may enter open
relation-geometry collection. It never reads a sealed split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "tests" / "gpt5" / "result" / "phase488_multimodel_three_channel_behavior"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase489_three_channel_behavior_analysis"
OUT_PATH = OUT_DIR / "phase489_three_channel_behavior_analysis.json"
AUTH_PATH = OUT_DIR / "phase489_open_physical_authorization.json"
MODELS = ("qwen3", "glm4", "deepseek7b")
NATIVE_TRACKS = ("identity", "native_plain_candidate")
Z = 1.96


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    if not n:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    k = sum(bool(row[field]) for row in rows)
    lcb, ucb = wilson(k, len(rows))
    return {"n": len(rows), "count": k, "rate": k / len(rows) if rows else 0.0, "lcb95": lcb, "ucb95": ucb}


def unique_semantic(rows: list[dict[str, Any]], track: str) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["track"] != track:
            continue
        previous = out.get(row["semantic_case_id"])
        if previous is not None and previous["semantic_candidate_correct"] != row["semantic_candidate_correct"]:
            raise RuntimeError("Semantic mapping duplicate disagreement")
        out[row["semantic_case_id"]] = row
    return list(out.values())


def native_intersection(rows: list[dict[str, Any]], field: str, semantic: bool) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["track"] not in NATIVE_TRACKS:
            continue
        key = row["semantic_case_id"] if semantic else row["sample_id"]
        if row["track"] in grouped[key]:
            if semantic:
                continue
            raise RuntimeError(f"Duplicate native row {key}/{row['track']}")
        grouped[key][row["track"]] = row
    complete = [pair for pair in grouped.values() if set(pair) == set(NATIVE_TRACKS)]
    k = sum(all(pair[track][field] for track in NATIVE_TRACKS) for pair in complete)
    lcb, ucb = wilson(k, len(complete))
    return {"n": len(complete), "count": k, "rate": k / len(complete), "lcb95": lcb, "ucb95": ucb}


def factor_report(rows: list[dict[str, Any]], factor: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for track in {row["track"] for row in rows}:
        for row in unique_semantic(rows, track):
            groups[(track, row[factor])].append(row)
    reports = []
    for (track, value), items in sorted(groups.items(), key=lambda item: str(item[0])):
        reports.append({"track": track, factor: value, **metric(items, "semantic_candidate_correct")})
    return reports


def analyze_model(model: str) -> dict[str, Any]:
    rows = load_jsonl(IN_DIR / f"phase488_{model}_rows.jsonl")
    original = load_json(IN_DIR / f"phase488_{model}_summary.json")
    if len(rows) != 2048 or any(row["model"] != model for row in rows):
        raise RuntimeError(f"Phase488 {model} count or model ledger failed")
    tracks = {}
    for track in sorted({row["track"] for row in rows}):
        event_rows = [row for row in rows if row["track"] == track]
        semantic_rows = unique_semantic(rows, track)
        unrecoverable = [row for row in event_rows if row["semantic_value"] is None]
        unrecoverable_lcb, unrecoverable_ucb = wilson(len(unrecoverable), len(event_rows))
        tracks[track] = {
            "semantic_candidate": metric(semantic_rows, "semantic_candidate_correct"),
            "label_candidate": metric(event_rows, "label_candidate_correct"),
            "semantic_generation": metric(event_rows, "semantic_generation_correct"),
            "label_generation": metric(event_rows, "label_generation_correct"),
            "strict_event": metric(event_rows, "strict_event_correct"),
            "unrecoverable": {
                "n": len(event_rows),
                "count": len(unrecoverable),
                "rate": len(unrecoverable) / len(event_rows),
                "lcb95": unrecoverable_lcb,
                "ucb95": unrecoverable_ucb,
            },
            "events": dict(Counter(row["event_type"] for row in event_rows)),
        }
    semantic_intersection = native_intersection(rows, "semantic_candidate_correct", semantic=True)
    label_intersection = native_intersection(rows, "label_candidate_correct", semantic=False)
    native_rows = [row for row in rows if row["track"] in NATIVE_TRACKS]
    native_unrecoverable = [row for row in native_rows if row["semantic_value"] is None]
    _lcb, native_unrecoverable_ucb = wilson(len(native_unrecoverable), len(native_rows))
    relation_pass = (
        tracks["identity"]["semantic_candidate"]["lcb95"] >= 0.95
        and tracks["native_plain_candidate"]["semantic_candidate"]["lcb95"] >= 0.95
        and semantic_intersection["lcb95"] >= 0.90
    )
    label_pass = (
        tracks["identity"]["label_candidate"]["lcb95"] >= 0.95
        and tracks["native_plain_candidate"]["label_candidate"]["lcb95"] >= 0.95
        and label_intersection["lcb95"] >= 0.90
    )
    strict_output_pass = (
        tracks["identity"]["strict_event"]["lcb95"] >= 0.90
        and tracks["native_plain_candidate"]["strict_event"]["lcb95"] >= 0.90
        and native_unrecoverable_ucb <= 0.05
    )
    return {
        "model": model,
        "row_count": len(rows),
        "tracks": tracks,
        "native_semantic_intersection": semantic_intersection,
        "native_label_intersection": label_intersection,
        "factor_reports": {
            "family": factor_report(rows, "family"),
            "truth_value": factor_report(rows, "truth_value"),
            "target_slot": factor_report(rows, "target_slot"),
            "mapping_position": factor_report(rows, "mapping_position"),
            "fact_order": factor_report(rows, "fact_order"),
        },
        "gates": {
            "relation_semantic_pass": relation_pass,
            "label_binding_pass": label_pass,
            "strict_output_event_pass": strict_output_pass,
            "recoverability_only_pass": native_unrecoverable_ucb <= 0.05,
        },
        "phase488_gate_name_correction": {
            "old_field": "output_event_behavior_pass",
            "correct_name": "recoverability_only_pass",
            "reason": "parseability without semantic and strict-event correctness is not an output-event pass",
            "old_value": original["gates"]["output_event_behavior_pass"],
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = {model: analyze_model(model) for model in MODELS}
    relation_models = [model for model in MODELS if models[model]["gates"]["relation_semantic_pass"]]
    output_models = [model for model in MODELS if models[model]["gates"]["strict_output_event_pass"]]
    label_models = [model for model in MODELS if models[model]["gates"]["label_binding_pass"]]
    output = {
        "schema_version": "phase489_three_channel_behavior_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "cross_model_behavior_analysis_complete",
        "sealed_split_read": False,
        "total_event_rows": sum(model["row_count"] for model in models.values()),
        "models": models,
        "cross_model_result": {
            "relation_semantic_pass_models": relation_models,
            "label_binding_pass_models": label_models,
            "strict_output_event_pass_models": output_models,
            "all_three_models_share_relation_entry": len(relation_models) == len(MODELS),
            "all_three_models_share_label_binding": len(label_models) == len(MODELS),
            "all_three_models_share_strict_output_event": len(output_models) == len(MODELS),
        },
        "allowed_claims": [
            "Direct relation truth selection is independently qualified for Qwen3 and GLM4 on the current frozen interface.",
            "DS7B does not qualify on the same interface.",
            "Mapped-label selection and strict output events are not qualified for any model.",
            "Order stress remains behaviorally distinct and is excluded from the native-core denominator.",
        ],
        "forbidden_claims": [
            "relation semantics are causally localized",
            "the native relation geometry is already positive",
            "label binding or serialization is closed",
            "the three small models share one universal relation route",
        ],
    }
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authorization = {
        "schema_version": "phase489_open_physical_authorization.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_after_phase488_before_physical_collection",
        "open_relation_geometry_models_in_order": relation_models,
        "open_output_event_map_models": [],
        "reason_output_event_map_closed": "no independent human parser precision qualification and no strict output-event pass",
        "physical_prediction_authorized": False,
        "sealed_read_authorized": False,
        "head_channel_neuron_scan_authorized": False,
        "causal_intervention_authorized": False,
    }
    AUTH_PATH.write_text(json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)
    print(AUTH_PATH)


if __name__ == "__main__":
    main()
