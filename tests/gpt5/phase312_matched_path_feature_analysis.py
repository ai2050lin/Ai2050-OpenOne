#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "Phase312"
SCHEMA_VERSION = "3.1.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
SOURCE = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/gpt5/result/phase312_matched_path_feature_analysis"
V2 = ROOT / "tests/gpt5/result/pattern_family_atlas/v2"
LEGACY_V2 = ROOT / "tests/result/pattern_family_atlas/v2"
POSITIONS = ["source", "query", "last"]
COMPONENT_FIELDS = {
    "attention": "delta_attn_semantic_margin",
    "mlp": "delta_mlp_semantic_margin",
    "residual": "delta_residual_semantic_margin",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def cosine(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    a, b = xs[:n], ys[:n]
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return round(dot / (na * nb), 6)


def normalized_segment(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm > 1e-12 else [0.0 for _ in values]


def profile_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[float]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["model"]), str(row["case_id"]), str(row["position_role"]), str(row.get("component") or ""))
        for component, field in COMPONENT_FIELDS.items():
            grouped[(key[0], key[1], key[2], component)].append({"layer_index": row["layer_index"], "value": safe_float(row.get(field))})
    profiles: dict[tuple[str, str, str, str], list[float]] = {}
    for key, vals in grouped.items():
        profiles[key] = [safe_float(v["value"]) for v in sorted(vals, key=lambda x: int(x["layer_index"]))]
    return profiles


def sign_flips(values: list[float], epsilon: float) -> int:
    signs = [1 if v > epsilon else -1 if v < -epsilon else 0 for v in values]
    signs = [s for s in signs if s]
    return sum(1 for a, b in zip(signs, signs[1:], strict=False) if a != b)


def build_event_rows(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        by_case[(str(row["model"]), str(row["case_id"]), str(row["position_role"]))].append(row)
    events: list[dict[str, Any]] = []
    for (model, case_id, position), rows in sorted(by_case.items()):
        ordered = sorted(rows, key=lambda r: int(r["layer_index"]))
        meta = ordered[0]
        layer_count = len(ordered)
        for component, field in COMPONENT_FIELDS.items():
            values = [safe_float(r.get(field)) for r in ordered]
            abs_values = [abs(v) for v in values]
            peak_abs = max(abs_values, default=0.0)
            peak_idx = abs_values.index(peak_abs) if abs_values else 0
            threshold = max(0.25, peak_abs * 0.25)
            active = [i for i, v in enumerate(abs_values) if v >= threshold]
            epsilon = max(0.05, peak_abs * 0.05)
            events.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "event_id": f"phase312:event:{model}:{case_id}:{position}:{component}",
                    "model": model,
                    "case_id": case_id,
                    "family_id": meta["family_id"],
                    "mechanism_id": meta["mechanism_id"],
                    "split": meta["split"],
                    "position_role": position,
                    "component": component,
                    "layer_count": layer_count,
                    "onset_layer": active[0] if active else None,
                    "peak_layer": int(ordered[peak_idx]["layer_index"]) if ordered else None,
                    "peak_normalized_depth": round(peak_idx / max(1, layer_count - 1), 6),
                    "peak_signed_value": round(values[peak_idx], 6) if values else 0.0,
                    "peak_absolute_value": round(peak_abs, 6),
                    "cumulative_signed_value": round(sum(values), 6),
                    "positive_sum": round(sum(max(0.0, v) for v in values), 6),
                    "negative_sum": round(sum(min(0.0, v) for v in values), 6),
                    "active_layer_count": len(active),
                    "persistence_rate": round(len(active) / max(1, layer_count), 6),
                    "sign_flip_count": sign_flips(values, epsilon),
                    "profile_l2": round(math.sqrt(sum(v * v for v in values)), 6),
                }
            )
    return events


def build_similarity_rows(cases: list[dict[str, Any]], profiles: dict[tuple[str, str, str, str], list[float]]) -> list[dict[str, Any]]:
    case_map = {(str(r["model"]), str(r["family_id"]), str(r["mechanism_id"]), int(r["item_index"])): r for r in cases}
    mechanisms: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in cases:
        key = (str(row["model"]), str(row["family_id"]))
        if str(row["mechanism_id"]) not in mechanisms[key]:
            mechanisms[key].append(str(row["mechanism_id"]))
    rows: list[dict[str, Any]] = []
    for left in sorted(cases, key=lambda r: (r["model"], r["family_id"], r["mechanism_id"], r["item_index"])):
        model = str(left["model"])
        family = str(left["family_id"])
        mechanism = str(left["mechanism_id"])
        item = int(left["item_index"])
        same = case_map.get((model, family, mechanism, (item + 1) % 5))
        same_target_controls = sorted(
            [
                r
                for r in cases
                if r["model"] == model
                and r["family_id"] == family
                and int(r["item_index"]) == item
                and r["mechanism_id"] != mechanism
                and str(r["target"]).lower() == str(left["target"]).lower()
            ],
            key=lambda r: str(r["mechanism_id"]),
        )
        if same_target_controls:
            control = same_target_controls[0]
            control_mechanism = str(control["mechanism_id"])
            control_target_matched = True
        else:
            mech_list = sorted(mechanisms[(model, family)])
            control_mechanism = mech_list[(mech_list.index(mechanism) + 1) % len(mech_list)]
            control = case_map.get((model, family, control_mechanism, item))
            control_target_matched = False
        if same is None or control is None:
            continue
        for position in POSITIONS:
            for component in COMPONENT_FIELDS:
                left_profile = profiles.get((model, str(left["case_id"]), position, component), [])
                same_profile = profiles.get((model, str(same["case_id"]), position, component), [])
                control_profile = profiles.get((model, str(control["case_id"]), position, component), [])
                same_cos = cosine(left_profile, same_profile)
                control_cos = cosine(left_profile, control_profile)
                same_reuse = (same_cos + 1.0) / 2.0
                control_reuse = (control_cos + 1.0) / 2.0
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "similarity_id": f"phase312:similarity:{model}:{left['case_id']}:{position}:{component}",
                        "model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "left_case_id": left["case_id"],
                        "within_case_id": same["case_id"],
                        "matched_control_case_id": control["case_id"],
                        "matched_control_mechanism_id": control_mechanism,
                        "matched_control_target_matched": control_target_matched,
                        "left_split": left["split"],
                        "within_split": same["split"],
                        "position_role": position,
                        "component": component,
                        "within_path_cosine": same_cos,
                        "matched_control_path_cosine": control_cos,
                        "within_reuse_score": round(same_reuse, 6),
                        "matched_control_reuse_score": round(control_reuse, 6),
                        "adjusted_reuse_score": round(same_reuse - control_reuse, 6),
                    }
                )
    return rows


def case_vector(model: str, case_id: str, profiles: dict[tuple[str, str, str, str], list[float]]) -> list[float]:
    vector: list[float] = []
    for position in POSITIONS:
        for component in COMPONENT_FIELDS:
            vector.extend(normalized_segment(profiles.get((model, case_id, position, component), [])))
    return vector


def prototype(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    n = min(len(v) for v in vectors)
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(n)]


def build_prediction_rows(cases: list[dict[str, Any]], profiles: dict[tuple[str, str, str, str], list[float]]) -> list[dict[str, Any]]:
    vectors = {(str(r["model"]), str(r["case_id"])): case_vector(str(r["model"]), str(r["case_id"]), profiles) for r in cases}
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_cases = [r for r in cases if r["model"] == model]
        train = [r for r in model_cases if r["split"] in {"discovery", "calibration"}]
        heldout = [r for r in model_cases if r["split"] == "heldout"]
        family_prototypes: dict[str, list[float]] = {}
        mechanism_prototypes: dict[tuple[str, str], list[float]] = {}
        for family in sorted({str(r["family_id"]) for r in train}):
            family_prototypes[family] = prototype([vectors[(model, str(r["case_id"]))] for r in train if r["family_id"] == family])
            for mechanism in sorted({str(r["mechanism_id"]) for r in train if r["family_id"] == family}):
                mechanism_prototypes[(family, mechanism)] = prototype(
                    [vectors[(model, str(r["case_id"]))] for r in train if r["family_id"] == family and r["mechanism_id"] == mechanism]
                )
        for case in heldout:
            vector = vectors[(model, str(case["case_id"]))]
            family_scores = {family: cosine(vector, proto) for family, proto in family_prototypes.items()}
            predicted_family = max(family_scores, key=family_scores.get) if family_scores else "unknown"
            true_family = str(case["family_id"])
            mechanism_scores = {
                mechanism: cosine(vector, proto)
                for (family, mechanism), proto in mechanism_prototypes.items()
                if family == true_family
            }
            predicted_mechanism = max(mechanism_scores, key=mechanism_scores.get) if mechanism_scores else "unknown"
            target_matched_mechanisms = {
                str(r["mechanism_id"])
                for r in train
                if r["family_id"] == true_family and str(r["target"]).lower() == str(case["target"]).lower()
            }
            target_conditioned_baseline = (
                1.0 / len(target_matched_mechanisms)
                if target_matched_mechanisms
                else 1.0 / max(1, len(mechanism_scores))
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "prediction_id": f"phase312:heldout:{model}:{case['case_id']}",
                    "model": model,
                    "case_id": case["case_id"],
                    "true_family_id": true_family,
                    "predicted_family_id": predicted_family,
                    "family_correct": predicted_family == true_family,
                    "family_best_cosine": round(family_scores.get(predicted_family, 0.0), 6),
                    "true_mechanism_id": case["mechanism_id"],
                    "predicted_mechanism_id": predicted_mechanism,
                    "mechanism_correct": predicted_mechanism == case["mechanism_id"],
                    "mechanism_best_cosine": round(mechanism_scores.get(predicted_mechanism, 0.0), 6),
                    "family_random_baseline": round(1.0 / max(1, len(family_prototypes)), 6),
                    "mechanism_random_baseline": round(1.0 / max(1, len(mechanism_scores)), 6),
                    "mechanism_target_conditioned_baseline": round(target_conditioned_baseline, 6),
                    "prediction_frozen_split": "heldout",
                }
            )
    return rows


def aggregate_rows(similarity: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in similarity:
        buckets[(str(row["model"]), str(row["family_id"]), str(row["position_role"]), str(row["component"]))].append(row)
    event_map: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        event_map[(str(row["model"]), str(row["family_id"]), str(row["position_role"]), str(row["component"]))].append(row)
    out = []
    for key, vals in sorted(buckets.items()):
        ev = event_map.get(key, [])
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": key[0],
                "family_id": key[1],
                "position_role": key[2],
                "component": key[3],
                "independent_left_cases": len(vals),
                "mean_within_reuse_score": mean_safe([safe_float(r["within_reuse_score"]) for r in vals]),
                "mean_matched_control_reuse_score": mean_safe([safe_float(r["matched_control_reuse_score"]) for r in vals]),
                "mean_adjusted_reuse_score": mean_safe([safe_float(r["adjusted_reuse_score"]) for r in vals]),
                "mean_peak_normalized_depth": mean_safe([safe_float(r["peak_normalized_depth"]) for r in ev]),
                "mean_persistence_rate": mean_safe([safe_float(r["persistence_rate"]) for r in ev]),
                "mean_sign_flip_count": mean_safe([safe_float(r["sign_flip_count"]) for r in ev]),
            }
        )
    return out


def write_report(path: Path, summary: dict[str, Any], aggregates: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase312 Matched Path Feature Analysis",
        "",
        "## Independent Evidence",
        "",
        f"- independent_model_cases: {summary['independent_model_cases']}",
        f"- layer_component_rows: {summary['layer_component_rows']}",
        f"- heldout_prediction_rows: {summary['heldout_prediction_rows']}",
        f"- heldout_family_accuracy: {summary['heldout_family_accuracy']}",
        f"- heldout_mechanism_accuracy: {summary['heldout_mechanism_accuracy']}",
        "",
        "## Adjusted Reuse By Family",
        "",
    ]
    for family, value in summary["adjusted_reuse_by_family"].items():
        lines.append(f"- {family}: {value}")
    lines += ["", "## Model / Family / Position / Component", ""]
    for row in aggregates:
        lines.append(
            f"- {row['model']} / {row['family_id']} / {row['position_role']} / {row['component']}: "
            f"adjusted={row['mean_adjusted_reuse_score']}, peak_depth={row['mean_peak_normalized_depth']}, "
            f"persistence={row['mean_persistence_rate']}"
        )
    lines += [
        "",
        "## Caution",
        "",
        "Adjusted reuse subtracts a same-family, same-item-index mechanism control. It is still observational and is not a causal subspace proof.",
        "Heldout prediction uses frozen item_index=4 cases and simple cosine prototypes; lexical/template leakage remains a possible baseline.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cases = read_jsonl(SOURCE / "phase311_core_language_case_result_rows.jsonl")
    component_rows = read_jsonl(SOURCE / "phase311_core_language_component_rows.jsonl")
    if not cases or not component_rows:
        raise SystemExit("Phase311 collected data is required")
    profiles = profile_index(component_rows)
    events = build_event_rows(component_rows)
    similarity = build_similarity_rows(cases, profiles)
    predictions = build_prediction_rows(cases, profiles)
    aggregates = aggregate_rows(similarity, events)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in similarity:
        family_groups[str(row["family_id"])].append(row)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "independent_model_cases": len(cases),
        "layer_component_rows": len(component_rows),
        "path_event_rows": len(events),
        "matched_similarity_rows": len(similarity),
        "aggregate_rows": len(aggregates),
        "heldout_prediction_rows": len(predictions),
        "heldout_family_accuracy": mean_safe([1.0 if r["family_correct"] else 0.0 for r in predictions]),
        "heldout_mechanism_accuracy": mean_safe([1.0 if r["mechanism_correct"] else 0.0 for r in predictions]),
        "family_random_baseline": mean_safe([safe_float(r["family_random_baseline"]) for r in predictions]),
        "mechanism_random_baseline": mean_safe([safe_float(r["mechanism_random_baseline"]) for r in predictions]),
        "mechanism_target_conditioned_baseline": mean_safe(
            [safe_float(r["mechanism_target_conditioned_baseline"]) for r in predictions]
        ),
        "mean_within_reuse_score": mean_safe([safe_float(r["within_reuse_score"]) for r in similarity]),
        "mean_matched_control_reuse_score": mean_safe([safe_float(r["matched_control_reuse_score"]) for r in similarity]),
        "mean_adjusted_reuse_score": mean_safe([safe_float(r["adjusted_reuse_score"]) for r in similarity]),
        "adjusted_reuse_by_family": {
            family: mean_safe([safe_float(r["adjusted_reuse_score"]) for r in vals]) for family, vals in sorted(family_groups.items())
        },
        "prediction_counts": {
            "family_correct": sum(1 for r in predictions if r["family_correct"]),
            "mechanism_correct": sum(1 for r in predictions if r["mechanism_correct"]),
        },
    }
    write_json(OUT / "phase312_matched_path_feature_summary.json", summary)
    write_jsonl(OUT / "phase312_path_event_rows.jsonl", events)
    write_jsonl(OUT / "phase312_matched_similarity_rows.jsonl", similarity)
    write_jsonl(OUT / "phase312_path_aggregate_rows.jsonl", aggregates)
    write_jsonl(OUT / "phase312_heldout_prediction_rows.jsonl", predictions)
    write_report(OUT / "phase312_report.md", summary, aggregates)
    for base in [V2, LEGACY_V2]:
        write_json(base / "phase312_matched_path_feature_summary.json", summary)
        write_jsonl(base / "phase312_path_event_rows.jsonl", events)
        write_jsonl(base / "phase312_matched_similarity_rows.jsonl", similarity)
        write_jsonl(base / "phase312_path_aggregate_rows.jsonl", aggregates)
        write_jsonl(base / "phase312_heldout_prediction_rows.jsonl", predictions)
        write_report(base / "phase312_report.md", summary, aggregates)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
