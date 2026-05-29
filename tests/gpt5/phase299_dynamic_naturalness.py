from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values: list[float]) -> float:
    vals = [float(x) for x in values if finite(x)]
    return sum(vals) / len(vals) if vals else 0.0


def std(values: list[float]) -> float:
    vals = [float(x) for x in values if finite(x)]
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return math.sqrt(sum((x - mu) ** 2 for x in vals) / len(vals))


def safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def load_model_rows(input_dir: Path, model: str) -> list[dict[str, Any]]:
    path = input_dir / f"{model}_phase294_dynamic_recompute_pilot.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["results"])


def build_reference(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int, str], dict[str, float]]:
    refs: dict[tuple[str, int, str], dict[tuple[str, int], tuple[float, float]]] = defaultdict(dict)
    for model, rows in rows_by_model.items():
        for row in rows:
            pair = str(row["pair"])
            layer = int(row["layer"])
            patch_type = str(row["patch_type"])
            key = (model, layer, patch_type)
            refs[key][(pair, layer)] = (
                safe_float(row, "a_ref_norm", float("nan")),
                safe_float(row, "b_ref_norm", float("nan")),
            )

    stats: dict[tuple[str, int, str], dict[str, float]] = {}
    for key, values in refs.items():
        flat: list[float] = []
        for a_norm, b_norm in values.values():
            if finite(a_norm) and a_norm > 0:
                flat.append(float(a_norm))
            if finite(b_norm) and b_norm > 0:
                flat.append(float(b_norm))
        stats[key] = {
            "n_refs": float(len(flat)),
            "mean": mean(flat),
            "std": std(flat),
            "min": min(flat) if flat else 0.0,
            "max": max(flat) if flat else 0.0,
        }
    return stats


def zscore(value: float, stat: dict[str, float]) -> float:
    sigma = stat.get("std", 0.0)
    if not finite(value) or sigma <= 1e-12:
        return 0.0
    return (value - stat.get("mean", 0.0)) / sigma


def classify_row(
    row: dict[str, Any],
    stat: dict[str, float],
    z_threshold: float,
    success_threshold: float,
    over_threshold: float,
    negative_threshold: float,
) -> dict[str, Any]:
    patch_norm = safe_float(row, "patch_norm", float("nan"))
    a_ref = safe_float(row, "a_ref_norm", float("nan"))
    b_ref = safe_float(row, "b_ref_norm", float("nan"))
    ratio_a = patch_norm / a_ref if finite(patch_norm) and finite(a_ref) and abs(a_ref) > 1e-12 else float("nan")
    ratio_b = patch_norm / b_ref if finite(patch_norm) and finite(b_ref) and abs(b_ref) > 1e-12 else float("nan")
    norm_z = zscore(patch_norm, stat)
    finite_flag = safe_float(row, "finite", 1.0) >= 0.5 and safe_float(row, "logits_finite", 1.0) >= 0.5
    progress = safe_float(row, "progress")

    ratio_bad = False
    for ratio in [ratio_a, ratio_b]:
        if finite(ratio) and (ratio < 0.5 or ratio > 2.0):
            ratio_bad = True

    off_manifold = (not finite_flag) or ratio_bad or abs(norm_z) >= z_threshold
    high_progress = progress >= success_threshold
    over_conversion = progress >= over_threshold
    negative_progress = progress <= negative_threshold

    labels: list[str] = []
    if not finite_flag:
        labels.append("numeric_illegal")
    if off_manifold:
        labels.append("off_manifold")
    if high_progress and off_manifold:
        labels.append("off_manifold_high_progress")
    elif high_progress:
        labels.append("on_manifold_high_progress")
    if over_conversion and off_manifold:
        labels.append("off_manifold_over_conversion")
    elif over_conversion:
        labels.append("on_manifold_over_conversion")
    if negative_progress and off_manifold:
        labels.append("off_manifold_negative_progress")
    elif negative_progress:
        labels.append("on_manifold_negative_progress")

    return {
        "patch_norm": patch_norm,
        "a_ref_norm": a_ref,
        "b_ref_norm": b_ref,
        "norm_ratio_to_a": ratio_a,
        "norm_ratio_to_b": ratio_b,
        "norm_z": norm_z,
        "max_abs_norm_z": abs(norm_z),
        "finite": 1.0 if finite_flag else 0.0,
        "off_manifold": 1.0 if off_manifold else 0.0,
        "high_progress": 1.0 if high_progress else 0.0,
        "over_conversion": 1.0 if over_conversion else 0.0,
        "negative_progress": 1.0 if negative_progress else 0.0,
        "labels": "|".join(labels),
    }


def analyze(
    rows_by_model: dict[str, list[dict[str, Any]]],
    stats: dict[tuple[str, int, str], dict[str, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    summary_counter: Counter[tuple[str, str]] = Counter()
    base_counter: Counter[str] = Counter()
    subtype_counter: Counter[tuple[str, str, str, str]] = Counter()
    patch_counter: Counter[tuple[str, str, str]] = Counter()
    layer_counter: Counter[tuple[str, int, str]] = Counter()

    for model, rows in rows_by_model.items():
        for row in rows:
            layer = int(row["layer"])
            patch_type = str(row["patch_type"])
            stat = stats.get((model, layer, patch_type), {})
            cls = classify_row(
                row,
                stat,
                args.z_threshold,
                args.success_threshold,
                args.over_threshold,
                args.negative_threshold,
            )
            labels = [label for label in str(cls["labels"]).split("|") if label]
            base_counter[model] += 1
            for label in labels:
                summary_counter[(model, label)] += 1
                subtype_counter[(model, str(row["category"]), str(row["subtype"]), label)] += 1
                patch_counter[(model, patch_type, label)] += 1
                layer_counter[(model, layer, label)] += 1

            if labels:
                event_rows.append({
                    "model": model,
                    "category": row.get("category", ""),
                    "subtype": row.get("subtype", ""),
                    "pair": row.get("pair", ""),
                    "layer": layer,
                    "patch_type": patch_type,
                    "alpha": safe_float(row, "alpha"),
                    "progress": safe_float(row, "progress"),
                    "kl_ratio": safe_float(row, "kl_ratio"),
                    "logit_delta_ratio": safe_float(row, "logit_delta_ratio"),
                    **cls,
                })

    summary_rows: list[dict[str, Any]] = []
    labels = sorted({label for _, label in summary_counter})
    for model in MODELS:
        total = base_counter[model]
        row: dict[str, Any] = {"model": model, "total_rows": total}
        for label in labels:
            count = summary_counter[(model, label)]
            row[f"{label}_rows"] = count
            row[f"{label}_rate"] = count / max(total, 1)
        summary_rows.append(row)

    subtype_rows = [
        {
            "model": model,
            "category": category,
            "subtype": subtype,
            "label": label,
            "count": count,
        }
        for (model, category, subtype, label), count in sorted(subtype_counter.items())
    ]
    patch_rows = [
        {
            "model": model,
            "patch_type": patch_type,
            "label": label,
            "count": count,
        }
        for (model, patch_type, label), count in sorted(patch_counter.items())
    ]
    layer_rows = [
        {
            "model": model,
            "layer": layer,
            "label": label,
            "count": count,
        }
        for (model, layer, label), count in sorted(layer_counter.items())
    ]
    return event_rows, summary_rows, subtype_rows, patch_rows + layer_rows


def write_reference(path: Path, stats: dict[tuple[str, int, str], dict[str, float]]) -> None:
    rows = []
    for (model, layer, patch_type), stat in sorted(stats.items()):
        rows.append({
            "model": model,
            "layer": layer,
            "patch_type": patch_type,
            **stat,
        })
    write_csv(path, [{k: fmt(v) for k, v in row.items()} for row in rows], ["model", "layer", "patch_type", "n_refs", "mean", "std", "min", "max"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/gpt5_phase298_expanded_dynamic_normal")
    parser.add_argument("--output-dir", default="results/gpt5_phase299_dynamic_naturalness")
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument("--success-threshold", type=float, default=0.8)
    parser.add_argument("--over-threshold", type=float, default=1.05)
    parser.add_argument("--negative-threshold", type=float, default=-0.05)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model = {model: load_model_rows(input_dir, model) for model in MODELS}
    stats = build_reference(rows_by_model)
    event_rows, summary_rows, subtype_rows, patch_layer_rows = analyze(rows_by_model, stats, args)

    write_reference(output_dir / "dynamic_norm_reference.csv", stats)
    event_fields = [
        "model", "category", "subtype", "pair", "layer", "patch_type", "alpha",
        "progress", "kl_ratio", "logit_delta_ratio", "patch_norm", "a_ref_norm",
        "b_ref_norm", "norm_ratio_to_a", "norm_ratio_to_b", "norm_z",
        "max_abs_norm_z", "finite", "off_manifold", "high_progress",
        "over_conversion", "negative_progress", "labels",
    ]
    write_csv(output_dir / "dynamic_naturalness_events.csv", [{k: fmt(v) for k, v in row.items()} for row in event_rows], event_fields)
    write_csv(output_dir / "dynamic_naturalness_summary.csv", [{k: fmt(v) for k, v in row.items()} for row in summary_rows], list(summary_rows[0].keys()))
    write_csv(output_dir / "dynamic_naturalness_subtype_summary.csv", [{k: fmt(v) for k, v in row.items()} for row in subtype_rows], ["model", "category", "subtype", "label", "count"])
    write_csv(output_dir / "dynamic_naturalness_patch_layer_summary.csv", [{k: fmt(v) for k, v in row.items()} for row in patch_layer_rows], sorted(set().union(*(set(row) for row in patch_layer_rows))) if patch_layer_rows else ["model"])

    report = [
        "# Phase 299 Dynamic Naturalness Report\n\n",
        f"- input_dir: `{input_dir}`\n",
        f"- z_threshold: {args.z_threshold}\n",
        f"- success_threshold: {args.success_threshold}\n",
        f"- over_threshold: {args.over_threshold}\n",
        "\n## Summary\n\n",
    ]
    for row in summary_rows:
        report.append(f"### {row['model']}\n")
        report.append(f"- total_rows: {row['total_rows']}\n")
        for key, value in row.items():
            if key.endswith("_rows") or key.endswith("_rate"):
                report.append(f"- {key}: {fmt(value)}\n")
        report.append("\n")
    report.append("## Notes\n\n")
    report.append("- This is norm/z-score naturalness only. It does not compute PCA, kNN, Mahalanobis, entropy, or loss.\n")
    report.append("- on_manifold/off_manifold labels are diagnostic filters, not final mechanism proof.\n")
    (output_dir / "DYNAMIC_NATURALNESS_REPORT.md").write_text("".join(report), encoding="utf-8")

    print(f"saved output_dir={output_dir}")
    print(f"events={len(event_rows)}")


if __name__ == "__main__":
    main()
