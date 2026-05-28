from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ["qwen3", "glm4", "deepseek7b"]
SOURCES = ["phase290", "phase291"]
PATCH_MODULES = ["attn", "mlp", "resid"]


def log(message: str) -> None:
    print(f"[phase293] {message}", flush=True)


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def ref_values_from_ratio(norm: float, ratio_a: float, ratio_b: float) -> list[float]:
    refs: list[float] = []
    if finite(norm) and finite(ratio_a) and abs(ratio_a) > 1e-12:
        refs.append(float(norm) / float(ratio_a))
    if finite(norm) and finite(ratio_b) and abs(ratio_b) > 1e-12:
        refs.append(float(norm) / float(ratio_b))
    return [x for x in refs if finite(x) and x > 0]


def norm_observations(source: str, row: dict[str, Any]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    if source == "phase290":
        layer = int(row["layer"])
        for module in PATCH_MODULES:
            key = f"patch_{module}_norm"
            if key not in row:
                continue
            observations.append({
                "source": source,
                "scope": f"L{layer}",
                "module": module,
                "metric": f"patch_{module}",
                "norm_key": key,
                "norm": safe_float(row, key, float("nan")),
                "ratio_a": safe_float(row, f"patch_{module}_norm_ratio_to_a", float("nan")),
                "ratio_b": safe_float(row, f"patch_{module}_norm_ratio_to_b", float("nan")),
            })
        for metric in ["next_resid_in", "next_layer_out"]:
            key = f"{metric}_norm"
            if key not in row:
                continue
            observations.append({
                "source": source,
                "scope": f"L{layer}",
                "module": metric,
                "metric": metric,
                "norm_key": key,
                "norm": safe_float(row, key, float("nan")),
                "ratio_a": safe_float(row, f"{metric}_norm_ratio_to_a", float("nan")),
                "ratio_b": safe_float(row, f"{metric}_norm_ratio_to_b", float("nan")),
            })
        return observations

    # Phase 291 stores one patch norm per patched layer inside the block.
    for key, value in row.items():
        if not key.startswith("L") or not key.endswith("_norm"):
            continue
        # Example: L20_attn_norm.
        parts = key.split("_")
        if len(parts) < 3:
            continue
        layer = parts[0]
        module = parts[1]
        if module not in PATCH_MODULES:
            continue
        observations.append({
            "source": source,
            "scope": layer,
            "module": module,
            "metric": f"patch_{module}",
            "norm_key": key,
            "norm": safe_float(row, key, float("nan")),
            "ratio_a": safe_float(row, f"{layer}_{module}_norm_ratio_to_a", float("nan")),
            "ratio_b": safe_float(row, f"{layer}_{module}_norm_ratio_to_b", float("nan")),
        })
    return observations


def build_reference_stats(rows_by_model_source: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str, str, str], dict[str, float]]:
    refs: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for (model, source), rows in rows_by_model_source.items():
        for row in rows:
            for obs in norm_observations(source, row):
                key = (model, source, str(obs["scope"]), str(obs["module"]))
                refs[key].extend(ref_values_from_ratio(float(obs["norm"]), float(obs["ratio_a"]), float(obs["ratio_b"])))
    stats: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key, values in refs.items():
        vals = [v for v in values if finite(v)]
        stats[key] = {
            "mean": mean(vals),
            "std": std(vals),
            "n": float(len(vals)),
            "min": min(vals) if vals else 0.0,
            "max": max(vals) if vals else 0.0,
        }
    return stats


def zscore(value: float, stat: dict[str, float]) -> float:
    sigma = stat.get("std", 0.0)
    if not finite(value) or sigma <= 1e-12:
        return 0.0
    return (value - stat.get("mean", 0.0)) / sigma


def numeric_illegal(row: dict[str, Any], source: str) -> bool:
    keys = ["finite", "logits_finite"]
    if source == "phase290":
        keys.extend(["next_resid_in_finite", "next_layer_out_finite"])
        keys.extend([f"patch_{module}_finite" for module in PATCH_MODULES])
    else:
        keys.extend([key for key in row if key.startswith("L") and key.endswith("_finite")])
    for key in keys:
        if key in row and safe_float(row, key, 1.0) < 0.5:
            return True
    return False


def row_scope(source: str, row: dict[str, Any]) -> str:
    if source == "phase290":
        return f"L{int(row['layer'])}"
    return str(row["block"])


def row_scope_key(source: str, row: dict[str, Any]) -> tuple[Any, ...]:
    if source == "phase290":
        return (
            row.get("category", ""),
            row.get("subtype", ""),
            row.get("pair", ""),
            int(row["layer"]),
        )
    return (
        row.get("category", ""),
        row.get("subtype", ""),
        row.get("pair", ""),
        str(row["block"]),
    )


def build_both_lookup(rows_by_model_source: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[tuple[str, str, tuple[Any, ...]], dict[str, float]]:
    lookup: dict[tuple[str, str, tuple[Any, ...]], dict[str, float]] = {}
    for (model, source), rows in rows_by_model_source.items():
        for row in rows:
            if str(row.get("patch_type")) != "both" or abs(safe_float(row, "alpha") - 1.0) > 1e-9:
                continue
            key = (model, source, row_scope_key(source, row))
            lookup[key] = {
                "both_progress": safe_float(row, "progress"),
                "both_kl_ratio": safe_float(row, "kl_ratio"),
            }
    return lookup


def analyze_rows(
    rows_by_model_source: dict[tuple[str, str], list[dict[str, Any]]],
    ref_stats: dict[tuple[str, str, str, str], dict[str, float]],
    both_lookup: dict[tuple[str, str, tuple[Any, ...]], dict[str, float]],
    z_threshold: float,
    drop_threshold: float,
    both_min: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_rows: list[dict[str, Any]] = []
    summary_counter: Counter[tuple[str, str, str]] = Counter()
    subtype_counter: Counter[tuple[str, str, str, str, str]] = Counter()

    for (model, source), rows in rows_by_model_source.items():
        for row in rows:
            observations = norm_observations(source, row)
            z_values: list[tuple[str, float]] = []
            norm_ratio_flag = False
            for obs in observations:
                stat = ref_stats.get((model, source, str(obs["scope"]), str(obs["module"])))
                z = zscore(float(obs["norm"]), stat or {})
                z_values.append((f"{obs['scope']}.{obs['module']}", z))
                ratio_a = float(obs["ratio_a"])
                ratio_b = float(obs["ratio_b"])
                if finite(ratio_a) and (ratio_a < 0.5 or ratio_a > 2.0):
                    norm_ratio_flag = True
                if finite(ratio_b) and (ratio_b < 0.5 or ratio_b > 2.0):
                    norm_ratio_flag = True
            max_metric, max_z = ("", 0.0)
            if z_values:
                max_metric, max_z = max(z_values, key=lambda item: abs(item[1]))
            num_bad = numeric_illegal(row, source)
            off_manifold = num_bad or norm_ratio_flag or abs(max_z) >= z_threshold or safe_float(row, "norm_illegal") >= 0.5

            both = both_lookup.get((model, source, row_scope_key(source, row)), {})
            both_progress = both.get("both_progress", 0.0)
            both_kl = both.get("both_kl_ratio", 0.0)
            progress = safe_float(row, "progress")
            progress_drop = both_progress - progress
            kl = safe_float(row, "kl_ratio")
            kl_ratio_vs_both = kl / max(both_kl, 1e-8) if both_kl > 0 else 0.0
            is_cross = str(row.get("patch_type", "")).startswith("cross_")
            functional_failure = is_cross and both_progress >= both_min and progress_drop >= drop_threshold

            labels: list[str] = []
            if num_bad:
                labels.append("numeric_illegal")
            if functional_failure and off_manifold:
                labels.append("off_manifold_functional_failure")
            elif functional_failure:
                labels.append("norm_normal_functional_failure")
            elif off_manifold:
                labels.append("off_manifold_no_drop")

            for label in labels:
                summary_counter[(model, source, label)] += 1
                subtype_counter[(model, source, str(row.get("category", "")), str(row.get("subtype", "")), label)] += 1

            if labels:
                event_rows.append({
                    "model": model,
                    "source": source,
                    "scope": row_scope(source, row),
                    "category": row.get("category", ""),
                    "subtype": row.get("subtype", ""),
                    "pair": row.get("pair", ""),
                    "patch_type": row.get("patch_type", ""),
                    "alpha": safe_float(row, "alpha"),
                    "progress": progress,
                    "both_progress": both_progress,
                    "progress_drop": progress_drop,
                    "kl_ratio": kl,
                    "both_kl_ratio": both_kl,
                    "kl_ratio_vs_both": kl_ratio_vs_both,
                    "max_abs_norm_z": abs(max_z),
                    "max_norm_z": max_z,
                    "max_z_metric": max_metric,
                    "norm_ratio_flag": 1.0 if norm_ratio_flag else 0.0,
                    "numeric_illegal": 1.0 if num_bad else 0.0,
                    "off_manifold": 1.0 if off_manifold else 0.0,
                    "functional_failure": 1.0 if functional_failure else 0.0,
                    "labels": "|".join(labels),
                })

    summary_rows = [
        {"model": model, "source": source, "label": label, "rows": count}
        for (model, source, label), count in sorted(summary_counter.items())
    ]
    subtype_rows = [
        {"model": model, "source": source, "category": category, "subtype": subtype, "label": label, "rows": count}
        for (model, source, category, subtype, label), count in sorted(subtype_counter.items())
    ]
    return event_rows, summary_rows, subtype_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase290-dir", default=str(REPO_ROOT / "results" / "gpt5_phase290_contract_break_full"))
    parser.add_argument("--phase291-dir", default=str(REPO_ROOT / "results" / "gpt5_phase291_block_contract_full"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase293_naturalness"))
    parser.add_argument("--z-threshold", type=float, default=4.0)
    parser.add_argument("--drop-threshold", type=float, default=0.5)
    parser.add_argument("--both-min", type=float, default=0.4)
    args = parser.parse_args()

    rows_by_model_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for model in MODELS:
        p290 = Path(args.phase290_dir) / f"{model}_phase290_contract_break_scan.json"
        p291 = Path(args.phase291_dir) / f"{model}_phase291_block_contract_scan.json"
        rows_by_model_source[(model, "phase290")] = load_json(p290)["results"]
        rows_by_model_source[(model, "phase291")] = load_json(p291)["results"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_stats = build_reference_stats(rows_by_model_source)
    both_lookup = build_both_lookup(rows_by_model_source)
    event_rows, summary_rows, subtype_rows = analyze_rows(
        rows_by_model_source,
        ref_stats,
        both_lookup,
        z_threshold=args.z_threshold,
        drop_threshold=args.drop_threshold,
        both_min=args.both_min,
    )

    ref_rows = []
    for (model, source, scope, module), stat in sorted(ref_stats.items()):
        ref_rows.append({
            "model": model,
            "source": source,
            "scope": scope,
            "module": module,
            "n_refs": stat["n"],
            "mean": stat["mean"],
            "std": stat["std"],
            "min": stat["min"],
            "max": stat["max"],
        })

    write_csv(
        out_dir / "natural_norm_reference.csv",
        [{k: fmt(v) for k, v in row.items()} for row in ref_rows],
        ["model", "source", "scope", "module", "n_refs", "mean", "std", "min", "max"],
    )
    write_csv(
        out_dir / "naturalness_events.csv",
        [{k: fmt(v) for k, v in row.items()} for row in event_rows],
        [
            "model", "source", "scope", "category", "subtype", "pair", "patch_type", "alpha",
            "progress", "both_progress", "progress_drop", "kl_ratio", "both_kl_ratio", "kl_ratio_vs_both",
            "max_abs_norm_z", "max_norm_z", "max_z_metric", "norm_ratio_flag", "numeric_illegal",
            "off_manifold", "functional_failure", "labels",
        ],
    )
    write_csv(out_dir / "naturalness_summary.csv", summary_rows, ["model", "source", "label", "rows"])
    write_csv(out_dir / "naturalness_subtype_summary.csv", subtype_rows, ["model", "source", "category", "subtype", "label", "rows"])

    report: list[str] = []
    report.append("# Phase 293 Naturalness Report\n")
    report.append("## Inputs\n")
    report.append(f"- phase290_dir: `{args.phase290_dir}`\n")
    report.append(f"- phase291_dir: `{args.phase291_dir}`\n")
    report.append(f"- z_threshold: `{args.z_threshold}`\n")
    report.append(f"- drop_threshold: `{args.drop_threshold}`\n")
    report.append(f"- both_min: `{args.both_min}`\n")
    report.append("\n## Event Summary\n")
    by_model_source: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_model_source[(row["model"], row["source"])].append(row)
    for key in sorted(by_model_source):
        model, source = key
        report.append(f"### {model} / {source}\n")
        for row in by_model_source[key]:
            report.append(f"- {row['label']}: {row['rows']}\n")

    report.append("\n## Top Functional Failures\n")
    failures = [r for r in event_rows if r["functional_failure"] >= 0.5]
    failures.sort(key=lambda r: (float(r["off_manifold"]), float(r["progress_drop"]), float(r["max_abs_norm_z"])), reverse=True)
    for row in failures[:30]:
        report.append(
            f"- {row['model']} {row['source']} {row['scope']} {row['patch_type']} "
            f"{row['subtype']} pair={row['pair']} drop={float(row['progress_drop']):.4f} "
            f"off={int(row['off_manifold'])} z={float(row['max_abs_norm_z']):.2f} labels={row['labels']}\n"
        )

    report.append("\n## Caution\n")
    report.append("- This is norm-based naturalness only; it is not PCA, kNN, or Mahalanobis manifold distance.\n")
    report.append("- A norm-normal functional failure is stronger than an off-manifold failure, but still not final proof of an internal contract.\n")
    (out_dir / "NATURALNESS_REPORT.md").write_text("".join(report), encoding="utf-8")
    (out_dir / "naturalness_stats.json").write_text(
        json.dumps({"reference_stats": {str(k): v for k, v in ref_stats.items()}, "summary": summary_rows}, indent=2),
        encoding="utf-8",
    )

    log(f"saved output_dir={out_dir}")
    log(f"reference_rows={len(ref_rows)} event_rows={len(event_rows)} summary_rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
