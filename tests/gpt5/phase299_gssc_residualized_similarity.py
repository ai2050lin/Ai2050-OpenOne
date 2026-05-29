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


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def dense_keys(vectors: dict[str, dict[str, float]]) -> list[str]:
    return sorted(set().union(*(set(vector) for vector in vectors.values())))


def mean_vector(vectors: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return {key: mean([vector.get(key, 0.0) for vector in vectors]) for key in keys}


def center(vectors: dict[str, dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    base = mean_vector(list(vectors.values()), keys)
    return {
        name: {key: vector.get(key, 0.0) - base.get(key, 0.0) for key in keys}
        for name, vector in vectors.items()
    }


def category_center(vectors: dict[str, dict[str, float]], categories: dict[str, str], keys: list[str]) -> dict[str, dict[str, float]]:
    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    for name, vector in vectors.items():
        by_category[categories.get(name, "")].append(vector)
    bases = {category: mean_vector(items, keys) for category, items in by_category.items()}
    return {
        name: {key: vector.get(key, 0.0) - bases[categories.get(name, "")].get(key, 0.0) for key in keys}
        for name, vector in vectors.items()
    }


def feature_group(key: str) -> str:
    if "dynamic_layer" in key or "p294.layer" in key:
        return "dynamic_layer"
    if "dynamic_alpha" in key or "p294.alpha" in key:
        return "dynamic_alpha"
    if "block_curve" in key or "p291.block" in key:
        return "block_curve"
    if "block_alpha" in key or "p291.alpha" in key:
        return "block_alpha"
    if "layer_curve" in key or "p290.layer" in key:
        return "layer_curve"
    if "single_alpha" in key or "p290.alpha" in key:
        return "single_alpha"
    if "naturalness" in key or "p293" in key:
        return "naturalness"
    if "summary" in key:
        return "summary"
    return "other"


def group_center(vectors: dict[str, dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        groups[feature_group(key)].append(key)
    out = {name: dict(vector) for name, vector in vectors.items()}
    for group_keys in groups.values():
        base = mean_vector(list(vectors.values()), group_keys)
        for name, vector in vectors.items():
            for key in group_keys:
                out[name][key] = vector.get(key, 0.0) - base.get(key, 0.0)
    return out


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


def pair_rows(names: list[str], categories: dict[str, str], vector_sets: dict[str, dict[str, dict[str, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            raw = cosine(vector_sets["raw"][a], vector_sets["raw"][b])
            model_centered = cosine(vector_sets["model_centered"][a], vector_sets["model_centered"][b])
            category_centered = cosine(vector_sets["category_centered"][a], vector_sets["category_centered"][b])
            group_centered = cosine(vector_sets["group_centered"][a], vector_sets["group_centered"][b])
            zscore = cosine(vector_sets["zscore"][a], vector_sets["zscore"][b])
            label = "ordinary"
            if raw >= 0.90 and model_centered >= 0.40 and category_centered >= 0.20:
                label = "residual_stable_reuse_candidate"
            elif raw >= 0.90 and model_centered < 0.10:
                label = "model_curve_artifact_candidate"
            elif raw >= 0.90 and category_centered < 0.05:
                label = "category_curve_artifact_candidate"
            elif raw <= 0.70 and model_centered <= 0.0:
                label = "stable_differentiation_candidate"
            rows.append({
                "a": a,
                "b": b,
                "category_a": categories.get(a, ""),
                "category_b": categories.get(b, ""),
                "same_category": categories.get(a, "") == categories.get(b, ""),
                "raw_similarity": raw,
                "model_centered_similarity": model_centered,
                "category_centered_similarity": category_centered,
                "group_centered_similarity": group_centered,
                "zscore_similarity": zscore,
                "min_centered_similarity": min(model_centered, category_centered, group_centered),
                "label": label,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/gpt5_phase298_expanded_gssc_v1_dynamic/global_contract_maps.json")
    parser.add_argument("--output-dir", default="results/gpt5_phase299_gssc_residualized_dynamic")
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    report: list[str] = ["# Phase 299 GSSC Residualized Dynamic Similarity\n\n"]
    all_pair_rows: list[dict[str, Any]] = []
    for model in MODELS:
        model_data = data[model]
        categories = dict(model_data["categories"])
        names = list(model_data["subtypes"])
        # Use group-normalized vectors from the GSSC builder as the raw comparable signature.
        vectors = {
            name: {key: float(value) for key, value in model_data["group_normalized_vectors"][name].items() if finite(value)}
            for name in names
        }
        keys = dense_keys(vectors)
        z_vectors = {
            name: {key: float(value) for key, value in model_data["zscore_vectors"].get(name, {}).items() if finite(value)}
            for name in names
        }
        vector_sets = {
            "raw": vectors,
            "model_centered": center(vectors, keys),
            "category_centered": category_center(vectors, categories, keys),
            "group_centered": group_center(vectors, keys),
            "zscore": z_vectors,
        }
        rows = pair_rows(names, categories, vector_sets)
        rows_sorted_raw = sorted(rows, key=lambda row: row["raw_similarity"], reverse=True)
        rows_sorted_stable = sorted(rows, key=lambda row: (row["label"] != "residual_stable_reuse_candidate", -row["min_centered_similarity"], -row["raw_similarity"]))
        rows_sorted_diff = sorted(rows, key=lambda row: (row["label"] != "stable_differentiation_candidate", row["raw_similarity"], row["model_centered_similarity"]))
        all_pair_rows.extend({"model": model, **row} for row in rows)

        label_counts = Counter(row["label"] for row in rows)
        raw_values = [float(row["raw_similarity"]) for row in rows]
        model_values = [float(row["model_centered_similarity"]) for row in rows]
        category_values = [float(row["category_centered_similarity"]) for row in rows]
        group_values = [float(row["group_centered_similarity"]) for row in rows]
        summary_rows.append({
            "model": model,
            "pairs": len(rows),
            "raw_mean": mean(raw_values),
            "raw_min": min(raw_values),
            "raw_max": max(raw_values),
            "model_centered_mean": mean(model_values),
            "category_centered_mean": mean(category_values),
            "group_centered_mean": mean(group_values),
            **{f"{label}_count": label_counts[label] for label in sorted(label_counts)},
        })

        fields = list(rows[0].keys())
        write_csv(out_dir / f"{model}_residualized_pair_diagnostics.csv", [{k: fmt(v) for k, v in row.items()} for row in rows], fields)
        write_csv(out_dir / f"{model}_stable_reuse_candidates.csv", [{k: fmt(v) for k, v in row.items()} for row in rows_sorted_stable[: args.top_k]], fields)
        write_csv(out_dir / f"{model}_stable_differentiation_candidates.csv", [{k: fmt(v) for k, v in row.items()} for row in rows_sorted_diff[: args.top_k]], fields)
        write_csv(out_dir / f"{model}_raw_top_pairs.csv", [{k: fmt(v) for k, v in row.items()} for row in rows_sorted_raw[: args.top_k]], fields)

        report.append(f"## {model}\n")
        report.append(f"- raw_mean: {mean(raw_values):.6f}\n")
        report.append(f"- model_centered_mean: {mean(model_values):.6f}\n")
        report.append(f"- category_centered_mean: {mean(category_values):.6f}\n")
        report.append(f"- group_centered_mean: {mean(group_values):.6f}\n")
        report.append(f"- label_counts: {dict(label_counts)}\n")
        report.append("- stable reuse candidates:\n")
        for row in rows_sorted_stable[:5]:
            report.append(
                f"  - {row['a']} / {row['b']}: raw={row['raw_similarity']:.4f}, "
                f"model={row['model_centered_similarity']:.4f}, category={row['category_centered_similarity']:.4f}, "
                f"group={row['group_centered_similarity']:.4f}, label={row['label']}\n"
            )
        report.append("- stable differentiation candidates:\n")
        for row in rows_sorted_diff[:5]:
            report.append(
                f"  - {row['a']} / {row['b']}: raw={row['raw_similarity']:.4f}, "
                f"model={row['model_centered_similarity']:.4f}, category={row['category_centered_similarity']:.4f}, "
                f"group={row['group_centered_similarity']:.4f}, label={row['label']}\n"
            )
        report.append("\n")

    summary_fields = sorted(set().union(*(set(row) for row in summary_rows)))
    write_csv(out_dir / "residualized_similarity_summary.csv", [{k: fmt(v) for k, v in row.items()} for row in summary_rows], summary_fields)
    all_fields = ["model"] + list(next(iter(all_pair_rows)).keys())
    write_csv(out_dir / "all_residualized_pair_diagnostics.csv", [{k: fmt(v) for k, v in row.items()} for row in all_pair_rows], all_fields)
    (out_dir / "RESIDUALIZED_SIMILARITY_REPORT.md").write_text("".join(report), encoding="utf-8")
    print(f"saved output_dir={out_dir}")


if __name__ == "__main__":
    main()
