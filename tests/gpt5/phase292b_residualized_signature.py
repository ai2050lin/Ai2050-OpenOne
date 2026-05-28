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


def log(message: str) -> None:
    print(f"[phase292b] {message}", flush=True)


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


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def dense_keys(vectors: dict[str, dict[str, float]]) -> list[str]:
    return sorted(set().union(*(set(v) for v in vectors.values())))


def mean_vector(vectors: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    return {key: mean([vector.get(key, 0.0) for vector in vectors]) for key in keys}


def subtract_mean(vectors: dict[str, dict[str, float]], means: dict[str, dict[str, float]], labels: dict[str, str], keys: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        base = means[labels[name]]
        out[name] = {key: vector.get(key, 0.0) - base.get(key, 0.0) for key in keys}
    return out


def model_center_vectors(vectors: dict[str, dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    base = mean_vector(list(vectors.values()), keys)
    return {name: {key: vector.get(key, 0.0) - base.get(key, 0.0) for key in keys} for name, vector in vectors.items()}


def category_center_vectors(vectors: dict[str, dict[str, float]], categories: dict[str, str], keys: list[str]) -> dict[str, dict[str, float]]:
    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for name, vector in vectors.items():
        category = categories.get(name, "")
        by_category[category].append(vector)
        labels[name] = category
    means = {category: mean_vector(items, keys) for category, items in by_category.items()}
    return subtract_mean(vectors, means, labels, keys)


def zscore_vectors(vectors: dict[str, dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    mus = {key: mean([vector.get(key, 0.0) for vector in vectors.values()]) for key in keys}
    sigmas = {key: std([vector.get(key, 0.0) for vector in vectors.values()]) for key in keys}
    out: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        current: dict[str, float] = {}
        for key in keys:
            sigma = sigmas[key]
            if sigma <= 1e-12:
                continue
            current[key] = (vector.get(key, 0.0) - mus[key]) / sigma
        out[name] = current
    return out


def feature_group(key: str) -> str:
    if ".alpha." in key:
        return "alpha_curve"
    if ".layer_pos" in key or ".layer." in key:
        if ".kl_ratio" in key:
            return "layer_kl"
        if ".progress" in key:
            return "layer_progress"
        return "layer_other"
    if ".block_pos" in key or ".block." in key:
        if ".kl_ratio" in key:
            return "block_kl"
        if ".progress" in key:
            return "block_progress"
        return "block_other"
    if "cross_battn_amlp" in key or "cross_aattn_bmlp" in key:
        return "cross_summary"
    if "best_block_width" in key or key.endswith(".width"):
        return "block_width"
    if "best_progress" in key or "mean_progress" in key:
        return "summary_progress"
    if "kl_ratio" in key:
        return "summary_kl"
    if "drop" in key:
        return "summary_drop"
    return "other"


def group_normalize_vectors(vectors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, value in vector.items():
            grouped[feature_group(key)].append((key, float(value)))
        current: dict[str, float] = {}
        for group, items in grouped.items():
            norm = math.sqrt(sum(value * value for _, value in items))
            if norm <= 1e-12:
                continue
            # Every active feature group gets comparable total weight.
            weight = 1.0 / math.sqrt(max(len(grouped), 1))
            for key, value in items:
                current[f"{group}:{key}"] = weight * value / norm
        out[name] = current
    return out


def matrix_rows(names: list[str], vectors: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for a in names:
        row: dict[str, Any] = {"subtype": a}
        for b in names:
            row[b] = f"{cosine(vectors[a], vectors[b]):.6f}"
        rows.append(row)
    return rows


def pair_rows(names: list[str], vectors: dict[str, dict[str, float]], categories: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({
                "a": a,
                "b": b,
                "category_a": categories.get(a, ""),
                "category_b": categories.get(b, ""),
                "same_category": categories.get(a, "") == categories.get(b, ""),
                "similarity": cosine(vectors[a], vectors[b]),
            })
    rows.sort(key=lambda r: r["similarity"], reverse=True)
    return rows


def build_pair_diagnostics(
    names: list[str],
    categories: dict[str, str],
    vector_sets: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            raw = cosine(vector_sets["raw"][a], vector_sets["raw"][b])
            model_resid = cosine(vector_sets["model_centered"][a], vector_sets["model_centered"][b])
            category_resid = cosine(vector_sets["category_centered"][a], vector_sets["category_centered"][b])
            group_norm = cosine(vector_sets["group_normalized"][a], vector_sets["group_normalized"][b])
            zscore = cosine(vector_sets["zscore_model"][a], vector_sets["zscore_model"][b])
            label = "ordinary"
            if raw >= 0.90 and model_resid >= 0.50 and category_resid >= 0.30:
                label = "residual_stable_candidate"
            elif raw >= 0.90 and model_resid < 0.20:
                label = "model_shape_candidate"
            elif raw >= 0.90 and category_resid < 0.10:
                label = "category_shape_candidate"
            elif raw <= 0.60 and model_resid <= 0.0:
                label = "stable_differentiation_candidate"
            rows.append({
                "a": a,
                "b": b,
                "category_a": categories.get(a, ""),
                "category_b": categories.get(b, ""),
                "same_category": categories.get(a, "") == categories.get(b, ""),
                "raw_similarity": raw,
                "model_centered_similarity": model_resid,
                "category_centered_similarity": category_resid,
                "group_normalized_similarity": group_norm,
                "zscore_model_similarity": zscore,
                "min_residual_similarity": min(model_resid, category_resid),
                "label": label,
            })
    rows.sort(key=lambda r: (r["label"], -float(r["raw_similarity"])))
    return rows


def cross_model_rows(all_sets: dict[str, Any], vector_set_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_subtypes = sorted(set().union(*(set(all_sets[m]["subtypes"]) for m in MODELS)))
    for subtype in all_subtypes:
        for i, model_a in enumerate(MODELS):
            for model_b in MODELS[i + 1:]:
                vec_a = all_sets[model_a][vector_set_name].get(subtype)
                vec_b = all_sets[model_b][vector_set_name].get(subtype)
                if vec_a is None or vec_b is None:
                    continue
                rows.append({
                    "subtype": subtype,
                    "category": all_sets[model_a]["categories"].get(subtype, ""),
                    "model_a": model_a,
                    "model_b": model_b,
                    "vector_set": vector_set_name,
                    "similarity": cosine(vec_a, vec_b),
                })
    rows.sort(key=lambda r: (r["subtype"], r["model_a"], r["model_b"], r["vector_set"]))
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPO_ROOT / "results" / "gpt5_phase292_contract_signature" / "contract_signatures.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase292b_residualized_signature"))
    parser.add_argument("--vector-kind", choices=["canonical", "full"], default="canonical")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))
    vector_key = "canonical_vectors" if args.vector_kind == "canonical" else "full_vectors"
    all_sets: dict[str, Any] = {}
    stability_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model in MODELS:
        model_data = data[model]
        names = list(model_data["subtypes"])
        vectors = {
            name: {key: float(value) for key, value in model_data[vector_key][name].items() if finite(value)}
            for name in names
        }
        categories = dict(model_data["categories"])
        keys = dense_keys(vectors)
        vector_sets = {
            "raw": vectors,
            "model_centered": model_center_vectors(vectors, keys),
            "category_centered": category_center_vectors(vectors, categories, keys),
        }
        vector_sets["group_normalized"] = group_normalize_vectors(vectors)
        vector_sets["model_centered_group_normalized"] = group_normalize_vectors(vector_sets["model_centered"])
        vector_sets["zscore_model"] = zscore_vectors(vectors, keys)

        all_sets[model] = {
            "categories": categories,
            "subtypes": names,
            **vector_sets,
        }

        for set_name, set_vectors in vector_sets.items():
            write_csv(out_dir / f"{model}_{set_name}_similarity.csv", matrix_rows(names, set_vectors), ["subtype"] + names)
            pairs = pair_rows(names, set_vectors, categories)
            write_csv(
                out_dir / f"{model}_{set_name}_top_pairs.csv",
                pairs[: args.top_k],
                ["a", "b", "category_a", "category_b", "same_category", "similarity"],
            )
            write_csv(
                out_dir / f"{model}_{set_name}_bottom_pairs.csv",
                list(reversed(pairs[-args.top_k:])),
                ["a", "b", "category_a", "category_b", "same_category", "similarity"],
            )

        diagnostics = build_pair_diagnostics(names, categories, vector_sets)
        for row in diagnostics:
            row["model"] = model
            stability_rows.append(row)
        write_csv(
            out_dir / f"{model}_pair_diagnostics.csv",
            diagnostics,
            [
                "model", "a", "b", "category_a", "category_b", "same_category",
                "raw_similarity", "model_centered_similarity", "category_centered_similarity",
                "group_normalized_similarity", "zscore_model_similarity", "min_residual_similarity", "label",
            ],
        )

        counts = Counter(row["label"] for row in diagnostics)
        raw_values = [float(row["raw_similarity"]) for row in diagnostics]
        model_resid_values = [float(row["model_centered_similarity"]) for row in diagnostics]
        category_resid_values = [float(row["category_centered_similarity"]) for row in diagnostics]
        summary_rows.append({
            "model": model,
            "subtypes": len(names),
            "pairs": len(diagnostics),
            "raw_mean": mean(raw_values),
            "raw_min": min(raw_values) if raw_values else 0.0,
            "raw_max": max(raw_values) if raw_values else 0.0,
            "model_centered_mean": mean(model_resid_values),
            "model_centered_min": min(model_resid_values) if model_resid_values else 0.0,
            "model_centered_max": max(model_resid_values) if model_resid_values else 0.0,
            "category_centered_mean": mean(category_resid_values),
            "category_centered_min": min(category_resid_values) if category_resid_values else 0.0,
            "category_centered_max": max(category_resid_values) if category_resid_values else 0.0,
            "residual_stable_candidate": counts["residual_stable_candidate"],
            "model_shape_candidate": counts["model_shape_candidate"],
            "category_shape_candidate": counts["category_shape_candidate"],
            "stable_differentiation_candidate": counts["stable_differentiation_candidate"],
        })

    write_csv(
        out_dir / "pair_stability_diagnostics.csv",
        stability_rows,
        [
            "model", "a", "b", "category_a", "category_b", "same_category",
            "raw_similarity", "model_centered_similarity", "category_centered_similarity",
            "group_normalized_similarity", "zscore_model_similarity", "min_residual_similarity", "label",
        ],
    )
    write_csv(
        out_dir / "residualized_summary.csv",
        summary_rows,
        [
            "model", "subtypes", "pairs",
            "raw_mean", "raw_min", "raw_max",
            "model_centered_mean", "model_centered_min", "model_centered_max",
            "category_centered_mean", "category_centered_min", "category_centered_max",
            "residual_stable_candidate", "model_shape_candidate", "category_shape_candidate",
            "stable_differentiation_candidate",
        ],
    )

    cross_rows: list[dict[str, Any]] = []
    for vector_set_name in ["raw", "model_centered", "category_centered", "group_normalized", "zscore_model"]:
        cross_rows.extend(cross_model_rows(all_sets, vector_set_name))
    write_csv(out_dir / "cross_model_same_subtype_residual_similarity.csv", cross_rows, ["subtype", "category", "model_a", "model_b", "vector_set", "similarity"])

    (out_dir / "residualized_signatures.json").write_text(json.dumps(all_sets, indent=2), encoding="utf-8")

    report: list[str] = []
    report.append("# Phase 292b Residualized Signature Report\n")
    report.append("## Inputs\n")
    report.append(f"- input: `{input_path}`\n")
    report.append(f"- vector_kind: `{args.vector_kind}`\n")
    report.append("\n## Model Summary\n")
    for row in summary_rows:
        report.append(f"### {row['model']}\n")
        report.append(f"- subtypes: {row['subtypes']}, pairs: {row['pairs']}\n")
        report.append(f"- raw similarity: mean={row['raw_mean']:.4f}, min={row['raw_min']:.4f}, max={row['raw_max']:.4f}\n")
        report.append(f"- model-centered similarity: mean={row['model_centered_mean']:.4f}, min={row['model_centered_min']:.4f}, max={row['model_centered_max']:.4f}\n")
        report.append(f"- category-centered similarity: mean={row['category_centered_mean']:.4f}, min={row['category_centered_min']:.4f}, max={row['category_centered_max']:.4f}\n")
        report.append(
            "- diagnostic labels: "
            f"stable={row['residual_stable_candidate']}, "
            f"model_shape={row['model_shape_candidate']}, "
            f"category_shape={row['category_shape_candidate']}, "
            f"differentiation={row['stable_differentiation_candidate']}\n"
        )
        model_rows = [r for r in stability_rows if r["model"] == row["model"]]
        stable = [r for r in model_rows if r["label"] == "residual_stable_candidate"][:5]
        shape = [r for r in model_rows if r["label"] in {"model_shape_candidate", "category_shape_candidate"}][:5]
        if stable:
            report.append("- top residual-stable candidates:\n")
            for item in stable:
                report.append(
                    f"  - {item['a']} / {item['b']}: "
                    f"raw={item['raw_similarity']:.4f}, "
                    f"model_resid={item['model_centered_similarity']:.4f}, "
                    f"category_resid={item['category_centered_similarity']:.4f}\n"
                )
        if shape:
            report.append("- high-raw but residual-weak candidates:\n")
            for item in shape:
                report.append(
                    f"  - {item['a']} / {item['b']}: "
                    f"label={item['label']}, raw={item['raw_similarity']:.4f}, "
                    f"model_resid={item['model_centered_similarity']:.4f}, "
                    f"category_resid={item['category_centered_similarity']:.4f}\n"
                )

    report.append("\n## Cross Model Same Subtype Means\n")
    by_key: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in cross_rows:
        by_key[(row["model_a"], row["model_b"], row["vector_set"])].append(float(row["similarity"]))
    for key, vals in sorted(by_key.items()):
        report.append(f"- {key[0]} vs {key[1]} / {key[2]}: mean={mean(vals):.4f}, n={len(vals)}\n")
    report.append("\n## Caution\n")
    report.append("- Residualized cosine values can be negative; they measure deviation-pattern similarity, not absolute functional strength.\n")
    report.append("- Diagnostic labels are screening labels, not proof of true reuse or differentiation.\n")
    (out_dir / "RESIDUALIZED_SIGNATURE_REPORT.md").write_text("".join(report), encoding="utf-8")

    # Keep numeric CSV values compact but leave JSON full precision.
    for path in out_dir.glob("*.csv"):
        rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
        if not rows:
            continue
        fieldnames = list(rows[0].keys())
        compact_rows = [{key: fmt(value) for key, value in row.items()} for row in rows]
        write_csv(path, compact_rows, fieldnames)

    log(f"saved output_dir={out_dir}")
    log(f"summary_rows={len(summary_rows)} pair_rows={len(stability_rows)} cross_rows={len(cross_rows)}")


if __name__ == "__main__":
    main()
