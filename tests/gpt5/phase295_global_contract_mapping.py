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
PATCH_TYPES = ["attn", "mlp", "both", "resid", "cross_battn_amlp", "cross_aattn_bmlp"]
DYNAMIC_PATCH_TYPES = ["resid_in", "resid_out", "attn_out", "mlp_out"]


def log(message: str) -> None:
    print(f"[phase295] {message}", flush=True)


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except Exception:
        return default
    return value if math.isfinite(value) else default


def group_key(feature: str) -> str:
    if feature.startswith("p290.layer."):
        return "layer_curve"
    if feature.startswith("p290.alpha."):
        return "single_alpha"
    if feature.startswith("p291.block."):
        return "block_curve"
    if feature.startswith("p291.alpha."):
        return "block_alpha"
    if feature.startswith("p293."):
        return "naturalness"
    if feature.startswith("p294.layer."):
        return "dynamic_layer"
    if feature.startswith("p294.alpha."):
        return "dynamic_alpha"
    if feature.startswith("summary."):
        return "summary"
    return "other"


def normalize_group_vectors(vectors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for key, value in vector.items():
            grouped[group_key(key)].append((key, value))
        current: dict[str, float] = {}
        active_groups = [group for group, items in grouped.items() if any(abs(v) > 1e-12 for _, v in items)]
        group_weight = 1.0 / math.sqrt(max(len(active_groups), 1))
        for group, items in grouped.items():
            norm = math.sqrt(sum(value * value for _, value in items))
            if norm <= 1e-12:
                continue
            for key, value in items:
                current[f"{group}:{key}"] = group_weight * value / norm
        out[name] = current
    return out


def zscore_vectors(vectors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted(set().union(*(set(v) for v in vectors.values())))
    mu = {key: mean([v.get(key, 0.0) for v in vectors.values()]) for key in keys}
    sigma = {key: std([v.get(key, 0.0) for v in vectors.values()]) for key in keys}
    out: dict[str, dict[str, float]] = {}
    for name, vector in vectors.items():
        current: dict[str, float] = {}
        for key in keys:
            if sigma[key] <= 1e-12:
                continue
            current[key] = (vector.get(key, 0.0) - mu[key]) / sigma[key]
        out[name] = current
    return out


def aggregate_phase290(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    categories: dict[str, str] = {}
    vectors: dict[str, dict[str, float]] = defaultdict(dict)
    layers_by_sub: dict[str, set[int]] = defaultdict(set)
    by_sub_layer_patch: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_sub_alpha_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subtype = str(row["subtype"])
        categories[subtype] = str(row["category"])
        layer = int(row["layer"])
        patch = str(row["patch_type"])
        layers_by_sub[subtype].add(layer)
        by_sub_alpha_patch[(subtype, str(row["alpha"]), patch)].append(row)
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            by_sub_layer_patch[(subtype, layer, patch)].append(row)

    for subtype, layers in layers_by_sub.items():
        sorted_layers = sorted(layers)
        layer_pos = {layer: idx for idx, layer in enumerate(sorted_layers)}
        denom = max(len(sorted_layers) - 1, 1)
        patch_progress: dict[str, list[float]] = defaultdict(list)
        for layer in sorted_layers:
            pos = layer_pos[layer]
            rel = pos / denom
            vectors[subtype][f"summary.p290.layer_rel.{pos}"] = rel
            for patch in PATCH_TYPES:
                items = by_sub_layer_patch.get((subtype, layer, patch), [])
                if not items:
                    continue
                progress = mean([safe_float(x, "progress") for x in items])
                kl_ratio = mean([safe_float(x, "kl_ratio") for x in items])
                delta = mean([safe_float(x, "logit_delta_ratio") for x in items])
                patch_progress[patch].append(progress)
                vectors[subtype][f"p290.layer.pos{pos}.{patch}.progress"] = progress
                vectors[subtype][f"p290.layer.pos{pos}.{patch}.kl_ratio"] = kl_ratio
                vectors[subtype][f"p290.layer.pos{pos}.{patch}.delta"] = delta
            both = vectors[subtype].get(f"p290.layer.pos{pos}.both.progress", 0.0)
            cross = vectors[subtype].get(f"p290.layer.pos{pos}.cross_battn_amlp.progress", 0.0)
            vectors[subtype][f"p290.layer.pos{pos}.cross_battn_amlp.drop"] = both - cross

        for patch in PATCH_TYPES:
            vals = patch_progress.get(patch, [])
            if vals:
                vectors[subtype][f"summary.p290.{patch}.mean_progress"] = mean(vals)
                vectors[subtype][f"summary.p290.{patch}.best_progress"] = max(vals)

        for (sub, alpha, patch), items in by_sub_alpha_patch.items():
            if sub != subtype:
                continue
            progress = mean([safe_float(x, "progress") for x in items])
            if patch in {"both", "cross_battn_amlp", "attn", "mlp", "resid"}:
                vectors[subtype][f"p290.alpha.{alpha}.{patch}.progress"] = progress
    return dict(vectors), categories


def aggregate_phase291(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = defaultdict(dict)
    by_sub_block_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_sub_alpha_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    blocks_by_sub: dict[str, set[str]] = defaultdict(set)
    width_by_block: dict[str, int] = {}
    for row in rows:
        subtype = str(row["subtype"])
        block = str(row["block"])
        patch = str(row["patch_type"])
        blocks_by_sub[subtype].add(block)
        width_by_block[block] = int(row.get("block_width", 1))
        by_sub_alpha_patch[(subtype, str(row["alpha"]), patch)].append(row)
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            by_sub_block_patch[(subtype, block, patch)].append(row)

    for subtype, blocks in blocks_by_sub.items():
        sorted_blocks = sorted(blocks, key=lambda b: (width_by_block.get(b, 1), b))
        block_pos = {block: idx for idx, block in enumerate(sorted_blocks)}
        patch_progress: dict[str, list[float]] = defaultdict(list)
        for block in sorted_blocks:
            pos = block_pos[block]
            width = width_by_block.get(block, 1)
            vectors[subtype][f"summary.p291.block.pos{pos}.width"] = float(width)
            for patch in PATCH_TYPES:
                items = by_sub_block_patch.get((subtype, block, patch), [])
                if not items:
                    continue
                progress = mean([safe_float(x, "progress") for x in items])
                kl_ratio = mean([safe_float(x, "kl_ratio") for x in items])
                delta = mean([safe_float(x, "logit_delta_ratio") for x in items])
                patch_progress[patch].append(progress)
                vectors[subtype][f"p291.block.pos{pos}.width{width}.{patch}.progress"] = progress
                vectors[subtype][f"p291.block.pos{pos}.width{width}.{patch}.kl_ratio"] = kl_ratio
                vectors[subtype][f"p291.block.pos{pos}.width{width}.{patch}.delta"] = delta
            both = vectors[subtype].get(f"p291.block.pos{pos}.width{width}.both.progress", 0.0)
            cross = vectors[subtype].get(f"p291.block.pos{pos}.width{width}.cross_battn_amlp.progress", 0.0)
            vectors[subtype][f"p291.block.pos{pos}.width{width}.cross_battn_amlp.drop"] = both - cross

        for patch in PATCH_TYPES:
            vals = patch_progress.get(patch, [])
            if vals:
                vectors[subtype][f"summary.p291.{patch}.mean_progress"] = mean(vals)
                vectors[subtype][f"summary.p291.{patch}.best_progress"] = max(vals)

        for (sub, alpha, patch), items in by_sub_alpha_patch.items():
            if sub != subtype:
                continue
            progress = mean([safe_float(x, "progress") for x in items])
            if patch in {"both", "cross_battn_amlp", "attn", "mlp", "resid"}:
                vectors[subtype][f"p291.alpha.{alpha}.{patch}.progress"] = progress
    return dict(vectors)


def aggregate_phase293(events: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = defaultdict(dict)
    counts: Counter[tuple[str, str]] = Counter()
    by_label: Counter[tuple[str, str]] = Counter()
    by_source_label: Counter[tuple[str, str, str]] = Counter()
    by_patch_label: Counter[tuple[str, str, str]] = Counter()
    for row in events:
        subtype = str(row["subtype"])
        labels = [item for item in str(row["labels"]).split("|") if item]
        patch = str(row["patch_type"])
        source = str(row["source"])
        counts[(subtype, source)] += 1
        for label in labels:
            by_label[(subtype, label)] += 1
            by_source_label[(subtype, source, label)] += 1
            by_patch_label[(subtype, patch, label)] += 1

    subtypes = sorted({sub for sub, _ in counts})
    labels = sorted({label for _, label in by_label})
    for subtype in subtypes:
        total = sum(counts[(subtype, source)] for source in ["phase290", "phase291"])
        for label in labels:
            vectors[subtype][f"p293.naturalness.label.{label}.count"] = float(by_label[(subtype, label)])
            vectors[subtype][f"p293.naturalness.label.{label}.rate"] = by_label[(subtype, label)] / max(total, 1)
        for source in ["phase290", "phase291"]:
            source_total = counts[(subtype, source)]
            vectors[subtype][f"p293.naturalness.{source}.event_count"] = float(source_total)
            for label in labels:
                vectors[subtype][f"p293.naturalness.{source}.{label}.rate"] = by_source_label[(subtype, source, label)] / max(source_total, 1)
        for patch in ["cross_battn_amlp", "cross_aattn_bmlp"]:
            patch_total = sum(by_patch_label[(subtype, patch, label)] for label in labels)
            vectors[subtype][f"p293.naturalness.{patch}.event_count"] = float(patch_total)
            for label in labels:
                vectors[subtype][f"p293.naturalness.{patch}.{label}.rate"] = by_patch_label[(subtype, patch, label)] / max(patch_total, 1)
    return dict(vectors)


def aggregate_phase294(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    vectors: dict[str, dict[str, float]] = defaultdict(dict)
    layers_by_sub: dict[str, set[int]] = defaultdict(set)
    by_sub_layer_patch: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_sub_alpha_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subtype = str(row["subtype"])
        layer = int(row["layer"])
        patch = str(row["patch_type"])
        layers_by_sub[subtype].add(layer)
        by_sub_layer_patch[(subtype, layer, patch)].append(row)
        by_sub_alpha_patch[(subtype, str(row["alpha"]), patch)].append(row)

    for subtype, layers in layers_by_sub.items():
        sorted_layers = sorted(layers)
        layer_pos = {layer: idx for idx, layer in enumerate(sorted_layers)}
        patch_progress: dict[str, list[float]] = defaultdict(list)
        for layer in sorted_layers:
            pos = layer_pos[layer]
            for patch in DYNAMIC_PATCH_TYPES:
                items = [x for x in by_sub_layer_patch.get((subtype, layer, patch), []) if abs(float(x["alpha"]) - 1.0) < 1e-9]
                if not items:
                    continue
                progress = mean([safe_float(x, "progress") for x in items])
                kl_ratio = mean([safe_float(x, "kl_ratio") for x in items])
                delta = mean([safe_float(x, "logit_delta_ratio") for x in items])
                patch_progress[patch].append(progress)
                vectors[subtype][f"p294.layer.pos{pos}.{patch}.progress"] = progress
                vectors[subtype][f"p294.layer.pos{pos}.{patch}.kl_ratio"] = kl_ratio
                vectors[subtype][f"p294.layer.pos{pos}.{patch}.delta"] = delta
        for patch in DYNAMIC_PATCH_TYPES:
            vals = patch_progress.get(patch, [])
            if vals:
                vectors[subtype][f"summary.p294.{patch}.mean_progress"] = mean(vals)
                vectors[subtype][f"summary.p294.{patch}.best_progress"] = max(vals)
        for (sub, alpha, patch), items in by_sub_alpha_patch.items():
            if sub != subtype:
                continue
            progress = mean([safe_float(x, "progress") for x in items])
            vectors[subtype][f"p294.alpha.{alpha}.{patch}.progress"] = progress
    return dict(vectors)


def merge_vectors(*vectors: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for vector in vectors:
        out.update(vector)
    return out


def group_subset(vector: dict[str, float], group: str) -> dict[str, float]:
    prefix = f"{group}:"
    return {
        key: value for key, value in vector.items()
        if key.startswith(prefix) or group_key(key) == group
    }


def pair_rows(names: list[str], vectors: dict[str, dict[str, float]], categories: dict[str, str]) -> list[dict[str, Any]]:
    group_names = ["layer_curve", "single_alpha", "block_curve", "block_alpha", "naturalness", "dynamic_layer", "dynamic_alpha", "summary"]
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            row: dict[str, Any] = {
                "a": a,
                "b": b,
                "category_a": categories.get(a, ""),
                "category_b": categories.get(b, ""),
                "same_category": categories.get(a, "") == categories.get(b, ""),
                "combined_similarity": cosine(vectors[a], vectors[b]),
            }
            for group in group_names:
                row[f"{group}_similarity"] = cosine(group_subset(vectors[a], group), group_subset(vectors[b], group))
            rows.append(row)
    rows.sort(key=lambda r: float(r["combined_similarity"]), reverse=True)
    return rows


def matrix_rows(names: list[str], vectors: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for a in names:
        row: dict[str, Any] = {"subtype": a}
        for b in names:
            row[b] = f"{cosine(vectors[a], vectors[b]):.6f}"
        rows.append(row)
    return rows


def build_model_map(args: argparse.Namespace, model: str) -> tuple[dict[str, dict[str, float]], dict[str, str], dict[str, Any]]:
    p290 = load_json(Path(args.phase290_dir) / f"{model}_phase290_contract_break_scan.json")
    p291 = load_json(Path(args.phase291_dir) / f"{model}_phase291_block_contract_scan.json")
    p294 = load_json(Path(args.phase294_dir) / f"{model}_phase294_dynamic_recompute_pilot.json")
    p293_events = [
        row for row in read_csv(Path(args.phase293_dir) / "naturalness_events.csv")
        if row.get("model") == model
    ]

    v290, categories = aggregate_phase290(p290["results"])
    v291 = aggregate_phase291(p291["results"])
    v293 = aggregate_phase293(p293_events)
    v294 = aggregate_phase294(p294["results"])

    subtypes = sorted(set(v290) | set(v291) | set(v293) | set(v294))
    vectors = {
        subtype: merge_vectors(v290.get(subtype, {}), v291.get(subtype, {}), v293.get(subtype, {}), v294.get(subtype, {}))
        for subtype in subtypes
    }
    meta = {
        "phase290_rows": len(p290["results"]),
        "phase291_rows": len(p291["results"]),
        "phase293_event_rows": len(p293_events),
        "phase294_rows": len(p294["results"]),
        "subtypes": len(subtypes),
    }
    return vectors, categories, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase290-dir", default=str(REPO_ROOT / "results" / "gpt5_phase290_contract_break_full"))
    parser.add_argument("--phase291-dir", default=str(REPO_ROOT / "results" / "gpt5_phase291_block_contract_full"))
    parser.add_argument("--phase293-dir", default=str(REPO_ROOT / "results" / "gpt5_phase293_naturalness"))
    parser.add_argument("--phase294-dir", default=str(REPO_ROOT / "results" / "gpt5_phase294b_dynamic_recompute_full"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase295_global_contract_mapping"))
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    report: list[str] = []
    report.append("# Phase 295 Global Functional Contract Mapping Report\n")
    report.append("## Inputs\n")
    report.append(f"- phase290_dir: `{args.phase290_dir}`\n")
    report.append(f"- phase291_dir: `{args.phase291_dir}`\n")
    report.append(f"- phase293_dir: `{args.phase293_dir}`\n")
    report.append(f"- phase294_dir: `{args.phase294_dir}`\n")

    for model in MODELS:
        raw_vectors, categories, meta = build_model_map(args, model)
        group_vectors = normalize_group_vectors(raw_vectors)
        z_vectors = zscore_vectors(group_vectors)
        subtypes = sorted(raw_vectors)
        pairs = pair_rows(subtypes, group_vectors, categories)
        z_pairs = pair_rows(subtypes, z_vectors, categories)

        all_data[model] = {
            "categories": categories,
            "subtypes": subtypes,
            "raw_vectors": raw_vectors,
            "group_normalized_vectors": group_vectors,
            "zscore_vectors": z_vectors,
            "meta": meta,
        }

        write_csv(out_dir / f"{model}_global_similarity.csv", matrix_rows(subtypes, group_vectors), ["subtype"] + subtypes)
        write_csv(out_dir / f"{model}_global_top_reuse_candidates.csv", [{k: fmt(v) for k, v in row.items()} for row in pairs[: args.top_k]], list(pairs[0].keys()))
        write_csv(out_dir / f"{model}_global_bottom_differentiation_candidates.csv", [{k: fmt(v) for k, v in row.items()} for row in list(reversed(pairs[-args.top_k:]))], list(pairs[0].keys()))
        write_csv(out_dir / f"{model}_global_zscore_top_pairs.csv", [{k: fmt(v) for k, v in row.items()} for row in z_pairs[: args.top_k]], list(z_pairs[0].keys()))
        write_csv(out_dir / f"{model}_global_zscore_bottom_pairs.csv", [{k: fmt(v) for k, v in row.items()} for row in list(reversed(z_pairs[-args.top_k:]))], list(z_pairs[0].keys()))

        sims = [float(row["combined_similarity"]) for row in pairs]
        same = [float(row["combined_similarity"]) for row in pairs if row["same_category"]]
        cross = [float(row["combined_similarity"]) for row in pairs if not row["same_category"]]
        summary_rows.append({
            "model": model,
            **meta,
            "features_min": min(len(v) for v in raw_vectors.values()),
            "features_max": max(len(v) for v in raw_vectors.values()),
            "combined_mean": mean(sims),
            "combined_min": min(sims),
            "combined_max": max(sims),
            "same_category_mean": mean(same),
            "cross_category_mean": mean(cross),
        })

        report.append(f"\n## {model}\n")
        report.append(f"- meta: {meta}\n")
        report.append(f"- features: min={min(len(v) for v in raw_vectors.values())}, max={max(len(v) for v in raw_vectors.values())}\n")
        report.append(f"- combined similarity: mean={mean(sims):.4f}, min={min(sims):.4f}, max={max(sims):.4f}\n")
        report.append(f"- same category mean={mean(same):.4f}, cross category mean={mean(cross):.4f}\n")
        report.append("- top reuse candidates:\n")
        for row in pairs[:5]:
            report.append(
                f"  - {row['a']} / {row['b']}: combined={row['combined_similarity']:.4f}, "
                f"layer={row['layer_curve_similarity']:.4f}, block={row['block_curve_similarity']:.4f}, "
                f"dynamic={row['dynamic_layer_similarity']:.4f}, natural={row['naturalness_similarity']:.4f}\n"
            )
        report.append("- bottom differentiation candidates:\n")
        for row in list(reversed(pairs[-5:])):
            report.append(
                f"  - {row['a']} / {row['b']}: combined={row['combined_similarity']:.4f}, "
                f"layer={row['layer_curve_similarity']:.4f}, block={row['block_curve_similarity']:.4f}, "
                f"dynamic={row['dynamic_layer_similarity']:.4f}, natural={row['naturalness_similarity']:.4f}\n"
            )

    write_csv(out_dir / "global_mapping_summary.csv", [{k: fmt(v) for k, v in row.items()} for row in summary_rows], list(summary_rows[0].keys()))
    (out_dir / "global_contract_maps.json").write_text(json.dumps(all_data, indent=2), encoding="utf-8")
    (out_dir / "GLOBAL_CONTRACT_MAPPING_REPORT.md").write_text("".join(report), encoding="utf-8")

    log(f"saved output_dir={out_dir}")
    log(f"models={len(MODELS)} summary_rows={len(summary_rows)}")


if __name__ == "__main__":
    main()
