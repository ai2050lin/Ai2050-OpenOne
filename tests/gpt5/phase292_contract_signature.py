from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = ["qwen3", "glm4", "deepseek7b"]
PATCH_TYPES = ["attn", "mlp", "both", "resid", "cross_battn_amlp", "cross_aattn_bmlp"]


def log(message: str) -> None:
    print(f"[phase292] {message}", flush=True)


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def category_by_subtype(rows: list[dict[str, Any]]) -> dict[str, str]:
    out = {}
    for row in rows:
        out[str(row["subtype"])] = str(row["category"])
    return out


def aggregate_phase290(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    full: dict[str, dict[str, float]] = defaultdict(dict)
    canon: dict[str, dict[str, float]] = defaultdict(dict)

    by_sub_layer_patch: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_sub_alpha_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    layers_by_sub: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        subtype = str(row["subtype"])
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

        patch_layer_progress: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for (sub, layer, patch), items in by_sub_layer_patch.items():
            if sub != subtype:
                continue
            progress = mean([float(x.get("progress", 0.0)) for x in items])
            kl_ratio = mean([float(x.get("kl_ratio", 0.0)) for x in items])
            delta = mean([float(x.get("logit_delta_ratio", 0.0)) for x in items])
            nonfinite = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
            norm_bad = mean([float(x.get("norm_illegal", 0.0)) for x in items])
            pos = layer_pos[layer]
            rel = pos / denom
            full[subtype][f"p290.layer.L{layer}.{patch}.progress"] = progress
            full[subtype][f"p290.layer.L{layer}.{patch}.kl_ratio"] = kl_ratio
            full[subtype][f"p290.layer.L{layer}.{patch}.delta"] = delta
            full[subtype][f"p290.layer_pos.{pos}.{patch}.progress"] = progress
            full[subtype][f"p290.layer_pos.{pos}.{patch}.kl_ratio"] = kl_ratio
            full[subtype][f"p290.layer.L{layer}.{patch}.nonfinite"] = nonfinite
            full[subtype][f"p290.layer.L{layer}.{patch}.norm_illegal"] = norm_bad
            patch_layer_progress[patch].append((layer, progress))
            if patch == "both":
                canon[subtype][f"p290.both.layer_pos_{pos}.progress"] = progress
                canon[subtype][f"p290.both.layer_pos_{pos}.kl_ratio"] = kl_ratio
                full[subtype][f"p290.layer.L{layer}.relative_position"] = rel

        for patch in PATCH_TYPES:
            vals = patch_layer_progress.get(patch, [])
            if vals:
                best_layer, best_progress = max(vals, key=lambda x: x[1])
                canon[subtype][f"p290.{patch}.best_progress"] = best_progress
                canon[subtype][f"p290.{patch}.mean_progress"] = mean([x[1] for x in vals])
                canon[subtype][f"p290.{patch}.best_layer_rel"] = layer_pos[best_layer] / denom

        for layer in sorted_layers:
            both = full[subtype].get(f"p290.layer.L{layer}.both.progress", 0.0)
            cross = full[subtype].get(f"p290.layer.L{layer}.cross_battn_amlp.progress", 0.0)
            drop = both - cross
            full[subtype][f"p290.layer.L{layer}.cross_battn_amlp.drop"] = drop
        drops = [v for k, v in full[subtype].items() if k.endswith(".cross_battn_amlp.drop")]
        canon[subtype]["p290.cross_battn_amlp.max_drop"] = max(drops) if drops else 0.0
        canon[subtype]["p290.cross_battn_amlp.mean_drop"] = mean(drops)

        for (sub, alpha, patch), items in by_sub_alpha_patch.items():
            if sub != subtype:
                continue
            progress = mean([float(x.get("progress", 0.0)) for x in items])
            kl_ratio = mean([float(x.get("kl_ratio", 0.0)) for x in items])
            full[subtype][f"p290.alpha.{alpha}.{patch}.progress"] = progress
            full[subtype][f"p290.alpha.{alpha}.{patch}.kl_ratio"] = kl_ratio
            if patch in {"both", "cross_battn_amlp"}:
                canon[subtype][f"p290.alpha.{alpha}.{patch}.progress"] = progress

    return dict(full), dict(canon)


def aggregate_phase291(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    full: dict[str, dict[str, float]] = defaultdict(dict)
    canon: dict[str, dict[str, float]] = defaultdict(dict)

    by_sub_block_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_sub_alpha_patch: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    blocks_by_sub: dict[str, set[str]] = defaultdict(set)
    widths_by_block: dict[str, int] = {}
    for row in rows:
        subtype = str(row["subtype"])
        block = str(row["block"])
        patch = str(row["patch_type"])
        blocks_by_sub[subtype].add(block)
        widths_by_block[block] = int(row.get("block_width", 1))
        by_sub_alpha_patch[(subtype, str(row["alpha"]), patch)].append(row)
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            by_sub_block_patch[(subtype, block, patch)].append(row)

    for subtype, blocks in blocks_by_sub.items():
        sorted_blocks = sorted(blocks, key=lambda b: (widths_by_block.get(b, 1), b))
        block_pos = {block: idx for idx, block in enumerate(sorted_blocks)}
        denom = max(len(sorted_blocks) - 1, 1)
        patch_block_progress: dict[str, list[tuple[str, float]]] = defaultdict(list)

        for (sub, block, patch), items in by_sub_block_patch.items():
            if sub != subtype:
                continue
            progress = mean([float(x.get("progress", 0.0)) for x in items])
            kl_ratio = mean([float(x.get("kl_ratio", 0.0)) for x in items])
            delta = mean([float(x.get("logit_delta_ratio", 0.0)) for x in items])
            nonfinite = mean([1.0 - float(x.get("finite", 1.0)) for x in items])
            norm_bad = mean([float(x.get("norm_illegal", 0.0)) for x in items])
            pos = block_pos[block]
            width = widths_by_block.get(block, 1)
            full[subtype][f"p291.block.{block}.{patch}.progress"] = progress
            full[subtype][f"p291.block.{block}.{patch}.kl_ratio"] = kl_ratio
            full[subtype][f"p291.block.{block}.{patch}.delta"] = delta
            full[subtype][f"p291.block_pos.{pos}.{patch}.progress"] = progress
            full[subtype][f"p291.block.{block}.{patch}.nonfinite"] = nonfinite
            full[subtype][f"p291.block.{block}.{patch}.norm_illegal"] = norm_bad
            full[subtype][f"p291.block.{block}.width"] = float(width)
            patch_block_progress[patch].append((block, progress))
            if patch == "both":
                canon[subtype][f"p291.both.block_pos_{pos}.progress"] = progress
                canon[subtype][f"p291.both.block_pos_{pos}.kl_ratio"] = kl_ratio
                canon[subtype][f"p291.both.width_{width}.progress"] = progress

        for patch in PATCH_TYPES:
            vals = patch_block_progress.get(patch, [])
            if vals:
                best_block, best_progress = max(vals, key=lambda x: x[1])
                canon[subtype][f"p291.{patch}.best_progress"] = best_progress
                canon[subtype][f"p291.{patch}.mean_progress"] = mean([x[1] for x in vals])
                canon[subtype][f"p291.{patch}.best_block_pos"] = block_pos[best_block] / denom
                canon[subtype][f"p291.{patch}.best_block_width"] = float(widths_by_block.get(best_block, 1))

        for block in sorted_blocks:
            both = full[subtype].get(f"p291.block.{block}.both.progress", 0.0)
            cross = full[subtype].get(f"p291.block.{block}.cross_battn_amlp.progress", 0.0)
            drop = both - cross
            full[subtype][f"p291.block.{block}.cross_battn_amlp.drop"] = drop
        drops = [v for k, v in full[subtype].items() if k.endswith(".cross_battn_amlp.drop")]
        canon[subtype]["p291.cross_battn_amlp.max_drop"] = max(drops) if drops else 0.0
        canon[subtype]["p291.cross_battn_amlp.mean_drop"] = mean(drops)

        for (sub, alpha, patch), items in by_sub_alpha_patch.items():
            if sub != subtype:
                continue
            progress = mean([float(x.get("progress", 0.0)) for x in items])
            full[subtype][f"p291.alpha.{alpha}.{patch}.progress"] = progress
            if patch in {"both", "cross_battn_amlp"}:
                canon[subtype][f"p291.alpha.{alpha}.{patch}.progress"] = progress

    return dict(full), dict(canon)


def merge_vectors(*vectors: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for vector in vectors:
        out.update(vector)
    return out


def matrix_rows(names: list[str], vectors: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for a in names:
        row = {"subtype": a}
        for b in names:
            row[b] = f"{cosine(vectors[a], vectors[b]):.6f}"
        rows.append(row)
    return rows


def top_pairs(names: list[str], vectors: dict[str, dict[str, float]], categories: dict[str, str], limit: int) -> list[dict[str, Any]]:
    rows = []
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
    return rows[:limit]


def bottom_pairs(names: list[str], vectors: dict[str, dict[str, float]], categories: dict[str, str], limit: int) -> list[dict[str, Any]]:
    rows = top_pairs(names, vectors, categories, limit=100000)
    rows.sort(key=lambda r: r["similarity"])
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase290-dir", default=str(REPO_ROOT / "results" / "gpt5_phase290_contract_break_full"))
    parser.add_argument("--phase291-dir", default=str(REPO_ROOT / "results" / "gpt5_phase291_block_contract_full"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase292_contract_signature"))
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    for model in MODELS:
        p290 = load_json(Path(args.phase290_dir) / f"{model}_phase290_contract_break_scan.json")
        p291 = load_json(Path(args.phase291_dir) / f"{model}_phase291_block_contract_scan.json")
        categories = category_by_subtype(p290["results"])
        p290_full, p290_canon = aggregate_phase290(p290["results"])
        p291_full, p291_canon = aggregate_phase291(p291["results"])
        subtypes = sorted(set(p290_full) | set(p291_full))

        full_vectors = {
            subtype: merge_vectors(p290_full.get(subtype, {}), p291_full.get(subtype, {}))
            for subtype in subtypes
        }
        canonical_vectors = {
            subtype: merge_vectors(p290_canon.get(subtype, {}), p291_canon.get(subtype, {}))
            for subtype in subtypes
        }

        all_data[model] = {
            "categories": categories,
            "subtypes": subtypes,
            "full_vectors": full_vectors,
            "canonical_vectors": canonical_vectors,
        }

        matrix = matrix_rows(subtypes, canonical_vectors)
        write_csv(out_dir / f"{model}_subtype_similarity.csv", matrix, ["subtype"] + subtypes)
        top = top_pairs(subtypes, canonical_vectors, categories, args.top_k)
        bottom = bottom_pairs(subtypes, canonical_vectors, categories, args.top_k)
        write_csv(
            out_dir / f"{model}_top_reuse_pairs.csv",
            top,
            ["a", "b", "category_a", "category_b", "same_category", "similarity"],
        )
        write_csv(
            out_dir / f"{model}_bottom_differentiation_pairs.csv",
            bottom,
            ["a", "b", "category_a", "category_b", "same_category", "similarity"],
        )

        for subtype in subtypes:
            vector = canonical_vectors[subtype]
            summary_rows.append({
                "model": model,
                "category": categories.get(subtype, ""),
                "subtype": subtype,
                "features": len(vector),
                "p290_both_best": vector.get("p290.both.best_progress", 0.0),
                "p291_both_best": vector.get("p291.both.best_progress", 0.0),
                "p290_cross_max_drop": vector.get("p290.cross_battn_amlp.max_drop", 0.0),
                "p291_cross_max_drop": vector.get("p291.cross_battn_amlp.max_drop", 0.0),
                "p291_best_block_width": vector.get("p291.both.best_block_width", 0.0),
            })

    for subtype in sorted(set().union(*(set(all_data[m]["subtypes"]) for m in MODELS))):
        for i, model_a in enumerate(MODELS):
            for model_b in MODELS[i + 1:]:
                vec_a = all_data[model_a]["canonical_vectors"].get(subtype)
                vec_b = all_data[model_b]["canonical_vectors"].get(subtype)
                if vec_a is None or vec_b is None:
                    continue
                cross_rows.append({
                    "subtype": subtype,
                    "category": all_data[model_a]["categories"].get(subtype, ""),
                    "model_a": model_a,
                    "model_b": model_b,
                    "similarity": cosine(vec_a, vec_b),
                })
    cross_rows.sort(key=lambda r: (r["subtype"], r["model_a"], r["model_b"]))

    write_csv(
        out_dir / "signature_summary.csv",
        summary_rows,
        [
            "model", "category", "subtype", "features",
            "p290_both_best", "p291_both_best",
            "p290_cross_max_drop", "p291_cross_max_drop", "p291_best_block_width",
        ],
    )
    write_csv(out_dir / "cross_model_same_subtype_similarity.csv", cross_rows, ["subtype", "category", "model_a", "model_b", "similarity"])

    (out_dir / "contract_signatures.json").write_text(json.dumps(all_data, indent=2), encoding="utf-8")

    report = []
    report.append("# Phase 292 Contract Signature Report\n")
    report.append("## Inputs\n")
    report.append(f"- Phase 290 dir: `{args.phase290_dir}`\n")
    report.append(f"- Phase 291 dir: `{args.phase291_dir}`\n")
    report.append("## Model Summaries\n")
    for model in MODELS:
        model_rows = [r for r in summary_rows if r["model"] == model]
        avg_p290 = mean([float(r["p290_both_best"]) for r in model_rows])
        avg_p291 = mean([float(r["p291_both_best"]) for r in model_rows])
        avg_drop = mean([float(r["p291_cross_max_drop"]) for r in model_rows])
        report.append(f"### {model}\n")
        report.append(f"- subtypes: {len(model_rows)}\n")
        report.append(f"- avg p290 both best: {avg_p290:.4f}\n")
        report.append(f"- avg p291 both best: {avg_p291:.4f}\n")
        report.append(f"- avg p291 cross max drop: {avg_drop:.4f}\n")
        top = top_pairs(all_data[model]["subtypes"], all_data[model]["canonical_vectors"], all_data[model]["categories"], 5)
        report.append("- top reuse pairs:\n")
        for row in top:
            report.append(f"  - {row['a']} / {row['b']}: {row['similarity']:.4f}\n")
    report.append("## Cross Model Same Subtype Similarity\n")
    by_pair = defaultdict(list)
    for row in cross_rows:
        by_pair[(row["model_a"], row["model_b"])].append(float(row["similarity"]))
    for pair, vals in sorted(by_pair.items()):
        report.append(f"- {pair[0]} vs {pair[1]}: mean={mean(vals):.4f}, n={len(vals)}\n")
    (out_dir / "CONTRACT_SIGNATURE_REPORT.md").write_text("".join(report), encoding="utf-8")

    log(f"saved output_dir={out_dir}")
    log(f"summary_rows={len(summary_rows)} cross_rows={len(cross_rows)}")


if __name__ == "__main__":
    main()
