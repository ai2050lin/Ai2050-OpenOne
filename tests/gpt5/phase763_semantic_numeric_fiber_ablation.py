#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


MODELS = ["qwen3", "glm4", "deepseek7b"]
IN_ROOT = Path("results/glm5_phase762_semantic_numeric_fiber_atlas")
OUT_ROOT = Path("results/glm5_phase763_semantic_numeric_fiber_ablation")


def safe_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def add_feature(bucket: dict[str, list[float]], feature: str, value: Any) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(val):
        bucket[feature].append(val)


def mean_dict(values: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(sum(v) / len(v)) for k, v in values.items() if v}


def cosine(a: dict[str, float], b: dict[str, float], features: list[str]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for f in features:
        av = float(a.get(f, 0.0))
        bv = float(b.get(f, 0.0))
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / math.sqrt(na * nb)


def center_vectors(vectors: dict[str, dict[str, float]], features: list[str]) -> dict[str, dict[str, float]]:
    means = {f: safe_mean([v.get(f, 0.0) for v in vectors.values()]) or 0.0 for f in features}
    return {obj: {f: vec.get(f, 0.0) - means[f] for f in features} for obj, vec in vectors.items()}


def pairwise(vectors: dict[str, dict[str, float]], meta: dict[str, dict[str, str]], features: list[str]) -> list[dict[str, Any]]:
    objects = sorted(vectors)
    rows = []
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            rows.append(
                {
                    "object_a": a,
                    "object_b": b,
                    "domain_a": meta[a]["domain"],
                    "domain_b": meta[b]["domain"],
                    "same_domain": meta[a]["domain"] == meta[b]["domain"],
                    "similarity": cosine(vectors[a], vectors[b], features),
                }
            )
    return rows


def same_diff_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    same = [float(r["similarity"]) for r in rows if r["same_domain"]]
    diff = [float(r["similarity"]) for r in rows if not r["same_domain"]]
    same_mean = safe_mean(same)
    diff_mean = safe_mean(diff)
    return {
        "same_domain_mean": same_mean,
        "different_domain_mean": diff_mean,
        "separation": (same_mean or 0.0) - (diff_mean or 0.0),
        "same_n": len(same),
        "different_n": len(diff),
    }


def nn_domain_accuracy(rows: list[dict[str, Any]], objects: list[str]) -> dict[str, Any]:
    neighbors: dict[str, tuple[str, float, bool]] = {}
    for row in rows:
        a = row["object_a"]
        b = row["object_b"]
        sim = float(row["similarity"])
        for x, y in [(a, b), (b, a)]:
            if x not in neighbors or sim > neighbors[x][1]:
                neighbors[x] = (y, sim, bool(row["same_domain"]))
    if not neighbors:
        return {"accuracy": None, "neighbors": {}}
    return {
        "accuracy": sum(1 for obj in objects if neighbors.get(obj, ("", 0.0, False))[2]) / len(objects),
        "neighbors": {
            obj: {
                "nearest": neighbors[obj][0],
                "similarity": neighbors[obj][1],
                "same_domain": neighbors[obj][2],
            }
            for obj in sorted(neighbors)
        },
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_meta(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    meta = {}
    for row in rows:
        obj = row.get("object")
        domain = row.get("domain")
        if obj and domain:
            meta[str(obj)] = {"object": str(obj), "domain": str(domain)}
    return meta


def configs() -> list[dict[str, Any]]:
    return [
        {"name": "phase762_exact", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": True},
        {"name": "all_direct4_relation_specific", "metrics": {"target", "attention", "direct4", "route_release", "margin_drop"}, "relation_prefix": True},
        {"name": "target_drop_only", "metrics": {"target"}, "relation_prefix": True},
        {"name": "attention_mass_only", "metrics": {"attention"}, "relation_prefix": True},
        {"name": "direct2_scores_only", "metrics": {"direct2"}, "relation_prefix": True},
        {"name": "direct4_scores_only", "metrics": {"direct4"}, "relation_prefix": True},
        {"name": "route_release_only", "metrics": {"route_release"}, "relation_prefix": True},
        {"name": "margin_drop_only", "metrics": {"margin_drop"}, "relation_prefix": True},
        {"name": "no_attention_mass", "metrics": {"target", "direct2", "route_release", "margin_drop"}, "relation_prefix": True},
        {"name": "no_direct_scores", "metrics": {"target", "attention", "route_release", "margin_drop"}, "relation_prefix": True},
        {"name": "no_route_features", "metrics": {"target", "attention", "direct2"}, "relation_prefix": True},
        {"name": "records_only", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": True, "sources": {"target_record_line", "target_value_tokens", "records_all"}},
        {"name": "object_relation_sources_only", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": True, "sources": {"object_tokens", "relation_tokens"}},
        {"name": "no_target_value_tokens", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": True, "exclude_sources": {"target_value_tokens"}},
        {"name": "no_object_tokens", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": True, "exclude_sources": {"object_tokens"}},
        {"name": "no_relation_tokens", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": True, "exclude_sources": {"relation_tokens"}},
        {"name": "all_relation_collapsed", "metrics": {"target", "attention", "direct2", "route_release", "margin_drop"}, "relation_prefix": False},
        {"name": "target_drop_relation_collapsed", "metrics": {"target"}, "relation_prefix": False},
        {"name": "route_release_relation_collapsed", "metrics": {"route_release"}, "relation_prefix": False},
    ]


def keep_source(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    src = row.get("source_group")
    if cfg.get("sources") is not None and src not in cfg["sources"]:
        return False
    if cfg.get("exclude_sources") is not None and src in cfg["exclude_sources"]:
        return False
    return True


def build_vectors(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, dict[str, float]], list[str]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metrics = set(cfg["metrics"])
    for row in rows:
        if row.get("row_kind") != "semantic_fiber_effect" or not keep_source(row, cfg):
            continue
        obj = row["object"]
        if cfg.get("relation_prefix", True):
            prefix = f"rel={row['relation']}|{row['subunit_id']}|{row['source_group']}"
        else:
            prefix = f"rel=*|{row['subunit_id']}|{row['source_group']}"
        if "target" in metrics:
            add_feature(buckets[obj], f"{prefix}|target_logit_drop", row.get("target_logit_drop"))
        if "attention" in metrics:
            add_feature(buckets[obj], f"{prefix}|attention_mass", row.get("attention_mass_to_source"))
        if "direct2" in metrics or "direct4" in metrics:
            direct = row.get("source_direct_score") or {}
            keys = ["direct_target_boost", "direct_total_route_suppression"]
            if "direct4" in metrics:
                keys += ["direct_mean_margin_gain", "direct_positive_route_count"]
            for key in keys:
                add_feature(buckets[obj], f"{prefix}|{key}", direct.get(key))
        route_matrix = row.get("route_matrix") or {}
        for route_group, cell in route_matrix.items():
            if "route_release" in metrics:
                add_feature(buckets[obj], f"{prefix}|route_release:{route_group}", cell.get("route_release"))
            if "margin_drop" in metrics:
                add_feature(buckets[obj], f"{prefix}|margin_drop:{route_group}", cell.get("margin_drop_target_vs_route"))
    vectors = {obj: mean_dict(feats) for obj, feats in buckets.items()}
    features = sorted({f for vec in vectors.values() for f in vec})
    return vectors, features


def summarize_config(rows: list[dict[str, Any]], meta: dict[str, dict[str, str]], cfg: dict[str, Any]) -> dict[str, Any]:
    vectors, features = build_vectors(rows, cfg)
    objects = sorted(vectors)
    centered = center_vectors(vectors, features)
    pair_rows = pairwise(centered, meta, features)
    return {
        "config": cfg["name"],
        "n_objects": len(objects),
        "n_features": len(features),
        "pair_summary": same_diff_summary(pair_rows),
        "nn_domain": nn_domain_accuracy(pair_rows, objects),
        "pair_rows": pair_rows,
    }


def pair_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    out = {}
    for row in rows:
        key = "||".join(sorted([row["object_a"], row["object_b"]]))
        out[key] = float(row["similarity"])
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 763 Semantic Numeric Fiber Ablation ({payload['round']})",
        "",
        "- Status: `complete`",
        "- Input: Phase 762 confirm rows; no model was loaded.",
        "- Purpose: identify which causal-fiber components carry same-domain object structure.",
        "",
        "## Ablation Results",
        "",
        "| model | config | features | NN | same | diff | sep |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for row in payload["by_model"].get(model, {}).get("configs", []):
            summary = row["pair_summary"]
            lines.append(
                f"| {model} | `{row['config']}` | {row['n_features']} | "
                f"{row['nn_domain']['accuracy']:.3f} | {summary['same_domain_mean']:.3f} | "
                f"{summary['different_domain_mean']:.3f} | {summary['separation']:.3f} |"
            )
    lines += [
        "",
        "## Cross-Model Correlations",
        "",
        "| config | pair | common pairs | pearson |",
        "|---|---|---:|---:|",
    ]
    for cfg_name, pairs in payload["cross_model_correlations"].items():
        for pair, corr in pairs.items():
            value = corr["pearson"]
            value_text = "null" if value is None else f"{value:.3f}"
            lines.append(f"| `{cfg_name}` | `{pair}` | {corr['common_pairs']} | {value_text} |")
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- A valid semantic-numeric interface should survive feature ablation and not depend on a single metric family.",
        "- If a feature family has high separation but poor nearest-neighbor accuracy, it is a weak topology signal rather than a solved semantic code.",
        "- This phase is an offline audit of Phase 762, not a new causal intervention.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-name", default="confirm")
    parser.add_argument("--input-root", default=str(IN_ROOT))
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    args = parser.parse_args()

    in_root = Path(args.input_root) / args.round_name
    out_dir = Path(args.output_root) / args.round_name
    by_model: dict[str, Any] = {}
    pair_maps: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for model in MODELS:
        row_path = in_root / f"phase762_{model}_rows.jsonl"
        if not row_path.exists():
            continue
        rows = load_rows(row_path)
        meta = load_meta(rows)
        config_rows = []
        for cfg in configs():
            summary = summarize_config(rows, meta, cfg)
            config_rows.append(summary)
            pair_maps[cfg["name"]][model] = pair_map(summary["pair_rows"])
        by_model[model] = {
            "row_path": str(row_path),
            "n_rows": len(rows),
            "n_objects": len(meta),
            "configs": config_rows,
        }

    correlations: dict[str, dict[str, Any]] = {}
    for cfg_name, model_maps in pair_maps.items():
        correlations[cfg_name] = {}
        models = sorted(model_maps)
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                keys = sorted(set(model_maps[a]) & set(model_maps[b]))
                correlations[cfg_name][f"{a}__{b}"] = {
                    "common_pairs": len(keys),
                    "pearson": pearson([model_maps[a][k] for k in keys], [model_maps[b][k] for k in keys]),
                }

    payload = {
        "phase": 763,
        "title": "Semantic Numeric Fiber Ablation",
        "round": args.round_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_root": str(in_root),
        "by_model": by_model,
        "cross_model_correlations": correlations,
        "strict_interpretation": "Offline ablation tests whether Phase 762's causal-fiber domain signal is carried by target drops, attention mass, direct scores, route release, margin drops, or source group subsets.",
    }
    write_json(out_dir / "phase763_ablation_summary.json", payload)
    write_markdown(out_dir / "phase763_ablation_summary.md", payload)
    print(json.dumps({"status": "complete", "out_dir": str(out_dir), "models": sorted(by_model)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
