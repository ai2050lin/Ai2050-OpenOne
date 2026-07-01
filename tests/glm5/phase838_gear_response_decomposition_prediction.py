#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


PHASE = 838
INPUT_ROOT = Path("tests/result/phase837_global_gear_response_atlas_pilot/confirm")
RESULT_ROOT = Path("tests/result/phase838_gear_response_decomposition_prediction")
MODELS = ("qwen3", "glm4", "deepseek7b")
DONOR_ORDER = ("exact_choices", "natural_category", "natural_question", "object_only")


VECTOR_KEYS = [
    "target_success",
    "target_rank_gain",
    "above_target_gain",
    "target_logit_gain",
    "target_minus_top_gain",
    "span_target_rank_gain",
    "span_target_margin_vs_contrast_gain",
    "span_target_margin_vs_generic_gain",
    "span_target_margin_vs_non_target_gain",
    "target_span_score_gain",
    "contrast_suppression_gain",
    "generic_suppression_gain",
    "echo_margin_gain",
    "blocker_reduction_score",
    "protocol_valid",
    "protocol_damage",
    "object_echo",
    "format_echo",
    "format_with_target",
    "near_miss",
    "harmful",
    "target_quality_score",
    "echo_risk_score",
    "harm_risk_score",
]

SHAPE_KEYS = [
    "target_quality_score",
    "target_success",
    "target_rank_gain",
    "span_target_margin_vs_contrast_gain",
    "span_target_margin_vs_generic_gain",
    "blocker_reduction_score",
    "echo_risk_score",
    "harm_risk_score",
]


def log(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def maybe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def bool01(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def squash(value: float, scale: float = 1.0) -> float:
    if scale <= 0:
        scale = 1.0
    return math.tanh(finite(value) / scale)


def load_rows(model: str) -> list[dict[str, Any]]:
    path = INPUT_ROOT / f"phase837_{model}_rows.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nested_score(row: dict[str, Any], key: str) -> float | None:
    item = row.get(key)
    if not isinstance(item, dict):
        return None
    return maybe_float(item.get("score_mean_logprob"))


def baseline_nested_score(row: dict[str, Any], key: str) -> float | None:
    span = row.get("baseline_span")
    if not isinstance(span, dict):
        return None
    item = span.get(key)
    if not isinstance(item, dict):
        return None
    return maybe_float(item.get("score_mean_logprob"))


def diff_or_zero(after: Any, before: Any) -> float:
    a = maybe_float(after)
    b = maybe_float(before)
    return 0.0 if a is None or b is None else a - b


def rank_gain(after_rank: Any, before_rank: Any) -> float:
    after = maybe_float(after_rank)
    before = maybe_float(before_rank)
    if after is None or before is None or before <= 0:
        return 0.0
    return max(-1.0, min(1.0, (before - after) / before))


def row_vector(row: dict[str, Any]) -> dict[str, float]:
    baseline_rank = (row.get("baseline_rank_profile") or {}).get("target_rank")
    baseline_above = (row.get("baseline_rank_profile") or {}).get("above_target_count")
    baseline_logit = (row.get("baseline_rank_profile") or {}).get("target_logit")
    baseline_tmt = (row.get("baseline_rank_profile") or {}).get("target_minus_top")

    patched_class = str(row.get("patched_boundary_class") or "")
    target_success = bool01(row.get("target_transition"))
    harmful = bool01(row.get("degraded_boundary"))
    object_echo = bool01(patched_class == "object_echo")
    format_echo = bool01(patched_class == "format_echo")
    format_with_target = bool01(patched_class == "format_with_target")
    near_miss = bool01(patched_class in {"close_near_miss", "broad_near_miss"})
    protocol_valid = bool01(row.get("patched_protocol_valid"))
    protocol_damage = bool01(bool(row.get("baseline_protocol_valid")) and not bool(row.get("patched_protocol_valid")))

    target_logit_gain = diff_or_zero(row.get("target_logit"), baseline_logit)
    target_minus_top_gain = diff_or_zero(row.get("target_minus_top"), baseline_tmt)
    target_rank_gain = rank_gain(row.get("target_rank"), baseline_rank)
    above_target_gain = rank_gain(row.get("above_target_count"), baseline_above)
    span_target_rank_gain = rank_gain(row.get("patch_span_target_rank"), row.get("patch_span_target_rank_baseline"))

    contrast_margin_gain = diff_or_zero(
        row.get("patch_span_target_margin_vs_contrast"),
        row.get("patch_span_target_margin_vs_contrast_baseline"),
    )
    generic_margin_gain = diff_or_zero(
        row.get("patch_span_target_margin_vs_generic"),
        row.get("patch_span_target_margin_vs_generic_baseline"),
    )
    non_target_margin_gain = diff_or_zero(
        row.get("patch_span_target_margin_vs_non_target"),
        row.get("patch_span_target_margin_vs_non_target_baseline"),
    )
    echo_margin_gain = diff_or_zero(
        row.get("patch_span_target_margin_vs_echo"),
        row.get("patch_span_target_margin_vs_echo_baseline"),
    )

    target_span_score_gain = diff_or_zero(
        nested_score(row, "patch_best_target"),
        baseline_nested_score(row, "best_target"),
    )
    contrast_suppression_gain = -diff_or_zero(
        nested_score(row, "patch_best_contrast"),
        baseline_nested_score(row, "best_contrast"),
    )
    generic_suppression_gain = -diff_or_zero(
        nested_score(row, "patch_best_generic_blocker"),
        baseline_nested_score(row, "best_generic_blocker"),
    )

    blocker_reduction_score = (
        0.35 * bool01(row.get("patch_span_contrast_cleared"))
        + 0.25 * bool01(row.get("patch_span_generic_cleared"))
        + 0.20 * bool01(row.get("above_target_decreased"))
        + 0.20 * bool01(row.get("target_rank_improved"))
    )
    echo_risk_score = (
        object_echo
        + 0.50 * format_echo
        + 0.20 * format_with_target
        + 0.20 * bool01(row.get("patch_best_candidate_class") == "object_echo")
        + max(0.0, -squash(echo_margin_gain, 2.0)) * 0.35
    )
    harm_risk_score = harmful + 0.25 * protocol_damage + 0.15 * near_miss

    target_quality_score = (
        1.00 * target_success
        + 0.25 * target_rank_gain
        + 0.15 * above_target_gain
        + 0.12 * squash(target_logit_gain, 5.0)
        + 0.12 * squash(target_minus_top_gain, 5.0)
        + 0.10 * span_target_rank_gain
        + 0.10 * squash(contrast_margin_gain, 2.0)
        + 0.06 * squash(generic_margin_gain, 2.0)
        + 0.04 * squash(target_span_score_gain, 2.0)
        - 0.85 * harmful
        - 0.70 * object_echo
        - 0.35 * format_echo
        - 0.25 * format_with_target
        - 0.30 * protocol_damage
    )

    return {
        "target_success": target_success,
        "target_rank_gain": target_rank_gain,
        "above_target_gain": above_target_gain,
        "target_logit_gain": target_logit_gain,
        "target_minus_top_gain": target_minus_top_gain,
        "span_target_rank_gain": span_target_rank_gain,
        "span_target_margin_vs_contrast_gain": contrast_margin_gain,
        "span_target_margin_vs_generic_gain": generic_margin_gain,
        "span_target_margin_vs_non_target_gain": non_target_margin_gain,
        "target_span_score_gain": target_span_score_gain,
        "contrast_suppression_gain": contrast_suppression_gain,
        "generic_suppression_gain": generic_suppression_gain,
        "echo_margin_gain": echo_margin_gain,
        "blocker_reduction_score": blocker_reduction_score,
        "protocol_valid": protocol_valid,
        "protocol_damage": protocol_damage,
        "object_echo": object_echo,
        "format_echo": format_echo,
        "format_with_target": format_with_target,
        "near_miss": near_miss,
        "harmful": harmful,
        "target_quality_score": target_quality_score,
        "echo_risk_score": echo_risk_score,
        "harm_risk_score": harm_risk_score,
    }


def ordered_unique(rows: list[dict[str, Any]], key: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        val = str(row.get(key) or "")
        if val and val not in seen:
            seen.add(val)
            out.append(val)
    return out


def avg(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = mean(xs)
    my = mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = rank
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    return pearson(ranks(xs), ranks(ys))


def family_from_metrics(metrics: dict[str, float]) -> str:
    target = metrics.get("mean_target_success", 0.0)
    quality = metrics.get("mean_target_quality_score", 0.0)
    echo = metrics.get("mean_echo_risk_score", 0.0)
    harm = metrics.get("mean_harm_risk_score", 0.0)
    rank = metrics.get("mean_target_rank_gain", 0.0)
    blocker = metrics.get("mean_blocker_reduction_score", 0.0)
    if target >= 0.70 and echo < 0.20 and harm < 0.15:
        return "broad_target_writer_family"
    if target >= 0.45 and echo < 0.35 and harm < 0.25:
        return "conditional_target_writer_family"
    if echo >= 0.30 or (target < 0.40 and echo >= 0.15):
        return "echo_dominated_family"
    if harm >= 0.20:
        return "harmful_mixer_family"
    if blocker >= 0.45 and rank > 0.05 and quality < 0.45:
        return "blocker_reducer_non_closer_family"
    return "weak_or_unresolved_family"


def component_metrics(rows: list[dict[str, Any]], train_cases: set[str], holdout_cases: set[str]) -> list[dict[str, Any]]:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_component[str(row.get("component_label_full"))].append(row)

    out: list[dict[str, Any]] = []
    for label, group_rows in sorted(by_component.items()):
        vecs = [r["_vector"] for r in group_rows]
        train = [r for r in group_rows if r.get("case_id") in train_cases]
        holdout = [r for r in group_rows if r.get("case_id") in holdout_cases]
        metrics = {
            "model": group_rows[0].get("model"),
            "component_label_full": label,
            "component_kind": group_rows[0].get("component_kind"),
            "component_source_case_id": group_rows[0].get("component_source_case_id"),
            "n_rows": len(group_rows),
            "n_cases": len({r.get("case_id") for r in group_rows}),
            "n_train_rows": len(train),
            "n_holdout_rows": len(holdout),
        }
        for key in VECTOR_KEYS:
            vals = [finite(v.get(key)) for v in vecs]
            metrics[f"mean_{key}"] = avg(vals)
        metrics["train_target_quality_score"] = avg([finite(r["_vector"].get("target_quality_score")) for r in train])
        metrics["holdout_target_quality_score"] = avg([finite(r["_vector"].get("target_quality_score")) for r in holdout])
        metrics["train_target_rate"] = avg([finite(r["_vector"].get("target_success")) for r in train])
        metrics["holdout_target_rate"] = avg([finite(r["_vector"].get("target_success")) for r in holdout])
        metrics["holdout_harm_rate"] = avg([finite(r["_vector"].get("harmful")) for r in holdout])
        metrics["holdout_echo_risk"] = avg([finite(r["_vector"].get("echo_risk_score")) for r in holdout])
        metrics["target_case_count"] = len({r.get("case_id") for r in group_rows if r["_vector"].get("target_success", 0.0) > 0.5})
        metrics["object_echo_case_count"] = len({r.get("case_id") for r in group_rows if r["_vector"].get("object_echo", 0.0) > 0.5})
        metrics["donor_stability_std"] = donor_stability(group_rows)
        metrics["family"] = family_from_metrics(metrics)
        out.append(metrics)
    return out


def donor_stability(rows: list[dict[str, Any]]) -> float:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_id"))].append(finite(row["_vector"].get("target_quality_score")))
    stds: list[float] = []
    for vals in by_case.values():
        if len(vals) <= 1:
            continue
        m = mean(vals)
        stds.append(math.sqrt(mean([(v - m) ** 2 for v in vals])))
    return avg(stds)


def component_shape_vector(rows: list[dict[str, Any]], cases: list[str]) -> dict[str, list[float]]:
    by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_component[str(row.get("component_label_full"))].append(row)
    out: dict[str, list[float]] = {}
    for label, group_rows in by_component.items():
        lookup: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in group_rows:
            case_id = str(row.get("case_id"))
            donor = str(row.get("donor_variant"))
            for key in SHAPE_KEYS:
                lookup[(case_id, donor, key)].append(finite(row["_vector"].get(key)))
        vec: list[float] = []
        for case_id in cases:
            for donor in DONOR_ORDER:
                for key in SHAPE_KEYS:
                    vals = lookup.get((case_id, donor, key), [])
                    vec.append(avg(vals))
        out[label] = vec
    return out


class UnionFind:
    def __init__(self, labels: list[str]) -> None:
        self.parent = {label: label for label in labels}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def clusters(self) -> list[list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for label in sorted(self.parent):
            groups[self.find(label)].append(label)
        return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def similarity_edges(shape: dict[str, list[float]], threshold: float) -> tuple[list[dict[str, Any]], list[list[str]]]:
    labels = sorted(shape)
    uf = UnionFind(labels)
    edges: list[dict[str, Any]] = []
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            sim = pearson(shape[left], shape[right])
            edge = {"left": left, "right": right, "pearson": sim}
            edges.append(edge)
            if sim is not None and sim >= threshold:
                uf.union(left, right)
    edges.sort(key=lambda e: (-999 if e["pearson"] is None else -float(e["pearson"]), e["left"], e["right"]))
    return edges, uf.clusters()


def prediction_summary(model: str, components: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    ordered = sorted(components, key=lambda c: c.get("train_target_quality_score", 0.0), reverse=True)
    top = ordered[:top_k]
    all_holdout_quality = avg([finite(c.get("holdout_target_quality_score")) for c in components])
    all_holdout_target = avg([finite(c.get("holdout_target_rate")) for c in components])
    top_holdout_quality = avg([finite(c.get("holdout_target_quality_score")) for c in top])
    top_holdout_target = avg([finite(c.get("holdout_target_rate")) for c in top])
    top_holdout_harm = avg([finite(c.get("holdout_harm_rate")) for c in top])
    top_holdout_echo = avg([finite(c.get("holdout_echo_risk")) for c in top])
    xs = [finite(c.get("train_target_quality_score")) for c in components]
    ys = [finite(c.get("holdout_target_quality_score")) for c in components]
    oracle = sorted(components, key=lambda c: c.get("holdout_target_quality_score", 0.0), reverse=True)[:top_k]
    return {
        "model": model,
        "n_components": len(components),
        "top_k": top_k,
        "train_to_holdout_pearson": pearson(xs, ys),
        "train_to_holdout_spearman": spearman(xs, ys),
        "all_holdout_quality": all_holdout_quality,
        "all_holdout_target_rate": all_holdout_target,
        "top_train_holdout_quality": top_holdout_quality,
        "top_train_holdout_target_rate": top_holdout_target,
        "top_train_holdout_harm_rate": top_holdout_harm,
        "top_train_holdout_echo_risk": top_holdout_echo,
        "lift_vs_random_quality": top_holdout_quality - all_holdout_quality,
        "lift_vs_random_target_rate": top_holdout_target - all_holdout_target,
        "oracle_holdout_quality": avg([finite(c.get("holdout_target_quality_score")) for c in oracle]),
        "top_train_components": [c["component_label_full"] for c in top],
        "oracle_holdout_components": [c["component_label_full"] for c in oracle],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Phase {PHASE}: gear response decomposition and prediction validation")
    lines.append("")
    lines.append(f"- timestamp: {summary['timestamp']}")
    lines.append(f"- input: `{INPUT_ROOT}`")
    lines.append("- boundary: offline analysis over Phase 837 confirm rows; no new model forward pass.")
    lines.append("")
    for model in summary["models"]:
        m = summary["model_summaries"][model]
        pred = m["prediction"]
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            "| rows | components | train cases | holdout cases | train->holdout pearson | train->holdout spearman | top-k lift quality | top-k lift target |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(
            f"| {m['n_rows']} | {m['n_components']} | {len(m['train_cases'])} | {len(m['holdout_cases'])} | "
            f"{fmt(pred['train_to_holdout_pearson'])} | {fmt(pred['train_to_holdout_spearman'])} | "
            f"{fmt(pred['lift_vs_random_quality'])} | {fmt(pred['lift_vs_random_target_rate'])} |"
        )
        lines.append("")
        lines.append("Family counts:")
        for fam, count in sorted(m["family_counts"].items()):
            lines.append(f"- {fam}: {count}")
        lines.append("")
        lines.append("Top train components:")
        for label in pred["top_train_components"]:
            lines.append(f"- `{label}`")
        lines.append("")
        lines.append("Top similarity edges:")
        for edge in m["top_similarity_edges"][:5]:
            lines.append(f"- `{edge['left']}` <-> `{edge['right']}`: {fmt(edge['pearson'])}")
        lines.append("")
        lines.append("Clusters:")
        for idx, cluster in enumerate(m["clusters"], 1):
            compact = ", ".join(f"`{x}`" for x in cluster)
            lines.append(f"- cluster {idx}: {compact}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{finite(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    parser.add_argument("--train-cases", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    all_component_rows: list[dict[str, Any]] = []
    all_similarity_rows: list[dict[str, Any]] = []
    all_row_vectors: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}

    for model in MODELS:
        log(f"loading phase837 rows for {model}")
        rows = load_rows(model)
        for row in rows:
            row["_vector"] = row_vector(row)
            all_row_vectors.append(
                {
                    "model": model,
                    "case_id": row.get("case_id"),
                    "donor_variant": row.get("donor_variant"),
                    "component_label_full": row.get("component_label_full"),
                    "component_kind": row.get("component_kind"),
                    "component_source_case_id": row.get("component_source_case_id"),
                    "response_type": row.get("response_type"),
                    "patched_boundary_class": row.get("patched_boundary_class"),
                    "patched_generated": row.get("patched_generated"),
                    "vector": row["_vector"],
                }
            )

        cases = ordered_unique(rows, "case_id")
        train_cases = set(cases[: int(args.train_cases)])
        holdout_cases = set(cases[int(args.train_cases) :])
        components = component_metrics(rows, train_cases, holdout_cases)
        shape = component_shape_vector(rows, cases)
        edges, clusters = similarity_edges(shape, float(args.similarity_threshold))
        pred = prediction_summary(model, components, int(args.top_k))

        for row in components:
            all_component_rows.append(row)
        for edge in edges:
            edge = dict(edge)
            edge["model"] = model
            all_similarity_rows.append(edge)

        family_counts = Counter(c["family"] for c in components)
        model_summaries[model] = {
            "model": model,
            "n_rows": len(rows),
            "n_components": len(components),
            "cases": cases,
            "train_cases": cases[: int(args.train_cases)],
            "holdout_cases": cases[int(args.train_cases) :],
            "family_counts": dict(family_counts),
            "prediction": pred,
            "clusters": clusters,
            "top_similarity_edges": [e for e in edges if e.get("pearson") is not None][:10],
        }
        log(f"{model}: components={len(components)} clusters={len(clusters)} lift={fmt(pred['lift_vs_random_quality'])}")

    summary = {
        "phase": PHASE,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_root": str(INPUT_ROOT),
        "result_root": str(RESULT_ROOT),
        "models": list(MODELS),
        "similarity_threshold": float(args.similarity_threshold),
        "train_cases": int(args.train_cases),
        "top_k": int(args.top_k),
        "model_summaries": model_summaries,
        "boundary": "Offline decomposition over Phase 837 confirm rows. It tests predictive signal in response fingerprints, not natural mechanism closure.",
    }

    write_jsonl(RESULT_ROOT / "phase838_component_vectors.jsonl", all_component_rows)
    write_jsonl(RESULT_ROOT / "phase838_similarity_edges.jsonl", all_similarity_rows)
    write_jsonl(RESULT_ROOT / "phase838_row_vectors.jsonl", all_row_vectors)
    (RESULT_ROOT / "phase838_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(RESULT_ROOT / "phase838_summary.md", summary)
    log(f"wrote {RESULT_ROOT}")


if __name__ == "__main__":
    main()
