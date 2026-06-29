#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase765_commonsense_context_identity_closure_test import center_vectors, cosine  # noqa: E402
from phase767_commonsense_failure_type_topk_audit import MODELS  # noqa: E402
from phase769_semantic_clean_fiber_reanalysis import (  # noqa: E402
    case_meta,
    fmt,
    load_jsonl,
    safe_mean,
    summarize_subset,
    write_json,
)


PHASE765_ROOT = Path("results/glm5_phase765_commonsense_context_identity_closure_test")
PHASE767_ROOT = Path("tests/result/phase767_commonsense_failure_type_topk_audit")
OUT_ROOT = Path("results/glm5_phase770_balanced_semantic_clean_fiber_reanalysis")
RESULT_ROOT = Path("tests/result/phase770_balanced_semantic_clean_fiber_reanalysis")


LabelFn = Callable[[dict[str, Any]], bool]


def is_exact_clean(row: dict[str, Any]) -> bool:
    return bool(row.get("exact_target_top1"))


def is_semantic_clean(row: dict[str, Any]) -> bool:
    return bool(row.get("target_top1"))


def is_semantic_only(row: dict[str, Any]) -> bool:
    return bool(row.get("target_top1")) and not bool(row.get("exact_target_top1"))


def is_semantic_fail(row: dict[str, Any]) -> bool:
    return not bool(row.get("target_top1"))


LABELS: dict[str, LabelFn] = {
    "exact_clean": is_exact_clean,
    "semantic_clean": is_semantic_clean,
    "semantic_only": is_semantic_only,
    "semantic_fail": is_semantic_fail,
}


CONTRASTS = [
    ("semantic_clean", "semantic_fail"),
    ("exact_clean", "semantic_only"),
    ("exact_clean", "semantic_fail"),
    ("semantic_only", "semantic_fail"),
]


def mean_dict(values: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(sum(v) / len(v)) for k, v in values.items() if v}


def add_feature(bucket: dict[str, list[float]], feature: str, value: Any) -> None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(val):
        bucket[feature].append(val)


def effect_rows_for(rows765: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows765 if r.get("row_kind") == "commonsense_fiber_effect"]


def phase767_path(model: str, round_name: str) -> Path:
    path = PHASE767_ROOT / round_name / f"phase767_{model}_rows.jsonl"
    if path.exists():
        return path
    return Path("results/glm5_phase767_commonsense_failure_type_topk_audit") / round_name / f"phase767_{model}_rows.jsonl"


def stratum(row: dict[str, Any], keys: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(k, "")) for k in keys)


def label_case_ids(rows767: list[dict[str, Any]], label: str) -> set[str]:
    pred = LABELS[label]
    return {r["case_id"] for r in rows767 if pred(r)}


def grouped_case_ids(rows767: list[dict[str, Any]], label: str, keys: list[str]) -> dict[tuple[str, ...], list[str]]:
    pred = LABELS[label]
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for row in rows767:
        if pred(row):
            groups[stratum(row, keys)].append(row["case_id"])
    return {k: sorted(v) for k, v in groups.items()}


def balanced_ids(
    rows767: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    keys: list[str],
    max_per_stratum: int | None = None,
) -> dict[str, Any]:
    groups_a = grouped_case_ids(rows767, label_a, keys)
    groups_b = grouped_case_ids(rows767, label_b, keys)
    ids_a: list[str] = []
    ids_b: list[str] = []
    strata_rows = []
    for key in sorted(set(groups_a) & set(groups_b)):
        n = min(len(groups_a[key]), len(groups_b[key]))
        if max_per_stratum is not None:
            n = min(n, max_per_stratum)
        if n <= 0:
            continue
        take_a = groups_a[key][:n]
        take_b = groups_b[key][:n]
        ids_a.extend(take_a)
        ids_b.extend(take_b)
        strata_rows.append(
            {
                "stratum": key,
                "label_a_available": len(groups_a[key]),
                "label_b_available": len(groups_b[key]),
                "taken_each": n,
            }
        )
    return {
        "label_a": label_a,
        "label_b": label_b,
        "strata_keys": keys,
        "n_strata": len(strata_rows),
        "label_a_balanced_ids": sorted(set(ids_a)),
        "label_b_balanced_ids": sorted(set(ids_b)),
        "strata": strata_rows,
        "label_a_available": sum(len(v) for v in groups_a.values()),
        "label_b_available": sum(len(v) for v in groups_b.values()),
    }


def metric_delta(a: dict[str, Any], b: dict[str, Any], key: str) -> float | None:
    av = a.get(key)
    bv = b.get(key)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


def summarize_contrast(
    effects: list[dict[str, Any]],
    meta: dict[str, dict[str, str]],
    rows767: list[dict[str, Any]],
    label_a: str,
    label_b: str,
    keys: list[str],
    max_per_stratum: int | None,
) -> dict[str, Any]:
    matched = balanced_ids(rows767, label_a, label_b, keys, max_per_stratum=max_per_stratum)
    ids_a = set(matched["label_a_balanced_ids"])
    ids_b = set(matched["label_b_balanced_ids"])
    summary_a = summarize_subset(effects, meta, ids_a)
    summary_b = summarize_subset(effects, meta, ids_b)
    return {
        **matched,
        "arm_a": summary_a,
        "arm_b": summary_b,
        "delta_a_minus_b": {
            "mean_context_separation": metric_delta(summary_a, summary_b, "mean_context_separation"),
            "mean_context_nn_domain_accuracy": metric_delta(summary_a, summary_b, "mean_context_nn_domain_accuracy"),
            "mean_object_stability_gap": metric_delta(summary_a, summary_b, "mean_object_stability_gap"),
            "mean_domain_stability_gap": metric_delta(summary_a, summary_b, "mean_domain_stability_gap"),
        },
    }


def case_vectors(effects: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in effects:
        case_id = row["case_id"]
        prefix = f"rel={row['relation']}|{row['subunit_id']}|{row['source_group']}"
        add_feature(buckets[case_id], f"{prefix}|target_logit_drop", row.get("target_logit_drop"))
        add_feature(buckets[case_id], f"{prefix}|attention_mass", row.get("attention_mass_to_source"))
        direct = row.get("source_direct_score") or {}
        add_feature(buckets[case_id], f"{prefix}|direct_target_boost", direct.get("direct_target_boost"))
        add_feature(buckets[case_id], f"{prefix}|direct_total_route_suppression", direct.get("direct_total_route_suppression"))
    return {case_id: mean_dict(feats) for case_id, feats in buckets.items()}


def pair_label(row_a: dict[str, Any], row_b: dict[str, Any], clean_fn: LabelFn) -> str:
    a = clean_fn(row_a)
    b = clean_fn(row_b)
    if a and b:
        return "both_clean"
    if (not a) and (not b):
        return "both_fail"
    return "mixed"


def lexical_pair_label(row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
    states = []
    for row in (row_a, row_b):
        if is_exact_clean(row):
            states.append("exact")
        elif is_semantic_only(row):
            states.append("semantic_only")
        elif is_semantic_fail(row):
            states.append("semantic_fail")
        else:
            states.append("other")
    return "__".join(sorted(states))


def paired_context_stability(rows767: list[dict[str, Any]], effects: list[dict[str, Any]]) -> dict[str, Any]:
    by_obj_rel: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows767:
        by_obj_rel[(row["object"], row["relation"])][row["context_format"]] = row
    vectors = case_vectors(effects)
    features = sorted({f for vec in vectors.values() for f in vec})
    centered = center_vectors(vectors, features) if features else {}
    pair_rows = []
    for (obj, relation), contexts in sorted(by_obj_rel.items()):
        q = contexts.get("commonsense_question")
        s = contexts.get("commonsense_statement")
        if not q or not s:
            continue
        qid = q["case_id"]
        sid = s["case_id"]
        if qid not in centered or sid not in centered:
            continue
        sim = cosine(centered[qid], centered[sid], features)
        pair_rows.append(
            {
                "object": obj,
                "domain": q["domain"],
                "relation": relation,
                "question_case_id": qid,
                "statement_case_id": sid,
                "context_cosine": sim,
                "semantic_pair": pair_label(q, s, is_semantic_clean),
                "exact_pair": pair_label(q, s, is_exact_clean),
                "lexical_pair": lexical_pair_label(q, s),
            }
        )

    def grouped_mean(key: str) -> dict[str, Any]:
        groups: dict[str, list[float]] = defaultdict(list)
        for row in pair_rows:
            groups[row[key]].append(row["context_cosine"])
        return {
            name: {
                "n_pairs": len(vals),
                "mean_context_cosine": safe_mean(vals),
            }
            for name, vals in sorted(groups.items())
        }

    return {
        "n_pairs": len(pair_rows),
        "mean_context_cosine": safe_mean([r["context_cosine"] for r in pair_rows]),
        "by_semantic_pair": grouped_mean("semantic_pair"),
        "by_exact_pair": grouped_mean("exact_pair"),
        "by_lexical_pair": grouped_mean("lexical_pair"),
        "rows": pair_rows,
    }


def audit_model(args: argparse.Namespace, model: str) -> dict[str, Any]:
    p765 = PHASE765_ROOT / args.phase765_round / f"phase765_{model}_rows.jsonl"
    p767 = phase767_path(model, args.phase767_round)
    rows765 = load_jsonl(p765)
    rows767 = load_jsonl(p767)
    effects = effect_rows_for(rows765)
    meta = case_meta(rows765)
    strata_keys = [x.strip() for x in args.strata_keys.split(",") if x.strip()]
    contrasts = {}
    for label_a, label_b in CONTRASTS:
        if not label_case_ids(rows767, label_a) or not label_case_ids(rows767, label_b):
            contrasts[f"{label_a}_vs_{label_b}"] = {
                "label_a": label_a,
                "label_b": label_b,
                "n_strata": 0,
                "label_a_balanced_ids": [],
                "label_b_balanced_ids": [],
                "skipped": "missing_one_side",
            }
            continue
        contrasts[f"{label_a}_vs_{label_b}"] = summarize_contrast(
            effects,
            meta,
            rows767,
            label_a,
            label_b,
            strata_keys,
            args.max_per_stratum,
        )
    return {
        "model": model,
        "phase765_rows": str(p765),
        "phase767_rows": str(p767),
        "label_counts": {label: len(label_case_ids(rows767, label)) for label in LABELS},
        "balanced_contrasts": contrasts,
        "paired_context_stability": paired_context_stability(rows767, effects),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 770 Balanced Semantic-Clean Fiber Reanalysis ({payload['phase765_round']} / {payload['phase767_round']})",
        "",
        "- Status: `complete`",
        "- Input: Phase 765 causal-fiber effect rows and Phase 767 semantic/exact labels.",
        "- This is an offline balanced reanalysis; no model was loaded.",
        f"- Balanced strata: `{','.join(payload['strata_keys'])}`",
        "",
        "## Label Counts",
        "",
        "| model | exact clean | semantic clean | semantic only | semantic fail |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        counts = payload["by_model"][model]["label_counts"]
        lines.append(
            f"| {model} | {counts['exact_clean']} | {counts['semantic_clean']} | "
            f"{counts['semantic_only']} | {counts['semantic_fail']} |"
        )

    lines += [
        "",
        "## Balanced Contrast Deltas",
        "",
        "Delta means arm A minus arm B after matching counts inside each stratum.",
        "",
        "| model | contrast | strata | cases each | delta sep | delta NN | delta object gap | delta domain gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"][model]
        for name, contrast in data["balanced_contrasts"].items():
            if contrast.get("skipped"):
                lines.append(f"| {model} | `{name}` | 0 | 0 | null | null | null | null |")
                continue
            delta = contrast["delta_a_minus_b"]
            n_each = len(contrast["label_a_balanced_ids"])
            lines.append(
                f"| {model} | `{name}` | {contrast['n_strata']} | {n_each} | "
                f"{fmt(delta['mean_context_separation'])} | {fmt(delta['mean_context_nn_domain_accuracy'])} | "
                f"{fmt(delta['mean_object_stability_gap'])} | {fmt(delta['mean_domain_stability_gap'])} |"
            )

    lines += [
        "",
        "## Balanced Arm Metrics",
        "",
        "| model | contrast | arm | cases | mean sep | mean NN | object gap | domain gap |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"][model]
        for name, contrast in data["balanced_contrasts"].items():
            if contrast.get("skipped"):
                continue
            for arm_name, arm_key in [(contrast["label_a"], "arm_a"), (contrast["label_b"], "arm_b")]:
                arm = contrast[arm_key]
                lines.append(
                    f"| {model} | `{name}` | `{arm_name}` | {arm['n_cases']} | "
                    f"{fmt(arm['mean_context_separation'])} | {fmt(arm['mean_context_nn_domain_accuracy'])} | "
                    f"{fmt(arm['mean_object_stability_gap'])} | {fmt(arm['mean_domain_stability_gap'])} |"
                )

    lines += [
        "",
        "## Paired Context Stability",
        "",
        "Each pair is the same object and same relation across `commonsense_question` and `commonsense_statement`.",
        "",
        "| model | group type | group | pairs | mean context cosine |",
        "|---|---|---|---:|---:|",
    ]
    for model in MODELS:
        paired = payload["by_model"][model]["paired_context_stability"]
        for group_type in ("by_semantic_pair", "by_exact_pair", "by_lexical_pair"):
            for group, row in paired[group_type].items():
                lines.append(
                    f"| {model} | `{group_type}` | `{group}` | {row['n_pairs']} | "
                    f"{fmt(row['mean_context_cosine'])} |"
                )

    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- If balanced deltas differ from Phase 769, the previous subset result was partly caused by object/relation/context distribution.",
        "- If semantic clean still fails to improve fiber metrics after balancing, output closure and internal fiber stability are genuinely separated.",
        "- Paired context stability is stricter than subset filtering because the object and relation are held fixed.",
        "- This audit is still offline and head/source-level; it does not replace new causal interventions or neuron/channel-level atlas work.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase765-round", default="confirm")
    parser.add_argument("--phase767-round", default="main")
    parser.add_argument("--strata-keys", default="domain,relation,context_format")
    parser.add_argument("--max-per-stratum", type=int, default=None)
    args = parser.parse_args()
    strata_keys = [x.strip() for x in args.strata_keys.split(",") if x.strip()]
    payload = {
        "phase": 770,
        "title": "Balanced Semantic-Clean Fiber Reanalysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase765_round": args.phase765_round,
        "phase767_round": args.phase767_round,
        "strata_keys": strata_keys,
        "max_per_stratum": args.max_per_stratum,
        "by_model": {model: audit_model(args, model) for model in MODELS},
    }
    out_name = f"{args.phase765_round}_x_{args.phase767_round}"
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / out_name
        write_json(out_dir / "phase770_balanced_semantic_clean_fiber_reanalysis.json", payload)
        write_markdown(out_dir / "phase770_balanced_semantic_clean_fiber_reanalysis.md", payload)
    print(json.dumps({"status": "complete", "models": MODELS, "out": out_name}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
