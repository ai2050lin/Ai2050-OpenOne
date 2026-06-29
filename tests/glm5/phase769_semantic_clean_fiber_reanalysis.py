#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from phase765_commonsense_context_identity_closure_test import (  # noqa: E402
    build_vectors,
    center_vectors,
    context_stability,
    pair_summary,
)
from phase767_commonsense_failure_type_topk_audit import MODELS  # noqa: E402


PHASE765_ROOT = Path("results/glm5_phase765_commonsense_context_identity_closure_test")
PHASE767_ROOT = Path("tests/result/phase767_commonsense_failure_type_topk_audit")
OUT_ROOT = Path("results/glm5_phase769_semantic_clean_fiber_reanalysis")
RESULT_ROOT = Path("tests/result/phase769_semantic_clean_fiber_reanalysis")


def safe_mean(values: list[Any]) -> float | None:
    vals = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            vals.append(val)
    return sum(vals) / len(vals) if vals else None


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def case_meta(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    meta = {}
    for row in rows:
        if row.get("row_kind") == "commonsense_task_observation":
            meta[row["object"]] = {"object": row["object"], "domain": row["domain"]}
    return meta


def subset_case_ids(phase767_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    out = {
        "all": set(),
        "exact_clean": set(),
        "semantic_clean": set(),
        "semantic_only": set(),
        "semantic_fail": set(),
        "rank_le2": set(),
    }
    for row in phase767_rows:
        cid = row["case_id"]
        out["all"].add(cid)
        if row.get("exact_target_top1"):
            out["exact_clean"].add(cid)
        if row.get("target_top1"):
            out["semantic_clean"].add(cid)
        if row.get("target_top1") and not row.get("exact_target_top1"):
            out["semantic_only"].add(cid)
        if not row.get("target_top1"):
            out["semantic_fail"].add(cid)
        if int(row.get("target_rank") or 999999) <= 2:
            out["rank_le2"].add(cid)
    return out


def summarize_subset(effect_rows: list[dict[str, Any]], meta: dict[str, dict[str, str]], case_ids: set[str]) -> dict[str, Any]:
    rows = [r for r in effect_rows if r.get("case_id") in case_ids]
    vectors_by_context = build_vectors(rows)
    context_rows = []
    for context, vectors in sorted(vectors_by_context.items()):
        features = sorted({f for vec in vectors.values() for f in vec})
        centered = center_vectors(vectors, features) if features else {}
        ps = pair_summary(centered, meta, features) if features else {}
        context_rows.append(
            {
                "context": context,
                "n_objects": len(vectors),
                "n_features": len(features),
                "nn_domain_accuracy": ps.get("nn_domain_accuracy"),
                "same_domain_mean": ps.get("same_domain_mean"),
                "different_domain_mean": ps.get("different_domain_mean"),
                "separation": ps.get("separation"),
            }
        )
    stability = context_stability(vectors_by_context, meta) if vectors_by_context else []
    return {
        "n_cases": len(case_ids),
        "n_effect_rows": len(rows),
        "contexts": context_rows,
        "mean_context_separation": safe_mean([r.get("separation") for r in context_rows]),
        "mean_context_nn_domain_accuracy": safe_mean([r.get("nn_domain_accuracy") for r in context_rows]),
        "cross_context_stability": stability,
        "mean_object_stability_gap": safe_mean([r.get("object_stability_gap") for r in stability]),
        "mean_domain_stability_gap": safe_mean([r.get("domain_stability_gap") for r in stability]),
    }


def audit_model(model: str, phase765_round: str, phase767_round: str) -> dict[str, Any]:
    p765 = PHASE765_ROOT / phase765_round / f"phase765_{model}_rows.jsonl"
    p767 = PHASE767_ROOT / phase767_round / f"phase767_{model}_rows.jsonl"
    if not p767.exists():
        p767 = Path("results/glm5_phase767_commonsense_failure_type_topk_audit") / phase767_round / f"phase767_{model}_rows.jsonl"
    rows765 = load_jsonl(p765)
    rows767 = load_jsonl(p767)
    effects = [r for r in rows765 if r.get("row_kind") == "commonsense_fiber_effect"]
    meta = case_meta(rows765)
    subsets = subset_case_ids(rows767)
    return {
        "model": model,
        "phase765_rows": str(p765),
        "phase767_rows": str(p767),
        "phase765_round": phase765_round,
        "phase767_round": phase767_round,
        "subsets": {name: summarize_subset(effects, meta, ids) for name, ids in subsets.items()},
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Phase 769 Semantic Clean Fiber Reanalysis ({payload['phase765_round']} / {payload['phase767_round']})",
        "",
        "- Status: `complete`",
        "- Input: Phase 765 causal-fiber effect rows filtered by Phase 767 semantic/exact subsets.",
        "- This is an offline reanalysis; no model was loaded.",
        "",
        "## Subset Summary",
        "",
        "| model | subset | cases | effect rows | mean sep | mean NN | object gap | domain gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        data = payload["by_model"].get(model)
        if not data:
            continue
        for subset, row in data["subsets"].items():
            lines.append(
                f"| {model} | `{subset}` | {row['n_cases']} | {row['n_effect_rows']} | "
                f"{fmt(row['mean_context_separation'])} | {fmt(row['mean_context_nn_domain_accuracy'])} | "
                f"{fmt(row['mean_object_stability_gap'])} | {fmt(row['mean_domain_stability_gap'])} |"
            )
    lines += [
        "",
        "## Strict Interpretation",
        "",
        "- If `semantic_clean` improves separation or stability over `all`, failed states were polluting the mechanism graph.",
        "- If `semantic_only` resembles `exact_clean`, lexical realization is likely a surface output issue.",
        "- Small subsets, especially qwen3 `semantic_only`, should not be over-interpreted.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase765-round", default="confirm")
    parser.add_argument("--phase767-round", default="main")
    args = parser.parse_args()
    by_model = {}
    for model in MODELS:
        by_model[model] = audit_model(model, args.phase765_round, args.phase767_round)
    payload = {
        "phase": 769,
        "title": "Semantic Clean Fiber Reanalysis",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "phase765_round": args.phase765_round,
        "phase767_round": args.phase767_round,
        "by_model": by_model,
    }
    for root in (OUT_ROOT, RESULT_ROOT):
        out_dir = root / f"{args.phase765_round}_x_{args.phase767_round}"
        write_json(out_dir / "phase769_semantic_clean_fiber_reanalysis.json", payload)
        write_markdown(out_dir / "phase769_semantic_clean_fiber_reanalysis.md", payload)
    print(json.dumps({"status": "complete", "models": MODELS}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
