#!/usr/bin/env python3
"""Phase447 read-only failure decomposition for Phase446 outputs.

No model loading, no CUDA, no physical traces.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE446_DIR = ROOT / "tests" / "gpt5" / "result" / "phase446_antishortcut_behavior"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase447_failure_decomposition"
OUT_PATH = OUT_DIR / "phase447_failure_decomposition.json"
MODELS = ("qwen3", "glm4", "deepseek7b")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_records(rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    labels: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        buckets[key][row["classification"]] += 1
        labels[key][row["normalized_generated"] or "<empty>"] += 1
    out = []
    for key, counts in sorted(buckets.items()):
        n = sum(counts.values())
        item = {field: value for field, value in zip(key_fields, key, strict=True)}
        item.update({
            "n": n,
            "semantic": counts["semantic"],
            "wrong": counts["wrong"],
            "other": counts["other"],
            "semantic_rate": counts["semantic"] / n if n else 0.0,
            "other_rate": counts["other"] / n if n else 0.0,
            "output_distribution": dict(labels[key]),
        })
        out.append(item)
    return out


def pair_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[(row["model"], f"{row['ability']}/{row['task']}")][row["pair_id"]].append(row)
    out = []
    for (model, task), pairs in sorted(grouped.items()):
        both = only_base = only_cf = neither = 0
        for pair_rows in pairs.values():
            roles = defaultdict(list)
            for row in pair_rows:
                roles[row["pair_role"]].append(row)
            base_ok = any(row["classification"] == "semantic" for row in roles["base"])
            cf_ok = any(row["classification"] == "semantic" for row in roles["counterfactual"])
            if base_ok and cf_ok:
                both += 1
            elif base_ok:
                only_base += 1
            elif cf_ok:
                only_cf += 1
            else:
                neither += 1
        n = both + only_base + only_cf + neither
        out.append({
            "model": model,
            "task": task,
            "n_pairs": n,
            "both_correct": both,
            "base_only": only_base,
            "counterfactual_only": only_cf,
            "neither": neither,
            "both_rate": both / n if n else 0.0,
        })
    return out


def failure_localization(transform_rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = []
    for row in transform_rows:
        fail = row["n"] - row["semantic"]
        failures.append((row["transform"], fail, row["n"], row["semantic_rate"]))
    total_fail = sum(item[1] for item in failures)
    failures_sorted = sorted(failures, key=lambda item: item[1], reverse=True)
    top_fail_share = failures_sorted[0][1] / total_fail if total_fail else 0.0
    rates = [item[3] for item in failures]
    return {
        "total_failures": total_fail,
        "top_failure_transform": failures_sorted[0][0] if failures_sorted else None,
        "top_failure_count": failures_sorted[0][1] if failures_sorted else 0,
        "top_failure_share": top_fail_share,
        "semantic_rate_range": (max(rates) - min(rates)) if rates else 0.0,
        "by_transform_failure_counts": [
            {"transform": name, "failures": fail, "n": n, "semantic_rate": rate}
            for name, fail, n, rate in failures_sorted
        ],
        "localized": top_fail_share >= 0.50,
    }


def classify_level(summary: dict[str, Any], task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    b = summary["behavior"]
    cf = summary["counterfactual"]
    orbit = summary["orbit"]
    if b["semantic_gain"] <= 0 or cf["consistent_pairs"] == 0:
        level = "S0_no_function_evidence"
    elif not cf["counterfactual_pass"]:
        level = "S1_partial_semantic_sensitivity"
    elif b["behavior_pass"] and b["shortcut_pass"] and cf["counterfactual_pass"] and not orbit["orbit_pass"]:
        level = "S2_conditional_function_candidate"
    elif b["behavior_pass"] and b["shortcut_pass"] and cf["counterfactual_pass"] and orbit["orbit_pass"]:
        level = "S3_behavior_stable_window"
    else:
        level = "S1_partial_semantic_sensitivity"
    loc = failure_localization(task_rows)
    atlas = (
        level == "S2_conditional_function_candidate"
        and loc["localized"]
        and b["other"] == 0
    )
    return {
        "level": level,
        "failure_localization": loc,
        "observation_atlas_authorized": atlas,
        "authorization_scope": (
            "limited_non_causal_non_sealed_observation_only"
            if atlas else "analysis_only_no_physical"
        ),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_generations = []
    summaries = {}
    for model in MODELS:
        gen_path = PHASE446_DIR / f"phase446_{model}_generations.jsonl"
        sum_path = PHASE446_DIR / f"phase446_{model}_summary.json"
        if not gen_path.exists() or not sum_path.exists():
            continue
        rows = load_jsonl(gen_path)
        all_generations.extend(rows)
        summaries[model] = load_json(sum_path)

    behavior_rows = [row for row in all_generations if row["stage"] == "behavior_discovery"]
    orbit_rows = [row for row in all_generations if row["stage"] == "counterfactual_orbit_holdout"]

    by_model_task_transform = summarize_records(orbit_rows, ["model", "ability", "task", "transform"])
    by_model_task_stage = summarize_records(all_generations, ["model", "ability", "task", "stage"])
    pairs = pair_summary(behavior_rows)

    task_diagnostics = {}
    for model, summary in summaries.items():
        task_diagnostics[model] = {}
        for task, task_summary in summary["by_task"].items():
            ability, task_name = task.split("/", 1)
            transform_rows = [
                row for row in by_model_task_transform
                if row["model"] == model and row["ability"] == ability and row["task"] == task_name
            ]
            task_diagnostics[model][task] = classify_level(task_summary, transform_rows)

    atlas_candidates = [
        {"model": model, "task": task, **diag}
        for model, tasks in task_diagnostics.items()
        for task, diag in tasks.items()
        if diag["observation_atlas_authorized"]
    ]

    out = {
        "schema_version": "phase447_failure_decomposition.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_model_run_no_cuda",
        "by_model_task_stage": by_model_task_stage,
        "by_model_task_transform": by_model_task_transform,
        "behavior_pair_summary": pairs,
        "task_diagnostics": task_diagnostics,
        "observation_atlas_candidates": atlas_candidates,
        "strict_physical_candidates": [],
        "next_authorized": (
            "limited_observation_atlas_protocol" if atlas_candidates else "interface_redesign_protocol"
        ),
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
