#!/usr/bin/env python3
"""Phase444 read-only analysis of Phase443 behavior boundary results.

No model loading, no CUDA, no sample/protocol edits.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE442_DIR = ROOT / "tests" / "gpt5" / "result" / "phase442_static_sample_contract"
PHASE443_DIR = ROOT / "tests" / "gpt5" / "result" / "phase443_behavior_qualification"
OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase444_behavior_boundary_analysis"
OUT_PATH = OUT_DIR / "phase444_behavior_boundary_analysis.json"

ANSWER_POOL = ["red", "blue", "green", "gold"]
SELECTED_TASKS = {
    "knowledge_network": "category_attribute_inheritance",
    "single_step_reasoning": "conditional_implication_one_step",
    "syntax_system": "active_passive_role_conversion",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\s:：,，.;。!！?？\"'`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.,;:!?，。；：！？\"'`")


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def normalized_edit(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return edit_distance(a, b) / denom


def lcp_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def generated_token(text: str) -> str:
    text = norm(text)
    if not text:
        return ""
    return re.split(r"\s+", text, maxsplit=1)[0].strip(".,;:!?，。；：！？\"'`")


def classify_glm4_syntax_failures(samples_by_id: dict[str, dict[str, Any]], generations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row for row in generations
        if row["stage"] == "surface_orbit_holdout"
        and row["ability"] == "syntax_system"
        and row["task"] == "active_passive_role_conversion"
    ]
    categories: Counter[str] = Counter()
    by_transform: dict[str, Counter[str]] = defaultdict(Counter)
    margins = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        sample = samples_by_id[row["sample_id"]]
        target = norm(row["canonical_answer"])
        opposite = norm(sample["role_nodes"].get("distractor", ""))
        out_full = norm(row["generated"])
        out = generated_token(row["generated"])
        target_prefix = bool(out) and target.startswith(out) and out != target
        target_contains = bool(target) and target in out_full
        opposite_prefix = bool(out) and opposite.startswith(out) and out != opposite
        opposite_contains = bool(opposite) and opposite in out_full

        if row["classification"] == "semantic":
            category = "exact_or_accepted_target"
        elif target_prefix:
            category = "target_prefix_truncation"
        elif target_contains:
            category = "target_complete_overgenerated"
        elif opposite_prefix:
            category = "opposite_role_prefix"
        elif opposite_contains or out == opposite:
            category = "opposite_role_target"
        elif out in {norm(value) for value in sample["role_nodes"].values()}:
            category = "other_known_entity"
        else:
            category = "unrelated_or_format_output"

        d_target = normalized_edit(out, target)
        d_opposite = normalized_edit(out, opposite) if opposite else 1.0
        margin = d_opposite - d_target
        prefix_ratio = lcp_len(out, target) / max(len(target), 1)
        record = {
            "sample_id": row["sample_id"],
            "transform": row["transform"],
            "target": target,
            "opposite": opposite,
            "output_token": out,
            "raw_generated": row["generated"],
            "category": category,
            "role_margin": margin,
            "target_prefix_ratio": prefix_ratio,
        }
        categories[category] += 1
        by_transform[row["transform"]][category] += 1
        margins.append(record)
        if len(examples[category]) < 5:
            examples[category].append(record)

    strict_failures = [item for item in margins if item["category"] != "exact_or_accepted_target"]
    positive_role_margin = sum(1 for item in strict_failures if item["role_margin"] > 0)
    high_prefix = sum(1 for item in strict_failures if item["target_prefix_ratio"] >= 0.75)
    return {
        "n": len(rows),
        "category_counts": dict(categories),
        "by_transform": {key: dict(value) for key, value in sorted(by_transform.items())},
        "strict_failure_count": len(strict_failures),
        "strict_failures_with_positive_role_margin": positive_role_margin,
        "strict_failures_with_target_prefix_ratio_ge_0_75": high_prefix,
        "mean_role_margin_on_failures": (
            sum(item["role_margin"] for item in strict_failures) / len(strict_failures)
            if strict_failures else None
        ),
        "mean_target_prefix_ratio_on_failures": (
            sum(item["target_prefix_ratio"] for item in strict_failures) / len(strict_failures)
            if strict_failures else None
        ),
        "examples": dict(examples),
        "interpretation": (
            "GLM4 syntax strict failures are dominated by serialization/prefix/form failures"
            if categories["target_prefix_truncation"] + categories["target_complete_overgenerated"] + categories["unrelated_or_format_output"] > categories["opposite_role_target"] + categories["opposite_role_prefix"]
            else "GLM4 syntax failures include substantial opposite-role evidence"
        ),
    }


def selected_holdout_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in samples
        if row["split"] == "surface_orbit_holdout"
        and SELECTED_TASKS.get(row["ability"]) == row["task"]
    ]


def baseline_predictions(row: dict[str, Any], all_answers_by_task: dict[str, Counter[str]]) -> dict[str, str]:
    task_key = f"{row['ability']}/{row['task']}"
    majority = all_answers_by_task[task_key].most_common(1)[0][0]
    text = row["input_text"]
    colors = [color for color in ANSWER_POOL if re.search(rf"\b{re.escape(color)}\b", text)]
    role_values = list(row["role_nodes"].values())
    entities = re.findall(r"\b[a-z]{2}_[a-z]{4}_soh_[0-9]{3}_ent_a[x]*\b", text)
    return {
        "majority": majority,
        "answer_length_mode": majority,
        "first_color_in_prompt": colors[0] if colors else "",
        "last_color_in_prompt": colors[-1] if colors else "",
        "first_role_node": role_values[0] if role_values else "",
        "last_role_node": role_values[-1] if role_values else "",
        "first_entity_token": entities[0] if entities else "",
        "last_entity_token": entities[-1] if entities else "",
        "query_last_token": norm(text.split("?")[0].split()[-1]) if "?" in text and text.split("?")[0].split() else "",
    }


def qwen_baseline_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows = selected_holdout_samples(samples)
    answers_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        answers_by_task[f"{row['ability']}/{row['task']}"][row["canonical_answer"]] += 1

    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        task_key = f"{row['ability']}/{row['task']}"
        preds = baseline_predictions(row, answers_by_task)
        for name, pred in preds.items():
            counts[task_key][name]["correct" if norm(pred) == norm(row["canonical_answer"]) else "wrong"] += 1

    report = {}
    closable = {}
    for task_key, baselines in sorted(counts.items()):
        report[task_key] = {}
        closable[task_key] = False
        for name, counter in sorted(baselines.items()):
            n = counter["correct"] + counter["wrong"]
            acc = counter["correct"] / n if n else 0.0
            report[task_key][name] = {"n": n, "correct": counter["correct"], "accuracy": acc}
            if acc >= 0.95:
                closable[task_key] = True
    return {
        "task_baselines": report,
        "tasks_with_simple_baseline_ge_0_95": [task for task, flag in closable.items() if flag],
        "interpretation": "At least one selected task has a near-perfect string/template baseline" if any(closable.values()) else "No selected task has a near-perfect checked baseline",
    }


def stop_scope_audit(protocol: dict[str, Any], phase443_aggregate: dict[str, Any]) -> dict[str, Any]:
    protocol_text = json.dumps(protocol, ensure_ascii=False)
    has_global_stop = any(
        phrase in protocol_text
        for phrase in [
            "global",
            "cross_model",
            "do_not_continue_after_model",
            "全局",
            "跨模型",
        ]
    )
    has_model_specific = "single_model_only_pass_record_model_specific_route" in protocol_text
    return {
        "phase443_stopped_at": phase443_aggregate.get("stopped_at"),
        "deepseek7b_status": phase443_aggregate["models"].get("deepseek7b", {}).get("status"),
        "explicit_global_cross_model_stop_found": has_global_stop,
        "model_specific_route_rule_found": has_model_specific,
        "conclusion": (
            "Phase443 global stop is explicitly supported by protocol text"
            if has_global_stop
            else "No explicit frozen global cross-model stop was found; DS7B should be treated as unknown, and Phase443 is incomplete for three-model behavior comparison"
        ),
    }


def authorization_decision(qwen_baselines: dict[str, Any], glm4_failure: dict[str, Any], stop_scope: dict[str, Any]) -> dict[str, Any]:
    blocked_tasks = set(qwen_baselines["tasks_with_simple_baseline_ge_0_95"])
    candidates = []
    for ability, task in SELECTED_TASKS.items():
        key = f"{ability}/{task}"
        if key not in blocked_tasks:
            candidates.append({"model": "qwen3", "ability": ability, "task": task, "authorization": "behavior_pass_baseline_not_closed"})
    for ability in ("knowledge_network", "single_step_reasoning"):
        task = SELECTED_TASKS[ability]
        key = f"{ability}/{task}"
        if key not in blocked_tasks:
            candidates.append({"model": "glm4", "ability": ability, "task": task, "authorization": "behavior_pass_baseline_not_closed"})
    return {
        "authorized_for_minimal_physical_after_phase444": candidates,
        "blocked_or_needs_redesign": sorted(blocked_tasks),
        "glm4_syntax_status": "strict_behavior_failed_role_vs_serialization_decomposed",
        "deepseek7b_status": "unknown_pending_protocol_consistent_completion" if not stop_scope["explicit_global_cross_model_stop_found"] else "not_run_by_global_stop",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = load_json(PHASE442_DIR / "phase442_protocol_v3_freeze.json")
    samples = load_jsonl(PHASE442_DIR / "phase442_samples.jsonl")
    samples_by_id = {row["sample_id"]: row for row in samples}
    glm4_generations = load_jsonl(PHASE443_DIR / "phase443_glm4_generations.jsonl")
    aggregate = load_json(PHASE443_DIR / "phase443_behavior_aggregate_summary.json")

    glm4_failure = classify_glm4_syntax_failures(samples_by_id, glm4_generations)
    qwen_baselines = qwen_baseline_audit(samples)
    stop_scope = stop_scope_audit(protocol, aggregate)
    authorization = authorization_decision(qwen_baselines, glm4_failure, stop_scope)

    out = {
        "schema_version": "phase444_behavior_boundary_analysis.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "analysis_only_no_cuda_no_model_run",
        "glm4_syntax_failure_decomposition": glm4_failure,
        "qwen3_selected_task_baseline_audit": qwen_baselines,
        "ds7b_stop_scope_audit": stop_scope,
        "authorization_decision": authorization,
    }
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
