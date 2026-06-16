#!/usr/bin/env python3
"""Summarize Phase 158 step-wise competition trace results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase158_stepwise_competition_trace")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["clean", "mlp_joint", "joint_k8_mlp_joint", "random_k8"]
DIFFICULT_FORMATS = {"label_colon", "answer_one_word", "quoted_answer", "list_answer"}


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase158_{model}_stepwise_competition_trace.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def hit(case: dict[str, Any], cond: str) -> float:
    return float(case["conditions"][cond]["hit_rate"])


def step_margin(case: dict[str, Any], cond: str, step_idx: int) -> float:
    return float(case["conditions"][cond]["steps"][step_idx]["competition"]["margins"]["correct_vs_competitor"])


def step_top1_rate(case: dict[str, Any], cond: str, step_idx: int, label: str) -> float:
    return float(case["conditions"][cond]["steps"][step_idx]["topk"]["top1_label_rates"].get(label, 0.0))


def top_trajectory(case: dict[str, Any], cond: str) -> str:
    rates = case["conditions"][cond]["trajectory_rates"]
    if not rates:
        return ""
    k, v = max(rates.items(), key=lambda kv: kv[1])
    return f"{k}:{v:.2f}"


def aggregate_traj(cases: list[dict[str, Any]], cond: str) -> Counter[str]:
    acc: Counter[str] = Counter()
    for case in cases:
        for name, rate in case["conditions"][cond]["trajectory_rates"].items():
            acc[name] += float(rate)
    return acc


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(cases)), "clean_hit": mean([hit(c, "clean") for c in cases])}
    for cond in CONDITIONS:
        out[f"{cond}_hit"] = mean([hit(c, cond) for c in cases])
        out[f"{cond}_hit_delta"] = out[f"{cond}_hit"] - out["clean_hit"]
        for step in range(3):
            out[f"{cond}_step{step+1}_margin"] = mean([step_margin(c, cond, step) for c in cases])
            out[f"{cond}_step{step+1}_correct_top1"] = mean([step_top1_rate(c, cond, step, "correct_expanded") for c in cases])
            out[f"{cond}_step{step+1}_wrong_top1"] = mean([step_top1_rate(c, cond, step, "wrong_category") for c in cases])
            out[f"{cond}_step{step+1}_generic_top1"] = mean([step_top1_rate(c, cond, step, "generic_continue") for c in cases])
            out[f"{cond}_step{step+1}_format_top1"] = mean([step_top1_rate(c, cond, step, "format_target") for c in cases])
    return out


def emit_table(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| group | n | clean_hit | mlp_hit | k8+mlp_hit | random_hit | "
        "mlp_delta | k8+mlp_delta | random_delta | "
        "clean_m1 | mlp_m1 | mlp_m2 | mlp_m3 | "
        "mlp_correct_top1_s1 | mlp_wrong_top1_s1 | mlp_generic_top1_s1 | mlp_format_top1_s1 | "
        "top_clean_traj | top_mlp_traj | top_k8mlp_traj |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group, cases in sorted(grouped.items()):
        s = summarize_cases(cases)
        clean_traj = aggregate_traj(cases, "clean").most_common(1)
        mlp_traj = aggregate_traj(cases, "mlp_joint").most_common(1)
        joint_traj = aggregate_traj(cases, "joint_k8_mlp_joint").most_common(1)
        lines.append(
            f"| {group} | {int(s['n'])} | {s['clean_hit']:.3f} | {s['mlp_joint_hit']:.3f} | "
            f"{s['joint_k8_mlp_joint_hit']:.3f} | {s['random_k8_hit']:.3f} | "
            f"{s['mlp_joint_hit_delta']:+.3f} | {s['joint_k8_mlp_joint_hit_delta']:+.3f} | "
            f"{s['random_k8_hit_delta']:+.3f} | "
            f"{s['clean_step1_margin']:.3f} | {s['mlp_joint_step1_margin']:.3f} | "
            f"{s['mlp_joint_step2_margin']:.3f} | {s['mlp_joint_step3_margin']:.3f} | "
            f"{s['mlp_joint_step1_correct_top1']:.3f} | {s['mlp_joint_step1_wrong_top1']:.3f} | "
            f"{s['mlp_joint_step1_generic_top1']:.3f} | {s['mlp_joint_step1_format_top1']:.3f} | "
            f"{clean_traj[0][0] if clean_traj else ''} | "
            f"{mlp_traj[0][0] if mlp_traj else ''} | "
            f"{joint_traj[0][0] if joint_traj else ''} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 158 Cross-model Step-wise Competition Trace Summary", ""]
    cross: dict[str, list[dict[str, Any]]] = {}
    for model in MODELS:
        result = load_result(model)
        lines.append(f"## {model}")
        lines.append("")
        if result is None:
            lines.append("Missing result.")
            lines.append("")
            continue
        cases_by_key = result["results"]
        cases = list(cases_by_key.values())
        cross[model] = cases
        lines.append(
            f"cases={len(cases)}, attention=L{result['attention_layer']}, mlp=L{result['mlp_layer']}, "
            f"heads={result['num_heads']}, steps={result['steps']}, top_k={result['top_k']}"
        )
        lines.append("")
        difficult = [c for c in cases if c["format"] in DIFFICULT_FORMATS]
        mc = [c for c in cases if c["format"] == "multiple_choice"]
        emit_table(lines, "All / difficult / multiple-choice", {"all": cases, "difficult_formats": difficult, "multiple_choice_control": mc})
        for title, field in [("By format", "format"), ("By category", "category"), ("By family", "family"), ("By split", "split")]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for case in cases:
                grouped[case[field]].append(case)
            emit_table(lines, title, grouped)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | clean | mlp | k8+mlp | random | clean_traj | mlp_traj | k8+mlp_traj |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, case in sorted(cases_by_key.items()):
            lines.append(
                f"| {key} | {hit(case, 'clean'):.2f} | {hit(case, 'mlp_joint'):.2f} | "
                f"{hit(case, 'joint_k8_mlp_joint'):.2f} | {hit(case, 'random_k8'):.2f} | "
                f"{top_trajectory(case, 'clean')} | {top_trajectory(case, 'mlp_joint')} | "
                f"{top_trajectory(case, 'joint_k8_mlp_joint')} |"
            )
        lines.append("")
    if cross:
        emit_table(
            lines,
            "Cross-model difficult-format core",
            {model: [c for c in cases if c["format"] in DIFFICULT_FORMATS] for model, cases in cross.items()},
        )
    path = OUT_ROOT / "phase158_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
