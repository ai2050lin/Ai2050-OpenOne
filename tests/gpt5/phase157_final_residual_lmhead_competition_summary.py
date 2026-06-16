#!/usr/bin/env python3
"""Summarize Phase 157 final residual and LM-head competition results."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase157_final_residual_lmhead_competition")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = ["mlp_joint", "joint_k8_mlp_joint", "random_k8"]
GROUPS = ["correct_expanded", "wrong_category", "format_target", "generic_continue", "object_copy", "option_label"]
DIFFICULT_FORMATS = {"label_colon", "answer_one_word", "quoted_answer", "list_answer"}


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase157_{model}_final_residual_lmhead_competition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    vals = [x for x in xs if np.isfinite(x)]
    return float(np.mean(vals)) if vals else 0.0


def comp(case: dict[str, Any], cond: str) -> dict[str, Any]:
    return case["conditions"][cond]["competition"]


def hidden(case: dict[str, Any], cond: str) -> dict[str, float]:
    return case["conditions"][cond]["hidden"]


def group_logit(case: dict[str, Any], cond: str, group: str) -> float:
    return float(comp(case, cond)["groups"][group]["max_logit"])


def margin(case: dict[str, Any], cond: str) -> float:
    return float(comp(case, cond)["margins"]["correct_vs_competitor"])


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {"n": float(len(cases)), "clean_margin": mean([margin(c, "clean") for c in cases])}
    for cond in CONDITIONS:
        out[f"{cond}_margin_delta"] = mean([margin(c, cond) - margin(c, "clean") for c in cases])
        out[f"{cond}_hidden_delta_norm"] = mean([hidden(c, cond)["delta_norm"] for c in cases])
        out[f"{cond}_delta_sem_proj"] = mean([hidden(c, cond)["delta_semantic_projection_norm"] for c in cases])
        out[f"{cond}_delta_fmt_proj"] = mean([hidden(c, cond)["delta_format_projection_norm"] for c in cases])
        for group in GROUPS:
            out[f"{cond}_{group}_delta"] = mean([group_logit(c, cond, group) - group_logit(c, "clean", group) for c in cases])
    return out


def emit_table(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| group | n | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | "
        "mlp_correctΔ | mlp_wrongΔ | mlp_formatΔ | mlp_genericΔ | "
        "k8+mlp_correctΔ | k8+mlp_wrongΔ | k8+mlp_formatΔ | k8+mlp_genericΔ | "
        "mlp_hiddenΔ | k8+mlp_hiddenΔ |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group, cases in sorted(grouped.items()):
        s = summarize_cases(cases)
        lines.append(
            f"| {group} | {int(s['n'])} | {s['clean_margin']:.3f} | "
            f"{s['mlp_joint_margin_delta']:+.3f} | {s['joint_k8_mlp_joint_margin_delta']:+.3f} | "
            f"{s['random_k8_margin_delta']:+.3f} | "
            f"{s['mlp_joint_correct_expanded_delta']:+.3f} | {s['mlp_joint_wrong_category_delta']:+.3f} | "
            f"{s['mlp_joint_format_target_delta']:+.3f} | {s['mlp_joint_generic_continue_delta']:+.3f} | "
            f"{s['joint_k8_mlp_joint_correct_expanded_delta']:+.3f} | {s['joint_k8_mlp_joint_wrong_category_delta']:+.3f} | "
            f"{s['joint_k8_mlp_joint_format_target_delta']:+.3f} | {s['joint_k8_mlp_joint_generic_continue_delta']:+.3f} | "
            f"{s['mlp_joint_hidden_delta_norm']:.3f} | {s['joint_k8_mlp_joint_hidden_delta_norm']:.3f} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 157 Cross-model Final Residual LM-head Competition Summary", ""]
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
            f"cases={len(cases)}, attention=L{result['attention_layer']}, "
            f"mlp=L{result['mlp_layer']}, heads={result['num_heads']}"
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
        lines.append("| case | clean_margin | mlp_marginΔ | k8+mlp_marginΔ | random_marginΔ | mlp_correctΔ | mlp_wrongΔ | mlp_genericΔ |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, case in sorted(cases_by_key.items()):
            lines.append(
                f"| {key} | {margin(case, 'clean'):.2f} | "
                f"{margin(case, 'mlp_joint') - margin(case, 'clean'):+.2f} | "
                f"{margin(case, 'joint_k8_mlp_joint') - margin(case, 'clean'):+.2f} | "
                f"{margin(case, 'random_k8') - margin(case, 'clean'):+.2f} | "
                f"{group_logit(case, 'mlp_joint', 'correct_expanded') - group_logit(case, 'clean', 'correct_expanded'):+.2f} | "
                f"{group_logit(case, 'mlp_joint', 'wrong_category') - group_logit(case, 'clean', 'wrong_category'):+.2f} | "
                f"{group_logit(case, 'mlp_joint', 'generic_continue') - group_logit(case, 'clean', 'generic_continue'):+.2f} |"
            )
        lines.append("")
    if cross:
        emit_table(
            lines,
            "Cross-model difficult-format core",
            {model: [c for c in cases if c["format"] in DIFFICULT_FORMATS] for model, cases in cross.items()},
        )
    path = OUT_ROOT / "phase157_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
