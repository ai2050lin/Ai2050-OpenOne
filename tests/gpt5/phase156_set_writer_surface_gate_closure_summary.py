#!/usr/bin/env python3
"""Summarize Phase 156 set-writer surface gate closure results."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase156_set_writer_surface_gate_closure")
MODELS = ["qwen3", "glm4", "deepseek7b"]
CONDITIONS = [
    "clean",
    "joint_k1",
    "joint_k4",
    "joint_k8",
    "answer_k4",
    "format_k4",
    "random_k4",
    "random_k8",
    "mlp_joint",
    "joint_k4_mlp_joint",
    "joint_k8_mlp_joint",
]
DIFFICULT_FORMATS = {"label_colon", "answer_one_word", "quoted_answer", "list_answer"}


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase156_{model}_set_writer_surface_gate_closure.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def hit(case: dict[str, Any], cond: str) -> float:
    return float(case["generations"][cond]["hit_rate"])


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, float]:
    clean = mean([hit(c, "clean") for c in cases])
    out = {"n": float(len(cases)), "clean": clean}
    for cond in CONDITIONS:
        val = mean([hit(c, cond) for c in cases])
        out[cond] = val
        out[f"{cond}_delta"] = val - clean
    return out


def emit_group_table(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| group | n | clean | joint_k4 | joint_k8 | random_k4 | random_k8 | "
        "mlp_joint | k4+mlp | k8+mlp | joint_k4_delta | joint_k8_delta | "
        "mlp_delta | k4+mlp_delta | k8+mlp_delta |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for group, cases in sorted(grouped.items()):
        s = summarize_cases(cases)
        lines.append(
            f"| {group} | {int(s['n'])} | {s['clean']:.3f} | "
            f"{s['joint_k4']:.3f} | {s['joint_k8']:.3f} | "
            f"{s['random_k4']:.3f} | {s['random_k8']:.3f} | "
            f"{s['mlp_joint']:.3f} | {s['joint_k4_mlp_joint']:.3f} | "
            f"{s['joint_k8_mlp_joint']:.3f} | "
            f"{s['joint_k4_delta']:+.3f} | {s['joint_k8_delta']:+.3f} | "
            f"{s['mlp_joint_delta']:+.3f} | {s['joint_k4_mlp_joint_delta']:+.3f} | "
            f"{s['joint_k8_mlp_joint_delta']:+.3f} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 156 Cross-model Set-Writer Surface Gate Closure Summary", ""]
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
            f"mlp=L{result['mlp_layer']}, heads={result['num_heads']}, steps={result['steps']}"
        )
        lines.append("")
        hard = [c for c in cases if c["format"] in DIFFICULT_FORMATS]
        control = [c for c in cases if c["format"] == "multiple_choice"]
        emit_group_table(lines, "All cases", {"all": cases, "difficult_formats": hard, "multiple_choice_control": control})
        for title, field in [("By category", "category"), ("By format", "format"), ("By family", "family"), ("By split", "split")]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for case in cases:
                grouped[case[field]].append(case)
            emit_group_table(lines, title, grouped)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | clean | joint_k4 | joint_k8 | random_k4 | mlp_joint | k4+mlp | k8+mlp |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, case in sorted(cases_by_key.items()):
            lines.append(
                f"| {key} | {hit(case, 'clean'):.2f} | {hit(case, 'joint_k4'):.2f} | "
                f"{hit(case, 'joint_k8'):.2f} | {hit(case, 'random_k4'):.2f} | "
                f"{hit(case, 'mlp_joint'):.2f} | {hit(case, 'joint_k4_mlp_joint'):.2f} | "
                f"{hit(case, 'joint_k8_mlp_joint'):.2f} |"
            )
        lines.append("")
    if cross:
        lines.append("## Cross-model Difficult-format Core")
        lines.append("")
        emit_group_table(
            lines,
            "Difficult formats by model",
            {m: [c for c in cases if c["format"] in DIFFICULT_FORMATS] for m, cases in cross.items()},
        )
    path = OUT_ROOT / "phase156_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
