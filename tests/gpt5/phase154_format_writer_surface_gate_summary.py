#!/usr/bin/env python3
"""Summarize Phase 154 format writer surface gate results."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


OUT_ROOT = Path("results/gpt5_phase154_format_writer_surface_gate")
MODELS = ["qwen3", "glm4", "deepseek7b"]
MODES = ["semantic_proj", "format_proj", "joint_proj"]


def load_result(model: str) -> dict[str, Any] | None:
    path = OUT_ROOT / f"phase154_{model}_format_writer_surface_gate.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def best(case: dict[str, Any], mode: str, field: str) -> dict[str, Any]:
    rows = [r for r in case.get("conditions", []) if r["mode"] == mode]
    return max(rows, key=lambda r: r[field]) if rows else {}


def row_id(row: dict[str, Any]) -> str:
    if not row:
        return ""
    return f"L{row['patch_layer']}:{row['component']}"


def emit(lines: list[str], title: str, grouped: dict[str, list[dict[str, Any]]]) -> None:
    lines.append(f"### By {title}")
    lines.append("")
    lines.append("| group | n | clean_answer_rank | clean_format_rank | sem_answer_damage | fmt_format_damage | joint_answer_damage | joint_format_damage | top_sem_writer | top_fmt_writer | top_joint_writer |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for group, cases in sorted(grouped.items()):
        sem_ans = [best(c, "semantic_proj", "answer_rank_delta") for c in cases]
        fmt_fmt = [best(c, "format_proj", "format_rank_delta") for c in cases]
        joint_ans = [best(c, "joint_proj", "answer_rank_delta") for c in cases]
        joint_fmt = [best(c, "joint_proj", "format_rank_delta") for c in cases]
        top_sem = Counter(row_id(x) for x in sem_ans if x).most_common(1)
        top_fmt = Counter(row_id(x) for x in fmt_fmt if x).most_common(1)
        top_joint = Counter(row_id(x) for x in joint_ans if x).most_common(1)
        lines.append(
            f"| {group} | {len(cases)} | "
            f"{mean([c['clean']['answer_rank'] for c in cases]):.1f} | "
            f"{mean([c['clean']['format_rank'] for c in cases]):.1f} | "
            f"{mean([x.get('answer_rank_delta', 0.0) for x in sem_ans]):+.1f} | "
            f"{mean([x.get('format_rank_delta', 0.0) for x in fmt_fmt]):+.1f} | "
            f"{mean([x.get('answer_rank_delta', 0.0) for x in joint_ans]):+.1f} | "
            f"{mean([x.get('format_rank_delta', 0.0) for x in joint_fmt]):+.1f} | "
            f"{top_sem[0][0] if top_sem else ''} | {top_fmt[0][0] if top_fmt else ''} | {top_joint[0][0] if top_joint else ''} |"
        )
    lines.append("")


def component_table(lines: list[str], cases: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        for r in c.get("conditions", []):
            grouped[(r["mode"], r["component"], int(r["patch_layer"]))].append(r)
    lines.append("### By Writer Condition")
    lines.append("")
    lines.append("| mode | component | layer | n | answer_rank_delta | format_rank_delta | answer_argmax_delta | format_argmax_delta |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for (mode, comp, layer), rows in sorted(grouped.items()):
        lines.append(
            f"| {mode} | {comp} | L{layer} | {len(rows)} | "
            f"{mean([r['answer_rank_delta'] for r in rows]):+.1f} | "
            f"{mean([r['format_rank_delta'] for r in rows]):+.1f} | "
            f"{mean([r['answer_argmax_delta'] for r in rows]):+.3f} | "
            f"{mean([r['format_argmax_delta'] for r in rows]):+.3f} |"
        )
    lines.append("")


def main() -> None:
    lines = ["# Phase 154 Cross-model Format Writer Surface Gate Summary", ""]
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
        lines.append(
            f"cases={len(cases)}, patch_layers={result.get('patch_layers')}, "
            f"formats={','.join(result.get('formats', []))}"
        )
        lines.append("")
        for title, idx in [("category", 3), ("format", 2), ("family", 1), ("split", 0)]:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for key, case in cases_by_key.items():
                grouped[key.split(":")[idx]].append(case)
            emit(lines, title, grouped)
        component_table(lines, cases)
        lines.append("### Cases")
        lines.append("")
        lines.append("| case | clean_ans | clean_fmt | sem_ans | fmt_fmt | joint_ans | joint_fmt |")
        lines.append("|---|---|---|---|---|---|---|")
        for key, case in sorted(cases_by_key.items()):
            sem = best(case, "semantic_proj", "answer_rank_delta")
            fmt = best(case, "format_proj", "format_rank_delta")
            ja = best(case, "joint_proj", "answer_rank_delta")
            jf = best(case, "joint_proj", "format_rank_delta")

            def f(row: dict[str, Any], field: str) -> str:
                return "" if not row else f"{row_id(row)} {row[field]:+.1f}"

            lines.append(
                f"| {key} | {case['clean']['answer_rank']:.1f} | {case['clean']['format_rank']:.1f} | "
                f"{f(sem, 'answer_rank_delta')} | {f(fmt, 'format_rank_delta')} | "
                f"{f(ja, 'answer_rank_delta')} | {f(jf, 'format_rank_delta')} |"
            )
        lines.append("")
    path = OUT_ROOT / "phase154_cross_model_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
