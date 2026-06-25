#!/usr/bin/env python3
"""
Summarize Phase 634 multi-position format source field closure results.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase634_multi_position_format_source_field_closure")
OUT = ROOT / "phase634_cross_model_summary.md"


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_rows(data: dict) -> list[dict]:
    rows = data["summary"]["by_mode"]
    base_order = ["base", "repair_prompt", "semantic_cumulative"]
    baselines = [r for name in base_order for r in rows if r["mode"] == name]
    important = [
        r for r in rows
        if r["mode"].endswith("_restore_semantic")
        or r["mode"].endswith("_random_semantic")
        or r["mode"].endswith("_reverse_semantic")
        or r["mode"].endswith("_remove_from_repair")
    ]
    important.sort(key=lambda x: x["mode"])
    extras = [r for r in rows if r not in baselines and r not in important]
    return baselines + important + extras[:16]


def table(data: dict) -> str:
    lines = [
        "| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |",
        "|---|---:|---:|---:|---:|---|",
    ]
    seen = set()
    for item in ordered_rows(data):
        if item["mode"] in seen:
            continue
        seen.add(item["mode"])
        top0 = ", ".join(f"{k}:{v}" for k, v in item.get("top0_text", {}).items())
        lines.append(
            f"| {item['mode']} | {item['tok0_hit']}/{item['n']} | "
            f"{item['exact']}/{item['n']} | {item['wrong_exact']}/{item['n']} | "
            f"{fmt(item['mean_prefix_margin'])} | {top0} |"
        )
    return "\n".join(lines)


def examples(data: dict, n: int = 10) -> str:
    keep = {
        "base",
        "semantic_cumulative",
        "single_prompt_last_restore_semantic",
        "answer_prompt_restore_semantic",
        "relation_answer_prompt_restore_semantic",
        "question_all_answer_prompt_restore_semantic",
        "all6_restore_semantic",
        "all6_random_semantic",
        "all6_reverse_semantic",
    }
    out = []
    for row in data.get("rows", [])[:320]:
        if row["mode"] not in keep or not row.get("generation_text"):
            continue
        out.append(
            f"- sample={row['sample_idx']} mode={row['mode']} tok0={row['tok0_text']!r} "
            f"exact={row['eval']['exact_correct']} wrong={row['eval']['exact_wrong']} "
            f"margin={fmt(row['prefix_margin'])} text={row['generation_text']!r}"
        )
        if len(out) >= n:
            break
    return "\n".join(out)


def main() -> None:
    paths = sorted(ROOT.glob("phase634_*_multi_position_format_source_field_closure_confirm.json"))
    if not paths:
        raise SystemExit(f"No confirm files found in {ROOT}")
    blocks = [
        "# Phase 634 Cross-Model Summary",
        "",
        "目标：测试多位置 source/format field 是否能补上 Phase633 排除 prompt_last 后剩余的 token0 prefix gate 缺口。",
        "",
    ]
    for path in paths:
        data = load(path)
        blocks.extend([
            f"## {data['model']}",
            "",
            f"- rows: {data['n_rows']} / raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']}",
            f"- layer_map: {data['layer_map']}",
            f"- set_defs: {data['set_defs']}",
            f"- downstream_layers: {data['downstream_layers']}",
            "",
            table(data),
            "",
            "### Examples",
            "",
            examples(data),
            "",
        ])
    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
