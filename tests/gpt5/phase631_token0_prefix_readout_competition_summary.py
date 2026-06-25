#!/usr/bin/env python3
"""
Summarize Phase 631 token0 prefix readout competition results.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase631_token0_prefix_readout_competition")
OUT = ROOT / "phase631_cross_model_summary.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def pick_modes(summary: dict) -> list[dict]:
    wanted = [
        "base",
        "repair_prompt",
        "semantic_cumulative",
        "best_source",
        "best_source_semantic",
    ]
    by_mode = summary["by_mode"]
    rows = [by_mode[m] for m in wanted if m in by_mode]
    readout = [
        item for item in by_mode.values()
        if item["mode"].startswith("readout_scale")
    ]
    readout.sort(key=lambda x: (x["tok0_hit"], x["exact"], x["mean_prefix_margin"]), reverse=True)
    for item in readout[:6]:
        if item not in rows:
            rows.append(item)
    return rows


def mode_table(data: dict) -> str:
    lines = [
        "| mode | tok0 | exact | wrong_exact | mean_prefix_margin | mean_prefix_logit | mean_competitor_logit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in pick_modes(data["summary"]):
        lines.append(
            f"| {item['mode']} | {item['tok0_hit']}/{item['n']} | {item['exact']}/{item['n']} | "
            f"{item['wrong_exact']}/{item['n']} | {fmt(item['mean_prefix_margin'])} | "
            f"{fmt(item['mean_prefix_logit'])} | {fmt(item['mean_competitor_logit'])} |"
        )
    return "\n".join(lines)


def example_block(data: dict) -> str:
    rows = data.get("rows", [])[:4]
    if not rows:
        return ""
    best_readout = None
    for item in data["summary"]["best_tok0"]:
        if item["mode"].startswith("readout_scale"):
            best_readout = item["mode"]
            break
    modes = ["base", "semantic_cumulative", "best_source_semantic"]
    if best_readout:
        modes.append(best_readout)
    out = []
    for row in rows:
        case = row["case"]
        out.append(
            f"- sample {row['sample_idx']} object={case['object']} relation={case['relation']} "
            f"correct={case['correct']} prefix={row['prefix_text']!r} competitor={row['competitor_text']!r}"
        )
        for mode in modes:
            item = row["modes"].get(mode)
            if not item:
                continue
            ev = item["eval"]
            margin = item["logit_summary"]["prefix_minus_competitor"]
            out.append(
                f"  - {mode}: tok0={item['tok0_text']!r} exact={ev['exact_correct']} "
                f"wrong={ev['exact_wrong']} margin={fmt(margin)} text={item['generation']['text']!r}"
            )
    return "\n".join(out)


def main() -> None:
    paths = sorted(ROOT.glob("phase631_*_token0_prefix_readout_competition_confirm.json"))
    if not paths:
        raise SystemExit(f"No confirm files found in {ROOT}")
    blocks = [
        "# Phase 631 Cross-Model Summary",
        "",
        "目标：直接审计第一个生成词元的 prefix/readout competition，并测试 final_norm unembedding 方向注入是否能替代缺失的格式/前缀门。",
        "",
    ]
    for path in paths:
        data = load(path)
        blocks.extend([
            f"## {data['model']}",
            "",
            f"- rows: {data['n_rows']} / raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']}",
            f"- source: {data['source']}",
            f"- downstream_layers: {data['downstream_layers']}",
            f"- scales: {data['scales']}",
            "",
            mode_table(data),
            "",
            "### Examples",
            "",
            example_block(data),
            "",
        ])
    OUT.write_text("\n".join(blocks), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
