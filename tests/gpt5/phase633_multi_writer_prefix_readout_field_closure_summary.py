#!/usr/bin/env python3
"""
Summarize Phase 633 multi-writer prefix readout field closure results.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase633_multi_writer_prefix_readout_field_closure")
OUT = ROOT / "phase633_cross_model_summary.md"


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ordered_modes(data: dict) -> list[dict]:
    rows = data["summary"]["by_mode"]
    baselines = [r for r in rows if r["mode"] in {"base", "repair_prompt", "semantic_cumulative"}]
    key_modes = [
        r for r in rows
        if r["mode"].endswith("_restore_semantic")
        or r["mode"].endswith("_random_semantic")
        or r["mode"].endswith("_reverse_semantic")
        or r["mode"].endswith("_remove_from_repair")
    ]
    key_modes.sort(key=lambda x: (x["mode"].split("_")[0], x["mode"]))
    best = [r for r in rows if r not in baselines and r not in key_modes]
    return baselines + key_modes + best[:12]


def mode_table(data: dict) -> str:
    lines = [
        "| mode | tok0 | exact | wrong_exact | mean_prefix_margin | top0_text |",
        "|---|---:|---:|---:|---:|---|",
    ]
    seen = set()
    for item in ordered_modes(data):
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


def examples(data: dict, n: int = 8) -> str:
    out = []
    for row in data.get("rows", [])[:260]:
        if row["mode"] not in {
            "base",
            "semantic_cumulative",
            "top1_restore_semantic",
            "top4_restore_semantic",
            "top8_restore_semantic",
            "top12_restore_semantic",
            "top12_random_semantic",
            "top12_reverse_semantic",
        }:
            continue
        if not row.get("generation_text"):
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
    paths = sorted(ROOT.glob("phase633_*_multi_writer_prefix_readout_field_closure_confirm.json"))
    if not paths:
        raise SystemExit(f"No confirm files found in {ROOT}")
    blocks = [
        "# Phase 633 Cross-Model Summary",
        "",
        "目标：对 Phase632 的自然 prefix writer 做去重后的多节点 cumulative restore，检查 prompt_last residual writer field 能否闭合 token0 prefix gate。",
        "",
    ]
    for path in paths:
        data = load(path)
        blocks.extend([
            f"## {data['model']}",
            "",
            f"- rows: {data['n_rows']} / raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']}",
            f"- candidate_nodes: {data['candidate_nodes']}",
            f"- set_defs: {data['set_defs']}",
            f"- downstream_layers: {data['downstream_layers']}",
            "",
            mode_table(data),
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
