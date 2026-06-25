#!/usr/bin/env python3
"""
Summarize Phase 632 natural prefix readout writer backtrace results.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase632_natural_prefix_readout_writer_backtrace")
OUT = ROOT / "phase632_cross_model_summary.md"


def fmt(x: float | int | None, digits: int = 3) -> str:
    if x is None:
        return "NA"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_table(data: dict, n: int = 12) -> str:
    lines = [
        "| rank | node | mean_margin_delta | positive_rate | mean_cos | mean_delta_norm | score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, item in enumerate(data["scan_rank"][:n], 1):
        lines.append(
            f"| {i} | {item['node']} | {fmt(item['mean_margin_delta'])} | "
            f"{fmt(item['positive_rate'])} | {fmt(item['mean_cos'])} | "
            f"{fmt(item['mean_delta_norm'])} | {fmt(item['score'])} |"
        )
    return "\n".join(lines)


def causal_table(data: dict, n: int = 24) -> str:
    rows = data["causal_summary"]["by_node_mode"]
    baseline = [r for r in rows if r["node"] == "__baseline__"]
    non_base = [r for r in rows if r["node"] != "__baseline__"]
    ordered = baseline + non_base[:n]
    lines = [
        "| node | mode | tok0 | exact | wrong_exact | mean_prefix_margin |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in ordered:
        lines.append(
            f"| {item['node']} | {item['mode']} | {item['tok0_hit']}/{item['n']} | "
            f"{item['exact']}/{item['n']} | {item['wrong_exact']}/{item['n']} | "
            f"{fmt(item['mean_prefix_margin'])} |"
        )
    return "\n".join(lines)


def examples(data: dict, n: int = 5) -> str:
    out = []
    for row in data.get("causal_rows", [])[:200]:
        if row["node"] == "__baseline__" and row["mode"] not in {"base", "semantic_cumulative"}:
            continue
        if row["node"] != "__baseline__" and row["mode"] not in {"restore_semantic", "random_semantic", "reverse_semantic"}:
            continue
        if not row.get("generation_text"):
            continue
        out.append(
            f"- sample={row['sample_idx']} node={row['node']} mode={row['mode']} "
            f"tok0={row['tok0_text']!r} exact={row['eval']['exact_correct']} "
            f"wrong={row['eval']['exact_wrong']} margin={fmt(row['prefix_margin'])} "
            f"text={row['generation_text']!r}"
        )
        if len(out) >= n:
            break
    return "\n".join(out)


def main() -> None:
    paths = sorted(ROOT.glob("phase632_*_natural_prefix_readout_writer_backtrace_confirm.json"))
    if not paths:
        raise SystemExit(f"No confirm files found in {ROOT}")
    blocks = [
        "# Phase 632 Cross-Model Summary",
        "",
        "目标：从 Phase631 的人工 readout direction 回溯自然写入器，审计每层/组件对 prefix margin 的自然差分贡献，并对 top writer 做 causal restore/remove/control。",
        "",
    ]
    for path in paths:
        data = load(path)
        blocks.extend([
            f"## {data['model']}",
            "",
            f"- rows: {data['n_rows']} / raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']}",
            f"- scan_layers: {data['scan_layers']}",
            f"- downstream_layers: {data['downstream_layers']}",
            f"- top_nodes: {data['top_nodes']}",
            "",
            "### Natural Margin Writer Scan",
            "",
            scan_table(data),
            "",
            "### Causal Patch Audit",
            "",
            causal_table(data),
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
