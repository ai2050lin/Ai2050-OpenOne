#!/usr/bin/env python3
"""Summarize Phase 609 query / o_proj input / head-slot decomposition."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase609_query_oproj_head_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def row(item: dict) -> str:
    head = "" if item.get("head") is None else f"H{item['head']}"
    return (
        f"| `{item['key']}` | L{item['layer']} | {item['mode']} | {head} | "
        f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
    )


def main() -> None:
    lines = ["# Phase609 Cross-Model Summary", "", "Query / o_proj-input / head-slot decomposition.", ""]
    for model in MODELS:
        path = ROOT / f"phase609_{model}_query_oproj_head_decomposition_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"layers={data['layers_to_scan']}, heads={data['heads_by_layer']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Patches")
        lines.append("")
        lines.append("| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:48]:
            lines.append(row(item))
        lines.append("")

        by_patch = data["summary"]["by_patch"]
        lines.append("### Core Modes")
        lines.append("")
        lines.append("| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        for li in data["layers_to_scan"]:
            for mode in ["q_delta", "q_random", "o_input_delta", "o_input_random"]:
                item = by_patch.get(f"L{li}|{mode}")
                if item:
                    lines.append(row(item))
        lines.append("")

        lines.append("### Head Delta Ranking")
        lines.append("")
        lines.append("| key | layer | mode | head | switch | margin_gain | correct_delta | old_wrong_delta |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|")
        heads = [
            item for item in by_patch.values()
            if item["mode"] == "head_delta"
        ]
        heads = sorted(heads, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
        for item in heads[:40]:
            lines.append(row(item))
        lines.append("")

    out = ROOT / "phase609_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
