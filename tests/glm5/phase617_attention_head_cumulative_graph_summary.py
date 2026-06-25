#!/usr/bin/env python3
"""Summarize Phase 617 attention head cumulative graph."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase617_attention_head_cumulative_graph")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 617 Cross Model Summary",
        "",
        "Layer/head-slot decomposition of the multi-layer attention cumulative path.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase617_{model}_attention_head_cumulative_graph_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append("missing")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"rows={data['n_rows']}, target_seen={data['n_target_cases_seen']}, "
            f"raw={data['n_raw_cases']}, filtered={data.get('filtered')}, "
            f"layers={data['layers_to_scan']}, heads={data['heads_by_layer']}, "
            f"specs={data['n_specs']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### best")
        lines.append("")
        lines.append("| rank | name | kind | random | ops | slots | switch | margin | correct_delta | wrong_delta |")
        lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(data["summary"]["best"][:32], 1):
            lines.append(
                f"| {rank} | `{item['name']}` | {item['kind']} | {item['random']} | "
                f"{item['n_ops']} | {item['n_slots']} | {item['switch']}/{item['n']} | "
                f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} |"
            )
        lines.append("")

        by = list(data["summary"]["by_patch"].values())
        for title, predicate in [
            ("all_heads_refs", lambda x: (not x["random"]) and x["kind"].startswith("all_heads")),
            ("known_top_cumulative", lambda x: (not x["random"]) and x["kind"].startswith("known_top")),
            ("single_heads", lambda x: (not x["random"]) and x["kind"].startswith("single")),
            ("random_controls", lambda x: x["random"] and (x["kind"].startswith("all_heads") or x["kind"].startswith("known_top"))),
        ]:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| name | kind | ops | slots | switch | margin | correct_delta | wrong_delta |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            items = [item for item in by if predicate(item)]
            items = sorted(items, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
            for item in items[:36]:
                lines.append(
                    f"| `{item['name']}` | {item['kind']} | {item['n_ops']} | {item['n_slots']} | "
                    f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
                    f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} |"
                )
            lines.append("")
    out = ROOT / "phase617_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
