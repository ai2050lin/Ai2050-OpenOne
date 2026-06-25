#!/usr/bin/env python3
"""Summarize Phase 619 rule-line token micro atlas."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase619_rule_line_token_micro_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 619 Cross Model Summary",
        "",
        "Rule-line token micro-atlas for source-localized pattern/content repair.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase619_{model}_rule_line_token_micro_atlas_confirm.json"
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
            f"top_heads={data['top_heads']}, specs={data['n_specs']}, "
            f"compact={data.get('compact')}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### best")
        lines.append("")
        lines.append("| rank | name | group | mode | random | ops | slots | switch | margin | correct_delta | wrong_delta |")
        lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|")
        for rank, item in enumerate(data["summary"]["best"][:40], 1):
            lines.append(
                f"| {rank} | `{item['name']}` | {item['group']} | {item['mode']} | {item['random']} | "
                f"{item['n_ops']} | {item['n_slots']} | {item['switch']}/{item['n']} | "
                f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} |"
            )
        lines.append("")

        by = list(data["summary"]["by_patch"].values())
        sections = [
            ("micro_real_top_paths", lambda x: (not x["random"]) and x["name"].startswith("top")),
            ("correct_line_vs_parts_real", lambda x: (not x["random"]) and x["group"].startswith("correct_")),
            ("wrong_line_controls_real", lambda x: (not x["random"]) and x["group"].startswith("wrong_")),
            ("single_head_real", lambda x: (not x["random"]) and x["name"].startswith("L")),
            ("random_controls", lambda x: x["random"] and x["name"].startswith("top")),
        ]
        for title, predicate in sections:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
            items = [item for item in by if predicate(item)]
            items = sorted(items, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
            for item in items[:64]:
                lines.append(
                    f"| `{item['name']}` | {item['group']} | {item['mode']} | "
                    f"{item['n_ops']} | {item['n_slots']} | {item['switch']}/{item['n']} | "
                    f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | "
                    f"{fmt(item['mean_wrong_delta'])} |"
                )
            lines.append("")
    out = ROOT / "phase619_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
