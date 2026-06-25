#!/usr/bin/env python3
"""Summarize Phase 618 attention source pattern/content decomposition."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase618_attention_source_pattern_content")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x: float) -> str:
    return f"{x:+.3f}"


def main() -> None:
    lines = [
        "# Phase 618 Cross Model Summary",
        "",
        "Source group and pattern/content decomposition for top attention head paths.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase618_{model}_attention_source_pattern_content_confirm.json"
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
        for rank, item in enumerate(data["summary"]["best"][:36], 1):
            lines.append(
                f"| {rank} | `{item['name']}` | {item['group']} | {item['mode']} | {item['random']} | "
                f"{item['n_ops']} | {item['n_slots']} | {item['switch']}/{item['n']} | "
                f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | "
                f"{fmt(item['mean_wrong_delta'])} |"
            )
        lines.append("")

        by = list(data["summary"]["by_patch"].values())
        for title, predicate in [
            ("top_path_real", lambda x: (not x["random"]) and x["name"].startswith("top")),
            ("single_head_rr_real", lambda x: (not x["random"]) and x["name"].startswith("L") and x["mode"] == "rr_pattern_content"),
            ("pattern_vs_content_real", lambda x: (not x["random"]) and x["group"] in ["all_source", "question_line", "final_object_category_line", "value_rule_lines"]),
            ("random_controls", lambda x: x["random"] and x["name"].startswith("top")),
        ]:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("| name | group | mode | ops | slots | switch | margin | correct_delta | wrong_delta |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
            items = [item for item in by if predicate(item)]
            items = sorted(items, key=lambda x: (x["switch"], x["mean_margin_gain"]), reverse=True)
            for item in items[:48]:
                lines.append(
                    f"| `{item['name']}` | {item['group']} | {item['mode']} | "
                    f"{item['n_ops']} | {item['n_slots']} | {item['switch']}/{item['n']} | "
                    f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_correct_delta'])} | "
                    f"{fmt(item['mean_wrong_delta'])} |"
                )
            lines.append("")
    out = ROOT / "phase618_cross_model_summary.md"
    ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
