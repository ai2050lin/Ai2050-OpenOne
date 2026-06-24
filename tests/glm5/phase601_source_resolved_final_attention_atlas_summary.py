#!/usr/bin/env python3
"""Summarize Phase 601 source-resolved final attention atlas."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase601_source_resolved_final_attention_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]
GROUPS = [
    "rule_relation",
    "rule_value",
    "object",
    "category_first",
    "query_relation",
    "query_category",
    "prompt_last",
    "punct_newline",
    "other",
]


def fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def delta_line(item: dict) -> str:
    best = max(GROUPS, key=lambda g: abs(item.get(f"delta_{g}", 0.0)))
    vals = " | ".join(fmt(item.get(f"delta_{g}", 0.0)) for g in GROUPS)
    return f"| `{item['key']}` | {item['trajectory']} | {item['n']} | `{best}` | {vals} | {fmt(item['entropy'])} | {fmt(item['top_mass'])} |"


def contrast_line(item: dict) -> str:
    best = max(GROUPS, key=lambda g: abs(item.get(f"nat_minus_art_{g}", 0.0)))
    vals = " | ".join(fmt(item.get(f"nat_minus_art_{g}", 0.0)) for g in GROUPS)
    return f"| `{item['key']}` | {item['n']} | `{best}` | {fmt(item['l1_nat_minus_artificial'])} | {vals} |"


def main() -> None:
    header_groups = " | ".join(GROUPS)
    align = "|---|---|---:|---|" + "|---:" * len(GROUPS) + "|---:|---:|"
    contrast_align = "|---|---:|---|---:|" + "|---:" * len(GROUPS) + "|"
    lines = [
        "# Phase601 Cross-Model Summary",
        "",
        "Source-resolved final attention atlas.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase601_{model}_source_resolved_final_attention_atlas_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"probe_layer={data['probe_layer']}, alpha={data['alpha']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Largest Attention Deltas")
        lines.append("")
        lines.append(f"| key | trajectory | n | max_group | {header_groups} | entropy | top_mass |")
        lines.append(align)
        for item in data["summary"]["best_deltas"][:24]:
            lines.append(delta_line(item))
        lines.append("")
        lines.append("### Natural Correct Minus Artificial Repair")
        lines.append("")
        lines.append(f"| key | n | max_group | l1 | {header_groups} |")
        lines.append(contrast_align)
        for item in data["summary"]["best_contrast"][:24]:
            lines.append(contrast_line(item))
        lines.append("")
        if model == "deepseek7b":
            by_key = data["summary"]["by_key"]
            watched = [
                "rule_value|L26",
                "prompt_last|L26",
                "query_relation|L19",
            ]
            trajectories = [
                "natural_correct",
                "natural_wrong",
                "artificial_repair",
                "artificial_random",
                "artificial_wrong",
            ]
            lines.append("### DS7B watched source deltas")
            lines.append("")
            lines.append(f"| key | trajectory | n | max_group | {header_groups} | entropy | top_mass |")
            lines.append(align)
            for base in watched:
                for traj in trajectories:
                    key = f"{base}|{traj}"
                    if key in by_key:
                        lines.append(delta_line(by_key[key]))
            lines.append("")
    out = ROOT / "phase601_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
