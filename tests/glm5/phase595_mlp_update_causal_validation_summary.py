#!/usr/bin/env python3
"""Summarize Phase 595 cross-model MLP update causal validation results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase595_mlp_update_causal_validation")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(model: str):
    path = ROOT / f"phase595_{model}_mlp_update_causal_validation_confirm.json"
    if not path.exists():
        return None, path
    return json.loads(path.read_text(encoding="utf-8")), path


def fmt(x) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def row_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_specific_margin_gain'])} | "
        f"{fmt(item['mean_common_delta'])} | {fmt(item['mean_correct_specific'])} | "
        f"{fmt(item['mean_old_top_wrong_specific'])} |"
    )


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase595 Cross-Model Summary",
        "",
        "MLP update component causal patch validation.",
        "",
    ]
    for model in MODELS:
        data, path = load(model)
        lines.append(f"## {model}")
        if data is None:
            lines.append("")
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        lines.append("")
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"alpha={data['alpha']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:12]:
            lines.append(row_line(item))
        lines.append("")

        if model == "deepseek7b":
            lines.append("### DS7B watched nodes")
            lines.append("")
            watched = [
                "rule_value|L26|mlp|raw",
                "rule_value|L26|mlp|specific_only",
                "rule_value|L26|mlp|common_only",
                "rule_value|L26|mlp|specific_norm_raw",
                "rule_value|L26|mlp|common_norm_raw",
                "rule_value|L26|mlp|random_same_norm",
                "query_relation|L19|mlp|raw",
                "query_relation|L19|mlp|specific_only",
                "query_relation|L19|mlp|common_only",
                "prompt_last|L26|mlp|raw",
                "prompt_last|L26|mlp|specific_only",
            ]
            by_key = data["summary"]["by_key"]
            lines.append("| key | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
            for key in watched:
                item = by_key.get(key)
                if item:
                    lines.append(row_line(item))
            lines.append("")
    out = ROOT / "phase595_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
