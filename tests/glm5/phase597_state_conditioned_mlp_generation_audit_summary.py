#!/usr/bin/env python3
"""Summarize Phase 597 state-conditioned MLP generation audit results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase597_state_conditioned_mlp_generation_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | {fmt(item['alpha'])} | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_margin_gain'])} | {fmt(item['mean_specific_margin_gain'])} | "
        f"{fmt(item['mean_common_delta'])} | {fmt(item['mean_correct_specific'])} | "
        f"{fmt(item['mean_old_top_wrong_specific'])} |"
    )


def proj_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | {fmt(item['alpha'])} | {item['n']} | "
        f"{fmt(item['mean_projection_specific_margin'])} | "
        f"{fmt(item['mean_projection_correct_specific'])} | "
        f"{fmt(item['mean_projection_old_top_wrong_specific'])} | "
        f"{fmt(item['positive_projection_rate'])} |"
    )


def main() -> None:
    lines = [
        "# Phase597 Cross-Model Summary",
        "",
        "State-conditioned MLP input interpolation and recomputation audit.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase597_{model}_state_conditioned_mlp_generation_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"alphas={data['alphas']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best causal state patches")
        lines.append("")
        lines.append("| key | kind | alpha | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_patches"][:14]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Best generated projections")
        lines.append("")
        lines.append("| key | kind | alpha | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_projections"][:14]:
            lines.append(proj_line(item))
        lines.append("")
        if model == "deepseek7b":
            by_patch = data["summary"]["patch_by_key"]
            by_proj = data["summary"]["projection_by_key"]
            watched = []
            for alpha in data["alphas"]:
                a = f"{alpha:g}"
                for kind in ["repair", "wrong", "random"]:
                    watched.append(f"rule_value|L26|{kind}_alpha{a}")
                    watched.append(f"query_relation|L19|{kind}_alpha{a}")
            lines.append("### DS7B watched causal patches")
            lines.append("")
            lines.append("| key | kind | alpha | n | switch | margin_gain | specific_margin_gain | common_delta | correct_specific | old_wrong_specific |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for key in watched:
                if key in by_patch:
                    lines.append(patch_line(by_patch[key]))
            lines.append("")
            lines.append("### DS7B watched generated projections")
            lines.append("")
            lines.append("| key | kind | alpha | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for key in [w + "|generated_down" for w in watched]:
                if key in by_proj:
                    lines.append(proj_line(by_proj[key]))
            lines.append("")
    out = ROOT / "phase597_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
