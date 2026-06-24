#!/usr/bin/env python3
"""Summarize Phase 599 final layer washout decomposition results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase599_final_layer_washout_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def comp_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | `{item['component']}` | {item['n']} | "
        f"{fmt(item['mean_projection_specific_margin'])} | "
        f"{fmt(item['mean_projection_correct_specific'])} | "
        f"{fmt(item['mean_projection_old_top_wrong_specific'])} | "
        f"{fmt(item['positive_projection_rate'])} |"
    )


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_generated_down_projection'])} | "
        f"{fmt(item['mean_full_margin_gain'])} | "
        f"{fmt(item['mean_first_token_logit_margin_gain'])} |"
    )


def main() -> None:
    lines = [
        "# Phase599 Cross-Model Summary",
        "",
        "Final layer washout decomposition.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase599_{model}_final_layer_washout_decomposition_confirm.json"
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
        lines.append("### Final Effects")
        lines.append("")
        lines.append("| key | kind | n | switch | generated_down_projection | full_margin_gain | first_token_logit_margin_gain |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_patches"][:12]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Component Projections")
        lines.append("")
        lines.append("| key | kind | component | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_components"][:24]:
            lines.append(comp_line(item))
        lines.append("")
        if model == "deepseek7b":
            by_patch = data["summary"]["by_patch"]
            by_comp = data["summary"]["by_component"]
            watched = [
                "rule_value|L26|repair_alpha2",
                "rule_value|L26|random_alpha2",
                "rule_value|L26|wrong_alpha2",
                "prompt_last|L26|repair_alpha2",
                "prompt_last|L26|random_alpha2",
                "prompt_last|L26|wrong_alpha2",
            ]
            components = [
                "layer_input",
                "attn_out",
                "mlp_input",
                "mlp_out",
                "layer_out",
                "final_norm_input",
                "final_norm_output",
            ]
            lines.append("### DS7B watched final effects")
            lines.append("")
            lines.append("| key | kind | n | switch | generated_down_projection | full_margin_gain | first_token_logit_margin_gain |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            for key in watched:
                if key in by_patch:
                    lines.append(patch_line(by_patch[key]))
            lines.append("")
            lines.append("### DS7B watched component path")
            lines.append("")
            lines.append("| key | kind | component | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
            lines.append("|---|---|---|---:|---:|---:|---:|---:|")
            for key in watched:
                for comp in components:
                    ckey = f"{key}|{comp}"
                    if ckey in by_comp:
                        lines.append(comp_line(by_comp[ckey]))
            lines.append("")
    out = ROOT / "phase599_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
