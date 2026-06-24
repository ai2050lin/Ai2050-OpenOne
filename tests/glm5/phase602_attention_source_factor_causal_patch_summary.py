#!/usr/bin/env python3
"""Summarize Phase 602 attention-source factor causal patch."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase602_attention_source_factor_causal_patch")
MODELS = ["qwen3", "glm4", "deepseek7b"]
WATCHED = [
    "rule_value|L26",
    "prompt_last|L26",
    "query_relation|L19",
]
MODES = [
    "mlp_repair_only",
    "attn_effect_only",
    "mlp_plus_attn_effect",
    "attn_random",
    "mlp_plus_attn_random",
    "mlp_random_plus_attn_effect",
]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['mode']} | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_full_margin_gain'])} | {fmt(item['mean_generated_down_projection'])} | "
        f"{fmt(item['mean_attn_delta_projection'])} | {fmt(item['mean_final_norm_projection'])} | "
        f"{fmt(item['mean_final_norm_cos_to_natural'])} | {fmt(item['positive_full_margin_rate'])} |"
    )


def main() -> None:
    lines = [
        "# Phase602 Cross-Model Summary",
        "",
        "Attention-source factor causal patch.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase602_{model}_attention_source_factor_causal_patch_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"probe_layer={data['probe_layer']}, alpha={data['alpha']}, attn_scale={data['attn_scale']}, "
            f"time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Best Effects")
        lines.append("")
        lines.append("| key | mode | n | switch | full_margin_gain | mlp_down_projection | attn_delta_projection | final_norm_projection | final_norm_cos_to_natural | positive_full_margin_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:30]:
            lines.append(line(item))
        lines.append("")
        if model == "deepseek7b":
            by = data["summary"]["by_patch"]
            lines.append("### DS7B watched combinations")
            lines.append("")
            lines.append("| key | mode | n | switch | full_margin_gain | mlp_down_projection | attn_delta_projection | final_norm_projection | final_norm_cos_to_natural | positive_full_margin_rate |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for base in WATCHED:
                for mode in MODES:
                    key = f"{base}|{mode}"
                    if key in by:
                        lines.append(line(by[key]))
            lines.append("")
    out = ROOT / "phase602_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
