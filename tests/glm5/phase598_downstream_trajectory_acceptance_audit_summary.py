#!/usr/bin/env python3
"""Summarize Phase 598 downstream trajectory acceptance audit results."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase598_downstream_trajectory_acceptance_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | {item['n']} | {item['switch']}/{item['n']} | "
        f"{fmt(item['mean_generated_down_projection'])} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_specific_margin_gain'])} | {fmt(item['mean_common_delta'])} |"
    )


def traj_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | H{item['hidden_index']} | {item['n']} | "
        f"{fmt(item['mean_projection_specific_margin'])} | "
        f"{fmt(item['mean_projection_correct_specific'])} | "
        f"{fmt(item['mean_projection_old_top_wrong_specific'])} | "
        f"{fmt(item['positive_projection_rate'])} |"
    )


def main() -> None:
    lines = [
        "# Phase598 Cross-Model Summary",
        "",
        "Downstream trajectory acceptance audit after MLP input state interpolation.",
        "",
    ]
    for model in MODELS:
        path = ROOT / f"phase598_{model}_downstream_trajectory_acceptance_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"alpha={data['alpha']}, window={data['window']}, time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        lines.append("### Final patch effects")
        lines.append("")
        lines.append("| key | kind | n | switch | generated_down_projection | margin_gain | specific_margin_gain | common_delta |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_patches"][:14]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Downstream hidden trajectory")
        lines.append("")
        lines.append("| key | kind | hidden | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best_trajectories"][:20]:
            lines.append(traj_line(item))
        lines.append("")
        if model == "deepseek7b":
            by_patch = data["summary"]["patch_by_key"]
            by_traj = data["summary"]["trajectory_by_key"]
            watched_patch = [
                "rule_value|L26|repair_alpha2",
                "rule_value|L26|random_alpha2",
                "rule_value|L26|wrong_alpha2",
                "prompt_last|L26|repair_alpha2",
                "prompt_last|L26|random_alpha2",
                "prompt_last|L26|wrong_alpha2",
                "query_relation|L19|repair_alpha2",
                "query_relation|L19|random_alpha2",
                "query_relation|L19|wrong_alpha2",
            ]
            lines.append("### DS7B watched final effects")
            lines.append("")
            lines.append("| key | kind | n | switch | generated_down_projection | margin_gain | specific_margin_gain | common_delta |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for key in watched_patch:
                if key in by_patch:
                    lines.append(patch_line(by_patch[key]))
            lines.append("")
            lines.append("### DS7B watched trajectories")
            lines.append("")
            lines.append("| key | kind | hidden | n | projection_specific_margin | correct_specific | old_wrong_specific | positive_rate |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
            for key in watched_patch:
                for hidx in range(20, 30):
                    tkey = f"{key}|H{hidx}"
                    if tkey in by_traj:
                        lines.append(traj_line(by_traj[tkey]))
            lines.append("")
    out = ROOT / "phase598_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
