#!/usr/bin/env python3
"""Summarize Phase 605 multi-token trajectory builder audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("results/glm5_phase605_multi_token_trajectory_builder_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
WATCH = [
    "input_prefix0",
    "input_digit1",
    "input_digit2",
    "input_digits",
    "input_all",
    "output_prefix0",
    "output_digit1",
    "output_digit2",
    "output_digits",
    "output_all",
    "input_digits_random",
    "output_digits_random",
]


def fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def patch_line(item: dict) -> str:
    return (
        f"| `{item['key']}` | {item['kind']} | {item['group']} | {item['random']} | {item['n']} | "
        f"{item['switch']}/{item['n']} | {fmt(item['mean_margin_gain'])} | "
        f"{fmt(item['mean_correct_delta'])} | {fmt(item['mean_wrong_delta'])} | "
        f"{fmt(item.get('mean_token0_correct_delta', 0.0))} | "
        f"{fmt(item.get('mean_token1_correct_delta', 0.0))} | "
        f"{fmt(item.get('mean_token2_correct_delta', 0.0))} | "
        f"{fmt(item.get('mean_token0_wrong_delta', 0.0))} | "
        f"{fmt(item.get('mean_token1_wrong_delta', 0.0))} | "
        f"{fmt(item.get('mean_token2_wrong_delta', 0.0))} |"
    )


def main() -> None:
    lines = ["# Phase605 Cross-Model Summary", "", "Multi-token trajectory builder audit.", ""]
    for model in MODELS:
        path = ROOT / f"phase605_{model}_multi_token_trajectory_builder_audit_confirm.json"
        lines.append(f"## {model}")
        lines.append("")
        if not path.exists():
            lines.append(f"Missing: `{path}`")
            lines.append("")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"cases={data['n_cases']}, rows={data['n_rows']}, target_cases_seen={data['n_target_cases_seen']}, "
            f"time_min={data.get('total_time_min', 0):.2f}"
        )
        lines.append("")
        if data["rows"]:
            lines.append("Tokenization example:")
            tok = data["rows"][0].get("tokenization", {})
            lines.append("")
            lines.append("```text")
            for ans, toks in tok.items():
                lines.append(f"{ans}: {toks}")
            lines.append("```")
            lines.append("")
        lines.append("### Best Patch Modes")
        lines.append("")
        lines.append("| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in data["summary"]["best"][:24]:
            lines.append(patch_line(item))
        lines.append("")
        lines.append("### Watched Patch Modes")
        lines.append("")
        lines.append("| key | kind | group | random | n | switch | margin_gain | correct_delta | old_wrong_delta | c_tok0 | c_tok1 | c_tok2 | w_tok0 | w_tok1 | w_tok2 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        by_patch = data["summary"]["by_patch"]
        for key in WATCH:
            if key in by_patch:
                lines.append(patch_line(by_patch[key]))
        lines.append("")
    out = ROOT / "phase605_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
