#!/usr/bin/env python3
"""Summarize Phase 639 protocol tail minimal causal unit audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase639_protocol_tail_minimal_causal_unit_audit")
MODE_ORDER = [
    "original",
    "inline",
    "final_output_inline_to_original",
    "patch_qmark",
    "patch_separator",
    "patch_answer_word",
    "patch_colon",
    "patch_prompt_last",
    "patch_qmark_separator",
    "patch_separator_answer",
    "patch_answer_colon",
    "patch_tail_all",
]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d):
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def rows_for(data, split):
    rows = [r for r in data["summary"]["by_mode_split"] if r["split"] == split]
    by_mode = {r["mode"]: r for r in rows}
    return [by_mode[m] for m in MODE_ORDER if m in by_mode]


def main() -> None:
    lines = []
    lines.append("# Phase 639 Cross-Model Summary\n")
    lines.append("目标：把 Phase638 的 question-tail protocol state 缩小到最小 token / token-pair 因果单位。\n")
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = OUT_ROOT / f"phase639_{model}_protocol_tail_minimal_causal_unit_audit_confirm.json"
        data = load_model(path)
        if data is None:
            lines.append(f"## {model}\n\nMissing: `{path}`\n")
            continue
        lines.append(f"## {model}\n")
        lines.append(f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / cases_written: {data['n_rows']} / mode_rows: {data['n_mode_rows']}")
        lines.append(f"- top_k: {data['top_k']}")
        lines.append(f"- filtered: `{data['filtered']}`")
        lines.append(f"- unit_token_lens_sample: `{data['unit_token_lens_sample']}`")
        for split in ["target", "non_target"]:
            rows = rows_for(data, split)
            if not rows:
                continue
            lines.append(f"\n### {split}\n")
            lines.append("| mode | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | margin-top | top0_category | top0_text |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
            for row in rows:
                n = row["n"]
                lines.append(
                    f"| {row['mode']} | {n} | {row['tok0_hit']}/{n} | {row['exact']}/{n} | "
                    f"{row['wrong_exact']}/{n} | {row['newline_top0']}/{n} | "
                    f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
                    f"{row['mean_prefix_margin_vs_top']:.3f} | {fmt_counts(row['top0_category'])} | {fmt_counts(row['top0_text'])} |"
                )
        lines.append("\n### Examples\n")
        for item in data["examples"][:16]:
            tops = ", ".join(f"{x['rank']}:{x['text_clean']}[{x['category']}]" for x in item["top"][:5])
            lines.append(
                f"- sample={item['sample_idx']} split={item['split']} mode={item['mode']} "
                f"top0={item['top0_text_clean']!r}/{item['top0_category']} rank={item['prefix_rank']} "
                f"exact={item['eval']['exact_correct']} text={item['generation_text']!r} ladder={tops}"
            )
        lines.append("")
    out = OUT_ROOT / "phase639_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
