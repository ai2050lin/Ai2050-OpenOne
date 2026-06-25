#!/usr/bin/env python3
"""Summarize Phase 637 newline prior suppression source audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase637_newline_prior_suppression_source_audit")
SHOW_VARIANTS = [
    "original",
    "no_qmark",
    "period",
    "inline_answer",
    "short_only",
    "no_explain",
    "no_qmark_short",
    "value_label",
    "direct_value_label",
]


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d):
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def rows_for(data, subject_kind, split):
    rows = [
        r for r in data["summary"]["by_mode_split"]
        if r["subject_kind"] == subject_kind and r["split"] == split
    ]
    by_variant = {r["variant"]: r for r in rows}
    return [by_variant[v] for v in SHOW_VARIANTS if v in by_variant]


def main() -> None:
    lines = []
    lines.append("# Phase 637 Cross-Model Summary\n")
    lines.append("目标：测试 prompt ablation 是否能压制 newline / format continuation prior，并记录 non-target 副作用。\n")
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = OUT_ROOT / f"phase637_{model}_newline_prior_suppression_source_audit_confirm.json"
        data = load_model(path)
        if data is None:
            lines.append(f"## {model}\n\nMissing: `{path}`\n")
            continue
        lines.append(f"## {model}\n")
        lines.append(f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / rows: {data['n_rows']}")
        lines.append(f"- top_k: {data['top_k']}")
        for subject_kind in ["base_subject", "repair_subject"]:
            for split in ["target", "non_target"]:
                rows = rows_for(data, subject_kind, split)
                if not rows:
                    continue
                lines.append(f"\n### {subject_kind} / {split}\n")
                lines.append("| variant | n | tok0 | exact | wrong_exact | newline_top0 | rank | prefix-newline | top0_category | top0_text |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
                for row in rows:
                    n = row["n"]
                    lines.append(
                        f"| {row['variant']} | {n} | {row['tok0_hit']}/{n} | {row['exact']}/{n} | "
                        f"{row['wrong_exact']}/{n} | {row['newline_top0']}/{n} | "
                        f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
                        f"{fmt_counts(row['top0_category'])} | {fmt_counts(row['top0_text'])} |"
                    )
        lines.append("\n### Examples\n")
        for item in data["examples"][:14]:
            tops = ", ".join(f"{x['rank']}:{x['text_clean']}[{x['category']}]" for x in item["top"][:5])
            lines.append(
                f"- sample={item['sample_idx']} split={item['split']} mode={item['mode']} "
                f"top0={item['top0_text_clean']!r}/{item['top0_category']} rank={item['prefix_rank']} "
                f"exact={item['eval']['exact_correct']} text={item['generation_text']!r} ladder={tops}"
            )
        lines.append("")
    out = OUT_ROOT / "phase637_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
