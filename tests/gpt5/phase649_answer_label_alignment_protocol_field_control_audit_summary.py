#!/usr/bin/env python3
"""Summarize Phase 649 answer-label alignment and protocol field controls."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase649_answer_label_alignment_protocol_field_control_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
POSITIONS = [
    "answer_word",
    "colon",
    "answer_colon",
    "answer_label_aligned",
    "separator",
    "prompt_last",
    "question_mark_answer",
    "relation_tail",
]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def short(row) -> str:
    if not row:
        return ""
    return f"{row['mode']} exact={row['exact']}/{row['n']} newline={row['newline_top0']}/{row['n']} rank={row['mean_prefix_rank']:.1f}"


def row_line(row) -> str:
    n = row["n"]
    return (
        f"| {row['mode']} | {n} | {row.get('position_unit') or ''} | {row.get('direction') or ''} | "
        f"{row.get('scope') or ''} | {row.get('interval') or ''} | "
        f"{row.get('layer') if row.get('layer') is not None else ''} | {row.get('component') or ''} | "
        f"{row.get('control') or ''} | {row['exact']}/{n} | {row['tok0_hit']}/{n} | "
        f"{row['newline_top0']}/{n} | {row['mean_prefix_rank']:.1f} | "
        f"{row['mean_prefix_minus_newline']:.3f} | {fmt_counts(row['top0_category'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 649 Cross-Model Summary\n")
    lines.append(
        "目标：修复 Phase648 的 answer_label 对齐缺口，并对最强 protocol field 候选加入 "
        "restore/random/reverse controls。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase649_{model}_answer_label_alignment_protocol_field_control_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / "
            f"cases_written: {data['n_cases_written']} / mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- positions: `{data['position_units']}`")
        lines.append(f"- filtered: `{data['filtered']}` / total_time_min: {data.get('total_time_min', 0):.2f}\n")

        base = [r for r in data["summary"]["by_mode"] if r["kind"] == "baseline"]
        lines.append("### Baselines\n")
        lines.append("| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |")
        lines.append("|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|")
        for row in base:
            lines.append(row_line(row))
        lines.append("")

        lines.append("### Position Control Overview\n")
        lines.append("| position | best restore sufficiency | best restore necessity/remove | best random | best reverse |")
        lines.append("|---|---|---|---|---|")
        for pos in POSITIONS:
            pd = data["summary"]["by_position"].get(pos, {})
            suff = (pd.get("best_sufficiency_restore") or [None])[0]
            nec = (pd.get("best_necessity_remove") or [None])[0]
            rnd = (pd.get("best_random_controls") or [None])[0]
            rev = (pd.get("best_reverse_controls") or [None])[0]
            lines.append(f"| {pos} | {short(suff)} | {short(nec)} | {short(rnd)} | {short(rev)} |")
        lines.append("")

        for pos in POSITIONS:
            pd = data["summary"]["by_position"].get(pos, {})
            lines.append(f"### {pos}\n")
            for title, group in [
                ("Best Sufficiency Restore", pd.get("best_sufficiency_restore", [])[:8]),
                ("Best Necessity Remove", pd.get("best_necessity_remove", [])[:8]),
                ("Best Random Controls", pd.get("best_random_controls", [])[:8]),
                ("Best Reverse Controls", pd.get("best_reverse_controls", [])[:8]),
            ]:
                lines.append(f"#### {title}\n")
                if not group:
                    lines.append("No rows.\n")
                    continue
                lines.append("| mode | n | position | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category |")
                lines.append("|---|---:|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|")
                for row in group:
                    lines.append(row_line(row))
                lines.append("")

        lines.append("### Global Top Notes\n")
        suff = data["summary"]["best_sufficiency_restore"][:10]
        nec = data["summary"]["best_necessity_remove"][:10]
        lines.append("- Top sufficiency: " + "; ".join(short(r) for r in suff))
        lines.append("- Top necessity/remove: " + "; ".join(short(r) for r in nec))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase649_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
