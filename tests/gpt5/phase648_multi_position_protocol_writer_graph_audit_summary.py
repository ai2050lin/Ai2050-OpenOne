#!/usr/bin/env python3
"""Summarize Phase 648 multi-position protocol writer graph audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase648_multi_position_protocol_writer_graph_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]
POSITIONS = ["separator", "answer_label", "prompt_last", "question_mark_answer", "relation_tail"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def row_line(row) -> str:
    n = row["n"]
    return (
        f"| {row['mode']} | {n} | {row.get('position_unit') or ''} | {row.get('direction') or ''} | "
        f"{row.get('scope') or ''} | {row.get('interval') or ''} | "
        f"{row.get('layer') if row.get('layer') is not None else ''} | {row.get('component') or ''} | "
        f"{row['exact']}/{n} | {row['tok0_hit']}/{n} | {row['newline_top0']}/{n} | "
        f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
        f"{fmt_counts(row['top0_category'])} | {fmt_counts(row['generation_text'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 648 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase647 的 separator writer candidate graph 扩展到多个提示边界位置，"
        "检查 value_short_answer_protocol 是单边界现象还是多位置协议场。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase648_{model}_multi_position_protocol_writer_graph_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / "
            f"cases_written: {data['n_cases_written']} / mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- layers: `{data['layers']}` / positions: `{data['position_units']}` / target_only: {data['target_only']}")
        lines.append(f"- filtered: `{data['filtered']}` / total_time_min: {data.get('total_time_min', 0):.2f}\n")

        base = [r for r in data["summary"]["by_mode"] if r["kind"] == "baseline"]
        lines.append("### Baselines\n")
        lines.append("| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |")
        lines.append("|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|")
        for row in base:
            lines.append(row_line(row))
        lines.append("")

        lines.append("### Position Best Rows\n")
        lines.append("| position | best sufficiency | exact | newline | rank | best necessity/remove | exact | newline | rank |")
        lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
        for pos in POSITIONS:
            pos_data = data["summary"]["by_position"].get(pos, {})
            suff = (pos_data.get("best_sufficiency_restore") or [None])[0]
            nec = (pos_data.get("best_necessity_remove") or [None])[0]
            if suff is None:
                suff_cells = ["", "", "", ""]
            else:
                suff_cells = [
                    suff["mode"],
                    f"{suff['exact']}/{suff['n']}",
                    f"{suff['newline_top0']}/{suff['n']}",
                    f"{suff['mean_prefix_rank']:.1f}",
                ]
            if nec is None:
                nec_cells = ["", "", "", ""]
            else:
                nec_cells = [
                    nec["mode"],
                    f"{nec['exact']}/{nec['n']}",
                    f"{nec['newline_top0']}/{nec['n']}",
                    f"{nec['mean_prefix_rank']:.1f}",
                ]
            lines.append(f"| {pos} | {' | '.join(suff_cells)} | {' | '.join(nec_cells)} |")
        lines.append("")

        for pos in POSITIONS:
            lines.append(f"### {pos}\n")
            pos_data = data["summary"]["by_position"].get(pos, {})
            for title, group in [
                ("Best Sufficiency Restore", pos_data.get("best_sufficiency_restore", [])[:12]),
                ("Best Necessity Remove", pos_data.get("best_necessity_remove", [])[:12]),
            ]:
                lines.append(f"#### {title}\n")
                if not group:
                    lines.append("No rows.\n")
                    continue
                lines.append("| mode | n | position | direction | scope | interval | layer | component | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |")
                lines.append("|---|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|")
                for row in group:
                    lines.append(row_line(row))
                lines.append("")

        lines.append("### Global Top Notes\n")
        suff = data["summary"]["best_sufficiency_restore"][:8]
        nec = data["summary"]["best_necessity_remove"][:8]
        lines.append("- Top sufficiency: " + "; ".join(
            f"{r['mode']} exact={r['exact']}/{r['n']} newline={r['newline_top0']}/{r['n']}" for r in suff
        ))
        lines.append("- Top necessity/remove: " + "; ".join(
            f"{r['mode']} exact={r['exact']}/{r['n']} newline={r['newline_top0']}/{r['n']}" for r in nec
        ))
        lines.append("")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "phase648_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
