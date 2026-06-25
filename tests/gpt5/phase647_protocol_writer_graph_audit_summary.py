#!/usr/bin/env python3
"""Summarize Phase 647 protocol writer graph audit."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase647_protocol_writer_graph_audit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d) -> str:
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def row_line(row) -> str:
    n = row["n"]
    return (
        f"| {row['mode']} | {n} | {row.get('direction') or ''} | {row.get('scope') or ''} | "
        f"{row.get('interval') or ''} | {row.get('layer') if row.get('layer') is not None else ''} | "
        f"{row.get('component') or ''} | {row.get('control') or ''} | "
        f"{row['exact']}/{n} | {row['tok0_hit']}/{n} | {row['newline_top0']}/{n} | "
        f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
        f"{fmt_counts(row['top0_category'])} | {fmt_counts(row['generation_text'])} |"
    )


def main() -> None:
    lines = []
    lines.append("# Phase 647 Cross-Model Summary\n")
    lines.append(
        "目标：把 Phase646 atlas 中的 value_short_answer_protocol 从 layer_out trajectory "
        "继续拆成 attention / MLP / residual carry writer graph。\n"
    )

    for model in MODELS:
        path = OUT_ROOT / f"phase647_{model}_protocol_writer_graph_audit_confirm.json"
        data = load(path)
        lines.append(f"## {model}\n")
        if data is None:
            lines.append(f"Missing: `{path}`\n")
            continue
        lines.append(
            f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / "
            f"cases_written: {data['n_cases_written']} / mode_rows: {data['n_mode_rows']}"
        )
        lines.append(f"- layers: `{data['layers']}` / components: `{data['components']}` / target_only: {data['target_only']}")
        lines.append(f"- filtered: `{data['filtered']}` / total_time_min: {data.get('total_time_min', 0):.2f}\n")

        rows = data["summary"]["by_mode"]
        base = [r for r in rows if r["kind"] == "baseline"]
        interval = [r for r in rows if r.get("scope") == "interval" and r.get("control") == "restore"]
        single = [r for r in rows if r.get("scope") == "single_layer" and r.get("control") == "restore"]
        controls = [r for r in rows if r.get("control") in {"random", "reverse"}]

        for title, group in [
            ("Baselines", base),
            ("Interval Restore", interval),
            ("Best Sufficiency Single-Layer Restore", data["summary"]["best_sufficiency_restore"][:16]),
            ("Best Necessity Single-Layer Remove", data["summary"]["best_necessity_remove"][:16]),
            ("Control Samples", controls[:20]),
        ]:
            lines.append(f"### {title}\n")
            if not group:
                lines.append("No rows.\n")
                continue
            lines.append("| mode | n | direction | scope | interval | layer | component | control | exact | tok0 | newline | rank | prefix-newline | top0_category | generation_text |")
            lines.append("|---|---:|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|")
            for row in group:
                lines.append(row_line(row))
            lines.append("")

        lines.append("### Writer Notes\n")
        suff = data["summary"]["best_sufficiency_restore"][:5]
        nec = data["summary"]["best_necessity_remove"][:5]
        lines.append("- Top sufficiency: " + "; ".join(
            f"{r['mode']} exact={r['exact']}/{r['n']} newline={r['newline_top0']}/{r['n']}" for r in suff
        ))
        lines.append("- Top necessity/remove: " + "; ".join(
            f"{r['mode']} exact={r['exact']}/{r['n']} newline={r['newline_top0']}/{r['n']}" for r in nec
        ))
        lines.append("")

    out = OUT_ROOT / "phase647_cross_model_summary.md"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
