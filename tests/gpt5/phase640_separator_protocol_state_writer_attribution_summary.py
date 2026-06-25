#!/usr/bin/env python3
"""Summarize Phase 640 separator protocol state writer attribution."""
from __future__ import annotations

import json
from pathlib import Path


OUT_ROOT = Path("results/glm5_phase640_separator_protocol_state_writer_attribution")


def load_model(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_counts(d):
    return ", ".join(f"{k}:{v}" for k, v in d.items())


def main() -> None:
    lines = []
    lines.append("# Phase 640 Cross-Model Summary\n")
    lines.append("目标：扫描 separator boundary 的 layer/component writer，定位 inline protocol state 的写入候选。\n")
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = OUT_ROOT / f"phase640_{model}_separator_protocol_state_writer_attribution_confirm.json"
        data = load_model(path)
        if data is None:
            lines.append(f"## {model}\n\nMissing: `{path}`\n")
            continue
        lines.append(f"## {model}\n")
        lines.append(f"- raw_cases: {data['n_raw_cases']} / target_seen: {data['n_target_cases_seen']} / cases_written: {data['n_cases_written']} / mode_rows: {data['n_mode_rows']}")
        lines.append(f"- target_only: {data['target_only']} / top_k: {data['top_k']}")
        lines.append(f"- scan_layers: `{data['scan_layers']}`")
        lines.append(f"- control_layers: `{data['control_layers']}`")
        lines.append(f"- filtered: `{data['filtered']}`")

        base_rows = [r for r in data["summary"]["by_mode"] if r["kind"] == "baseline"]
        lines.append("\n### Baselines\n")
        lines.append("| mode | n | tok0 | newline_top0 | rank | prefix-newline | top0_category | top0_text |")
        lines.append("|---|---:|---:|---:|---:|---:|---|---|")
        for row in base_rows:
            n = row["n"]
            lines.append(
                f"| {row['mode']} | {n} | {row['tok0_hit']}/{n} | {row['newline_top0']}/{n} | "
                f"{row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} | "
                f"{fmt_counts(row['top0_category'])} | {fmt_counts(row['top0_text'])} |"
            )

        lines.append("\n### Best Restore Candidates\n")
        lines.append("| layer | component | n | tok0 | newline_top0 | rank | prefix-newline |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in data["summary"]["best_restore"][:32]:
            n = row["n"]
            lines.append(
                f"| {row['layer']} | {row['component']} | {n} | {row['tok0_hit']}/{n} | "
                f"{row['newline_top0']}/{n} | {row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} |"
            )

        lines.append("\n### Control Candidates\n")
        lines.append("| layer | component | control | n | tok0 | newline_top0 | rank | prefix-newline |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for row in data["summary"]["best_controls"][:32]:
            n = row["n"]
            lines.append(
                f"| {row['layer']} | {row['component']} | {row['control']} | {n} | {row['tok0_hit']}/{n} | "
                f"{row['newline_top0']}/{n} | {row['mean_prefix_rank']:.1f} | {row['mean_prefix_minus_newline']:.3f} |"
            )

        lines.append("\n### Component Timeline Restore\n")
        for comp in data["components"]:
            rows = [
                r for r in data["summary"]["by_layer_component"]
                if r["control"] == "restore" and r["component"] == comp
            ]
            rows = sorted(rows, key=lambda r: r["layer"])
            compact = "; ".join(
                f"L{r['layer']} tok0={r['tok0_hit']}/{r['n']} nl={r['newline_top0']}/{r['n']} pmn={r['mean_prefix_minus_newline']:.2f}"
                for r in rows
            )
            lines.append(f"- {comp}: {compact}")
        lines.append("")

    out = OUT_ROOT / "phase640_cross_model_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
