#!/usr/bin/env python3
"""Summary for Phase556 donor specificity decomposition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase556_donor_specificity_decomposition")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase556_{model}_donor_specificity_decomposition.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_restore(row: dict[str, Any]) -> bool:
    return bool(row.get("donor_category"))


def fmt(model: str, row: dict[str, Any]) -> str:
    return (
        f"| {model} | {row['combo']} | {','.join(str(x) for x in row['layers'])} | "
        f"{row['route']} | {row['condition']} | {row.get('donor_category', '')} | "
        f"{row.get('donor_variant', '')} | {row.get('donor_state', '')} | "
        f"{row['base_clean_non_object_rate']:.2f} | {row['clean_non_object_rate']:.2f} | "
        f"{row['clean_delta']:+.2f} | {row['remove_delta']:+.2f} | "
        f"{row['restore_gain']:+.2f} | {row['label_delta']:+.2f} | {row['class']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase556 Donor Specificity Decomposition Summary", ""]
    all_rows: list[tuple[str, dict[str, Any]]] = []
    for model, d in data.items():
        rows = d.get("compact_rows", [])
        all_rows.extend((model, row) for row in rows)
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pair={d['pair']}, window={d['window']}, combos={list(d['combos'])}, "
            f"conditions={d['conditions']}, routes={[r['name'] for r in d['routes']]}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, sample_seeds={d['sample_seeds']}"
        )
        lines.append("")
        lines.append("| model | combo | layers | route | condition | donor category | donor variant | donor state | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |")
        lines.append("|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: (is_restore(r), r.get("restore_gain", 0.0), -abs(r.get("label_delta", 0.0))), reverse=True):
            lines.append(fmt(model, row))
        lines.append("")

    if all_rows:
        lines.append("## Decomposition By Route")
        lines.append("")
        lines.append("| model | combo | route | vehicle | tool | unrelated best | vehicle shuffled | tool shuffled | category gap | task/shared gap | shuffle loss |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            model_rows = [r for m, r in all_rows if m == model and is_restore(r)]
            keys = sorted({(r["combo"], r["route"]) for r in model_rows})
            for combo, route in keys:
                group = [r for r in model_rows if r["combo"] == combo and r["route"] == route]

                def best(cat: str, variant: str = "aligned") -> dict[str, Any] | None:
                    vals = [r for r in group if r.get("donor_category") == cat and r.get("donor_variant", "aligned") == variant]
                    return max(vals, key=lambda r: r["restore_gain"]) if vals else None

                veh = best("vehicle")
                tool = best("tool")
                veh_shuf = best("vehicle", "shuffle")
                tool_shuf = best("tool", "shuffle")
                unrelated_vals = [
                    r for r in group
                    if r.get("donor_category") in {"furniture", "animal", "fruit"}
                    and r.get("donor_variant", "aligned") == "aligned"
                ]
                unrelated = max(unrelated_vals, key=lambda r: r["restore_gain"]) if unrelated_vals else None
                if not veh:
                    continue
                veh_g = veh["restore_gain"]
                tool_g = tool["restore_gain"] if tool else 0.0
                unrel_g = unrelated["restore_gain"] if unrelated else 0.0
                veh_shuf_g = veh_shuf["restore_gain"] if veh_shuf else 0.0
                tool_shuf_g = tool_shuf["restore_gain"] if tool_shuf else 0.0
                category_gap = veh_g - max(tool_g, unrel_g)
                task_gap = tool_g - unrel_g
                shuffle_loss = veh_g - veh_shuf_g
                lines.append(
                    f"| {model} | {combo} | {route} | {veh_g:+.2f} | {tool_g:+.2f} | "
                    f"{unrel_g:+.2f} | {veh_shuf_g:+.2f} | {tool_shuf_g:+.2f} | "
                    f"{category_gap:+.2f} | {task_gap:+.2f} | {shuffle_loss:+.2f} |"
                )
        lines.append("")

        lines.append("## Strongest Restores")
        lines.append("")
        lines.append("| model | combo | layers | route | condition | donor category | donor variant | donor state | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |")
        lines.append("|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        restore_rows = [(m, r) for m, r in all_rows if is_restore(r)]
        for model, row in sorted(restore_rows, key=lambda mr: (mr[1]["class"] == "restore_success", mr[1].get("restore_gain", 0.0)), reverse=True)[:100]:
            lines.append(fmt(model, row))
        lines.append("")

    out = root / "phase556_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
