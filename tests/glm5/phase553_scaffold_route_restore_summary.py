#!/usr/bin/env python3
"""Summary for Phase553 scaffold route restore."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase553_scaffold_route_restore")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase553_{model}_scaffold_route_restore.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(model: str, row: dict[str, Any]) -> str:
    return (
        f"| {model} | {row['combo']} | {','.join(str(x) for x in row['layers'])} | "
        f"{row['scaffold']} | {row['mode']} | {row['condition']} | "
        f"{row['base_clean_non_object_rate']:.2f} | {row['clean_non_object_rate']:.2f} | "
        f"{row['clean_delta']:+.2f} | {row['base_label_violation_rate']:.2f} | "
        f"{row['label_violation_rate']:.2f} | {row['label_delta']:+.2f} | "
        f"{row['object_echo_rate']:.2f} | {row['prompt_echo_rate']:.2f} | "
        f"{row['score_delta']:+.2f} | {row['random_delta']:+.2f} | "
        f"{row['drop_vs_random']:+.2f} | {row.get('remove_delta', 0.0):+.2f} | "
        f"{row.get('restore_gain', 0.0):+.2f} | {row['class']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase553 Scaffold Route Restore Summary", ""]
    all_rows = []
    for model, d in data.items():
        rows = d.get("compact_rows", [])
        all_rows.extend((model, row) for row in rows)
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pair={d['pair']}, window={d['window']}, combos={list(d['combos'])}, "
            f"conditions={d['conditions']}, scaffold_modes={d['scaffold_modes']}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, sample_seeds={d['sample_seeds']}, "
            f"remove_scale={d['remove_scale']}, add_alpha={d['add_alpha']}"
        )
        lines.append("")
        lines.append("| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: (r["class"] == "restore_success", r["restore_gain"], r["class"] == "necessity_drop", -r["clean_delta"]), reverse=True):
            lines.append(fmt(model, row))
        lines.append("")

        lines.append("### Best drops by condition")
        lines.append("")
        lines.append("| model | condition | best row | clean delta | score delta | random delta | restore gain | class |")
        lines.append("|---|---|---|---:|---:|---:|---:|---|")
        for cond in d["conditions"]:
            if cond == "baseline":
                continue
            rows_c = [r for r in rows if r["condition"] == cond]
            if not rows_c:
                continue
            if "restore" in cond:
                best = max(rows_c, key=lambda r: (r.get("restore_gain", 0.0), r["clean_delta"]))
            else:
                best = min(rows_c, key=lambda r: (r["clean_delta"], r["score_delta"]))
            desc = f"{best['combo']} {best['scaffold']} {best['mode']}"
            lines.append(f"| {model} | {cond} | {desc} | {best['clean_delta']:+.2f} | {best['score_delta']:+.2f} | {best['random_delta']:+.2f} | {best.get('restore_gain', 0.0):+.2f} | {best['class']} |")
        lines.append("")

    if all_rows:
        lines.append("## Cross-Model Restore Checks")
        lines.append("")
        lines.append("| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        restore_rows = [(m, r) for m, r in all_rows if "restore" in r["condition"]]
        for model, row in sorted(restore_rows, key=lambda mr: (mr[1]["class"] == "restore_success", mr[1].get("restore_gain", 0.0), mr[1]["clean_delta"]), reverse=True)[:100]:
            lines.append(fmt(model, row))
        lines.append("")

        lines.append("## Cross-Model Strongest Drops")
        lines.append("")
        lines.append("| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for model, row in sorted(all_rows, key=lambda mr: (mr[1]["class"] == "necessity_drop", -mr[1]["clean_delta"], -mr[1]["score_delta"]), reverse=True)[:100]:
            lines.append(fmt(model, row))
        lines.append("")

        lines.append("## Cross-Model Add Checks")
        lines.append("")
        lines.append("| model | combo | layers | scaffold | mode | condition | base clean-no | clean-no | clean delta | base label | label | label delta | object echo | prompt echo | score delta | random delta | drop-vs-random | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        add_rows = [(m, r) for m, r in all_rows if r["condition"] == "add_perp"]
        for model, row in sorted(add_rows, key=lambda mr: mr[1]["clean_delta"], reverse=True)[:40]:
            lines.append(fmt(model, row))
        lines.append("")

    out = root / "phase553_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
