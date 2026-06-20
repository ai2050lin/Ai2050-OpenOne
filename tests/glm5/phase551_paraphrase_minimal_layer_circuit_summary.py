#!/usr/bin/env python3
"""Summary for Phase551 paraphrase minimal layer circuit audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase551_paraphrase_minimal_layer_circuit")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase551_{model}_paraphrase_minimal_layer_circuit.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, Any], model: str) -> str:
    return (
        f"| {model} | {row['combo']} | {','.join(str(x) for x in row['layers'])} | "
        f"{row['scaffold']} | {row['mode']} | {row['component']} | "
        f"{row['base_clean_non_object_rate']:.2f} | {row['clean_non_object_rate']:.2f} | "
        f"{row['clean_gain']:+.2f} | {row['base_label_violation_rate']:.2f} | "
        f"{row['label_violation_rate']:.2f} | {row['label_gain']:+.2f} | "
        f"{row['object_echo_rate']:.2f} | {row['prompt_echo_rate']:.2f} | "
        f"{row['score_gain']:+.2f} | {row['random_gain']:+.2f} | "
        f"{row['above_random']:+.2f} | {row['class']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase551 Paraphrase Minimal Layer Circuit Summary", ""]
    all_rows = []
    for model, d in data.items():
        rows = d.get("compact_rows", [])
        for row in rows:
            all_rows.append((model, row))
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pair={d['pair']}, window={d['window']}, combos={list(d['layer_combos'])}, "
            f"components={d['components']}, scaffolds={d['scaffolds']}, modes={d['decode_modes']}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, sample_seeds={d['sample_seeds']}, alpha={d['alpha']}"
        )
        lines.append("")
        lines.append("| model | combo | layers | scaffold | mode | component | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score gain | random gain | above random | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: (r["class"] == "robust_clean_positive", r["clean_gain"], r["above_random"]), reverse=True):
            lines.append(fmt(row, model))
        lines.append("")

        lines.append("### Best by component")
        lines.append("")
        lines.append("| model | component | best row | clean gain | above random | class |")
        lines.append("|---|---|---|---:|---:|---|")
        for component in d["components"]:
            if component == "baseline":
                continue
            rows_c = [r for r in rows if r["component"] == component]
            if not rows_c:
                continue
            best = max(rows_c, key=lambda r: (r["class"] == "robust_clean_positive", r["clean_gain"], r["above_random"]))
            desc = f"{best['combo']} {best['scaffold']} {best['mode']}"
            lines.append(f"| {model} | {component} | {desc} | {best['clean_gain']:+.2f} | {best['above_random']:+.2f} | {best['class']} |")
        lines.append("")

    if all_rows:
        lines.append("## Cross-Model Best Rows")
        lines.append("")
        lines.append("| model | combo | layers | scaffold | mode | component | base clean-no | clean-no | clean gain | base label | label | label gain | object echo | prompt echo | score gain | random gain | above random | class |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        rows_sorted = sorted(
            all_rows,
            key=lambda mr: (mr[1]["class"] == "robust_clean_positive", mr[1]["clean_gain"], mr[1]["above_random"]),
            reverse=True,
        )
        for model, row in rows_sorted[:80]:
            lines.append(fmt(row, model))
        lines.append("")

    out = root / "phase551_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
