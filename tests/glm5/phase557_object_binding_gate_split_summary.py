#!/usr/bin/env python3
"""Summary for Phase557 object-binding and generic-gate split audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase557_object_binding_gate_split")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase557_{model}_object_binding_gate_split.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def is_restore(row: dict[str, Any]) -> bool:
    return bool(row.get("donor_category"))


def fmt(model: str, row: dict[str, Any]) -> str:
    return (
        f"| {model} | {row['combo']} | {','.join(str(x) for x in row['layers'])} | "
        f"{row['route']} | {row['condition']} | {row.get('donor_category', '')} | "
        f"{row.get('donor_variant', '')} | {row['base_clean_non_object_rate']:.2f} | "
        f"{row['clean_non_object_rate']:.2f} | {row['clean_delta']:+.2f} | "
        f"{row['remove_delta']:+.2f} | {row['restore_gain']:+.2f} | "
        f"{row['label_delta']:+.2f} | {row['class']} |"
    )


def best(group: list[dict[str, Any]], category: str, variant: str) -> dict[str, Any] | None:
    vals = [r for r in group if r.get("donor_category") == category and r.get("donor_variant") == variant]
    return max(vals, key=lambda r: r["restore_gain"]) if vals else None


def gain(row: dict[str, Any] | None) -> float:
    return float(row["restore_gain"]) if row else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase557 Object-Binding and Generic-Gate Split Summary", ""]
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
        lines.append("| model | combo | layers | route | condition | donor category | donor variant | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |")
        lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: (is_restore(r), r.get("restore_gain", 0.0), -abs(r.get("label_delta", 0.0))), reverse=True):
            lines.append(fmt(model, row))
        lines.append("")

    if all_rows:
        lines.append("## Object Binding Split")
        lines.append("")
        lines.append("| model | combo | route | same | near | far | shuffle | repeat | random cache | same-near | same-far | same-shuffle | same-random |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model in MODELS:
            model_rows = [r for m, r in all_rows if m == model and is_restore(r)]
            keys = sorted({(r["combo"], r["route"]) for r in model_rows})
            for combo, route in keys:
                group = [r for r in model_rows if r["combo"] == combo and r["route"] == route]
                same = best(group, "vehicle", "same")
                near = best(group, "vehicle", "near")
                far = best(group, "vehicle", "far")
                shuffle = best(group, "vehicle", "shuffle")
                repeat = best(group, "vehicle", "repeat")
                random_cache = best(group, "vehicle", "random_cache")
                if not same:
                    continue
                sg, ng, fg, shg, rg, rcg = map(gain, [same, near, far, shuffle, repeat, random_cache])
                lines.append(
                    f"| {model} | {combo} | {route} | {sg:+.2f} | {ng:+.2f} | {fg:+.2f} | "
                    f"{shg:+.2f} | {rg:+.2f} | {rcg:+.2f} | {sg-ng:+.2f} | {sg-fg:+.2f} | "
                    f"{sg-shg:+.2f} | {sg-rcg:+.2f} |"
                )
        lines.append("")

        lines.append("## Strongest Restores")
        lines.append("")
        lines.append("| model | combo | layers | route | condition | donor category | donor variant | base clean-no | clean-no | clean delta | remove delta | restore gain | label delta | class |")
        lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for model, row in sorted(
            [(m, r) for m, r in all_rows if is_restore(r)],
            key=lambda mr: (mr[1]["class"] == "restore_success", mr[1].get("restore_gain", 0.0)),
            reverse=True,
        )[:100]:
            lines.append(fmt(model, row))
        lines.append("")

    out = root / "phase557_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
