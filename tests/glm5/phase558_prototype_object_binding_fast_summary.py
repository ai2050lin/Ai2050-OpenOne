#!/usr/bin/env python3
"""Summary for Phase558 fast prototype/object-binding next-token audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase558_prototype_object_binding_fast")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase558_{model}_prototype_object_binding_fast.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def gain(row: dict[str, Any] | None) -> float:
    return float(row["restore_gain"]) if row else 0.0


def best(group: list[dict[str, Any]], category: str, variant: str) -> dict[str, Any] | None:
    vals = [r for r in group if r.get("donor_category") == category and r.get("donor_variant") == variant]
    return max(vals, key=lambda r: r["restore_gain"]) if vals else None


def fmt(row: dict[str, Any], model: str) -> str:
    return (
        f"| {model} | {row['combo']} | {','.join(str(x) for x in row['layers'])} | {row['route']} | "
        f"{row['condition']} | {row.get('donor_category', '')} | {row.get('donor_variant', '')} | "
        f"{row['base_target_margin_mean']:+.3f} | {row['target_margin_mean']:+.3f} | "
        f"{row['margin_delta']:+.3f} | {row['remove_delta']:+.3f} | {row['restore_gain']:+.3f} | "
        f"{row['target_rank_mean']:.1f} | {row['target_top1_rate']:.2f} | {row['class']} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    lines = ["# Phase558 Fast Prototype/Object-Binding Summary", ""]

    all_rows: list[tuple[str, dict[str, Any]]] = []
    for model, d in data.items():
        rows = d.get("compact_rows", [])
        all_rows.extend((model, row) for row in rows)
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pair={d['pair']}, window={d['window']}, combos={list(d['combos'])}, "
            f"routes={[r['name'] for r in d['routes']]}, train_n={d['train_n']}, test_n={d['test_n']}"
        )
        lines.append("")
        lines.append("| model | combo | layers | route | condition | donor category | donor variant | base margin | margin | margin delta | remove delta | restore gain | target rank | top1 | class |")
        lines.append("|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: r["restore_gain"], reverse=True):
            lines.append(fmt(row, model))
        lines.append("")

    lines.append("## Prototype Split")
    lines.append("")
    lines.append("| model | combo | route | same | shuffle | best repeat | best repeat id | mean | pca1 | pca3 | random | same-shuffle | repeat-mean | pca3-random |")
    lines.append("|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        rows = [r for m, r in all_rows if m == model and r.get("donor_category") == "vehicle"]
        for combo, route in sorted({(r["combo"], r["route"]) for r in rows}):
            group = [r for r in rows if r["combo"] == combo and r["route"] == route]
            same = best(group, "vehicle", "same")
            shuffle = best(group, "vehicle", "shuffle")
            repeats = [r for r in group if str(r.get("donor_variant", "")).startswith("repeat")]
            best_repeat = max(repeats, key=lambda r: r["restore_gain"]) if repeats else None
            mean = best(group, "vehicle", "mean_cache")
            pca1 = best(group, "vehicle", "pca1_cache")
            pca3 = best(group, "vehicle", "pca3_cache")
            random = best(group, "vehicle", "random_cache")
            sg, shg, rg, mg, p1g, p3g, randg = map(gain, [same, shuffle, best_repeat, mean, pca1, pca3, random])
            lines.append(
                f"| {model} | {combo} | {route} | {sg:+.3f} | {shg:+.3f} | {rg:+.3f} | "
                f"{best_repeat.get('donor_variant', '') if best_repeat else ''} | {mg:+.3f} | {p1g:+.3f} | "
                f"{p3g:+.3f} | {randg:+.3f} | {sg-shg:+.3f} | {rg-mg:+.3f} | {p3g-randg:+.3f} |"
            )
    lines.append("")

    lines.append("## Repeat Matrix")
    lines.append("")
    lines.append("| model | combo | route | repeat0 | repeat1 | repeat2 | repeat3 | repeat4 | repeat5 | spread |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in MODELS:
        rows = [r for m, r in all_rows if m == model and r.get("donor_category") == "vehicle"]
        for combo, route in sorted({(r["combo"], r["route"]) for r in rows}):
            group = [r for r in rows if r["combo"] == combo and r["route"] == route]
            vals = [gain(best(group, "vehicle", f"repeat{i}")) for i in range(6)]
            spread = max(vals) - min(vals) if vals else 0.0
            lines.append(
                f"| {model} | {combo} | {route} | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + f" | {spread:+.3f} |"
            )

    out = root / "phase558_fast_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
