#!/usr/bin/env python3
"""Summary for Phase555 wrong-donor specificity audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/glm5_phase555_wrong_donor_specificity")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def load(root: Path, model: str) -> dict[str, Any] | None:
    path = root / f"phase555_{model}_wrong_donor_specificity.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(model: str, row: dict[str, Any]) -> str:
    return (
        f"| {model} | {row['combo']} | {','.join(str(x) for x in row['layers'])} | "
        f"{row['route']} | {row['recipient_scaffold']} | {row['donor_scaffold']} | "
        f"{row['mode']} | {row['condition']} | {row.get('donor_category', '')} | "
        f"{row.get('donor_state', '')} | {row['base_clean_non_object_rate']:.2f} | "
        f"{row['clean_non_object_rate']:.2f} | {row['clean_delta']:+.2f} | "
        f"{row['label_violation_rate']:.2f} | {row['label_delta']:+.2f} | "
        f"{row['score_delta']:+.2f} | {row['random_delta']:+.2f} | "
        f"{row['remove_delta']:+.2f} | {row['restore_gain']:+.2f} | {row['class']} |"
    )


def is_restore(row: dict[str, Any]) -> bool:
    return bool(row.get("donor_category"))


def route_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["combo"], row["route"], row["donor_state"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    data = {m: load(root, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}

    lines = ["# Phase555 Wrong-Donor Specificity Summary", ""]
    all_rows: list[tuple[str, dict[str, Any]]] = []
    for model, d in data.items():
        rows = d.get("compact_rows", [])
        all_rows.extend((model, row) for row in rows)
        lines.append(f"## {model}")
        lines.append("")
        lines.append(
            f"pair={d['pair']}, window={d['window']}, combos={list(d['combos'])}, "
            f"conditions={d['conditions']}, routes={[r['name'] for r in d['routes']]}, "
            f"train_n={d['train_n']}, test_n={d['test_n']}, sample_seeds={d['sample_seeds']}, "
            f"remove_scale={d['remove_scale']}, add_alpha={d['add_alpha']}"
        )
        lines.append("")
        lines.append("| model | combo | layers | route | recipient scaffold | donor scaffold | mode | condition | donor category | donor state | base clean-no | clean-no | clean delta | label | label delta | score delta | random delta | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for row in sorted(rows, key=lambda r: (is_restore(r), r.get("restore_gain", 0.0), -abs(r.get("label_delta", 0.0))), reverse=True):
            lines.append(fmt(model, row))
        lines.append("")

    if all_rows:
        lines.append("## Same-Category Versus Wrong-Donor Specificity")
        lines.append("")
        lines.append("| model | combo | route | donor state | vehicle gain | best wrong category | best wrong gain | specificity gap | vehicle class | wrong class |")
        lines.append("|---|---|---|---|---:|---|---:|---:|---|---|")
        for model in MODELS:
            model_rows = [r for m, r in all_rows if m == model and is_restore(r)]
            keys = sorted({route_key(r) for r in model_rows})
            for combo, route, donor_state in keys:
                group = [r for r in model_rows if route_key(r) == (combo, route, donor_state)]
                vehicle = [r for r in group if r.get("donor_category") == "vehicle"]
                wrong = [r for r in group if r.get("donor_category") != "vehicle"]
                if not vehicle or not wrong:
                    continue
                best_vehicle = max(vehicle, key=lambda r: r["restore_gain"])
                best_wrong = max(wrong, key=lambda r: r["restore_gain"])
                gap = best_vehicle["restore_gain"] - best_wrong["restore_gain"]
                lines.append(
                    f"| {model} | {combo} | {route} | {donor_state} | "
                    f"{best_vehicle['restore_gain']:+.2f} | {best_wrong['donor_category']} | "
                    f"{best_wrong['restore_gain']:+.2f} | {gap:+.2f} | "
                    f"{best_vehicle['class']} | {best_wrong['class']} |"
                )
        lines.append("")

        lines.append("## Strongest Specific Restores")
        lines.append("")
        lines.append("| model | combo | layers | route | recipient scaffold | donor scaffold | mode | condition | donor category | donor state | base clean-no | clean-no | clean delta | label | label delta | score delta | random delta | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        restore_rows = [(m, r) for m, r in all_rows if is_restore(r)]
        for model, row in sorted(restore_rows, key=lambda mr: (mr[1]["class"] == "restore_success", mr[1].get("restore_gain", 0.0)), reverse=True)[:80]:
            lines.append(fmt(model, row))
        lines.append("")

        lines.append("## Strongest Necessity Drops")
        lines.append("")
        lines.append("| model | combo | layers | route | recipient scaffold | donor scaffold | mode | condition | donor category | donor state | base clean-no | clean-no | clean delta | label | label delta | score delta | random delta | remove delta | restore gain | class |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for model, row in sorted(all_rows, key=lambda mr: (mr[1]["class"] == "necessity_drop", -mr[1]["clean_delta"], -mr[1]["score_delta"]), reverse=True)[:60]:
            lines.append(fmt(model, row))
        lines.append("")

    out = root / "phase555_cross_model_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
