from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def summarize(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "clean_value_top1": avg([float(v["clean_value_top1"]) for v in vals]),
        "zero_value_top1": avg([float(v["zero_value_top1"]) for v in vals]),
        "restore_value_top1": avg([float(v["restore_value_top1"]) for v in vals]),
        "clean_letter_top1": avg([float(v["clean_letter_top1"]) for v in vals]),
        "zero_letter_top1": avg([float(v["zero_letter_top1"]) for v in vals]),
        "restore_letter_top1": avg([float(v["restore_letter_top1"]) for v in vals]),
        "clean_choice_top1": avg([float(v["clean_choice_correct"]) for v in vals]),
        "zero_choice_top1": avg([float(v["zero_choice_correct"]) for v in vals]),
        "restore_choice_top1": avg([float(v["restore_choice_correct"]) for v in vals]),
        "value_drop": avg([float(v["value_drop"]) for v in vals]),
        "value_restore_gain": avg([float(v["value_restore_gain"]) for v in vals]),
        "value_restore_gap": avg([float(v["value_restore_gap"]) for v in vals]),
        "letter_drop": avg([float(v["letter_drop"]) for v in vals]),
        "letter_restore_gain": avg([float(v["letter_restore_gain"]) for v in vals]),
        "letter_restore_gap": avg([float(v["letter_restore_gap"]) for v in vals]),
        "choice_drop": avg([float(v["choice_drop"]) for v in vals]),
        "choice_restore_gain": avg([float(v["choice_restore_gain"]) for v in vals]),
        "choice_restore_gap": avg([float(v["choice_restore_gap"]) for v in vals]),
        "clean_choice_valid": avg([float(v["clean_choice_valid"]) for v in vals]),
        "zero_choice_valid": avg([float(v["zero_choice_valid"]) for v in vals]),
        "restore_choice_valid": avg([float(v["restore_choice_valid"]) for v in vals]),
    }


def group(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] if k != "node" else f"L{row['layer']}:{row['component']}" for k in keys)
        grouped.setdefault(key, []).append(row)
    return {":".join(map(str, k)): summarize(v) for k, v in sorted(grouped.items(), key=lambda kv: str(kv[0]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out_dir.glob("*_phase91_component_restore_reader_closure.json")):
        model = path.name.split("_phase91_")[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows", [])
        by_model[model] = rows
        for row in rows:
            r = dict(row)
            r["model"] = model
            all_rows.append(r)
    summary = {
        "total_rows": len(all_rows),
        "by_model": {m: summarize(rows) for m, rows in by_model.items()},
        "by_model_node": group(all_rows, ["model", "node"]),
        "by_model_node_slot": group(all_rows, ["model", "node", "slot"]),
        "by_model_slot": group(all_rows, ["model", "slot"]),
    }
    path = out_dir / "phase91_component_restore_reader_closure_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_model_node"], ensure_ascii=False, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
