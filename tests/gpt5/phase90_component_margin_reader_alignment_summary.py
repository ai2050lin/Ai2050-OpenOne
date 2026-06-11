from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def summarize(vals: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [v for v in vals if v["component"] == "clean"]
    ablated = [v for v in vals if v["component"] != "clean"]
    gen_vals = [v for v in vals if v.get("generated", "") != ""]
    return {
        "n": len(vals),
        "clean_n": len(clean),
        "ablated_n": len(ablated),
        "value_top1": avg([float(v["value_top1"]) for v in vals]),
        "letter_top1": avg([float(v["letter_top1"]) for v in vals]),
        "choice_top1": avg([float(v["choice_correct"]) for v in gen_vals]),
        "choice_valid": avg([float(v["choice_valid"]) for v in gen_vals]),
        "value_top1_margin": avg([float(v["value_top1_margin"]) for v in vals]),
        "value_mean_margin": avg([float(v["value_mean_margin"]) for v in vals]),
        "letter_top1_margin": avg([float(v["letter_top1_margin"]) for v in vals]),
        "letter_mean_margin": avg([float(v["letter_mean_margin"]) for v in vals]),
        "component_value_effect_top1": avg([float(v.get("component_value_effect_top1", 0.0)) for v in ablated]),
        "component_value_effect_mean": avg([float(v.get("component_value_effect_mean", 0.0)) for v in ablated]),
        "component_letter_effect_top1": avg([float(v.get("component_letter_effect_top1", 0.0)) for v in ablated]),
        "component_letter_effect_mean": avg([float(v.get("component_letter_effect_mean", 0.0)) for v in ablated]),
        "choice_drop": avg([float(v.get("choice_drop", 0.0)) for v in ablated if v.get("generated", "") != ""]),
    }


def group(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return {":".join(map(str, k)): summarize(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    by_model_rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out_dir.glob("*_phase90_component_margin_reader_alignment.json")):
        model = path.name.split("_phase90_")[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows", [])
        by_model_rows[model] = rows
        for row in rows:
            r = dict(row)
            r["model"] = model
            all_rows.append(r)
    summary = {
        "total_rows": len(all_rows),
        "by_model": {m: summarize(rows) for m, rows in by_model_rows.items()},
        "by_model_component": group(all_rows, ["model", "component"]),
        "by_model_layer_component": group(all_rows, ["model", "layer", "component"]),
        "by_model_slot_component": group(all_rows, ["model", "slot", "component"]),
        "by_model_layer_slot_component": group(all_rows, ["model", "layer", "slot", "component"]),
    }
    path = out_dir / "phase90_component_margin_reader_alignment_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_model_component"], ensure_ascii=False, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
