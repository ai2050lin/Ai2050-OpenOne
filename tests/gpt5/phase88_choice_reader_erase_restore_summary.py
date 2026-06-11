from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def rows_from(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("rows", []))


def summarize(vals: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [v for v in vals if v["base_choice_correct"]]
    closed_eligible = [v for v in vals if int(v["closed_base_rank"]) == 1]
    return {
        "n": len(vals),
        "eligible_n": len(eligible),
        "closed_eligible_n": len(closed_eligible),
        "base_choice_top1": avg([float(v["base_choice_correct"]) for v in vals]),
        "erase_choice_top1": avg([float(v["erase_choice_correct"]) for v in vals]),
        "restore_choice_top1": avg([float(v["restore_choice_correct"]) for v in vals]),
        "choice_drop": avg([float(v["choice_drop"]) for v in vals]),
        "choice_restore_gain": avg([float(v["choice_restore_gain"]) for v in vals]),
        "choice_restore_gap": avg([float(v["choice_restore_gap"]) for v in vals]),
        "eligible_choice_drop": avg([float(v["choice_drop"]) for v in eligible]),
        "eligible_choice_restore_gain": avg([float(v["choice_restore_gain"]) for v in eligible]),
        "closed_base_top1": avg([float(int(v["closed_base_rank"]) == 1) for v in vals]),
        "closed_erase_top1": avg([float(int(v["closed_erase_rank"]) == 1) for v in vals]),
        "closed_restore_top1": avg([float(int(v["closed_restore_rank"]) == 1) for v in vals]),
        "closed_drop": avg([float(v["closed_drop"]) for v in vals]),
        "closed_restore_gain": avg([float(v["closed_restore_gain"]) for v in vals]),
        "closed_restore_gap": avg([float(v["closed_restore_gap"]) for v in vals]),
        "closed_base_margin": avg([float(v["closed_base_margin"]) for v in vals]),
        "closed_erase_margin": avg([float(v["closed_erase_margin"]) for v in vals]),
        "closed_restore_margin": avg([float(v["closed_restore_margin"]) for v in vals]),
        "closed_choice_agreement_base": avg([float(v["closed_choice_agreement_base"]) for v in vals]),
        "closed_choice_agreement_erase": avg([float(v["closed_choice_agreement_erase"]) for v in vals]),
        "closed_choice_agreement_restore": avg([float(v["closed_choice_agreement_restore"]) for v in vals]),
    }


def group(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        k = tuple(row[x] for x in keys)
        grouped.setdefault(k, []).append(row)
    return {":".join(map(str, k)): summarize(v) for k, v in sorted(grouped.items(), key=lambda kv: str(kv[0]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(out_dir.glob("*_phase88_choice_reader_erase_restore.json")):
        model = path.name.split("_phase88_")[0]
        rows = rows_from(path)
        model_rows[model] = rows
        for row in rows:
            row = dict(row)
            row["model"] = model
            all_rows.append(row)
    summary = {
        "total_rows": len(all_rows),
        "by_model": {m: summarize(rows) for m, rows in model_rows.items()},
        "by_model_condition": group(all_rows, ["model", "condition"]),
        "by_model_condition_order": group(all_rows, ["model", "condition", "order_key"]),
        "by_model_condition_template": group(all_rows, ["model", "condition", "template_key"]),
        "by_model_condition_path": group(all_rows, ["model", "condition", "destroy_layer", "restore_layer"]),
        "by_model_condition_relation": group(all_rows, ["model", "condition", "relation"]),
        "by_condition": group(all_rows, ["condition"]),
    }
    summary_path = out_dir / "phase88_choice_reader_erase_restore_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_model"], ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
