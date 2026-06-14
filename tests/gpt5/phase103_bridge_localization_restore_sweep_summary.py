from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def group(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = ":".join(str(row[k]) for k in keys)
        buckets.setdefault(name, []).append(row)
    return {
        name: {
            "n": len(vals),
            "value_delta": round(avg([float(v["value_delta"]) for v in vals]), 4),
            "letter_delta": round(avg([float(v["letter_delta"]) for v in vals]), 4),
            "value_top1_delta": round(avg([float(v["value_top1_delta"]) for v in vals]), 4),
            "letter_top1_delta": round(avg([float(v["letter_top1_delta"]) for v in vals]), 4),
        }
        for name, vals in buckets.items()
    }


def load_one(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    bad = 0
    for row in rows:
        for key in ("value_delta", "letter_delta", "value_top1_delta", "letter_top1_delta"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                bad += 1
                break
    by_restore = group(rows, ["restore_node"])
    return {
        "model": data["model"],
        "phase": data.get("phase"),
        "rows": len(rows),
        "bad_numeric_rows": bad,
        "num_items": data.get("num_items"),
        "value_layer": data.get("value_layer"),
        "value_component": data.get("value_component"),
        "restore_nodes": data.get("restore_nodes"),
        "factors": data.get("factors"),
        "basis_dims": data.get("basis_dims"),
        "by_condition": group(rows, ["condition"]),
        "by_restore_node": by_restore,
        "by_factor_restore": group(rows, ["factor", "restore_node"]),
        "best_value_restore": [{"key": k, **v} for k, v in sorted(by_restore.items(), key=lambda kv: kv[1]["value_delta"], reverse=True)[:12]],
        "best_letter_restore": [{"key": k, **v} for k, v in sorted(by_restore.items(), key=lambda kv: kv[1]["letter_delta"], reverse=True)[:12]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    paths = sorted(out_dir.glob("*_phase103_bridge_localization_restore_sweep.json"))
    summary = {"phase": 103, "output_dir": str(out_dir), "models": [load_one(path) for path in paths]}
    summary["total_rows"] = sum(m["rows"] for m in summary["models"])
    summary["total_bad_numeric_rows"] = sum(m["bad_numeric_rows"] for m in summary["models"])
    out_path = out_dir / "phase103_bridge_localization_restore_sweep_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
