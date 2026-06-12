from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def load_model_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    bad = 0
    for row in rows:
        for key in ("value_delta", "letter_delta", "value_top1_delta", "letter_top1_delta"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                bad += 1
                break

    def group(keys: list[str]) -> dict[str, dict[str, float]]:
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

    by_condition = group(["condition"])
    strongest_letter = sorted(by_condition.items(), key=lambda kv: kv[1]["letter_delta"])[:12]
    strongest_value = sorted(by_condition.items(), key=lambda kv: kv[1]["value_delta"])[:12]
    by_position_condition = group(["position_kind", "condition"])
    return {
        "model": data["model"],
        "phase": data.get("phase"),
        "rows": len(rows),
        "bad_numeric_rows": bad,
        "layer": data.get("layer"),
        "n_heads": data.get("n_heads"),
        "head_sets": data.get("head_sets"),
        "num_items": data.get("num_items"),
        "positions": data.get("positions"),
        "by_condition": by_condition,
        "by_head_set": group(["head_set_name"]),
        "by_position": group(["position_kind"]),
        "by_position_condition": by_position_condition,
        "strongest_letter_conditions": [{"key": k, **v} for k, v in strongest_letter],
        "strongest_value_conditions": [{"key": k, **v} for k, v in strongest_value],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    paths = sorted(out_dir.glob("*_phase99_head_set_route_closure.json"))
    summary = {
        "phase": 99,
        "output_dir": str(out_dir),
        "models": [load_model_result(path) for path in paths],
    }
    summary["total_rows"] = sum(m["rows"] for m in summary["models"])
    summary["total_bad_numeric_rows"] = sum(m["bad_numeric_rows"] for m in summary["models"])
    out_path = out_dir / "phase99_head_set_route_closure_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
