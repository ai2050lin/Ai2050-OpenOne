from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
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
    by_condition = group_rows(rows, ["condition"])
    strongest_letter = sorted(by_condition.items(), key=lambda kv: kv[1]["letter_delta"])[:16]
    strongest_value = sorted(by_condition.items(), key=lambda kv: kv[1]["value_delta"])[:16]
    source = data.get("source_attention", {})
    source_top: list[dict[str, Any]] = []
    by_head_label = source.get("by_head_label", {}) if isinstance(source, dict) else {}
    for key, value in sorted(by_head_label.items(), key=lambda kv: kv[1], reverse=True)[:30]:
        source_top.append({"key": key, "attn": value})
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
        "by_head_set": group_rows(rows, ["head_set_name"]),
        "by_position": group_rows(rows, ["position_kind"]),
        "by_position_condition": group_rows(rows, ["position_kind", "condition"]),
        "strongest_letter_conditions": [{"key": k, **v} for k, v in strongest_letter],
        "strongest_value_conditions": [{"key": k, **v} for k, v in strongest_value],
        "source_attention_items": source.get("items") if isinstance(source, dict) else None,
        "source_attention_error": source.get("error") if isinstance(source, dict) else None,
        "source_attention_top": source_top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    paths = sorted(out_dir.glob("*_phase100_head_route_restore_source.json"))
    summary = {"phase": 100, "output_dir": str(out_dir), "models": [load_one(path) for path in paths]}
    summary["total_rows"] = sum(m["rows"] for m in summary["models"])
    summary["total_bad_numeric_rows"] = sum(m["bad_numeric_rows"] for m in summary["models"])
    out_path = out_dir / "phase100_head_route_restore_source_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
