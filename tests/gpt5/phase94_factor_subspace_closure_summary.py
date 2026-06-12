from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return v == v and abs(v) != float("inf")


def compact(block: dict[str, Any], limit: int = 120) -> dict[str, Any]:
    out = {}
    for k, v in sorted(block.items())[:limit]:
        out[k] = {
            "n": v.get("n", 0),
            "value_delta": round(float(v.get("value_delta", 0.0)), 4),
            "letter_delta": round(float(v.get("letter_delta", 0.0)), 4),
            "value_top1_delta": round(float(v.get("value_top1_delta", 0.0)), 4),
            "letter_top1_delta": round(float(v.get("letter_top1_delta", 0.0)), 4),
        }
    return out


def load_one(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    bad = 0
    for row in rows:
        if any(not finite(row.get(k)) for k in ("clean_value_margin", "patched_value_margin", "clean_letter_margin", "patched_letter_margin", "value_delta", "letter_delta")):
            bad += 1
    return {
        "model": data.get("model"),
        "phase": data.get("phase"),
        "rows": len(rows),
        "bad_numeric_rows": bad,
        "nodes": data.get("nodes", []),
        "rank": data.get("rank"),
        "pool_mode": data.get("pool_mode"),
        "copy_mode": data.get("copy_mode"),
        "basis_dims": data.get("basis_dims", {}),
        "by_condition": compact(data.get("summary", {}).get("by_condition", {})),
        "by_factor": compact(data.get("summary", {}).get("by_factor", {})),
        "by_node_condition": compact(data.get("summary", {}).get("by_node_condition", {}), 200),
        "by_node_factor": compact(data.get("summary", {}).get("by_node_factor", {}), 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    models = [load_one(p) for p in sorted(out_dir.glob("*_phase94_factor_subspace_closure.json"))]
    summary = {
        "phase": 94,
        "output_dir": str(out_dir),
        "models": models,
        "total_rows": sum(m["rows"] for m in models),
        "total_bad_numeric_rows": sum(m["bad_numeric_rows"] for m in models),
    }
    path = out_dir / "phase94_factor_subspace_closure_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
