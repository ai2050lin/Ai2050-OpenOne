from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def finite(x: Any) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return v == v and abs(v) != float("inf")


def compact(block: dict[str, Any], limit: int = 500) -> dict[str, Any]:
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
        keys = (
            "clean_value_margin",
            "patched_value_margin",
            "clean_letter_margin",
            "patched_letter_margin",
            "value_delta",
            "letter_delta",
        )
        if any(not finite(row.get(k)) for k in keys):
            bad += 1
    summary = data.get("summary", {})
    return {
        "model": data.get("model"),
        "phase": data.get("phase"),
        "rows": len(rows),
        "bad_numeric_rows": bad,
        "nodes": data.get("nodes", []),
        "num_items": data.get("num_items"),
        "positions": data.get("positions"),
        "donor_kinds": data.get("donor_kinds"),
        "by_node": compact(summary.get("by_node", {})),
        "by_position": compact(summary.get("by_position", {})),
        "by_condition": compact(summary.get("by_condition", {})),
        "by_node_position": compact(summary.get("by_node_position", {})),
        "by_node_condition": compact(summary.get("by_node_condition", {})),
        "by_node_position_condition": compact(summary.get("by_node_position_condition", {}), 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    models = [load_one(p) for p in sorted(out_dir.glob("*_phase97_token_route_local_patch.json"))]
    merged = {
        "phase": 97,
        "output_dir": str(out_dir),
        "models": models,
        "total_rows": sum(m["rows"] for m in models),
        "total_bad_numeric_rows": sum(m["bad_numeric_rows"] for m in models),
    }
    path = out_dir / "phase97_token_route_local_patch_summary.json"
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
