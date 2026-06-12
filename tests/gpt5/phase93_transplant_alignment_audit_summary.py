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


def small(vals: dict[str, Any]) -> dict[str, Any]:
    return {
        "n": vals.get("n", 0),
        "value_drop": round(float(vals.get("value_drop", 0.0)), 4),
        "value_patch_gain": round(float(vals.get("value_patch_gain", 0.0)), 4),
        "value_patch_gap": round(float(vals.get("value_patch_gap", 0.0)), 4),
        "letter_drop": round(float(vals.get("letter_drop", 0.0)), 4),
        "letter_patch_gain": round(float(vals.get("letter_patch_gain", 0.0)), 4),
        "letter_patch_gap": round(float(vals.get("letter_patch_gap", 0.0)), 4),
        "value_top1_patch_gain": round(float(vals.get("value_top1_patch_gain", 0.0)), 4),
        "letter_top1_patch_gain": round(float(vals.get("letter_top1_patch_gain", 0.0)), 4),
    }


def compact(block: dict[str, Any], limit: int = 120) -> dict[str, Any]:
    return {k: small(v) for k, v in sorted(block.items())[:limit]}


def load_model_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    numeric_keys = [
        "clean_value_margin", "zero_value_margin", "patched_value_margin",
        "clean_letter_margin", "zero_letter_margin", "patched_letter_margin",
        "value_drop", "value_patch_gain", "value_patch_gap",
        "letter_drop", "letter_patch_gain", "letter_patch_gap",
    ]
    bad = 0
    for row in rows:
        if any(not finite(row.get(k)) for k in numeric_keys):
            bad += 1
    by_copy_mode = {}
    for mode in sorted({r.get("copy_mode", "") for r in rows}):
        vals = [r for r in rows if r.get("copy_mode") == mode]
        by_copy_mode[mode] = {
            "n": len(vals),
            "value_patch_gain": avg([float(v["value_patch_gain"]) for v in vals]),
            "letter_patch_gain": avg([float(v["letter_patch_gain"]) for v in vals]),
            "value_patch_gap": avg([float(v["value_patch_gap"]) for v in vals]),
            "letter_patch_gap": avg([float(v["letter_patch_gap"]) for v in vals]),
            "value_top1_patch_gain": avg([float(v["value_top1_patch_gain"]) for v in vals]),
            "letter_top1_patch_gain": avg([float(v["letter_top1_patch_gain"]) for v in vals]),
        }
    return {
        "model": data.get("model"),
        "phase": data.get("phase"),
        "num_rows": len(rows),
        "bad_numeric_rows": bad,
        "nodes": data.get("nodes", []),
        "num_items": data.get("num_items", 0),
        "copy_modes": data.get("copy_modes", []),
        "donor_kinds": data.get("donor_kinds", []),
        "by_copy_mode": by_copy_mode,
        "by_node_copy_mode": compact(data.get("summary", {}).get("by_node_copy_mode", {})),
        "by_node_copy_mode_donor_kind": compact(data.get("summary", {}).get("by_node_copy_mode_donor_kind", {}), 200),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    files = sorted(out_dir.glob("*_phase93_transplant_alignment_audit.json"))
    models = [load_model_result(p) for p in files]
    summary = {
        "phase": 93,
        "output_dir": str(out_dir),
        "models": models,
        "total_rows": sum(m["num_rows"] for m in models),
        "total_bad_numeric_rows": sum(m["bad_numeric_rows"] for m in models),
    }
    path = out_dir / "phase93_transplant_alignment_audit_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
