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


def compact_block(summary: dict[str, Any], key: str, limit: int = 80) -> dict[str, Any]:
    block = summary.get(key, {})
    out = {}
    for name, vals in sorted(block.items())[:limit]:
        out[name] = {
            "n": vals.get("n", 0),
            "value_drop": round(float(vals.get("value_drop", 0.0)), 4),
            "value_patch_gain": round(float(vals.get("value_patch_gain", 0.0)), 4),
            "value_patch_gap": round(float(vals.get("value_patch_gap", 0.0)), 4),
            "letter_drop": round(float(vals.get("letter_drop", 0.0)), 4),
            "letter_patch_gain": round(float(vals.get("letter_patch_gain", 0.0)), 4),
            "letter_patch_gap": round(float(vals.get("letter_patch_gap", 0.0)), 4),
            "value_top1_drop": round(float(vals.get("value_top1_drop", 0.0)), 4),
            "value_top1_patch_gain": round(float(vals.get("value_top1_patch_gain", 0.0)), 4),
            "letter_top1_drop": round(float(vals.get("letter_top1_drop", 0.0)), 4),
            "letter_top1_patch_gain": round(float(vals.get("letter_top1_patch_gain", 0.0)), 4),
        }
    return out


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
        for key in numeric_keys:
            if not finite(row.get(key)):
                bad += 1
                break
    by_kind = {}
    for kind in sorted({r.get("donor_kind", "") for r in rows}):
        vals = [r for r in rows if r.get("donor_kind") == kind]
        by_kind[kind] = {
            "n": len(vals),
            "value_drop": avg([float(v["value_drop"]) for v in vals]),
            "value_patch_gain": avg([float(v["value_patch_gain"]) for v in vals]),
            "value_patch_gap": avg([float(v["value_patch_gap"]) for v in vals]),
            "letter_drop": avg([float(v["letter_drop"]) for v in vals]),
            "letter_patch_gain": avg([float(v["letter_patch_gain"]) for v in vals]),
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
        "copy_mode": data.get("copy_mode", ""),
        "by_kind": by_kind,
        "by_node": compact_block(data.get("summary", {}), "by_node"),
        "by_node_donor_kind": compact_block(data.get("summary", {}), "by_node_donor_kind"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    model_files = sorted(out_dir.glob("*_phase92_cross_item_component_transplant.json"))
    models = [load_model_result(p) for p in model_files]
    summary = {
        "phase": 92,
        "output_dir": str(out_dir),
        "models": models,
        "total_rows": sum(m["num_rows"] for m in models),
        "total_bad_numeric_rows": sum(m["bad_numeric_rows"] for m in models),
    }
    path = out_dir / "phase92_cross_item_component_transplant_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
