#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 865
PHASE864_SUMMARY = Path("tests/result/phase864_answer_lift_vs_blocker_weakening_route_separation/phase864_summary.json")
RESULT_ROOT = Path("tests/result/phase865_route_purity_and_side_effect_filter")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def purity_class(row: dict[str, Any], object_delta_threshold: float) -> str:
    clear_gain = int(row.get("clear_gain") or 0)
    clear_loss = int(row.get("clear_loss") or 0)
    route = str(row.get("route_class"))
    object_delta = finite(row.get("mean_object_delta"))
    object_induced = int(row.get("object_echo_induced") or 0)
    format_induced = int(row.get("format_or_other_induced") or 0)
    answer_delta = finite(row.get("mean_answer_delta"))
    blocker_reduction = finite(row.get("mean_class_blocker_reduction"))
    blocker_delta = finite(row.get("mean_original_blocker_delta"))
    if clear_gain <= 0 and clear_loss <= 0:
        return "inactive_or_weak"
    if clear_loss > 0:
        return "harmful_or_unstable"
    if object_induced > 0 or object_delta > float(object_delta_threshold):
        return "object_side_effect_risk"
    if format_induced > 0:
        return "format_side_effect_risk"
    if "mixed_answer_lift_and_blocker_weakening" == route and answer_delta > 0 and blocker_reduction > 0 and blocker_delta < 0:
        return "clean_mixed_answer_blocker_route"
    if route == "answer_lift_dominant" and answer_delta > 0:
        return "clean_answer_lift_route"
    if route == "blocker_weakening_dominant" and blocker_reduction > 0 and blocker_delta < 0:
        return "clean_blocker_weakening_route"
    return "positive_but_unresolved"


def route_rows(object_delta_threshold: float) -> list[dict[str, Any]]:
    payload = read_json(PHASE864_SUMMARY)
    out = []
    for row in payload.get("route_rows") or []:
        if row.get("condition_type") not in {"full_set", "single_channel"}:
            continue
        copied = dict(row)
        copied["purity_class"] = purity_class(copied, object_delta_threshold)
        copied["object_delta_threshold"] = object_delta_threshold
        copied["purity_key"] = f"{copied.get('model')}:{copied.get('domain')}:{copied.get('condition_type')}:{copied.get('subset_name')}:{copied.get('edit_mode')}"
        out.append(copied)
    out.sort(key=lambda row: (str(row.get("model")), str(row.get("domain")), str(row.get("condition_type")), str(row.get("subset_name")), str(row.get("edit_mode"))))
    return out


def summarize(rows: list[dict[str, Any]], object_delta_threshold: float) -> dict[str, Any]:
    full_rows = [row for row in rows if row.get("condition_type") == "full_set"]
    dominant_rows = [
        row
        for row in rows
        if row.get("condition_type") == "single_channel"
        and any(str(role).startswith("dominant") for role in row.get("channel_role_classes") or [])
    ]
    clean_full = [row for row in full_rows if str(row.get("purity_class")).startswith("clean_")]
    clean_dominant = [row for row in dominant_rows if str(row.get("purity_class")).startswith("clean_")]
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_rows:
        by_domain[f"{row.get('model')}:{row.get('domain')}"].append(row)
    return {
        "phase": PHASE,
        "title": "Route Purity and Side-Effect Filter",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(PHASE864_SUMMARY),
        "object_delta_threshold": object_delta_threshold,
        "n_rows": len(rows),
        "purity_class_counts": dict(Counter(str(row.get("purity_class")) for row in rows)),
        "full_set_purity_class_counts": dict(Counter(str(row.get("purity_class")) for row in full_rows)),
        "dominant_channel_purity_class_counts": dict(Counter(str(row.get("purity_class")) for row in dominant_rows)),
        "clean_full_set_routes": clean_full,
        "clean_dominant_channel_routes": clean_dominant,
        "domain_full_set_classes": {
            domain: dict(Counter(str(row.get("purity_class")) for row in group))
            for domain, group in sorted(by_domain.items())
        },
        "boundary": "offline filter from Phase 864 route rows; no new model intervention and not closure",
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 865 Route Purity and Side-Effect Filter",
        "",
        "- Source: Phase 864 route separation.",
        "- Boundary: offline filter, not new model intervention and not closure.",
        "",
        "## Summary",
        "",
        f"- full_set_purity_class_counts: `{summary.get('full_set_purity_class_counts')}`",
        f"- dominant_channel_purity_class_counts: `{summary.get('dominant_channel_purity_class_counts')}`",
        "",
        "## Clean Full-Set Routes",
        "",
        "| model | domain | mode | route | gain/loss | answer delta | blocker reduction | blocker delta | object delta | purity |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("clean_full_set_routes") or []:
        lines.append(
            f"| {row.get('model')} | {row.get('domain')} | `{row.get('edit_mode')}` | `{row.get('route_class')}` | "
            f"{row.get('clear_gain', 0)}/{row.get('clear_loss', 0)} | {finite(row.get('mean_answer_delta')):.4f} | "
            f"{finite(row.get('mean_class_blocker_reduction')):.4f} | {finite(row.get('mean_original_blocker_delta')):.4f} | "
            f"{finite(row.get('mean_object_delta')):.4f} | `{row.get('purity_class')}` |"
        )
    lines += [
        "",
        "## Domain Full-Set Classes",
        "",
        f"`{summary.get('domain_full_set_classes')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    parser.add_argument("--output-dir", default=str(RESULT_ROOT))
    args = parser.parse_args()
    rows = route_rows(float(args.object_delta_threshold))
    summary = summarize(rows, float(args.object_delta_threshold))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "phase865_summary.json", summary)
    write_jsonl(out_dir / "phase865_route_purity_rows.jsonl", rows)
    (out_dir / "phase865_summary.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"phase": PHASE, "purity_class_counts": summary["purity_class_counts"], "full_set": summary["full_set_purity_class_counts"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
