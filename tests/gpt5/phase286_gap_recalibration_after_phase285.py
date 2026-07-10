#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 286
SCHEMA_VERSION = "2.13.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase286_gap_recalibration_after_phase285"
FILL_PHASES = ["phase275", "phase277", "phase279", "phase283"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model")), str(row.get("case_id"))


def load_all_components() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for phase in FILL_PHASES:
        for row in read_jsonl(V2 / f"{phase}_component_summary_rows.jsonl"):
            out[key(row)] = row
    return out


def load_all_causal() -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for phase in FILL_PHASES:
        for row in read_jsonl(V2 / f"{phase}_causal_fill_rows.jsonl"):
            out[key(row)].append(row)
    return out


def closure_quality_map() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for phase in ["phase281", "phase285"]:
        for row in read_jsonl(V2 / f"{phase}_closure_quality_rows.jsonl"):
            out[key(row)] = row
    return out


def recalibrate(row: dict[str, Any], components: dict[tuple[str, str], dict[str, Any]], causal: dict[tuple[str, str], list[dict[str, Any]]], closure: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    k = key(row)
    flags = dict(row.get("gap_flags") or {})
    comp = components.get(k)
    crows = causal.get(k, [])
    q = closure.get(k)
    low_rows = [r for r in crows if r.get("side_effect_level") == "lower"]
    low_supported = any(r.get("causal_effect_supported") and not r.get("side_effect_risk") for r in low_rows)
    any_supported = any(r.get("causal_effect_supported") for r in crows)
    four_closed = bool(q and q.get("four_condition_closed"))
    closure_checked = bool(q)
    closure_rejected = bool(q and not q.get("four_condition_closed"))
    remaining = dict(flags)
    if comp:
        remaining["need_component_path"] = False
        remaining["need_layer_path"] = False
    if crows:
        remaining["need_causal_audit"] = False
    if low_supported:
        remaining["good_readout_low_causal"] = False
    if closure_checked:
        remaining["need_closure_quality"] = False
        if remaining.get("candidate_not_closed"):
            remaining["candidate_not_closed"] = not four_closed
        if four_closed:
            remaining["need_readout_competition"] = False
    dimensions = [name for name, value in remaining.items() if value and name.startswith("need_")]
    if remaining.get("candidate_not_closed"):
        dimensions.append("candidate_closure_verification")
    if closure_rejected:
        dimensions.append("closure_rejected")
    priority = safe_float(row.get("priority_score"))
    priority -= 2.0 if comp else 0.0
    priority -= 2.0 if crows else 0.0
    priority -= 1.0 if low_supported else 0.0
    priority -= 1.0 if closure_checked else 0.0
    priority -= 3.0 if four_closed else 0.0
    priority = round(max(0.0, priority), 6)
    resolved_any = bool(comp or crows or closure_checked)
    status = "filled_by_phase275_277_279_281_283_285" if not [d for d in dimensions if d != "closure_rejected"] else ("partially_filled_by_phase275_277_279_281_283_285" if resolved_any else "still_open")
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase286_created_at": now(),
        "phase286_resolution": {
            "component_path_filled": bool(comp),
            "causal_audit_filled": bool(crows),
            "low_side_effect_causal_supported": bool(low_supported),
            "any_causal_effect_supported": bool(any_supported),
            "closure_quality_checked": closure_checked,
            "four_condition_closed": four_closed,
            "closure_rejected": closure_rejected,
            "closure_reclassification": q.get("closure_reclassification") if q else None,
        },
        "remaining_gap_flags": remaining,
        "remaining_dimensions": dimensions,
        "priority_score_after_phase285": priority,
        "phase286_status": status,
    }


def select_next(rows: list[dict[str, Any]], max_total: int = 54) -> list[dict[str, Any]]:
    open_rows = [r for r in rows if r.get("phase286_status") != "filled_by_phase275_277_279_281_283_285"]
    open_rows.sort(key=lambda r: (-safe_float(r.get("priority_score_after_phase285")), str(r.get("family_id")), str(r.get("model")), str(r.get("case_id"))))
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in open_rows:
        k = (str(row.get("family_id")), str(row.get("model")))
        if counts[k] >= 2:
            continue
        selected.append(row)
        counts[k] += 1
        if len(selected) >= max_total:
            break
    for rank, row in enumerate(selected, start=1):
        row["phase286_next_batch_rank"] = rank
    return selected


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files["phase286_recalibrated_gap_rows"] = "phase286_recalibrated_gap_rows.jsonl"
    files["phase286_next_batch_rows"] = "phase286_next_batch_rows.jsonl"
    files["phase286_summary"] = "phase286_summary.json"
    files["phase286_report"] = "phase286_report.md"
    manifest["latest_gap_recalibration_phase"] = "Phase286"
    manifest["phase286_summary"] = summary
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase286_summary.json", "phase286_next_batch_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase286_summary_ref"] = "phase286_summary.json"
    client["phase286_next_batch_ref"] = "phase286_next_batch_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase286_recalibrated_gap_rows"] = "Phase274 gaps recalibrated after Phase285 expanded closure-quality scan"
    tables["phase286_next_batch_rows"] = "next queue after closure-quality expansion"
    write_json(V2 / "schema.json", schema)


def write_report(summary: dict[str, Any], next_batch: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase286 Gap Recalibration After Phase285",
        "",
        f"- source_gap_rows: {summary['source_gap_rows']}",
        f"- closure_quality_checked_rows: {summary['closure_quality_checked_rows']}",
        f"- closure_rejected_rows: {summary['closure_rejected_rows']}",
        f"- remaining_gap_counts: {json.dumps(summary['remaining_gap_counts'], ensure_ascii=False)}",
        f"- next_batch_rows: {summary['next_batch_rows']}",
        "",
        "## Next Batch Preview",
        "",
    ]
    for row in next_batch[:12]:
        lines.append(
            f"- rank={row.get('phase286_next_batch_rank')} {row.get('model')} {row.get('family_id')} {row.get('case_id')} "
            f"status={row.get('phase286_status')} priority={row.get('priority_score_after_phase285')} remaining={row.get('remaining_dimensions')}"
        )
    text = "\n".join(lines) + "\n"
    (V2 / "phase286_report.md").write_text(text, encoding="utf-8")
    (OUT / "phase286_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    source = read_jsonl(V2 / "phase274_gap_rows.jsonl")
    components = load_all_components()
    causal = load_all_causal()
    closure = closure_quality_map()
    recalibrated = [recalibrate(row, components, causal, closure) for row in source]
    next_batch = select_next(recalibrated, max_total=54)
    status_counts = Counter(str(r.get("phase286_status")) for r in recalibrated)
    remaining_counts = Counter()
    closure_checked = 0
    closure_rejected = 0
    for row in recalibrated:
        res = row.get("phase286_resolution") or {}
        closure_checked += int(bool(res.get("closure_quality_checked")))
        closure_rejected += int(bool(res.get("closure_rejected")))
        for name, value in (row.get("remaining_gap_flags") or {}).items():
            if value:
                remaining_counts[name] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase286",
        "created_at": now(),
        "source_gap_rows": len(source),
        "total_component_summary_rows": len(components),
        "total_causal_fill_rows": sum(len(v) for v in causal.values()),
        "closure_quality_checked_rows": closure_checked,
        "closure_rejected_rows": closure_rejected,
        "phase286_status_counts": dict(status_counts),
        "remaining_gap_counts": dict(remaining_counts),
        "next_batch_rows": len(next_batch),
        "next_batch_by_model": dict(Counter(str(r.get("model")) for r in next_batch)),
        "next_batch_by_family": dict(Counter(str(r.get("family_id")) for r in next_batch)),
        "progress_estimate": {
            "pattern_family_atlas": 0.62,
            "physical_distribution_puzzle": 0.60,
            "component_path_coverage": 0.50,
            "causal_audit_coverage": 0.40,
            "closure_quality_measurement": 0.25,
            "closure": 0.20,
        },
        "mean_component_mlp_delta_all_fills": mean_safe([safe_float(r.get("sum_positive_mlp_delta")) for r in components.values()]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase286_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(OUT / "phase286_next_batch_rows.jsonl", next_batch)
    write_json(OUT / "phase286_summary.json", summary)
    write_jsonl(V2 / "phase286_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(V2 / "phase286_next_batch_rows.jsonl", next_batch)
    write_json(V2 / "phase286_summary.json", summary)
    update_v2(summary)
    write_report(summary, next_batch)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
