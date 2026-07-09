#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")

PHASE = 282
SCHEMA_VERSION = "2.9.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase282_gap_recalibration_after_phase281"


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


def recalibrate(row: dict[str, Any], closure: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    flags = dict(row.get("remaining_gap_flags") or row.get("gap_flags") or {})
    c = closure.get(key(row))
    candidate_verified = bool(c)
    four_closed = bool(c and c.get("four_condition_closed"))
    weak_survived = bool(c and c.get("weak_candidate_survived"))
    remaining = dict(flags)
    if candidate_verified:
        remaining["candidate_not_closed"] = not four_closed
        if four_closed:
            remaining["need_closure_quality"] = False
            remaining["need_readout_competition"] = False
        elif weak_survived:
            remaining["need_closure_quality"] = True
        else:
            remaining["need_closure_quality"] = True
    dimensions = [name for name, value in remaining.items() if value and name.startswith("need_")]
    if remaining.get("candidate_not_closed"):
        dimensions.append("candidate_closure_verification")
    priority = safe_float(row.get("priority_score_after_phase279", row.get("priority_score_after_phase275", row.get("priority_score"))))
    if candidate_verified:
        priority -= 0.75
    if four_closed:
        priority -= 3.0
    elif weak_survived:
        priority -= 0.5
    priority = round(max(0.0, priority), 6)
    if not dimensions:
        status = "filled_by_phase275_277_279_281"
    elif candidate_verified:
        status = "closure_quality_rechecked_phase281"
    else:
        status = row.get("phase280_status", "still_open")
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase282_created_at": now(),
        "phase282_closure_quality_ref": c.get("closure_quality_id") if c else None,
        "phase282_closure_resolution": {
            "candidate_verified": candidate_verified,
            "semantic_done": bool(c and c.get("semantic_done")),
            "stop_wins": bool(c and c.get("stop_wins")),
            "continue_suppressed": bool(c and c.get("continue_suppressed")),
            "rollout_stable": bool(c and c.get("rollout_stable")),
            "four_condition_closed": four_closed,
            "weak_candidate_survived": weak_survived,
            "closure_blockers": c.get("closure_blockers") if c else [],
        },
        "remaining_gap_flags": remaining,
        "remaining_dimensions": dimensions,
        "priority_score_after_phase281": priority,
        "phase282_status": status,
    }


def select_next(rows: list[dict[str, Any]], max_total: int = 54) -> list[dict[str, Any]]:
    open_rows = [r for r in rows if r.get("phase282_status") != "filled_by_phase275_277_279_281"]
    open_rows.sort(key=lambda r: (-safe_float(r.get("priority_score_after_phase281")), str(r.get("family_id")), str(r.get("model")), str(r.get("case_id"))))
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
        row["phase282_next_batch_rank"] = rank
    return selected


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files["phase282_recalibrated_gap_rows"] = "phase282_recalibrated_gap_rows.jsonl"
    files["phase282_next_batch_rows"] = "phase282_next_batch_rows.jsonl"
    files["phase282_summary"] = "phase282_summary.json"
    files["phase282_report"] = "phase282_report.md"
    manifest["latest_gap_recalibration_phase"] = "Phase282"
    manifest["phase282_summary"] = summary
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase282_summary.json", "phase282_next_batch_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase282_summary_ref"] = "phase282_summary.json"
    client["phase282_next_batch_ref"] = "phase282_next_batch_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase282_recalibrated_gap_rows"] = "Phase280 gaps recalibrated after strict Phase281 candidate closure verification"
    tables["phase282_next_batch_rows"] = "next queue after candidate closure recheck"
    write_json(V2 / "schema.json", schema)


def write_report(summary: dict[str, Any], next_batch: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase282 Gap Recalibration After Phase281",
        "",
        f"- source_gap_rows: {summary['source_gap_rows']}",
        f"- closure_quality_rows: {summary['closure_quality_rows']}",
        f"- four_condition_closed_count: {summary['four_condition_closed_count']}",
        f"- remaining_gap_counts: {json.dumps(summary['remaining_gap_counts'], ensure_ascii=False)}",
        f"- next_batch_rows: {summary['next_batch_rows']}",
        "",
        "## Next Batch Preview",
        "",
    ]
    for row in next_batch[:12]:
        lines.append(
            f"- rank={row.get('phase282_next_batch_rank')} {row.get('model')} {row.get('family_id')} {row.get('case_id')} "
            f"status={row.get('phase282_status')} priority={row.get('priority_score_after_phase281')} remaining={row.get('remaining_dimensions')}"
        )
    text = "\n".join(lines) + "\n"
    (V2 / "phase282_report.md").write_text(text, encoding="utf-8")
    (OUT / "phase282_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    source = read_jsonl(V2 / "phase280_recalibrated_gap_rows.jsonl")
    closure_rows = read_jsonl(V2 / "phase281_closure_quality_rows.jsonl")
    closure = {key(row): row for row in closure_rows}
    recalibrated = [recalibrate(row, closure) for row in source]
    next_batch = select_next(recalibrated, max_total=54)
    status_counts = Counter(str(r.get("phase282_status")) for r in recalibrated)
    remaining_counts = Counter()
    for row in recalibrated:
        for name, value in (row.get("remaining_gap_flags") or {}).items():
            if value:
                remaining_counts[name] += 1
    four_closed = sum(1 for r in closure_rows if r.get("four_condition_closed"))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase282",
        "created_at": now(),
        "source_gap_rows": len(source),
        "closure_quality_rows": len(closure_rows),
        "four_condition_closed_count": four_closed,
        "weak_candidate_survived_count": sum(1 for r in closure_rows if r.get("weak_candidate_survived")),
        "phase282_status_counts": dict(status_counts),
        "remaining_gap_counts": dict(remaining_counts),
        "next_batch_rows": len(next_batch),
        "next_batch_by_model": dict(Counter(str(r.get("model")) for r in next_batch)),
        "next_batch_by_family": dict(Counter(str(r.get("family_id")) for r in next_batch)),
        "mean_candidate_stop_continue_margin": mean_safe([safe_float(r.get("stop_continue_margin")) for r in closure_rows]),
        "progress_estimate": {
            "pattern_family_atlas": 0.58,
            "physical_distribution_puzzle": 0.56,
            "component_path_coverage": 0.40,
            "causal_audit_coverage": 0.29,
            "closure": 0.20 if not four_closed else 0.21,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase282_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(OUT / "phase282_next_batch_rows.jsonl", next_batch)
    write_json(OUT / "phase282_summary.json", summary)
    write_jsonl(V2 / "phase282_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(V2 / "phase282_next_batch_rows.jsonl", next_batch)
    write_json(V2 / "phase282_summary.json", summary)
    update_v2(summary)
    write_report(summary, next_batch)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
