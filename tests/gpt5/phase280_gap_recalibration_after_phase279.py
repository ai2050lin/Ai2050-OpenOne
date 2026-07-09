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

PHASE = 280
SCHEMA_VERSION = "2.7.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase280_gap_recalibration_after_phase279"


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
    for phase in ["phase275", "phase277", "phase279"]:
        for row in read_jsonl(V2 / f"{phase}_component_summary_rows.jsonl"):
            out[key(row)] = row
    return out


def load_all_causal() -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for phase in ["phase275", "phase277", "phase279"]:
        for row in read_jsonl(V2 / f"{phase}_causal_fill_rows.jsonl"):
            out[key(row)].append(row)
    return out


def recalibrate(row: dict[str, Any], components: dict[tuple[str, str], dict[str, Any]], causal: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    k = key(row)
    flags = dict(row.get("gap_flags") or {})
    comp = components.get(k)
    crows = causal.get(k, [])
    low_rows = [r for r in crows if r.get("side_effect_level") == "lower"]
    low_supported = any(r.get("causal_effect_supported") and not r.get("side_effect_risk") for r in low_rows)
    any_supported = any(r.get("causal_effect_supported") for r in crows)
    remaining = dict(flags)
    if comp:
        remaining["need_component_path"] = False
        remaining["need_layer_path"] = False
    if crows:
        remaining["need_causal_audit"] = False
    if low_supported:
        remaining["good_readout_low_causal"] = False
    dimensions = [name for name, value in remaining.items() if value and name.startswith("need_")]
    if remaining.get("candidate_not_closed"):
        dimensions.append("candidate_closure_verification")
    priority = safe_float(row.get("priority_score_after_phase275", row.get("priority_score")))
    priority -= 2.0 if comp else 0.0
    priority -= 2.0 if crows else 0.0
    priority -= 1.0 if low_supported else 0.0
    priority = round(max(0.0, priority), 6)
    status = "filled_by_phase275_277_279" if not dimensions else ("partially_filled_by_phase275_277_279" if comp or crows else "still_open")
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase280_created_at": now(),
        "phase280_resolution": {
            "component_path_filled": bool(comp),
            "causal_audit_filled": bool(crows),
            "low_side_effect_causal_supported": bool(low_supported),
            "any_causal_effect_supported": bool(any_supported),
        },
        "remaining_gap_flags": remaining,
        "remaining_dimensions": dimensions,
        "priority_score_after_phase279": priority,
        "phase280_status": status,
    }


def select_next(rows: list[dict[str, Any]], max_total: int = 54) -> list[dict[str, Any]]:
    open_rows = [r for r in rows if r.get("phase280_status") != "filled_by_phase275_277_279"]
    open_rows.sort(key=lambda r: (-safe_float(r.get("priority_score_after_phase279")), str(r.get("family_id")), str(r.get("model")), str(r.get("case_id"))))
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
        row["phase280_next_batch_rank"] = rank
    return selected


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files.update(
        {
            "phase280_recalibrated_gap_rows": "phase280_recalibrated_gap_rows.jsonl",
            "phase280_next_batch_rows": "phase280_next_batch_rows.jsonl",
            "phase280_summary": "phase280_summary.json",
            "phase280_report": "phase280_report.md",
        }
    )
    manifest["latest_gap_recalibration_phase"] = "Phase280"
    manifest["phase280_summary"] = summary
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase280_summary.json", "phase280_next_batch_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase280_summary_ref"] = "phase280_summary.json"
    client["phase280_next_batch_ref"] = "phase280_next_batch_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase280_recalibrated_gap_rows"] = "Phase274 gaps recalibrated after Phase275, Phase277, and Phase279 fills"
    tables["phase280_next_batch_rows"] = "next queue after three physical-path fill batches"
    write_json(V2 / "schema.json", schema)


def write_report(summary: dict[str, Any], next_batch: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase280 Gap Recalibration After Phase279",
        "",
        f"- source_gap_rows: {summary['source_gap_rows']}",
        f"- total_component_summary_rows: {summary['total_component_summary_rows']}",
        f"- total_causal_fill_rows: {summary['total_causal_fill_rows']}",
        f"- filled: {summary['phase280_status_counts'].get('filled_by_phase275_277_279', 0)}",
        f"- partially_filled: {summary['phase280_status_counts'].get('partially_filled_by_phase275_277_279', 0)}",
        f"- still_open: {summary['phase280_status_counts'].get('still_open', 0)}",
        f"- next_batch_rows: {summary['next_batch_rows']}",
        "",
        "## Next Batch Preview",
        "",
    ]
    for row in next_batch[:12]:
        lines.append(
            f"- rank={row.get('phase280_next_batch_rank')} {row.get('model')} {row.get('family_id')} {row.get('case_id')} "
            f"status={row.get('phase280_status')} priority={row.get('priority_score_after_phase279')} remaining={row.get('remaining_dimensions')}"
        )
    text = "\n".join(lines) + "\n"
    (V2 / "phase280_report.md").write_text(text, encoding="utf-8")
    (OUT / "phase280_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    source = read_jsonl(V2 / "phase274_gap_rows.jsonl")
    components = load_all_components()
    causal = load_all_causal()
    recalibrated = [recalibrate(row, components, causal) for row in source]
    next_batch = select_next(recalibrated, max_total=54)
    status_counts = Counter(str(r.get("phase280_status")) for r in recalibrated)
    remaining_counts = Counter()
    for row in recalibrated:
        for name, value in (row.get("remaining_gap_flags") or {}).items():
            if value:
                remaining_counts[name] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase280",
        "created_at": now(),
        "source_gap_rows": len(source),
        "total_component_summary_rows": len(components),
        "total_causal_fill_rows": sum(len(v) for v in causal.values()),
        "phase280_status_counts": dict(status_counts),
        "remaining_gap_counts": dict(remaining_counts),
        "next_batch_rows": len(next_batch),
        "next_batch_by_model": dict(Counter(str(r.get("model")) for r in next_batch)),
        "next_batch_by_family": dict(Counter(str(r.get("family_id")) for r in next_batch)),
        "progress_estimate": {
            "pattern_family_atlas": 0.57,
            "physical_distribution_puzzle": 0.56,
            "component_path_coverage": 0.40,
            "causal_audit_coverage": 0.29,
            "closure": 0.20,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase280_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(OUT / "phase280_next_batch_rows.jsonl", next_batch)
    write_json(OUT / "phase280_summary.json", summary)
    write_jsonl(V2 / "phase280_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(V2 / "phase280_next_batch_rows.jsonl", next_batch)
    write_json(V2 / "phase280_summary.json", summary)
    update_v2(summary)
    write_report(summary, next_batch)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
