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

PHASE = 276
SCHEMA_VERSION = "2.3.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase276_gap_recalibration_after_phase275"


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


def recalibrate_gap(row: dict[str, Any], component_by_key: dict[tuple[str, str], dict[str, Any]], causal_by_key: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    k = key(row)
    flags = dict(row.get("gap_flags") or {})
    component = component_by_key.get(k)
    causal_rows = causal_by_key.get(k, [])
    low_rows = [r for r in causal_rows if r.get("side_effect_level") == "lower"]
    low_supported = any(r.get("causal_effect_supported") and not r.get("side_effect_risk") for r in low_rows)
    any_supported = any(r.get("causal_effect_supported") for r in causal_rows)

    resolved = {
        "component_path_filled": bool(component),
        "causal_audit_filled": bool(causal_rows),
        "low_side_effect_causal_supported": bool(low_supported),
        "any_causal_effect_supported": bool(any_supported),
    }
    remaining_flags = dict(flags)
    if component:
        remaining_flags["need_component_path"] = False
        remaining_flags["need_layer_path"] = False
    if causal_rows:
        remaining_flags["need_causal_audit"] = False
    if low_supported:
        remaining_flags["good_readout_low_causal"] = False

    remaining_dimensions = [name for name, value in remaining_flags.items() if value and name.startswith("need_")]
    if remaining_flags.get("candidate_not_closed"):
        remaining_dimensions.append("candidate_closure_verification")
    priority_after = safe_float(row.get("priority_score"))
    priority_after -= 2.0 if component else 0.0
    priority_after -= 2.0 if causal_rows else 0.0
    priority_after -= 1.0 if low_supported else 0.0
    priority_after = round(max(0.0, priority_after), 6)
    return {
        **row,
        "schema_version": SCHEMA_VERSION,
        "phase276_recalibrated": True,
        "phase276_created_at": now(),
        "phase275_resolution": resolved,
        "remaining_gap_flags": remaining_flags,
        "remaining_dimensions": remaining_dimensions,
        "priority_score_after_phase275": priority_after,
        "phase276_status": "filled_by_phase275" if not remaining_dimensions else ("partially_filled_by_phase275" if component or causal_rows else "still_open"),
    }


def select_next_queue(rows: list[dict[str, Any]], max_total: int = 54) -> list[dict[str, Any]]:
    open_rows = [r for r in rows if r.get("phase276_status") != "filled_by_phase275"]
    open_rows.sort(key=lambda r: (-safe_float(r.get("priority_score_after_phase275")), str(r.get("family_id")), str(r.get("model")), str(r.get("case_id"))))
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
        row["phase276_next_batch_rank"] = rank
    return selected


def coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("family_id")), str(row.get("model")))].append(row)
    out: list[dict[str, Any]] = []
    for (family, model), vals in sorted(grouped.items()):
        statuses = Counter(str(v.get("phase276_status")) for v in vals)
        remaining = Counter()
        resolved = Counter()
        for v in vals:
            for name, value in (v.get("remaining_gap_flags") or {}).items():
                if value:
                    remaining[name] += 1
            for name, value in (v.get("phase275_resolution") or {}).items():
                if value:
                    resolved[name] += 1
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "coverage_id": f"phase276:coverage:{family}:{model}",
                "family_id": family,
                "model": model,
                "rows": len(vals),
                "status_counts": dict(statuses),
                "remaining_gap_counts": dict(remaining),
                "resolved_counts": dict(resolved),
                "mean_priority_after_phase275": mean_safe([safe_float(v.get("priority_score_after_phase275")) for v in vals]),
            }
        )
    out.sort(key=lambda r: (-safe_float(r["mean_priority_after_phase275"]), str(r["family_id"]), str(r["model"])))
    return out


def update_v2(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files.update(
        {
            "phase276_recalibrated_gap_rows": "phase276_recalibrated_gap_rows.jsonl",
            "phase276_next_batch_rows": "phase276_next_batch_rows.jsonl",
            "phase276_coverage_rows": "phase276_coverage_rows.jsonl",
            "phase276_summary": "phase276_summary.json",
            "phase276_report": "phase276_report.md",
        }
    )
    manifest["latest_gap_recalibration_phase"] = "Phase276"
    manifest["phase276_summary"] = summary
    write_json(V2 / "manifest.json", manifest)

    client = read_json(V2 / "client_index.json")
    for view in ["recalibrated_gaps", "next_gap_batch"]:
        if view not in client.setdefault("views", []):
            client["views"].append(view)
    for item in ["phase276_summary.json", "phase276_coverage_rows.jsonl", "phase276_next_batch_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase276_summary_ref"] = "phase276_summary.json"
    client["phase276_next_batch_ref"] = "phase276_next_batch_rows.jsonl"
    write_json(V2 / "client_index.json", client)

    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase276_recalibrated_gap_rows"] = "Phase274 gaps with Phase275 fill resolution and remaining dimensions"
    tables["phase276_next_batch_rows"] = "next queue after Phase275 physical-path fills"
    tables["phase276_coverage_rows"] = "family-model remaining gap coverage after Phase275"
    write_json(V2 / "schema.json", schema)


def write_report(summary: dict[str, Any], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Phase276 Gap Recalibration After Phase275",
        "",
        f"- source_gap_rows: {summary['source_gap_rows']}",
        f"- phase275_component_summary_rows: {summary['phase275_component_summary_rows']}",
        f"- phase275_causal_fill_rows: {summary['phase275_causal_fill_rows']}",
        f"- filled_by_phase275: {summary['phase276_status_counts'].get('filled_by_phase275', 0)}",
        f"- partially_filled_by_phase275: {summary['phase276_status_counts'].get('partially_filled_by_phase275', 0)}",
        f"- still_open: {summary['phase276_status_counts'].get('still_open', 0)}",
        f"- next_batch_rows: {summary['next_batch_rows']}",
        "",
        "## Next Highest Priority Rows",
        "",
    ]
    for row in top_rows[:12]:
        lines.append(
            f"- rank={row.get('phase276_next_batch_rank')} {row.get('model')} {row.get('family_id')} {row.get('case_id')} "
            f"status={row.get('phase276_status')} priority={row.get('priority_score_after_phase275')} remaining={row.get('remaining_dimensions')}"
        )
    text = "\n".join(lines) + "\n"
    (V2 / "phase276_report.md").write_text(text, encoding="utf-8")
    (OUT / "phase276_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    gaps = read_jsonl(V2 / "phase274_gap_rows.jsonl")
    components = read_jsonl(V2 / "phase275_component_summary_rows.jsonl")
    causal_rows = read_jsonl(V2 / "phase275_causal_fill_rows.jsonl")
    component_by_key = {key(r): r for r in components}
    causal_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in causal_rows:
        causal_by_key[key(row)].append(row)
    recalibrated = [recalibrate_gap(row, component_by_key, causal_by_key) for row in gaps]
    next_batch = select_next_queue(recalibrated, max_total=54)
    coverage_rows = coverage(recalibrated)
    status_counts = Counter(str(r.get("phase276_status")) for r in recalibrated)
    remaining_counts = Counter()
    for row in recalibrated:
        for name, value in (row.get("remaining_gap_flags") or {}).items():
            if value:
                remaining_counts[name] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase276",
        "created_at": now(),
        "source_gap_rows": len(gaps),
        "phase275_component_summary_rows": len(components),
        "phase275_causal_fill_rows": len(causal_rows),
        "phase276_status_counts": dict(status_counts),
        "remaining_gap_counts": dict(remaining_counts),
        "next_batch_rows": len(next_batch),
        "next_batch_by_model": dict(Counter(str(r.get("model")) for r in next_batch)),
        "next_batch_by_family": dict(Counter(str(r.get("family_id")) for r in next_batch)),
        "progress_estimate": {
            "pattern_family_atlas": 0.53,
            "physical_distribution_puzzle": 0.50,
            "component_path_coverage": 0.34,
            "causal_audit_coverage": 0.23,
            "closure": 0.19,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase276_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(OUT / "phase276_next_batch_rows.jsonl", next_batch)
    write_jsonl(OUT / "phase276_coverage_rows.jsonl", coverage_rows)
    write_json(OUT / "phase276_summary.json", summary)
    write_jsonl(V2 / "phase276_recalibrated_gap_rows.jsonl", recalibrated)
    write_jsonl(V2 / "phase276_next_batch_rows.jsonl", next_batch)
    write_jsonl(V2 / "phase276_coverage_rows.jsonl", coverage_rows)
    write_json(V2 / "phase276_summary.json", summary)
    update_v2(summary)
    write_report(summary, next_batch)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
