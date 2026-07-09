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

PHASE = 274
SCHEMA_VERSION = "2.1.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
SCORE_FIELDS = ["behavior", "readout", "layer_path", "component_path", "causal", "rollout", "closure", "overall"]

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase274_pattern_family_atlas_v2_gap_queue"


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


def gap_flags(row: dict[str, Any]) -> dict[str, bool]:
    scores = row.get("scores", {})
    sig = row.get("path_signature", {})
    dominant_layers = sig.get("dominant_layers") or []
    return {
        "need_component_path": safe_float(scores.get("component_path")) < 0.08 or not dominant_layers,
        "need_causal_audit": safe_float(scores.get("causal")) < 0.08,
        "need_closure_quality": safe_float(scores.get("closure")) < 0.35,
        "need_layer_path": safe_float(scores.get("layer_path")) < 0.12,
        "need_readout_competition": safe_float(scores.get("readout")) < 0.22,
        "candidate_not_closed": row.get("status") in {"high_quality_candidate_not_closed", "path_candidate_not_closed"},
        "good_behavior_low_path": safe_float(scores.get("behavior")) >= 0.66 and safe_float(scores.get("component_path")) < 0.08,
        "good_readout_low_causal": safe_float(scores.get("readout")) >= 0.35 and safe_float(scores.get("causal")) < 0.08,
    }


def priority(row: dict[str, Any], flags: dict[str, bool], family_pressure: float) -> float:
    scores = row.get("scores", {})
    status = row.get("status")
    score = 0.0
    score += 6.0 if status == "high_quality_candidate_not_closed" else 0.0
    score += 4.0 if status == "path_candidate_not_closed" else 0.0
    score += 2.0 * safe_float(scores.get("overall"))
    score += 1.4 * safe_float(scores.get("behavior"))
    score += 1.2 * safe_float(scores.get("readout"))
    score += 0.8 * safe_float(scores.get("rollout"))
    score += 0.8 * safe_float(scores.get("closure"))
    score += 0.7 * family_pressure
    score += sum(0.35 for v in flags.values() if v)
    if flags["good_behavior_low_path"]:
        score += 1.0
    if flags["good_readout_low_causal"]:
        score += 1.0
    if row.get("variant_id") in {"base", "answer_only", "structured_json"}:
        score += 0.25
    return round(score, 6)


def batch_kind(flags: dict[str, bool]) -> str:
    if flags["candidate_not_closed"]:
        return "candidate_closure_path_fill"
    if flags["good_behavior_low_path"] or flags["good_readout_low_causal"]:
        return "high_signal_missing_mechanism"
    if flags["need_component_path"] and flags["need_causal_audit"]:
        return "component_and_causal_gap"
    if flags["need_component_path"]:
        return "component_path_gap"
    if flags["need_causal_audit"]:
        return "causal_audit_gap"
    return "coverage_balance"


def make_gap_rows(path_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        grouped[(str(row.get("family_id")), str(row.get("model")))].append(row)

    family_pressure: dict[tuple[str, str], float] = {}
    for key, rows in grouped.items():
        component = mean_safe([safe_float(r.get("scores", {}).get("component_path")) for r in rows])
        causal = mean_safe([safe_float(r.get("scores", {}).get("causal")) for r in rows])
        closure = mean_safe([safe_float(r.get("scores", {}).get("closure")) for r in rows])
        family_pressure[key] = round((1.0 - component) * 0.45 + (1.0 - causal) * 0.45 + (1.0 - closure) * 0.10, 6)

    gap_rows: list[dict[str, Any]] = []
    for row in path_rows:
        flags = gap_flags(row)
        key = (str(row.get("family_id")), str(row.get("model")))
        p = priority(row, flags, family_pressure[key])
        missing = [name for name, value in flags.items() if value and name.startswith("need_")]
        if flags["candidate_not_closed"]:
            missing.append("candidate_closure_verification")
        gap_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "gap_id": f"phase274:gap:{row.get('model')}:{row.get('case_id')}",
                "source_signature_id": row.get("signature_id"),
                "case_id": row.get("case_id"),
                "model": row.get("model"),
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row.get("variant_id"),
                "status": row.get("status"),
                "detail_ref": row.get("detail_ref"),
                "scores": row.get("scores", {}),
                "gap_flags": flags,
                "missing_dimensions": missing,
                "batch_kind": batch_kind(flags),
                "priority_score": p,
                "family_model_pressure": family_pressure[key],
                "recommended_next_test": {
                    "component_path": "layerwise attention/mlp/residual contribution sweep" if flags["need_component_path"] else "skip",
                    "causal": "low-side-effect single/window MLP direction audit with random same-norm control" if flags["need_causal_audit"] else "skip",
                    "closure_quality": "candidate closure verification plus strict protocol and answer integrity recheck" if flags["candidate_not_closed"] else ("strict protocol and answer integrity recheck" if flags["need_closure_quality"] else "skip"),
                },
            }
        )
    gap_rows.sort(key=lambda r: (-safe_float(r["priority_score"]), str(r["family_id"]), str(r["model"]), str(r["case_id"])))
    return gap_rows, family_pressure


def select_batch(gap_rows: list[dict[str, Any]], per_family_model: int, max_total: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    kind_counts: Counter[str] = Counter()
    mandatory_status = {"high_quality_candidate_not_closed", "path_candidate_not_closed"}

    for row in gap_rows:
        key = (str(row["family_id"]), str(row["model"]))
        if row["status"] in mandatory_status and counts[key] < per_family_model + 2:
            selected.append(row)
            counts[key] += 1
            kind_counts[str(row["batch_kind"])] += 1

    for row in gap_rows:
        if row in selected:
            continue
        key = (str(row["family_id"]), str(row["model"]))
        if counts[key] >= per_family_model or len(selected) >= max_total:
            continue
        # Keep the first batch diverse enough to expose different failure sources.
        if kind_counts[str(row["batch_kind"])] > max(8, max_total // 4) and row["batch_kind"] not in {"candidate_closure_path_fill", "high_signal_missing_mechanism"}:
            continue
        selected.append(row)
        counts[key] += 1
        kind_counts[str(row["batch_kind"])] += 1
        if len(selected) >= max_total:
            break

    for rank, row in enumerate(selected, start=1):
        row["batch_rank"] = rank
        row["phase274_selected"] = True
    return selected


def coverage_rows(path_rows: list[dict[str, Any]], gap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped_paths: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_gaps: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in path_rows:
        grouped_paths[(str(row.get("family_id")), str(row.get("model")))].append(row)
    for row in gap_rows:
        grouped_gaps[(str(row.get("family_id")), str(row.get("model")))].append(row)

    rows: list[dict[str, Any]] = []
    for key in sorted(grouped_paths):
        family, model = key
        paths = grouped_paths[key]
        gaps = grouped_gaps[key]
        scores = {name: mean_safe([safe_float(r.get("scores", {}).get(name)) for r in paths]) for name in SCORE_FIELDS}
        flags = Counter()
        for gap in gaps:
            for name, value in gap["gap_flags"].items():
                if value:
                    flags[name] += 1
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "coverage_id": f"phase274:coverage:{family}:{model}",
                "family_id": family,
                "model": model,
                "case_count": len(paths),
                "scores": scores,
                "status_counts": dict(Counter(str(r.get("status")) for r in paths)),
                "gap_counts": dict(flags),
                "physical_distribution_progress": round((scores["layer_path"] + scores["component_path"] + scores["readout"]) / 3.0, 6),
                "closure_readiness": round((scores["component_path"] + scores["causal"] + scores["closure"]) / 3.0, 6),
                "priority_pressure": round((1.0 - scores["component_path"]) * 0.45 + (1.0 - scores["causal"]) * 0.45 + (1.0 - scores["closure"]) * 0.10, 6),
            }
        )
    rows.sort(key=lambda r: (-safe_float(r["priority_pressure"]), str(r["family_id"]), str(r["model"])))
    return rows


def update_v2_index(summary: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    client_index = read_json(V2 / "client_index.json")
    schema = read_json(V2 / "schema.json")

    files = manifest.setdefault("files", {})
    files.update(
        {
            "phase274_gap_rows": "phase274_gap_rows.jsonl",
            "phase274_selected_batch_rows": "phase274_selected_batch_rows.jsonl",
            "phase274_coverage_matrix_rows": "phase274_coverage_matrix_rows.jsonl",
            "phase274_gap_summary": "phase274_gap_summary.json",
            "phase274_gap_report": "phase274_gap_report.md",
        }
    )
    manifest["latest_gap_phase"] = "Phase274"
    manifest["phase274_gap_summary"] = summary

    views = client_index.setdefault("views", [])
    for view in ["gap_matrix", "gap_queue", "batch_planner"]:
        if view not in views:
            views.append(view)
    initial_files = client_index.setdefault("initial_files", [])
    for item in ["phase274_gap_summary.json", "phase274_coverage_matrix_rows.jsonl", "phase274_selected_batch_rows.jsonl"]:
        if item not in initial_files:
            initial_files.append(item)
    client_index["phase274_gap_summary"] = summary
    client_index["gap_queue_ref"] = "phase274_gap_rows.jsonl"
    client_index["selected_gap_batch_ref"] = "phase274_selected_batch_rows.jsonl"
    client_index["coverage_matrix_ref"] = "phase274_coverage_matrix_rows.jsonl"

    tables = schema.setdefault("tables", {})
    tables["phase274_gap_rows"] = "one row per v2 signature gap with objective missing dimensions and priority"
    tables["phase274_selected_batch_rows"] = "first batch queue for physical path and low-side-effect causal fill"
    tables["phase274_coverage_matrix_rows"] = "family x model gap pressure and progress table"
    status_values = schema.setdefault("status_values", [])
    if "gap_queue_only" not in status_values:
        status_values.append("gap_queue_only")

    write_json(V2 / "manifest.json", manifest)
    write_json(V2 / "client_index.json", client_index)
    write_json(V2 / "schema.json", schema)


def write_report(summary: dict[str, Any], coverage: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    top_cov = coverage[:8]
    top_sel = selected[:12]
    lines = [
        "# Phase274 Pattern Family Atlas v2 Gap Queue",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- source_path_signatures: {summary['source_path_signatures']}",
        f"- gap_rows: {summary['gap_rows']}",
        f"- selected_batch_rows: {summary['selected_batch_rows']}",
        f"- model_test_status: {summary['model_test_status']}",
        "",
        "This phase does not claim new causal evidence. It turns Phase273 v2 into an explicit gap matrix and first batch queue for physical path completion.",
        "",
        "## Highest Pressure Family-Model Cells",
        "",
    ]
    for row in top_cov:
        lines.append(
            f"- {row['family_id']} / {row['model']}: pressure={row['priority_pressure']}, "
            f"physical={row['physical_distribution_progress']}, closure_readiness={row['closure_readiness']}, gaps={row['gap_counts']}"
        )
    lines.extend(["", "## First Batch Queue", ""])
    for row in top_sel:
        lines.append(
            f"- rank={row['batch_rank']} {row['model']} {row['family_id']} {row['case_id']} "
            f"kind={row['batch_kind']} priority={row['priority_score']} missing={row['missing_dimensions']}"
        )
    lines.extend(
        [
            "",
            "## Caution",
            "",
            "Scores are coverage and prioritization signals. They are not closure proof. Small local models may have coarse internal structure, so selected gaps should be retested across qwen3, GLM4, and DS7B before theoretical claims are upgraded.",
        ]
    )
    (V2 / "phase274_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "phase274_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    path_rows = read_jsonl(V2 / "path_signature_rows.jsonl")
    if not path_rows:
        raise SystemExit(f"missing v2 path_signature_rows: {V2}")

    gap_rows, _ = make_gap_rows(path_rows)
    selected = select_batch(gap_rows, per_family_model=2, max_total=72)
    coverage = coverage_rows(path_rows, gap_rows)
    generated_at = now()
    flag_counts = Counter()
    kind_counts = Counter()
    for row in gap_rows:
        kind_counts[str(row["batch_kind"])] += 1
        for name, value in row["gap_flags"].items():
            if value:
                flag_counts[name] += 1
    selected_by_model = Counter(str(row["model"]) for row in selected)
    selected_by_family = Counter(str(row["family_id"]) for row in selected)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase274",
        "generated_at": generated_at,
        "model_test_status": "not_run_gap_queue_only",
        "source_path_signatures": len(path_rows),
        "gap_rows": len(gap_rows),
        "selected_batch_rows": len(selected),
        "status_counts": dict(Counter(str(r.get("status")) for r in path_rows)),
        "gap_flag_counts": dict(flag_counts),
        "batch_kind_counts": dict(kind_counts),
        "selected_by_model": dict(selected_by_model),
        "selected_by_family": dict(selected_by_family),
        "top_pressure_cells": coverage[:8],
        "priority_rule": "candidate statuses first, then high behavior/readout with missing component/causal, then family-model pressure balancing",
        "next_executable_phase": "Phase275 should run selected_batch_rows sequentially on qwen3, GLM4, and DS7B for component path and low-side-effect causal fill.",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "gap_rows.jsonl", gap_rows)
    write_jsonl(OUT / "selected_batch_rows.jsonl", selected)
    write_jsonl(OUT / "coverage_matrix_rows.jsonl", coverage)
    write_json(OUT / "phase274_gap_summary.json", summary)

    write_jsonl(V2 / "phase274_gap_rows.jsonl", gap_rows)
    write_jsonl(V2 / "phase274_selected_batch_rows.jsonl", selected)
    write_jsonl(V2 / "phase274_coverage_matrix_rows.jsonl", coverage)
    write_json(V2 / "phase274_gap_summary.json", summary)
    update_v2_index(summary)
    write_report(summary, coverage, selected)

    print(
        json.dumps(
            {
                "phase": PHASE,
                "status": "complete",
                "model_test_status": summary["model_test_status"],
                "source_path_signatures": len(path_rows),
                "gap_rows": len(gap_rows),
                "selected_batch_rows": len(selected),
                "selected_by_model": dict(selected_by_model),
                "top_pressure_cells": [
                    {
                        "family_id": r["family_id"],
                        "model": r["model"],
                        "priority_pressure": r["priority_pressure"],
                        "physical_distribution_progress": r["physical_distribution_progress"],
                        "closure_readiness": r["closure_readiness"],
                    }
                    for r in coverage[:5]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
