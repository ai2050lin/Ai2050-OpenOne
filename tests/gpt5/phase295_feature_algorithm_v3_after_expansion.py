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

PHASE = 295
SCHEMA_VERSION = "2.22.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase295_feature_algorithm_v3_after_expansion"


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


def clamp(v: float) -> float:
    return round(max(0.0, min(1.0, v)), 6)


def mean_safe(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def feature_v3_rows() -> list[dict[str, Any]]:
    base = {(str(r["family_id"]), str(r["model"])): r for r in read_jsonl(V2 / "phase292_feature_matrix_v2_rows.jsonl")}
    measured = {(str(r["family_id"]), str(r["model"])): r for r in read_jsonl(V2 / "phase294_expanded_family_model_update_rows.jsonl")}
    rows = []
    for cell, b in sorted(base.items()):
        m = measured.get(cell, {})
        answer = safe_float(m.get("answer_correct_proxy_rate"))
        pattern = safe_float(m.get("pattern_matched_proxy_rate"))
        stop_exec = safe_float(m.get("model_stop_executed_rate"))
        stop_win = safe_float(m.get("stop_winner_rate"))
        continue_win = safe_float(m.get("continue_winner_rate"), 1.0)
        margin = safe_float(m.get("mean_top_continue_vs_stop_margin"))
        old_completion = safe_float(b.get("atlas_completion_v2"))
        measured_score = clamp(0.25 * answer + 0.20 * pattern + 0.20 * stop_exec + 0.20 * stop_win + 0.15 * (1.0 - continue_win))
        readout_penalty = clamp(continue_win * 0.65 + min(max(margin, 0.0), 15.0) / 15.0 * 0.35)
        completion_v3 = clamp(0.55 * old_completion + 0.35 * measured_score + 0.10 * (1.0 - readout_penalty))
        if continue_win >= 0.95 and stop_exec < 0.1:
            priority = "hard_readout_stop_failure"
        elif pattern < 0.3:
            priority = "protocol_pattern_failure"
        elif stop_exec > 0.4 and continue_win >= 0.95:
            priority = "generation_stop_without_readout_stop"
        else:
            priority = "measured_expansion_followup"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase295",
                "created_at": now(),
                "feature_v3_id": f"phase295:feature_v3:{cell[0]}:{cell[1]}",
                "family_id": cell[0],
                "model": cell[1],
                "atlas_completion_v2": old_completion,
                "expanded_answer_correct_proxy_rate": answer,
                "expanded_pattern_matched_proxy_rate": pattern,
                "expanded_model_stop_executed_rate": stop_exec,
                "expanded_continue_winner_rate": continue_win,
                "expanded_stop_winner_rate": stop_win,
                "expanded_mean_top_continue_vs_stop_margin": margin,
                "expanded_measured_score": measured_score,
                "readout_penalty": readout_penalty,
                "atlas_completion_v3": completion_v3,
                "completion_delta_v3_minus_v2": round(completion_v3 - old_completion, 6),
                "next_priority": priority,
                "base_next_priority_v2": b.get("next_priority"),
            }
        )
    rows.sort(key=lambda r: (safe_float(r["atlas_completion_v3"]), str(r["model"]), str(r["family_id"])))
    return rows


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[float]] = {}
    by_family: dict[str, list[float]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(safe_float(row["atlas_completion_v3"]))
        by_family.setdefault(str(row["family_id"]), []).append(safe_float(row["atlas_completion_v3"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase295",
        "created_at": now(),
        "feature_v3_rows": len(rows),
        "mean_atlas_completion_v3": mean_safe([safe_float(r["atlas_completion_v3"]) for r in rows]),
        "mean_completion_delta_v3_minus_v2": mean_safe([safe_float(r["completion_delta_v3_minus_v2"]) for r in rows]),
        "mean_expanded_answer_correct_proxy_rate": mean_safe([safe_float(r["expanded_answer_correct_proxy_rate"]) for r in rows]),
        "mean_expanded_pattern_matched_proxy_rate": mean_safe([safe_float(r["expanded_pattern_matched_proxy_rate"]) for r in rows]),
        "mean_expanded_model_stop_executed_rate": mean_safe([safe_float(r["expanded_model_stop_executed_rate"]) for r in rows]),
        "mean_expanded_continue_winner_rate": mean_safe([safe_float(r["expanded_continue_winner_rate"]) for r in rows]),
        "next_priority_counts": dict(Counter(str(r["next_priority"]) for r in rows)),
        "model_completion_v3": {k: mean_safe(v) for k, v in sorted(by_model.items())},
        "family_completion_v3": {k: mean_safe(v) for k, v in sorted(by_family.items())},
        "progress_estimate": {
            "pattern_family_atlas": 0.75,
            "sample_type_coverage": 0.68,
            "feature_mining": 0.58,
            "physical_distribution_puzzle": 0.67,
            "mechanism_audit": 0.44,
            "closure": 0.21,
        },
    }


def update_v2(payload: dict[str, Any]) -> None:
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files["phase295_feature_matrix_v3_rows"] = "phase295_feature_matrix_v3_rows.jsonl"
    files["phase295_summary"] = "phase295_summary.json"
    files["phase295_report"] = "phase295_report.md"
    manifest["latest_feature_algorithm_phase"] = "Phase295"
    manifest["phase295_summary"] = payload
    write_json(V2 / "manifest.json", manifest)

    client = read_json(V2 / "client_index.json")
    for item in ["phase295_summary.json", "phase295_feature_matrix_v3_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase295_summary_ref"] = "phase295_summary.json"
    write_json(V2 / "client_index.json", client)

    schema = read_json(V2 / "schema.json")
    schema.setdefault("tables", {})["phase295_feature_matrix_v3_rows"] = "feature algorithm v3 after measured expanded sample batch"
    write_json(V2 / "schema.json", schema)


def main() -> None:
    rows = feature_v3_rows()
    payload = summary(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase295_feature_matrix_v3_rows.jsonl", rows)
    write_jsonl(V2 / "phase295_feature_matrix_v3_rows.jsonl", rows)
    write_json(OUT / "phase295_summary.json", payload)
    write_json(V2 / "phase295_summary.json", payload)
    report = "\n".join(
        [
            "# Phase295 Feature Algorithm V3 After Expansion",
            "",
            f"- feature_v3_rows: {payload['feature_v3_rows']}",
            f"- mean_atlas_completion_v3: {payload['mean_atlas_completion_v3']}",
            f"- mean_completion_delta_v3_minus_v2: {payload['mean_completion_delta_v3_minus_v2']}",
            f"- mean_expanded_answer_correct_proxy_rate: {payload['mean_expanded_answer_correct_proxy_rate']}",
            f"- mean_expanded_pattern_matched_proxy_rate: {payload['mean_expanded_pattern_matched_proxy_rate']}",
            f"- mean_expanded_model_stop_executed_rate: {payload['mean_expanded_model_stop_executed_rate']}",
            f"- mean_expanded_continue_winner_rate: {payload['mean_expanded_continue_winner_rate']}",
            f"- next_priority_counts: {json.dumps(payload['next_priority_counts'], ensure_ascii=False)}",
            "",
            "V3 uses measured expanded behavior/readout rows. It still does not include layer/component/causal evidence for the expanded cases.",
        ]
    ) + "\n"
    (OUT / "phase295_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase295_report.md").write_text(report, encoding="utf-8")
    update_v2(payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
