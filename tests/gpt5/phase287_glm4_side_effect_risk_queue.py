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

PHASE = 287
SCHEMA_VERSION = "2.14.0"
ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
OUT = ROOT / "tests/result/phase287_glm4_side_effect_risk_queue"
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


def load_causal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in FILL_PHASES:
        for row in read_jsonl(V2 / f"{phase}_causal_fill_rows.jsonl"):
            if row.get("model") == "glm4":
                row = dict(row)
                row["source_fill_phase"] = phase
                rows.append(row)
    return rows


def load_component_map() -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for phase in FILL_PHASES:
        for row in read_jsonl(V2 / f"{phase}_component_summary_rows.jsonl"):
            if row.get("model") == "glm4":
                out[key(row)] = row
    return out


def load_gap_map() -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in read_jsonl(V2 / "phase286_recalibrated_gap_rows.jsonl"):
        if row.get("model") == "glm4":
            out[key(row)] = row
    return out


def classify_risk(row: dict[str, Any], component: dict[str, Any] | None, gap: dict[str, Any] | None) -> tuple[str, list[str], list[str]]:
    reasons: list[str] = []
    tests: list[str] = []
    if row.get("side_effect_risk"):
        reasons.append("side_effect_risk")
    if row.get("winner_changed"):
        reasons.append("winner_changed")
    if safe_float(row.get("delta_target_logit")) < -0.1:
        reasons.append("target_logit_damaged")
    if safe_float(row.get("delta_continue_stop_margin")) > -0.25:
        reasons.append("weak_continue_suppression")
    if component and component.get("dominant_positive_component") == "attention":
        reasons.append("attention_dominant_case")
        tests.append("attention_mlp_joint_audit")
    if gap and (gap.get("remaining_gap_flags") or {}).get("need_readout_competition"):
        reasons.append("readout_competition_gap_open")
        tests.append("channel_level_stop_continue_audit")
    if row.get("side_effect_level") == "lower" and row.get("side_effect_risk"):
        tests.append("source_restricted_low_side_effect_audit")
    if row.get("patch_type") == "mlp_zero_last_token":
        tests.append("subspace_or_mean_replace_audit")
    tests.append("random_same_norm_control")
    if not tests:
        tests = ["manual_review"]
    if "target_logit_damaged" in reasons and "weak_continue_suppression" in reasons:
        bucket = "coupled_target_continue_risk"
    elif "attention_dominant_case" in reasons:
        bucket = "attention_mlp_joint_risk"
    elif "readout_competition_gap_open" in reasons:
        bucket = "readout_competition_risk"
    else:
        bucket = "generic_side_effect_risk"
    return bucket, sorted(set(reasons)), sorted(set(tests))


def main() -> None:
    causal = load_causal_rows()
    component_map = load_component_map()
    gap_map = load_gap_map()
    risk_source = [r for r in causal if r.get("side_effect_risk")]
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(risk_source, start=1):
        k = key(row)
        comp = component_map.get(k)
        gap = gap_map.get(k)
        bucket, reasons, tests = classify_risk(row, comp, gap)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase287",
                "created_at": now(),
                "risk_queue_id": f"phase287:glm4_risk:{idx:04d}:{row.get('case_id')}:{row.get('patch_type')}",
                "model": "glm4",
                "case_id": row.get("case_id"),
                "family_id": row.get("family_id"),
                "mode_id": row.get("mode_id"),
                "variant_id": row.get("variant_id"),
                "source_fill_phase": row.get("source_fill_phase"),
                "source_causal_fill_id": row.get("causal_fill_id"),
                "risk_bucket": bucket,
                "risk_reasons": reasons,
                "recommended_audits": tests,
                "patch_type": row.get("patch_type"),
                "side_effect_level": row.get("side_effect_level"),
                "strongest_mlp_layer": row.get("strongest_mlp_layer_phase275"),
                "base_winner": row.get("base_winner"),
                "patched_winner": row.get("patched_winner"),
                "delta_continue_stop_margin": row.get("delta_continue_stop_margin"),
                "delta_target_logit": row.get("delta_target_logit"),
                "remaining_gap_flags": (gap or {}).get("remaining_gap_flags"),
                "component_dominant": (comp or {}).get("dominant_positive_component"),
                "sum_positive_mlp_delta": (comp or {}).get("sum_positive_mlp_delta"),
                "sum_positive_attn_delta": (comp or {}).get("sum_positive_attn_delta"),
            }
        )
    rows.sort(key=lambda r: (str(r["risk_bucket"]), str(r["family_id"]), str(r["case_id"]), str(r["patch_type"])))
    next_rows = rows[:36]
    for rank, row in enumerate(next_rows, start=1):
        row["phase287_next_audit_rank"] = rank
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": "Phase287",
        "created_at": now(),
        "source_glm4_causal_rows": len(causal),
        "glm4_side_effect_risk_rows": len(rows),
        "next_audit_rows": len(next_rows),
        "risk_bucket_counts": dict(Counter(str(r.get("risk_bucket")) for r in rows)),
        "family_counts": dict(Counter(str(r.get("family_id")) for r in rows)),
        "recommended_audit_counts": dict(Counter(a for r in rows for a in r.get("recommended_audits", []))),
        "patch_type_counts": dict(Counter(str(r.get("patch_type")) for r in rows)),
        "mean_delta_continue_stop_margin": mean_safe([safe_float(r.get("delta_continue_stop_margin")) for r in rows]),
        "mean_delta_target_logit": mean_safe([safe_float(r.get("delta_target_logit")) for r in rows]),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "phase287_glm4_side_effect_risk_rows.jsonl", rows)
    write_jsonl(OUT / "phase287_glm4_next_audit_rows.jsonl", next_rows)
    write_json(OUT / "phase287_summary.json", summary)
    write_jsonl(V2 / "phase287_glm4_side_effect_risk_rows.jsonl", rows)
    write_jsonl(V2 / "phase287_glm4_next_audit_rows.jsonl", next_rows)
    write_json(V2 / "phase287_summary.json", summary)
    report = "\n".join(
        [
            "# Phase287 GLM4 Side-Effect Risk Queue",
            "",
            f"- source_glm4_causal_rows: {summary['source_glm4_causal_rows']}",
            f"- glm4_side_effect_risk_rows: {summary['glm4_side_effect_risk_rows']}",
            f"- next_audit_rows: {summary['next_audit_rows']}",
            f"- risk_bucket_counts: {json.dumps(summary['risk_bucket_counts'], ensure_ascii=False)}",
            f"- recommended_audit_counts: {json.dumps(summary['recommended_audit_counts'], ensure_ascii=False)}",
            "",
            "This phase creates the fixed input queue for source-restricted/channel-level GLM4 audits.",
        ]
    ) + "\n"
    (OUT / "phase287_report.md").write_text(report, encoding="utf-8")
    (V2 / "phase287_report.md").write_text(report, encoding="utf-8")
    manifest = read_json(V2 / "manifest.json")
    files = manifest.setdefault("files", {})
    files["phase287_glm4_side_effect_risk_rows"] = "phase287_glm4_side_effect_risk_rows.jsonl"
    files["phase287_glm4_next_audit_rows"] = "phase287_glm4_next_audit_rows.jsonl"
    files["phase287_summary"] = "phase287_summary.json"
    files["phase287_report"] = "phase287_report.md"
    manifest["latest_glm4_risk_phase"] = "Phase287"
    manifest["phase287_summary"] = summary
    write_json(V2 / "manifest.json", manifest)
    client = read_json(V2 / "client_index.json")
    for item in ["phase287_summary.json", "phase287_glm4_side_effect_risk_rows.jsonl", "phase287_glm4_next_audit_rows.jsonl"]:
        if item not in client.setdefault("initial_files", []):
            client["initial_files"].append(item)
    client["phase287_summary_ref"] = "phase287_summary.json"
    client["phase287_glm4_next_audit_ref"] = "phase287_glm4_next_audit_rows.jsonl"
    write_json(V2 / "client_index.json", client)
    schema = read_json(V2 / "schema.json")
    tables = schema.setdefault("tables", {})
    tables["phase287_glm4_side_effect_risk_rows"] = "GLM4 side-effect risk rows grouped into fine-grained audit buckets"
    tables["phase287_glm4_next_audit_rows"] = "fixed queue for GLM4 source-restricted/channel-level audits"
    write_json(V2 / "schema.json", schema)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
