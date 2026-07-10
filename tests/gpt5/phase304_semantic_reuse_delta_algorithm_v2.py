#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "tests/result/pattern_family_atlas/v2"
PHASE = "Phase304"
SCHEMA_VERSION = "2.31.0"


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


def clamp(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def main() -> None:
    reuse = read_jsonl(V2 / "phase303_semantic_reuse_matrix_rows.jsonl")
    delta = {str(r.get("delta_id")).replace("phase303:delta:", ""): r for r in read_jsonl(V2 / "phase303_semantic_delta_matrix_rows.jsonl")}
    objects = {str(r.get("object_id")): r for r in read_jsonl(V2 / "phase303_semantic_object_summary_rows.jsonl")}
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in reuse:
        left = str(row.get("left_object_id"))
        right = str(row.get("right_object_id"))
        key = f"{left}:{right}"
        l_obj = objects.get(left, {})
        r_obj = objects.get(right, {})
        theoretical = safe_float(row.get("theoretical_reuse_score"))
        measured = safe_float(row.get("measured_reuse_score"))
        combined_old = safe_float(row.get("combined_reuse_score"))
        left_success = safe_float(l_obj.get("attribute_success_rate"))
        right_success = safe_float(r_obj.get("attribute_success_rate"))
        evidence_quality = (left_success + right_success) / 2.0
        same_category = row.get("left_category_id") == row.get("right_category_id")
        same_subclass = row.get("left_subclass_id") == row.get("right_subclass_id")
        category_bonus = 0.10 if same_category else -0.10
        subclass_bonus = 0.12 if same_subclass else 0.0
        measured_weight = 0.20 + 0.25 * evidence_quality
        theoretical_weight = 1.0 - measured_weight
        corrected_reuse = clamp(theoretical_weight * theoretical + measured_weight * measured + category_bonus + subclass_bonus)
        corrected_delta = clamp(1.0 - corrected_reuse)
        false_high = bool(combined_old >= 0.55 and corrected_reuse < 0.55)
        likely_shared_backbone = bool(corrected_reuse >= 0.55 and same_category)
        semantic_relation = (
            "subclass_shared_backbone"
            if same_subclass and corrected_reuse >= 0.55
            else "category_shared_backbone"
            if same_category and corrected_reuse >= 0.45
            else "contrast_control"
            if not same_category and corrected_delta >= 0.60
            else "ambiguous_or_needs_more_evidence"
        )
        out = {
            **row,
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "reuse_v2_id": f"phase304:reuse_v2:{left}:{right}",
            "left_attribute_success_rate": left_success,
            "right_attribute_success_rate": right_success,
            "evidence_quality": round(evidence_quality, 6),
            "same_category": same_category,
            "same_subclass": same_subclass,
            "measured_weight_v2": round(measured_weight, 6),
            "theoretical_weight_v2": round(theoretical_weight, 6),
            "combined_reuse_score_v1": combined_old,
            "corrected_reuse_score_v2": corrected_reuse,
            "corrected_delta_score_v2": corrected_delta,
            "false_high_reuse_flag": false_high,
            "likely_shared_backbone": likely_shared_backbone,
            "semantic_relation_v2": semantic_relation,
        }
        rows.append(out)
        if false_high or semantic_relation == "ambiguous_or_needs_more_evidence":
            d = delta.get(key, {})
            audits.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "audit_id": f"phase304:audit:{left}:{right}",
                    "left_object_id": left,
                    "right_object_id": right,
                    "old_reuse_score": combined_old,
                    "corrected_reuse_score_v2": corrected_reuse,
                    "corrected_delta_score_v2": corrected_delta,
                    "old_delta_score": d.get("delta_score"),
                    "reason": "false_high_reuse_from_low_quality_measured_profile" if false_high else "ambiguous_relation_needs_more_evidence",
                    "recommended_next": "add richer aliases or component-path probe before treating as shared backbone",
                }
            )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "created_at": now(),
        "status": "complete",
        "input_reuse_rows": len(reuse),
        "corrected_reuse_rows": len(rows),
        "audit_rows": len(audits),
        "mean_reuse_v1": mean_safe([safe_float(r.get("combined_reuse_score_v1")) for r in rows]),
        "mean_corrected_reuse_v2": mean_safe([safe_float(r.get("corrected_reuse_score_v2")) for r in rows]),
        "mean_corrected_delta_v2": mean_safe([safe_float(r.get("corrected_delta_score_v2")) for r in rows]),
        "false_high_reuse_count": sum(1 for r in rows if r.get("false_high_reuse_flag")),
        "likely_shared_backbone_count": sum(1 for r in rows if r.get("likely_shared_backbone")),
        "semantic_relation_counts": dict(Counter(str(r.get("semantic_relation_v2")) for r in rows)),
        "progress": {
            "language_pattern_family_atlas": 0.81,
            "semantic_reuse_delta_subatlas": 0.38,
            "sample_type_coverage": 0.72,
            "large_data_feature_mining": 0.71,
            "physical_distribution_puzzle": 0.75,
            "mechanism_causal_audit": 0.52,
            "closure": 0.21,
        },
    }
    write_jsonl(V2 / "phase304_semantic_reuse_matrix_v2_rows.jsonl", rows)
    write_jsonl(V2 / "phase304_semantic_reuse_false_high_audit_rows.jsonl", audits)
    write_json(V2 / "phase304_semantic_reuse_delta_algorithm_v2_summary.json", summary)
    write_json(V2 / "progress.json", {**read_json(V2 / "progress.json"), **summary["progress"], "last_phase": PHASE, "updated_at": now()})
    update_manifest(summary)
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def update_manifest(summary: dict[str, Any]) -> None:
    path = V2 / "manifest.json"
    manifest = read_json(path)
    manifest.setdefault("generated_files", [])
    for name in [
        "phase304_semantic_reuse_matrix_v2_rows.jsonl",
        "phase304_semantic_reuse_false_high_audit_rows.jsonl",
        "phase304_semantic_reuse_delta_algorithm_v2_summary.json",
    ]:
        if name not in manifest["generated_files"]:
            manifest["generated_files"].append(name)
    manifest["last_phase"] = PHASE
    manifest["updated_at"] = now()
    manifest["phase304_summary"] = {
        "corrected_reuse_rows": summary["corrected_reuse_rows"],
        "false_high_reuse_count": summary["false_high_reuse_count"],
        "likely_shared_backbone_count": summary["likely_shared_backbone_count"],
    }
    write_json(path, manifest)


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase304 Semantic Reuse-Delta Algorithm v2",
        "",
        f"- corrected_reuse_rows: {summary['corrected_reuse_rows']}",
        f"- audit_rows: {summary['audit_rows']}",
        f"- mean_reuse_v1: {summary['mean_reuse_v1']}",
        f"- mean_corrected_reuse_v2: {summary['mean_corrected_reuse_v2']}",
        f"- false_high_reuse_count: {summary['false_high_reuse_count']}",
        f"- likely_shared_backbone_count: {summary['likely_shared_backbone_count']}",
        f"- semantic_relation_counts: {json.dumps(summary['semantic_relation_counts'], ensure_ascii=False)}",
        "",
        "v2 downweights coarse measured agreement when object-level semantic answer quality is low.",
    ]
    (V2 / "phase304_semantic_reuse_delta_algorithm_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
