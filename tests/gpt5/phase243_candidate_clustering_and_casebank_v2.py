#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PHASE = 243
SOURCE_PHASE = 242
SCHEMA_VERSION = "1.0.0"
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE242_ROOT = Path("tests/result/phase242_negative_multilabel_and_trace_selection")
RESULT_ROOT = Path("tests/result/phase243_candidate_clustering_and_casebank_v2")
ROUND_DEFAULT = "candidate_clustering_and_casebank_v2"

TRACE_TARGET_COUNTS = {
    "readout_competitor_trace": 40,
    "protocol_gate_product_residual_trace": 25,
    "stepwise_rollout_trace": 20,
    "rollout_closure_trace": 10,
    "cross_model_structure_comparison": 5,
}


def utc_now() -> str:
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


def append_unique_jsonl(path: Path, new_rows: list[dict[str, Any]], id_key: str) -> None:
    old_rows = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in old_rows + new_rows:
        key = str(row.get(id_key) or row.get("metric_id") or row.get("edge_id") or row.get("observation_id") or row.get("case_id") or len(by_id))
        by_id[key] = row
    write_jsonl(path, list(by_id.values()))


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def split_name(key: str) -> str:
    bucket = stable_hash(key) % 10
    if bucket < 6:
        return "explore"
    if bucket < 8:
        return "validate"
    return "frozen"


def load_phase242(round_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = PHASE242_ROOT / round_name
    candidates = read_jsonl(root / "phase242_high_value_hook_candidates.jsonl")
    calibration = read_jsonl(root / "phase242_case_bank_calibration_rows.jsonl")
    multilabel = read_jsonl(root / "phase242_multilabel_negative_rows.jsonl")
    summary = read_json(root / "phase242_summary.json")
    if not candidates:
        raise FileNotFoundError(f"missing candidates under {root}")
    return candidates, calibration, multilabel, summary


def cluster_key(candidate: dict[str, Any]) -> str:
    reasons = set(candidate.get("selection_reasons") or [])
    if "stable_readout_competitor" in reasons:
        failure_group = "stable_readout"
    elif "semantic_correct_protocol_failure" in reasons or "high_target_pressure_protocol_failure" in reasons:
        failure_group = "protocol_failure"
    elif "semantic_correct_rollout_failure" in reasons:
        failure_group = "rollout_failure"
    elif "semantic_correct_closure_failure" in reasons:
        failure_group = "closure_failure"
    elif "cross_model_divergence" in reasons:
        failure_group = "cross_model_divergence"
    else:
        failure_group = "mixed"
    margin = safe_float(candidate.get("mean_target_margin_vs_winner"))
    if margin >= 1.0:
        margin_bucket = "margin_positive"
    elif margin >= -1.0:
        margin_bucket = "margin_near"
    else:
        margin_bucket = "margin_negative"
    return "::".join(
        [
            str(candidate.get("family_id")),
            str(candidate.get("mode_id")),
            str(candidate.get("recommended_next_test")),
            str(candidate.get("stable_winner_regime")),
            failure_group,
            margin_bucket,
        ]
    )


def dedup_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        key = "::".join(
            [
                str(cand.get("case_id")),
                str(cand.get("variant_id")),
                str(cand.get("recommended_next_test")),
            ]
        )
        buckets[key].append(cand)
    rows = []
    now = utc_now()
    for key, items in buckets.items():
        items.sort(key=lambda x: (safe_float(x.get("candidate_score")), -safe_float(x.get("scoring_risk_rate"))), reverse=True)
        best = dict(items[0])
        best.update(
            {
                "phase_id": "Phase243",
                "created_at": now,
                "dedup_id": f"phase243:dedup:{key}",
                "duplicate_count": len(items),
                "cluster_key": cluster_key(best),
                "data_split": split_name(str(best.get("case_id")) + ":" + str(best.get("variant_id"))),
            }
        )
        rows.append(best)
    rows.sort(key=lambda x: (safe_float(x.get("candidate_score")), x.get("should_enter_hook", False)), reverse=True)
    return rows


def cluster_rows(dedup_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedup_rows:
        groups[row["cluster_key"]].append(row)
    out = []
    now = utc_now()
    for key, items in groups.items():
        items.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
        parts = key.split("::")
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "cluster_id": f"phase243:cluster:{stable_hash(key)}",
                "cluster_key": key,
                "family_id": parts[0],
                "mode_id": parts[1],
                "recommended_next_test": parts[2],
                "stable_winner_regime": parts[3],
                "failure_group": parts[4],
                "margin_bucket": parts[5],
                "candidate_count": len(items),
                "hook_ready_count": sum(1 for x in items if x.get("should_enter_hook")),
                "mean_candidate_score": round(mean(safe_float(x.get("candidate_score")) for x in items), 4),
                "max_candidate_score": round(max(safe_float(x.get("candidate_score")) for x in items), 4),
                "selection_reasons": dict(Counter(r for x in items for r in x.get("selection_reasons", [])).most_common()),
                "data_splits": dict(Counter(str(x.get("data_split")) for x in items).most_common()),
                "representative_candidate_ids": [x.get("candidate_id") for x in items[:5]],
            }
        )
    out.sort(key=lambda x: (x["hook_ready_count"], x["mean_candidate_score"]), reverse=True)
    return out


def select_trace_rows(dedup_rows: list[dict[str, Any]], max_total: int) -> list[dict[str, Any]]:
    rows_by_test: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedup_rows:
        if row.get("should_enter_hook"):
            rows_by_test[str(row.get("recommended_next_test"))].append(row)
    for rows in rows_by_test.values():
        rows.sort(key=lambda x: (safe_float(x.get("candidate_score")), -safe_float(x.get("scoring_risk_rate"))), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for test, target in TRACE_TARGET_COUNTS.items():
        for row in rows_by_test.get(test, [])[:target]:
            key = (str(row.get("case_id")), str(row.get("variant_id")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    if len(selected) < max_total:
        leftovers = [x for x in dedup_rows if x.get("should_enter_hook") and (str(x.get("case_id")), str(x.get("variant_id"))) not in seen]
        leftovers.sort(key=lambda x: safe_float(x.get("candidate_score")), reverse=True)
        for row in leftovers:
            if len(selected) >= max_total:
                break
            key = (str(row.get("case_id")), str(row.get("variant_id")))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
    selected = selected[:max_total]
    out = []
    now = utc_now()
    for rank, row in enumerate(selected, start=1):
        out.append(
            {
                **row,
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "trace_selection_id": f"phase243:trace:{rank:03d}:{row['case_id']}:{row['variant_id']}",
                "trace_rank": rank,
                "trace_batch": "phase244_batch_a" if rank <= 60 else "phase244_batch_b",
                "selected_for_internal_trace": True,
                "selection_balance_bucket": str(row.get("recommended_next_test")),
            }
        )
    return out


def case_bank_v2_rows(calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    out = []
    for row in calibration_rows:
        risk = str(row.get("scoring_risk"))
        review = bool(row.get("manual_review_flag"))
        aliases = list(row.get("target_aliases") or [])
        acceptable = list(row.get("acceptable_answers") or aliases)
        if review and row.get("target") not in acceptable:
            acceptable.append(row.get("target"))
        status = "v2_needs_manual_review" if review else "v2_auto_stable"
        out.append(
            {
                **row,
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "case_bank_v2_id": f"phase243:casebank_v2:{row['case_id']}",
                "target_aliases_v2": sorted(set(str(x) for x in aliases if str(x))),
                "acceptable_answers_v2": sorted(set(str(x) for x in acceptable if str(x))),
                "relation_schema_v2": row.get("relation_schema"),
                "answer_policy_v2": row.get("answer_policy"),
                "scoring_risk_score_v2": row.get("mean_scoring_risk"),
                "manual_review_status_v2": status,
                "use_for_explore": True,
                "use_for_validate": risk != "high",
                "use_for_frozen": (not review) and risk == "low",
            }
        )
    return out


def data_split_rows(dedup_rows: list[dict[str, Any]], case_v2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    case_flags = {row["case_id"]: row for row in case_v2_rows}
    out = []
    for row in dedup_rows:
        case = case_flags.get(row["case_id"], {})
        split = row.get("data_split")
        if split == "frozen" and not case.get("use_for_frozen", False):
            split = "validate" if case.get("use_for_validate", False) else "explore"
        if split == "validate" and not case.get("use_for_validate", True):
            split = "explore"
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "split_id": f"phase243:split:{row['case_id']}:{row['variant_id']}",
                "candidate_id": row.get("candidate_id"),
                "case_id": row["case_id"],
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "variant_id": row["variant_id"],
                "data_split": split,
                "original_split": row.get("data_split"),
                "split_reason": "hash_split_adjusted_by_casebank_risk" if split != row.get("data_split") else "hash_split",
                "candidate_score": row.get("candidate_score"),
                "recommended_next_test": row.get("recommended_next_test"),
            }
        )
    return out


def metrics(clusters: list[dict[str, Any]], selected: list[dict[str, Any]], case_v2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for cluster in clusters:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "metric_id": f"phase243:cluster:{cluster['cluster_id']}",
                "scope": "candidate_cluster",
                "family_id": cluster["family_id"],
                "mode_id": cluster["mode_id"],
                "metric_name": "candidate_cluster_size",
                "metric_value": cluster["candidate_count"],
                "hook_ready_count": cluster["hook_ready_count"],
                "mean_candidate_score": cluster["mean_candidate_score"],
                "recommended_next_test": cluster["recommended_next_test"],
                "stable_winner_regime": cluster["stable_winner_regime"],
            }
        )
    selected_by_test = Counter(str(x.get("recommended_next_test")) for x in selected)
    for test, count in selected_by_test.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "metric_id": f"phase243:selected:{test}",
                "scope": "trace_selection",
                "metric_name": "selected_internal_trace_count",
                "metric_value": count,
                "recommended_next_test": test,
            }
        )
    rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase243",
            "created_at": now,
            "metric_id": "phase243:casebank_v2:manual_review",
            "scope": "case_bank",
            "metric_name": "manual_review_case_count",
            "metric_value": sum(1 for x in case_v2 if x["manual_review_status_v2"] == "v2_needs_manual_review"),
            "case_count": len(case_v2),
        }
    )
    return rows


def observations(selected: list[dict[str, Any]], clusters: list[dict[str, Any]], case_v2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for row in selected:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "observation_id": f"phase243:trace_selection:{row['trace_rank']:03d}",
                "run_id": "phase243:candidate_clustering",
                "case_id": row["case_id"],
                "model": "cross_model",
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "level": "internal_trace_selection",
                "metric_name": "candidate_score",
                "metric_value": safe_float(row.get("candidate_score")),
                "metric_unit": "score",
                "variant_id": row["variant_id"],
                "negative_result": True,
                "negative_category": ",".join(row.get("selection_reasons", [])),
                "mechanism_hint": row.get("recommended_next_test"),
                "should_enter_hook": True,
            }
        )
    for cluster in clusters:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "observation_id": f"phase243:cluster:{cluster['cluster_id']}",
                "run_id": "phase243:candidate_clustering",
                "case_id": cluster["cluster_id"],
                "model": "cross_model",
                "family_id": cluster["family_id"],
                "mode_id": cluster["mode_id"],
                "level": "candidate_cluster",
                "metric_name": "candidate_count",
                "metric_value": cluster["candidate_count"],
                "metric_unit": "count",
                "negative_result": True,
                "negative_category": cluster["failure_group"],
                "mechanism_hint": cluster["recommended_next_test"],
                "should_enter_hook": cluster["hook_ready_count"] > 0,
            }
        )
    for row in case_v2:
        if row["manual_review_status_v2"] != "v2_needs_manual_review":
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "observation_id": f"phase243:casebank_v2:{row['case_id']}",
                "run_id": "phase243:casebank_v2",
                "case_id": row["case_id"],
                "model": "cross_model",
                "family_id": row["family_id"],
                "mode_id": row["mode_id"],
                "level": "case_bank_v2",
                "metric_name": "scoring_risk_score_v2",
                "metric_value": safe_float(row.get("scoring_risk_score_v2")),
                "metric_unit": "score",
                "negative_result": True,
                "negative_category": "case_bank_review",
                "mechanism_hint": row.get("review_reason"),
                "should_enter_hook": False,
            }
        )
    return rows


def edges(clusters: list[dict[str, Any]], selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for cluster in clusters[:80]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "edge_id": f"phase243:{cluster['cluster_id']}:to:{cluster['recommended_next_test']}",
                "source": f"cluster:{cluster['cluster_id']}",
                "target": f"internal_trace:{cluster['recommended_next_test']}",
                "edge_type": "cluster_to_internal_trace_plan",
                "family_id": cluster["family_id"],
                "mode_id": cluster["mode_id"],
                "model": "cross_model",
                "evidence_type": "candidate_cluster",
                "effect_direction": "candidate_support",
                "effect_size": cluster["candidate_count"],
                "confidence": round(0.35 + min(0.40, cluster["hook_ready_count"] / 20), 4),
                "supporting_phases": ["Phase241", "Phase242", "Phase243"],
                "status": "source_candidate",
            }
        )
    selected_by_test = Counter(str(x["recommended_next_test"]) for x in selected)
    for test, count in selected_by_test.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase243",
                "created_at": now,
                "edge_id": f"phase243:selected_batch:{test}",
                "source": "node:HighValueInternalTraceSelection",
                "target": f"internal_trace:{test}",
                "edge_type": "selected_trace_batch",
                "family_id": "cross_family",
                "mode_id": test,
                "model": "cross_model",
                "evidence_type": "balanced_candidate_selection",
                "effect_direction": "positive",
                "effect_size": count,
                "confidence": 0.62,
                "supporting_phases": ["Phase243"],
                "status": "source_candidate",
            }
        )
    return rows


def write_report(path: Path, payload: dict[str, Any], clusters: list[dict[str, Any]], selected: list[dict[str, Any]], case_v2: list[dict[str, Any]]) -> None:
    lines = ["# Phase243 Candidate Clustering And Case Bank V2", ""]
    for key in [
        "input_candidates",
        "dedup_candidates",
        "cluster_count",
        "trace_selection_rows",
        "case_bank_v2_rows",
        "manual_review_cases",
    ]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(["", "## Trace Selection By Test", ""])
    for key, value in payload["trace_selection_by_test"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Data Splits", ""])
    for key, value in payload["data_split_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Clusters", "", "| count | hook | score | family | mode | test | winner | group | margin |", "| ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |"])
    for c in clusters[:40]:
        lines.append(
            f"| {c['candidate_count']} | {c['hook_ready_count']} | {c['mean_candidate_score']} | {c['family_id']} | {c['mode_id']} | "
            f"{c['recommended_next_test']} | {c['stable_winner_regime']} | {c['failure_group']} | {c['margin_bucket']} |"
        )
    lines.extend(["", "## Selected Internal Trace Rows", "", "| rank | score | family | mode | variant | test | split | winner |", "| ---: | ---: | --- | --- | --- | --- | --- | --- |"])
    for row in selected[:80]:
        lines.append(
            f"| {row['trace_rank']} | {row['candidate_score']} | {row['family_id']} | {row['mode_id']} | {row['variant_id']} | "
            f"{row['recommended_next_test']} | {row['data_split']} | {row['stable_winner_regime']} |"
        )
    lines.extend(["", "## Case Bank V2 Review", ""])
    lines.append(f"manual_review_cases: {sum(1 for x in case_v2 if x['manual_review_status_v2'] == 'v2_needs_manual_review')}")
    lines.append("")
    lines.append("Phase243 does not run model hooks. It prepares a balanced first internal-trace batch and data splits.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_atlas(payload: dict[str, Any], obs: list[dict[str, Any]], mets: list[dict[str, Any]], eds: list[dict[str, Any]]) -> None:
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", obs, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", mets, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", eds, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase243"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.69
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.52
        progress.setdefault("levels", {})["candidate_clustering"] = 0.40
        progress.setdefault("levels", {})["case_bank_calibration"] = 0.36
        progress.setdefault("levels", {})["high_value_trace_selection"] = 0.48
        progress["next_phase"] = "Phase244_first_internal_trace_batch"
        progress["latest_phase"] = {
            "phase_id": "Phase243",
            "title": "候选聚类、数据划分与 case bank v2",
            "dedup_candidates": payload["dedup_candidates"],
            "cluster_count": payload["cluster_count"],
            "trace_selection_rows": payload["trace_selection_rows"],
            "manual_review_cases": payload["manual_review_cases"],
            "trace_selection_by_test": payload["trace_selection_by_test"],
            "data_split_counts": payload["data_split_counts"],
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    old = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase243 Candidate Clustering And Case Bank V2 Update"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- dedup_candidates: {payload['dedup_candidates']}\n"
        f"- cluster_count: {payload['cluster_count']}\n"
        f"- trace_selection_rows: {payload['trace_selection_rows']}\n"
        f"- case_bank_v2_rows: {payload['case_bank_v2_rows']}\n"
        f"- manual_review_cases: {payload['manual_review_cases']}\n"
        f"- trace_selection_by_test: {payload['trace_selection_by_test']}\n"
        f"- data_split_counts: {payload['data_split_counts']}\n"
    )
    summary_path.write_text(old.rstrip() + "\n" + addition, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates, calibration, _multilabel, phase242_summary = load_phase242(args.phase242_round)
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = dedup_candidates(candidates)
    clusters = cluster_rows(dedup)
    selected = select_trace_rows(dedup, args.max_trace_rows)
    case_v2 = case_bank_v2_rows(calibration)
    splits = data_split_rows(dedup, case_v2)
    mets = metrics(clusters, selected, case_v2)
    obs = observations(selected, clusters, case_v2)
    eds = edges(clusters, selected)
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Candidate clustering, data split, and case bank v2",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "phase242_round": args.phase242_round,
        "input_candidates": len(candidates),
        "input_multilabel_rows": phase242_summary.get("multilabel_rows", 0),
        "dedup_candidates": len(dedup),
        "cluster_count": len(clusters),
        "trace_selection_rows": len(selected),
        "case_bank_v2_rows": len(case_v2),
        "manual_review_cases": sum(1 for x in case_v2 if x["manual_review_status_v2"] == "v2_needs_manual_review"),
        "data_split_counts": dict(Counter(str(x["data_split"]) for x in splits).most_common()),
        "trace_selection_by_test": dict(Counter(str(x["recommended_next_test"]) for x in selected).most_common()),
        "top_clusters": clusters[:30],
        "progress_estimate": {
            "pattern_family_atlas": 0.69,
            "candidate_clustering": 0.40,
            "case_bank_calibration": 0.36,
            "high_value_trace_selection": 0.48,
            "model_internal_closure": 0.46,
            "general_language_mechanism_confidence": 0.52,
        },
    }
    write_json(out_dir / "phase243_pattern_mining_summary.json", payload)
    write_jsonl(out_dir / "phase243_candidate_dedup_rows.jsonl", dedup)
    write_jsonl(out_dir / "phase243_candidate_cluster_rows.jsonl", clusters)
    write_jsonl(out_dir / "phase243_trace_selection_rows.jsonl", selected)
    write_jsonl(out_dir / "phase243_case_bank_v2_rows.jsonl", case_v2)
    write_jsonl(out_dir / "phase243_data_split_rows.jsonl", splits)
    write_jsonl(out_dir / "phase243_observations.jsonl", obs)
    write_jsonl(out_dir / "phase243_metrics.jsonl", mets)
    write_jsonl(out_dir / "phase243_graph_edges.jsonl", eds)
    write_report(out_dir / "phase243_internal_trace_plan.md", payload, clusters, selected, case_v2)
    update_atlas(payload, obs, mets, eds)
    print(json.dumps({"phase": PHASE, "status": "complete", "clusters": len(clusters), "trace_selection_rows": len(selected)}, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase243 candidate clustering and case bank v2")
    parser.add_argument("--phase242-round", default="negative_multilabel_and_trace_selection")
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-trace-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
