#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PHASE = 242
SOURCE_PHASE = 241
SCHEMA_VERSION = "1.0.0"
MODELS = ["qwen3", "glm4", "deepseek7b"]
ATLAS_ROOT = Path("tests/result/pattern_family_atlas/v1")
PHASE241_ROOT = Path("tests/result/phase241_large_scale_pattern_atlas_benchmark")
RESULT_ROOT = Path("tests/result/phase242_negative_multilabel_and_trace_selection")
ROUND_DEFAULT = "negative_multilabel_and_trace_selection"

SCORING_RISK_MODES = {
    "location_fact",
    "causal_fact",
    "classify",
    "function_answer",
    "part_whole",
    "material_answer",
    "EN_to_FR",
    "FR_to_EN",
    "cross_lingual_reasoning",
    "translate",
    "summarize",
    "rewrite",
    "compare",
}

HOOK_PRIORITY_CATEGORIES = {
    "protocol",
    "readout",
    "rollout",
    "closure",
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


def load_phase241(round_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = PHASE241_ROOT / round_name
    behavior = read_jsonl(root / "phase241_large_scale_behavior_rows.jsonl")
    cases = read_jsonl(root / "phase241_large_scale_case_rows.jsonl")
    summary = read_json(root / "phase241_cross_model_summary.json")
    if not behavior:
        raise FileNotFoundError(f"missing Phase241 behavior rows under {root}")
    return behavior, cases, summary


def risk_reason(row: dict[str, Any]) -> tuple[str, float]:
    family = str(row.get("family_id", ""))
    mode = str(row.get("mode_id", ""))
    variant = str(row.get("variant_id", ""))
    reasons = []
    risk = 0.0
    if mode in SCORING_RISK_MODES:
        reasons.append("known_relation_or_answer_policy_risk")
        risk += 0.35
    if family in {"cross_lingual", "language_action"}:
        reasons.append("language_or_action_surface_variance")
        risk += 0.20
    if row.get("negative_category") == "semantic_failure" and safe_float(row.get("target_rank"), 9999) <= 20:
        reasons.append("target_readout_near_but_string_match_failed")
        risk += 0.25
    if variant == "target_seeded":
        reasons.append("target_seeded_upper_bound_not_natural_case")
        risk += 0.25
    if row.get("negative_category") == "semantic_failure":
        risk += 0.15
    return "+".join(reasons) if reasons else "low", round(min(1.0, risk), 4)


def multilabel_for(row: dict[str, Any]) -> dict[str, bool]:
    semantic_match = bool(row.get("semantic_match"))
    protocol_match = bool(row.get("protocol_match"))
    closure_signal = bool(row.get("closure_signal"))
    over_generation = bool(row.get("over_generation"))
    margin = safe_float(row.get("target_margin_vs_winner"))
    target_rank = safe_float(row.get("target_rank"), 9999)
    negative_category = str(row.get("negative_category", ""))
    risk_text, scoring_risk = risk_reason(row)
    labels = {
        "semantic": not semantic_match,
        "protocol": semantic_match and not protocol_match,
        "readout": margin < -1.0 or (semantic_match and target_rank > 50 and negative_category in {"readout_negative", "protocol_negative"}),
        "rollout": over_generation or (semantic_match and safe_float(row.get("output_token_count")) > 16),
        "closure": semantic_match and (not closure_signal or negative_category == "closure_negative"),
        "scoring": scoring_risk >= 0.45,
    }
    row["scoring_risk"] = scoring_risk
    row["scoring_risk_reason"] = risk_text
    return labels


def build_multilabel_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    now = utc_now()
    for row in rows:
        labels = multilabel_for(dict(row))
        label_count = sum(1 for v in labels.values() if v)
        primary = ",".join([k for k, v in labels.items() if v]) or "none"
        out.append(
            {
                **row,
                "phase": PHASE,
                "source_phase": SOURCE_PHASE,
                "phase_id": "Phase242",
                "created_at": now,
                "multilabel_id": f"phase242:{row['model']}:{row['case_id']}:{row['variant_id']}",
                "negative_labels": labels,
                "negative_label_count": label_count,
                "primary_negative_signature": primary,
                "scoring_risk": row.get("scoring_risk", 0.0),
                "scoring_risk_reason": row.get("scoring_risk_reason", "low"),
                "should_enter_probe": should_enter_probe(row, labels),
                "should_enter_ablation": False,
                "should_enter_closure": False,
            }
        )
    return out


def should_enter_probe(row: dict[str, Any], labels: dict[str, bool]) -> bool:
    if str(row.get("variant_id")) == "target_seeded":
        return False
    if labels.get("scoring") and not (labels.get("readout") or labels.get("closure") or labels.get("rollout")):
        return False
    return any(labels.get(k) for k in HOOK_PRIORITY_CATEGORIES) and bool(row.get("semantic_match"))


def case_bank_calibration_rows(cases: list[dict[str, Any]], multilabel_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in multilabel_rows:
        buckets[row["case_id"]].append(row)
    now = utc_now()
    out = []
    for case in cases:
        items = buckets.get(case["case_id"], [])
        if not items:
            continue
        semantic_failure_rate = sum(1 for x in items if x["negative_labels"]["semantic"]) / len(items)
        scoring_risk = mean(safe_float(x.get("scoring_risk")) for x in items)
        risk_mode = str(case.get("mode_id")) in SCORING_RISK_MODES
        manual_review = semantic_failure_rate >= 0.45 or scoring_risk >= 0.35 or risk_mode
        aliases = list(case.get("target_aliases") or [case.get("target")])
        acceptable = sorted(set(str(x) for x in aliases + [case.get("target", "")] if str(x)))
        relation_schema = relation_schema_for(case)
        answer_policy = answer_policy_for(case)
        out.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase242",
                "created_at": now,
                "case_id": case["case_id"],
                "family_id": case["family_id"],
                "mode_id": case["mode_id"],
                "target": case.get("target", ""),
                "target_aliases": aliases,
                "acceptable_answers": acceptable,
                "relation_schema": relation_schema,
                "answer_policy": answer_policy,
                "semantic_failure_rate": round(semantic_failure_rate, 4),
                "mean_scoring_risk": round(scoring_risk, 4),
                "scoring_risk": "high" if scoring_risk >= 0.55 else "medium" if scoring_risk >= 0.30 or risk_mode else "low",
                "semantic_review_status": "needs_manual_review" if manual_review else "auto_ok",
                "manual_review_flag": bool(manual_review),
                "review_reason": review_reason(case, semantic_failure_rate, scoring_risk, risk_mode),
            }
        )
    return out


def relation_schema_for(case: dict[str, Any]) -> str:
    family = str(case.get("family_id"))
    mode = str(case.get("mode_id"))
    if family == "content_knowledge":
        return f"content_relation::{mode}"
    if family == "output_protocol":
        return f"same_content_different_output_protocol::{mode}"
    if family == "reasoning_constraint":
        return f"constraint_state::{mode}"
    if family == "syntax_structure":
        return f"surface_boundary_or_syntax::{mode}"
    if family == "language_action":
        return f"language_action::{mode}"
    if family == "cross_lingual":
        return f"language_transfer::{mode}"
    if family == "readout_competition":
        return f"readout_competitor_probe::{mode}"
    if family == "state_drift":
        return f"rollout_drift_probe::{mode}"
    if family == "closure":
        return f"closure_condition::{mode}"
    return mode


def answer_policy_for(case: dict[str, Any]) -> str:
    mode = str(case.get("mode_id"))
    if mode in {"one_word", "short_answer", "stop_after_answer", "pattern_matched", "boundary_stable", "done_state_stable", "model_stop_executed", "no_drift"}:
        return "short_answer_exact_or_alias"
    if mode in {"explain_answer", "because_reason", "explain"}:
        return "answer_first_then_reason"
    if mode in {"json_answer", "table_answer", "list_answer"}:
        return "structured_format_with_answer"
    if "ZH" in mode or "EN_to_ZH" in mode:
        return "language_specific_alias"
    return "semantic_alias_or_brief_answer"


def review_reason(case: dict[str, Any], semantic_failure_rate: float, scoring_risk: float, risk_mode: bool) -> str:
    reasons = []
    if semantic_failure_rate >= 0.45:
        reasons.append("high_semantic_failure_rate")
    if scoring_risk >= 0.35:
        reasons.append("high_scoring_risk")
    if risk_mode:
        reasons.append("known_rough_relation_schema")
    return "+".join(reasons) if reasons else "none"


def cross_model_groups(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["variant_id"])].append(row)
    return groups


def high_value_candidates(rows: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    groups = cross_model_groups(rows)
    now = utc_now()
    candidates = []
    for (case_id, variant_id), items in groups.items():
        if variant_id == "target_seeded":
            continue
        models = sorted(set(x["model"] for x in items))
        if len(models) < 2:
            continue
        semantic_rate = sum(1 for x in items if x.get("semantic_match")) / len(items)
        protocol_fail_rate = sum(1 for x in items if x["negative_labels"]["protocol"]) / len(items)
        readout_rate = sum(1 for x in items if x["negative_labels"]["readout"]) / len(items)
        rollout_rate = sum(1 for x in items if x["negative_labels"]["rollout"]) / len(items)
        closure_rate = sum(1 for x in items if x["negative_labels"]["closure"]) / len(items)
        scoring_rate = sum(1 for x in items if x["negative_labels"]["scoring"]) / len(items)
        failed_models = sorted(set(x["model"] for x in items if x.get("negative_label_count", 0) > 0))
        success_models = sorted(set(x["model"] for x in items if x.get("negative_label_count", 0) == 0))
        winners = Counter(str(x.get("winning_regime")) for x in items)
        stable_winner, stable_winner_count = winners.most_common(1)[0]
        mean_margin = mean(safe_float(x.get("target_margin_vs_winner")) for x in items)
        reasons = []
        if len(failed_models) == 3:
            reasons.append("cross_model_stable_failure")
        if semantic_rate >= 0.66 and rollout_rate >= 0.50:
            reasons.append("semantic_correct_rollout_failure")
        if semantic_rate >= 0.66 and closure_rate >= 0.50:
            reasons.append("semantic_correct_closure_failure")
        if semantic_rate >= 0.66 and protocol_fail_rate >= 0.50:
            reasons.append("semantic_correct_protocol_failure")
        if readout_rate >= 0.50 and stable_winner_count >= 2:
            reasons.append("stable_readout_competitor")
        if success_models and failed_models:
            reasons.append("cross_model_divergence")
        if mean_margin > -1.0 and protocol_fail_rate >= 0.50:
            reasons.append("high_target_pressure_protocol_failure")
        if scoring_rate >= 0.66 and semantic_rate < 0.50:
            reasons.append("scoring_or_case_bank_risk")
        hook_ready = bool(set(reasons) - {"scoring_or_case_bank_risk"}) and scoring_rate < 0.75
        score = (
            0.25 * min(1.0, len(failed_models) / 3)
            + 0.20 * semantic_rate
            + 0.15 * max(protocol_fail_rate, readout_rate, rollout_rate, closure_rate)
            + 0.15 * min(1.0, stable_winner_count / 3)
            + 0.15 * (1.0 if success_models and failed_models else 0.0)
            + 0.10 * (1.0 if mean_margin > -1.0 else 0.0)
            - 0.20 * scoring_rate
        )
        template = items[0]
        if reasons:
            candidates.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase242",
                    "created_at": now,
                    "candidate_id": f"phase242:{case_id}:{variant_id}",
                    "case_id": case_id,
                    "family_id": template["family_id"],
                    "mode_id": template["mode_id"],
                    "variant_id": variant_id,
                    "models": models,
                    "failed_models": failed_models,
                    "success_models": success_models,
                    "semantic_rate": round(semantic_rate, 4),
                    "protocol_fail_rate": round(protocol_fail_rate, 4),
                    "readout_fail_rate": round(readout_rate, 4),
                    "rollout_fail_rate": round(rollout_rate, 4),
                    "closure_fail_rate": round(closure_rate, 4),
                    "scoring_risk_rate": round(scoring_rate, 4),
                    "mean_target_margin_vs_winner": round(mean_margin, 4),
                    "stable_winner_regime": stable_winner,
                    "winner_regimes": dict(winners.most_common()),
                    "selection_reasons": reasons,
                    "candidate_score": round(max(0.0, score), 4),
                    "should_enter_hook": bool(hook_ready),
                    "should_enter_case_bank_review": scoring_rate >= 0.50 or "scoring_or_case_bank_risk" in reasons,
                    "recommended_next_test": recommended_next_test(reasons),
                }
            )
    candidates.sort(key=lambda x: (x["should_enter_hook"], x["candidate_score"], -x["scoring_risk_rate"]), reverse=True)
    return candidates[:max_candidates]


def recommended_next_test(reasons: list[str]) -> str:
    if "stable_readout_competitor" in reasons:
        return "readout_competitor_trace"
    if "semantic_correct_protocol_failure" in reasons or "high_target_pressure_protocol_failure" in reasons:
        return "protocol_gate_product_residual_trace"
    if "semantic_correct_closure_failure" in reasons:
        return "rollout_closure_trace"
    if "semantic_correct_rollout_failure" in reasons:
        return "stepwise_rollout_trace"
    if "cross_model_divergence" in reasons:
        return "cross_model_structure_comparison"
    return "case_bank_review"


def trace_selection_matrix(candidates: list[dict[str, Any]], calibration_rows: list[dict[str, Any]], multilabel_rows: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: dict[str, Any] = {"by_family_mode": {}, "by_reason": {}, "by_next_test": {}, "case_bank_review": {}}
    for cand in candidates:
        key = f"{cand['family_id']}::{cand['mode_id']}"
        bucket = matrix["by_family_mode"].setdefault(key, {"candidates": 0, "hook_ready": 0, "case_review": 0, "reasons": Counter(), "next_tests": Counter()})
        bucket["candidates"] += 1
        bucket["hook_ready"] += int(cand["should_enter_hook"])
        bucket["case_review"] += int(cand["should_enter_case_bank_review"])
        bucket["reasons"].update(cand["selection_reasons"])
        bucket["next_tests"].update([cand["recommended_next_test"]])
        for reason in cand["selection_reasons"]:
            matrix["by_reason"][reason] = matrix["by_reason"].get(reason, 0) + 1
        matrix["by_next_test"][cand["recommended_next_test"]] = matrix["by_next_test"].get(cand["recommended_next_test"], 0) + 1
    for key, bucket in matrix["by_family_mode"].items():
        bucket["reasons"] = dict(bucket["reasons"].most_common())
        bucket["next_tests"] = dict(bucket["next_tests"].most_common())
    review_rows = [x for x in calibration_rows if x["manual_review_flag"]]
    matrix["case_bank_review"] = {
        "review_cases": len(review_rows),
        "high_risk_cases": sum(1 for x in review_rows if x["scoring_risk"] == "high"),
        "top_review_modes": dict(Counter(f"{x['family_id']}::{x['mode_id']}" for x in review_rows).most_common(20)),
    }
    matrix["multilabel_counts"] = {
        key: sum(1 for x in multilabel_rows if x["negative_labels"][key])
        for key in ["semantic", "protocol", "readout", "rollout", "closure", "scoring"]
    }
    return matrix


def metric_rows(multilabel_rows: list[dict[str, Any]], candidates: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in multilabel_rows:
        buckets[(row["family_id"], row["mode_id"])].append(row)
    cand_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        cand_buckets[(cand["family_id"], cand["mode_id"])].append(cand)
    cal_buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        cal_buckets[(row["family_id"], row["mode_id"])].append(row)
    for (family, mode), items in buckets.items():
        labels = {
            key: sum(1 for x in items if x["negative_labels"][key]) / len(items)
            for key in ["semantic", "protocol", "readout", "rollout", "closure", "scoring"]
        }
        cands = cand_buckets.get((family, mode), [])
        cals = cal_buckets.get((family, mode), [])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase242",
                "created_at": now,
                "metric_id": f"phase242:{family}:{mode}:multilabel_selection",
                "scope": "mode",
                "family_id": family,
                "mode_id": mode,
                "metric_name": "multilabel_negative_and_trace_selection",
                "metric_value": round(max(labels.values()), 4),
                "negative_label_rates": {k: round(v, 4) for k, v in labels.items()},
                "hook_candidate_count": sum(1 for x in cands if x["should_enter_hook"]),
                "case_review_candidate_count": sum(1 for x in cands if x["should_enter_case_bank_review"]),
                "manual_review_case_count": sum(1 for x in cals if x["manual_review_flag"]),
                "mean_candidate_score": round(mean([x["candidate_score"] for x in cands]), 4) if cands else 0.0,
                "rows": len(items),
            }
        )
    return rows


def observation_rows(candidates: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    for cand in candidates:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase242",
                "created_at": now,
                "observation_id": f"phase242:candidate:{cand['case_id']}:{cand['variant_id']}",
                "run_id": "phase242:selection",
                "case_id": cand["case_id"],
                "model": "cross_model",
                "family_id": cand["family_id"],
                "mode_id": cand["mode_id"],
                "level": "high_value_trace_selection",
                "metric_name": "candidate_score",
                "metric_value": cand["candidate_score"],
                "metric_unit": "score",
                "variant_id": cand["variant_id"],
                "negative_result": True,
                "negative_category": ",".join(cand["selection_reasons"]),
                "mechanism_hint": cand["recommended_next_test"],
                "should_enter_hook": cand["should_enter_hook"],
            }
        )
    for cal in calibration_rows:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase242",
                "created_at": now,
                "observation_id": f"phase242:case_calibration:{cal['case_id']}",
                "run_id": "phase242:case_bank_calibration",
                "case_id": cal["case_id"],
                "model": "cross_model",
                "family_id": cal["family_id"],
                "mode_id": cal["mode_id"],
                "level": "case_bank_calibration",
                "metric_name": "mean_scoring_risk",
                "metric_value": cal["mean_scoring_risk"],
                "metric_unit": "score",
                "negative_result": cal["manual_review_flag"],
                "negative_category": "scoring_risk" if cal["manual_review_flag"] else "none",
                "mechanism_hint": cal["review_reason"],
                "should_enter_hook": False,
            }
        )
    return rows


def graph_edges(candidates: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = utc_now()
    rows = []
    reason_counts = Counter()
    for cand in candidates:
        reason_counts.update(cand["selection_reasons"])
    for reason, count in reason_counts.items():
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase242",
                "created_at": now,
                "edge_id": f"phase242:negative_taxonomy:{reason}",
                "source": f"negative_label:{reason}",
                "target": "node:HighValueInternalTraceSelection",
                "edge_type": "negative_to_trace_candidate",
                "family_id": "cross_family",
                "mode_id": reason,
                "model": "cross_model",
                "evidence_type": "phase241_multilabel_selection",
                "effect_direction": "candidate_source",
                "effect_size": count,
                "confidence": round(0.35 + min(0.45, count / 300), 4),
                "supporting_phases": ["Phase241", "Phase242"],
                "status": "source_candidate",
            }
        )
    top_metrics = sorted(metrics, key=lambda x: x.get("hook_candidate_count", 0), reverse=True)[:30]
    for metric in top_metrics:
        if metric.get("hook_candidate_count", 0) <= 0:
            continue
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase242",
                "created_at": now,
                "edge_id": f"phase242:mode:{metric['family_id']}:{metric['mode_id']}:hook_candidates",
                "source": f"mode:{metric['family_id']}:{metric['mode_id']}",
                "target": "node:HighValueInternalTraceSelection",
                "edge_type": "mode_to_hook_candidate_pool",
                "family_id": metric["family_id"],
                "mode_id": metric["mode_id"],
                "model": "cross_model",
                "evidence_type": "multilabel_candidate_count",
                "effect_direction": "positive",
                "effect_size": metric["hook_candidate_count"],
                "confidence": round(0.40 + min(0.40, metric["hook_candidate_count"] / 20), 4),
                "supporting_phases": ["Phase241", "Phase242"],
                "status": "source_candidate",
            }
        )
    return rows


def write_report(path: Path, payload: dict[str, Any], matrix: dict[str, Any], candidates: list[dict[str, Any]], calibration_rows: list[dict[str, Any]]) -> None:
    lines = ["# Phase242 Negative Multilabel And Trace Selection", ""]
    for key in [
        "source_behavior_rows",
        "multilabel_rows",
        "high_value_candidates",
        "hook_ready_candidates",
        "case_bank_review_rows",
        "manual_review_cases",
    ]:
        lines.append(f"{key}: {payload[key]}")
    lines.extend(["", "## Multilabel Counts", ""])
    for key, value in payload["multilabel_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate Reasons", ""])
    for key, value in matrix["by_reason"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Next Tests", ""])
    for key, value in matrix["by_next_test"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Hook Candidates", "", "| score | family | mode | variant | reasons | next | winner | margin |", "| ---: | --- | --- | --- | --- | --- | --- | ---: |"])
    for cand in candidates[:40]:
        lines.append(
            f"| {cand['candidate_score']} | {cand['family_id']} | {cand['mode_id']} | {cand['variant_id']} | "
            f"{','.join(cand['selection_reasons'])} | {cand['recommended_next_test']} | {cand['stable_winner_regime']} | {cand['mean_target_margin_vs_winner']} |"
        )
    lines.extend(["", "## Case Bank Review Hotspots", "", "| family | mode | review cases |", "| --- | --- | ---: |"])
    review_counts = Counter(f"{x['family_id']}::{x['mode_id']}" for x in calibration_rows if x["manual_review_flag"])
    for key, count in review_counts.most_common(30):
        family, mode = key.split("::", 1)
        lines.append(f"| {family} | {mode} | {count} |")
    lines.extend(["", "## Caution", "", "This phase does not run models or hooks. It upgrades Phase241 observations into multilabel negatives, case-bank calibration targets, and internal-trace candidates."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_atlas(payload: dict[str, Any], observations: list[dict[str, Any]], metrics: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    append_unique_jsonl(ATLAS_ROOT / "observations.jsonl", observations, "observation_id")
    append_unique_jsonl(ATLAS_ROOT / "metrics.jsonl", metrics, "metric_id")
    append_unique_jsonl(ATLAS_ROOT / "graph_edges.jsonl", edges, "edge_id")
    progress = read_json(ATLAS_ROOT / "progress.json")
    if progress:
        progress["phase_id"] = "Phase242"
        progress["created_at"] = utc_now()
        progress.setdefault("global_progress", {})["pattern_family_atlas"] = 0.66
        progress.setdefault("global_progress", {})["general_language_mechanism_confidence"] = 0.51
        progress.setdefault("levels", {})["large_scale_negative_taxonomy"] = 0.50
        progress.setdefault("levels", {})["case_bank_calibration"] = 0.28
        progress.setdefault("levels", {})["high_value_trace_selection"] = 0.34
        progress.setdefault("levels", {})["behavior"] = max(0.68, safe_float(progress.get("levels", {}).get("behavior")))
        progress.setdefault("levels", {})["readout_competition"] = max(0.56, safe_float(progress.get("levels", {}).get("readout_competition")))
        progress["next_phase"] = "Phase243_mode_trace_clustering_and_candidate_validation"
        progress["latest_phase"] = {
            "phase_id": "Phase242",
            "title": "负面结果多标签化与高价值内部脉络选择",
            "source_behavior_rows": payload["source_behavior_rows"],
            "multilabel_rows": payload["multilabel_rows"],
            "high_value_candidates": payload["high_value_candidates"],
            "hook_ready_candidates": payload["hook_ready_candidates"],
            "manual_review_cases": payload["manual_review_cases"],
            "multilabel_counts": payload["multilabel_counts"],
        }
        write_json(ATLAS_ROOT / "progress.json", progress)
    summary_path = ATLAS_ROOT / "summary.md"
    old = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""
    marker = "## Phase242 Negative Multilabel And Trace Selection Update"
    if marker in old:
        old = old.split(marker, 1)[0].rstrip()
    addition = (
        f"\n{marker}\n\n"
        f"- source_behavior_rows: {payload['source_behavior_rows']}\n"
        f"- multilabel_rows: {payload['multilabel_rows']}\n"
        f"- high_value_candidates: {payload['high_value_candidates']}\n"
        f"- hook_ready_candidates: {payload['hook_ready_candidates']}\n"
        f"- case_bank_review_rows: {payload['case_bank_review_rows']}\n"
        f"- manual_review_cases: {payload['manual_review_cases']}\n"
        f"- multilabel_counts: {payload['multilabel_counts']}\n"
    )
    summary_path.write_text(old.rstrip() + "\n" + addition, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    behavior_rows, case_rows, phase241_summary = load_phase241(args.phase241_round)
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    multilabel_rows = build_multilabel_rows(behavior_rows)
    calibration = case_bank_calibration_rows(case_rows, multilabel_rows)
    candidates = high_value_candidates(multilabel_rows, args.max_candidates)
    matrix = trace_selection_matrix(candidates, calibration, multilabel_rows)
    metrics = metric_rows(multilabel_rows, candidates, calibration)
    observations = observation_rows(candidates, calibration)
    edges = graph_edges(candidates, metrics)
    hook_ready = [x for x in candidates if x["should_enter_hook"]]
    review_candidates = [x for x in calibration if x["manual_review_flag"]]
    payload = {
        "phase": PHASE,
        "source_phase": SOURCE_PHASE,
        "title": "Negative multilabel and high-value internal trace selection",
        "status": "complete",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "schema_version": SCHEMA_VERSION,
        "phase241_round": args.phase241_round,
        "source_behavior_rows": len(behavior_rows),
        "source_negative_rows": phase241_summary.get("negative_rows", 0),
        "multilabel_rows": len(multilabel_rows),
        "case_bank_review_rows": len(calibration),
        "manual_review_cases": len(review_candidates),
        "high_value_candidates": len(candidates),
        "hook_ready_candidates": len(hook_ready),
        "observation_rows": len(observations),
        "metric_rows": len(metrics),
        "graph_edges": len(edges),
        "multilabel_counts": matrix["multilabel_counts"],
        "candidate_reason_counts": matrix["by_reason"],
        "recommended_next_tests": matrix["by_next_test"],
        "top_hook_candidates": hook_ready[:30],
        "top_case_bank_review_modes": matrix["case_bank_review"]["top_review_modes"],
        "progress_estimate": {
            "pattern_family_atlas": 0.66,
            "large_scale_negative_taxonomy": 0.50,
            "case_bank_calibration": 0.28,
            "high_value_trace_selection": 0.34,
            "model_internal_closure": 0.46,
            "general_language_mechanism_confidence": 0.51,
        },
    }
    write_json(out_dir / "phase242_summary.json", payload)
    write_jsonl(out_dir / "phase242_multilabel_negative_rows.jsonl", multilabel_rows)
    write_jsonl(out_dir / "phase242_high_value_hook_candidates.jsonl", candidates)
    write_jsonl(out_dir / "phase242_case_bank_calibration_rows.jsonl", calibration)
    write_json(out_dir / "phase242_trace_selection_matrix.json", matrix)
    write_jsonl(out_dir / "phase242_observations.jsonl", observations)
    write_jsonl(out_dir / "phase242_metrics.jsonl", metrics)
    write_jsonl(out_dir / "phase242_graph_edges.jsonl", edges)
    write_report(out_dir / "phase242_internal_trace_plan.md", payload, matrix, candidates, calibration)
    update_atlas(payload, observations, metrics, edges)
    print(json.dumps({"phase": PHASE, "status": "complete", "multilabel_rows": len(multilabel_rows), "hook_ready_candidates": len(hook_ready)}, ensure_ascii=False, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase242 negative multilabel and trace selection")
    parser.add_argument("--phase241-round", default="large_scale_pattern_atlas_benchmark")
    parser.add_argument("--round-name", default=ROUND_DEFAULT)
    parser.add_argument("--max-candidates", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
