#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402
import phase928_punctuation_specific_protocol_gear_search as p928  # noqa: E402


PHASE = 935
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_ROOT = Path("tests/result/phase935_case_residual_gate_candidate_audit")
PHASE930_ROOT = Path("tests/result/phase930_natural_gate_strict_clean_transition_audit")
PHASE934_ROOT = Path("tests/result/phase934_case_residual_size_control_audit")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float | int | None]) -> float | None:
    vals = [float(x) for x in values if x is not None]
    return None if not vals else float(sum(vals) / len(vals))


def median(values: list[float | int | None]) -> float | None:
    vals = [float(x) for x in values if x is not None]
    return None if not vals else float(statistics.median(vals))


def row_success(row: dict[str, Any]) -> bool:
    return bool(row.get("new_top1_vs_coordinate_base") or row.get("new_margin_closure_vs_coordinate_base"))


def state_success_map(rows: list[dict[str, Any]], group_kind: str, factor: float) -> dict[str, bool]:
    out: dict[str, bool] = defaultdict(bool)
    for row in rows:
        if row.get("candidate_group_kind") != group_kind:
            continue
        if abs(float(row.get("candidate_group_factor") or 0.0) - factor) > 1e-9:
            continue
        key = str(row.get("target_state_key"))
        out[key] = bool(out[key] or row_success(row))
    return dict(out)


def best_binary_split(rows: list[dict[str, Any]], feature: str, target: str) -> dict[str, Any] | None:
    pairs = []
    for row in rows:
        value = numeric(row.get(feature))
        if value is None or row.get(target) is None:
            continue
        pairs.append((value, bool(row.get(target))))
    if len(pairs) < 6:
        return None
    values = sorted(set(value for value, _target in pairs))
    if len(values) < 2:
        return None
    cuts = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
    best: dict[str, Any] | None = None
    for cut in cuts:
        for polarity in ["le_true", "ge_true"]:
            correct = 0
            for value, target_value in pairs:
                pred = value <= cut if polarity == "le_true" else value >= cut
                correct += int(pred == target_value)
            item = {
                "feature": feature,
                "target": target,
                "threshold": float(cut),
                "polarity": polarity,
                "correct": int(correct),
                "total": int(len(pairs)),
                "accuracy": float(correct / len(pairs)),
                "true_count": int(sum(1 for _value, t in pairs if t)),
                "false_count": int(sum(1 for _value, t in pairs if not t)),
            }
            if best is None or (item["accuracy"], item["correct"]) > (best["accuracy"], best["correct"]):
                best = item
    return best


def case_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("target_case_id"))].append(row)
    out = []
    for case_id, srows in sorted(grouped.items()):
        out.append(
            {
                "target_case_id": case_id,
                "states": len(srows),
                "fixed_success_2_25": sum(1 for row in srows if row.get("fixed_success_2_25")),
                "residual_needed_2_25": sum(1 for row in srows if row.get("residual_needed_2_25")),
                "true_loso_repair_success_2_25": sum(1 for row in srows if row.get("true_loso_repair_success_2_25")),
                "size_control_success_2_25": sum(1 for row in srows if row.get("size_control_success_2_25")),
                "true_beats_controls_2_25": sum(1 for row in srows if row.get("true_beats_controls_2_25")),
                "opening_threshold_median": median([row.get("opening_threshold_factor") for row in srows]),
                "route_delta_norm_mean": mean([row.get("target_route_delta_norm") for row in srows]),
                "boundary_eos_rank_median": median([row.get("target_boundary_eos_rank") for row in srows]),
                "margin_pos_mean_score_mean": mean([row.get("l39_margin_pos_mean_score") for row in srows]),
            }
        )
    return out


def feature_split_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    features = p928.parse_csv(args.gate_features)
    targets = p928.parse_csv(args.targets)
    splits = []
    for target in targets:
        true_cases = sorted({str(row.get("target_case_id")) for row in rows if row.get(target)})
        false_cases = sorted({str(row.get("target_case_id")) for row in rows if not row.get(target)})
        case_confounded = len(true_cases) <= 1 or len(false_cases) <= 1
        for feature in features:
            split = best_binary_split(rows, feature, target)
            if not split:
                continue
            split["true_cases"] = true_cases
            split["false_cases"] = false_cases
            split["case_confounded"] = bool(case_confounded)
            splits.append(split)
    splits.sort(key=lambda row: (float(row.get("accuracy") or 0.0), int(row.get("correct") or 0)), reverse=True)
    return splits[:120]


def state_rows(model_name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    thresholds_path = PHASE930_ROOT / args.phase930_round / f"phase930_{model_name}_thresholds.jsonl"
    phase934_rows_path = PHASE934_ROOT / args.phase934_round / f"phase934_{model_name}_rows.jsonl"
    thresholds = read_jsonl(thresholds_path)
    rows934 = read_jsonl(phase934_rows_path)
    if not thresholds or not rows934:
        return []
    success_maps = {
        "fixed": state_success_map(rows934, "fixed_topfreq_64", 2.25),
        "true_inter": state_success_map(rows934, "fixed_plus_loso_case_inter_residual", 2.25),
        "true_union": state_success_map(rows934, "fixed_plus_loso_case_union_residual", 2.25),
        "noncase_inter": state_success_map(rows934, "fixed_plus_noncase_inter_size_control", 2.25),
        "global_inter": state_success_map(rows934, "fixed_plus_global_inter_size_control", 2.25),
        "random_inter": state_success_map(rows934, "fixed_plus_pseudorandom_inter_size_control", 2.25),
        "noncase_union": state_success_map(rows934, "fixed_plus_noncase_union_size_control", 2.25),
        "global_union": state_success_map(rows934, "fixed_plus_global_union_size_control", 2.25),
        "random_union": state_success_map(rows934, "fixed_plus_pseudorandom_union_size_control", 2.25),
    }
    out = []
    for feat in thresholds:
        key = str(feat.get("target_state_key"))
        fixed = bool(success_maps["fixed"].get(key))
        true_success = bool(success_maps["true_inter"].get(key) or success_maps["true_union"].get(key))
        control_success = bool(
            success_maps["noncase_inter"].get(key)
            or success_maps["global_inter"].get(key)
            or success_maps["random_inter"].get(key)
            or success_maps["noncase_union"].get(key)
            or success_maps["global_union"].get(key)
            or success_maps["random_union"].get(key)
        )
        row = dict(feat)
        row["phase"] = PHASE
        row["row_kind"] = "phase935_case_residual_gate_candidate_state"
        row["fixed_success_2_25"] = fixed
        row["residual_needed_2_25"] = not fixed
        row["true_loso_repair_success_2_25"] = true_success
        row["size_control_success_2_25"] = control_success
        row["true_beats_controls_2_25"] = bool(true_success and not control_success)
        row["target_case_is_chair"] = str(row.get("target_case_id")) == "p856_035_object_chair"
        out.append(row)
    out.sort(key=lambda row: (str(row.get("target_case_id")), str(row.get("target_state_key"))))
    return out


def evidence_label(rows: list[dict[str, Any]], splits: list[dict[str, Any]]) -> str:
    if not rows:
        return "no_phase934_gate_data"
    strong = [row for row in splits if float(row.get("accuracy") or 0.0) >= 0.9]
    if any(not row.get("case_confounded") for row in strong):
        return "case_residual_gate_candidate_not_case_confounded"
    if strong:
        return "case_residual_gate_candidate_case_confounded"
    return "no_strong_case_residual_gate_candidate"


def eval_model(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = RESULT_ROOT / args.round_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = state_rows(args.model, args)
    splits = feature_split_summary(rows, args) if rows else []
    cases = case_summary(rows) if rows else []
    label = evidence_label(rows, splits)
    payload = {
        "phase": PHASE,
        "title": "Case Residual Gate Candidate Audit",
        "model": args.model,
        "status": "complete" if rows else "no_phase934_gate_data",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "state_rows": len(rows),
        "case_summary": cases,
        "top_feature_splits": splits,
        "target_counts": {
            "fixed_success_2_25": sum(1 for row in rows if row.get("fixed_success_2_25")),
            "residual_needed_2_25": sum(1 for row in rows if row.get("residual_needed_2_25")),
            "true_loso_repair_success_2_25": sum(1 for row in rows if row.get("true_loso_repair_success_2_25")),
            "size_control_success_2_25": sum(1 for row in rows if row.get("size_control_success_2_25")),
            "true_beats_controls_2_25": sum(1 for row in rows if row.get("true_beats_controls_2_25")),
        },
        "evidence_label": label,
        "boundary": "observational gate-candidate audit only; no causal natural gate claim",
    }
    p846.write_json(out_dir / f"phase935_{args.model}_summary.json", payload)
    p846.write_jsonl(out_dir / f"phase935_{args.model}_state_rows.jsonl", rows)
    print(json.dumps({"phase": PHASE, "model": args.model, "status": label, "states": len(rows), "target_counts": payload["target_counts"]}, ensure_ascii=False, indent=2), flush=True)
    return payload


def summarize_round(round_name: str) -> dict[str, Any]:
    out_dir = RESULT_ROOT / round_name
    summaries = [read_json(out_dir / f"phase935_{model}_summary.json") for model in MODELS]
    summaries = [summary for summary in summaries if summary]
    evidence_counts: Counter[str] = Counter()
    all_splits = []
    overall: dict[str, int] = {}
    for summary in summaries:
        evidence_counts[str(summary.get("evidence_label"))] += 1
        overall["state_rows"] = overall.get("state_rows", 0) + int(summary.get("state_rows") or 0)
        for key, value in (summary.get("target_counts") or {}).items():
            overall[key] = overall.get(key, 0) + int(value or 0)
        for split in summary.get("top_feature_splits") or []:
            item = dict(split)
            item["model"] = summary.get("model")
            all_splits.append(item)
    all_splits.sort(key=lambda row: (float(row.get("accuracy") or 0.0), int(row.get("correct") or 0)), reverse=True)
    payload = {
        "phase": PHASE,
        "round": round_name,
        "status": "complete" if summaries else "missing",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [summary.get("model") for summary in summaries],
        "overall": overall,
        "evidence_label_counts": dict(evidence_counts),
        "top_feature_splits": all_splits[:120],
        "model_summaries": summaries,
    }
    p846.write_json(out_dir / "phase935_cross_model_summary.json", payload)
    write_summary_md(out_dir / "phase935_cross_model_summary.md", payload)
    return payload


def write_summary_md(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Phase 935 case residual gate candidate audit", "", "## Overall", ""]
    for key, value in sorted((payload.get("overall") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Evidence", ""]
    for key, value in sorted((payload.get("evidence_label_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines += ["", "## Top Feature Splits", ""]
    lines.append("| model | target | feature | threshold | polarity | accuracy | true | false | case_confounded |")
    lines.append("| --- | --- | --- | ---: | --- | ---: | ---: | ---: | --- |")
    for row in payload.get("top_feature_splits") or []:
        lines.append(
            "| {model} | {target} | {feature} | {threshold} | {polarity} | {accuracy} | {true_count} | {false_count} | {case_confounded} |".format(
                **row
            )
        )
    lines += ["", "## Case Summary", ""]
    for summary in payload.get("model_summaries") or []:
        if not summary.get("case_summary"):
            continue
        lines.append(f"### {summary.get('model')}")
        lines.append("")
        for row in summary.get("case_summary") or []:
            lines.append(
                "- {target_case_id}: states={states}, fixed={fixed_success_2_25}, residual_needed={residual_needed_2_25}, true_repair={true_loso_repair_success_2_25}, size_control={size_control_success_2_25}, true_beats_controls={true_beats_controls_2_25}".format(
                    **row
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round-name", default="case_residual_gate_candidate_audit")
    parser.add_argument("--phase930-round", default="natural_gate_strict_clean_transition_audit")
    parser.add_argument("--phase934-round", default="case_residual_size_control_audit")
    parser.add_argument("--gate-features", default="opening_threshold_factor,target_route_delta_norm,target_boundary_eos_margin_vs_blocker,target_boundary_eos_rank,boundary_period_gap_vs_eos,boundary_punctuation_gap_vs_eos,l39_activation_abs_top,l39_activation_abs_median,l39_margin_pos_mean_score,l39_margin_pos_max_score,l39_margin_pos_min_score,l39_eos_support_mean_score,l39_neg_margin_mean_score,phase925_factor")
    parser.add_argument("--targets", default="residual_needed_2_25,true_beats_controls_2_25,fixed_success_2_25")
    parser.add_argument("--summarize-round", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.summarize_round:
        payload = summarize_round(args.round_name)
        print(json.dumps({"phase": PHASE, "status": payload["status"], "overall": payload["overall"]}, ensure_ascii=False, indent=2))
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-round is set")
    eval_model(args)


if __name__ == "__main__":
    main()
