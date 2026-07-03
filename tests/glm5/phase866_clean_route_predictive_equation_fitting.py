#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase846_geometry_boundary_equation_fitting as p846  # noqa: E402


PHASE = 866
PHASE865_ROWS = Path("tests/result/phase865_route_purity_and_side_effect_filter/phase865_route_purity_rows.jsonl")
RESULT_ROOT = Path("tests/result/phase866_clean_route_predictive_equation_fitting")


def finite(value: Any, default: float = 0.0) -> float:
    return p846.finite(value, default)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return p846.read_jsonl(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def rule_answer_blocker_object(row: dict[str, Any], object_delta_threshold: float) -> bool:
    return (
        finite(row.get("mean_answer_delta")) > 0
        and finite(row.get("mean_class_blocker_reduction")) > 0
        and finite(row.get("mean_original_blocker_delta")) < 0
        and finite(row.get("mean_object_delta")) <= float(object_delta_threshold)
        and int(row.get("object_echo_induced") or 0) == 0
        and int(row.get("format_or_other_induced") or 0) == 0
    )


def rule_answer_blocker_only(row: dict[str, Any], _object_delta_threshold: float) -> bool:
    return (
        finite(row.get("mean_answer_delta")) > 0
        and finite(row.get("mean_class_blocker_reduction")) > 0
        and finite(row.get("mean_original_blocker_delta")) < 0
    )


def rule_answer_only(row: dict[str, Any], object_delta_threshold: float) -> bool:
    return (
        finite(row.get("mean_answer_delta")) > 0
        and finite(row.get("mean_object_delta")) <= float(object_delta_threshold)
        and int(row.get("object_echo_induced") or 0) == 0
    )


RULES = {
    "answer_blocker_object_rule": rule_answer_blocker_object,
    "answer_blocker_only_rule": rule_answer_blocker_only,
    "answer_only_object_rule": rule_answer_only,
}


def evaluate_rule(rows: list[dict[str, Any]], rule_name: str, object_delta_threshold: float) -> dict[str, Any]:
    rule = RULES[rule_name]
    details = []
    counts = Counter()
    for row in rows:
        target = str(row.get("purity_class")) == "clean_mixed_answer_blocker_route"
        pred = bool(rule(row, object_delta_threshold))
        if pred and target:
            counts["tp"] += 1
        elif pred and not target:
            counts["fp"] += 1
        elif not pred and target:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
        details.append(
            {
                "rule": rule_name,
                "prediction": pred,
                "target_clean_mixed": target,
                "correct": pred == target,
                "model": row.get("model"),
                "domain": row.get("domain"),
                "condition_type": row.get("condition_type"),
                "subset_name": row.get("subset_name"),
                "edit_mode": row.get("edit_mode"),
                "purity_class": row.get("purity_class"),
                "route_class": row.get("route_class"),
                "mean_answer_delta": row.get("mean_answer_delta"),
                "mean_class_blocker_reduction": row.get("mean_class_blocker_reduction"),
                "mean_original_blocker_delta": row.get("mean_original_blocker_delta"),
                "mean_object_delta": row.get("mean_object_delta"),
                "object_echo_induced": row.get("object_echo_induced"),
                "format_or_other_induced": row.get("format_or_other_induced"),
            }
        )
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    return {
        "rule": rule_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / max(1, tp + fp + fn + tn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "details": details,
    }


def selected_rows(scope: str) -> list[dict[str, Any]]:
    rows = read_jsonl(PHASE865_ROWS)
    if scope == "full_set":
        return [row for row in rows if row.get("condition_type") == "full_set"]
    if scope == "dominant_channel":
        return [
            row
            for row in rows
            if row.get("condition_type") == "single_channel"
            and any(str(role).startswith("dominant") for role in row.get("channel_role_classes") or [])
        ]
    if scope == "full_and_dominant":
        return [
            row
            for row in rows
            if row.get("condition_type") == "full_set"
            or (
                row.get("condition_type") == "single_channel"
                and any(str(role).startswith("dominant") for role in row.get("channel_role_classes") or [])
            )
        ]
    return rows


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 866 Clean Route Predictive Equation Fitting",
        "",
        "- Source: Phase 865 route purity rows.",
        "- Boundary: simple empirical rule check, not a learned model and not closure.",
        "",
        "## Rule Results",
        "",
        "| scope | rule | TP | FP | FN | TN | precision | recall | accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("rule_results") or []:
        lines.append(
            f"| {row.get('scope')} | `{row.get('rule')}` | {row.get('tp')} | {row.get('fp')} | {row.get('fn')} | {row.get('tn')} | "
            f"{float(row.get('precision')):.3f} | {float(row.get('recall')):.3f} | {float(row.get('accuracy')):.3f} |"
        )
    lines += [
        "",
        "## Selected Equation",
        "",
        "```text",
        "CleanMixedRoute(g,d,m) =",
        "  [answer_delta > 0]",
        "  and [blocker_reduction > 0]",
        "  and [original_blocker_delta < 0]",
        "  and [object_delta <= 0.25]",
        "  and [object_echo_induced = 0]",
        "  and [format_or_other_induced = 0]",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-delta-threshold", type=float, default=0.25)
    parser.add_argument("--output-dir", default=str(RESULT_ROOT))
    args = parser.parse_args()
    rule_results = []
    detail_rows = []
    for scope in ("full_set", "dominant_channel", "full_and_dominant"):
        rows = selected_rows(scope)
        for rule_name in RULES:
            result = evaluate_rule(rows, rule_name, float(args.object_delta_threshold))
            detail_rows.extend(result.pop("details"))
            result["scope"] = scope
            result["n_rows"] = len(rows)
            rule_results.append(result)
    payload = {
        "phase": PHASE,
        "title": "Clean Route Predictive Equation Fitting",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(PHASE865_ROWS),
        "object_delta_threshold": float(args.object_delta_threshold),
        "rule_results": rule_results,
        "selected_rule": "answer_blocker_object_rule",
        "boundary": "basic rule evaluation over existing route rows; no new model intervention",
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "phase866_summary.json", payload)
    write_jsonl(out_dir / "phase866_rule_details.jsonl", detail_rows)
    (out_dir / "phase866_summary.md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"phase": PHASE, "selected_rule": payload["selected_rule"], "rule_results": rule_results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
