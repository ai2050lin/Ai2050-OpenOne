#!/usr/bin/env python3
"""Recompute the five frozen Phase1203 behavior ledgers and seal Phase1204."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import phase1203_object_attribute_behavior_protocol as phase1203
import phase1204_object_attribute_behavior_execution as execution


PHASE = 1204
OUT_ROOT = execution.OUT_ROOT
VERDICT_PATH = OUT_ROOT / "analysis/behavior_verdict.json"
RESULT_AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"
FINAL_PATH = OUT_ROOT / "analysis/final.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def grouped(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def rate(values: Iterable[bool]) -> float:
    materialized = [bool(value) for value in values]
    return sum(materialized) / len(materialized) if materialized else 0.0


def close(left: float | None, right: float | None, tolerance: float = 2e-6) -> bool:
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def recompute_case(manifest: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    if raw["item_id"] != manifest["item_id"] or raw["execution_index"] != manifest["execution_index"]:
        raise RuntimeError("raw/manifest ordering mismatch")
    labels = list(manifest["candidate_labels"])
    if set(raw["candidate_scores"]) != set(labels):
        raise RuntimeError("candidate score labels mismatch")
    score_values = list(raw["candidate_scores"].values())
    score_finite = all(value is not None and math.isfinite(float(value)) for value in score_values)
    finite = bool(raw["all_vocab_logits_finite"] and score_finite)
    scores = {label: float(raw["candidate_scores"][label]) for label in labels} if score_finite else {}
    if finite:
        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        gap = ranked[0][1] - ranked[1][1]
        tie = gap <= phase1203.TIE_TOLERANCE
        prediction = "UNRESOLVED_TIE" if tie else ranked[0][0]
        gold = manifest["gold_candidate"]
        gold_margin = scores[gold] - max(value for label, value in scores.items() if label != gold)
    else:
        gap = None
        tie = False
        prediction = "NONFINITE"
        gold_margin = None
    correct = bool(finite and not tie and prediction == manifest["gold_candidate"])
    if raw["prediction"] != prediction or raw["correct"] is not correct:
        raise RuntimeError(f"stored verdict mismatch for {manifest['item_id']}")
    if raw["unresolved_tie"] is not tie:
        raise RuntimeError(f"stored tie mismatch for {manifest['item_id']}")
    if not close(raw["top_two_gap"], gap) or not close(raw["gold_margin"], gold_margin):
        raise RuntimeError(f"stored margin mismatch for {manifest['item_id']}")
    return {
        **{key: manifest[key] for key in (
            "execution_index", "item_id", "split", "panel", "world", "attribute",
            "template", "gold_position", "candidate_order", "binding_state", "combination_id",
            "gold_candidate", "input_length",
        )},
        "finite": finite,
        "prediction": prediction,
        "correct": correct,
        "gold_margin": gold_margin,
        "unresolved_tie": tie,
    }


def panel_pair_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for panel in phase1203.PANELS:
        panel_rows = [row for row in rows if row["panel"] == panel]
        groups = grouped(panel_rows, ("combination_id", "panel", "template", "candidate_order"))
        success: list[bool] = []
        for members in groups.values():
            if len(members) != 2 or {row["binding_state"] for row in members} != {0, 1}:
                raise RuntimeError(f"incomplete panel pair for {panel}")
            ordered = sorted(members, key=lambda row: row["binding_state"])
            common = all(row["finite"] and row["correct"] for row in ordered)
            if panel == "active":
                success.append(common and ordered[0]["prediction"] != ordered[1]["prediction"])
            else:
                success.append(common and ordered[0]["prediction"] == ordered[1]["prediction"])
        result[panel] = rate(success)
    return result


def invariance_rate(rows: list[dict[str, Any]], variant: str) -> float:
    if variant == "candidate_order":
        fields = ("combination_id", "panel", "template", "binding_state")
        expected_count = 3
        expected_levels = {0, 1, 2}
    else:
        fields = ("combination_id", "panel", "candidate_order", "binding_state")
        expected_count = 2
        expected_levels = set(phase1203.read_json(execution.UPSTREAM_PROTOCOL)["five_behavior_ledgers"]["L2_identity"].get("template_levels", []))
        if not expected_levels:
            expected_levels = {"profile_prose", "compact_ledger"}
    groups = grouped(rows, fields)
    success: list[bool] = []
    for members in groups.values():
        if len(members) != expected_count or {row[variant] for row in members} != expected_levels:
            raise RuntimeError(f"incomplete {variant} invariance group")
        success.append(
            all(row["finite"] and row["correct"] for row in members)
            and len({row["prediction"] for row in members}) == 1
        )
    return rate(success)


def split_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finite_rate = rate(row["finite"] for row in rows)
    accuracy = rate(row["correct"] for row in rows)
    marginal_cells: dict[str, dict[str, float]] = {}
    for axis in phase1203.WORST_CELL_AXES:
        cells: dict[str, float] = {}
        for level, members in grouped(rows, (axis,)).items():
            cells[str(level[0])] = rate(row["correct"] for row in members)
        marginal_cells[axis] = cells
    worst_axis, worst_level, worst_value = min(
        (
            (axis, level, value)
            for axis, cells in marginal_cells.items()
            for level, value in cells.items()
        ),
        key=lambda item: item[2],
    )
    panels = panel_pair_rates(rows)
    order_invariance = invariance_rate(rows, "candidate_order")
    template_invariance = invariance_rate(rows, "template")
    ledger_pass = {
        "L1_numerical": finite_rate >= phase1203.FINITE_RATE_MIN,
        "L2_identity": accuracy >= phase1203.OVERALL_ACCURACY_MIN
        and worst_value >= phase1203.WORST_MARGINAL_CELL_MIN,
        "L3_panel_logic": all(value >= phase1203.PANEL_PAIR_MIN for value in panels.values()),
        "L4_interface_invariance": order_invariance >= phase1203.CANDIDATE_ORDER_INVARIANCE_MIN
        and template_invariance >= phase1203.TEMPLATE_INVARIANCE_MIN,
    }
    return {
        "case_count": len(rows),
        "finite_rate": finite_rate,
        "accuracy": accuracy,
        "marginal_cells": marginal_cells,
        "worst_marginal_cell": {
            "axis": worst_axis,
            "level": worst_level,
            "accuracy": worst_value,
        },
        "panel_pair_success": panels,
        "candidate_order_invariance": order_invariance,
        "template_invariance": template_invariance,
        "ledger_pass": ledger_pass,
        "split_pass": all(ledger_pass.values()),
    }


def evaluate_model(model_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    manifest = execution.load_manifest(model_name, contract)
    raw = execution.read_jsonl(execution.raw_path(model_name))
    summary = execution.read_json(execution.summary_path(model_name))
    execution.validate_embedded_digest(summary, "summary_digest")
    if summary["contract_digest"] != contract["contract_digest"]:
        raise RuntimeError(f"{model_name} contract link mismatch")
    if len(raw) != len(manifest) or execution.digest(raw) != summary["raw_digest"]:
        raise RuntimeError(f"{model_name} raw output mismatch")
    rows = [recompute_case(manifest_row, raw_row) for manifest_row, raw_row in zip(manifest, raw)]
    overall_finite_rate = rate(row["finite"] for row in rows)
    split_results = {
        split: split_metrics([row for row in rows if row["split"] == split])
        for split in phase1203.SPLITS
    }
    overall_finite_pass = overall_finite_rate >= phase1203.FINITE_RATE_MIN
    model_pass = overall_finite_pass and all(result["split_pass"] for result in split_results.values())
    return {
        "model": model_name,
        "case_count": len(rows),
        "raw_digest": summary["raw_digest"],
        "run_summary_digest": summary["summary_digest"],
        "overall_finite_rate": overall_finite_rate,
        "overall_finite_pass": overall_finite_pass,
        "overall_accuracy_descriptive": rate(row["correct"] for row in rows),
        "unresolved_tie_rate_descriptive": rate(row["unresolved_tie"] for row in rows),
        "splits": split_results,
        "model_pass": model_pass,
        "precision_audit": summary["precision_audit"],
        "placement": summary["placement"],
        "elapsed_seconds": summary["runtime"]["elapsed_seconds"],
    }


def analyze() -> None:
    if VERDICT_PATH.exists() or RESULT_AUDIT_PATH.exists() or FINAL_PATH.exists():
        raise RuntimeError("Phase1204 analysis output already exists")
    contract = execution.verify_contract()
    execution.verify_preexecution_audit(contract)
    models = {model: evaluate_model(model, contract) for model in execution.MODEL_ORDER}
    passing_models = [model for model in execution.MODEL_ORDER if models[model]["model_pass"]]
    cross_model_pass = len(passing_models) >= phase1203.MIN_CROSS_MODEL_PASSES
    if cross_model_pass:
        status = "cross_model_behavior_qualified"
        proposed_k = {
            "id": "K184",
            "scope": "controlled object-attribute behavior",
            "statement": "At least two frozen FP16 model interfaces satisfy every sealed numerical, identity, panel, invariance, and unseen-composition ledger.",
        }
    elif len(passing_models) == 1:
        status = "model_specific_behavior_only"
        proposed_k = {
            "id": "K184",
            "scope": "cross-model behavior boundary",
            "statement": "Only one frozen FP16 model interface satisfies the complete object-attribute behavior contract; cross-model hidden claims are denied.",
        }
    else:
        status = "interface_behavior_failed"
        proposed_k = {
            "id": "K184",
            "scope": "interface behavior boundary",
            "statement": "No frozen FP16 model interface satisfies the complete object-attribute behavior contract; hidden-state access is denied for this interface.",
        }
    verdict: dict[str, Any] = {
        "phase": PHASE,
        "schema_version": "phase1204.object_attribute.behavior_verdict.v1",
        "contract_digest": contract["contract_digest"],
        "models": models,
        "passing_models": passing_models,
        "passing_model_count": len(passing_models),
        "cross_model_behavior_pass": cross_model_pass,
        "status": status,
        "proposed_k_item_pending_independent_audit": proposed_k,
        "claim_boundary": {
            "behavior_evidence": True,
            "hidden_state_evidence": False,
            "causal_evidence": False,
            "natural_use_evidence": False,
            "mechanism_claim": False,
        },
    }
    verdict["verdict_digest"] = execution.digest(verdict)
    write_json(VERDICT_PATH, verdict)
    print(json.dumps({
        "status": status,
        "passing_models": passing_models,
        "cross_model_behavior_pass": cross_model_pass,
        "verdict_digest": verdict["verdict_digest"],
    }, ensure_ascii=False, indent=2))


def finalize() -> None:
    if FINAL_PATH.exists():
        raise RuntimeError("Phase1204 final already exists")
    contract = execution.verify_contract()
    verdict = execution.read_json(VERDICT_PATH)
    result_audit = execution.read_json(RESULT_AUDIT_PATH)
    execution.validate_embedded_digest(verdict, "verdict_digest")
    execution.validate_embedded_digest(result_audit, "audit_digest")
    if not result_audit.get("gate_pass") or result_audit.get("verdict_digest") != verdict["verdict_digest"]:
        raise RuntimeError("Phase1204 independent result audit failed")
    cross_model_pass = bool(verdict["cross_model_behavior_pass"])
    final: dict[str, Any] = {
        "phase": PHASE,
        "status": verdict["status"],
        "contract_digest": contract["contract_digest"],
        "verdict_digest": verdict["verdict_digest"],
        "independent_result_audit_digest": result_audit["audit_digest"],
        "passing_models": verdict["passing_models"],
        "passing_model_count": verdict["passing_model_count"],
        "cross_model_behavior_pass": cross_model_pass,
        "new_k_item": verdict["proposed_k_item_pending_independent_audit"],
        "evidence_scope": verdict["claim_boundary"],
        "authorized_next": {
            "phase1205_hidden_specificity_preregistration": cross_model_pass,
            "automatic_hidden_state_execution": False,
            "cross_model_hidden_claim": False,
            "causal_intervention": False,
            "natural_use_claim": False,
            "new_mechanism_algebra": False,
        },
        "stop_rule": (
            "At least two models passed; only a separate zero-output hidden-specificity preregistration is authorized."
            if cross_model_pass
            else "Fewer than two models passed; cross-model hidden-state work is denied for this frozen interface."
        ),
    }
    final["final_digest"] = execution.digest(final)
    write_json(FINAL_PATH, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze", "finalize"))
    args = parser.parse_args()
    {"analyze": analyze, "finalize": finalize}[args.command]()


if __name__ == "__main__":
    main()
