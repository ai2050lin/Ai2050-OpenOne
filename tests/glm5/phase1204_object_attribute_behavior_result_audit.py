#!/usr/bin/env python3
"""Independent recomputation audit for all Phase1204 behavior ledgers."""

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
AUDIT_PATH = OUT_ROOT / "audit/independent_result_audit.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def add(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None) -> None:
    checks.append({"name": name, "pass": bool(passed), "detail": detail})


def grouped(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    result: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def rate(values: Iterable[bool]) -> float:
    items = [bool(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def recompute_rows(model_name: str, contract: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = execution.load_manifest(model_name, contract)
    raw = execution.read_jsonl(execution.raw_path(model_name))
    summary = execution.read_json(execution.summary_path(model_name))
    execution.validate_embedded_digest(summary, "summary_digest")
    if len(raw) != len(manifest) or execution.digest(raw) != summary["raw_digest"]:
        raise RuntimeError(f"{model_name} raw digest mismatch")
    rows: list[dict[str, Any]] = []
    positions_by_length: dict[int, int] = defaultdict(int)
    counts_by_length: dict[int, int] = defaultdict(int)
    for item in manifest:
        counts_by_length[int(item["input_length"])] += 1
    for item, observed in zip(manifest, raw):
        if item["item_id"] != observed["item_id"] or item["execution_index"] != observed["execution_index"]:
            raise RuntimeError(f"{model_name} raw order mismatch")
        labels = list(item["candidate_labels"])
        score_values = observed["candidate_scores"]
        if set(score_values) != set(labels):
            raise RuntimeError(f"{model_name} candidate labels mismatch")
        score_finite = all(value is not None and math.isfinite(float(value)) for value in score_values.values())
        finite = bool(observed["all_vocab_logits_finite"] and score_finite)
        if finite:
            scores = {label: float(score_values[label]) for label in labels}
            ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
            tie = ranked[0][1] - ranked[1][1] <= phase1203.TIE_TOLERANCE
            prediction = "UNRESOLVED_TIE" if tie else ranked[0][0]
        else:
            tie = False
            prediction = "NONFINITE"
        correct = bool(finite and not tie and prediction == item["gold_candidate"])
        if observed["prediction"] != prediction or observed["correct"] is not correct:
            raise RuntimeError(f"{model_name} stored prediction mismatch")
        length = int(item["input_length"])
        position = positions_by_length[length]
        full_batch = phase1203.MODEL_BATCH_SIZE[model_name]
        remaining = counts_by_length[length] - (position // full_batch) * full_batch
        expected_runtime_batch = min(full_batch, remaining)
        if observed["runtime_batch_size"] != expected_runtime_batch:
            raise RuntimeError(f"{model_name} runtime batch mismatch")
        positions_by_length[length] += 1
        rows.append(
            {
                **{key: item[key] for key in (
                    "split", "panel", "world", "attribute", "template", "gold_position",
                    "candidate_order", "binding_state", "combination_id",
                )},
                "finite": finite,
                "correct": correct,
                "prediction": prediction,
                "tie": tie,
            }
        )
    return rows, summary


def panel_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    output: dict[str, float] = {}
    for panel in phase1203.PANELS:
        members = [row for row in rows if row["panel"] == panel]
        groups = grouped(members, ("combination_id", "panel", "template", "candidate_order"))
        outcomes: list[bool] = []
        for pair in groups.values():
            if len(pair) != 2 or {row["binding_state"] for row in pair} != {0, 1}:
                raise RuntimeError("panel pair is incomplete")
            pair = sorted(pair, key=lambda row: row["binding_state"])
            correct = all(row["finite"] and row["correct"] for row in pair)
            identity_relation = pair[0]["prediction"] != pair[1]["prediction"] if panel == "active" else pair[0]["prediction"] == pair[1]["prediction"]
            outcomes.append(correct and identity_relation)
        output[panel] = rate(outcomes)
    return output


def invariance(rows: list[dict[str, Any]], axis: str) -> float:
    if axis == "candidate_order":
        key = ("combination_id", "panel", "template", "binding_state")
        levels: set[Any] = {0, 1, 2}
    else:
        key = ("combination_id", "panel", "candidate_order", "binding_state")
        levels = {"profile_prose", "compact_ledger"}
    outcomes: list[bool] = []
    for members in grouped(rows, key).values():
        if {row[axis] for row in members} != levels or len(members) != len(levels):
            raise RuntimeError(f"{axis} group is incomplete")
        outcomes.append(
            all(row["finite"] and row["correct"] for row in members)
            and len({row["prediction"] for row in members}) == 1
        )
    return rate(outcomes)


def recompute_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    marginal: dict[str, dict[str, float]] = {}
    for axis in phase1203.WORST_CELL_AXES:
        marginal[axis] = {
            str(level[0]): rate(row["correct"] for row in members)
            for level, members in grouped(rows, (axis,)).items()
        }
    worst_axis, worst_level, worst_value = min(
        ((axis, level, value) for axis, cells in marginal.items() for level, value in cells.items()),
        key=lambda item: item[2],
    )
    finite_rate = rate(row["finite"] for row in rows)
    accuracy = rate(row["correct"] for row in rows)
    panels = panel_rates(rows)
    order = invariance(rows, "candidate_order")
    template = invariance(rows, "template")
    ledgers = {
        "L1_numerical": finite_rate >= phase1203.FINITE_RATE_MIN,
        "L2_identity": accuracy >= phase1203.OVERALL_ACCURACY_MIN and worst_value >= phase1203.WORST_MARGINAL_CELL_MIN,
        "L3_panel_logic": all(value >= phase1203.PANEL_PAIR_MIN for value in panels.values()),
        "L4_interface_invariance": order >= phase1203.CANDIDATE_ORDER_INVARIANCE_MIN and template >= phase1203.TEMPLATE_INVARIANCE_MIN,
    }
    return {
        "case_count": len(rows),
        "finite_rate": finite_rate,
        "accuracy": accuracy,
        "marginal_cells": marginal,
        "worst_marginal_cell": {"axis": worst_axis, "level": worst_level, "accuracy": worst_value},
        "panel_pair_success": panels,
        "candidate_order_invariance": order,
        "template_invariance": template,
        "ledger_pass": ledgers,
        "split_pass": all(ledgers.values()),
    }


def recompute_model(model_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    rows, summary = recompute_rows(model_name, contract)
    overall_finite = rate(row["finite"] for row in rows)
    splits = {
        split: recompute_split([row for row in rows if row["split"] == split])
        for split in phase1203.SPLITS
    }
    finite_pass = overall_finite >= phase1203.FINITE_RATE_MIN
    return {
        "model": model_name,
        "case_count": len(rows),
        "raw_digest": summary["raw_digest"],
        "run_summary_digest": summary["summary_digest"],
        "overall_finite_rate": overall_finite,
        "overall_finite_pass": finite_pass,
        "overall_accuracy_descriptive": rate(row["correct"] for row in rows),
        "unresolved_tie_rate_descriptive": rate(row["tie"] for row in rows),
        "splits": splits,
        "model_pass": finite_pass and all(value["split_pass"] for value in splits.values()),
        "precision_audit": summary["precision_audit"],
        "placement": summary["placement"],
        "elapsed_seconds": summary["runtime"]["elapsed_seconds"],
    }


def audit(write: bool) -> dict[str, Any]:
    if AUDIT_PATH.exists() and write:
        raise RuntimeError("Phase1204 independent result audit already exists")
    contract = execution.verify_contract()
    preaudit = execution.verify_preexecution_audit(contract)
    verdict = execution.read_json(VERDICT_PATH)
    execution.validate_embedded_digest(verdict, "verdict_digest")
    checks: list[dict[str, Any]] = []
    add(checks, "phase", verdict.get("phase") == PHASE)
    add(checks, "contract_link", verdict.get("contract_digest") == contract["contract_digest"])
    add(checks, "preexecution_gate", preaudit.get("gate_pass") is True)
    add(checks, "preexecution_was_zero_output", preaudit.get("behavior_cases_scored") == 0)
    recomputed: dict[str, Any] = {}
    for model_name in execution.MODEL_ORDER:
        model = recompute_model(model_name, contract)
        recomputed[model_name] = model
        add(checks, f"{model_name}_case_count", model["case_count"] == execution.EXPECTED_CASES)
        add(checks, f"{model_name}_metrics", execution.digest(model) == execution.digest(verdict["models"][model_name]))
        precision = model["precision_audit"]
        add(
            checks,
            f"{model_name}_fp16_no_quantization",
            precision["has_fp16_parameters"]
            and not precision["has_bf16_parameters"]
            and not precision["has_quantized_modules"]
            and set(precision["parameter_dtypes"]) == {"float16"},
        )
        add(checks, f"{model_name}_all_splits", set(model["splits"]) == set(phase1203.SPLITS))
        add(checks, f"{model_name}_no_case_deletion", sum(split["case_count"] for split in model["splits"].values()) == execution.EXPECTED_CASES)
    passing = [model for model in execution.MODEL_ORDER if recomputed[model]["model_pass"]]
    cross_pass = len(passing) >= phase1203.MIN_CROSS_MODEL_PASSES
    add(checks, "passing_models", passing == verdict["passing_models"], passing)
    add(checks, "passing_count", len(passing) == verdict["passing_model_count"])
    add(checks, "cross_model_gate", cross_pass is verdict["cross_model_behavior_pass"])
    add(checks, "no_hidden_claim", verdict["claim_boundary"]["hidden_state_evidence"] is False)
    add(checks, "no_causal_claim", verdict["claim_boundary"]["causal_evidence"] is False)
    add(checks, "no_natural_use_claim", verdict["claim_boundary"]["natural_use_evidence"] is False)
    gate = all(check["pass"] for check in checks)
    output: dict[str, Any] = {
        "phase": PHASE,
        "kind": "independent_phase1204_result_recomputation",
        "contract_digest": contract["contract_digest"],
        "verdict_digest": verdict["verdict_digest"],
        "gate_pass": gate,
        "checks_passed": sum(check["pass"] for check in checks),
        "checks_total": len(checks),
        "checks": checks,
        "recomputed_models": recomputed,
        "passing_models": passing,
        "cross_model_behavior_pass": cross_pass,
    }
    output["audit_digest"] = execution.digest(output)
    if write:
        write_json(AUDIT_PATH, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    output = audit(args.write)
    print(json.dumps({
        "gate_pass": output["gate_pass"],
        "checks_passed": output["checks_passed"],
        "checks_total": output["checks_total"],
        "passing_models": output["passing_models"],
        "cross_model_behavior_pass": output["cross_model_behavior_pass"],
        "audit_digest": output["audit_digest"],
    }, ensure_ascii=False, indent=2))
    if not output["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
