#!/usr/bin/env python3
"""Independent pre/final audit for Phase1286 C026 behavior mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tests/glm5/result/phase1285_c026_conditional_response_mapping_contract"
OUT = ROOT / "tests/glm5/result/phase1286_c026_qwen3_conditional_response_mapping_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = INPUT / "material/frozen_binary_status_worlds.jsonl"
RAW = OUT / "raw/response_scores.jsonl"
GENERATIONS = OUT / "raw/confirmation_generations.jsonl"
DECISION = OUT / "analysis/frozen_selection_decision.json"
FINAL = OUT / "analysis/final.json"
COMPLETE = OUT / "protocol/formal_run_complete.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"

SURFACES = ("official_decision", "binary_audit", "closed_review", "signed_assessment")
SOURCE = SURFACES[0]
TARGETS = SURFACES[1:]
PANELS = (
    "consistency", "reversal", "lexical_consistency", "lexical_reversal",
    "role_consistency", "role_reversal",
)
ROLES = ("expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1")
FAMILIES = ("H0_constant", "H1_identity", "H2_diagonal_affine", "H3_full_affine")
PERMUTATIONS = ((2, 3, 0, 1, 4, 5), (1, 0, 3, 2, 5, 4), (4, 5, 2, 3, 0, 1))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def close_tree(left: Any, right: Any, atol: float = 1.0e-10) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close_tree(left[key], right[key], atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close_tree(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=0.0))
    return left == right


def preaudit() -> None:
    protocol = read_json(PROTOCOL)
    parent = read_json(INPUT / "analysis/final.json")
    parent_audit = read_json(INPUT / "audit/independent_final_audit.json")
    dependencies = {
        "phase1285_protocol": INPUT / "protocol/preregistration.json",
        "phase1285_material": INPUT / "material/frozen_binary_status_worlds.jsonl",
        "phase1285_final": INPUT / "analysis/final.json",
        "phase1285_audit": INPUT / "audit/independent_final_audit.json",
    }
    checks = [
        check("phase_campaign_contract", protocol["phase"] == 1286 and protocol["campaign"] == "C026" and protocol["contract_id"] == "EXP-C026-WP01-001"),
        check("parent_authorization", parent["authorization"] == "phase1286_qwen3_conditional_response_mapping_behavior" and parent_audit["all_checks_passed"]),
        check("dependency_hashes", all(protocol["dependencies"][key] == file_sha256(path) for key, path in dependencies.items())),
        check("frozen_dimensions", protocol["row_count"] == 192 and protocol["context_count"] == 4608 and protocol["scored_sequence_count"] == 27648 and protocol["generation_count"] == 512),
        check("surface_role_panel_registry", tuple(protocol["surface_order"]) == SURFACES and tuple(protocol["role_order"]) == ROLES and tuple(protocol["panels"]) == PANELS),
        check("map_registry", protocol["source_surface"] == SOURCE and tuple(protocol["target_surfaces"]) == TARGETS and tuple(protocol["family_order"]) == FAMILIES),
        check("zero_models", tuple(tuple(value) for value in protocol["zero_models"]["fixed_role_permutations"]) == PERMUTATIONS),
        check("length_account", "token" in (INPUT / "protocol/preregistration.json").read_text(encoding="utf-8") and protocol["map_fit"]["selection"].startswith("Choose minimum")),
        check("generation_registry", protocol["generation"]["partition"] == "confirmation" and protocol["generation"]["max_new_tokens"] == 8 and protocol["generation"]["do_sample"] is False),
        check("unblinding_order", protocol["unblinding_order"][2].startswith("fit discovery") and protocol["unblinding_order"][3].startswith("write immutable") and protocol["unblinding_order"][4].startswith("evaluate behavior")),
        check("single_run_and_stop", protocol["formal_run_budget"] == 1 and len(protocol["hard_stops"]) >= 5),
    ]
    result = {
        "phase": 1286,
        "campaign": "C026",
        "audit_type": "independent_preaudit",
        "checks": checks,
        "passed_count": sum(value["passed"] for value in checks),
        "check_count": len(checks),
        "all_checks_passed": all(value["passed"] for value in checks),
    }
    atomic_json(PREAUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def response(keys: dict[tuple[str, str, str], dict[str, Any]], row_id: str, surface: str, right: str, left: str, account: str) -> np.ndarray:
    rv = np.asarray([keys[(row_id, surface, right)][account][role] for role in ROLES], dtype=np.float64)
    lv = np.asarray([keys[(row_id, surface, left)][account][role] for role in ROLES], dtype=np.float64)
    value = rv - lv
    return value - value.mean()


def signatures(raw: list[dict[str, Any]], rows: list[dict[str, Any]], account: str) -> dict[tuple[str, str], dict[str, Any]]:
    keys = {(value["row_id"], value["surface"], value["panel"]): value for value in raw}
    output = {}
    for row in rows:
        for surface in SURFACES:
            active = response(keys, row["row_id"], surface, "reversal", "consistency", account)
            lexical = response(keys, row["row_id"], surface, "lexical_reversal", "lexical_consistency", account)
            role = response(keys, row["row_id"], surface, "role_reversal", "role_consistency", account)
            target_scale = max(float(np.mean(np.abs(active[:4]))), 1.0e-12)
            output[(row["row_id"], surface)] = {
                "active": active, "lexical": lexical, "role": role,
                "effect": float(np.mean(active[2:4]) - np.mean(active[0:2])),
                "active_norm": float(np.linalg.norm(active)),
                "lexical_norm": float(np.linalg.norm(lexical)),
                "role_norm": float(np.linalg.norm(role)),
                "control_leakage": float(np.mean(np.abs(active[4:6])) / target_scale),
            }
    return output


def recompute_behavior(raw: list[dict[str, Any]], rows: list[dict[str, Any]], threshold: dict[str, float]) -> dict[str, Any]:
    primary = signatures(raw, rows, "mean_log_prob")
    total = signatures(raw, rows, "total_log_prob")
    surface_cells, axis_cells = {}, {}
    for partition in ("discovery", "selection", "confirmation"):
        part = [row for row in rows if row["partition"] == partition]
        axes = sorted({row["axis"] for row in part})
        for surface in SURFACES:
            values = [primary[(row["row_id"], surface)] for row in part]
            totals = [total[(row["row_id"], surface)] for row in part]
            active = float(np.median([value["active_norm"] for value in values]))
            lexical = float(np.median([value["lexical_norm"] for value in values]))
            role = float(np.median([value["role_norm"] for value in values]))
            surface_cells[f"{partition}.{surface}"] = {
                "n_worlds": len(values),
                "positive_fraction": float(np.mean([value["effect"] > 0 for value in values])),
                "median_effect": float(np.median([value["effect"] for value in values])),
                "median_active_norm": active,
                "median_lexical_norm": lexical,
                "median_role_norm": role,
                "lexical_norm_ratio": lexical / max(active, 1.0e-12),
                "role_norm_ratio": role / max(active, 1.0e-12),
                "median_active_minus_lexical_norm": float(np.median([value["active_norm"] - value["lexical_norm"] for value in values])),
                "median_active_minus_role_norm": float(np.median([value["active_norm"] - value["role_norm"] for value in values])),
                "median_control_leakage": float(np.median([value["control_leakage"] for value in values])),
                "total_mean_effect_sign_agreement": float(np.mean([(value["effect"] > 0) == (other["effect"] > 0) for value, other in zip(values, totals)])),
            }
            for axis in axes:
                subset = [primary[(row["row_id"], surface)] for row in part if row["axis"] == axis]
                axis_cells[f"{partition}.{surface}.{axis}"] = {
                    "n_worlds": len(subset),
                    "positive_fraction": float(np.mean([value["effect"] > 0 for value in subset])),
                    "median_effect": float(np.median([value["effect"] for value in subset])),
                    "median_active_norm": float(np.median([value["active_norm"] for value in subset])),
                }
    axis_counts = {}
    for partition in ("discovery", "selection", "confirmation"):
        axes = sorted({row["axis"] for row in rows if row["partition"] == partition})
        axis_counts[partition] = sum(
            min(axis_cells[f"{partition}.{surface}.{axis}"]["positive_fraction"] for surface in SURFACES)
            >= threshold["axis_positive_fraction_min"] for axis in axes
        )
    gates = {
        "finite": float(np.mean([value["finite"] for value in raw])) >= threshold["finite_fraction_min"],
        "positive_fraction": min(value["positive_fraction"] for value in surface_cells.values()) >= threshold["partition_surface_positive_fraction_min"],
        "median_effect": min(value["median_effect"] for value in surface_cells.values()) >= threshold["partition_surface_median_effect_min"],
        "active_norm": min(value["median_active_norm"] for value in surface_cells.values()) >= threshold["partition_surface_median_active_norm_min"],
        "axis_coverage": min(axis_counts.values()) >= threshold["axis_pass_count_per_partition_min"],
        "lexical_null": max(value["lexical_norm_ratio"] for value in surface_cells.values()) <= threshold["lexical_null_norm_ratio_max"],
        "role_null": max(value["role_norm_ratio"] for value in surface_cells.values()) <= threshold["role_null_norm_ratio_max"],
        "control_leakage": max(value["median_control_leakage"] for value in surface_cells.values()) <= threshold["control_leakage_ratio_max"],
        "length_sensitivity": min(value["total_mean_effect_sign_agreement"] for value in surface_cells.values()) >= threshold["total_mean_effect_sign_agreement_min"],
    }
    return {
        "finite_fraction": float(np.mean([value["finite"] for value in raw])),
        "surface_cells": surface_cells,
        "axis_cells": axis_cells,
        "axis_pass_counts": axis_counts,
        "gate_extrema": {
            "positive_fraction_min": min(value["positive_fraction"] for value in surface_cells.values()),
            "median_effect_min": min(value["median_effect"] for value in surface_cells.values()),
            "active_norm_min": min(value["median_active_norm"] for value in surface_cells.values()),
            "lexical_ratio_max": max(value["lexical_norm_ratio"] for value in surface_cells.values()),
            "role_ratio_max": max(value["role_norm_ratio"] for value in surface_cells.values()),
            "control_leakage_max": max(value["median_control_leakage"] for value in surface_cells.values()),
            "length_agreement_min": min(value["total_mean_effect_sign_agreement"] for value in surface_cells.values()),
            "axis_pass_count_min": min(axis_counts.values()),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def matrix(rows: list[dict[str, Any]], sig: dict[tuple[str, str], dict[str, Any]], partition: str, surface: str, kind: str = "active") -> np.ndarray:
    selected = sorted((row for row in rows if row["partition"] == partition), key=lambda row: row["row_id"])
    return np.stack([sig[(row["row_id"], surface)][kind] for row in selected], axis=0)


def fit(family: str, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if family == "H0_constant":
        return {"family": family, "mean": y.mean(axis=0).tolist()}
    if family == "H1_identity":
        return {"family": family}
    if family == "H2_diagonal_affine":
        gains, offsets = [], []
        for column in range(y.shape[1]):
            design = np.column_stack([x[:, column], np.ones(len(x))])
            w = np.linalg.solve(design.T @ design + np.diag([1.0e-3, 0.0]), design.T @ y[:, column])
            gains.append(float(w[0])); offsets.append(float(w[1]))
        return {"family": family, "gains": gains, "offsets": offsets}
    design = np.column_stack([x, np.ones(len(x))])
    w = np.linalg.solve(design.T @ design + np.diag([1.0e-2] * x.shape[1] + [0.0]), design.T @ y)
    return {"family": family, "matrix": w[:-1].tolist(), "offset": w[-1].tolist()}


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if model["family"] == "H0_constant":
        return np.repeat(np.asarray(model["mean"])[None, :], len(x), axis=0)
    if model["family"] == "H1_identity":
        return x.copy()
    if model["family"] == "H2_diagonal_affine":
        return x * np.asarray(model["gains"]) + np.asarray(model["offsets"])
    return x @ np.asarray(model["matrix"]) + np.asarray(model["offset"])


def metrics(y: np.ndarray, yhat: np.ndarray, h0: np.ndarray | None = None) -> dict[str, Any]:
    error = float(np.square(y - yhat).sum()); energy = max(float(np.square(y).sum()), 1.0e-12)
    cosines = [cosine(a, b) for a, b in zip(y, yhat)]
    result = {
        "n": len(y), "squared_error": error, "energy": energy,
        "nrmse": float(np.sqrt(error / energy)), "median_cosine": float(np.median(cosines)),
        "positive_fraction": float(np.mean(np.asarray(cosines) > 0)), "minimum_cosine": float(np.min(cosines)),
    }
    if h0 is not None:
        h0_error = max(float(np.square(y - h0).sum()), 1.0e-12)
        result["gain_over_h0"] = 1.0 - error / h0_error
        result["h0_nrmse"] = float(np.sqrt(h0_error / energy))
        result["nrmse_improvement_over_h0"] = result["h0_nrmse"] - result["nrmse"]
    return result


def select_family(rows: list[dict[str, Any]], sig: dict[tuple[str, str], dict[str, Any]], tolerance: float, account: str) -> dict[str, Any]:
    fits, selection, pooled = {}, {}, {}
    for family in FAMILIES:
        fits[family], selection[family] = {}, {}
        error = energy = 0.0
        for target in TARGETS:
            xd = matrix(rows, sig, "discovery", SOURCE); yd = matrix(rows, sig, "discovery", target)
            model = fit(family, xd, yd); fits[family][target] = model
            xs = matrix(rows, sig, "selection", SOURCE); ys = matrix(rows, sig, "selection", target)
            value = metrics(ys, predict(model, xs), predict(fit("H0_constant", xd, yd), xs))
            selection[family][target] = value; error += value["squared_error"]; energy += value["energy"]
        pooled[family] = float(np.sqrt(error / max(energy, 1.0e-12)))
    best = min(pooled.values()); eligible = [family for family in FAMILIES if pooled[family] <= best + tolerance]
    return {
        "account": account, "family_order": list(FAMILIES), "discovery_fits": fits,
        "selection_metrics": selection, "pooled_selection_nrmse": pooled,
        "minimum_selection_nrmse": best, "simplicity_tolerance": tolerance,
        "eligible_within_tolerance": eligible, "selected_family": eligible[0],
    }


def evaluate(rows: list[dict[str, Any]], sig: dict[tuple[str, str], dict[str, Any]], family: str, threshold: dict[str, float], account: str) -> dict[str, Any]:
    per_target, refits = {}, {}
    for target in TARGETS:
        x_train = np.concatenate([matrix(rows, sig, "discovery", SOURCE), matrix(rows, sig, "selection", SOURCE)])
        y_train = np.concatenate([matrix(rows, sig, "discovery", target), matrix(rows, sig, "selection", target)])
        model = fit(family, x_train, y_train); refits[target] = model
        x = matrix(rows, sig, "confirmation", SOURCE); y = matrix(rows, sig, "confirmation", target)
        h0 = predict(fit("H0_constant", x_train, y_train), x); yhat = predict(model, x)
        value = metrics(y, yhat, h0)
        perm = [metrics(y, yhat[:, list(p)])["nrmse"] for p in PERMUTATIONS]
        value["role_permutation_nrmse"] = perm; value["best_role_permutation_nrmse"] = min(perm)
        value["nrmse_improvement_over_best_role_permutation"] = min(perm) - value["nrmse"]
        value["nulls"] = {}
        for kind in ("lexical", "role"):
            xn = matrix(rows, sig, "confirmation", SOURCE, kind); yn = matrix(rows, sig, "confirmation", target, kind)
            null_train = np.concatenate([matrix(rows, sig, "discovery", target, kind), matrix(rows, sig, "selection", target, kind)])
            null_h0 = np.repeat(null_train.mean(axis=0, keepdims=True), len(xn), axis=0)
            value["nulls"][kind] = metrics(yn, predict(model, xn), null_h0)
        value["active_minus_lexical_gain"] = value["gain_over_h0"] - value["nulls"]["lexical"]["gain_over_h0"]
        value["active_minus_role_gain"] = value["gain_over_h0"] - value["nulls"]["role"]["gain_over_h0"]
        per_target[target] = value
    map_gates = {
        "median_cosine": min(v["median_cosine"] for v in per_target.values()) >= threshold["mapping_confirmation_median_cosine_min"],
        "positive_fraction": min(v["positive_fraction"] for v in per_target.values()) >= threshold["mapping_confirmation_positive_fraction_min"],
        "nrmse": max(v["nrmse"] for v in per_target.values()) <= threshold["mapping_confirmation_nrmse_max"],
        "h0_improvement": min(v["nrmse_improvement_over_h0"] for v in per_target.values()) >= threshold["mapping_nrmse_improvement_over_h0_min"],
        "role_permutation_improvement": min(v["nrmse_improvement_over_best_role_permutation"] for v in per_target.values()) >= threshold["mapping_nrmse_improvement_over_role_permutation_min"],
    }
    spec_gates = {
        "active_gain": min(v["gain_over_h0"] for v in per_target.values()) >= threshold["mapping_active_gain_min"],
        "lexical_advantage": min(v["active_minus_lexical_gain"] for v in per_target.values()) >= threshold["mapping_active_minus_lexical_gain_min"],
        "role_advantage": min(v["active_minus_role_gain"] for v in per_target.values()) >= threshold["mapping_active_minus_role_gain_min"],
    }
    return {
        "account": account, "selected_family": family, "refits": refits, "confirmation": per_target,
        "map_gates": map_gates, "mapping_passed": all(map_gates.values()),
        "specificity_gates": spec_gates, "specificity_passed": all(spec_gates.values()),
    }


def generation_summary(rows: list[dict[str, Any]], threshold: dict[str, float]) -> dict[str, Any]:
    cells = {}
    for surface in SURFACES:
        for panel in ("consistency", "reversal"):
            subset = [row for row in rows if row["surface"] == surface and row["panel"] == panel]
            parsed = [row for row in subset if row["parsed"]]
            cells[f"{surface}.{panel}"] = {
                "n": len(subset), "coverage": float(np.mean([row["parsed"] for row in subset])),
                "accuracy_given_parsed": float(np.mean([row["correct"] for row in parsed])) if parsed else 0.0,
            }
    gates = {
        "coverage": min(value["coverage"] for value in cells.values()) >= threshold["generation_coverage_min"],
        "accuracy": min(value["accuracy_given_parsed"] for value in cells.values()) >= threshold["generation_accuracy_min"],
    }
    return {"cells": cells, "coverage_min": min(v["coverage"] for v in cells.values()), "accuracy_min": min(v["accuracy_given_parsed"] for v in cells.values()), "gates": gates, "passed": all(gates.values())}


def final_audit() -> None:
    protocol = read_json(PROTOCOL); final = read_json(FINAL); complete = read_json(COMPLETE); decision = read_json(DECISION)
    rows = read_jsonl(MATERIAL); raw = read_jsonl(RAW); generations = read_jsonl(GENERATIONS)
    threshold = protocol["thresholds"]
    recomputed_behavior = recompute_behavior(raw, rows, threshold)
    recomputed_generation = generation_summary(generations, threshold)
    primary_sig = signatures(raw, rows, "mean_log_prob")
    total_sig = signatures(raw, rows, "total_log_prob")
    primary_selection = select_family(rows, primary_sig, threshold["selection_simplicity_tolerance"], "mean_log_prob")
    primary_mapping = evaluate(rows, primary_sig, primary_selection["selected_family"], threshold, "mean_log_prob")
    total_selection = select_family(rows, total_sig, threshold["selection_simplicity_tolerance"], "total_log_prob")
    total_mapping = evaluate(rows, total_sig, total_selection["selected_family"], threshold, "total_log_prob")
    all_passed = recomputed_behavior["passed"] and recomputed_generation["passed"] and primary_mapping["mapping_passed"] and primary_mapping["specificity_passed"]
    decision_without_digest = {key: value for key, value in decision.items() if key != "decision_digest"}
    expected_authorization = "phase1287_qwen3_hidden_conditional_mapping" if all_passed else "stop_c026_at_qwen_behavior_mapping"
    precision = final["precision_audit"]
    hidden_artifacts = list(OUT.rglob("*hidden*"))
    checks = [
        check("raw_counts", len(raw) == 4608 and len(generations) == 512, {"raw": len(raw), "generations": len(generations)}),
        check("raw_unique_and_complete", len({(v["row_id"], v["surface"], v["panel"]) for v in raw}) == 4608 and all(set(v["mean_log_prob"]) == set(ROLES) and set(v["total_log_prob"]) == set(ROLES) for v in raw)),
        check("all_finite", all(value["finite"] for value in raw)),
        check("behavior_recompute", close_tree(recomputed_behavior, final["behavior"])),
        check("generation_recompute", close_tree(recomputed_generation, final["generation"])),
        check("selection_recompute", decision["selected_family"] == primary_selection["selected_family"] and close_tree(decision["pooled_selection_nrmse"], primary_selection["pooled_selection_nrmse"]) and close_tree(decision["selection_metrics"], primary_selection["selection_metrics"])),
        check("selection_decision_digest", decision["confirmation_mapping_metrics_read"] is False and decision["decision_digest"] == digest(decision_without_digest)),
        check("selection_before_completion", datetime.fromisoformat(decision["written_at_utc"]) < datetime.fromisoformat(complete["completed_at_utc"])),
        check("primary_mapping_recompute", close_tree(primary_mapping, final["mapping"])),
        check("selection_final_summary", final["selection"]["selected_family"] == primary_selection["selected_family"] and close_tree(final["selection"]["pooled_selection_nrmse"], primary_selection["pooled_selection_nrmse"]) and final["selection"]["decision_digest"] == decision["decision_digest"]),
        check("total_length_sensitivity", final["length_sensitivity"] == {"selected_family": total_selection["selected_family"], "mapping_passed": total_mapping["mapping_passed"], "specificity_passed": total_mapping["specificity_passed"]}),
        check("verdict_and_authorization", final["authorization"] == expected_authorization and (final["verdict"] == "qwen3_conditional_response_mapping_qualified") == all_passed),
        check("fp16_cuda_unquantized", set(precision["parameter_dtypes"]) == {"float16"} and not precision["has_quantized_modules"] and not precision["has_bf16_parameters"]),
        check("artifact_hashes", complete["raw_sha256"] == file_sha256(RAW) and complete["generation_sha256"] == file_sha256(GENERATIONS) and complete["selection_decision_sha256"] == file_sha256(DECISION) and complete["final_sha256"] == file_sha256(FINAL)),
        check("hidden_branch_obeyed", not hidden_artifacts if not all_passed else True, [str(path) for path in hidden_artifacts]),
    ]
    result = {
        "phase": 1286, "campaign": "C026", "audit_type": "independent_final_recomputation_audit",
        "checks": checks, "passed_count": sum(v["passed"] for v in checks), "check_count": len(checks),
        "all_checks_passed": all(v["passed"] for v in checks), "scientific_gate_passed": all_passed,
        "authorization": expected_authorization if all(v["passed"] for v in checks) else "deny_all_followup",
    }
    atomic_json(FINAL_AUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pre", "final"))
    args = parser.parse_args()
    preaudit() if args.action == "pre" else final_audit()
