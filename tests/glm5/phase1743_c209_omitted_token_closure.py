#!/usr/bin/env python3
"""C209: compare role-only and progressively fuller token views for q24->q25 closure."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C209
PHASE, CAMPAIGN = 1743, "C209"
MODELS = ("boundary_only", "six_roles", "six_roles_plus_nonrole", "six_roles_plus_quartiles")


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C208 / "audit/independent_final_audit.json")
    rows = core.rows(common.C206 / "compiled/qwen3_anchors.jsonl")
    checks = {"authorization": parent["all_checks_passed"], "cases": len(rows) == 36, "partitions": {row["unit"] for row in rows} == {1, 2, 5, 6}, "models": len(MODELS) == 4}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "omitted_token_closure_frozen",
        "source": "C206 full-token q24 response",
        "target": "C206 q25 boundary-role response at the same physical activation coordinate",
        "examples": "36 cases x six doses x odd/even, split by lexical unit before fitting",
        "models": list(MODELS),
        "model_details": {
            "boundary_only": "q24 boundary response at the same coordinate",
            "six_roles": "six registered q24 role means at the same coordinate",
            "six_roles_plus_nonrole": "six roles plus pre-query and post-query non-role token means",
            "six_roles_plus_quartiles": "six roles plus four equal-position full-token-bin means",
        },
        "fit": "independent zero-intercept least squares per physical activation coordinate; discovery units 1/2 only",
        "selection": "model table is frozen; confirmation unit 5 is reported; primary full-token comparison is revealed once on fresh unit 6",
        "gates": {"fresh_nrmse_max": 0.75, "fresh_improvement_over_six_roles_min": 0.05, "both_odd_even_required": True},
        "claim_boundary": "predictive sufficiency of registered token summaries for one local transition; no unique Markov state or semantic closure is inferred",
        "forbidden": ["attention", "MLP", "weights", "PCA", "case leakage across units"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_C206_full_token_fields_then_C210_natural_edits",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "models": list(MODELS)}, indent=2))


def token_views(q24: np.ndarray, row: dict, length: int) -> dict[str, np.ndarray]:
    role_values = []
    registered = set()
    for role in common.ROLES:
        positions = row["role_positions"][role]
        registered.update(positions)
        role_values.append(q24[positions].mean(axis=0))
    query_start = min(row["role_positions"]["query"])
    pre = [position for position in range(query_start) if position not in registered]
    post = [position for position in range(query_start, length) if position not in registered]
    nonrole = [q24[pre].mean(axis=0) if pre else np.zeros(common.DIM, np.float32), q24[post].mean(axis=0) if post else np.zeros(common.DIM, np.float32)]
    bins = []
    for positions in np.array_split(np.arange(length), 4):
        bins.append(q24[positions].mean(axis=0))
    roles = np.stack(role_values)
    return {
        "boundary_only": roles[common.ROLES.index("boundary")][None],
        "six_roles": roles,
        "six_roles_plus_nonrole": np.concatenate([roles, np.stack(nonrole)], axis=0),
        "six_roles_plus_quartiles": np.concatenate([roles, np.stack(bins)], axis=0),
    }


def build_examples(component: int):
    fields = np.load(common.C206 / "raw/joint_effects.float16.npy", mmap_mode="r")
    rows = core.rows(common.C206 / "compiled/qwen3_anchors.jsonl")
    index = core.rows(common.C206 / "raw/index.jsonl")
    features = {name: [] for name in MODELS}
    targets = []
    units = []
    for case_i, row in enumerate(rows):
        length = index[case_i]["length"]
        boundary = row["role_positions"]["boundary"]
        for dose_i in range(len(common.DOSES)):
            q24 = np.asarray(fields[case_i, dose_i, component, 0, :length], np.float32)
            q25 = np.asarray(fields[case_i, dose_i, component, 1, boundary], np.float32).mean(axis=0)
            views = token_views(q24, row, length)
            for name in MODELS:
                features[name].append(views[name])
            targets.append(q25)
            units.append(row["unit"])
    return {name: np.stack(value) for name, value in features.items()}, np.stack(targets), np.asarray(units)


def fit_coordinatewise(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    predictors = features.shape[1]
    coefficients = np.empty((predictors, common.DIM), np.float32)
    for coordinate in range(common.DIM):
        x = features[:, :, coordinate].astype(np.float64)
        y = target[:, coordinate].astype(np.float64)
        coefficients[:, coordinate] = np.linalg.lstsq(x, y, rcond=None)[0].astype(np.float32)
    return coefficients


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.einsum("npc,pc->nc", features.astype(np.float32), coefficients.astype(np.float32), optimize=True)


def analyze() -> None:
    report_components = {}
    saved = {}
    for component, label in ((0, "odd"), (1, "even")):
        features, target, units = build_examples(component)
        masks = {"discovery": np.isin(units, [1, 2]), "confirmation": units == 5, "fresh": units == 6}
        table = {}
        for name in MODELS:
            coefficient = fit_coordinatewise(features[name][masks["discovery"]], target[masks["discovery"]])
            saved[f"{label}_{name}"] = coefficient
            table[name] = {}
            for split, mask in masks.items():
                prediction = predict(features[name][mask], coefficient)
                table[name][split] = {"nrmse": common.nrmse(prediction, target[mask]), "weighted_sign": common.weighted_sign(prediction, target[mask])}
        report_components[label] = table
    (OUT / "analysis/operators").mkdir(parents=True, exist_ok=True)
    for name, value in saved.items():
        np.save(OUT / f"analysis/operators/{name}.float32.npy", value)
    gates = core.load(OUT / "protocol/preregistration.json")["gates"]
    decisions = {}
    for label in ("odd", "even"):
        full = report_components[label]["six_roles_plus_quartiles"]["fresh"]["nrmse"]
        roles = report_components[label]["six_roles"]["fresh"]["nrmse"]
        decisions[label] = {"full_token_nrmse": full, "six_role_nrmse": roles, "improvement": roles - full, "passed": full <= gates["fresh_nrmse_max"] and roles - full >= gates["fresh_improvement_over_six_roles_min"]}
    passed = all(value["passed"] for value in decisions.values())
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "omitted_token_closure_adjudicated", "components": report_components, "fresh_decisions": decisions, "full_token_closure_gate_passed": passed, "interpretation": "A fuller token summary is useful only if it prospectively improves fresh-unit prediction over the six registered role means. Failure does not prove omitted tokens are irrelevant; the tested summaries may be insufficient.", "next_authorization": "C210_natural_paraphrase_edit_trajectory"}
    core.save(OUT / "analysis/omitted_token_closure.json", report)
    checks = {"two_components": set(report_components) == {"odd", "even"}, "four_models": all(set(value) == set(MODELS) for value in report_components.values()), "three_splits": all(set(splits) == {"discovery", "confirmation", "fresh"} for component in report_components.values() for splits in component.values()), "finite": bool(np.isfinite([metric[key] for component in report_components.values() for splits in component.values() for metric in splits.values() for key in ("nrmse", "weighted_sign")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"fresh_decisions": decisions, "full_token_closure_gate_passed": passed, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/omitted_token_closure.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()

