#!/usr/bin/env python3
"""C215: close C205-C214 and publish a full-coordinate response atlas."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C215
PHASE, CAMPAIGN = 1749, "C215"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c215_response_interval_composition_atlas.json"
UPSTREAM = tuple(range(205, 215))


def audit_path(campaign: int) -> Path:
    directory = {
        205: common.C205,
        206: common.C206,
        207: common.C207,
        208: common.C208,
        209: common.C209,
        210: common.C210,
        211: common.C211,
        212: common.C212,
        213: common.C213,
        214: common.C214,
    }[campaign]
    return directory / "audit/independent_final_audit.json"


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    upstream = {f"C{campaign}": core.load(audit_path(campaign))["all_checks_passed"] for campaign in UPSTREAM}
    checks = {
        "all_upstream_audited": all(upstream.values()),
        "ten_campaigns": len(upstream) == 10,
        "asset_is_json": ASSET.suffix == ".json",
        "full_coordinate_width": common.DIM == 2560,
    }
    if not all(checks.values()):
        raise RuntimeError({"checks": checks, "upstream": upstream})
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "campaign_synthesis_and_heatmap_frozen",
        "inputs": [f"C{campaign}" for campaign in UPSTREAM],
        "heatmap": {
            "dimensions": 2560,
            "baseline_panel": "one fresh anchor per nine programs; relation and boundary roles; embedding/q23/q24/q25",
            "intervention_panel": "same anchors and roles; dose-1 odd/even q24/q25 response",
            "composition_panel": "fresh path-factorial mean combined, additive prediction and interaction at q24/q25 for all six roles",
        },
        "theory_readiness_criteria": {
            "fresh_multi_program_natural_response": "C210 q24->q25 sign >= 0.90 in all nine fresh programs",
            "unseen_intervention_prediction": "C208 complete direction gate",
            "two_construction_composition": "C212 both-arm gate",
            "typed_causal_deletion_rescue": "C213 tested and passed",
            "three_model_functional_invariant": "C214 all-pair topology gate",
        },
        "claim_boundary": "The atlas contains physical activation coordinates, not weights, neurons, a closed causal state, or a discovered new mathematics.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "retroactive gate changes"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "synthesize_C205_C214_without_new_model_execution",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "upstream": upstream, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "upstream": upstream}, indent=2))


def role_positions(anchor: dict, role: str) -> list[int]:
    positions = [int(value) for value in anchor["role_positions"][role]]
    if not positions:
        raise RuntimeError((anchor["case_id"], role, positions))
    return positions


def add_row(rows: list[dict], values: np.ndarray, **metadata) -> None:
    vector = np.asarray(values, np.float32).reshape(-1)
    if vector.shape != (common.DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((metadata, vector.shape))
    rows.append({**metadata, "values": vector.astype(float).tolist()})


def synthesize() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    anchors = core.rows(common.C205 / "compiled/qwen3_anchors.jsonl")
    index_rows = core.rows(common.C206 / "raw/index.jsonl")
    by_case = {row["case_id"]: row for row in anchors}
    baseline = np.load(common.C206 / "raw/baseline_full.float16.npy", mmap_mode="r")
    effects = np.load(common.C206 / "raw/joint_effects.float16.npy", mmap_mode="r")
    rows: list[dict] = []

    fresh_indices = [row["case_index"] for row in index_rows if row["partition"] == "fresh"]
    if len(fresh_indices) != 9:
        raise RuntimeError(fresh_indices)
    checkpoint_names = ("embedding", "q23", "q24", "q25")
    for case_i in fresh_indices:
        index = index_rows[case_i]
        anchor = by_case[index["case_id"]]
        for role in ("relation", "boundary"):
            tokens = role_positions(anchor, role)
            for state_i, checkpoint in enumerate(checkpoint_names):
                add_row(
                    rows,
                    np.asarray(baseline[case_i, state_i, tokens], np.float32).mean(axis=0),
                    kind="fresh_baseline",
                    case_id=index["case_id"],
                    program=index["program"],
                    partition="fresh",
                    role=role,
                    token_positions=tokens,
                    checkpoint=checkpoint,
                    label=f"{index['program']} / {role} / {checkpoint} baseline",
                )
            for parity_i, parity in enumerate(("odd", "even")):
                for target_i, checkpoint in enumerate(("q24", "q25")):
                    add_row(
                        rows,
                        np.asarray(effects[case_i, 3, parity_i, target_i, tokens], np.float32).mean(axis=0),
                        kind="dose1_intervention_response",
                        case_id=index["case_id"],
                        program=index["program"],
                        partition="fresh",
                        role=role,
                        token_positions=tokens,
                        checkpoint=checkpoint,
                        parity=parity,
                        dose=1.0,
                        label=f"{index['program']} / {role} / {checkpoint} dose1 {parity}",
                    )

    factorial_states = np.load(common.C212 / "raw/role_states.float16.npy", mmap_mode="r")
    factorial_index = core.rows(common.C212 / "raw/hidden_index.jsonl")
    index_map = {(row["arm"], row["unit"], row["factor_a"], row["factor_b"]): row["hidden_index"] for row in factorial_index}
    for checkpoint_i, checkpoint in ((2, "q24"), (3, "q25")):
        combined_rows, additive_rows, interaction_rows = [], [], []
        for unit in (8, 9, 10, 11):
            h00 = np.asarray(factorial_states[index_map[("path_factorial", unit, 0, 0)], checkpoint_i], np.float32)
            h01 = np.asarray(factorial_states[index_map[("path_factorial", unit, 0, 1)], checkpoint_i], np.float32)
            h10 = np.asarray(factorial_states[index_map[("path_factorial", unit, 1, 0)], checkpoint_i], np.float32)
            h11 = np.asarray(factorial_states[index_map[("path_factorial", unit, 1, 1)], checkpoint_i], np.float32)
            combined_rows.append(h11 - h00)
            additive_rows.append((h10 - h00) + (h01 - h00))
            interaction_rows.append(h11 - h10 - h01 + h00)
        for role_i, role in enumerate(common.ROLES):
            for kind, vectors in (
                ("path_combined", combined_rows),
                ("path_additive_prediction", additive_rows),
                ("path_interaction", interaction_rows),
            ):
                add_row(
                    rows,
                    np.mean(np.stack(vectors), axis=0)[role_i],
                    kind=kind,
                    program="path_factorial",
                    partition="fresh",
                    role=role,
                    checkpoint=checkpoint,
                    label=f"path factorial / {role} / {checkpoint} / {kind}",
                )

    c208 = core.load(common.C208 / "analysis/orthogonal_prediction.json")
    c210 = core.load(common.C210 / "analysis/natural_edit_trajectory.json")
    c212 = core.load(common.C212 / "analysis/factorial_composition.json")
    c213 = core.load(common.C213 / "analysis/final.json")["headline"]
    c214 = core.load(common.C214 / "analysis/final.json")["headline"]
    natural_all_programs = (
        len(c210["per_program_fresh"]) == 9
        and min(value["weighted_sign"] for value in c210["per_program_fresh"].values()) >= 0.90
        and c210["trajectory_sign"]["q24_to_q25_sign"] >= 0.90
    )
    readiness = {
        "fresh_multi_program_natural_response": natural_all_programs,
        "unseen_intervention_prediction": bool(c208["predictive_gate_passed"]),
        "two_construction_composition": bool(c212["factorial_composition_gate_passed"]),
        "typed_causal_deletion_rescue": bool(c213.get("causal_tested", False) and c213.get("causal_gate_passed", False)),
        "three_model_functional_invariant": bool(c214["cross_model_gate_passed"]),
    }
    matrix = np.asarray([row["values"] for row in rows], np.float32)
    coordinate_score = np.mean(np.abs(matrix), axis=0)
    default_coordinates = np.argsort(-coordinate_score)[:64].astype(int).tolist()
    summary = {
        "rows": len(rows),
        "fresh_programs": 9,
        "natural_q24_q25_sign": c210["trajectory_sign"]["q24_to_q25_sign"],
        "c208_odd_nrmse": c208["pooled"]["odd"]["nrmse"],
        "c208_even_nrmse": c208["pooled"]["even"]["nrmse"],
        "surface_composition_fresh_nrmse": c212["arm_summaries"]["surface_factorial"]["fresh"]["median_nrmse"],
        "path_composition_fresh_nrmse": c212["arm_summaries"]["path_factorial"]["fresh"]["median_nrmse"],
        "cross_model_pair_tests": c214["pair_tests"],
        "theory_readiness": readiness,
        "theory_readiness_passed": sum(readiness.values()),
        "theory_readiness_total": len(readiness),
    }
    asset = {
        "schema": "c215_response_interval_composition_atlas.v1",
        "result_type": "response_interval_composition_atlas_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B (coordinate panels); Qwen3/GLM4/DS7B (topology summary)",
        "title": "C205-C215 Response Interval and Composition Atlas",
        "dimensions": list(range(common.DIM)),
        "default_coordinates": default_coordinates,
        "rows": rows,
        "summary": summary,
        "coordinate_semantics": "Columns 0-2559 are Qwen3-4B physical activation coordinates. Baselines include embedding and HiddenState; response rows are signed activation changes.",
        "claim_boundary": protocol["claim_boundary"],
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    core.save(ASSET, asset)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "campaign_synthesized",
        "heatmap_asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"),
        "heatmap_rows": len(rows),
        "heatmap_dimensions": common.DIM,
        "summary": summary,
        "theory_readiness_gate_passed": all(readiness.values()),
        "new_foundational_mathematics_gate": False,
        "interpretation": "C205-C214 reveal a reproducible late natural response field and one positive path-composition slice, but reject a fixed linear coordinate gear, full-token closure, both-arm composition, typed rescue, and three-model invariance under the frozen gates.",
        "next_authorization": "freeze_a_new_multi_family_conditional_response_state_campaign; do_not_reuse_a_fixed_coordinate_formula",
    }
    core.save(OUT / "analysis/campaign_synthesis.json", report)
    checks = {
        "rows": len(rows) == 180,
        "dimensions": matrix.shape == (180, common.DIM),
        "nine_programs": len({row["program"] for row in rows if row["kind"] == "fresh_baseline"}) == 9,
        "embedding_present": any(row.get("checkpoint") == "embedding" for row in rows),
        "hidden_present": all(any(row.get("checkpoint") == checkpoint for row in rows) for checkpoint in ("q23", "q24", "q25")),
        "interaction_present": any(row["kind"] == "path_interaction" for row in rows),
        "asset": ASSET.exists(),
        "finite": bool(np.isfinite(matrix).all()),
    }
    core.save(OUT / "audit/internal_synthesis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "summary": summary}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/campaign_synthesis.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "synthesis": core.load(OUT / "audit/internal_synthesis_audit.json")["all_checks_passed"],
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "synthesize", "close"))
    args = parser.parse_args()
    {"contract": contract, "synthesize": synthesize, "close": close}[args.command]()


if __name__ == "__main__":
    main()
