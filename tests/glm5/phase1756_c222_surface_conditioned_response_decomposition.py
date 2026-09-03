#!/usr/bin/env python3
"""C222: post-reveal decomposition of response-state surface conditioning."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common
import phase1750_c216_multi_family_conditional_response_state as c216
import phase1751_c217_reworded_response_state_validation as c217
import phase1753_c219_shared_interface_response_state_confirmation as c219
import phase1754_c220_response_state_minimality_collision_controls as c220
import phase1755_c221_independent_response_state_prediction as c221

core = common.core
OUT = common.RESULT / "phase1756_c222_surface_conditioned_response_decomposition"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c222_surface_conditioned_response_atlas.json"
PHASE, CAMPAIGN = 1756, "C222"
EFFECTS = ("factor_a", "factor_b", "interaction")


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(c221.OUT / "audit/independent_final_audit.json")
    subset = core.load(c220.OUT / "analysis/minimality_collision_controls.json")["selected_subset"]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C222_amplitude_conditioning_observation_without_causal_claim",
        "fixed_subset": subset["name"] == "q24_q25_relation_boundary",
        "sources_exist": all((path / "raw/role_states.float16.npy").exists() for path in (c216.OUT, c217.OUT, c219.OUT, c221.OUT)),
        "dimensions": common.DIM == 2560,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post_reveal_surface_conditioning_decomposition_frozen",
        "selected_subset": subset,
        "sources": ["C216 original", "C217 reworded", "C219 shared interface", "C221 third material"],
        "questions": [
            "Does C221 classify within its own surface from confirmation to fresh words?",
            "Does a common surface-offset correction restore C216-to-C221 arm identity?",
            "Can oracle scalar amplitude correction rescue exact raw-field prediction?",
            "Which prior surface bank is nearest to each C221 response?",
        ],
        "epistemic_status": "post-reveal exploratory failure decomposition; no confirmation or causal claim",
        "claim_boundary": "Panel centering uses the full five-arm panel and is not a single-example decoder. Oracle scaling is diagnostic data leakage and cannot count as prediction.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "causal language", "rewriting C221"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "existing_data_decomposition_and_coordinate_atlas_only",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def source_cubes(directory: Path, units: list[int], subset: dict):
    states, key = c220.load_source(directory)
    return {(arm, unit): c220.response_cube(states, key, arm, unit, subset) for arm in c216.ARMS for unit in units}


def normalized(cubes: dict) -> dict:
    return {key: c220.normalize(value) for key, value in cubes.items()}


def templates(values: dict, units: list[int]) -> dict:
    return {arm: np.mean(np.stack([values[(arm, unit)] for unit in units]), axis=0) for arm in c216.ARMS}


def classify(template_bank: dict, values: dict, units: list[int]) -> dict:
    rows = []
    for arm in c216.ARMS:
        for unit in units:
            distances = {candidate: float(np.sqrt(np.mean(np.square(values[(arm, unit)] - template, dtype=np.float64)))) for candidate, template in template_bank.items()}
            prediction = min(c216.ARMS, key=lambda candidate: distances[candidate])
            rows.append({"arm": arm, "unit": unit, "prediction": prediction, "correct": prediction == arm})
    return {"support": len(rows), "accuracy": float(np.mean([row["correct"] for row in rows])), "by_arm_accuracy": {arm: float(np.mean([row["correct"] for row in rows if row["arm"] == arm])) for arm in c216.ARMS}}


def exact_summary(template_bank: dict, cubes: dict, units: list[int], oracle_scale: bool = False) -> dict:
    rows = []
    for arm in c216.ARMS:
        for unit in units:
            prediction = np.asarray(template_bank[arm], np.float32)
            actual = np.asarray(cubes[(arm, unit)], np.float32)
            scale = 1.0
            if oracle_scale:
                scale = max(0.0, float(np.sum(prediction * actual, dtype=np.float64) / max(np.sum(prediction * prediction, dtype=np.float64), 1e-30)))
            rows.append({"arm": arm, "unit": unit, "scale": scale, "nrmse": common.nrmse(scale * prediction, actual), "weighted_sign": common.weighted_sign(scale * prediction, actual)})
    return {
        "support": len(rows),
        "median_nrmse": float(np.median([row["nrmse"] for row in rows])),
        "median_weighted_sign": float(np.median([row["weighted_sign"] for row in rows])),
        "by_arm": {arm: {"median_nrmse": float(np.median([row["nrmse"] for row in rows if row["arm"] == arm])), "median_weighted_sign": float(np.median([row["weighted_sign"] for row in rows if row["arm"] == arm])), "median_scale": float(np.median([row["scale"] for row in rows if row["arm"] == arm]))} for arm in c216.ARMS},
    }


def add_row(rows: list[dict], vector: np.ndarray, **metadata) -> None:
    value = np.asarray(vector, np.float32).reshape(-1)
    if value.shape != (common.DIM,) or not np.isfinite(value).all():
        raise RuntimeError((metadata, value.shape))
    rows.append({**metadata, "values": value.astype(float).tolist()})


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    subset = protocol["selected_subset"]
    cubes = {
        "C216_original": source_cubes(c216.OUT, [0, 1, 2, 3], subset),
        "C217_reworded": source_cubes(c217.OUT, [8, 9, 10, 11], subset),
        "C219_shared": source_cubes(c219.OUT, list(range(8)), subset),
        "C221_third": source_cubes(c221.OUT, list(range(8)), subset),
    }
    values = {name: normalized(source) for name, source in cubes.items()}
    banks = {
        "C216_original": templates(values["C216_original"], [0, 1, 2, 3]),
        "C217_reworded": templates(values["C217_reworded"], [8, 9, 10, 11]),
        "C219_shared": templates(values["C219_shared"], [0, 1, 2, 3]),
        "C221_confirmation": templates(values["C221_third"], [0, 1, 2, 3]),
    }
    per_bank = {name: classify(bank, values["C221_third"], [4, 5, 6, 7]) for name, bank in banks.items()}

    old_global = np.mean(np.stack(list(banks["C216_original"].values())), axis=0)
    new_global = np.mean(np.stack(list(banks["C221_confirmation"].values())), axis=0)
    shifted_bank = {arm: c220.normalize(banks["C216_original"][arm] - old_global + new_global) for arm in c216.ARMS}
    common_offset = classify(shifted_bank, values["C221_third"], [4, 5, 6, 7])

    centered_templates = {arm: c220.normalize(banks["C216_original"][arm] - old_global) for arm in c216.ARMS}
    centered_queries = {}
    for unit in range(4, 8):
        unit_mean = np.mean(np.stack([values["C221_third"][(arm, unit)] for arm in c216.ARMS]), axis=0)
        for arm in c216.ARMS:
            centered_queries[(arm, unit)] = c220.normalize(values["C221_third"][(arm, unit)] - unit_mean)
    panel_centered = classify(centered_templates, centered_queries, [4, 5, 6, 7])

    joint_rows = []
    for arm in c216.ARMS:
        for unit in range(4, 8):
            query = values["C221_third"][(arm, unit)]
            distances = {(source, candidate): float(np.sqrt(np.mean(np.square(query - template, dtype=np.float64)))) for source, bank in banks.items() for candidate, template in bank.items()}
            source, prediction = min(distances, key=distances.get)
            joint_rows.append({"arm": arm, "unit": unit, "nearest_source": source, "prediction": prediction, "correct": prediction == arm})
    joint = {"support": len(joint_rows), "arm_accuracy": float(np.mean([row["correct"] for row in joint_rows])), "nearest_source_counts": {source: sum(row["nearest_source"] == source for row in joint_rows) for source in banks}}

    raw_old = {arm: np.mean(np.stack([cubes["C216_original"][(arm, unit)] for unit in range(4)]), axis=0) for arm in c216.ARMS}
    raw_self = {arm: np.mean(np.stack([cubes["C221_third"][(arm, unit)] for unit in range(4)]), axis=0) for arm in c216.ARMS}
    exact = {
        "C216_raw_no_scaling": exact_summary(raw_old, cubes["C221_third"], [4, 5, 6, 7], False),
        "C216_raw_oracle_positive_scalar": exact_summary(raw_old, cubes["C221_third"], [4, 5, 6, 7], True),
        "C221_confirmation_raw_to_fresh": exact_summary(raw_self, cubes["C221_third"], [4, 5, 6, 7], False),
    }

    atlas_rows = []
    checkpoint_names = {2: "q24", 3: "q25"}
    raw_sources = {
        "C216_discovery_template": raw_old,
        "C221_confirmation_template": raw_self,
        "C221_fresh_mean": {arm: np.mean(np.stack([cubes["C221_third"][(arm, unit)] for unit in range(4, 8)]), axis=0) for arm in c216.ARMS},
    }
    for arm in c216.ARMS:
        for source, bank in raw_sources.items():
            cube = bank[arm]
            for effect_i, effect in enumerate(EFFECTS):
                for checkpoint_i, checkpoint in enumerate(subset["checkpoints"]):
                    for role_i, role in enumerate(subset["roles"]):
                        add_row(atlas_rows, cube[effect_i, checkpoint_i, role_i], source=source, arm=arm, effect=effect, checkpoint=checkpoint_names[checkpoint], role=common.ROLES[role], label=f"{source} / {arm} / {effect} / {checkpoint_names[checkpoint]} / {common.ROLES[role]}")
        difference = raw_sources["C221_fresh_mean"][arm] - raw_old[arm]
        for effect_i, effect in enumerate(EFFECTS):
            for checkpoint_i, checkpoint in enumerate(subset["checkpoints"]):
                for role_i, role in enumerate(subset["roles"]):
                    add_row(atlas_rows, difference[effect_i, checkpoint_i, role_i], source="C221_fresh_minus_C216", arm=arm, effect=effect, checkpoint=checkpoint_names[checkpoint], role=common.ROLES[role], label=f"difference / {arm} / {effect} / {checkpoint_names[checkpoint]} / {common.ROLES[role]}")
    matrix = np.asarray([row["values"] for row in atlas_rows], np.float32)
    summary = {"within_C221_surface": per_bank["C221_confirmation"], "per_prior_bank": per_bank, "common_surface_offset": common_offset, "panel_centered_arm_residual": panel_centered, "joint_bank": joint, "exact_field": exact}
    asset = {
        "schema": "c222_surface_conditioned_response_atlas.v1",
        "result_type": "surface_conditioned_response_atlas_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C222 Surface-Conditioned Signed Response Atlas",
        "dimensions": list(range(common.DIM)),
        "default_coordinates": np.argsort(-np.mean(np.abs(matrix), axis=0))[:64].astype(int).tolist(),
        "rows": atlas_rows,
        "summary": summary,
        "coordinate_semantics": "Columns are the original 2560 physical HiddenState coordinates at q24/q25 relation and boundary roles; displayed values are signed, unprojected responses.",
        "claim_boundary": protocol["claim_boundary"],
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    core.save(ASSET, asset)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "surface_conditioning_decomposed",
        "summary": summary,
        "heatmap_asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"),
        "interpretation": "Within-surface transfer separates lexical stability from cross-surface invariance. Common-offset and panel-centered diagnostics ask whether a surface-wide response component masks arm residuals. Oracle scaling isolates amplitude mismatch but is not predictive evidence.",
        "next_authorization": "close_C205_C222_major_stage_and_freeze_future_surface_conditioned_state_model",
    }
    core.save(OUT / "analysis/surface_conditioning.json", report)
    core.write_rows(OUT / "analysis/joint_bank_rows.jsonl", joint_rows)
    checks = {"four_banks": len(banks) == 4, "fresh_support": all(item["support"] == 20 for item in per_bank.values()), "joint_support": joint["support"] == 20, "three_exact": len(exact) == 3, "atlas": matrix.shape == (240, 2560), "finite": bool(np.isfinite(matrix).all()), "asset": ASSET.exists()}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "summary": summary}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/surface_conditioning.json")
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
