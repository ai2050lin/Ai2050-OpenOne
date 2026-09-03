#!/usr/bin/env python3
"""C220: preregistered minimality ladder and collision controls for response states."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common
import phase1750_c216_multi_family_conditional_response_state as c216
import phase1753_c219_shared_interface_response_state_confirmation as c219

core = common.core
OUT = common.RESULT / "phase1754_c220_response_state_minimality_collision_controls"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c220_response_state_minimality_atlas.json"
PHASE, CAMPAIGN = 1754, "C220"
EFFECTS = ("factor_a", "factor_b", "interaction")
SUBSETS = (
    {"name": "q25_boundary", "checkpoints": [3], "roles": [5]},
    {"name": "q24_q25_relation_boundary", "checkpoints": [2, 3], "roles": [2, 5]},
    {"name": "q23_q25_relation_boundary", "checkpoints": [1, 2, 3], "roles": [2, 5]},
    {"name": "q24_q25_six_roles", "checkpoints": [2, 3], "roles": [0, 1, 2, 3, 4, 5]},
    {"name": "q23_q25_six_roles", "checkpoints": [1, 2, 3], "roles": [0, 1, 2, 3, 4, 5]},
)


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(c219.OUT / "audit/internal_run_audit.json")
    checks = {
        "c219_run_complete": parent["all_checks_passed"],
        "c219_unrevealed": not (c219.OUT / "analysis/shared_interface_confirmation.json").exists(),
        "ordered_ladder": [item["name"] for item in SUBSETS] == [
            "q25_boundary",
            "q24_q25_relation_boundary",
            "q23_q25_relation_boundary",
            "q24_q25_six_roles",
            "q23_q25_six_roles",
        ],
        "dimensions": common.DIM == 2560,
        "roles": list(common.ROLES) == ["primary", "secondary", "relation", "context", "query", "boundary"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "response_state_minimality_and_collision_controls_frozen_before_C219_reveal",
        "sources": {
            "templates": "C216 discovery units 0-3",
            "selection": "C219 confirmation units 0-3",
            "single_reveal": "C219 fresh units 4-7",
        },
        "subset_ladder": list(SUBSETS),
        "selection_rule": "choose the first subset with confirmation accuracy >= 0.60 and at least three arms >= 0.50; otherwise choose the full subset",
        "fresh_gate": {
            "accuracy_min": 0.60,
            "arms_at_or_above_half_min": 3,
            "advantage_over_best_negative_min": 0.15,
        },
        "negative_controls": [
            "factor-A/factor-B order swap in each query",
            "fixed query-only permutation of all 2560 physical coordinates",
            "coordinate-collapsed RMS energy signature",
        ],
        "normalization": "one RMS normalization after concatenating the signed coordinate responses",
        "claim_boundary": "Passing identifies a sufficient member of this finite observation ladder for arm discrimination. It does not establish a unique minimal state, semantic invariance, or causal use.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "post-reveal subset edits", "template refitting", "causal language"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_C219_only_after_this_contract_is_frozen",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "subset_ladder": protocol["subset_ladder"]}, indent=2))


def load_source(directory: Path):
    states = np.load(directory / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(directory / "raw/hidden_index.jsonl")
    key = {(row["arm"], row["unit"], row["factor_a"], row["factor_b"]): row["hidden_index"] for row in index}
    return states, key


def response_cube(states: np.ndarray, key: dict, arm: str, unit: int, subset: dict) -> np.ndarray:
    h00 = np.asarray(states[key[(arm, unit, 0, 0)]], np.float32)
    h10 = np.asarray(states[key[(arm, unit, 1, 0)]], np.float32)
    h01 = np.asarray(states[key[(arm, unit, 0, 1)]], np.float32)
    h11 = np.asarray(states[key[(arm, unit, 1, 1)]], np.float32)
    cube = np.stack((h10 - h00, h01 - h00, h11 - h10 - h01 + h00))
    return cube[:, subset["checkpoints"]][:, :, subset["roles"]]


def normalize(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, np.float32).reshape(-1)
    return flat / max(float(np.sqrt(np.mean(np.square(flat, dtype=np.float64)))), 1e-12)


def classify(templates: dict[str, np.ndarray], values: dict[tuple[str, int], np.ndarray], units: range) -> dict:
    rows = []
    for arm in c216.ARMS:
        for unit in units:
            value = values[(arm, unit)]
            distances = {candidate: float(np.sqrt(np.mean(np.square(value - template, dtype=np.float64)))) for candidate, template in templates.items()}
            prediction = min(c216.ARMS, key=lambda candidate: distances[candidate])
            rows.append({
                "arm": arm,
                "unit": unit,
                "prediction": prediction,
                "correct": prediction == arm,
                "own_distance": distances[arm],
                "nearest_wrong_distance": min(distance for candidate, distance in distances.items() if candidate != arm),
            })
    return {
        "support": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "by_arm_accuracy": {arm: float(np.mean([row["correct"] for row in rows if row["arm"] == arm])) for arm in c216.ARMS},
        "collision_fraction": float(np.mean([row["nearest_wrong_distance"] <= row["own_distance"] for row in rows])),
        "rows": rows,
    }


def add_atlas_row(rows: list[dict], values: np.ndarray, **metadata) -> None:
    vector = np.asarray(values, np.float32).reshape(-1)
    if vector.shape != (common.DIM,) or not np.isfinite(vector).all():
        raise RuntimeError((metadata, vector.shape))
    rows.append({**metadata, "values": vector.astype(float).tolist()})


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    old_states, old_key = load_source(c216.OUT)
    new_states, new_key = load_source(c219.OUT)
    subset_reports = {}
    cached = {}
    for subset in SUBSETS:
        old_cubes = {(arm, unit): response_cube(old_states, old_key, arm, unit, subset) for arm in c216.ARMS for unit in range(4)}
        new_cubes = {(arm, unit): response_cube(new_states, new_key, arm, unit, subset) for arm in c216.ARMS for unit in range(8)}
        templates = {arm: np.mean(np.stack([normalize(old_cubes[(arm, unit)]) for unit in range(4)]), axis=0) for arm in c216.ARMS}
        values = {key: normalize(cube) for key, cube in new_cubes.items()}
        confirmation = classify(templates, values, range(4))
        fresh = classify(templates, values, range(4, 8))
        subset_reports[subset["name"]] = {
            "coordinates_per_signature": int(np.prod(new_cubes[(c216.ARMS[0], 0)].shape)),
            "confirmation": {key: value for key, value in confirmation.items() if key != "rows"},
            "fresh": {key: value for key, value in fresh.items() if key != "rows"},
        }
        cached[subset["name"]] = (old_cubes, new_cubes, templates, values, fresh["rows"])

    selected = SUBSETS[-1]
    for subset in SUBSETS:
        result = subset_reports[subset["name"]]["confirmation"]
        if result["accuracy"] >= 0.60 and sum(value >= 0.50 for value in result["by_arm_accuracy"].values()) >= 3:
            selected = subset
            break

    old_cubes, new_cubes, templates, values, fresh_rows = cached[selected["name"]]
    permutation = np.random.default_rng(22003).permutation(common.DIM)
    swapped_values = {(arm, unit): normalize(cube[[1, 0, 2]]) for (arm, unit), cube in new_cubes.items()}
    permuted_values = {(arm, unit): normalize(cube[..., permutation]) for (arm, unit), cube in new_cubes.items()}
    energy_templates = {
        arm: np.mean(np.stack([normalize(np.sqrt(np.mean(np.square(old_cubes[(arm, unit)], dtype=np.float64), axis=-1))) for unit in range(4)]), axis=0)
        for arm in c216.ARMS
    }
    energy_values = {key: normalize(np.sqrt(np.mean(np.square(cube, dtype=np.float64), axis=-1))) for key, cube in new_cubes.items()}
    controls = {
        "factor_order_swap": classify(templates, swapped_values, range(4, 8)),
        "query_coordinate_permutation": classify(templates, permuted_values, range(4, 8)),
        "coordinate_collapsed_energy": classify(energy_templates, energy_values, range(4, 8)),
    }
    controls_summary = {name: {key: value for key, value in result.items() if key != "rows"} for name, result in controls.items()}
    fresh = subset_reports[selected["name"]]["fresh"]
    best_negative = max(result["accuracy"] for result in controls_summary.values())
    gate = (
        fresh["accuracy"] >= protocol["fresh_gate"]["accuracy_min"]
        and sum(value >= 0.50 for value in fresh["by_arm_accuracy"].values()) >= protocol["fresh_gate"]["arms_at_or_above_half_min"]
        and fresh["accuracy"] - best_negative >= protocol["fresh_gate"]["advantage_over_best_negative_min"]
    )

    atlas_rows = []
    checkpoint_names = {0: "embedding", 1: "q23", 2: "q24", 3: "q25"}
    for arm in c216.ARMS:
        template_cube = np.mean(np.stack([old_cubes[(arm, unit)] for unit in range(4)]), axis=0)
        fresh_cube = np.mean(np.stack([new_cubes[(arm, unit)] for unit in range(4, 8)]), axis=0)
        for effect_i, effect in enumerate(EFFECTS):
            for checkpoint_i, checkpoint in enumerate(selected["checkpoints"]):
                for role_i, role in enumerate(selected["roles"]):
                    metadata = {
                        "arm": arm,
                        "effect": effect,
                        "checkpoint": checkpoint_names[checkpoint],
                        "role": common.ROLES[role],
                    }
                    add_atlas_row(atlas_rows, template_cube[effect_i, checkpoint_i, role_i], source="C216_discovery_template", label=f"template / {arm} / {effect} / {checkpoint_names[checkpoint]} / {common.ROLES[role]}", **metadata)
                    add_atlas_row(atlas_rows, fresh_cube[effect_i, checkpoint_i, role_i], source="C219_shared_interface_fresh", label=f"fresh / {arm} / {effect} / {checkpoint_names[checkpoint]} / {common.ROLES[role]}", **metadata)
                    add_atlas_row(atlas_rows, fresh_cube[effect_i, checkpoint_i, role_i] - template_cube[effect_i, checkpoint_i, role_i], source="fresh_minus_template", label=f"difference / {arm} / {effect} / {checkpoint_names[checkpoint]} / {common.ROLES[role]}", **metadata)
    matrix = np.asarray([row["values"] for row in atlas_rows], np.float32)
    asset = {
        "schema": "c220_response_state_minimality_atlas.v1",
        "result_type": "response_state_minimality_atlas_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C220 Shared-Interface Response-State Minimality Atlas",
        "dimensions": list(range(common.DIM)),
        "default_coordinates": np.argsort(-np.mean(np.abs(matrix), axis=0))[:64].astype(int).tolist(),
        "rows": atlas_rows,
        "summary": {"selected_subset": selected["name"], "fresh": fresh, "negative_controls": controls_summary, "gate_passed": gate},
        "coordinate_semantics": "Each column is one physical Qwen3-4B HiddenState activation coordinate; no PCA or coordinate averaging is used in the displayed rows.",
        "claim_boundary": protocol["claim_boundary"],
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    core.save(ASSET, asset)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "response_state_minimality_and_collision_controls_adjudicated",
        "subset_ladder": subset_reports,
        "selected_subset": selected,
        "selection_partition": subset_reports[selected["name"]]["confirmation"],
        "single_reveal_fresh": fresh,
        "negative_controls": controls_summary,
        "fresh_advantage_over_best_negative": fresh["accuracy"] - best_negative,
        "minimality_collision_gate_passed": gate,
        "heatmap_asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"),
        "interpretation": "The ordered ladder asks how much typed response state is needed to preserve family discrimination. Negative controls test factor order, physical coordinate arrangement and coordinate-collapsed energy. Results remain observational and task-family conditioned.",
        "next_authorization": "C221_independent_material_response_state_prediction_then_targeted_causal_test" if gate else "retain_full_response_state_and_design_independent_collision_panel",
    }
    core.save(OUT / "analysis/minimality_collision_controls.json", report)
    core.write_rows(OUT / "analysis/selected_fresh_classification_rows.jsonl", fresh_rows)
    checks = {
        "five_subsets": len(subset_reports) == 5,
        "selection_frozen": selected["name"] in [item["name"] for item in SUBSETS],
        "fresh_support": fresh["support"] == 20,
        "three_controls": set(controls_summary) == {"factor_order_swap", "query_coordinate_permutation", "coordinate_collapsed_energy"},
        "atlas_dimensions": matrix.ndim == 2 and matrix.shape[1] == common.DIM,
        "finite": bool(np.isfinite(matrix).all()),
        "asset": ASSET.exists(),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "report": report}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/minimality_collision_controls.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
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
