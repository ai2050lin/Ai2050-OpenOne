#!/usr/bin/env python3
"""C218: descriptive cross-surface response-state distance audit and atlas."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1739_c205_response_ecology_common as common
import phase1750_c216_multi_family_conditional_response_state as c216
import phase1751_c217_reworded_response_state_validation as c217

core = common.core
OUT = common.RESULT / "phase1752_c218_cross_surface_response_state_atlas"
ASSET = common.ROOT / "frontend/public/vis_data/research_kernel/c218_cross_surface_response_state_atlas.json"
PHASE, CAMPAIGN = 1752, "C218"


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(c217.OUT / "audit/independent_final_audit.json")
    checks = {"authorization": parent["all_checks_passed"], "source_arms": len(c216.ARMS) == 5, "paired_units": [8, 9, 10, 11] == list(range(8, 12)), "dimensions": common.DIM == 2560}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "cross_surface_response_state_atlas_frozen", "inputs": ["C216 original surfaces", "C217 reworded surfaces"], "paired_units": [8, 9, 10, 11], "arms": list(c216.ARMS), "distances": ["same_arm_same_unit_cross_surface", "wrong_arm_same_unit_cross_surface", "matched_unit_five_way_rank"], "heatmap": "q24/q25 x relation/boundary x factor-A/factor-B/interaction for both surfaces and their difference", "epistemic_status": "post-reveal descriptive decomposition; cannot confirm the semantic response-state hypothesis", "claim_boundary": "same-arm dominance may still reflect task family, answer protocol, role compiler and controlled-English structure", "forbidden": ["attention", "MLP", "weights", "PCA", "calling this confirmatory"], "producer_sha256": core.sha(Path(__file__)), "authorization": "analyze_existing_C216_C217_and_freeze_C219_shared_interface_test"}
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def load_source(directory: Path):
    states = np.load(directory / "raw/role_states.float16.npy", mmap_mode="r")
    index = core.rows(directory / "raw/hidden_index.jsonl")
    key = {(item["arm"], item["unit"], item["factor_a"], item["factor_b"]): item["hidden_index"] for item in index}
    return states, key


def components(states: np.ndarray, key: dict, arm: str, unit: int):
    h00 = np.asarray(states[key[(arm, unit, 0, 0)]], np.float32)
    h10 = np.asarray(states[key[(arm, unit, 1, 0)]], np.float32)
    h01 = np.asarray(states[key[(arm, unit, 0, 1)]], np.float32)
    h11 = np.asarray(states[key[(arm, unit, 1, 1)]], np.float32)
    return {"factor_a": h10 - h00, "factor_b": h01 - h00, "interaction": h11 - h10 - h01 + h00}


def normalized(parts: dict) -> np.ndarray:
    value = np.concatenate([parts[kind][1:4].reshape(-1) for kind in ("factor_a", "factor_b", "interaction")]).astype(np.float32)
    return value / max(float(np.sqrt(np.mean(np.square(value, dtype=np.float64)))), 1e-12)


def add_row(rows: list[dict], vector: np.ndarray, **metadata):
    value = np.asarray(vector, np.float32).reshape(-1)
    if value.shape != (2560,) or not np.isfinite(value).all():
        raise RuntimeError((metadata, value.shape))
    rows.append({**metadata, "values": value.astype(float).tolist()})


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    old_states, old_key = load_source(c216.OUT)
    new_states, new_key = load_source(c217.OUT)
    distance_rows, atlas_rows = [], []
    for unit in range(8, 12):
        old_vectors = {arm: normalized(components(old_states, old_key, arm, unit)) for arm in c216.ARMS}
        new_vectors = {arm: normalized(components(new_states, new_key, arm, unit)) for arm in c216.ARMS}
        for arm in c216.ARMS:
            distances = {candidate: float(np.sqrt(np.mean(np.square(new_vectors[arm] - old_vectors[candidate], dtype=np.float64)))) for candidate in c216.ARMS}
            prediction = min(c216.ARMS, key=lambda candidate: distances[candidate])
            distance_rows.append({"unit": unit, "arm": arm, "prediction": prediction, "correct": prediction == arm, "same_arm_cross_surface": distances[arm], "nearest_wrong_cross_surface": min(value for candidate, value in distances.items() if candidate != arm), "same_arm_margin": min(value for candidate, value in distances.items() if candidate != arm) - distances[arm]})

    for source, states, key in (("C216_original", old_states, old_key), ("C217_reworded", new_states, new_key)):
        for arm in c216.ARMS:
            unit_parts = [components(states, key, arm, unit) for unit in range(8, 12)]
            for checkpoint_i, checkpoint in ((2, "q24"), (3, "q25")):
                for role_i, role in ((2, "relation"), (5, "boundary")):
                    for kind in ("factor_a", "factor_b", "interaction"):
                        vector = np.mean(np.stack([parts[kind][checkpoint_i, role_i] for parts in unit_parts]), axis=0)
                        add_row(atlas_rows, vector, source=source, arm=arm, kind=kind, checkpoint=checkpoint, role=role, label=f"{source} / {arm} / {kind} / {checkpoint} / {role}")
    original_map = {(row["arm"], row["kind"], row["checkpoint"], row["role"]): np.asarray(row["values"], np.float32) for row in atlas_rows if row["source"] == "C216_original"}
    reworded_map = {(row["arm"], row["kind"], row["checkpoint"], row["role"]): np.asarray(row["values"], np.float32) for row in atlas_rows if row["source"] == "C217_reworded"}
    for key, original in original_map.items():
        arm, kind, checkpoint, role = key
        add_row(atlas_rows, reworded_map[key] - original, source="cross_surface_difference", arm=arm, kind=kind, checkpoint=checkpoint, role=role, label=f"difference / {arm} / {kind} / {checkpoint} / {role}")

    by_arm = {}
    for arm in c216.ARMS:
        selected = [item for item in distance_rows if item["arm"] == arm]
        by_arm[arm] = {"support": 4, "rank_accuracy": float(np.mean([item["correct"] for item in selected])), "median_same_arm_distance": float(np.median([item["same_arm_cross_surface"] for item in selected])), "median_nearest_wrong_distance": float(np.median([item["nearest_wrong_cross_surface"] for item in selected])), "median_margin": float(np.median([item["same_arm_margin"] for item in selected]))}
    summary = {"support": 20, "matched_unit_rank_accuracy": float(np.mean([item["correct"] for item in distance_rows])), "positive_margin_fraction": float(np.mean([item["same_arm_margin"] > 0 for item in distance_rows])), "by_arm": by_arm, "descriptive_same_arm_dominance": sum(value["median_margin"] > 0 for value in by_arm.values())}
    matrix = np.asarray([row["values"] for row in atlas_rows], np.float32)
    asset = {"schema": "c218_cross_surface_response_state_atlas.v1", "result_type": "cross_surface_response_state_atlas_heatmap", "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B", "title": "C216-C218 Cross-Surface Conditional Response-State Atlas", "dimensions": list(range(2560)), "default_coordinates": np.argsort(-np.mean(np.abs(matrix), axis=0))[:64].astype(int).tolist(), "rows": atlas_rows, "summary": summary, "coordinate_semantics": "Each column is a Qwen3-4B physical HiddenState activation coordinate; rows are mean q24/q25 factor or interaction responses and cross-surface differences.", "claim_boundary": protocol["claim_boundary"]}
    ASSET.parent.mkdir(parents=True, exist_ok=True); core.save(ASSET, asset)
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "cross_surface_response_state_described", "summary": summary, "heatmap_asset": str(ASSET.relative_to(common.ROOT)).replace("\\", "/"), "heatmap_rows": len(atlas_rows), "epistemic_status": protocol["epistemic_status"], "interpretation": "Matched-unit comparisons test whether the response signature stays closer to the same arm across rewording than to other arms. This is a diagnostic on already revealed data and freezes, but does not itself confirm, a shared-interface test.", "next_authorization": "C219_shared_interface_cross_relation_confirmation"}
    core.save(OUT / "analysis/cross_surface_atlas.json", report); core.write_rows(OUT / "analysis/distance_rows.jsonl", distance_rows)
    checks = {"distance_rows": len(distance_rows) == 20, "atlas_rows": len(atlas_rows) == 180, "dimensions": matrix.shape == (180, 2560), "five_arms": set(by_arm) == set(c216.ARMS), "finite": bool(np.isfinite(matrix).all()), "asset": ASSET.exists()}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks, "summary": summary}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/cross_surface_atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "analyze", "close")); args = parser.parse_args(); {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
