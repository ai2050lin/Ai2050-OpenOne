#!/usr/bin/env python3
"""Phase1486: full 2x2x2 factorial and surface atlas for the C079 field."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1484_c084_batch_deep_mining_contract"
PARENT = RESULT / "phase1485_c084_coordinate_stability_atlas"
DISCOVERY = RESULT / "phase1465_c079_discovery_full_field_capture"
HOLDOUT = RESULT / "phase1467_c079_holdout_capture_and_validation"
C079 = RESULT / "phase1463_c079_aggregate_observation_contract"
PRIOR = RESULT / "phase1476_c082_coordinate_atlas"
OUT = RESULT / "phase1486_c084_factorial_surface_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


EFFECTS = ["relation", "entity", "object", "relation_entity", "relation_object", "entity_object", "relation_entity_object"]
ORDERS = {"relation": 1, "entity": 1, "object": 1, "relation_entity": 2, "relation_object": 2, "entity_object": 2, "relation_entity_object": 3}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.astype(np.float64, copy=False), right.astype(np.float64, copy=False)) / denominator) if denominator > 1e-12 else 0.0


def compact(values: list[float]) -> dict:
    return {"minimum": float(min(values)), "median": float(np.median(values)), "mean": float(np.mean(values)), "maximum": float(max(values))}


def load_sources() -> tuple[dict, dict, dict]:
    discovery_field = np.load(DISCOVERY / "raw/discovery_role_field.float16.npy", mmap_mode="r")
    holdout_field = np.load(HOLDOUT / "raw/holdout_role_field.float16.npy", mmap_mode="r")
    discovery_index = core.rows(DISCOVERY / "raw/discovery_role_field_index.jsonl")
    holdout_index = core.rows(HOLDOUT / "raw/holdout_role_field_index.jsonl")
    fields = {"response_discovery": discovery_field, "confirmation": holdout_field, "lockbox": holdout_field}
    indexes = {"response_discovery": discovery_index, "confirmation": holdout_index, "lockbox": holdout_index}
    lookups = {
        split: {(row["family"], row["index"], row["record_relation_id"], row["surface"], row["cell"]): row for row in index if row["partition"] == split}
        for split, index in indexes.items()
    }
    return fields, indexes, lookups


def panel_keys(index: list[dict], split: str, relation: str) -> list[tuple[str, int]]:
    return sorted({(row["family"], row["index"]) for row in index if row["partition"] == split and row["record_relation_id"] == relation})


def signs(row: dict) -> dict[str, int]:
    r = 1 if row["relation_match"] else -1
    e = 1 if row["entity_match"] else -1
    o = 1 if row["object_match"] else -1
    return {"relation": r, "entity": e, "object": o, "relation_entity": r * e, "relation_object": r * o, "entity_object": e * o, "relation_entity_object": r * e * o}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1486 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    contract = core.load(CONTRACT / "protocol/preregistration.json")
    c079 = core.load(C079 / "protocol/preregistration.json")
    dmeta = core.load(DISCOVERY / "analysis/capture_metadata.json")
    hmeta = core.load(HOLDOUT / "analysis/holdout_summary.json")
    checks = {
        "parent": parent["authorization"] == "run_phase1486_c084_factorial_surface_atlas" and parent_audit["all_checks_passed"],
        "discovery_hash": core.sha(DISCOVERY / "raw/discovery_role_field.float16.npy") == dmeta["raw_sha256"],
        "holdout_hash": core.sha(HOLDOUT / "raw/holdout_role_field.float16.npy") == hmeta["raw_sha256"],
        "effects": contract["factorial_branch"]["contrasts"] == EFFECTS,
        "cells": c079["cells"] == ["111", "110", "101", "100", "011", "010", "001", "000"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    relations = contract["axes"]["relations"]
    splits = contract["axes"]["splits"]
    surfaces = contract["axes"]["surfaces"]
    roles = contract["axes"]["roles"]
    cells = c079["cells"]
    shape = (7, 6, 3, 2, 37, 9, 2560)
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)
    atlas_path = OUT / "atlas/factorial_contrast_mean.float32.npy"
    atlas = np.lib.format.open_memmap(atlas_path, mode="w+", dtype=np.float32, shape=shape)
    sample_counts = np.zeros((6, 3, 2), dtype=np.int32)
    fields, indexes, lookups = load_sources()
    finite = True
    for relation_index, relation in enumerate(relations):
        for split_index, split in enumerate(splits):
            keys = panel_keys(indexes[split], split, relation)
            for surface_index, surface in enumerate(surfaces):
                total = np.zeros((7, 37, 9, 2560), dtype=np.float64)
                for family, index in keys:
                    rows = [lookups[split][(family, index, relation, surface, cell)] for cell in cells]
                    block = np.asarray(fields[split][[row["row_index"] for row in rows]], dtype=np.float32)
                    weight = np.asarray([[signs(row)[effect] * (2 ** ORDERS[effect]) / 8.0 for row in rows] for effect in EFFECTS], dtype=np.float32)
                    contrasts = np.tensordot(weight, block, axes=(1, 0))
                    finite = finite and bool(np.isfinite(contrasts).all())
                    total += contrasts
                atlas[:, relation_index, split_index, surface_index] = (total / len(keys)).astype(np.float32)
                sample_counts[relation_index, split_index, surface_index] = len(keys)
    atlas.flush()
    del atlas
    np.save(OUT / "atlas/sample_counts.int32.npy", sample_counts)
    atlas = np.load(atlas_path, mmap_mode="r")
    prior = np.load(PRIOR / "atlas/mean_effect.float32.npy", mmap_mode="r")
    relation_reproduction_max_abs = float(np.max(np.abs(np.asarray(atlas[0], dtype=np.float32) - np.asarray(prior, dtype=np.float32))))
    layer_rows = []
    for relation_index, relation in enumerate(relations):
        pooled = np.mean(atlas[:, relation_index], axis=(1, 2), dtype=np.float64).astype(np.float32)
        for state in range(37):
            for role_index, role in enumerate(roles):
                vectors = pooled[:, state, role_index]
                norms = np.linalg.norm(vectors, axis=1)
                beta_norms = np.asarray([norms[index] / (2 ** ORDERS[effect]) for index, effect in enumerate(EFFECTS)])
                beta_energy = np.square(beta_norms)
                energy_total = float(np.sum(beta_energy))
                row = {
                    "relation": relation,
                    "state": state,
                    "role": role,
                    "effects": {},
                }
                for effect_index, effect in enumerate(EFFECTS):
                    row["effects"][effect] = {
                        "full_contrast_norm": float(norms[effect_index]),
                        "full_contrast_norm_ratio_to_relation": float(norms[effect_index] / norms[0]) if norms[0] > 0 else 0.0,
                        "factorial_coefficient_norm": float(beta_norms[effect_index]),
                        "factorial_coefficient_norm_ratio_to_relation": float(beta_norms[effect_index] / beta_norms[0]) if beta_norms[0] > 0 else 0.0,
                        "factorial_coefficient_energy_fraction": float(beta_energy[effect_index] / energy_total) if energy_total > 0 else 0.0,
                        "cosine_to_relation_contrast": cosine(vectors[effect_index], vectors[0]),
                    }
                layer_rows.append(row)
    core.write_rows(OUT / "analysis/layer_factorial_metrics.jsonl", layer_rows)
    surface_rows = []
    for effect_index, effect in enumerate(EFFECTS):
        for relation_index, relation in enumerate(relations):
            for split_index, split in enumerate(splits):
                for state in range(37):
                    for role_index, role in enumerate(roles):
                        left = np.asarray(atlas[effect_index, relation_index, split_index, 0, state, role_index], dtype=np.float32)
                        right = np.asarray(atlas[effect_index, relation_index, split_index, 1, state, role_index], dtype=np.float32)
                        denominator = 0.5 * (float(np.linalg.norm(left)) + float(np.linalg.norm(right)))
                        surface_rows.append({
                            "effect": effect,
                            "relation": relation,
                            "split": split,
                            "state": state,
                            "role": role,
                            "cosine": cosine(left, right),
                            "normalized_distance": float(np.linalg.norm(left - right) / denominator) if denominator > 1e-12 else 0.0,
                        })
    core.write_rows(OUT / "analysis/cross_surface_metrics.jsonl", surface_rows)
    boundary_role = roles.index("boundary")
    boundary_rows = []
    for effect_index, effect in enumerate(EFFECTS):
        for relation_index, relation in enumerate(relations):
            for split_index, split in enumerate(splits):
                for surface_index, surface in enumerate(surfaces):
                    vector = np.asarray(atlas[effect_index, relation_index, split_index, surface_index, 35, boundary_role], dtype=np.float32)
                    relation_vector = np.asarray(atlas[0, relation_index, split_index, surface_index, 35, boundary_role], dtype=np.float32)
                    boundary_rows.append({
                        "effect": effect,
                        "relation": relation,
                        "split": split,
                        "surface": surface,
                        "full_contrast_norm": float(np.linalg.norm(vector)),
                        "full_contrast_ratio_to_relation": float(np.linalg.norm(vector) / np.linalg.norm(relation_vector)) if np.linalg.norm(relation_vector) > 0 else 0.0,
                        "factorial_coefficient_ratio_to_relation": float((np.linalg.norm(vector) / (2 ** ORDERS[effect])) / (np.linalg.norm(relation_vector) / 2)) if np.linalg.norm(relation_vector) > 0 else 0.0,
                        "cosine_to_relation": cosine(vector, relation_vector),
                    })
    core.write_rows(OUT / "analysis/boundary_state35_factorial_panels.jsonl", boundary_rows)
    state35_layer = [row for row in layer_rows if row["state"] == 35 and row["role"] == "boundary"]
    effect_summary = {}
    for effect in EFFECTS:
        coefficient_ratios = [row["effects"][effect]["factorial_coefficient_norm_ratio_to_relation"] for row in layer_rows]
        boundary_ratios = [row["effects"][effect]["factorial_coefficient_norm_ratio_to_relation"] for row in state35_layer]
        boundary_cosines = [row["effects"][effect]["cosine_to_relation_contrast"] for row in state35_layer]
        effect_surface = [row for row in surface_rows if row["effect"] == effect and row["state"] == 35 and row["role"] == "boundary"]
        effect_summary[effect] = {
            "all_state_role_coefficient_ratio_to_relation": compact(coefficient_ratios),
            "boundary_state35_coefficient_ratio_to_relation": compact(boundary_ratios),
            "boundary_state35_cosine_to_relation": compact(boundary_cosines),
            "boundary_state35_cross_surface_cosine": compact([row["cosine"] for row in effect_surface]),
            "boundary_state35_cross_surface_normalized_distance": compact([row["normalized_distance"] for row in effect_surface]),
        }
    output_checks = {
        "finite": finite and bool(np.isfinite(atlas).all()),
        "shape": list(atlas.shape) == list(shape),
        "counts": int(np.sum(sample_counts)) == 414,
        "relation_reproduction": relation_reproduction_max_abs <= 1e-5,
        "layer_rows": len(layer_rows) == 1998,
        "surface_rows": len(surface_rows) == 41958,
        "boundary_rows": len(boundary_rows) == 252,
    }
    if not all(output_checks.values()):
        raise RuntimeError(output_checks)
    summary = {
        "phase": 1486,
        "campaign": "C084",
        "shape": list(shape),
        "axis_order": ["effect", "relation", "split", "surface", "state", "role", "coordinate"],
        "effects": EFFECTS,
        "orders": ORDERS,
        "relation_reproduction_max_abs": relation_reproduction_max_abs,
        "effect_summary": effect_summary,
        "output_checks": output_checks,
        "factorial_atlas": {"bytes": atlas_path.stat().st_size, "sha256": core.sha(atlas_path)},
        "sample_counts": {"bytes": (OUT / "atlas/sample_counts.int32.npy").stat().st_size, "sha256": core.sha(OUT / "atlas/sample_counts.int32.npy")},
        "interpretation_boundary": "orthogonal contrasts over the complete frozen condition cube; all outputs remain retrospective and success-domain-specific",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/factorial_atlas_summary.json", summary)
    core.save(OUT / "analysis/final.json", {
        "phase": 1486,
        "campaign": "C084",
        "status": "factorial_surface_atlas_complete",
        "output_checks": output_checks,
        "model_run": False,
        "hidden_access": "read existing legal C079 raw fields",
        "authorization": "run_phase1487_c084_joint_synthesis_and_prediction_freeze",
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
