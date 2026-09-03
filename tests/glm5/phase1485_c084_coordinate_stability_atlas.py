#!/usr/bin/env python3
"""Phase1485: sign-, threshold-, and layer-aware C079 coordinate atlas."""
from __future__ import annotations

import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1484_c084_batch_deep_mining_contract"
SOURCE = RESULT / "phase1476_c082_coordinate_atlas"
MANIFEST = RESULT / "phase1477_c082_atlas_synthesis/frozen/future_prediction_manifest.json"
OUT = RESULT / "phase1485_c084_coordinate_stability_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left.astype(np.float64, copy=False), right.astype(np.float64, copy=False)) / denominator) if denominator > 1e-12 else 0.0


def pairwise(values: np.ndarray) -> list[float]:
    return [cosine(values[i], values[j]) for i, j in itertools.combinations(range(len(values)), 2)]


def top_mask(vector: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(-np.square(vector.astype(np.float64, copy=False)), kind="stable")
    mask = np.zeros(vector.shape[-1], dtype=np.uint8)
    mask[order[:count]] = 1
    return mask


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def compact(values: list[float]) -> dict:
    return {"minimum": float(min(values)), "mean": float(np.mean(values)), "maximum": float(max(values))}


def cyclic_geometry(vectors: np.ndarray, label: str) -> dict:
    values = pairwise(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    centroid = np.mean(vectors, axis=0)
    return {
        "panel": label,
        "pairwise_cosine": compact(values),
        "simplex_reference": -0.2,
        "mean_minus_simplex_reference": float(np.mean(values) + 0.2),
        "centroid_norm_over_mean_vector_norm": float(np.linalg.norm(centroid) / np.mean(norms)),
        "sum_vector_norm": float(np.linalg.norm(np.sum(vectors, axis=0))),
        "norm_coefficient_of_variation": float(np.std(norms) / np.mean(norms)),
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1485 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    source_meta = core.load(SOURCE / "analysis/atlas_metadata.json")
    manifest = core.load(MANIFEST)
    source_path = SOURCE / "atlas/mean_effect.float32.npy"
    checks = {
        "parent": parent["authorization"] == "run_phase1485_c084_coordinate_stability_atlas" and parent_audit["all_checks_passed"],
        "source_hash": core.sha(source_path) == source_meta["files"]["mean_effect.float32.npy"]["sha256"],
        "manifest_source": manifest["source_atlas_sha256"] == source_meta["files"]["mean_effect.float32.npy"]["sha256"],
        "shape": source_meta["shape"] == [6, 3, 2, 37, 9, 2560],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    means = np.load(source_path, mmap_mode="r")
    relation_mean = np.mean(means, axis=(1, 2), dtype=np.float64).astype(np.float32)
    signs = np.sign(relation_mean).astype(np.int8)
    counts = protocol["coordinate_branch"]["support_counts"]
    shape = (len(counts),) + relation_mean.shape
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)
    support = np.lib.format.open_memmap(OUT / "atlas/support_membership.uint8.npy", mode="w+", dtype=np.uint8, shape=shape)
    for threshold_index, count in enumerate(counts):
        for relation in range(6):
            for state in range(37):
                for role in range(9):
                    support[threshold_index, relation, state, role] = top_mask(relation_mean[relation, state, role], count)
    support.flush()
    del support
    np.save(OUT / "atlas/relation_mean_sign.int8.npy", signs)
    support = np.load(OUT / "atlas/support_membership.uint8.npy", mmap_mode="r")
    layer_rows = []
    previous_common: list[np.ndarray | None] = [None] * 9
    for state in range(37):
        for role in range(9):
            vectors = relation_mean[:, state, role]
            relation_cosines = pairwise(vectors)
            common = np.mean(vectors, axis=0)
            loo = [cosine(vectors[r], np.mean(np.delete(vectors, r, axis=0), axis=0)) for r in range(6)]
            parallel = [cosine(vectors[r], common) ** 2 for r in range(6)]
            threshold_rows = []
            for threshold_index, (fraction, count) in enumerate(zip(protocol["coordinate_branch"]["support_fractions"], counts)):
                masks = np.asarray(support[threshold_index, :, state, role], dtype=bool)
                intersection = np.all(masks, axis=0)
                union = np.any(masks, axis=0)
                pair_jaccard = [jaccard(masks[i], masks[j]) for i, j in itertools.combinations(range(6), 2)]
                coords = np.flatnonzero(intersection)
                unanimous = int(np.sum(np.all(signs[:, state, role, coords] == signs[0, state, role, coords], axis=0) & (signs[0, state, role, coords] != 0))) if len(coords) else 0
                energies = []
                for relation in range(6):
                    energy = np.square(vectors[relation].astype(np.float64, copy=False))
                    energies.append(float(np.sum(energy[intersection]) / np.sum(energy)) if np.sum(energy) > 0 else 0.0)
                threshold_rows.append({
                    "fraction": fraction,
                    "count_per_relation": count,
                    "intersection_count": int(np.count_nonzero(intersection)),
                    "union_count": int(np.count_nonzero(union)),
                    "pairwise_jaccard": compact(pair_jaccard),
                    "same_nonzero_sign_intersection_count": unanimous,
                    "intersection_energy_fraction": compact(energies),
                })
            current_common = np.all(np.asarray(support[1, :, state, role], dtype=bool), axis=0)
            adjacent = None if previous_common[role] is None else jaccard(current_common, previous_common[role])
            previous_common[role] = current_common
            layer_rows.append({
                "state": state,
                "role": protocol["axes"]["roles"][role],
                "relation_pairwise_cosine": compact(relation_cosines),
                "leave_one_relation_common_cosine": compact(loo),
                "common_projection_energy_fraction": compact(parallel),
                "top_1pct_common_support_adjacent_jaccard": adjacent,
                "thresholds": threshold_rows,
            })
    core.write_rows(OUT / "analysis/layer_coordinate_metrics.jsonl", layer_rows)
    boundary_role = protocol["axes"]["roles"].index("boundary")
    query_label = protocol["axes"]["roles"].index("query_label")
    query_relation = protocol["axes"]["roles"].index("query_relation")
    early = []
    for role_index, role_name in ((query_label, "query_label"), (query_relation, "query_relation")):
        early.append(cyclic_geometry(relation_mean[:, 0, role_index], f"global|{role_name}"))
        for split_index, split in enumerate(protocol["axes"]["splits"]):
            for surface_index, surface in enumerate(protocol["axes"]["surfaces"]):
                early.append(cyclic_geometry(np.asarray(means[:, split_index, surface_index, 0, role_index], dtype=np.float32), f"{split}|{surface}|{role_name}"))
    core.write_rows(OUT / "analysis/state0_cyclic_geometry.jsonl", early)
    panel_loo = []
    for relation_index, relation in enumerate(protocol["axes"]["relations"]):
        panels = np.asarray(means[relation_index, :, :, 35, boundary_role], dtype=np.float32).reshape(6, 2560)
        for panel_index, vector in enumerate(panels):
            panel_loo.append({
                "relation": relation,
                "panel_index": panel_index,
                "cosine_to_other_five_panel_mean": cosine(vector, np.mean(np.delete(panels, panel_index, axis=0), axis=0)),
            })
    core.write_rows(OUT / "analysis/boundary_state35_panel_loo.jsonl", panel_loo)
    frozen17 = manifest["frozen_coordinates"]["boundary_state35_top_1pct_intersection"]
    state35_vectors = relation_mean[:, 35, boundary_role]
    state35_panels = np.asarray(means[:, :, :, 35, boundary_role], dtype=np.float32).reshape(36, 2560)
    fixed_rows = []
    for coordinate in frozen17:
        relation_signs = np.sign(state35_vectors[:, coordinate]).astype(int).tolist()
        panel_signs = np.sign(state35_panels[:, coordinate]).astype(int).tolist()
        fixed_rows.append({
            "coordinate": coordinate,
            "relation_signs": relation_signs,
            "relation_unanimous_nonzero": len(set(relation_signs)) == 1 and relation_signs[0] != 0,
            "panel_positive_count": panel_signs.count(1),
            "panel_negative_count": panel_signs.count(-1),
            "all_36_panels_unanimous_nonzero": len(set(panel_signs)) == 1 and panel_signs[0] != 0,
        })
    core.write_rows(OUT / "analysis/frozen17_sign_audit.jsonl", fixed_rows)
    boundary_row = next(row for row in layer_rows if row["state"] == 35 and row["role"] == "boundary")
    summary = {
        "phase": 1485,
        "campaign": "C084",
        "source_sha256": source_meta["files"]["mean_effect.float32.npy"]["sha256"],
        "state0_cyclic_geometry": {row["panel"]: row for row in early if row["panel"].startswith("global")},
        "boundary_state35": boundary_row,
        "frozen17": {
            "coordinates": frozen17,
            "relation_unanimous_nonzero_count": sum(row["relation_unanimous_nonzero"] for row in fixed_rows),
            "all_36_panels_unanimous_nonzero_count": sum(row["all_36_panels_unanimous_nonzero"] for row in fixed_rows),
        },
        "boundary_panel_loo_cosine": compact([row["cosine_to_other_five_panel_mean"] for row in panel_loo]),
        "interpretation_boundary": "retrospective sign and support structure in the behavior-qualified C079 success domain; no coordinate is named a neuron, semantic anchor, or causal unit",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    support_path = OUT / "atlas/support_membership.uint8.npy"
    sign_path = OUT / "atlas/relation_mean_sign.int8.npy"
    output_checks = {
        "finite": bool(np.isfinite(relation_mean).all()),
        "support_shape": list(np.load(support_path, mmap_mode="r").shape) == list(shape),
        "sign_shape": list(np.load(sign_path, mmap_mode="r").shape) == list(relation_mean.shape),
        "support_counts": all(np.all(np.sum(support[index], axis=-1) == count) for index, count in enumerate(counts)),
        "layer_rows": len(layer_rows) == 333,
        "early_rows": len(early) == 14,
        "panel_loo_rows": len(panel_loo) == 36,
        "frozen17_rows": len(fixed_rows) == 17,
    }
    if not all(output_checks.values()):
        raise RuntimeError(output_checks)
    summary["output_checks"] = output_checks
    summary["files"] = {
        "support_membership.uint8.npy": {"bytes": support_path.stat().st_size, "sha256": core.sha(support_path)},
        "relation_mean_sign.int8.npy": {"bytes": sign_path.stat().st_size, "sha256": core.sha(sign_path)},
    }
    core.save(OUT / "analysis/coordinate_atlas_summary.json", summary)
    core.save(OUT / "analysis/final.json", {
        "phase": 1485,
        "campaign": "C084",
        "status": "coordinate_stability_atlas_complete",
        "output_checks": output_checks,
        "model_run": False,
        "hidden_access": "read existing legal C082 derivative only",
        "authorization": "run_phase1486_c084_factorial_surface_atlas",
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
