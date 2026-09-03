#!/usr/bin/env python3
"""Phase1559: reveal the frozen C096 field predictions and cross-material diagnostics."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1557_c096_fresh_human_relation_field_contract"
PARENT = RESULT / "phase1558_c096_unified_behavior_and_all_state_capture"
C095 = RESULT / "phase1554_c095_triadic_interaction_and_field_atlas"
OUT = RESULT / "phase1559_c096_fresh_prediction_atlas_and_adjudication"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
CONCRETENESS = ("concrete", "abstract")
FAMILIES = ("similarity", "class_inclusion", "whole_part")
FAMILY_PAIRS = (("similarity", "class_inclusion"), ("similarity", "whole_part"), ("class_inclusion", "whole_part"))
ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
FOCUS_STATES = (31, 32)
SUPPORT_COUNTS = (16, 64, 256, 1024)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def selected_pairs(pairs: list[dict], partition: str, family: str, concreteness: str) -> list[dict]:
    return sorted([row for row in pairs if row["partition"] == partition and row["family"] == family and row["concreteness"] == concreteness], key=lambda row: row["pair_id"])


def top_indices(vector: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(vector), kind="stable")[-k:]


def energy_fraction(vector: np.ndarray, indices: np.ndarray) -> float:
    denominator = float(np.dot(vector, vector))
    return float(np.dot(vector[indices], vector[indices]) / denominator) if denominator > 0 else 0.0


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1559 exists")
    parent = core.load(PARENT / "analysis/final.json")
    parent_audit = core.load(PARENT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1559_c096_fresh_prediction_atlas_and_adjudication" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1558 authorization missing")
    field = np.load(PARENT / "raw/c096_all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/c096_all_role_field_index.jsonl")
    pairs = core.rows(CONTRACT / "material/frozen_fresh_pairs.jsonl")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): int(row["row_index"]) for row in index}

    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    individual_path = raw_dir / "c096_triadic_individual_interactions.float16.npy"
    individual = np.lib.format.open_memmap(individual_path, mode="w+", dtype=np.float16, shape=(180, 37, 4, 2560))
    individual_index: list[dict] = []
    group_rows: dict[tuple[str, str, str, str, str], list[int]] = {}
    cursor = 0
    for partition in PARTITIONS:
        for surface in SURFACES:
            for concreteness in CONCRETENESS:
                family_rows = {family: selected_pairs(pairs, partition, family, concreteness) for family in FAMILIES}
                for family_a, family_b in FAMILY_PAIRS:
                    key = (partition, surface, concreteness, family_a, family_b)
                    group_rows[key] = []
                    for matched_rank, (pair_a, pair_b) in enumerate(zip(family_rows[family_a], family_rows[family_b], strict=True)):
                        h_aa = field[lookup[(pair_a["pair_id"], surface, family_a)]].astype(np.float32)
                        h_bb = field[lookup[(pair_b["pair_id"], surface, family_b)]].astype(np.float32)
                        h_ab = field[lookup[(pair_a["pair_id"], surface, family_b)]].astype(np.float32)
                        h_ba = field[lookup[(pair_b["pair_id"], surface, family_a)]].astype(np.float32)
                        individual[cursor] = (0.5 * (h_aa + h_bb - h_ab - h_ba)).astype(np.float16)
                        group_rows[key].append(cursor)
                        individual_index.append({"row_index": cursor, "partition": partition, "surface": surface, "concreteness": concreteness, "family_a": family_a, "family_b": family_b, "matched_rank": matched_rank, "pair_a": pair_a["pair_id"], "pair_b": pair_b["pair_id"]})
                        cursor += 1
    individual.flush()
    individual_index_path = raw_dir / "c096_triadic_individual_interactions_index.jsonl"
    core.write_rows(individual_index_path, individual_index)

    centroid_path = raw_dir / "c096_triadic_interaction_centroids.float32.npy"
    centroids = np.lib.format.open_memmap(centroid_path, mode="w+", dtype=np.float32, shape=(3, 2, 2, 3, 37, 4, 2560))
    interaction_atlas = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                    stack = np.asarray(individual[group_rows[(partition, surface, concreteness, family_a, family_b)]], dtype=np.float32)
                    centroid = stack.mean(axis=0)
                    centroids[pi, si, ci, fi] = centroid
                    vector_norms = np.linalg.norm(stack, axis=-1)
                    centroid_norms = np.linalg.norm(centroid, axis=-1)
                    dots = np.einsum("nsrc,src->nsr", stack, centroid, optimize=True)
                    denominators = vector_norms * centroid_norms[None, :, :]
                    alignments = np.divide(dots, denominators, out=np.zeros_like(dots), where=denominators > 0)
                    for state in range(37):
                        for ri, role in enumerate(ROLES):
                            values = alignments[:, state, ri]
                            interaction_atlas.append({"partition": partition, "surface": surface, "concreteness": concreteness, "family_a": family_a, "family_b": family_b, "state": state, "role": role, "n_quartets": 5, "centroid_norm": float(centroid_norms[state, ri]), "median_individual_alignment": float(np.median(values)), "minimum_individual_alignment": float(np.min(values))})
    centroids.flush()
    interaction_path = OUT / "analysis/c096_interaction_atlas.jsonl"
    core.write_rows(interaction_path, interaction_atlas)

    triadic = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for state in range(37):
                    for ri, role in enumerate(ROLES):
                        vectors = [np.asarray(centroids[pi, si, ci, fi, state, ri], dtype=np.float64) for fi in range(3)]
                        values = [cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2]), cosine(vectors[1], vectors[2])]
                        triadic.append({"partition": partition, "surface": surface, "concreteness": concreteness, "state": state, "role": role, "similarity_class_vs_similarity_whole": values[0], "similarity_class_vs_class_whole": values[1], "similarity_whole_vs_class_whole": values[2], "minimum_pairwise_cosine": float(min(values)), "median_pairwise_cosine": float(np.median(values))})
    triadic_path = OUT / "analysis/c096_triadic_family_pair_cosines.jsonl"
    core.write_rows(triadic_path, triadic)

    cross_partition = []
    partition_pairs = ((0, 1), (0, 2), (1, 2))
    for si, surface in enumerate(SURFACES):
        for ci, concreteness in enumerate(CONCRETENESS):
            for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                for state in range(37):
                    for ri, role in enumerate(ROLES):
                        values = []
                        row = {"surface": surface, "concreteness": concreteness, "family_a": family_a, "family_b": family_b, "state": state, "role": role}
                        for left, right in partition_pairs:
                            value = cosine(np.asarray(centroids[left, si, ci, fi, state, ri], dtype=np.float64), np.asarray(centroids[right, si, ci, fi, state, ri], dtype=np.float64))
                            row[f"{PARTITIONS[left]}_vs_{PARTITIONS[right]}"] = value
                            values.append(value)
                        row["minimum_partition_cosine"] = float(min(values))
                        row["median_partition_cosine"] = float(np.median(values))
                        cross_partition.append(row)
    cross_partition_path = OUT / "analysis/c096_cross_partition_cosines.jsonl"
    core.write_rows(cross_partition_path, cross_partition)

    concentration = []
    stability = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                    for state in FOCUS_STATES:
                        vector = np.asarray(centroids[pi, si, ci, fi, state, 3], dtype=np.float64)
                        for k in SUPPORT_COUNTS:
                            indices = top_indices(vector, k)
                            concentration.append({"partition": partition, "surface": surface, "concreteness": concreteness, "family_a": family_a, "family_b": family_b, "state": state, "role": "boundary", "k": k, "energy_fraction": energy_fraction(vector, indices)})
    for si, surface in enumerate(SURFACES):
        for ci, concreteness in enumerate(CONCRETENESS):
            for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                for state in FOCUS_STATES:
                    reference = np.asarray(centroids[0, si, ci, fi, state, 3], dtype=np.float64)
                    for target_pi in (1, 2):
                        target = np.asarray(centroids[target_pi, si, ci, fi, state, 3], dtype=np.float64)
                        for k in SUPPORT_COUNTS:
                            ref_idx = top_indices(reference, k)
                            tgt_idx = top_indices(target, k)
                            nonzero = (reference[ref_idx] != 0) & (target[ref_idx] != 0)
                            stability.append({
                                "reference_partition": PARTITIONS[0], "target_partition": PARTITIONS[target_pi], "surface": surface, "concreteness": concreteness, "family_a": family_a, "family_b": family_b, "state": state, "role": "boundary", "k": k,
                                "sign_agreement": float(np.mean(np.sign(reference[ref_idx][nonzero]) == np.sign(target[ref_idx][nonzero]))) if np.any(nonzero) else 0.0,
                                "support_jaccard": float(len(set(ref_idx.tolist()) & set(tgt_idx.tolist())) / len(set(ref_idx.tolist()) | set(tgt_idx.tolist()))),
                                "restricted_cosine": cosine(reference[ref_idx], target[ref_idx]),
                                "full_cosine": cosine(reference, target),
                            })
    concentration_path = OUT / "analysis/c096_raw_coordinate_concentration.jsonl"
    stability_path = OUT / "analysis/c096_raw_coordinate_stability.jsonl"
    core.write_rows(concentration_path, concentration)
    core.write_rows(stability_path, stability)

    c095_centroids = np.load(C095 / "raw/triadic_interaction_centroids.float32.npy", mmap_mode="r")
    cross_material = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                    for state in FOCUS_STATES:
                        old = np.asarray(c095_centroids[pi, si, ci, fi, state, 3], dtype=np.float64)
                        new = np.asarray(centroids[pi, si, ci, fi, state, 3], dtype=np.float64)
                        old64 = top_indices(old, 64)
                        new64 = top_indices(new, 64)
                        nonzero = (old[old64] != 0) & (new[old64] != 0)
                        cross_material.append({
                            "partition": partition, "surface": surface, "concreteness": concreteness, "family_a": family_a, "family_b": family_b, "state": state, "role": "boundary",
                            "full_cosine": cosine(old, new),
                            "old_top64_restricted_cosine": cosine(old[old64], new[old64]),
                            "old_top64_sign_agreement": float(np.mean(np.sign(old[old64][nonzero]) == np.sign(new[old64][nonzero]))) if np.any(nonzero) else 0.0,
                            "top64_support_jaccard": float(len(set(old64.tolist()) & set(new64.tolist())) / len(set(old64.tolist()) | set(new64.tolist()))),
                        })
    cross_material_path = OUT / "analysis/c091_to_c096_cross_material_coordinate_cosines.jsonl"
    core.write_rows(cross_material_path, cross_material)

    prequery_focus = [row for row in triadic if row["surface"] == "prequery" and row["role"] == "boundary" and row["state"] in FOCUS_STATES]
    postquery_focus = [row for row in triadic if row["surface"] == "postquery" and row["role"] == "boundary" and row["state"] in FOCUS_STATES]
    cross_focus = [row for row in cross_partition if row["role"] == "boundary" and row["state"] in FOCUS_STATES]
    k64 = [row for row in concentration if row["k"] == 64]
    stability64 = [row for row in stability if row["k"] == 64]
    pre_median = float(np.median([row["minimum_pairwise_cosine"] for row in prequery_focus]))
    post_median = float(np.median([row["minimum_pairwise_cosine"] for row in postquery_focus]))
    measurements = {
        "P096_1_min_prequery_triadic": float(min(row["minimum_pairwise_cosine"] for row in prequery_focus)),
        "P096_2_min_cross_partition": float(min(row["minimum_partition_cosine"] for row in cross_focus)),
        "P096_3_median_top64_energy": float(np.median([row["energy_fraction"] for row in k64])),
        "P096_4_min_top64_sign_agreement": float(min(row["sign_agreement"] for row in stability64)),
        "P096_4_min_top64_restricted_cosine": float(min(row["restricted_cosine"] for row in stability64)),
        "P096_5_prequery_median": pre_median,
        "P096_5_postquery_median": post_median,
        "P096_5_gap_pre_minus_post": pre_median - post_median,
    }
    predictions = {
        "P096_1_prequery_common_field": {"passed": measurements["P096_1_min_prequery_triadic"] >= 0.75, "threshold": ">=0.75", "observed": measurements["P096_1_min_prequery_triadic"]},
        "P096_2_cross_partition": {"passed": measurements["P096_2_min_cross_partition"] >= 0.75, "threshold": ">=0.75", "observed": measurements["P096_2_min_cross_partition"]},
        "P096_3_top64_energy": {"passed": 0.15 <= measurements["P096_3_median_top64_energy"] <= 0.25, "threshold": "[0.15,0.25]", "observed": measurements["P096_3_median_top64_energy"]},
        "P096_4_coordinate_stability": {"passed": measurements["P096_4_min_top64_sign_agreement"] >= 0.90 and measurements["P096_4_min_top64_restricted_cosine"] >= 0.85, "threshold": "sign>=0.90 and restricted cosine>=0.85", "observed": {"minimum_sign_agreement": measurements["P096_4_min_top64_sign_agreement"], "minimum_restricted_cosine": measurements["P096_4_min_top64_restricted_cosine"]}},
        "P096_5_order_conditioning": {"passed": measurements["P096_5_gap_pre_minus_post"] >= 0.05, "threshold": "prequery median - postquery median >=0.05", "observed": {"prequery_median": pre_median, "postquery_median": post_median, "gap": pre_median - post_median}},
    }
    passed_count = sum(item["passed"] for item in predictions.values())
    summary = {
        "phase": 1559,
        "campaign": "C096",
        "status": "fresh_predictions_individually_adjudicated",
        "predictions": predictions,
        "passed_predictions": passed_count,
        "total_predictions": 5,
        "all_predictions_passed": passed_count == 5,
        "cross_material_c091_to_c096": {
            "minimum_full_cosine": float(min(row["full_cosine"] for row in cross_material)),
            "median_full_cosine": float(np.median([row["full_cosine"] for row in cross_material])),
            "minimum_old_top64_sign_agreement": float(min(row["old_top64_sign_agreement"] for row in cross_material)),
            "median_old_top64_sign_agreement": float(np.median([row["old_top64_sign_agreement"] for row in cross_material])),
            "minimum_top64_support_jaccard": float(min(row["top64_support_jaccard"] for row in cross_material)),
            "median_top64_support_jaccard": float(np.median([row["top64_support_jaccard"] for row in cross_material])),
        },
        "coverage": {"individual_interactions": len(individual_index), "interaction_rows": len(interaction_atlas), "triadic_rows": len(triadic), "cross_partition_rows": len(cross_partition), "concentration_rows": len(concentration), "stability_rows": len(stability), "cross_material_rows": len(cross_material)},
        "files": {
            "individual": {"path": str(individual_path.relative_to(ROOT)), "sha256": core.sha(individual_path), "shape": [180, 37, 4, 2560]},
            "individual_index": {"path": str(individual_index_path.relative_to(ROOT)), "sha256": core.sha(individual_index_path), "rows": len(individual_index)},
            "centroids": {"path": str(centroid_path.relative_to(ROOT)), "sha256": core.sha(centroid_path), "shape": [3, 2, 2, 3, 37, 4, 2560]},
            "interaction": {"path": str(interaction_path.relative_to(ROOT)), "sha256": core.sha(interaction_path), "rows": len(interaction_atlas)},
            "triadic": {"path": str(triadic_path.relative_to(ROOT)), "sha256": core.sha(triadic_path), "rows": len(triadic)},
            "cross_partition": {"path": str(cross_partition_path.relative_to(ROOT)), "sha256": core.sha(cross_partition_path), "rows": len(cross_partition)},
            "concentration": {"path": str(concentration_path.relative_to(ROOT)), "sha256": core.sha(concentration_path), "rows": len(concentration)},
            "stability": {"path": str(stability_path.relative_to(ROOT)), "sha256": core.sha(stability_path), "rows": len(stability)},
            "cross_material": {"path": str(cross_material_path.relative_to(ROOT)), "sha256": core.sha(cross_material_path), "rows": len(cross_material)},
        },
        "evidence_typing": {"whole_part": "behavior-qualified fresh lexical material", "similarity": "M_BEHAVIOR diagnostic", "class_inclusion": "M_BEHAVIOR diagnostic", "causal": "M_CAUSAL", "external": "M_EXTERNAL"},
        "claim_boundary": protocol["claim_boundary"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1560_c096_major_stage_closure",
    }
    core.save(OUT / "analysis/c096_prediction_adjudication.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1559, "campaign": "C096", "status": summary["status"], "authorization": summary["authorization"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
