#!/usr/bin/env python3
"""Phase1554: compute the C095 triadic interaction and complete 3x3 field atlas."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1553_c095_existing_field_batch_mining_contract"
FIELD_SOURCE = RESULT / "phase1539_c091_canonical_all_state_capture"
PAIR_SOURCE = RESULT / "phase1536_c091_human_validated_chinese_relation_contract"
OUT = RESULT / "phase1554_c095_triadic_interaction_and_field_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PARTITIONS = ("response_discovery", "confirmation", "lockbox")
SURFACES = ("prequery", "postquery")
CONCRETENESS = ("concrete", "abstract")
FAMILIES = ("similarity", "class_inclusion", "whole_part")
FAMILY_PAIRS = (("similarity", "class_inclusion"), ("similarity", "whole_part"), ("class_inclusion", "whole_part"))
ROLES = ("source_word", "target_word", "relation_anchor", "boundary")
STATES = 37
COORDINATES = 2560


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def selected_pairs(pairs: list[dict], partition: str, family: str, concreteness: str) -> list[dict]:
    return sorted(
        [row for row in pairs if row["partition"] == partition and row["family"] == family and row["concreteness"] == concreteness],
        key=lambda row: row["pair_id"],
    )


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1554 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    contract = core.load(CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1554_c095_triadic_interaction_and_field_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1553 authorization missing")

    field_path = ROOT / contract["source_assets"]["field"]["path"]
    index_path = ROOT / contract["source_assets"]["index"]["path"]
    field = np.load(field_path, mmap_mode="r")
    index = core.rows(index_path)
    pairs = core.rows(PAIR_SOURCE / "material/frozen_pairs.jsonl")
    lookup = {(row["pair_id"], row["surface"], row["query_family"]): int(row["row_index"]) for row in index}

    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    individual_path = raw_dir / "triadic_individual_interactions.float16.npy"
    individual = np.lib.format.open_memmap(individual_path, mode="w+", dtype=np.float16, shape=(180, STATES, len(ROLES), COORDINATES))
    individual_index: list[dict] = []
    group_rows: dict[tuple[str, str, str, str, str], list[int]] = {}
    row_cursor = 0
    for partition in PARTITIONS:
        for surface in SURFACES:
            for concreteness in CONCRETENESS:
                family_rows = {family: selected_pairs(pairs, partition, family, concreteness) for family in FAMILIES}
                if any(len(rows) != 5 for rows in family_rows.values()):
                    raise RuntimeError({key: len(value) for key, value in family_rows.items()})
                for family_a, family_b in FAMILY_PAIRS:
                    group_key = (partition, surface, concreteness, family_a, family_b)
                    group_rows[group_key] = []
                    for matched_rank, (pair_a, pair_b) in enumerate(zip(family_rows[family_a], family_rows[family_b], strict=True)):
                        h_aa = field[lookup[(pair_a["pair_id"], surface, family_a)]].astype(np.float32)
                        h_bb = field[lookup[(pair_b["pair_id"], surface, family_b)]].astype(np.float32)
                        h_ab = field[lookup[(pair_a["pair_id"], surface, family_b)]].astype(np.float32)
                        h_ba = field[lookup[(pair_b["pair_id"], surface, family_a)]].astype(np.float32)
                        individual[row_cursor] = (0.5 * (h_aa + h_bb - h_ab - h_ba)).astype(np.float16)
                        group_rows[group_key].append(row_cursor)
                        individual_index.append({
                            "row_index": row_cursor,
                            "partition": partition,
                            "surface": surface,
                            "concreteness": concreteness,
                            "family_a": family_a,
                            "family_b": family_b,
                            "matched_rank": matched_rank,
                            "pair_a": pair_a["pair_id"],
                            "pair_b": pair_b["pair_id"],
                        })
                        row_cursor += 1
    individual.flush()
    if row_cursor != 180:
        raise RuntimeError(row_cursor)
    individual_index_path = raw_dir / "triadic_individual_interactions_index.jsonl"
    core.write_rows(individual_index_path, individual_index)

    centroid_path = raw_dir / "triadic_interaction_centroids.float32.npy"
    centroid_shape = (len(PARTITIONS), len(SURFACES), len(CONCRETENESS), len(FAMILY_PAIRS), STATES, len(ROLES), COORDINATES)
    centroids = np.lib.format.open_memmap(centroid_path, mode="w+", dtype=np.float32, shape=centroid_shape)
    interaction_atlas: list[dict] = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                    rows = group_rows[(partition, surface, concreteness, family_a, family_b)]
                    stack = np.asarray(individual[rows], dtype=np.float32)
                    centroid = stack.mean(axis=0)
                    centroids[pi, si, ci, fi] = centroid
                    vector_norms = np.linalg.norm(stack, axis=-1)
                    centroid_norms = np.linalg.norm(centroid, axis=-1)
                    dots = np.einsum("nsrc,src->nsr", stack, centroid, optimize=True)
                    denominators = vector_norms * centroid_norms[None, :, :]
                    alignments = np.divide(dots, denominators, out=np.zeros_like(dots), where=denominators > 0)
                    for state in range(STATES):
                        for ri, role in enumerate(ROLES):
                            values = alignments[:, state, ri]
                            interaction_atlas.append({
                                "partition": partition,
                                "surface": surface,
                                "concreteness": concreteness,
                                "family_a": family_a,
                                "family_b": family_b,
                                "state": state,
                                "role": role,
                                "n_quartets": len(rows),
                                "centroid_norm": float(centroid_norms[state, ri]),
                                "median_individual_alignment": float(np.median(values)),
                                "minimum_individual_alignment": float(np.min(values)),
                                "maximum_individual_alignment": float(np.max(values)),
                            })
    centroids.flush()
    interaction_atlas_path = OUT / "analysis/triadic_interaction_atlas.jsonl"
    core.write_rows(interaction_atlas_path, interaction_atlas)

    triadic_cosines: list[dict] = []
    for pi, partition in enumerate(PARTITIONS):
        for si, surface in enumerate(SURFACES):
            for ci, concreteness in enumerate(CONCRETENESS):
                for state in range(STATES):
                    for ri, role in enumerate(ROLES):
                        vectors = [np.asarray(centroids[pi, si, ci, fi, state, ri], dtype=np.float64) for fi in range(3)]
                        values = [cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2]), cosine(vectors[1], vectors[2])]
                        triadic_cosines.append({
                            "partition": partition,
                            "surface": surface,
                            "concreteness": concreteness,
                            "state": state,
                            "role": role,
                            "similarity_class_vs_similarity_whole": values[0],
                            "similarity_class_vs_class_whole": values[1],
                            "similarity_whole_vs_class_whole": values[2],
                            "minimum_pairwise_cosine": float(min(values)),
                            "median_pairwise_cosine": float(np.median(values)),
                        })
    triadic_cosines_path = OUT / "analysis/triadic_family_pair_cosines.jsonl"
    core.write_rows(triadic_cosines_path, triadic_cosines)

    cross_partition: list[dict] = []
    partition_pairs = ((0, 1), (0, 2), (1, 2))
    for si, surface in enumerate(SURFACES):
        for ci, concreteness in enumerate(CONCRETENESS):
            for fi, (family_a, family_b) in enumerate(FAMILY_PAIRS):
                for state in range(STATES):
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
    cross_partition_path = OUT / "analysis/cross_partition_interaction_cosines.jsonl"
    core.write_rows(cross_partition_path, cross_partition)

    component_labels = ["mu"] + [f"pair_main:{family}" for family in FAMILIES] + [f"query_main:{family}" for family in FAMILIES] + [f"interaction:{pair_family}:{query_family}" for pair_family in FAMILIES for query_family in FAMILIES]
    component_path = raw_dir / "three_by_three_raw_components.float16.npy"
    components = np.lib.format.open_memmap(component_path, mode="w+", dtype=np.float16, shape=(12, len(component_labels), STATES, len(ROLES), COORDINATES))
    panel_index: list[dict] = []
    decomposition_atlas: list[dict] = []
    panel_cursor = 0
    max_reconstruction_error = 0.0
    for partition in PARTITIONS:
        for surface in SURFACES:
            for concreteness in CONCRETENESS:
                family_rows = {family: selected_pairs(pairs, partition, family, concreteness) for family in FAMILIES}
                cells = np.empty((3, 3, STATES, len(ROLES), COORDINATES), dtype=np.float32)
                for fi, pair_family in enumerate(FAMILIES):
                    row_ids = family_rows[pair_family]
                    for qi, query_family in enumerate(FAMILIES):
                        rows = [lookup[(item["pair_id"], surface, query_family)] for item in row_ids]
                        cells[fi, qi] = np.asarray(field[rows], dtype=np.float32).mean(axis=0)
                mu = cells.mean(axis=(0, 1))
                pair_main = cells.mean(axis=1) - mu[None, ...]
                query_main = cells.mean(axis=0) - mu[None, ...]
                interaction = cells - mu[None, None, ...] - pair_main[:, None, ...] - query_main[None, :, ...]
                reconstructed = mu[None, None, ...] + pair_main[:, None, ...] + query_main[None, :, ...] + interaction
                reconstruction_error = float(np.max(np.abs(reconstructed - cells)))
                max_reconstruction_error = max(max_reconstruction_error, reconstruction_error)
                packed = np.concatenate((mu[None, ...], pair_main, query_main, interaction.reshape(9, STATES, len(ROLES), COORDINATES)), axis=0)
                components[panel_cursor] = packed.astype(np.float16)
                panel_index.append({"panel_index": panel_cursor, "partition": partition, "surface": surface, "concreteness": concreteness})
                for mode in ("raw", "dynamic_state0_subtracted"):
                    current = packed if mode == "raw" else packed - packed[:, 0:1, :, :]
                    for state in range(STATES):
                        for ri, role in enumerate(ROLES):
                            vectors = current[:, state, ri].astype(np.float64)
                            decomposition_atlas.append({
                                "partition": partition,
                                "surface": surface,
                                "concreteness": concreteness,
                                "mode": mode,
                                "state": state,
                                "role": role,
                                "mu_norm": float(np.linalg.norm(vectors[0])),
                                "pair_main_mean_energy": float(np.mean(np.sum(vectors[1:4] ** 2, axis=1))),
                                "query_main_mean_energy": float(np.mean(np.sum(vectors[4:7] ** 2, axis=1))),
                                "interaction_mean_energy": float(np.mean(np.sum(vectors[7:16] ** 2, axis=1))),
                                "reconstruction_max_abs": reconstruction_error,
                            })
                panel_cursor += 1
    components.flush()
    panel_index_path = raw_dir / "three_by_three_panel_index.jsonl"
    core.write_rows(panel_index_path, panel_index)
    component_labels_path = OUT / "protocol/three_by_three_component_labels.json"
    core.save(component_labels_path, {"labels": component_labels, "dynamic_rule": "component[state] - component[state0]"})
    decomposition_atlas_path = OUT / "analysis/three_by_three_decomposition_atlas.jsonl"
    core.write_rows(decomposition_atlas_path, decomposition_atlas)

    focus = [row for row in triadic_cosines if row["role"] == "boundary" and row["state"] in {31, 32}]
    cross_focus = [row for row in cross_partition if row["role"] == "boundary" and row["state"] in {31, 32}]
    summary = {
        "phase": 1554,
        "campaign": "C095",
        "status": "triadic_and_three_by_three_atlas_complete",
        "model_run": False,
        "triadic_focus": {
            "n_panels": len(focus),
            "minimum_of_minimum_pairwise_cosine": float(min(row["minimum_pairwise_cosine"] for row in focus)),
            "median_minimum_pairwise_cosine": float(np.median([row["minimum_pairwise_cosine"] for row in focus])),
            "maximum_of_minimum_pairwise_cosine": float(max(row["minimum_pairwise_cosine"] for row in focus)),
        },
        "cross_partition_focus": {
            "n_rows": len(cross_focus),
            "minimum_of_minimum_partition_cosine": float(min(row["minimum_partition_cosine"] for row in cross_focus)),
            "median_minimum_partition_cosine": float(np.median([row["minimum_partition_cosine"] for row in cross_focus])),
        },
        "max_reconstruction_error": max_reconstruction_error,
        "coverage": {
            "individual_interactions": len(individual_index),
            "interaction_atlas_rows": len(interaction_atlas),
            "triadic_cosine_rows": len(triadic_cosines),
            "cross_partition_rows": len(cross_partition),
            "decomposition_rows": len(decomposition_atlas),
            "panels": len(panel_index),
        },
        "files": {
            "individual_interactions": {"path": str(individual_path.relative_to(ROOT)), "sha256": core.sha(individual_path), "shape": [180, 37, 4, 2560]},
            "individual_index": {"path": str(individual_index_path.relative_to(ROOT)), "sha256": core.sha(individual_index_path), "rows": len(individual_index)},
            "centroids": {"path": str(centroid_path.relative_to(ROOT)), "sha256": core.sha(centroid_path), "shape": list(centroid_shape)},
            "interaction_atlas": {"path": str(interaction_atlas_path.relative_to(ROOT)), "sha256": core.sha(interaction_atlas_path), "rows": len(interaction_atlas)},
            "triadic_cosines": {"path": str(triadic_cosines_path.relative_to(ROOT)), "sha256": core.sha(triadic_cosines_path), "rows": len(triadic_cosines)},
            "cross_partition": {"path": str(cross_partition_path.relative_to(ROOT)), "sha256": core.sha(cross_partition_path), "rows": len(cross_partition)},
            "components": {"path": str(component_path.relative_to(ROOT)), "sha256": core.sha(component_path), "shape": [12, 16, 37, 4, 2560]},
            "panel_index": {"path": str(panel_index_path.relative_to(ROOT)), "sha256": core.sha(panel_index_path), "rows": len(panel_index)},
            "component_labels": {"path": str(component_labels_path.relative_to(ROOT)), "sha256": core.sha(component_labels_path)},
            "decomposition_atlas": {"path": str(decomposition_atlas_path.relative_to(ROOT)), "sha256": core.sha(decomposition_atlas_path), "rows": len(decomposition_atlas)},
        },
        "claim_boundary": {
            "allowed": "retrospective descriptive triadic and 3x3 structure in the immutable C091 field",
            "missing": ["behavior qualification for similarity/class", "output-code orthogonalization", "causal intervention", "new blind split", "external model/task"],
            "forbidden": ["pure relation semantics", "causal circuit", "identified neurons", "cross-model law", "new mathematics"],
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1555_c095_behavior_stratified_and_raw_coordinate_atlas",
    }
    core.save(OUT / "analysis/triadic_and_field_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1554, "campaign": "C095", "status": summary["status"], "authorization": summary["authorization"]})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
