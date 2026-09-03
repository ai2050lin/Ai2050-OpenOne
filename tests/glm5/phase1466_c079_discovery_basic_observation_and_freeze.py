#!/usr/bin/env python3
"""Phase1466: basic discovery observation and full-vector candidate freeze."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

CONTRACT = TESTS / "result/phase1463_c079_aggregate_observation_contract"
CAPTURE = TESTS / "result/phase1465_c079_discovery_full_field_capture"
OUT = TESTS / "result/phase1466_c079_discovery_basic_observation_and_freeze"
FACTORS = {"entity_nuisance": 0, "object_nuisance": 1, "relation_label": 2}


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def factor_effect(block: np.ndarray, cells: list[str], bit: int) -> np.ndarray:
    by_cell = {cell: block[index] for index, cell in enumerate(cells)}
    values = []
    for cell in cells:
        if cell[bit] == "1":
            opposite = cell[:bit] + "0" + cell[bit + 1:]
            values.append(by_cell[cell] - by_cell[opposite])
    return np.mean(np.stack(values, axis=0), axis=0, dtype=np.float32)


def select_candidates(pool: list[dict], relations: list[str]) -> list[dict]:
    selected = []
    for relation in relations:
        rows = [row for row in pool if row["relation"] == relation]
        best_by_role = {}
        for row in rows:
            role = row["role"]
            if role not in best_by_role or (row["score"], -row["state"]) > (best_by_role[role]["score"], -best_by_role[role]["state"]):
                best_by_role[role] = row
        selected.extend(sorted(best_by_role.values(), key=lambda row: (-row["score"], row["state"], row["role"]))[:3])
    return selected


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1466 exists")
    capture = core.load(CAPTURE / "analysis/final.json")
    capture_audit = core.load(CAPTURE / "audit/independent_final_audit.json")
    metadata = core.load(CAPTURE / "analysis/capture_metadata.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if capture["authorization"] != "run_phase1466_c079_discovery_basic_observation_and_freeze" or not capture_audit["all_checks_passed"]:
        raise RuntimeError("Phase1465 did not authorize observation")
    raw_path = CAPTURE / "raw/discovery_role_field.float16.npy"
    if core.sha(raw_path) != metadata["raw_sha256"]:
        raise RuntimeError("raw field changed")
    field = np.load(raw_path, mmap_mode="r")
    index = core.rows(CAPTURE / "raw/discovery_role_field_index.jsonl")
    if not all(row["partition"] == "response_discovery" for row in index):
        raise RuntimeError("holdout present")
    lookup = {(row["family"], row["index"], row["record_relation_id"], row["surface"], row["cell"]): row["row_index"] for row in index}
    set_keys = sorted({(row["family"], row["index"], row["record_relation_id"]) for row in index})
    surface_rows, cross_rows, pool = [], [], []
    stats = {}
    relation_means = {}
    roles = protocol["role_slots"]
    relations = list(protocol["relations"])
    cells = list(protocol["cells"])
    for relation in relations:
        current_sets = [key for key in set_keys if key[2] == relation]
        effect_samples = {surface: {factor: [] for factor in FACTORS} for surface in protocol["surfaces"]}
        for family, index_value, _ in current_sets:
            for surface in protocol["surfaces"]:
                row_ids = [lookup[(family, index_value, relation, surface, cell)] for cell in cells]
                block = np.asarray(field[row_ids], dtype=np.float32)
                for factor, bit in FACTORS.items():
                    effect_samples[surface][factor].append(factor_effect(block, cells, bit))
                del block
        means = {}
        for surface in protocol["surfaces"]:
            for factor in FACTORS:
                samples = np.stack(effect_samples[surface][factor], axis=0)
                mean = np.mean(samples, axis=0, dtype=np.float32)
                sample_norm = np.linalg.norm(samples, axis=-1)
                mean_norm = np.linalg.norm(mean, axis=-1)
                direction = np.mean(np.sum(samples * mean[None], axis=-1) / np.maximum(sample_norm * mean_norm[None], 1e-12), axis=0)
                sign = np.mean(np.sign(samples) == np.sign(mean[None]), axis=(0, 3))
                means[(surface, factor)] = mean
                stats[(relation, surface, factor)] = {"mean_norm": mean_norm, "sample_norm": np.mean(sample_norm, axis=0), "direction": direction, "sign": sign}
                for state in range(field.shape[1]):
                    for role_index, role in enumerate(roles):
                        surface_rows.append({"relation": relation, "surface": surface, "effect": factor, "sample_count": len(samples), "state": state, "role": role, "mean_l2": float(mean_norm[state, role_index]), "mean_sample_l2": float(np.mean(sample_norm[:, state, role_index])), "direction_consistency": float(direction[state, role_index]), "coordinate_sign_consistency": float(sign[state, role_index])})
                del samples
        for factor in FACTORS:
            left = means[(protocol["surfaces"][0], factor)]
            right = means[(protocol["surfaces"][1], factor)]
            cross = cosine(left, right)
            for state in range(field.shape[1]):
                for role_index, role in enumerate(roles):
                    cross_rows.append({"relation": relation, "effect": factor, "state": state, "role": role, "cross_surface_cosine": float(cross[state, role_index])})
        relation_means[relation] = {surface: means[(surface, "relation_label")] for surface in protocol["surfaces"]}
        cross_relation = cosine(means[(protocol["surfaces"][0], "relation_label")], means[(protocol["surfaces"][1], "relation_label")])
        for state in range(field.shape[1]):
            for role_index, role in enumerate(roles):
                directions = [stats[(relation, surface, "relation_label")]["direction"][state, role_index] for surface in protocol["surfaces"]]
                selectivities = []
                for surface in protocol["surfaces"]:
                    rel_norm = stats[(relation, surface, "relation_label")]["mean_norm"][state, role_index]
                    nuisance = max(stats[(relation, surface, "entity_nuisance")]["mean_norm"][state, role_index], stats[(relation, surface, "object_nuisance")]["mean_norm"][state, role_index], 1e-12)
                    selectivities.append(float(rel_norm / nuisance))
                cross_value = float(cross_relation[state, role_index])
                direction_value = float(min(directions))
                selectivity_value = float(min(selectivities))
                score = max(cross_value, 0.0) * max(direction_value, 0.0) * min(max(selectivity_value, 0.0), 1.0)
                pool.append({"relation": relation, "state": state, "role": role, "cross_surface_cosine": cross_value, "minimum_direction_consistency": direction_value, "minimum_selectivity_ratio": selectivity_value, "score": score})
    selected = select_candidates(pool, relations)
    vectors = {}
    candidates = []
    for candidate_index, row in enumerate(selected):
        role_index = roles.index(row["role"])
        candidate_id = f"{row['relation']}__state{row['state']:02d}__{row['role']}"
        candidates.append({"candidate_id": candidate_id, "candidate_index": candidate_index, **row})
        for surface in protocol["surfaces"]:
            vectors[f"{candidate_id}__{surface}"] = relation_means[row["relation"]][surface][row["state"], role_index].astype(np.float32)
    OUT.joinpath("frozen").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "frozen/discovery_candidate_mean_vectors.npz", **vectors)
    core.write_rows(OUT / "analysis/surface_effect_metrics.jsonl", surface_rows)
    core.write_rows(OUT / "analysis/cross_surface_effect_metrics.jsonl", cross_rows)
    core.write_rows(OUT / "analysis/relation_candidate_pool.jsonl", pool)
    state_profile = []
    role_profile = []
    for state in range(field.shape[1]):
        values = [row["score"] for row in pool if row["state"] == state]
        state_profile.append({"state": state, "mean_score": float(np.mean(values)), "max_score": float(np.max(values))})
    for role in roles:
        values = [row["score"] for row in pool if row["role"] == role]
        role_profile.append({"role": role, "mean_score": float(np.mean(values)), "max_score": float(np.max(values))})
    manifest = {
        "phase": 1466,
        "campaign": "C079",
        "source_raw_sha256": metadata["raw_sha256"],
        "source_partition": "response_discovery",
        "factor_formula": "mean over four paired full-vector differences: factor bit 1 minus factor bit 0 while holding the other two bits fixed",
        "selection_rule": "for each relation choose the highest-score cell within each role, then retain the top three distinct roles",
        "score_formula": "max(cross_surface_cosine,0) * max(minimum_direction_consistency,0) * min(max(minimum_selectivity_ratio,0),1)",
        "candidates": candidates,
        "validation_thresholds": {"cosine_to_discovery_each_surface_min": 0.70, "holdout_cross_surface_cosine_min": 0.70, "direction_to_discovery_min": 0.50, "selectivity_ratio_min": 1.0, "both_confirmation_and_lockbox_required": True},
        "claim_scope": "full-vector layer-role regularities of an explicit label-identity effect; not unlabeled semantics or causal use",
        "holdout_accessed": False,
        "vector_npz_sha256": core.sha(OUT / "frozen/discovery_candidate_mean_vectors.npz"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest["freeze_sha256"] = core.digest(manifest)
    core.save(OUT / "frozen/candidate_manifest.json", manifest)
    summary = {"phase": 1466, "campaign": "C079", "relation_set_counts": {relation: sum(key[2] == relation for key in set_keys) for relation in relations}, "surface_metric_rows": len(surface_rows), "cross_metric_rows": len(cross_rows), "candidate_pool_rows": len(pool), "candidate_count": len(candidates), "candidates": candidates, "state_profile": state_profile, "role_profile": role_profile, "raw_hash_unchanged": core.sha(raw_path) == metadata["raw_sha256"], "holdout_accessed": False}
    core.save(OUT / "analysis/observation_summary.json", summary)
    core.save(OUT / "analysis/final.json", {"phase": 1466, "campaign": "C079", "observation_complete": True, "freeze_sha256": manifest["freeze_sha256"], "authorization": "run_phase1467_c079_holdout_capture_and_validation"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
