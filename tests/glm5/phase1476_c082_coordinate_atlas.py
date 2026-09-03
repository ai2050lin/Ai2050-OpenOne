#!/usr/bin/env python3
"""Phase1476: build the full coordinate-resolved retrospective C079 atlas."""
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
CONTRACT = RESULT / "phase1475_c082_coordinate_atlas_contract"
C079_CONTRACT = RESULT / "phase1463_c079_aggregate_observation_contract"
DISCOVERY = RESULT / "phase1465_c079_discovery_full_field_capture"
HOLDOUT = RESULT / "phase1467_c079_holdout_capture_and_validation"
OUT = RESULT / "phase1476_c082_coordinate_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64), where=denominator > 1e-12)


def sample_effect(field: np.ndarray, row_ids: list[int]) -> np.ndarray:
    block = np.asarray(field[row_ids], dtype=np.float32)
    return ((block[0] - block[1]) + (block[2] - block[3]) + (block[4] - block[5]) + (block[6] - block[7])) * 0.25


def concentration(vector: np.ndarray) -> tuple[float, int, int, int]:
    energy = np.square(vector.astype(np.float64, copy=False))
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0, 0, 0, 0
    ordered = np.sort(energy)[::-1]
    cumulative = np.cumsum(ordered) / total
    counts = [int(np.searchsorted(cumulative, threshold, side="left") + 1) for threshold in (0.5, 0.8, 0.9)]
    return float(ordered[0] / total), counts[0], counts[1], counts[2]


def load_sources() -> tuple[dict, dict, dict]:
    discovery_field = np.load(DISCOVERY / "raw/discovery_role_field.float16.npy", mmap_mode="r")
    holdout_field = np.load(HOLDOUT / "raw/holdout_role_field.float16.npy", mmap_mode="r")
    discovery_index = core.rows(DISCOVERY / "raw/discovery_role_field_index.jsonl")
    holdout_index = core.rows(HOLDOUT / "raw/holdout_role_field_index.jsonl")
    fields = {"response_discovery": discovery_field, "confirmation": holdout_field, "lockbox": holdout_field}
    indexes = {"response_discovery": discovery_index, "confirmation": holdout_index, "lockbox": holdout_index}
    lookups = {
        split: {
            (row["family"], row["index"], row["record_relation_id"], row["surface"], row["cell"]): row["row_index"]
            for row in index if row["partition"] == split
        }
        for split, index in indexes.items()
    }
    return fields, indexes, lookups


def panel_set_keys(index: list[dict], split: str, relation: str) -> list[tuple[str, int]]:
    return sorted({(row["family"], row["index"]) for row in index if row["partition"] == split and row["record_relation_id"] == relation})


def row_ids_for(lookup: dict, family: str, index: int, relation: str, surface: str, cells: list[str]) -> list[int]:
    return [lookup[(family, index, relation, surface, cell)] for cell in cells]


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1476 exists")
    parent = core.load(CONTRACT / "analysis/final.json")
    parent_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    c079 = core.load(C079_CONTRACT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1476_c082_coordinate_atlas" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1475 did not authorize Phase1476")
    source_checks = {
        "discovery": core.sha(DISCOVERY / "raw/discovery_role_field.float16.npy") == protocol["source"]["discovery_raw_sha256"],
        "holdout": core.sha(HOLDOUT / "raw/holdout_role_field.float16.npy") == protocol["source"]["holdout_raw_sha256"],
    }
    if not all(source_checks.values()):
        raise RuntimeError(source_checks)
    relations = protocol["axes"]["relations"]
    splits = protocol["axes"]["splits"]
    surfaces = protocol["axes"]["surfaces"]
    roles = protocol["axes"]["roles"]
    cells = list(c079["cells"])
    shape = (len(relations), len(splits), len(surfaces), 37, len(roles), 2560)
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)
    means = np.lib.format.open_memmap(OUT / "atlas/mean_effect.float32.npy", mode="w+", dtype=np.float32, shape=shape)
    signs = np.lib.format.open_memmap(OUT / "atlas/sign_consistency.float16.npy", mode="w+", dtype=np.float16, shape=shape)
    counts = np.zeros(shape[:3], dtype=np.int32)
    fields, indexes, lookups = load_sources()
    layer_rows: list[dict] = []
    onset_rows: list[dict] = []
    finite = True
    for relation_index, relation in enumerate(relations):
        for split_index, split in enumerate(splits):
            keys = panel_set_keys(indexes[split], split, relation)
            if not keys:
                raise RuntimeError((relation, split, "no samples"))
            for surface_index, surface in enumerate(surfaces):
                total = np.zeros(shape[3:], dtype=np.float64)
                positive = np.zeros(shape[3:], dtype=np.uint16)
                negative = np.zeros(shape[3:], dtype=np.uint16)
                ids = [row_ids_for(lookups[split], family, index, relation, surface, cells) for family, index in keys]
                for row_ids in ids:
                    effect = sample_effect(fields[split], row_ids)
                    finite = finite and bool(np.isfinite(effect).all())
                    total += effect
                    positive += effect > 0
                    negative += effect < 0
                mean = (total / len(ids)).astype(np.float32)
                means[relation_index, split_index, surface_index] = mean
                consistency = np.where(mean > 0, positive / len(ids), np.where(mean < 0, negative / len(ids), (len(ids) - positive - negative) / len(ids)))
                signs[relation_index, split_index, surface_index] = consistency.astype(np.float16)
                counts[relation_index, split_index, surface_index] = len(ids)
                direction_sum = np.zeros(mean.shape[:-1], dtype=np.float64)
                for row_ids in ids:
                    effect = sample_effect(fields[split], row_ids)
                    direction_sum += cosine(effect, mean)
                direction = direction_sum / len(ids)
                norms = np.linalg.norm(mean, axis=-1)
                for state in range(37):
                    for role_index, role in enumerate(roles):
                        vector = mean[state, role_index]
                        max_share, k50, k80, k90 = concentration(vector)
                        adjacent = 0.0 if state == 0 else float(cosine(vector[None, :], mean[state - 1, role_index][None, :])[0])
                        layer_rows.append({
                            "relation": relation,
                            "split": split,
                            "surface": surface,
                            "state": state,
                            "role": role,
                            "sample_count": len(ids),
                            "mean_l2_norm": float(norms[state, role_index]),
                            "mean_sample_to_mean_cosine": float(direction[state, role_index]),
                            "adjacent_state_cosine": adjacent,
                            "maximum_coordinate_energy_share": max_share,
                            "k50": k50,
                            "k80": k80,
                            "k90": k90,
                            "mean_coordinate_sign_consistency": float(np.mean(consistency[state, role_index], dtype=np.float64)),
                        })
                for role_index, role in enumerate(roles):
                    trajectory = norms[:, role_index]
                    peak = float(np.max(trajectory))
                    row = {"relation": relation, "split": split, "surface": surface, "role": role, "peak_norm": peak, "peak_state": int(np.argmax(trajectory))}
                    for threshold in (0.1, 0.5, 0.9):
                        reached = np.flatnonzero(trajectory >= peak * threshold) if peak > 0 else np.array([], dtype=np.int64)
                        row[f"first_state_{int(threshold * 100)}pct"] = int(reached[0]) if len(reached) else None
                    onset_rows.append(row)
    means.flush()
    signs.flush()
    np.save(OUT / "atlas/sample_counts.int32.npy", counts)
    del means, signs
    means = np.load(OUT / "atlas/mean_effect.float32.npy", mmap_mode="r")
    panel_rows = []
    structural_zero_roles = {"record_label", "record_target", "record_relation", "record_object"}
    upstream_zero = True
    for relation_index, relation in enumerate(relations):
        for state in range(37):
            for role_index, role in enumerate(roles):
                panel = np.asarray(means[relation_index, :, :, state, role_index], dtype=np.float32).reshape(-1, 2560)
                pairwise = [float(cosine(panel[left][None, :], panel[right][None, :])[0]) for left, right in itertools.combinations(range(len(panel)), 2)]
                unanimous = np.mean((np.all(panel > 0, axis=0) | np.all(panel < 0, axis=0)))
                all_zero = bool(np.all(panel == 0))
                if role in structural_zero_roles:
                    upstream_zero = upstream_zero and all_zero
                panel_rows.append({
                    "relation": relation,
                    "state": state,
                    "role": role,
                    "minimum_panel_cosine": min(pairwise),
                    "mean_panel_cosine": float(np.mean(pairwise)),
                    "unanimous_nonzero_sign_fraction": float(unanimous),
                    "all_panels_exact_zero": all_zero,
                })
    core.write_rows(OUT / "analysis/layer_role_metrics.jsonl", layer_rows)
    core.write_rows(OUT / "analysis/onset_metrics.jsonl", onset_rows)
    core.write_rows(OUT / "analysis/panel_stability.jsonl", panel_rows)
    atlas_files = [
        OUT / "atlas/mean_effect.float32.npy",
        OUT / "atlas/sign_consistency.float16.npy",
        OUT / "atlas/sample_counts.int32.npy",
    ]
    output_checks = {
        "finite": finite and all(np.isfinite(np.load(path, mmap_mode="r")).all() for path in atlas_files),
        "shape": list(np.load(atlas_files[0], mmap_mode="r").shape) == list(shape) and list(np.load(atlas_files[1], mmap_mode="r").shape) == list(shape),
        "rows": len(layer_rows) == 11988 and len(onset_rows) == 324 and len(panel_rows) == 1998,
        "counts": int(np.sum(counts)) == 207 * 2,
        "upstream_zero": upstream_zero,
        "source_hashes": all(source_checks.values()),
    }
    if not all(output_checks.values()):
        raise RuntimeError({key: value for key, value in output_checks.items() if not value})
    metadata = {
        "phase": 1476,
        "campaign": "C082",
        "shape": list(shape),
        "axis_order": ["relation", "split", "surface", "state", "role", "coordinate"],
        "relations": relations,
        "splits": splits,
        "surfaces": surfaces,
        "roles": roles,
        "source_checks": source_checks,
        "output_checks": output_checks,
        "files": {path.name: {"size": path.stat().st_size, "sha256": core.sha(path)} for path in atlas_files},
        "model_run": False,
        "evidence_scope": protocol["evidence_scope"],
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/atlas_metadata.json", metadata)
    core.save(OUT / "analysis/final.json", {"phase": 1476, "campaign": "C082", "atlas_complete": True, "output_checks": output_checks, "authorization": "run_phase1477_c082_atlas_audit_and_synthesis"})
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
