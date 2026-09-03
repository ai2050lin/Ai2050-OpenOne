#!/usr/bin/env python3
"""Phase1499: exact four-factor atlas for the frozen C086 full-state field."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
BEHAVIOR = RESULT / "phase1497_c086_behavior_stratification"
CAPTURE = RESULT / "phase1498_c086_all_case_field_capture"
OUT = RESULT / "phase1499_c086_four_factor_atlas"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

FACTORS = ("relation", "entity", "object", "code")
EFFECTS = tuple(
    "_".join(parts)
    for order in range(1, 5)
    for parts in combinations(FACTORS, order)
)
KEY_EFFECTS = ("relation", "code", "relation_code")
STRATA = ("all", "success", "mixed", "failed")


def signs(row):
    base = {
        "relation": 1 if row["relation_match"] else -1,
        "entity": 1 if row["entity_match"] else -1,
        "object": 1 if row["object_match"] else -1,
        "code": int(row["code_sign"]),
    }
    return {
        effect: int(np.prod([base[name] for name in effect.split("_")]))
        for effect in EFFECTS
    }


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1499 exists")
    capture_final = core.load(CAPTURE / "analysis/final.json")
    capture_audit = core.load(CAPTURE / "audit/independent_final_audit.json")
    capture_meta = core.load(CAPTURE / "analysis/capture_metadata.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if (
        capture_final["authorization"] != "run_phase1499_c086_four_factor_atlas"
        or not capture_audit["all_checks_passed"]
        or core.sha(CAPTURE / "raw/all_role_field.float16.npy")
        != capture_meta["raw_sha256"]
    ):
        raise RuntimeError("Phase1498 authorization/integrity missing")

    field = np.load(CAPTURE / "raw/all_role_field.float16.npy", mmap_mode="r")
    index = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    lookup = {r["case_id"]: r for r in index}
    groups = core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
    relations = protocol["relations"]
    splits = protocol["partitions"]
    surfaces = protocol["surfaces"]
    codebooks = tuple(protocol["codebooks"])
    cells = protocol["cells"]
    OUT.joinpath("atlas").mkdir(parents=True, exist_ok=True)
    full_path = OUT / "atlas/all_four_factor_contrast_mean.float32.npy"
    key_path = OUT / "atlas/stratum_key_effect_mean.float32.npy"
    full = np.lib.format.open_memmap(
        full_path, mode="w+", dtype=np.float32, shape=(15, 6, 3, 2, 37, 7, 2560)
    )
    full[:] = 0
    counts = np.zeros((4, 6, 3, 2), dtype=np.int32)
    finite = True
    effect_index = {effect: i for i, effect in enumerate(EFFECTS)}

    for ri, relation in enumerate(relations):
        for pi, split in enumerate(splits):
            selected = [
                group
                for group in groups
                if group["record_relation_id"] == relation and group["partition"] == split
            ]
            for ui, surface in enumerate(surfaces):
                counts[0, ri, pi, ui] = len(selected)
                for si, stratum in enumerate(STRATA[1:], start=1):
                    counts[si, ri, pi, ui] = sum(
                        group["stratum"] == stratum for group in selected
                    )
                total = np.zeros((15, 37, 7, 2560), dtype=np.float64)
                for group in selected:
                    rows = [
                        lookup[group[f"{surface}_{codebook}_{cell}"]]
                        for codebook in codebooks
                        for cell in cells
                    ]
                    block = np.asarray(
                        field[[row["row_index"] for row in rows]], dtype=np.float32
                    )
                    weights = np.asarray(
                        [
                            [
                                signs(row)[effect]
                                * (2 ** len(effect.split("_")))
                                / 16.0
                                for row in rows
                            ]
                            for effect in EFFECTS
                        ],
                        dtype=np.float32,
                    )
                    contrast = np.tensordot(weights, block, axes=(1, 0))
                    finite = finite and bool(np.isfinite(contrast).all())
                    total += contrast
                full[:, ri, pi, ui] = (total / len(selected)).astype(np.float32)
    full.flush()
    del full

    full_read = np.load(full_path, mmap_mode="r")
    key = np.lib.format.open_memmap(
        key_path, mode="w+", dtype=np.float32, shape=(4, 3, 6, 3, 2, 37, 7, 2560)
    )
    key[:] = 0
    for ki, effect in enumerate(KEY_EFFECTS):
        source = full_read[effect_index[effect]]
        key[0, ki] = source
        # C086 happened to contain only mixed sets; the typed missing strata stay zero.
        key[2, ki] = source
    key.flush()
    del key
    np.save(OUT / "atlas/stratum_sample_counts.int32.npy", counts)

    key_read = np.load(key_path, mmap_mode="r")
    checks = {
        "finite": finite and bool(np.isfinite(full_read).all()) and bool(np.isfinite(key_read).all()),
        "full_shape": list(full_read.shape) == [15, 6, 3, 2, 37, 7, 2560],
        "key_shape": list(key_read.shape) == [4, 3, 6, 3, 2, 37, 7, 2560],
        "all_panel_counts": bool(np.all(counts[0] == 12)),
        "mixed_only": bool(np.all(counts[2] == counts[0]))
        and bool(np.all(counts[1] == 0))
        and bool(np.all(counts[3] == 0)),
        "key_identity": all(
            float(
                np.max(
                    np.abs(
                        np.asarray(key_read[0, ki])
                        - np.asarray(full_read[effect_index[effect]])
                    )
                )
            )
            == 0.0
            for ki, effect in enumerate(KEY_EFFECTS)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary = {
        "phase": 1499,
        "campaign": "C086",
        "factors": FACTORS,
        "effects": EFFECTS,
        "key_effects": KEY_EFFECTS,
        "strata": STRATA,
        "axis_orders": {
            "full": ["effect", "relation", "split", "surface", "state", "role", "coordinate"],
            "stratum_key": ["stratum", "effect", "relation", "split", "surface", "state", "role", "coordinate"],
        },
        "contrast_formula": "C_S = 2^|S| / 16 * sum_x product_{j in S}(x_j) H(x)",
        "conditional_relation_formula": "D_standard=C_R+0.5*C_RP; D_reversed=C_R-0.5*C_RP",
        "counts": counts.tolist(),
        "checks": checks,
        "missingness": {
            "success": "M2: no 32/32 composition set observed; no field imputed",
            "failed": "M2: no 0/32 composition set observed; no field imputed",
        },
        "files": {
            "full": {"bytes": full_path.stat().st_size, "sha256": core.sha(full_path)},
            "key": {"bytes": key_path.stat().st_size, "sha256": core.sha(key_path)},
            "counts": {"sha256": core.sha(OUT / "atlas/stratum_sample_counts.int32.npy")},
        },
        "interpretation_boundary": "the exact atlas describes the mixed-behavior controlled field; it is not a successful-behavior mechanism atlas",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/four_factor_atlas_summary.json", summary)
    core.save(
        OUT / "analysis/final.json",
        {
            "phase": 1499,
            "campaign": "C086",
            "status": "four_factor_atlas_complete",
            "authorization": "run_phase1500_c086_discovery_observation_freeze",
        },
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "counts"}, indent=2))


if __name__ == "__main__":
    main()
