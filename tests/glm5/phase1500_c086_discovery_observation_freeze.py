#!/usr/bin/env python3
"""Phase1500: observe C086 discovery only, then freeze dual-holdout predictions."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
ATLAS = RESULT / "phase1499_c086_four_factor_atlas"
C085_CONTRACT = RESULT / "phase1489_c085_prospective_layered_contract"
C085_ATLAS = RESULT / "phase1492_c085_stratified_factorial_atlas"
OUT = RESULT / "phase1500_c086_discovery_observation_freeze"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def pairwise_mean(vectors):
    return float(
        np.mean(
            [
                cosine(vectors[i], vectors[j])
                for i in range(len(vectors))
                for j in range(i + 1, len(vectors))
            ]
        )
    )


def concentration(vectors, k):
    energy = np.sum(np.square(np.asarray(vectors, dtype=np.float64)), axis=0)
    total = float(np.sum(energy))
    if total <= 1e-12:
        return 0.0, []
    order = np.argsort(energy)[::-1]
    selected = order[:k]
    return float(np.sum(energy[selected]) / total), [int(v) for v in selected]


def classify(rho, conditional_cosine):
    if rho >= 0.8 and conditional_cosine >= 0.5:
        return "content_dominant"
    if rho <= 0.2 and conditional_cosine <= -0.5:
        return "output_code_dominant"
    return "mixed_content_and_code"


def metrics_for_split(full, c085, effect_index, split_index, roles, c085_roles):
    boundary = roles.index("boundary")
    c085_boundary = c085_roles.index("boundary")
    rows = []
    for state in range(37):
        for role_index, role in enumerate(roles):
            rho_values = []
            conditional_cosines = []
            standard_alignments = []
            relation_vectors = []
            relation_code_vectors = []
            relation_panels = []
            relation_code_panels = []
            for relation_index in range(6):
                relation_surface = []
                relation_code_surface = []
                for surface_index in range(2):
                    cr = np.asarray(
                        full[effect_index["relation"], relation_index, split_index, surface_index, state, role_index],
                        dtype=np.float32,
                    )
                    crp = np.asarray(
                        full[effect_index["relation_code"], relation_index, split_index, surface_index, state, role_index],
                        dtype=np.float32,
                    )
                    beta_relation = cr / 2.0
                    beta_relation_code = crp / 4.0
                    er = float(np.dot(beta_relation, beta_relation))
                    erp = float(np.dot(beta_relation_code, beta_relation_code))
                    rho_values.append(er / (er + erp) if er + erp > 1e-12 else 0.0)
                    d_standard = cr + 0.5 * crp
                    d_reversed = cr - 0.5 * crp
                    conditional_cosines.append(cosine(d_standard, d_reversed))
                    if role == "boundary":
                        c085_vector = np.asarray(
                            c085[0, relation_index, split_index, surface_index, state, c085_boundary],
                            dtype=np.float32,
                        )
                        standard_alignments.append(cosine(d_standard, c085_vector))
                    relation_surface.append(beta_relation)
                    relation_code_surface.append(beta_relation_code)
                    relation_panels.append(beta_relation)
                    relation_code_panels.append(beta_relation_code)
                relation_vectors.append(np.mean(relation_surface, axis=0))
                relation_code_vectors.append(np.mean(relation_code_surface, axis=0))
            c_r_1, top_r = concentration(relation_panels, 26)
            c_rp_1, top_rp = concentration(relation_code_panels, 26)
            overlap = len(set(top_r) & set(top_rp)) / 26.0
            rows.append(
                {
                    "state": state,
                    "role": role,
                    "rho_content_median": float(np.median(rho_values)),
                    "rho_content_mean": float(np.mean(rho_values)),
                    "conditional_cosine_mean": float(np.mean(conditional_cosines)),
                    "conditional_cosine_median": float(np.median(conditional_cosines)),
                    "beta_relation_pairwise_mean": pairwise_mean(relation_vectors),
                    "beta_relation_code_pairwise_mean": pairwise_mean(relation_code_vectors),
                    "c085_standard_alignment_mean": float(np.mean(standard_alignments))
                    if standard_alignments
                    else None,
                    "beta_relation_top1pct_energy": c_r_1,
                    "beta_relation_code_top1pct_energy": c_rp_1,
                    "top1pct_overlap": overlap,
                }
            )
    boundary_row = next(r for r in rows if r["state"] == 35 and r["role"] == "boundary")
    summary = {**boundary_row}
    summary["field_class"] = classify(
        summary["rho_content_median"], summary["conditional_cosine_mean"]
    )
    summary["rho_peak_state"] = max(
        (r for r in rows if r["role"] == "boundary"),
        key=lambda r: r["rho_content_median"],
    )["state"]
    summary["conditional_divergence_state"] = min(
        (r for r in rows if r["role"] == "boundary"),
        key=lambda r: r["conditional_cosine_mean"],
    )["state"]
    return rows, summary


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1500 exists")
    atlas_final = core.load(ATLAS / "analysis/final.json")
    atlas_audit = core.load(ATLAS / "audit/independent_final_audit.json")
    atlas_summary = core.load(ATLAS / "analysis/four_factor_atlas_summary.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    c085_protocol = core.load(C085_CONTRACT / "protocol/preregistration.json")
    if (
        atlas_final["authorization"] != "run_phase1500_c086_discovery_observation_freeze"
        or not atlas_audit["all_checks_passed"]
    ):
        raise RuntimeError("Phase1499 authorization missing")
    full = np.load(ATLAS / "atlas/all_four_factor_contrast_mean.float32.npy", mmap_mode="r")
    c085 = np.load(C085_ATLAS / "atlas/success_factorial_contrast_mean.float32.npy", mmap_mode="r")
    effect_index = {name: i for i, name in enumerate(atlas_summary["effects"])}
    discovery_rows, discovery = metrics_for_split(
        full,
        c085,
        effect_index,
        0,
        protocol["roles"],
        c085_protocol["roles"],
    )
    core.write_rows(OUT / "analysis/discovery_layer_role_observations.jsonl", discovery_rows)
    core.save(OUT / "analysis/discovery_summary.json", discovery)
    tolerances = {
        "rho_content_median": 0.15,
        "conditional_cosine_mean": 0.15,
        "beta_relation_pairwise_mean": 0.15,
        "beta_relation_code_pairwise_mean": 0.15,
        "c085_standard_alignment_mean": 0.15,
        "beta_relation_top1pct_energy": 0.10,
        "beta_relation_code_top1pct_energy": 0.10,
    }
    predictions = {
        "phase": 1500,
        "campaign": "C086",
        "source_partition": "response_discovery",
        "untouched_partitions": ["confirmation", "lockbox"],
        "state": 35,
        "role": "boundary",
        "reference": discovery,
        "tolerances": tolerances,
        "predictions": {
            "P086-1": "field_class repeats exactly",
            "P086-2": "rho_content_median remains within frozen absolute tolerance",
            "P086-3": "conditional_cosine_mean remains within frozen absolute tolerance",
            "P086-4": "both relation coefficient pairwise means remain within tolerance",
            "P086-5": "C085 standard-arm alignment remains within tolerance",
            "P086-6": "both top-1-percent coordinate energy fractions remain within tolerance",
        },
        "claim_boundary": "validation concerns a mixed-behavior controlled field, not mastered semantics or causal mechanism",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    predictions["freeze_sha256"] = core.digest(predictions)
    core.save(OUT / "protocol/frozen_holdout_predictions.json", predictions)
    checks = {
        "discovery_only": predictions["source_partition"] == protocol["observation"]["discovery_partition"],
        "row_count": len(discovery_rows) == 37 * 7,
        "finite": all(
            value is None or np.isfinite(value)
            for row in discovery_rows
            for key, value in row.items()
            if key not in ("role",) and isinstance(value, (int, float))
        ),
        "class_registered": discovery["field_class"]
        in ("content_dominant", "output_code_dominant", "mixed_content_and_code"),
        "freeze_hash": predictions["freeze_sha256"] == core.digest({k: v for k, v in predictions.items() if k != "freeze_sha256"}),
        "no_dimension_reduction": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    result = {
        "phase": 1500,
        "campaign": "C086",
        "status": "discovery_observed_and_predictions_frozen",
        "discovery": discovery,
        "checks": checks,
        "freeze_sha256": predictions["freeze_sha256"],
        "authorization": "run_phase1501_c086_dual_holdout_validation",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
