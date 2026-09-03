#!/usr/bin/env python3
"""C174: discovery-frozen signed target-edge compression and held-out support transfer."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
C173 = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
OUT = RESULT / "phase1708_c174_signed_target_edge_compression"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1708, "C174"
ROLES = ("primary", "query")
ALLIANCES = ("own", "wrong_role", "relation")
PARTITIONS = ("confirmation", "fresh")
RELATIONS = ("is_a", "part_of", "located_in", "precedes")
FRACTIONS = (0.50, 0.80, 0.90, 0.95)


def now():
    return datetime.now(timezone.utc).isoformat()


def mask_for_energy(vector, fraction):
    energy = np.square(vector.astype(np.float64, copy=False))
    order = np.argsort(energy)[::-1]
    cumulative = np.cumsum(energy[order])
    count = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left") + 1) if cumulative[-1] > 0 else 0
    mask = np.zeros(len(vector), bool)
    mask[order[:count]] = True
    return mask, count


def main():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C173 / "audit/independent_final_audit.json")
    discovery = np.load(C173 / "analysis/discovery_relation_components.float16.npy", mmap_mode="r")
    validation = np.load(C173 / "analysis/validation_relation_components.float16.npy", mmap_mode="r")
    checks = {
        "authorization": parent["all_checks_passed"] and "C174" in parent["authorization"],
        "discovery": list(discovery.shape) == [2, 3, 4, 64, 6, 2560],
        "validation": list(validation.shape) == [2, 3, 2, 4, 64, 6, 2560],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "signed_target_edge_compression_frozen",
        "object": "source-coordinate x target-role x target-coordinate signed edges, separately by source role/alliance/relation",
        "energy_fractions": list(FRACTIONS),
        "selection": "smallest discovery-only edge set whose squared response reaches each requested fraction",
        "holdout_metrics": ["retained actual energy", "discovery-to-holdout signed NRMSE on locked edges", "sign agreement", "source-permuted mask advantage"],
        "descriptive_compactness": {"edge_fraction_at_80pct_max": 0.10, "fresh_retained_energy_min": 0.70, "fresh_source_mask_advantage_min": 0.05},
        "claim_boundary": "compression of an effective local response field, not identification of synapses, Attention/MLP edges, or a unique circuit",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)

    total_edges = int(np.prod(discovery.shape[-3:]))
    rows = []
    masks_80 = np.zeros(discovery.shape, bool)
    for role_i, role in enumerate(ROLES):
        for alliance_i, alliance in enumerate(ALLIANCES):
            for relation_i, relation in enumerate(RELATIONS):
                d = np.asarray(discovery[role_i, alliance_i, relation_i], np.float32).reshape(-1)
                for fraction in FRACTIONS:
                    mask, count = mask_for_energy(d, fraction)
                    if fraction == 0.80:
                        masks_80[role_i, alliance_i, relation_i] = mask.reshape(64, 6, 2560)
                    for part_i, partition in enumerate(PARTITIONS):
                        a = np.asarray(validation[role_i, alliance_i, part_i, relation_i], np.float32).reshape(-1)
                        retained = float(np.sum(np.square(a[mask], dtype=np.float64)) / max(np.sum(np.square(a, dtype=np.float64)), 1e-12))
                        source_mask = mask.reshape(64, -1)
                        permuted_mask = np.roll(source_mask, 1, axis=0).reshape(-1)
                        permuted_retained = float(np.sum(np.square(a[permuted_mask], dtype=np.float64)) / max(np.sum(np.square(a, dtype=np.float64)), 1e-12))
                        p = d[mask].astype(np.float64)
                        av = a[mask].astype(np.float64)
                        nrmse = float(np.linalg.norm(av - p) / max(np.linalg.norm(av), 1e-12))
                        sign = float(np.mean(np.sign(av) == np.sign(p))) if len(p) else 0.0
                        rows.append({
                            "source_role": role,
                            "alliance": alliance,
                            "relation": relation,
                            "energy_fraction": fraction,
                            "partition": partition,
                            "edge_count": count,
                            "edge_fraction": count / total_edges,
                            "heldout_retained_energy": retained,
                            "source_permuted_retained_energy": permuted_retained,
                            "source_mask_advantage": retained - permuted_retained,
                            "locked_edge_signed_nrmse": nrmse,
                            "locked_edge_sign_agreement": sign,
                        })
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/discovery_80pct_edge_masks.bool.npy", masks_80)
    rows80 = [r for r in rows if r["energy_fraction"] == 0.80]
    summary = {}
    for role in ROLES:
        summary[role] = {}
        for alliance in ALLIANCES:
            selected = [r for r in rows80 if r["source_role"] == role and r["alliance"] == alliance]
            summary[role][alliance] = {
                "median_edge_count": float(np.median([r["edge_count"] for r in selected])),
                "median_edge_fraction": float(np.median([r["edge_fraction"] for r in selected])),
                "confirmation_retained_energy": float(np.median([r["heldout_retained_energy"] for r in selected if r["partition"] == "confirmation"])),
                "fresh_retained_energy": float(np.median([r["heldout_retained_energy"] for r in selected if r["partition"] == "fresh"])),
                "fresh_source_mask_advantage": float(np.median([r["source_mask_advantage"] for r in selected if r["partition"] == "fresh"])),
                "fresh_sign_agreement": float(np.median([r["locked_edge_sign_agreement"] for r in selected if r["partition"] == "fresh"])),
            }
    compactness = {}
    criteria = protocol["descriptive_compactness"]
    for role in ROLES:
        own = summary[role]["own"]
        compactness[role] = all((own["median_edge_fraction"] <= criteria["edge_fraction_at_80pct_max"], own["fresh_retained_energy"] >= criteria["fresh_retained_energy_min"], own["fresh_source_mask_advantage"] >= criteria["fresh_source_mask_advantage_min"]))
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "edge_compression_adjudicated",
        "total_edges_per_relation": total_edges,
        "summary_80pct": summary,
        "compactness": compactness,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
        "next_authorization": "run_C175_pairwise_hyperedge_scan_and_C176_broad_family_reuse_regardless_of_compactness",
    }
    core.save(OUT / "analysis/edge_compression_atlas.json", report)
    final_checks = {**checks, "rows": len(rows) == 2 * 3 * 4 * 4 * 2, "finite": all(np.isfinite([v for k, v in r.items() if isinstance(v, (int, float))]).all() for r in rows), "mask_shape": list(masks_80.shape) == [2, 3, 4, 64, 6, 2560]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": {"summary_80pct": summary, "compactness": compactness}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
