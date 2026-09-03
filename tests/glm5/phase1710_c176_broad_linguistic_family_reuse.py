#!/usr/bin/env python3
"""C176: broad linguistic-factor reuse audit for C173 response coordinates."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1710_c176_broad_linguistic_family_reuse"
C162 = RESULT / "phase1696_c162_linguistic_program_field"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C173 = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
C175 = RESULT / "phase1709_c175_role_pair_hyperedge_response"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1710, "C176"
DIM = 2560
CHECKPOINTS = tuple(range(24, 35))
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
PARTITIONS = {"discovery": (0, 1, 2, 3), "confirmation": (4, 5), "fresh": (6, 7)}


def now():
    return datetime.now(timezone.utc).isoformat()


def signed_metrics(pred, actual):
    p = pred.astype(np.float64, copy=False)
    a = actual.astype(np.float64, copy=False)
    nrmse = float(np.linalg.norm(a - p) / max(np.linalg.norm(a), 1e-12))
    explained = float(1.0 - np.sum((a - p) ** 2) / max(np.sum(a * a), 1e-12))
    threshold = np.quantile(np.abs(p), 0.95)
    active = np.abs(p) >= threshold
    sign = float(np.mean(np.sign(a[active]) == np.sign(p[active])))
    return nrmse, explained, sign


def main():
    if OUT.exists():
        raise RuntimeError(OUT)
    audits = [core.load(path / "audit/independent_final_audit.json") for path in (C162, C173, C175)]
    fields = np.load(C162 / "analysis/unit_term_fields.float16.npy", mmap_mode="r")
    terms = core.rows(C162 / "analysis/term_index.jsonl")
    lock = core.load(C173 / "protocol/role_specific_coordinate_lock.json")
    coordinate_sets = {
        "query": lock["roles"]["query"]["coordinates"],
        "primary": lock["roles"]["primary"]["coordinates"],
        "relation": core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64],
    }
    checks = {
        "parents": all(a["all_checks_passed"] for a in audits),
        "shape": list(fields.shape) == [8, 21, 11, 6, DIM],
        "terms": len(terms) == 21,
        "coordinate_sets": all(len(v) == 64 and len(set(v)) == 64 for v in coordinate_sets.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "broad_linguistic_family_reuse_contract_frozen",
        "material": "C162 controlled English program: 6 semantic factors, 15 pair interactions, 8 lexical units, q24-q34, 6 roles",
        "coordinate_sets": {k: list(v) for k, v in coordinate_sets.items()},
        "tests": ["full-field signed discovery-to-confirmation/fresh transfer", "locked-coordinate energy enrichment", "locked-vs-rolled coordinate control", "role/checkpoint formation atlas", "cross-term active-coordinate overlap"],
        "claim_boundary": "reuse comparison between state-contrast fields and local-response source coordinates; these are distinct mathematical objects",
        "forbidden": ["attention", "MLP", "weights", "PCA", "calling C162 Chinese", "equating enrichment with causality"],
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)

    transfer_rows = []
    reuse_rows = []
    formation = []
    for term in terms:
        ti = term["term_index"]
        discovery = np.asarray(fields[list(PARTITIONS["discovery"]), ti], np.float32).mean(axis=0)
        for part in ("confirmation", "fresh"):
            for unit in PARTITIONS[part]:
                actual = np.asarray(fields[unit, ti], np.float32)
                for qi, checkpoint in enumerate(CHECKPOINTS):
                    for role_i, role in enumerate(ROLES):
                        nrmse, explained, sign = signed_metrics(discovery[qi, role_i], actual[qi, role_i])
                        transfer_rows.append({"term": term["name"], "order": term["order"], "partition": part, "unit": unit, "checkpoint": checkpoint, "role": role, "signed_nrmse": nrmse, "signed_explained_energy": explained, "active_sign_agreement": sign})
        for set_name, coordinates in coordinate_sets.items():
            role_i = ROLES.index(set_name)
            coords = np.asarray(coordinates, int)
            rolled = (coords + 371) % DIM
            for qi, checkpoint in enumerate(CHECKPOINTS):
                d = discovery[qi, role_i]
                total = float(np.sum(np.square(d, dtype=np.float64)))
                locked_energy = float(np.sum(np.square(d[coords], dtype=np.float64)) / max(total, 1e-12))
                rolled_energy = float(np.sum(np.square(d[rolled], dtype=np.float64)) / max(total, 1e-12))
                fresh_values = []
                fresh_signs = []
                for unit in PARTITIONS["fresh"]:
                    a = np.asarray(fields[unit, ti, qi, role_i], np.float32)
                    fresh_values.append(float(np.sum(np.square(a[coords], dtype=np.float64)) / max(np.sum(np.square(a, dtype=np.float64)), 1e-12)))
                    fresh_signs.append(signed_metrics(d[coords], a[coords])[2])
                reuse_rows.append({
                    "term": term["name"],
                    "order": term["order"],
                    "coordinate_set": set_name,
                    "checkpoint": checkpoint,
                    "discovery_energy_fraction": locked_energy,
                    "uniform_enrichment": locked_energy / (len(coords) / DIM),
                    "rolled_energy_fraction": rolled_energy,
                    "rolled_advantage": locked_energy - rolled_energy,
                    "fresh_energy_fraction": float(np.median(fresh_values)),
                    "fresh_active_sign_agreement": float(np.median(fresh_signs)),
                })
        fresh_rows = [r for r in transfer_rows if r["term"] == term["name"] and r["partition"] == "fresh"]
        grouped = []
        for checkpoint in CHECKPOINTS:
            for role in ROLES:
                rows = [r for r in fresh_rows if r["checkpoint"] == checkpoint and r["role"] == role]
                grouped.append({"checkpoint": checkpoint, "role": role, "median_sign": float(np.median([r["active_sign_agreement"] for r in rows])), "median_nrmse": float(np.median([r["signed_nrmse"] for r in rows]))})
        best = max(grouped, key=lambda r: (r["median_sign"], -r["median_nrmse"]))
        formation.append({"term": term["name"], "order": term["order"], "best_checkpoint": best["checkpoint"], "best_role": best["role"], "fresh_sign_agreement": best["median_sign"], "fresh_signed_nrmse": best["median_nrmse"]})

    # Active coordinate overlap across first-order terms, without reducing dimensions.
    first_terms = [t for t in terms if t["order"] == 1]
    overlap_rows = []
    for qi, checkpoint in enumerate(CHECKPOINTS):
        for role_i, role in enumerate(ROLES):
            active = {}
            for term in first_terms:
                vector = np.asarray(fields[list(PARTITIONS["discovery"]), term["term_index"], qi, role_i], np.float32).mean(axis=0)
                active[term["name"]] = set(np.argsort(np.abs(vector))[-128:].tolist())
            names = list(active)
            values = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    values.append(len(active[names[i]] & active[names[j]]) / len(active[names[i]] | active[names[j]]))
            overlap_rows.append({"checkpoint": checkpoint, "role": role, "median_first_order_jaccard": float(np.median(values)), "chance_reference": 128 / (2 * DIM - 128)})

    summary = {}
    for set_name in coordinate_sets:
        rows = [r for r in reuse_rows if r["coordinate_set"] == set_name and r["checkpoint"] == 24]
        summary[set_name] = {
            "q24_median_uniform_enrichment": float(np.median([r["uniform_enrichment"] for r in rows])),
            "q24_median_rolled_advantage": float(np.median([r["rolled_advantage"] for r in rows])),
            "q24_median_fresh_energy_fraction": float(np.median([r["fresh_energy_fraction"] for r in rows])),
            "q24_median_fresh_sign_agreement": float(np.median([r["fresh_active_sign_agreement"] for r in rows])),
            "terms_enriched_over_uniform": int(sum(r["uniform_enrichment"] > 1 for r in rows)),
        }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "broad_linguistic_family_reuse_adjudicated",
        "summary_q24": summary,
        "formation": formation,
        "overlap": overlap_rows,
        "transfer_row_count": len(transfer_rows),
        "reuse_row_count": len(reuse_rows),
        "claim_boundary": protocol["claim_boundary"],
        "next_authorization": "run_C177_natural_knowledge_ecology_and_C178_cross_model_eligibility",
    }
    core.save(OUT / "analysis/broad_family_atlas.json", report)
    core.write_rows(OUT / "analysis/transfer_rows.jsonl", transfer_rows)
    core.write_rows(OUT / "analysis/reuse_rows.jsonl", reuse_rows)
    core.write_rows(OUT / "analysis/formation_rows.jsonl", formation)
    final_checks = {**checks, "transfer_rows": len(transfer_rows) == 21 * 4 * 11 * 6, "reuse_rows": len(reuse_rows) == 21 * 3 * 11, "formation": len(formation) == 21, "finite": all(np.isfinite(list(v.values())).all() for v in summary.values())}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": {"summary_q24": summary, "formation": formation}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
