#!/usr/bin/env python3
"""C177: missing-aware repair of the broad linguistic-family reuse atlas."""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1711_c177_missing_aware_broad_family_atlas"
C162 = RESULT / "phase1696_c162_linguistic_program_field"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C173 = RESULT / "phase1707_c173_role_specific_full_coordinate_response"
C176 = RESULT / "phase1710_c176_broad_linguistic_family_reuse"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1711, "C177"
DIM, TAU = 2560, 1e-8
CHECKPOINTS = tuple(range(24, 35))
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
HOLDOUTS = {"confirmation": (4, 5), "fresh": (6, 7)}


def now():
    return datetime.now(timezone.utc).isoformat()


def valid_metrics(pred, actual):
    p = pred.astype(np.float64, copy=False)
    a = actual.astype(np.float64, copy=False)
    pn, an = np.linalg.norm(p), np.linalg.norm(a)
    if pn <= TAU or an <= TAU:
        return None
    threshold = np.quantile(np.abs(p), 0.95)
    active = np.abs(p) >= threshold
    return {
        "signed_nrmse": float(np.linalg.norm(a - p) / an),
        "signed_explained_energy": float(1 - np.sum((a - p) ** 2) / np.sum(a * a)),
        "active_sign_agreement": float(np.mean(np.sign(a[active]) == np.sign(p[active]))),
    }


def main():
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = [core.load(C162 / "audit/independent_final_audit.json"), core.load(C173 / "audit/independent_final_audit.json"), core.load(C176 / "audit/independent_final_audit.json")]
    fields = np.load(C162 / "analysis/unit_term_fields.float16.npy", mmap_mode="r")
    terms = core.rows(C162 / "analysis/term_index.jsonl")
    lock = core.load(C173 / "protocol/role_specific_coordinate_lock.json")
    coordinate_sets = {
        "primary": lock["roles"]["primary"]["coordinates"],
        "query": lock["roles"]["query"]["coordinates"],
        "relation": core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64],
    }
    checks = {"parents": all(a["all_checks_passed"] for a in parents), "c176_invalid": not core.load(C176 / "analysis/final.json")["scientific_result_valid"], "shape": list(fields.shape) == [8, 21, 11, 6, DIM], "terms": len(terms) == 21}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "missing_aware_broad_family_contract_frozen",
        "missing_rule": f"metric undefined if discovery or holdout L2 norm <= {TAU}",
        "fixed_material": "C162 fields unchanged",
        "fixed_coordinates": coordinate_sets,
        "primary_outputs": ["support map", "valid-only signed transfer", "valid-only coordinate enrichment", "formation among supported cells"],
        "claim_boundary": "state-contrast reuse atlas; not local-response causality or natural language closure",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)

    rows = []
    support = []
    enrichment = []
    formation = []
    for term in terms:
        ti = term["term_index"]
        discovery = np.asarray(fields[:4, ti], np.float32).mean(axis=0)
        candidates = []
        for qi, checkpoint in enumerate(CHECKPOINTS):
            for role_i, role in enumerate(ROLES):
                discovery_norm = float(np.linalg.norm(discovery[qi, role_i]))
                valid_count = 0
                values = []
                for partition, units in HOLDOUTS.items():
                    for unit in units:
                        metric = valid_metrics(discovery[qi, role_i], np.asarray(fields[unit, ti, qi, role_i], np.float32))
                        row = {"term": term["name"], "order": term["order"], "checkpoint": checkpoint, "role": role, "partition": partition, "unit": unit, "supported": metric is not None}
                        if metric is not None:
                            row.update(metric); values.append(metric); valid_count += 1
                        rows.append(row)
                support.append({"term": term["name"], "order": term["order"], "checkpoint": checkpoint, "role": role, "discovery_norm": discovery_norm, "valid_holdout_cells": valid_count, "possible_holdout_cells": 4})
                fresh = [r for r in rows if r["term"] == term["name"] and r["checkpoint"] == checkpoint and r["role"] == role and r["partition"] == "fresh" and r["supported"]]
                if fresh:
                    candidates.append({"checkpoint": checkpoint, "role": role, "fresh_support": len(fresh), "fresh_sign": float(np.median([r["active_sign_agreement"] for r in fresh])), "fresh_nrmse": float(np.median([r["signed_nrmse"] for r in fresh]))})
        if candidates:
            best = max(candidates, key=lambda x: (x["fresh_sign"], -x["fresh_nrmse"], x["fresh_support"]))
            formation.append({"term": term["name"], "order": term["order"], "status": "supported", **best})
        else:
            formation.append({"term": term["name"], "order": term["order"], "status": "missing"})

        for set_name, coordinates in coordinate_sets.items():
            role_i = ROLES.index(set_name)
            coords = np.asarray(coordinates, int)
            rolled = (coords + 371) % DIM
            for qi, checkpoint in enumerate(CHECKPOINTS):
                d = discovery[qi, role_i]
                total = float(np.sum(np.square(d, dtype=np.float64)))
                if total <= TAU:
                    enrichment.append({"term": term["name"], "order": term["order"], "coordinate_set": set_name, "checkpoint": checkpoint, "supported": False})
                    continue
                locked = float(np.sum(np.square(d[coords], dtype=np.float64)) / total)
                rolled_energy = float(np.sum(np.square(d[rolled], dtype=np.float64)) / total)
                fresh = []
                signs = []
                for unit in HOLDOUTS["fresh"]:
                    actual = np.asarray(fields[unit, ti, qi, role_i], np.float32)
                    if np.linalg.norm(actual) > TAU:
                        fresh.append(float(np.sum(np.square(actual[coords], dtype=np.float64)) / np.sum(np.square(actual, dtype=np.float64))))
                        coordinate_metric = valid_metrics(d[coords], actual[coords])
                        signs.append(coordinate_metric["active_sign_agreement"] if coordinate_metric is not None else np.nan)
                enrichment.append({"term": term["name"], "order": term["order"], "coordinate_set": set_name, "checkpoint": checkpoint, "supported": bool(fresh), "discovery_energy_fraction": locked, "uniform_enrichment": locked / (64 / DIM), "rolled_advantage": locked - rolled_energy, "fresh_energy_fraction": float(np.median(fresh)) if fresh else None, "fresh_sign_agreement": float(np.nanmedian(signs)) if signs and not np.all(np.isnan(signs)) else None})

    role_support = {}
    for role in ROLES:
        cells = [r for r in support if r["role"] == role]
        role_support[role] = {"supported_fraction": float(np.mean([r["valid_holdout_cells"] > 0 for r in cells])), "fully_supported_fraction": float(np.mean([r["valid_holdout_cells"] == 4 for r in cells]))}
    reuse_summary = {}
    for set_name in coordinate_sets:
        cells = [r for r in enrichment if r["coordinate_set"] == set_name and r["checkpoint"] == 24 and r["supported"]]
        reuse_summary[set_name] = {
            "supported_terms": len(cells),
            "median_uniform_enrichment": float(np.median([r["uniform_enrichment"] for r in cells])) if cells else None,
            "median_rolled_advantage": float(np.median([r["rolled_advantage"] for r in cells])) if cells else None,
            "median_fresh_energy_fraction": float(np.median([r["fresh_energy_fraction"] for r in cells])) if cells else None,
            "median_fresh_sign_agreement": float(np.median([r["fresh_sign_agreement"] for r in cells if r["fresh_sign_agreement"] is not None])) if cells and any(r["fresh_sign_agreement"] is not None for r in cells) else None,
        }
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "missing_aware_broad_family_adjudicated", "role_support": role_support, "reuse_q24": reuse_summary, "formation": formation, "valid_transfer_rows": sum(r["supported"] for r in rows), "missing_transfer_rows": sum(not r["supported"] for r in rows), "claim_boundary": protocol["claim_boundary"], "next_authorization": "run_C178_natural_knowledge_ecology_and_C179_cross_model_eligibility"}
    core.save(OUT / "analysis/missing_aware_atlas.json", report)
    core.write_rows(OUT / "analysis/transfer_rows.jsonl", rows)
    core.write_rows(OUT / "analysis/support_rows.jsonl", support)
    core.write_rows(OUT / "analysis/enrichment_rows.jsonl", enrichment)
    final_checks = {**checks, "rows": len(rows) == 21 * 11 * 6 * 4, "support": len(support) == 21 * 11 * 6, "enrichment": len(enrichment) == 21 * 3 * 11, "accounting": report["valid_transfer_rows"] + report["missing_transfer_rows"] == len(rows)}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": {"role_support": role_support, "reuse_q24": reuse_summary, "supported_formation_terms": sum(r["status"] == "supported" for r in formation)}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/internal_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values())})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
