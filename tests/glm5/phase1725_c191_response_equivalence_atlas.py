#!/usr/bin/env python3
"""C191: missing-aware response-equivalence atlas over all C190 factorial cells."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1725_c191_response_equivalence_atlas"
C190 = RESULT / "phase1724_c190_relation_phrase_wrapper_factorial"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c191_response_equivalence_atlas.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1724_c190_relation_phrase_wrapper_factorial as c190

PHASE, CAMPAIGN = 1725, "C191"


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C190 / "audit/independent_final_audit.json")
    checks = {
        "authorization": parent["all_checks_passed"] and "C191" in parent["authorization"],
        "parent_closed": core.load(C190 / "analysis/final.json")["status"] == "closed",
        "raw_immutable": (C190 / "raw/off_diagonal_relation_response.float16.npy").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "response_equivalence_atlas_frozen",
        "data": "52 observed cells and four registered behavior-missing cells from C186-C190",
        "object": "normalized q25 target-coordinate energy profile of the relation-source q24 local response",
        "comparison": "full 2560-coordinate total-variation similarity; leave-one-cell-out nearest neighbor",
        "registered_labels": ["family", "unit", "phrase_variant", "wrapper_variant"],
        "summary": "nearest-neighbor observed match rate minus exact available-peer baseline for each label",
        "interpretation_gate": {"dominant_advantage_min": 0.25, "lead_over_next_min": 0.10},
        "missing_policy": "no imputation; unavailable cells absent from neighbor candidates and explicitly carried in metadata",
        "claim_boundary": "descriptive response-equivalence organization in one model/task; not a semantic ontology or causal circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "clustering hyperparameter search", "editing parent arrays"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "build_atlas_and_parameter_coordinate_heatmap",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def profile(values):
    energy = np.square(values, dtype=np.float64).sum(axis=(0, 1))
    return (energy / max(energy.sum(), 1e-30)).astype(np.float32)


def similarity(left, right):
    return float(1.0 - 0.5 * np.abs(left.astype(np.float64) - right.astype(np.float64)).sum())


def build():
    cells = c190.load_cells()
    keys = sorted(cells)
    profiles = np.stack([profile(cells[key]) for key in keys])
    count = len(keys)
    matrix = np.eye(count, dtype=np.float32)
    pairs = []
    for left in range(count):
        for right in range(left + 1, count):
            score = similarity(profiles[left], profiles[right])
            matrix[left, right] = matrix[right, left] = score
            lk, rk = keys[left], keys[right]
            pairs.append({
                "left": left,
                "right": right,
                "similarity": score,
                "same_family": lk[0] == rk[0],
                "same_unit": lk[1] == rk[1],
                "same_phrase_variant": lk[2] == rk[2],
                "same_wrapper_variant": lk[3] == rk[3],
            })
    labels = {
        "family": [key[0] for key in keys],
        "unit": [key[1] for key in keys],
        "phrase_variant": [key[2] for key in keys],
        "wrapper_variant": [key[3] for key in keys],
    }
    nearest = []
    summaries = {}
    for index, key in enumerate(keys):
        candidates = matrix[index].copy(); candidates[index] = -np.inf
        neighbor = int(np.argmax(candidates))
        nearest.append({
            "cell_index": index,
            "neighbor_index": neighbor,
            "similarity": float(matrix[index, neighbor]),
            **{f"same_{name}": labels[name][index] == labels[name][neighbor] for name in labels},
        })
    for name, values in labels.items():
        matches = [row[f"same_{name}"] for row in nearest]
        baselines = []
        for index, value in enumerate(values):
            baselines.append(sum(other == value for j, other in enumerate(values) if j != index) / (count - 1))
        observed = float(np.mean(matches)); baseline = float(np.mean(baselines))
        summaries[name] = {"nearest_match_rate": observed, "available_peer_baseline": baseline, "advantage": observed - baseline, "match_count": int(sum(matches)), "support": count}
    ordered = sorted(summaries, key=lambda name: summaries[name]["advantage"], reverse=True)
    gate = core.load(OUT / "protocol/preregistration.json")["interpretation_gate"]
    lead = summaries[ordered[0]]["advantage"] - summaries[ordered[1]]["advantage"]
    dominant = ordered[0] if summaries[ordered[0]]["advantage"] >= gate["dominant_advantage_min"] and lead >= gate["lead_over_next_min"] else None
    group_similarity = {}
    for label in ("family", "unit", "phrase_variant", "wrapper_variant"):
        same = [row["similarity"] for row in pairs if row[f"same_{label}"]]
        different = [row["similarity"] for row in pairs if not row[f"same_{label}"]]
        group_similarity[label] = {"median_same": float(np.median(same)), "median_different": float(np.median(different)), "difference": float(np.median(same) - np.median(different)), "same_pairs": len(same), "different_pairs": len(different)}
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "response_equivalence_atlas_complete",
        "observed_cells": count,
        "registered_missing": core.load(C190 / "analysis/factorial_response_atlas.json")["registered_missing"],
        "nearest_neighbor_summary": summaries,
        "nearest_neighbor_rows": nearest,
        "pair_group_similarity": group_similarity,
        "dominant_registered_label": dominant,
        "dominance_lead": lead,
        "interpretation": "No dominant label means response geometry is jointly conditioned or organized by an unregistered factor; it does not mean the field is random or that relation identity is absent.",
        "next_authorization": "freeze_C192_multi_pattern_role_conditioned_equivalence_campaign_before_new_model_runs",
    }
    core.save(OUT / "analysis/response_equivalence_atlas.json", report)
    core.write_rows(OUT / "analysis/pairwise_similarity.jsonl", pairs)
    cell_rows = []
    for index, key in enumerate(keys):
        family, unit, phrase_variant, wrapper_variant = key
        cell_rows.append({"cell_index": index, "family": family, "unit": unit, "phrase_variant": phrase_variant, "wrapper_variant": wrapper_variant, "label": f"{family} / unit{unit} / phrase{phrase_variant} / wrapper{wrapper_variant}", "values": profiles[index].tolist()})
    variation = np.var(profiles, axis=0)
    payload = {
        "schema": "c191_response_equivalence_atlas.v1",
        "result_type": "response_equivalence_atlas_heatmap",
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": "Qwen3-4B",
        "title": "C191 Missing-Aware Response Equivalence Atlas",
        "dimensions": list(range(2560)),
        "default_coordinates": np.argsort(-variation)[:64].astype(int).tolist(),
        "rows": cell_rows,
        "similarity_matrix": matrix.tolist(),
        "nearest_neighbor_summary": summaries,
        "dominant_registered_label": dominant,
        "registered_missing": report["registered_missing"],
        "coordinate_semantics": "Each row is a normalized q25 target-coordinate energy profile from a relation-source q24 local response; every column is one physical Qwen3 activation coordinate.",
        "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"],
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    asset = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size, "rows": count, "schema": payload["schema"]}
    core.save(OUT / "analysis/public_asset.json", asset)
    checks = {"cells": count == 52, "missing": len(report["registered_missing"]) == 4, "pairs": len(pairs) == count * (count - 1) // 2, "nearest": len(nearest) == count, "matrix": list(matrix.shape) == [52, 52], "all_2560": profiles.shape == (52, 2560), "finite": bool(np.isfinite(matrix).all())}
    core.save(OUT / "audit/internal_build_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"summary": summaries, "dominant": dominant, "lead": lead, "asset": asset, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/response_equivalence_atlas.json"); asset = core.load(OUT / "analysis/public_asset.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "build": core.load(OUT / "audit/internal_build_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"], "asset_hash": core.sha(PUBLIC) == asset["sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {key: report[key] for key in ("observed_cells", "nearest_neighbor_summary", "pair_group_similarity", "dominant_registered_label", "dominance_lead", "registered_missing")}, "asset": asset, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "build", "close")); args = parser.parse_args(); {"contract": contract, "build": build, "close": close}[args.command]()


if __name__ == "__main__":
    main()
