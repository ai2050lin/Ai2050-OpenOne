#!/usr/bin/env python3
"""C188: prospective generic response-scaffold validation on C186/C187 material."""
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
OUT = RESULT / "phase1722_c188_new_material_generic_scaffold_prediction"
C180 = RESULT / "phase1714_c180_reachable_target_choice_ecology"
C184 = RESULT / "phase1718_c184_response_ecology_invariant_discovery"
C186 = RESULT / "phase1720_c186_new_material_response_ecology_prediction"
C187 = RESULT / "phase1721_c187_vocabulary_paraphrase_failure_decomposition"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1722, "C188"
TARGET_ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C187 / "audit/independent_final_audit.json")
    c184_protocol = core.load(C184 / "protocol/preregistration.json")
    checks = {
        "authorization": parent["all_checks_passed"] and "C188" in parent["authorization"],
        "c184_thresholds": set(c184_protocol["observation_labels"]) == {"routing_stable", "source_rank_stable", "target_support_stable", "amplitude_compatible"},
        "existing": (C186 / "raw/new_relation_role_response.float16.npy").exists(),
        "cross": (C187 / "raw/cross_cell_relation_response.float16.npy").exists(),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "new_material_generic_scaffold_contract_frozen",
        "model": "Qwen3-4B existing C180/C186/C187 observations",
        "predictor": "C180 discovery aggregate relation-source response scaffold",
        "cells": "unit0/unit3 x canonical/paraphrase; reports_to unit3 canonical missing",
        "metrics": c184_protocol["metrics"],
        "labels": c184_protocol["observation_labels"],
        "missing_policy": "aggregate each cell over observed families and report denominator",
        "claim_boundary": "generic local propagation scaffold, not abstract relation identity or exact signed edges",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "imputation"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_four_factorial_cells",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def routing_energy(values):
    return np.mean(np.square(values, dtype=np.float64), axis=(0, 3))


def target_energy(values):
    return np.mean(np.square(values, dtype=np.float64), axis=(0, 1, 2))


def normalized_rows(values):
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)


def adaptive_support(energy, fraction):
    order = np.argsort(-energy); cumulative = np.cumsum(energy[order])
    count = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left") + 1)
    return np.sort(order[:count])


def load_cells():
    cells = {}
    existing = np.load(C186 / "raw/new_relation_role_response.float16.npy", mmap_mode="r")
    for row in core.rows(C186 / "raw/response_anchor_index.jsonl"):
        unit = 0 if row["partition"] == "new_confirmation" else 3
        cells[(row["family"], unit, row["phrase_variant"])] = np.asarray(existing[row["anchor_index"]], dtype=np.float32)
    cross = np.load(C187 / "raw/cross_cell_relation_response.float16.npy", mmap_mode="r")
    for row in core.rows(C187 / "raw/response_index.jsonl"):
        cells[(row["family"], row["unit"], row["phrase_variant"])] = np.asarray(cross[row["anchor_index"]], dtype=np.float32)
    return cells


def analyze():
    protocol = core.load(OUT / "protocol/preregistration.json")
    response = np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    anchors = core.rows(C180 / "raw/anchor_index.jsonl")
    families = core.load(C180 / "protocol/behavior_eligibility_lock.json")["eligible_families"]
    lookup = {(row["partition"], row["family"]): row["anchor_index"] for row in anchors}
    discovery = np.asarray(response[2, [lookup[("discovery", family)] for family in families]], dtype=np.float32)
    route = routing_energy(discovery)
    route_profile = normalized_rows(route)
    route_winner = np.argmax(route, axis=1)
    source_top16 = set(np.argsort(-route.sum(axis=1))[:16].tolist())
    target = target_energy(discovery)
    support = adaptive_support(target, 0.80)
    target_top256 = set(np.argsort(-target)[:256].tolist())
    discovery_amplitude = float(np.mean(np.square(discovery, dtype=np.float64)))
    cells = load_cells()
    limits = protocol["labels"]
    rows = []
    for unit in (0, 3):
        for phrase_variant in (0, 1):
            observed_families = [family for family in families if (family, unit, phrase_variant) in cells]
            values = np.stack([cells[(family, unit, phrase_variant)] for family in observed_families])
            held_route = routing_energy(values)
            held_profile = normalized_rows(held_route)
            route_similarity = 1.0 - 0.5 * np.abs(route_profile - held_profile).sum(axis=1)
            winner_consistency = float(np.mean(route_winner == np.argmax(held_route, axis=1)))
            held_source_top16 = set(np.argsort(-held_route.sum(axis=1))[:16].tolist())
            source_overlap = len(source_top16 & held_source_top16) / 16.0
            held_target = target_energy(values)
            retained = float(held_target[support].sum() / max(held_target.sum(), 1e-30))
            held_top256 = set(np.argsort(-held_target)[:256].tolist())
            target_overlap = len(target_top256 & held_top256) / 256.0
            amplitude_ratio = float(np.mean(np.square(values, dtype=np.float64)) / max(discovery_amplitude, 1e-30))
            labels = {"routing_stable": float(np.median(route_similarity)) >= limits["routing_stable"]["profile_min"] and winner_consistency >= limits["routing_stable"]["winner_min"], "source_rank_stable": source_overlap >= limits["source_rank_stable"]["top16_overlap_min"], "target_support_stable": retained >= limits["target_support_stable"]["retained_energy_min"] and target_overlap >= limits["target_support_stable"]["top256_overlap_min"], "amplitude_compatible": limits["amplitude_compatible"]["ratio_min"] <= amplitude_ratio <= limits["amplitude_compatible"]["ratio_max"]}
            rows.append({"unit": unit, "phrase_variant": phrase_variant, "observed_families": observed_families, "observed_family_count": len(observed_families), "median_route_profile_similarity": float(np.median(route_similarity)), "route_winner_consistency": winner_consistency, "source_top16_overlap": source_overlap, "discovery_target_support_count_80pct": int(len(support)), "target_support_retained_energy": retained, "target_top256_overlap": target_overlap, "amplitude_ratio": amplitude_ratio, "labels": labels})
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "new_material_generic_scaffold_adjudicated", "rows": rows, "all_cells_all_labels": all(all(row["labels"].values()) for row in rows), "claim_boundary": protocol["claim_boundary"], "next_authorization": "run_C189_campaign_synthesis_and_extended_heatmap"}
    core.save(OUT / "analysis/scaffold_prediction_atlas.json", report)
    checks = {"four_cells": len(rows) == 4, "missing_registered": sorted(row["observed_family_count"] for row in rows) == [6, 7, 7, 7], "finite": all(np.isfinite([value for value in row.values() if isinstance(value, float)]).all() for row in rows), "typed": isinstance(report["all_cells_all_labels"], bool)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"rows": rows, "all_cells_all_labels": report["all_cells_all_labels"], "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/scaffold_prediction_atlas.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"rows": report["rows"], "all_cells_all_labels": report["all_cells_all_labels"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "analyze", "close")); args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__": main()
