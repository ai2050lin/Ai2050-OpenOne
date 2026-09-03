#!/usr/bin/env python3
"""C184: discover simple response-ecology invariants in existing C180 full-coordinate data."""
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
OUT = RESULT / "phase1718_c184_response_ecology_invariant_discovery"
C180 = RESULT / "phase1714_c180_reachable_target_choice_ecology"
C183 = RESULT / "phase1717_c183_response_ecology_synthesis_heatmap"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1718, "C184"
SOURCE_ROLES = ("primary", "query", "relation")
TARGET_ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
PARTITIONS = ("confirmation", "fresh")


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C183 / "audit/independent_final_audit.json")
    response = np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    checks = {
        "authorization": parent["all_checks_passed"] and "C184" in parent["authorization"],
        "shape": list(response.shape)[0:1] == [3] and list(response.shape)[2:] == [64, 6, 2560],
        "partitions": {row["partition"] for row in core.rows(C180 / "raw/anchor_index.jsonl")} == {"discovery", "confirmation", "fresh"},
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "existing_response_invariant_contract_frozen",
        "model": "Qwen3-4B; C180 audited observations; no new model run",
        "objects": {
            "role_routing": "per-source-coordinate fraction of squared response sent to six target roles",
            "source_ranking": "total squared response rank among 64 frozen source coordinates",
            "target_support": "target-coordinate set needed for 80% aggregate discovery energy",
        },
        "metrics": {
            "route_profile_similarity": "1 - 0.5 * L1(normalized routing profiles)",
            "route_winner_consistency": "fraction of source coordinates with same maximum-energy target role",
            "source_top16_overlap": "intersection size divided by 16",
            "target_support_retained_energy": "held-out energy on frozen discovery support divided by total held-out energy",
            "target_top256_overlap": "intersection size divided by 256",
            "amplitude_ratio": "held-out total squared response divided by discovery total squared response",
        },
        "observation_labels": {
            "routing_stable": {"profile_min": 0.85, "winner_min": 0.65},
            "source_rank_stable": {"top16_overlap_min": 0.50},
            "target_support_stable": {"retained_energy_min": 0.60, "top256_overlap_min": 0.40},
            "amplitude_compatible": {"ratio_min": 0.50, "ratio_max": 2.00},
        },
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine as primary metric", "claiming unique causal edges"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_discovery_then_confirmation_and_fresh",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def routing_energy(values: np.ndarray) -> np.ndarray:
    # values: family x source-coordinate x target-role x target-coordinate
    return np.mean(np.square(values, dtype=np.float64), axis=(0, 3))


def aggregate_target_energy(values: np.ndarray) -> np.ndarray:
    return np.mean(np.square(values, dtype=np.float64), axis=(0, 1, 2))


def normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-30)


def adaptive_support(energy: np.ndarray, fraction: float) -> np.ndarray:
    order = np.argsort(-energy)
    cumulative = np.cumsum(energy[order])
    count = int(np.searchsorted(cumulative, fraction * cumulative[-1], side="left") + 1)
    return np.sort(order[:count])


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    response = np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    anchors = core.rows(C180 / "raw/anchor_index.jsonl")
    families = core.load(C180 / "protocol/behavior_eligibility_lock.json")["eligible_families"]
    lookup = {(row["partition"], row["family"]): row["anchor_index"] for row in anchors}
    rows = []
    role_summaries = {}
    for source_i, source_role in enumerate(SOURCE_ROLES):
        arrays = {
            partition: np.asarray(
                response[source_i, [lookup[(partition, family)] for family in families]],
                dtype=np.float32,
            )
            for partition in ("discovery",) + PARTITIONS
        }
        discovery_route = routing_energy(arrays["discovery"])
        discovery_profile = normalized_rows(discovery_route)
        discovery_winner = np.argmax(discovery_route, axis=1)
        discovery_source_energy = discovery_route.sum(axis=1)
        discovery_source_top16 = set(np.argsort(-discovery_source_energy)[:16].tolist())
        discovery_target_energy = aggregate_target_energy(arrays["discovery"])
        discovery_target_support = adaptive_support(discovery_target_energy, 0.80)
        discovery_target_top256 = set(np.argsort(-discovery_target_energy)[:256].tolist())
        source_rows = []
        for partition in PARTITIONS:
            held_route = routing_energy(arrays[partition])
            held_profile = normalized_rows(held_route)
            profile_similarity = 1.0 - 0.5 * np.abs(discovery_profile - held_profile).sum(axis=1)
            winner_consistency = float(np.mean(discovery_winner == np.argmax(held_route, axis=1)))
            held_source_top16 = set(np.argsort(-held_route.sum(axis=1))[:16].tolist())
            source_top16_overlap = len(discovery_source_top16 & held_source_top16) / 16.0
            held_target_energy = aggregate_target_energy(arrays[partition])
            retained = float(held_target_energy[discovery_target_support].sum() / max(held_target_energy.sum(), 1e-30))
            held_target_top256 = set(np.argsort(-held_target_energy)[:256].tolist())
            target_top256_overlap = len(discovery_target_top256 & held_target_top256) / 256.0
            amplitude_ratio = float(np.square(arrays[partition], dtype=np.float64).sum() / max(np.square(arrays["discovery"], dtype=np.float64).sum(), 1e-30))
            limits = protocol["observation_labels"]
            labels = {
                "routing_stable": float(np.median(profile_similarity)) >= limits["routing_stable"]["profile_min"] and winner_consistency >= limits["routing_stable"]["winner_min"],
                "source_rank_stable": source_top16_overlap >= limits["source_rank_stable"]["top16_overlap_min"],
                "target_support_stable": retained >= limits["target_support_stable"]["retained_energy_min"] and target_top256_overlap >= limits["target_support_stable"]["top256_overlap_min"],
                "amplitude_compatible": limits["amplitude_compatible"]["ratio_min"] <= amplitude_ratio <= limits["amplitude_compatible"]["ratio_max"],
            }
            row = {
                "source_role": source_role,
                "partition": partition,
                "median_route_profile_similarity": float(np.median(profile_similarity)),
                "route_winner_consistency": winner_consistency,
                "source_top16_overlap": source_top16_overlap,
                "discovery_target_support_count_80pct": int(len(discovery_target_support)),
                "target_support_retained_energy": retained,
                "target_top256_overlap": target_top256_overlap,
                "amplitude_ratio": amplitude_ratio,
                "labels": labels,
            }
            rows.append(row)
            source_rows.append(row)
        role_summaries[source_role] = {
            "discovery_target_support_count_80pct": int(len(discovery_target_support)),
            "discovery_target_support_fraction": float(len(discovery_target_support) / 2560),
            "confirmation": source_rows[0],
            "fresh": source_rows[1],
            "both_partition_labels": {key: all(row["labels"][key] for row in source_rows) for key in source_rows[0]["labels"]},
            "discovery_routing_profile": {
                str(i): {TARGET_ROLES[j]: float(discovery_profile[i, j]) for j in range(len(TARGET_ROLES))}
                for i in range(64)
            },
            "discovery_target_support": discovery_target_support.astype(int).tolist(),
        }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "response_ecology_invariants_adjudicated",
        "families": families,
        "rows": rows,
        "role_summaries": role_summaries,
        "interpretation": "routing and ranking may remain stable even when exact fixed target edges do not; each label is descriptive, not a causal closure gate",
        "next_authorization": "run_C185_family_by_role_routing_grammar_and_C186_new_material_prediction",
    }
    core.save(OUT / "analysis/invariant_atlas.json", report)
    checks = {
        "rows": len(rows) == 6,
        "roles": set(role_summaries) == set(SOURCE_ROLES),
        "finite": all(np.isfinite([v for v in row.values() if isinstance(v, float)]).all() for row in rows),
        "support_nonempty": all(summary["discovery_target_support_count_80pct"] > 0 for summary in role_summaries.values()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"rows": rows, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    atlas = core.load(OUT / "analysis/invariant_atlas.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "closed",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "headline": {
            role: {
                "both_partition_labels": summary["both_partition_labels"],
                "support_count_80pct": summary["discovery_target_support_count_80pct"],
                "fresh": {key: value for key, value in summary["fresh"].items() if key not in ("source_role", "partition")},
            }
            for role, summary in atlas["role_summaries"].items()
        },
        "next_authorization": atlas["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
