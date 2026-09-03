#!/usr/bin/env python3
"""C185: test family-conditioned modulation above the generic local response scaffold."""
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
OUT = RESULT / "phase1719_c185_family_conditioned_routing_grammar"
C180 = RESULT / "phase1714_c180_reachable_target_choice_ecology"
C184 = RESULT / "phase1718_c184_response_ecology_invariant_discovery"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core

PHASE, CAMPAIGN = 1719, "C185"
SOURCE_ROLES = ("primary", "query", "relation")
PROFILE_TYPES = ("routing", "source", "target")


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C184 / "audit/independent_final_audit.json")
    checks = {
        "authorization": parent["all_checks_passed"] and "C185" in parent["authorization"],
        "response": (C180 / "raw/anchor_role_response.float16.npy").exists(),
        "families": len(core.load(C180 / "protocol/behavior_eligibility_lock.json")["eligible_families"]) == 7,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "family_modulation_contract_frozen",
        "model": "Qwen3-4B; C180 existing response observations",
        "profiles": {
            "routing": "normalized squared response over 64 source coordinates x 6 target roles",
            "source": "normalized squared response over 64 source coordinates",
            "target": "normalized squared response over 2560 target coordinates",
        },
        "same_family_similarity": "1 - 0.5 * L1(discovery family profile, held-out same-family profile)",
        "wrong_family_control": "median and maximum similarity to the other six discovery-family profiles",
        "descriptive_label": {"median_advantage_min": 0.01, "positive_families_min": 5},
        "claim_boundary": "family modulation of response-energy distributions, not exact signed field identity or causal circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "post-reveal threshold changes"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "analyze_all_three_profiles_on_both_holdouts",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def normalize(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1).astype(np.float64)
    return flat / max(float(flat.sum()), 1e-30)


def profiles(values: np.ndarray) -> dict[str, np.ndarray]:
    # values: source-coordinate x target-role x target-coordinate
    energy = np.square(values, dtype=np.float64)
    return {
        "routing": normalize(energy.sum(axis=2)),
        "source": normalize(energy.sum(axis=(1, 2))),
        "target": normalize(energy.sum(axis=(0, 1))),
    }


def similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.abs(left - right).sum())


def analyze() -> None:
    response = np.load(C180 / "raw/anchor_role_response.float16.npy", mmap_mode="r")
    anchors = core.rows(C180 / "raw/anchor_index.jsonl")
    families = core.load(C180 / "protocol/behavior_eligibility_lock.json")["eligible_families"]
    lookup = {(row["partition"], row["family"]): row["anchor_index"] for row in anchors}
    limits = core.load(OUT / "protocol/preregistration.json")["descriptive_label"]
    rows = []
    summary = {}
    for source_i, source_role in enumerate(SOURCE_ROLES):
        discovery = {
            family: profiles(np.asarray(response[source_i, lookup[("discovery", family)]], dtype=np.float32))
            for family in families
        }
        summary[source_role] = {}
        for partition in ("confirmation", "fresh"):
            summary[source_role][partition] = {}
            for profile_type in PROFILE_TYPES:
                profile_rows = []
                for family in families:
                    held = profiles(np.asarray(response[source_i, lookup[(partition, family)]], dtype=np.float32))[profile_type]
                    same = similarity(discovery[family][profile_type], held)
                    wrong = [similarity(discovery[other][profile_type], held) for other in families if other != family]
                    row = {
                        "source_role": source_role,
                        "partition": partition,
                        "profile_type": profile_type,
                        "family": family,
                        "same_similarity": same,
                        "median_wrong_similarity": float(np.median(wrong)),
                        "max_wrong_similarity": float(np.max(wrong)),
                        "median_wrong_advantage": same - float(np.median(wrong)),
                        "hard_wrong_advantage": same - float(np.max(wrong)),
                    }
                    rows.append(row)
                    profile_rows.append(row)
                median_advantage = float(np.median([row["median_wrong_advantage"] for row in profile_rows]))
                positive_count = int(sum(row["median_wrong_advantage"] > 0 for row in profile_rows))
                summary[source_role][partition][profile_type] = {
                    "median_same_similarity": float(np.median([row["same_similarity"] for row in profile_rows])),
                    "median_wrong_advantage": median_advantage,
                    "median_hard_wrong_advantage": float(np.median([row["hard_wrong_advantage"] for row in profile_rows])),
                    "positive_family_count": positive_count,
                    "family_modulation_label": median_advantage >= limits["median_advantage_min"] and positive_count >= limits["positive_families_min"],
                }
        summary[source_role]["both_partition_labels"] = {
            profile_type: all(summary[source_role][partition][profile_type]["family_modulation_label"] for partition in ("confirmation", "fresh"))
            for profile_type in PROFILE_TYPES
        }
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "family_conditioned_routing_grammar_adjudicated",
        "families": families,
        "summary": summary,
        "rows": rows,
        "interpretation": "positive advantage means family identity modulates an energy profile above the shared scaffold; it does not identify exact signed computation",
        "next_authorization": "run_C186_new_material_response_ecology_prediction_for_supported_profiles",
    }
    core.save(OUT / "analysis/family_routing_atlas.json", report)
    checks = {
        "rows": len(rows) == 3 * 2 * 3 * 7,
        "roles": set(summary) == set(SOURCE_ROLES),
        "finite": all(np.isfinite([value for value in row.values() if isinstance(value, float)]).all() for row in rows),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"summary": summary, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    atlas = core.load(OUT / "analysis/family_routing_atlas.json")
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
        "headline": {role: value["both_partition_labels"] for role, value in atlas["summary"].items()},
        "fresh_summary": {role: value["fresh"] for role, value in atlas["summary"].items()},
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
