#!/usr/bin/env python3
"""Independent C123 contract/discovery audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1657_c123_role_transition_atlas_discovery"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator <= 1e-12 else float(np.dot(left, right) / denominator)


def main() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    nomination = core.load(OUT / "protocol/frozen_discovery_nomination.json")
    summary = core.load(OUT / "analysis/discovery_summary.json")
    amendment = core.load(OUT / "protocol/phase1657_execution_amendment.json")
    candidates = core.rows(OUT / "analysis/discovery_candidate_table.jsonl")
    common = core.rows(OUT / "analysis/discovery_common_candidate_table.jsonl")
    trajectories = np.load(OUT / "analysis/discovery_selected_role_trajectories.float32.npy", mmap_mode="r")
    increments = np.load(OUT / "analysis/discovery_selected_role_increments.float32.npy", mmap_mode="r")
    winners = []
    for family in protocol["families"]:
        rows = [row for row in candidates if row["family"] == family]
        rows.sort(key=lambda row: (-row["score"], -row["split_half_cosine"], row["to_state"], row["role_index"]))
        winners.append(rows[0])
    common.sort(key=lambda row: (-row["minimum_family_score_fraction"], -row["mean_family_score_fraction"], row["to_state"], row["role_index"]))
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "internal": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"],
        "execution_amendment": amendment["original_producer_sha256"] == protocol["producer_sha256"] and amendment["repaired_producer_sha256"] == core.sha(TESTS / "phase1657_c123_role_transition_atlas.py") and set(amendment["unchanged"]) >= {"research object", "source hashes", "discovery partitions", "candidate score", "C124 gates"},
        "source_hashes": all(core.sha(Path(protocol["source_paths"][name])) == digest for name, digest in protocol["source_hashes"].items()),
        "counts": len(candidates) == 1008 and len(common) == 252,
        "winners": all((left["family"], left["role"], left["to_state"]) == (right["family"], right["role"], right["to_state"]) for left, right in zip(winners, nomination["family_nominations"], strict=True)),
        "common": (common[0]["role"], common[0]["to_state"]) == (nomination["common_nomination"]["role"], nomination["common_nomination"]["to_state"]),
        "shapes": list(trajectories.shape) == [4, 37, 2560] and list(increments.shape) == [4, 36, 2560],
        "hashes": core.sha(OUT / "analysis/discovery_selected_role_trajectories.float32.npy") == nomination["trajectory_sha256"] and core.sha(OUT / "analysis/discovery_selected_role_increments.float32.npy") == nomination["increment_sha256"],
        "telescoping": all(np.allclose(np.asarray(trajectories[index, -1] - trajectories[index, 0]), np.sum(np.asarray(increments[index]), axis=0), rtol=1e-5, atol=1e-5) for index in range(4)),
        "summary": summary["authorization"] == "freeze_c124_validation_without_reselection",
    }
    report = {
        "phase": 1657,
        "campaign": "C123",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "all_checks_passed": all(checks.values()),
        "authorization": "execute_c124_validation" if all(checks.values()) else "stop",
    }
    core.save(OUT / "audit/independent_discovery_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
