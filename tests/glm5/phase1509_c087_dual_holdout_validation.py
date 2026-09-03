#!/usr/bin/env python3
"""Phase1509: one-time C087 confirmation and lockbox reveal."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
ATLAS = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
FREEZE = RESULT / "phase1508_c087_discovery_observation_freeze"
C086_ATLAS = RESULT / "phase1499_c086_four_factor_atlas"
C086_CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
OUT = RESULT / "phase1509_c087_dual_holdout_validation"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1508_c087_discovery_observation_freeze as discovery


def reveal_summary(observations):
    boundary = next(row for row in observations if row["state"] == 35 and row["role"] == "boundary")
    source = next(row for row in observations if row["state"] == 35 and row["role"] == "source_relation")
    onset = min(
        row["state"] for row in observations
        if row["role"] == "boundary"
        and row["shared_energy_fraction"] >= 0.4
        and row["within_group_surface_cosine_mean"] >= 0.4
    )
    field_class = (
        "shared_late_boundary_response"
        if boundary["surface_centroid_cosine"] >= 0.8
        and boundary["within_group_surface_cosine_mean"] >= 0.6
        and boundary["shared_energy_fraction"] >= 0.4
        and boundary["group_pairwise_cosine_mean"] >= 0.4
        else "heterogeneous_or_surface_specific_response"
    )
    return {"field_class": field_class, "onset_state": onset, "boundary": boundary, "source_relation_state35": source}


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1509 exists")
    parent = core.load(FREEZE / "analysis/final.json")
    parent_audit = core.load(FREEZE / "audit/independent_final_audit.json")
    frozen = core.load(FREEZE / "protocol/frozen_holdout_predictions.json")
    if parent["authorization"] != "run_phase1509_c087_dual_holdout_validation" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1508 authorization missing")
    if frozen["freeze_sha256"] != core.digest({key: value for key, value in frozen.items() if key != "freeze_sha256"}):
        raise RuntimeError("frozen prediction mutation")
    group = np.load(ATLAS / "atlas/group_semantic_contrast.float32.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/group_semantic_contrast_index.jsonl")
    c086 = np.load(C086_ATLAS / "atlas/all_four_factor_contrast_mean.float32.npy", mmap_mode="r")
    c086_summary = core.load(C086_ATLAS / "analysis/four_factor_atlas_summary.json")
    c086_protocol = core.load(C086_CONTRACT / "protocol/preregistration.json")
    results = {}
    primary_names = (
        "surface_centroid_cosine",
        "within_group_surface_cosine_mean",
        "shared_energy_fraction",
        "group_pairwise_cosine_mean",
    )
    reference = frozen["reference"]
    for partition in frozen["untouched_partitions"]:
        observations = discovery.metrics(
            group,
            group_index,
            partition,
            c086,
            c086_summary["effects"].index("relation"),
            c086_protocol["roles"].index("boundary"),
        )
        summary = reveal_summary(observations)
        checks = {
            "P087-1_field_class": summary["field_class"] == reference["field_class"],
            **{
                f"P087-2_{name}": abs(summary["boundary"][name] - reference["boundary"][name]) <= frozen["primary_tolerances"][name]
                for name in primary_names
            },
            "P087-3_onset": abs(summary["onset_state"] - reference["onset_state"]) <= frozen["primary_tolerances"]["onset_state"],
            "P087-4_source_zero": summary["source_relation_state35"]["centroid_norm"] == 0.0,
        }
        diagnostics = {
            name: {
                "reference": reference["boundary"][name],
                "observed": summary["boundary"][name],
                "absolute_difference": abs(summary["boundary"][name] - reference["boundary"][name]),
                "within_diagnostic_tolerance": abs(summary["boundary"][name] - reference["boundary"][name]) <= tolerance,
            }
            for name, tolerance in frozen["diagnostic_tolerances"].items()
        }
        summary["primary_checks"] = checks
        summary["primary_pass"] = all(checks.values())
        summary["diagnostics"] = diagnostics
        results[partition] = summary
        core.write_rows(OUT / f"analysis/{partition}_layer_role_observations.jsonl", observations)
    dual_holdout_pass = all(row["primary_pass"] for row in results.values())
    result = {
        "phase": 1509,
        "campaign": "C087",
        "status": "dual_holdout_revealed",
        "freeze_sha256": frozen["freeze_sha256"],
        "holdouts": results,
        "dual_holdout_primary_pass": dual_holdout_pass,
        "evidence_scope": "prospective descriptive replication; Phase1506 execution identity failure prevents confirmatory mechanism status",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "run_phase1510_c087_stratum_and_c086_diagnostics",
    }
    core.save(OUT / "analysis/dual_holdout_validation.json", result)
    core.save(OUT / "analysis/final.json", {key: value for key, value in result.items() if key != "holdouts"})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
