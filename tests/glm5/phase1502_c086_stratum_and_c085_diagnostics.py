#!/usr/bin/env python3
"""Phase1502: diagnose C086 behavior, late field formation, and C085 pairing."""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
BEHAVIOR = RESULT / "phase1497_c086_behavior_stratification"
DISCOVERY = RESULT / "phase1500_c086_discovery_observation_freeze"
VALIDATION = RESULT / "phase1501_c086_dual_holdout_validation"
OUT = RESULT / "phase1502_c086_stratum_and_c085_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core


def mean(values):
    return float(np.mean(values)) if values else None


def persistent_onset(rows_by_split, metric, threshold):
    for state in range(1, 37):
        if all(
            all(row[metric] >= threshold for row in rows if row["state"] >= state)
            for rows in rows_by_split.values()
        ):
            return state
    return None


def margin_atlas(rows, groups, protocol):
    by = {row["case_id"]: row for row in rows}
    summaries = []
    for split in protocol["partitions"]:
        for relation in protocol["relations"]:
            selected = [
                group
                for group in groups
                if group["partition"] == split and group["record_relation_id"] == relation
            ]
            for surface in protocol["surfaces"]:
                effects = {name: [] for name in ("relation", "code", "relation_code")}
                for group in selected:
                    panel = [
                        by[group[f"{surface}_{codebook}_{cell}"]]
                        for codebook in protocol["codebooks"]
                        for cell in protocol["cells"]
                    ]
                    values = np.asarray(
                        [row["scores"][0] - row["scores"][1] for row in panel], dtype=np.float64
                    )
                    r = np.asarray([1 if row["relation_match"] else -1 for row in panel])
                    p = np.asarray([row["code_sign"] for row in panel])
                    effects["relation"].append(float(np.sum(r * values) * 2 / 16))
                    effects["code"].append(float(np.sum(p * values) * 2 / 16))
                    effects["relation_code"].append(float(np.sum(r * p * values) * 4 / 16))
                cr = mean(effects["relation"])
                cp = mean(effects["code"])
                crp = mean(effects["relation_code"])
                br, bp, brp = cr / 2.0, cp / 2.0, crp / 4.0
                summaries.append(
                    {
                        "split": split,
                        "relation": relation,
                        "surface": surface,
                        "C_R": cr,
                        "C_P": cp,
                        "C_RP": crp,
                        "beta_R": br,
                        "beta_P": bp,
                        "beta_RP": brp,
                        "rho_logit_content": br * br / (br * br + brp * brp)
                        if br * br + brp * brp > 1e-12
                        else 0.0,
                        "D_standard": cr + 0.5 * crp,
                        "D_reversed": cr - 0.5 * crp,
                    }
                )
    return summaries


def main():
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1502 exists")
    validation_final = core.load(VALIDATION / "analysis/final.json")
    validation_audit = core.load(VALIDATION / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if (
        validation_final["authorization"] != "run_phase1502_c086_stratum_and_c085_diagnostics"
        or not validation_audit["all_checks_passed"]
    ):
        raise RuntimeError("Phase1501 authorization missing")
    behavior = core.rows(BEHAVIOR / "raw/behavior.jsonl")
    groups = core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
    rows_by_split = {
        "response_discovery": [
            r
            for r in core.rows(DISCOVERY / "analysis/discovery_layer_role_observations.jsonl")
            if r["role"] == "boundary"
        ],
        "confirmation": [
            r
            for r in core.rows(VALIDATION / "analysis/confirmation_layer_role_observations.jsonl")
            if r["role"] == "boundary"
        ],
        "lockbox": [
            r
            for r in core.rows(VALIDATION / "analysis/lockbox_layer_role_observations.jsonl")
            if r["role"] == "boundary"
        ],
    }
    trajectory = []
    for state in range(37):
        state_rows = [rows[state] for rows in rows_by_split.values()]
        trajectory.append(
            {
                "state": state,
                **{
                    key: mean([row[key] for row in state_rows])
                    for key in (
                        "rho_content_median",
                        "conditional_cosine_mean",
                        "beta_relation_pairwise_mean",
                        "beta_relation_code_pairwise_mean",
                        "c085_standard_alignment_mean",
                        "beta_relation_top1pct_energy",
                        "beta_relation_code_top1pct_energy",
                        "top1pct_overlap",
                    )
                },
            }
        )
    core.write_rows(OUT / "analysis/three_split_boundary_trajectory.jsonl", trajectory)

    behavior_matrix = []
    for surface in protocol["surfaces"]:
        for codebook in protocol["codebooks"]:
            for truth in (True, False):
                selected = [
                    row
                    for row in behavior
                    if row["surface"] == surface
                    and row["codebook"] == codebook
                    and row["semantic_truth"] == truth
                ]
                behavior_matrix.append(
                    {
                        "surface": surface,
                        "codebook": codebook,
                        "semantic_truth": truth,
                        "count": len(selected),
                        "accuracy": mean([float(row["correct"]) for row in selected]),
                        "predicted_yes_rate": mean(
                            [float(row["prediction"] == 0) for row in selected]
                        ),
                    }
                )
    core.write_rows(OUT / "analysis/behavior_truth_code_matrix.jsonl", behavior_matrix)
    logit_rows = margin_atlas(behavior, groups, protocol)
    core.write_rows(OUT / "analysis/logit_margin_four_factor_summary.jsonl", logit_rows)
    summary = {
        "phase": 1502,
        "campaign": "C086",
        "behavior": {
            "global_accuracy": mean([float(r["correct"]) for r in behavior]),
            "predicted_yes_rate": mean([float(r["prediction"] == 0) for r in behavior]),
            "predicted_no_rate": mean([float(r["prediction"] == 1) for r in behavior]),
            "strata": dict(Counter(group["stratum"] for group in groups)),
            "matrix": behavior_matrix,
        },
        "logit_margin": {
            "rho_content_median": float(np.median([r["rho_logit_content"] for r in logit_rows])),
            "D_standard_mean": mean([r["D_standard"] for r in logit_rows]),
            "D_reversed_mean": mean([r["D_reversed"] for r in logit_rows]),
            "beta_code_mean": mean([r["beta_P"] for r in logit_rows]),
        },
        "field_formation": {
            "generic_relation_onset_at_pairwise_0_8": persistent_onset(
                rows_by_split, "beta_relation_pairwise_mean", 0.8
            ),
            "generic_relation_code_onset_at_pairwise_0_8": persistent_onset(
                rows_by_split, "beta_relation_code_pairwise_mean", 0.8
            ),
            "c085_alignment_onset_at_0_7": persistent_onset(
                rows_by_split, "c085_standard_alignment_mean", 0.7
            ),
            "state35": trajectory[35],
        },
        "interpretation": {
            "supported": "a code-invariant same-versus-different relation-match response becomes highly shared across six lexical relations late in the controlled Qwen3 field",
            "not_supported": [
                "six distinct lexical relation identities share one semantic vector",
                "the field is used causally for correct counterbalanced behavior",
                "natural-language relation coding is solved",
                "a small fixed neuron set implements the field",
            ],
            "missing_strata": "success and failed are M2 missing; only mixed-behavior diagnostics exist",
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "behavior_count": len(behavior) == 6912,
        "group_count": len(groups) == 216,
        "mixed_only": summary["behavior"]["strata"] == {"mixed": 216},
        "trajectory": len(trajectory) == 37,
        "matrix": len(behavior_matrix) == 8 and all(row["count"] == 864 for row in behavior_matrix),
        "logit_panels": len(logit_rows) == 36,
        "finite": all(
            np.isfinite(value)
            for row in trajectory + behavior_matrix + logit_rows
            for value in row.values()
            if isinstance(value, (int, float))
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    summary["checks"] = checks
    core.save(OUT / "analysis/diagnostic_summary.json", summary)
    core.save(
        OUT / "analysis/final.json",
        {
            "phase": 1502,
            "campaign": "C086",
            "status": "stratum_and_c085_diagnostics_complete",
            "checks": checks,
            "authorization": "run_phase1503_c086_major_stage_closure",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
