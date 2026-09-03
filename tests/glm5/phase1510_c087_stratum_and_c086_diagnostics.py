#!/usr/bin/env python3
"""Phase1510: C087 behavior-stratum, execution, and C086 paired diagnostics."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
CONTRACT = RESULT / "phase1504_c087_cross_root_semeval_contract"
BEHAVIOR = RESULT / "phase1505_c087_behavior_stratification"
CAPTURE = RESULT / "phase1506_c087_all_case_field_capture"
ATLAS = RESULT / "phase1507_c087_descriptive_semantic_contrast_atlas"
VALIDATION = RESULT / "phase1509_c087_dual_holdout_validation"
C086_ATLAS = RESULT / "phase1499_c086_four_factor_atlas"
C086_CONTRACT = RESULT / "phase1496_c086_unlabeled_counterbalanced_contract"
OUT = RESULT / "phase1510_c087_stratum_and_c086_diagnostics"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
import phase1508_c087_discovery_observation_freeze as metric


def field_metrics(group, selected, state, role):
    panel = np.asarray(group[selected, :, state, role], dtype=np.float64)
    per_group = panel.mean(axis=1)
    centroids = panel.mean(axis=0)
    return {
        "count": len(selected),
        "surface_centroid_cosine": metric.cosine(centroids[0], centroids[1]),
        "within_group_surface_cosine_mean": float(np.mean([metric.cosine(row[0], row[1]) for row in panel])),
        "group_pairwise_cosine_mean": metric.pairwise_mean(per_group),
        "shared_energy_fraction": metric.coherence(per_group),
        "top1pct_coordinate_energy": metric.concentration(per_group),
        "centroid_norm": float(np.linalg.norm(per_group.mean(axis=0))),
    }


def summarize(values):
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "minimum": float(np.min(values)) if values else None,
        "maximum": float(np.max(values)) if values else None,
    }


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1510 exists")
    parent = core.load(VALIDATION / "analysis/final.json")
    parent_audit = core.load(VALIDATION / "audit/independent_final_audit.json")
    if parent["authorization"] != "run_phase1510_c087_stratum_and_c086_diagnostics" or not parent_audit["all_checks_passed"]:
        raise RuntimeError("Phase1509 authorization missing")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    group = np.load(ATLAS / "atlas/group_semantic_contrast.float32.npy", mmap_mode="r")
    group_index = core.rows(ATLAS / "atlas/group_semantic_contrast_index.jsonl")
    behavior = core.rows(BEHAVIOR / "raw/behavior.jsonl")
    capture_index = core.rows(CAPTURE / "raw/all_role_field_index.jsonl")
    capture_by = {row["case_id"]: row for row in capture_index}
    behavior_by = {row["case_id"]: row for row in behavior}
    partitions = tuple(protocol["partitions"])

    stratum_field = {}
    for partition in partitions:
        stratum_field[partition] = {}
        for stratum in ("all", "success", "mixed"):
            selected = [
                row["group_index"] for row in group_index
                if row["partition"] == partition and (stratum == "all" or row["stratum"] == stratum)
            ]
            stratum_field[partition][stratum] = field_metrics(group, selected, 35, 2)

    compiled_by = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    group_stratum = {
        key: row["stratum"]
        for row in core.rows(BEHAVIOR / "material/stratified_composition_sets.jsonl")
        for key in (row["a_natural_same"], row["a_natural_different"], row["b_natural_same"], row["b_natural_different"])
    }
    margin_cells = defaultdict(list)
    for row in behavior:
        candidates = compiled_by[row["case_id"]]["candidates"]
        same_score = row["scores"][candidates.index("same")]
        different_score = row["scores"][candidates.index("different")]
        semantic_axis = same_score - different_score
        correct_margin = semantic_axis if row["semantic_match"] else -semantic_axis
        keys = (
            "global",
            f"partition:{row['partition']}",
            f"surface:{row['surface']}",
            f"truth:{str(row['semantic_match']).lower()}",
            f"stratum:{group_stratum[row['case_id']]}",
        )
        for key in keys:
            margin_cells[key].append(correct_margin)
    behavior_margins = {key: summarize(values) for key, values in sorted(margin_cells.items())}

    prediction_disagreements = [
        {
            "case_id": row["case_id"],
            "partition": row["partition"],
            "surface": row["surface"],
            "semantic_label": row["semantic_label"],
            "behavior_prediction": behavior_by[row["case_id"]]["prediction"],
            "capture_prediction": row["capture_prediction"],
            "behavior_scores": behavior_by[row["case_id"]]["scores"],
            "capture_scores": row["capture_scores"],
        }
        for row in capture_index if row["capture_prediction"] != behavior_by[row["case_id"]]["prediction"]
    ]

    all_centroids = []
    for partition in partitions:
        selected = [row["group_index"] for row in group_index if row["partition"] == partition]
        all_centroids.append(np.asarray(group[selected, :, 35, 2], dtype=np.float64).mean(axis=(0, 1)))
    cross_partition_cosines = {
        f"{partitions[i]}__{partitions[j]}": metric.cosine(all_centroids[i], all_centroids[j])
        for i in range(3) for j in range(i + 1, 3)
    }

    c086 = np.load(C086_ATLAS / "atlas/all_four_factor_contrast_mean.float32.npy", mmap_mode="r")
    c086_summary = core.load(C086_ATLAS / "analysis/four_factor_atlas_summary.json")
    c086_protocol = core.load(C086_CONTRACT / "protocol/preregistration.json")
    effect = c086_summary["effects"].index("relation")
    boundary86 = c086_protocol["roles"].index("boundary")
    c086_alignment_trajectory = []
    for state in range(37):
        c086_vector = np.asarray(c086[effect, :, :, :, state, boundary86], dtype=np.float64).mean(axis=(0, 1, 2))
        values = []
        for partition_index, partition in enumerate(partitions):
            selected = [row["group_index"] for row in group_index if row["partition"] == partition]
            centroids = np.asarray(group[selected, :, state, 2], dtype=np.float64).mean(axis=0)
            values.extend(metric.cosine(surface, c086_vector) for surface in centroids)
        c086_alignment_trajectory.append({"state": state, "mean_alignment": float(np.mean(values))})
    c086_peak = max(c086_alignment_trajectory, key=lambda row: row["mean_alignment"])

    validation = core.load(VALIDATION / "analysis/dual_holdout_validation.json")
    lockbox = validation["holdouts"]["lockbox"]
    failed_checks = [name for name, value in lockbox["primary_checks"].items() if not value]
    lockbox_failure_anatomy = {
        "failed_checks": failed_checks,
        "only_failed_check": len(failed_checks) == 1,
        "pairwise_difference_from_discovery": lockbox["boundary"]["group_pairwise_cosine_mean"] - core.load(RESULT / "phase1508_c087_discovery_observation_freeze/analysis/discovery_summary.json")["boundary"]["group_pairwise_cosine_mean"],
        "direction": "stronger_than_discovery" if lockbox["boundary"]["group_pairwise_cosine_mean"] > core.load(RESULT / "phase1508_c087_discovery_observation_freeze/analysis/discovery_summary.json")["boundary"]["group_pairwise_cosine_mean"] else "weaker_than_discovery",
        "gate_remains_failed": not validation["dual_holdout_primary_pass"],
    }

    diagnostics = {
        "phase": 1510,
        "campaign": "C087",
        "state35_boundary_by_behavior_stratum": stratum_field,
        "behavior_correct_margin": behavior_margins,
        "capture_execution_disagreements": prediction_disagreements,
        "capture_disagreement_count": len(prediction_disagreements),
        "capture_disagreement_partition_counts": dict(Counter(row["partition"] for row in prediction_disagreements)),
        "cross_partition_state35_boundary_cosines": cross_partition_cosines,
        "c086_alignment_trajectory": c086_alignment_trajectory,
        "c086_alignment_peak": c086_peak,
        "lockbox_failure_anatomy": lockbox_failure_anatomy,
        "interpretation": {
            "supported": "a late boundary same-minus-different response repeats across three disjoint lexical-item partitions and two prompt surfaces",
            "not_supported": "a universal semantic comparator, semantic understanding without output-code confounding, causal use, coordinate-local mechanism, or cross-model law",
            "central_confound": "C087 semantic truth and same/different answer direction are not factorially separated",
        },
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = {
        "stratum_counts": all(stratum_field[p]["all"]["count"] == 72 for p in partitions) and sum(stratum_field[p]["success"]["count"] for p in partitions) == 128 and sum(stratum_field[p]["mixed"]["count"] for p in partitions) == 88,
        "behavior_count": behavior_margins["global"]["count"] == 864,
        "execution_disagreements": len(prediction_disagreements) == 4,
        "cross_partition": len(cross_partition_cosines) == 3 and all(np.isfinite(value) for value in cross_partition_cosines.values()),
        "trajectory": len(c086_alignment_trajectory) == 37 and all(np.isfinite(row["mean_alignment"]) for row in c086_alignment_trajectory),
        "strict_gate_preserved": lockbox_failure_anatomy["gate_remains_failed"],
        "single_overstrength_failure": lockbox_failure_anatomy["only_failed_check"] and lockbox_failure_anatomy["direction"] == "stronger_than_discovery",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    diagnostics["checks"] = checks
    core.save(OUT / "analysis/stratum_and_c086_diagnostics.json", diagnostics)
    core.save(OUT / "analysis/final.json", {
        "phase": 1510,
        "campaign": "C087",
        "status": "stratum_execution_and_paired_diagnostics_complete",
        "checks": checks,
        "authorization": "run_phase1511_c087_major_stage_closure",
    })
    print(json.dumps({key: value for key, value in diagnostics.items() if key not in ("c086_alignment_trajectory", "capture_execution_disagreements")}, indent=2))


if __name__ == "__main__":
    main()
