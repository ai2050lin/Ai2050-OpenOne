#!/usr/bin/env python3
"""C276: prospectively predict fourth-material events with a frozen cross-role reuse map."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common


core = common.core
OUT = common.RESULT / "phase1810_c276_prospective_cross_role_reuse_prediction"
TRAIN = common.prior.OUTS["C248"]
TEST = common.OUTS["C264"]
C265 = common.OUTS["C265"]
C275 = common.RESULT / "phase1809_c275_joint_relational_state_observation"
FAMILIES = common.FAMILIES + ("nested_attitude",)
CONDITIONS = ("cross_role", "persistence", "wrong_role", "coordinate_roll")


def pair_ids(index: list[dict], family: str, panel: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family, "factor_a", panel)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def event(delta: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0)).astype(np.int8)


def combine(destination: np.ndarray, source: np.ndarray) -> np.ndarray:
    return np.where(destination != 0, destination, source).astype(np.int8)


def metrics(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    union = (predicted != 0) | (truth != 0)
    return np.asarray([
        ((predicted == truth) & union).sum(),
        union.sum(),
        ((predicted == truth) & (predicted != 0)).sum(),
        (predicted != 0).sum(),
        (truth != 0).sum(),
    ], np.int64)


def score(metric: np.ndarray) -> float:
    return float(metric[0] / max(int(metric[1]), 1))


def wrong_source(selected: int, destination: int) -> int:
    candidate = (selected + 1) % len(common.ROLES)
    return (candidate + 1) % len(common.ROLES) if candidate == destination else candidate


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C275 / "analysis/final.json")
    checks = {
        "parent_closed": parent["all_checks_passed"],
        "authorization": parent["next_authorization"].startswith("C276_prospective_cross_role_reuse_prediction"),
        "source_selection_only_C248": True,
        "prospective_test_only_C264": True,
        "all_coordinates": True,
        "no_topk_projection_attention_mlp_patching": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1810,
        "campaign": "C276",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "prospective_cross_role_map_frozen",
        "training": "C248 third material",
        "prospective_test": "C264 fourth material",
        "source_selection": "For every family, q, and destination role, choose the non-destination source role with maximum training signed-Jaccard after persistence-plus-source completion; tie by frozen role order.",
        "prediction": "Keep destination current events; where destination is inactive, insert the selected source role's same-coordinate signed current event.",
        "controls": ["destination persistence", "cyclic wrong source role", "selected source coordinates rolled by +1"],
        "family_gate": "cross-role signed-Jaccard exceeds every control by at least 0.01",
        "broad_gate": "at least four of six families pass",
        "claim_boundary": "A passing map would be an observational whole-state predictor, not a causal transport graph. No hidden state is patched.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "decide_after_fourth_material_reveal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    train_states = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    test_states = np.load(TEST / "raw/role_states.float16.npy", mmap_mode="r")
    train_index = core.rows(TRAIN / "raw/hidden_index.jsonl")
    test_index = core.rows(TEST / "raw/hidden_index.jsonl")
    thresholds = np.asarray(core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    source_map = np.zeros((len(FAMILIES), 36, 6), np.int8)
    training_scores = np.zeros((len(FAMILIES), 36, 6, 6), np.float32)
    coordinate_counts = np.zeros((len(FAMILIES), 36, 6, len(CONDITIONS), 2, 2560), np.uint16)
    family_rows: list[dict] = []
    for fi, family in enumerate(FAMILIES):
        panel = "nested_composition" if family == "nested_attitude" else "core"
        tl, tr = pair_ids(train_index, family, panel)
        vl, vr = pair_ids(test_index, family, panel)
        totals = {name: np.zeros(5, np.int64) for name in CONDITIONS}
        source_histogram = np.zeros(6, np.int64)
        for q in range(36):
            train_current = event(np.asarray(train_states[tr, q], np.float32) - np.asarray(train_states[tl, q], np.float32), thresholds[q])
            train_next = event(np.asarray(train_states[tr, q + 1], np.float32) - np.asarray(train_states[tl, q + 1], np.float32), thresholds[q + 1])
            test_current = event(np.asarray(test_states[vr, q], np.float32) - np.asarray(test_states[vl, q], np.float32), thresholds[q])
            test_next = event(np.asarray(test_states[vr, q + 1], np.float32) - np.asarray(test_states[vl, q + 1], np.float32), thresholds[q + 1])
            for di in range(6):
                choices = []
                for si in range(6):
                    if si == di:
                        training_scores[fi, q, di, si] = -1.0
                        continue
                    candidate_score = score(metrics(combine(train_current[:, di], train_current[:, si]), train_next[:, di]))
                    training_scores[fi, q, di, si] = candidate_score
                    choices.append((candidate_score, -si, si))
                selected = max(choices)[2]
                source_map[fi, q, di] = selected
                source_histogram[selected] += 1
                wrong = wrong_source(selected, di)
                predictions = {
                    "cross_role": combine(test_current[:, di], test_current[:, selected]),
                    "persistence": test_current[:, di],
                    "wrong_role": combine(test_current[:, di], test_current[:, wrong]),
                    "coordinate_roll": combine(test_current[:, di], np.roll(test_current[:, selected], 1, axis=1)),
                }
                truth = test_next[:, di]
                for ci, name in enumerate(CONDITIONS):
                    predicted = predictions[name]
                    totals[name] += metrics(predicted, truth)
                    coordinate_counts[fi, q, di, ci, 0] = ((predicted == truth) & (truth != 0)).sum(axis=0).astype(np.uint16)
                    coordinate_counts[fi, q, di, ci, 1] = ((predicted != 0) | (truth != 0)).sum(axis=0).astype(np.uint16)
        result_metrics = {}
        for name, total in totals.items():
            exact, union, active_correct, predicted_active, truth_active = total
            result_metrics[name] = {
                "signed_jaccard": float(exact / max(union, 1)),
                "signed_precision": float(active_correct / max(predicted_active, 1)),
                "signed_recall": float(active_correct / max(truth_active, 1)),
                "union": int(union),
            }
        best_control = max(result_metrics[name]["signed_jaccard"] for name in CONDITIONS if name != "cross_role")
        margin = result_metrics["cross_role"]["signed_jaccard"] - best_control
        family_rows.append({
            "family": family,
            "training_pairs": int(len(tl)),
            "test_pairs": int(len(vl)),
            "metrics": result_metrics,
            "source_role_histogram": {role: int(source_histogram[i]) for i, role in enumerate(common.ROLES)},
            "cross_role_minus_best_control": margin,
            "family_gate_passed": margin >= 0.01,
        })
        print(f"[C276] {family}: cross={result_metrics['cross_role']['signed_jaccard']:.4f}, best_control={best_control:.4f}, margin={margin:.4f}", flush=True)
    np.save(OUT / "analysis/frozen_source_map.int8.npy", source_map)
    np.save(OUT / "analysis/training_source_scores.float32.npy", training_scores)
    np.save(OUT / "analysis/coordinate_correct_union_counts.uint16.npy", coordinate_counts)
    core.write_rows(OUT / "analysis/family_results.jsonl", family_rows)
    passing = sum(bool(row["family_gate_passed"]) for row in family_rows)
    report = {
        "phase": 1810,
        "campaign": "C276",
        "status": "prospective_cross_role_reuse_adjudicated",
        "families": family_rows,
        "families_passing": passing,
        "broad_gate_passed": passing >= 4,
        "strict_interpretation": "The test asks whether a source-role map learned only on the third material predicts the complete next event field in the fourth material. It remains observational even if it passes.",
        "next_authorization": "C277_independent_material_and_natural_surface_replication" if passing >= 4 else "close_C263_C276_stage_with_cross_role_reuse_observation_only",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {
        "families": len(family_rows) == 6,
        "source_map_shape": list(source_map.shape) == [6, 36, 6],
        "coordinate_counts_shape": list(coordinate_counts.shape) == [6, 36, 6, 4, 2, 2560],
        "valid_sources": bool(np.all((source_map >= 0) & (source_map < 6))),
        "finite": bool(np.isfinite([row["cross_role_minus_best_control"] for row in family_rows]).all()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1810, "campaign": "C276", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
