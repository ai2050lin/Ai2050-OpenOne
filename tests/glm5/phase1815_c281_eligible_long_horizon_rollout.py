#!/usr/bin/env python3
"""C281: autonomously roll the prospectively qualified C280 joint words."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common
import phase1813_c279_joint_state_word_partition as partition
import phase1814_c280_multisource_one_step_prediction as one_step

core, OUT = common.core, common.OUTS["C281"]
C248 = common.previous.prior.OUTS["C248"]
C264 = common.previous.OUTS["C264"]
C265 = common.previous.OUTS["C265"]
C278 = common.OUTS["C278"]
C280 = common.OUTS["C280"]
HORIZONS = (8, 16, 24, 36)


def pair_ids(index: list[dict], family: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C280 / "analysis/final.json")
    passing = tuple(parent["headline"]["passing_candidates"])
    checks = {
        "parent": parent["all_checks_passed"],
        "eligible": parent["headline"]["broad_gate_passed"],
        "passing_candidates": bool(passing),
        "starts_only_from_embedding_response": True,
        "no_intermediate_truth_input": True,
        "all_coordinates": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1815,
        "campaign": "C281",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "autonomous_rollout_frozen",
        "candidates": list(passing),
        "initial_state": "observed fifth-material embedding response at canonical q0",
        "rollout": "At each q, form the exact joint word only from the previously predicted six-role event field; apply the frozen q-specific map; retain active destination events and legally abstain where the map is undefined.",
        "horizons": list(HORIZONS),
        "baselines": ["hold embedding event field fixed", "coordinate-rolled autonomous prediction"],
        "family_gate": "median q16/q24/q36 signed-Jaccard exceeds both controls by 0.01",
        "broad_gate": "at least four families pass",
        "claim_boundary": "Autonomous signed-event rollout is not continuous HiddenState emulation and does not identify a unique circuit.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C282_C284_regardless; C285_only_if_long_horizon_qualified",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    train_a = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    train_b = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    test_raw = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    indices = {
        "a": core.rows(C248 / "raw/hidden_index.jsonl"),
        "b": core.rows(C264 / "raw/hidden_index.jsonl"),
        "test": core.rows(C278 / "raw/hidden_index.jsonl"),
    }
    threshold = common.thresholds()
    family_rows = []
    for family in common.FAMILIES:
        al, ar = pair_ids(indices["a"], family)
        bl, br = pair_ids(indices["b"], family)
        tl, tr = pair_ids(indices["test"], family)
        train_events = []
        train_next = []
        truth = []
        for q in range(36):
            train_events.append(np.concatenate((
                common.event(np.asarray(train_a[ar, q], np.float32) - np.asarray(train_a[al, q], np.float32), threshold[q]),
                common.event(np.asarray(train_b[br, q], np.float32) - np.asarray(train_b[bl, q], np.float32), threshold[q]),
            ), axis=0))
            train_next.append(np.concatenate((
                common.event(np.asarray(train_a[ar, q + 1], np.float32) - np.asarray(train_a[al, q + 1], np.float32), threshold[q + 1]),
                common.event(np.asarray(train_b[br, q + 1], np.float32) - np.asarray(train_b[bl, q + 1], np.float32), threshold[q + 1]),
            ), axis=0))
            truth.append(common.event(
                np.asarray(test_raw[tr, common.CANONICAL_NEW_INDICES[q + 1]], np.float32) - np.asarray(test_raw[tl, common.CANONICAL_NEW_INDICES[q + 1]], np.float32),
                threshold[q + 1],
            ))
        initial = common.event(
            np.asarray(test_raw[tr, common.CANONICAL_NEW_INDICES[0]], np.float32) - np.asarray(test_raw[tl, common.CANONICAL_NEW_INDICES[0]], np.float32),
            threshold[0],
        )
        candidate_rows = {}
        for name in passing:
            roles = partition.CANDIDATES[name]
            states = 3 ** len(roles)
            predicted = initial.copy()
            horizon_metrics = {}
            for q in range(36):
                train_code = partition.code_word(train_events[q], roles)
                predicted_code = partition.code_word(predicted, roles)
                next_predicted = np.zeros_like(predicted)
                for destination in range(6):
                    fitted, _key, _support = one_step.fit_map(train_code, train_next[q][:, destination], states, 4, 0.70)
                    word = one_step.lookup(fitted, predicted_code, states)
                    next_predicted[:, destination] = np.where(predicted[:, destination] != 0, predicted[:, destination], word)
                predicted = next_predicted
                horizon = q + 1
                if horizon in HORIZONS:
                    primary = common.metric_dict(common.event_metrics(predicted, truth[q]))
                    fixed = common.metric_dict(common.event_metrics(initial, truth[q]))
                    rolled = common.metric_dict(common.event_metrics(np.roll(predicted, 1, axis=2), truth[q]))
                    horizon_metrics[str(horizon)] = {"autonomous": primary, "fixed_embedding": fixed, "coordinate_roll": rolled}
            long_horizons = [horizon_metrics[str(q)] for q in (16, 24, 36)]
            margins = [row["autonomous"]["signed_jaccard"] - max(row["fixed_embedding"]["signed_jaccard"], row["coordinate_roll"]["signed_jaccard"]) for row in long_horizons]
            candidate_rows[name] = {
                "horizons": horizon_metrics,
                "long_horizon_median_margin": float(np.median(margins)),
                "family_gate_passed": float(np.median(margins)) >= 0.01,
            }
        family_rows.append({"family": family, "candidates": candidate_rows})
        print(f"[C281] {family}: " + ", ".join(f"{name}={candidate_rows[name]['long_horizon_median_margin']:+.4f}" for name in passing), flush=True)
    core.write_rows(OUT / "analysis/family_results.jsonl", family_rows)
    candidate_summary = []
    for name in passing:
        count = sum(bool(row["candidates"][name]["family_gate_passed"]) for row in family_rows)
        candidate_summary.append({"candidate": name, "families_passing": count, "broad_gate_passed": count >= 4})
    broad = any(row["broad_gate_passed"] for row in candidate_summary)
    report = {
        "phase": 1815,
        "campaign": "C281",
        "status": "autonomous_rollout_adjudicated",
        "candidate_summary": candidate_summary,
        "families": family_rows,
        "broad_gate_passed": broad,
        "strict_interpretation": "One-step qualification does not guarantee autonomous closure. The rollout never reads the fifth-material intermediate truth after q0.",
        "next_authorization": "C285_may_use_only_candidates_with_broad_long_horizon_gate" if broad else "C285_registered_no_test; continue_composition_generation_cross_model_routes",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"families": len(family_rows) == 6, "candidates": len(candidate_summary) == len(passing), "finite": bool(np.isfinite([row["candidates"][name]["long_horizon_median_margin"] for row in family_rows for name in passing]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1815, "campaign": "C281", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

