#!/usr/bin/env python3
"""C280: prospectively test exact multi-role words on the fifth material."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common
import phase1813_c279_joint_state_word_partition as partition

core, OUT = common.core, common.OUTS["C280"]
C248 = common.previous.prior.OUTS["C248"]
C264 = common.previous.OUTS["C264"]
C265 = common.previous.OUTS["C265"]
C278 = common.OUTS["C278"]
C279 = common.OUTS["C279"]
CANDIDATES = partition.CANDIDATES
CONDITIONS = tuple(f"{name}_{variant}" for name in CANDIDATES for variant in ("pure", "completion")) + ("persistence",)


def pair_ids(index: list[dict], family: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def fit_map(train_code: np.ndarray, train_truth: np.ndarray, states: int, support_min: int, agreement_min: float):
    coordinates = np.arange(common.DIM, dtype=np.int64)[None, :]
    key = (train_code.astype(np.int64) + coordinates * states).ravel()
    size = common.DIM * states
    support = np.bincount(key, minlength=size).astype(np.int32)
    flat_truth = train_truth.ravel()
    negative = np.bincount(key, weights=(flat_truth == -1), minlength=size).astype(np.int32)
    positive = np.bincount(key, weights=(flat_truth == 1), minlength=size).astype(np.int32)
    zero = support - negative - positive
    counts = np.stack((negative, zero, positive), axis=0)
    choice = counts.argmax(axis=0)
    agreement = counts.max(axis=0) / np.maximum(support, 1)
    prediction = np.where((support >= support_min) & (agreement >= agreement_min), choice - 1, 0).astype(np.int8)
    return prediction, key, support


def lookup(prediction: np.ndarray, code: np.ndarray, states: int) -> np.ndarray:
    coordinates = np.arange(common.DIM, dtype=np.int64)[None, :]
    return prediction[code.astype(np.int64) + coordinates * states]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C279 / "analysis/final.json")
    gates = core.load(common.OUTS["C277"] / "protocol/preregistration.json")["gates"]
    checks = {
        "parent": parent["all_checks_passed"],
        "training_only_third_fourth": True,
        "prospective_only_fifth": True,
        "all_coordinates": True,
        "three_frozen_words": tuple(CANDIDATES) == ("relation_query", "primary_relation_query", "all_six_roles"),
        "no_topk_pca_cosine_attention_mlp": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1814,
        "campaign": "C280",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "one_step_rule_frozen",
        "training": ["C248 third", "C264 fourth"],
        "prospective_test": "C278 fifth",
        "candidate_words": {name: [common.ROLES[i] for i in roles] for name, roles in CANDIDATES.items()},
        "rule": "For each family, canonical transition, destination role, physical coordinate and exact joint word, predict the dominant next sign when support>=4 and agreement>=0.70.",
        "primary_variant": "completion keeps destination persistence where active and uses the word only where the destination is currently silent",
        "controls": ["destination persistence", "joint prediction rolled by one physical coordinate", "joint word code cyclically permuted within coordinate"],
        "amplitude": "For all-six completion only, minimum/maximum matching training magnitude; reported conditionally, not used to rescue the sign gate.",
        "family_gate": "completion signed-Jaccard exceeds persistence, coordinate roll, and word-code permutation by at least 0.01",
        "broad_gate": "at least four of six families pass for one frozen candidate",
        "claim_boundary": "Passing is a reusable observational transition rule, not a unique causal hyperedge.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "C281_only_for_passing_candidate; C282_C284_C286_C288_run_regardless",
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
    atlas = np.zeros((len(common.FAMILIES), 36, len(CANDIDATES), 2, common.DIM), np.uint32)
    family_rows = []
    for fi, family in enumerate(common.FAMILIES):
        al, ar = pair_ids(indices["a"], family)
        bl, br = pair_ids(indices["b"], family)
        tl, tr = pair_ids(indices["test"], family)
        totals = {name: np.zeros(5, np.int64) for name in CONDITIONS}
        controls = {name: {control: np.zeros(5, np.int64) for control in ("coordinate_roll", "word_permutation")} for name in CANDIDATES}
        amplitude_hit = 0
        amplitude_den = 0
        for q in range(36):
            train_event = np.concatenate((
                common.event(np.asarray(train_a[ar, q], np.float32) - np.asarray(train_a[al, q], np.float32), threshold[q]),
                common.event(np.asarray(train_b[br, q], np.float32) - np.asarray(train_b[bl, q], np.float32), threshold[q]),
            ), axis=0)
            test_event = common.event(
                np.asarray(test_raw[tr, common.CANONICAL_NEW_INDICES[q]], np.float32) - np.asarray(test_raw[tl, common.CANONICAL_NEW_INDICES[q]], np.float32),
                threshold[q],
            )
            train_next_delta = np.concatenate((
                np.asarray(train_a[ar, q + 1], np.float32) - np.asarray(train_a[al, q + 1], np.float32),
                np.asarray(train_b[br, q + 1], np.float32) - np.asarray(train_b[bl, q + 1], np.float32),
            ), axis=0)
            test_next_delta = np.asarray(test_raw[tr, common.CANONICAL_NEW_INDICES[q + 1]], np.float32) - np.asarray(test_raw[tl, common.CANONICAL_NEW_INDICES[q + 1]], np.float32)
            train_next = common.event(train_next_delta, threshold[q + 1])
            test_next = common.event(test_next_delta, threshold[q + 1])
            for ci, (name, roles) in enumerate(CANDIDATES.items()):
                states = 3 ** len(roles)
                train_code = partition.code_word(train_event, roles)
                test_code = partition.code_word(test_event, roles)
                permuted_code = (test_code + 1) % states
                for destination in range(6):
                    fitted, train_key, _support = fit_map(train_code, train_next[:, destination], states, gates["word_support_min"], gates["word_agreement_min"])
                    pure = lookup(fitted, test_code, states)
                    completion = np.where(test_event[:, destination] != 0, test_event[:, destination], pure).astype(np.int8)
                    word_null = lookup(fitted, permuted_code, states)
                    word_completion = np.where(test_event[:, destination] != 0, test_event[:, destination], word_null).astype(np.int8)
                    rolled = np.roll(completion, 1, axis=1)
                    if ci == 0:
                        totals["persistence"] += common.event_metrics(test_event[:, destination], test_next[:, destination])
                    totals[f"{name}_pure"] += common.event_metrics(pure, test_next[:, destination])
                    totals[f"{name}_completion"] += common.event_metrics(completion, test_next[:, destination])
                    controls[name]["coordinate_roll"] += common.event_metrics(rolled, test_next[:, destination])
                    controls[name]["word_permutation"] += common.event_metrics(word_completion, test_next[:, destination])
                    atlas[fi, q, ci, 0] += ((completion == test_next[:, destination]) & (test_next[:, destination] != 0)).sum(axis=0).astype(np.uint32)
                    atlas[fi, q, ci, 1] += ((completion != 0) | (test_next[:, destination] != 0)).sum(axis=0).astype(np.uint32)
                    if name == "all_six_roles":
                        size = common.DIM * states
                        chosen_train = fitted[train_key]
                        flat_truth = train_next[:, destination].ravel()
                        flat_magnitude = np.abs(train_next_delta[:, destination]).ravel()
                        eligible = (chosen_train != 0) & (chosen_train == flat_truth)
                        lo = np.full(size, np.inf, np.float32)
                        hi = np.full(size, -np.inf, np.float32)
                        np.minimum.at(lo, train_key[eligible], flat_magnitude[eligible])
                        np.maximum.at(hi, train_key[eligible], flat_magnitude[eligible])
                        coordinates = np.arange(common.DIM, dtype=np.int64)[None, :]
                        test_key = test_code.astype(np.int64) + coordinates * states
                        valid = (pure != 0) & np.isfinite(lo[test_key]) & (pure == test_next[:, destination])
                        magnitude = np.abs(test_next_delta[:, destination])
                        amplitude_hit += int((valid & (magnitude >= lo[test_key]) & (magnitude <= hi[test_key])).sum())
                        amplitude_den += int(valid.sum())
        metrics = {name: common.metric_dict(total) for name, total in totals.items()}
        control_metrics = {name: {control: common.metric_dict(total) for control, total in values.items()} for name, values in controls.items()}
        candidates = {}
        for name in CANDIDATES:
            score = metrics[f"{name}_completion"]["signed_jaccard"]
            baselines = [metrics["persistence"]["signed_jaccard"]] + [row["signed_jaccard"] for row in control_metrics[name].values()]
            margin = score - max(baselines)
            candidates[name] = {
                "pure": metrics[f"{name}_pure"],
                "completion": metrics[f"{name}_completion"],
                "controls": control_metrics[name],
                "minus_best_control": margin,
                "family_gate_passed": margin >= gates["one_step_margin_min"],
            }
        row = {
            "family": family,
            "pairs": {"training": int(len(al) + len(bl)), "test": int(len(tl))},
            "persistence": metrics["persistence"],
            "candidates": candidates,
            "all_six_amplitude_interval_coverage_given_correct_sign": float(amplitude_hit / max(amplitude_den, 1)),
            "all_six_amplitude_interval_denominator": amplitude_den,
        }
        family_rows.append(row)
        print(f"[C280] {family}: " + ", ".join(f"{name}={candidates[name]['completion']['signed_jaccard']:.4f}/{candidates[name]['minus_best_control']:+.4f}" for name in CANDIDATES), flush=True)
    np.save(OUT / "analysis/coordinate_correct_union_counts.uint32.npy", atlas)
    core.write_rows(OUT / "analysis/family_results.jsonl", family_rows)
    candidate_summary = []
    for name in CANDIDATES:
        passing = sum(bool(row["candidates"][name]["family_gate_passed"]) for row in family_rows)
        candidate_summary.append({
            "candidate": name,
            "families_passing": passing,
            "median_margin": float(np.median([row["candidates"][name]["minus_best_control"] for row in family_rows])),
            "broad_gate_passed": passing >= gates["broad_families_min"],
        })
    passing_candidates = [row["candidate"] for row in candidate_summary if row["broad_gate_passed"]]
    report = {
        "phase": 1814,
        "campaign": "C280",
        "status": "multisource_prediction_adjudicated",
        "candidate_summary": candidate_summary,
        "families": family_rows,
        "broad_gate_passed": bool(passing_candidates),
        "passing_candidates": passing_candidates,
        "strict_interpretation": "Exact multi-role words were tested only after support was established. Passing would be an observational one-step automaton; failing would reject these registered partitions, not every joint state.",
        "next_authorization": "C281_roll_passing_candidates_and_run_C282_C284_C286_C288" if passing_candidates else "C281_registered_no_test_and_run_C282_C284_C286_C288",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {
        "families": len(family_rows) == 6,
        "candidate_summary": len(candidate_summary) == 3,
        "atlas_shape": list(atlas.shape) == [6, 36, 3, 2, 2560],
        "finite": bool(np.isfinite([row["median_margin"] for row in candidate_summary]).all()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1814, "campaign": "C280", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
