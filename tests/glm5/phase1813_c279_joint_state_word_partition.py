#!/usr/bin/env python3
"""C279: partition full-coordinate responses by exact multi-role and token words."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common

core, OUT = common.core, common.OUTS["C279"]
C248 = common.previous.prior.OUTS["C248"]
C264 = common.previous.OUTS["C264"]
C265 = common.previous.OUTS["C265"]
C278 = common.OUTS["C278"]
CANDIDATES = {
    "relation_query": (common.ROLES.index("relation"), common.ROLES.index("query")),
    "primary_relation_query": (common.ROLES.index("primary"), common.ROLES.index("relation"), common.ROLES.index("query")),
    "all_six_roles": tuple(range(6)),
}


def pair_ids(index: list[dict], family: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def code_word(events: np.ndarray, roles: tuple[int, ...]) -> np.ndarray:
    code = np.zeros(events.shape[0:1] + events.shape[2:], np.int16)
    factor = 1
    for role in roles:
        code += (events[:, role] + 1).astype(np.int16) * factor
        factor *= 3
    return code


def lookup_support(train_codes: np.ndarray, test_codes: np.ndarray, states: int) -> np.ndarray:
    coordinates = np.arange(common.DIM, dtype=np.int64)[None, :]
    train_key = (train_codes.astype(np.int64) + coordinates * states).ravel()
    counts = np.bincount(train_key, minlength=common.DIM * states)
    test_key = test_codes.astype(np.int64) + coordinates * states
    return counts[test_key]


def token_hash(delta: np.ndarray, threshold: float, seed: int) -> np.ndarray:
    # Two independent unsigned hashes are used in the caller. Every token and
    # coordinate contributes; this is an exact-pattern index, not a projection.
    rng = np.random.default_rng(seed)
    coefficients = rng.integers(1, np.iinfo(np.uint64).max, size=delta.shape[1], dtype=np.uint64)
    result = np.zeros((delta.shape[0], delta.shape[2]), np.uint64)
    for token in range(delta.shape[1]):
        encoded = common.event(delta[:, token], threshold).astype(np.int16) + 1
        result += encoded.astype(np.uint64) * coefficients[token]
    return result


def exact_signature_support(train_a, train_b, test_a, test_b, threshold: float) -> tuple[float, float]:
    train_delta = np.concatenate((np.asarray(train_a, np.float32), np.asarray(train_b, np.float32)), axis=0)
    first_train = token_hash(train_delta, threshold, 27901)
    second_train = token_hash(train_delta, threshold, 27902)
    del train_delta
    test_delta = np.asarray(test_b, np.float32)
    first_test = token_hash(test_delta, threshold, 27901)
    second_test = token_hash(test_delta, threshold, 27902)
    # Coordinate identity is retained. Two hashes make accidental equality
    # negligible; no coordinate is selected or discarded.
    salt = np.arange(common.DIM, dtype=np.uint64)[None, :] * np.uint64(0x9E3779B185EBCA87)
    key_train = np.empty(first_train.size, dtype=[("a", "<u8"), ("b", "<u8")])
    key_train["a"] = (first_train ^ salt).ravel()
    key_train["b"] = (second_train ^ (salt << np.uint64(1))).ravel()
    unique, counts = np.unique(key_train, return_counts=True)
    key_test = np.empty(first_test.size, dtype=key_train.dtype)
    key_test["a"] = (first_test ^ salt).ravel()
    key_test["b"] = (second_test ^ (salt << np.uint64(1))).ravel()
    positions = np.searchsorted(unique, key_test)
    valid = positions < len(unique)
    matched = np.zeros(len(key_test), bool)
    matched[valid] = unique[positions[valid]] == key_test[valid]
    support = np.zeros(len(key_test), np.int32)
    support[matched] = counts[positions[matched]]
    return float(np.mean(support > 0)), float(np.mean(support >= 4))


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = [core.load(path / "analysis/final.json") for path in (C264, C278)]
    gates = core.load(common.OUTS["C277"] / "protocol/preregistration.json")["gates"]
    checks = {
        "parents": all(item["all_checks_passed"] for item in parents),
        "fifth_behavior": parents[-1]["headline"]["behavior_eligible"],
        "old_training_fields": (C248 / "raw/full_fields.float16.npy").exists() and (C264 / "raw/full_fields.float16.npy").exists(),
        "all_coordinates": True,
        "no_topk_pca_cosine_attention_mlp": True,
        "frozen_candidates": list(CANDIDATES) == ["relation_query", "primary_relation_query", "all_six_roles"],
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1813,
        "campaign": "C279",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partition_contract_frozen",
        "training_materials": ["C248 third", "C264 fourth"],
        "prospective_material": "C278 fifth",
        "role_words": {name: [common.ROLES[i] for i in roles] for name, roles in CANDIDATES.items()},
        "token_word": "double-uint64 identity of the complete 128-token signed event string for each physical coordinate; used only for exact repeat support",
        "support_min": gates["word_support_min"],
        "canonical_checkpoints": list(common.CANONICAL_CHECKPOINTS),
        "claim_boundary": "A repeated exact word is a partition cell, not a causal state or a semantic class. Low exact-token repetition may reflect lexical and positional change.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "C280_multisource_one_step_prediction",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    train_a_states = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    train_b_states = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    test_raw_states = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    train_a_fields = np.load(C248 / "raw/full_fields.float16.npy", mmap_mode="r")
    train_b_fields = np.load(C264 / "raw/full_fields.float16.npy", mmap_mode="r")
    test_raw_fields = np.load(C278 / "raw/full_fields.float16.npy", mmap_mode="r")
    indices = {
        "a": core.rows(C248 / "raw/hidden_index.jsonl"),
        "b": core.rows(C264 / "raw/hidden_index.jsonl"),
        "test": core.rows(C278 / "raw/hidden_index.jsonl"),
    }
    threshold = common.thresholds()
    transfer = np.zeros((len(common.FAMILIES), 36, len(CANDIDATES), 3), np.float32)
    token_transfer = np.zeros((len(common.FAMILIES), 36, 2), np.float32)
    family_rows = []
    for fi, family in enumerate(common.FAMILIES):
        al, ar = pair_ids(indices["a"], family)
        bl, br = pair_ids(indices["b"], family)
        tl, tr = pair_ids(indices["test"], family)
        if min(len(al), len(bl), len(tl)) < 1:
            raise RuntimeError((family, len(al), len(bl), len(tl)))
        for q in range(36):
            train_a_event = common.event(np.asarray(train_a_states[ar, q], np.float32) - np.asarray(train_a_states[al, q], np.float32), threshold[q])
            train_b_event = common.event(np.asarray(train_b_states[br, q], np.float32) - np.asarray(train_b_states[bl, q], np.float32), threshold[q])
            test_event = common.event(np.asarray(test_raw_states[tr, common.CANONICAL_NEW_INDICES[q]], np.float32) - np.asarray(test_raw_states[tl, common.CANONICAL_NEW_INDICES[q]], np.float32), threshold[q])
            for ci, roles in enumerate(CANDIDATES.values()):
                train_code = np.concatenate((code_word(train_a_event, roles), code_word(train_b_event, roles)), axis=0)
                test_code = code_word(test_event, roles)
                support = lookup_support(train_code, test_code, 3 ** len(roles))
                transfer[fi, q, ci] = [float(np.mean(support > 0)), float(np.mean(support >= gates["word_support_min"])), float(np.median(support))]
            first, supported = exact_signature_support(
                np.asarray(train_a_fields[ar, q], np.float32) - np.asarray(train_a_fields[al, q], np.float32),
                np.asarray(train_b_fields[br, q], np.float32) - np.asarray(train_b_fields[bl, q], np.float32),
                None,
                np.asarray(test_raw_fields[tr, common.CANONICAL_NEW_INDICES[q]], np.float32) - np.asarray(test_raw_fields[tl, common.CANONICAL_NEW_INDICES[q]], np.float32),
                threshold[q],
            )
            token_transfer[fi, q] = [first, supported]
        row = {
            "family": family,
            "pairs": {"third": int(len(al)), "fourth": int(len(bl)), "fifth": int(len(tl))},
            "role_partitions": {
                name: {
                    "median_seen_fraction": float(np.median(transfer[fi, :, ci, 0])),
                    "median_support4_fraction": float(np.median(transfer[fi, :, ci, 1])),
                    "median_cell_support": float(np.median(transfer[fi, :, ci, 2])),
                }
                for ci, name in enumerate(CANDIDATES)
            },
            "exact_token_signature_median_seen_fraction": float(np.median(token_transfer[fi, :, 0])),
            "exact_token_signature_median_support4_fraction": float(np.median(token_transfer[fi, :, 1])),
        }
        family_rows.append(row)
        print(f"[C279] {family}: all6 support4={row['role_partitions']['all_six_roles']['median_support4_fraction']:.4f}, token support4={row['exact_token_signature_median_support4_fraction']:.6f}", flush=True)
    np.save(OUT / "analysis/role_word_transfer.float32.npy", transfer)
    np.save(OUT / "analysis/exact_token_signature_transfer.float32.npy", token_transfer)
    core.write_rows(OUT / "analysis/family_results.jsonl", family_rows)
    report = {
        "phase": 1813,
        "campaign": "C279",
        "status": "joint_partition_observed",
        "families": family_rows,
        "strict_interpretation": "The result measures support of exact physical event partitions. It does not assume a low-dimensional Euclidean state and does not turn a repeated hash into a mechanism.",
        "next_authorization": "C280_test_all_three_registered_joint_words_prospectively",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {
        "families": len(family_rows) == 6,
        "transfer_shape": list(transfer.shape) == [6, 36, 3, 3],
        "token_shape": list(token_transfer.shape) == [6, 36, 2],
        "finite": bool(np.isfinite(transfer).all() and np.isfinite(token_transfer).all()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1813, "campaign": "C279", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

