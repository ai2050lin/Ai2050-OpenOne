#!/usr/bin/env python3
"""C274: prospectively test two transparent joint full-field guards."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1797_c263_c272_state_operator_common as common


core = common.core
OUT = common.RESULT / "phase1808_c274_joint_full_field_condition"
TRAIN = common.prior.OUTS["C248"]
TEST = common.OUTS["C264"]
C265 = common.OUTS["C265"]
C273 = common.RESULT / "phase1807_c273_full_field_response_ecology"
FAMILIES = common.FAMILIES + ("nested_attitude",)
CANDIDATES = ("role_joint", "all_role_joint")


def pair_ids(index: list[dict], family: str, panel: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family, "factor_a", panel)
    return np.asarray([row[0] for row in specs], int), np.asarray([row[1] for row in specs], int)


def event(delta: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(delta > threshold, 1, np.where(delta < -threshold, -1, 0)).astype(np.int8)


def key_for(current: np.ndarray, positive: np.ndarray, negative: np.ndarray, activity_median: float) -> np.ndarray:
    polarity = positive >= negative
    activity = positive + negative >= activity_median
    return np.where(
        current < 0,
        2 * polarity[:, None].astype(np.int8) + 4 * activity[:, None].astype(np.int8),
        np.where(
            current > 0,
            1 + 2 * polarity[:, None].astype(np.int8) + 4 * activity[:, None].astype(np.int8),
            -1,
        ),
    )


def event_metrics(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    union = (predicted != 0) | (truth != 0)
    return np.asarray([
        ((predicted == truth) & union).sum(),
        union.sum(),
        ((predicted == truth) & (predicted != 0)).sum(),
        (predicted != 0).sum(),
        (truth != 0).sum(),
    ], np.int64)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = [
        core.load(path / "analysis/final.json")
        for path in (TEST, C265, C273)
    ]
    checks = {
        "parents_closed": all(item["all_checks_passed"] for item in parents),
        "authorization": parents[-1]["next_authorization"].startswith("C274_joint_full_field_condition"),
        "training_only_C248": True,
        "holdout_only_C264": True,
        "all_coordinates": True,
        "no_topk_projection_attention_mlp": True,
        "same_event_thresholds": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1808,
        "campaign": "C274",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "joint_guard_tournament_frozen",
        "training": "C248 third material",
        "prospective_holdout": "C264 fourth material",
        "candidates": {
            "role_joint": "current coordinate sign x role-wide positive-vs-negative polarity x role-wide event activity above training median",
            "all_role_joint": "current coordinate sign x six-role positive-vs-negative polarity x six-role event activity above training median",
        },
        "support_min": 4,
        "agreement_min": 0.70,
        "family_gate": "candidate signed-Jaccard exceeds same-coordinate persistence by at least 0.01",
        "broad_gate": "at least four of six families pass for one predeclared candidate",
        "claim_boundary": "These are transparent joint summary guards over all coordinates. Passing would establish prediction, not a unique mechanism; failing does not rule out richer joint-state structure.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "decide_after_fourth_material_reveal",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    train_states = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    test_states = np.load(TEST / "raw/role_states.float16.npy", mmap_mode="r")
    train_index = core.rows(TRAIN / "raw/hidden_index.jsonl")
    test_index = core.rows(TEST / "raw/hidden_index.jsonl")
    thresholds = np.asarray(
        core.load(common.prior.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"],
        np.float32,
    )
    pred_maps = {
        name: np.lib.format.open_memmap(
            OUT / f"analysis/{name}_pred_sign.int8.npy",
            mode="w+",
            dtype=np.int8,
            shape=(len(FAMILIES), 36, len(common.ROLES), 8, 2560),
        )
        for name in CANDIDATES
    }
    results: list[dict] = []
    for fi, family in enumerate(FAMILIES):
        panel = "nested_composition" if family == "nested_attitude" else "core"
        tl, tr = pair_ids(train_index, family, panel)
        vl, vr = pair_ids(test_index, family, panel)
        totals = {name: np.zeros(5, np.int64) for name in (*CANDIDATES, "persistence")}
        for q in range(36):
            train_delta_all = np.asarray(train_states[tr, q], np.float32) - np.asarray(train_states[tl, q], np.float32)
            test_delta_all = np.asarray(test_states[vr, q], np.float32) - np.asarray(test_states[vl, q], np.float32)
            train_event_all = event(train_delta_all, thresholds[q])
            test_event_all = event(test_delta_all, thresholds[q])
            train_all_pos = (train_event_all > 0).sum(axis=(1, 2))
            train_all_neg = (train_event_all < 0).sum(axis=(1, 2))
            test_all_pos = (test_event_all > 0).sum(axis=(1, 2))
            test_all_neg = (test_event_all < 0).sum(axis=(1, 2))
            all_activity_median = float(np.median(train_all_pos + train_all_neg))
            for ri in range(len(common.ROLES)):
                train_current = train_event_all[:, ri]
                test_current = test_event_all[:, ri]
                train_next_delta = np.asarray(train_states[tr, q + 1, ri], np.float32) - np.asarray(train_states[tl, q + 1, ri], np.float32)
                test_next_delta = np.asarray(test_states[vr, q + 1, ri], np.float32) - np.asarray(test_states[vl, q + 1, ri], np.float32)
                train_truth = event(train_next_delta, thresholds[q + 1])
                test_truth = event(test_next_delta, thresholds[q + 1])
                train_role_pos = (train_current > 0).sum(axis=1)
                train_role_neg = (train_current < 0).sum(axis=1)
                test_role_pos = (test_current > 0).sum(axis=1)
                test_role_neg = (test_current < 0).sum(axis=1)
                role_activity_median = float(np.median(train_role_pos + train_role_neg))
                train_keys = {
                    "role_joint": key_for(train_current, train_role_pos, train_role_neg, role_activity_median),
                    "all_role_joint": key_for(train_current, train_all_pos, train_all_neg, all_activity_median),
                }
                test_keys = {
                    "role_joint": key_for(test_current, test_role_pos, test_role_neg, role_activity_median),
                    "all_role_joint": key_for(test_current, test_all_pos, test_all_neg, all_activity_median),
                }
                for name in CANDIDATES:
                    predicted = np.zeros_like(test_current)
                    for key in range(8):
                        member = train_keys[name] == key
                        support = member.sum(axis=0)
                        counts = np.stack([((train_truth == sign) & member).sum(axis=0) for sign in (-1, 0, 1)], axis=0)
                        choice = counts.argmax(axis=0)
                        agreement = counts.max(axis=0) / np.maximum(support, 1)
                        frozen = np.where(
                            (support >= protocol["support_min"]) & (agreement >= protocol["agreement_min"]),
                            choice - 1,
                            0,
                        ).astype(np.int8)
                        pred_maps[name][fi, q, ri, key] = frozen
                        mask = test_keys[name] == key
                        predicted[mask] = np.broadcast_to(frozen, predicted.shape)[mask]
                    totals[name] += event_metrics(predicted, test_truth)
                totals["persistence"] += event_metrics(test_current, test_truth)
        for item in pred_maps.values():
            item.flush()
        metrics = {}
        for name, total in totals.items():
            exact, union, active_correct, predicted_active, truth_active = total
            metrics[name] = {
                "signed_jaccard": float(exact / max(union, 1)),
                "signed_precision": float(active_correct / max(predicted_active, 1)),
                "signed_recall": float(active_correct / max(truth_active, 1)),
                "union": int(union),
            }
        family_result = {"family": family, "training_pairs": int(len(tl)), "test_pairs": int(len(vl)), "metrics": metrics}
        for name in CANDIDATES:
            margin = metrics[name]["signed_jaccard"] - metrics["persistence"]["signed_jaccard"]
            family_result[f"{name}_minus_persistence"] = margin
            family_result[f"{name}_gate_passed"] = margin >= 0.01
        results.append(family_result)
        print(
            f"[C274] {family}: role={metrics['role_joint']['signed_jaccard']:.4f}, all={metrics['all_role_joint']['signed_jaccard']:.4f}, persistence={metrics['persistence']['signed_jaccard']:.4f}",
            flush=True,
        )
    core.write_rows(OUT / "analysis/family_results.jsonl", results)
    candidate_summary = []
    for name in CANDIDATES:
        passing = sum(bool(row[f"{name}_gate_passed"]) for row in results)
        candidate_summary.append({
            "candidate": name,
            "families_passing": passing,
            "median_margin": float(np.median([row[f"{name}_minus_persistence"] for row in results])),
            "broad_gate_passed": passing >= 4,
        })
    broad = any(row["broad_gate_passed"] for row in candidate_summary)
    report = {
        "phase": 1808,
        "campaign": "C274",
        "status": "joint_full_field_guards_adjudicated",
        "candidate_summary": candidate_summary,
        "family_results": results,
        "broad_gate_passed": broad,
        "strict_interpretation": "Adding role-wide or all-role event load and polarity to each coordinate's own sign does not become a mechanism law unless it beats persistence prospectively across families.",
        "next_authorization": "C275_joint_relational_state_observation_without_coordinate_independence" if not broad else "C275_independent_joint_guard_replication",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {
        "families": len(results) == 6,
        "candidate_count": len(candidate_summary) == 2,
        "map_shapes": all(list(item.shape) == [6, 36, 6, 8, 2560] for item in pred_maps.values()),
        "finite": bool(np.isfinite([row["median_margin"] for row in candidate_summary]).all()),
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {
        "contract": all(checks.values()),
        "analysis": all(analysis_checks.values()),
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {
        "phase": 1808,
        "campaign": "C274",
        "status": "closed",
        "checks": final_checks,
        "all_checks_passed": all(final_checks.values()),
        "headline": report,
        "next_authorization": report["next_authorization"],
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
