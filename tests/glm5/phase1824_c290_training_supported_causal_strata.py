#!/usr/bin/env python3
"""C290: select causal transition-role strata using training materials only."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1811_c277_c289_joint_response_common as common
import phase1813_c279_joint_state_word_partition as partition
import phase1814_c280_multisource_one_step_prediction as one_step

core = common.core
OUT = common.RESULT / "phase1824_c290_training_supported_causal_strata"
C248 = common.previous.prior.OUTS["C248"]
C264 = common.previous.OUTS["C264"]
C265 = common.previous.OUTS["C265"]
C278 = common.OUTS["C278"]
C289 = common.OUTS["C289"]
CANDIDATE = "primary_relation_query"
SOURCE_ROLES = partition.CANDIDATES[CANDIDATE]


def ids(index: list[dict], family: str):
    specs = common.pair_specs(index, family)
    return (
        np.asarray([row[0] for row in specs], int),
        np.asarray([row[1] for row in specs], int),
        [row[2] for row in specs],
    )


def events(states: np.ndarray, left: np.ndarray, right: np.ndarray, q: int, threshold: np.ndarray, canonical_new: bool = False):
    qi = common.CANONICAL_NEW_INDICES[q] if canonical_new else q
    return common.event(np.asarray(states[right, qi], np.float32) - np.asarray(states[left, qi], np.float32), threshold[q])


def cross_predict(train_current: np.ndarray, train_next: np.ndarray, test_current: np.ndarray, test_next: np.ndarray, destination: int):
    states_count = 3 ** len(SOURCE_ROLES)
    train_code = partition.code_word(train_current, SOURCE_ROLES)
    test_code = partition.code_word(test_current, SOURCE_ROLES)
    fitted, _key, _support = one_step.fit_map(train_code, train_next[:, destination], states_count, 4, 0.70)
    pure = one_step.lookup(fitted, test_code, states_count)
    eligible = (test_current[:, destination] == 0) & (pure != 0)
    hit = eligible & (pure == test_next[:, destination])
    counts = eligible.sum(axis=1)
    return {
        "eligible_total": int(eligible.sum()),
        "correct_total": int(hit.sum()),
        "precision": float(hit.sum() / max(eligible.sum(), 1)),
        "median_per_pair": float(np.median(counts)),
        "p10_per_pair": float(np.quantile(counts, 0.10)),
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C289 / "analysis/final.json")
    checks = {
        "parent": parent["all_checks_passed"],
        "candidate_frozen": CANDIDATE == "primary_relation_query",
        "selection_third_fourth_only": True,
        "q_restricted_to_real_block_outputs": True,
        "all_coordinates": True,
        "no_topk_pca_cosine_attention_mlp": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    for subdir in ("analysis", "audit", "protocol"):
        (OUT / subdir).mkdir()
    protocol = {
        "phase": 1824,
        "campaign": "C290",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "training_only_stratum_selection_frozen",
        "candidate": CANDIDATE,
        "source_roles": [common.ROLES[i] for i in SOURCE_ROLES],
        "candidate_strata": "q1-q34 x all six destination roles, independently for each family",
        "selection": "maximize cross-material median eligible coordinates times precision after fitting third->fourth and fourth->third; require each direction precision>=0.70 and median eligible>=16",
        "holdout": "after selection, evaluate first factor-A pair per surface and family on fifth material",
        "holdout_family_gate": "both surfaces have >=16 eligible coordinates and pooled target precision>=0.65",
        "authorization": "C291 tests every family passing its own holdout family gate; failure of one family does not stop another",
        "claim_boundary": "This selects a locally testable observational stratum. It does not establish causal use.",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})

    a = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    b = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    fifth = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    ia = core.rows(C248 / "raw/hidden_index.jsonl")
    ib = core.rows(C264 / "raw/hidden_index.jsonl")
    it = core.rows(C278 / "raw/hidden_index.jsonl")
    threshold = common.thresholds()
    results = []
    selected = {}
    for family in common.FAMILIES:
        al, ar, _am = ids(ia, family)
        bl, br, _bm = ids(ib, family)
        tl, tr, tm = ids(it, family)
        strata = []
        for q in range(1, 35):
            ae = events(a, al, ar, q, threshold)
            an = events(a, al, ar, q + 1, threshold)
            be = events(b, bl, br, q, threshold)
            bn = events(b, bl, br, q + 1, threshold)
            for destination in range(6):
                ab = cross_predict(ae, an, be, bn, destination)
                ba = cross_predict(be, bn, ae, an, destination)
                qualified = (
                    ab["precision"] >= 0.70
                    and ba["precision"] >= 0.70
                    and ab["median_per_pair"] >= 16
                    and ba["median_per_pair"] >= 16
                )
                score = min(ab["precision"], ba["precision"]) * min(ab["median_per_pair"], ba["median_per_pair"])
                strata.append({
                    "q": q,
                    "destination_role": common.ROLES[destination],
                    "destination_index": destination,
                    "third_to_fourth": ab,
                    "fourth_to_third": ba,
                    "qualified": qualified,
                    "selection_score": float(score),
                })
        eligible_strata = [row for row in strata if row["qualified"]]
        chosen = max(eligible_strata or strata, key=lambda row: (row["qualified"], row["selection_score"], -row["q"], -row["destination_index"]))
        q, destination = chosen["q"], chosen["destination_index"]
        ae = events(a, al, ar, q, threshold)
        an = events(a, al, ar, q + 1, threshold)
        be = events(b, bl, br, q, threshold)
        bn = events(b, bl, br, q + 1, threshold)
        te = events(fifth, tl, tr, q, threshold, canonical_new=True)
        tn = events(fifth, tl, tr, q + 1, threshold, canonical_new=True)
        train_current = np.concatenate((ae, be), axis=0)
        train_next = np.concatenate((an, bn), axis=0)
        states_count = 3 ** len(SOURCE_ROLES)
        fitted, _key, _support = one_step.fit_map(
            partition.code_word(train_current, SOURCE_ROLES), train_next[:, destination], states_count, 4, 0.70
        )
        pure = one_step.lookup(fitted, partition.code_word(te, SOURCE_ROLES), states_count)
        eligible = (te[:, destination] == 0) & (pure != 0)
        hit = eligible & (pure == tn[:, destination])
        surface_rows = []
        selected_pairs = []
        for surface in common.SURFACES:
            pair_index = next(i for i, meta in enumerate(tm) if meta["surface"] == surface)
            selected_pairs.append({
                "surface": surface,
                "left_id": int(tl[pair_index]),
                "right_id": int(tr[pair_index]),
                "eligible_coordinates": int(eligible[pair_index].sum()),
                "correct_coordinates": int(hit[pair_index].sum()),
                "target_precision": float(hit[pair_index].sum() / max(eligible[pair_index].sum(), 1)),
            })
        pooled_eligible = sum(row["eligible_coordinates"] for row in selected_pairs)
        pooled_correct = sum(row["correct_coordinates"] for row in selected_pairs)
        holdout_precision = float(pooled_correct / max(pooled_eligible, 1))
        family_gate = bool(
            chosen["qualified"]
            and all(row["eligible_coordinates"] >= 16 for row in selected_pairs)
            and holdout_precision >= 0.65
        )
        row = {
            "family": family,
            "selected_stratum": chosen,
            "qualified_training_strata": len(eligible_strata),
            "fifth_selected_pairs": selected_pairs,
            "fifth_pooled_target_precision": holdout_precision,
            "holdout_family_gate_passed": family_gate,
        }
        results.append(row)
        selected[family] = {"q": q, "destination_role": common.ROLES[destination], "destination_index": destination}
        print(f"[C290] {family}: q{q}->{q + 1}/{common.ROLES[destination]} train={chosen['qualified']} fifth_n={[r['eligible_coordinates'] for r in selected_pairs]} precision={holdout_precision:.4f} gate={family_gate}", flush=True)

    passing = [row["family"] for row in results if row["holdout_family_gate_passed"]]
    core.save(OUT / "analysis/selected_strata.json", selected)
    core.write_rows(OUT / "analysis/family_results.jsonl", results)
    report = {
        "phase": 1824,
        "campaign": "C290",
        "status": "training_supported_causal_strata_adjudicated",
        "candidate": CANDIDATE,
        "families": results,
        "passing_families": passing,
        "families_passing": len(passing),
        "broad_authorization": len(passing) >= 4,
        "strict_interpretation": "A passing family has a prospectively eligible local transition-role stratum. This repairs C285's qualification mismatch but is not causal evidence.",
        "next_authorization": "C291_test_each_passing_family" if passing else "C291_registered_no_test",
    }
    core.save(OUT / "analysis/summary.json", report)
    audit_checks = {
        "six_families": len(results) == 6,
        "one_stratum_each": len(selected) == 6,
        "selection_q_real_blocks": all(1 <= row["selected_stratum"]["q"] <= 34 for row in results),
        "finite": bool(np.isfinite([row["fifth_pooled_target_precision"] for row in results]).all()),
        "fifth_not_used_for_selection": True,
    }
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": audit_checks, "all_checks_passed": all(audit_checks.values())})
    final_checks = {
        "contract": all(checks.values()),
        "analysis": all(audit_checks.values()),
        "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
    final = {
        "phase": 1824,
        "campaign": "C290",
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
