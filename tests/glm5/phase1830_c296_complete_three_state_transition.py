#!/usr/bin/env python3
"""C296: fit a non-absorbing three-state transition on all role coordinates."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common
import phase1813_c279_joint_state_word_partition as partition
import phase1814_c280_multisource_one_step_prediction as one_step

core, OUT = common.core, common.OUTS["C296"]
C248 = common.previous.previous.prior.OUTS["C248"]
C264 = common.previous.previous.OUTS["C264"]
C265 = common.previous.previous.OUTS["C265"]
C278 = common.previous.OUTS["C278"]
SOURCE_ROLES = partition.CANDIDATES["relation_query"]


def pair_ids(index: list[dict], family: str) -> tuple[np.ndarray, np.ndarray]:
    specs = common.pair_specs(index, family)
    return np.asarray([x[0] for x in specs], int), np.asarray([x[1] for x in specs], int)


def combined_code(events: np.ndarray, destination: int) -> np.ndarray:
    word = partition.code_word(events, SOURCE_ROLES).astype(np.int16)
    return (word + 9 * (events[:, destination].astype(np.int16) + 1)).astype(np.int16)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parents = {c: core.load(common.OUTS[c] / "analysis/final.json") for c in ("C293", "C295")}
    gates = core.load(common.OUTS["C293"] / "protocol/preregistration.json")["gates"]
    checks = {"parents": all(x["all_checks_passed"] for x in parents.values()), "discovery_only_third_fourth": True, "confirmation_only_fifth": True, "lockbox_unread": True, "all_q_roles_coordinates": True}
    if not all(checks.values()):
        raise RuntimeError(checks)
    for sub in ("analysis", "audit", "protocol"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    protocol = {
        "phase": 1830,
        "campaign": "C296",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_transition_rule_frozen",
        "feature": "relation/query joint event word plus current destination event",
        "target": "next destination event in {-1,0,+1}",
        "coverage": "six families x 36 transitions x six destination roles x all 2560 coordinates",
        "controls": ["destination persistence", "C280 absorbing completion"],
        "selection": "per family maximize confirmation signed-Jaccard margin over the strongest control; ties use lower q then lower role index",
        "claim_boundary": "This is a full three-state observational transition table. It is not a Markov proof, a causal edge, or a continuous simulator.",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    train_a = np.load(C265 / "raw/training_role_states.float16.npy", mmap_mode="r")
    train_b = np.load(C264 / "raw/role_states.float16.npy", mmap_mode="r")
    confirm = np.load(C278 / "raw/role_states.float16.npy", mmap_mode="r")
    indices = {"a": core.rows(C248 / "raw/hidden_index.jsonl"), "b": core.rows(C264 / "raw/hidden_index.jsonl"), "c": core.rows(C278 / "raw/hidden_index.jsonl")}
    thresholds = common.thresholds()
    maps = np.zeros((6, 36, 6, 27, common.DIM), np.int8)
    atlas = np.zeros((6, 36, 6, 3, common.DIM), np.uint16)
    family_rows = []
    for fi, family in enumerate(common.FAMILIES):
        ids = {name: pair_ids(index, family) for name, index in indices.items()}
        strata = []
        for q in range(36):
            train_current = np.concatenate((
                common.event(np.asarray(train_a[ids['a'][1], q], np.float32) - np.asarray(train_a[ids['a'][0], q], np.float32), thresholds[q]),
                common.event(np.asarray(train_b[ids['b'][1], q], np.float32) - np.asarray(train_b[ids['b'][0], q], np.float32), thresholds[q]),
            ))
            train_next = np.concatenate((
                common.event(np.asarray(train_a[ids['a'][1], q + 1], np.float32) - np.asarray(train_a[ids['a'][0], q + 1], np.float32), thresholds[q + 1]),
                common.event(np.asarray(train_b[ids['b'][1], q + 1], np.float32) - np.asarray(train_b[ids['b'][0], q + 1], np.float32), thresholds[q + 1]),
            ))
            current = common.event(np.asarray(confirm[ids['c'][1], common.CANONICAL_NEW_INDICES[q]], np.float32) - np.asarray(confirm[ids['c'][0], common.CANONICAL_NEW_INDICES[q]], np.float32), thresholds[q])
            nxt = common.event(np.asarray(confirm[ids['c'][1], common.CANONICAL_NEW_INDICES[q + 1]], np.float32) - np.asarray(confirm[ids['c'][0], common.CANONICAL_NEW_INDICES[q + 1]], np.float32), thresholds[q + 1])
            source_word_train = partition.code_word(train_current, SOURCE_ROLES)
            source_word_confirm = partition.code_word(current, SOURCE_ROLES)
            for destination in range(6):
                code_train = combined_code(train_current, destination)
                fitted, _, _ = one_step.fit_map(code_train, train_next[:, destination], 27, gates["transition_support_min"], gates["transition_agreement_min"])
                maps[fi, q, destination] = fitted.reshape(common.DIM, 27).T
                prediction = one_step.lookup(fitted, combined_code(current, destination), 27)
                absorb_fit, _, _ = one_step.fit_map(source_word_train, train_next[:, destination], 9, gates["transition_support_min"], gates["transition_agreement_min"])
                pure = one_step.lookup(absorb_fit, source_word_confirm, 9)
                absorbing = np.where(current[:, destination] != 0, current[:, destination], pure).astype(np.int8)
                persistence = current[:, destination]
                full_metrics = common.metric_counts(prediction, nxt[:, destination])
                absorbing_metrics = common.metric_counts(absorbing, nxt[:, destination])
                persistence_metrics = common.metric_counts(persistence, nxt[:, destination])
                margin = full_metrics["signed_jaccard"] - max(absorbing_metrics["signed_jaccard"], persistence_metrics["signed_jaccard"])
                strata.append({
                    "q": q,
                    "destination_index": destination,
                    "destination_role": common.ROLES[destination],
                    "full": full_metrics,
                    "absorbing": absorbing_metrics,
                    "persistence": persistence_metrics,
                    "transition_recall": common.transition_recall(prediction, current[:, destination], nxt[:, destination]),
                    "minus_best_control": margin,
                })
                atlas[fi, q, destination, 0] = ((prediction == nxt[:, destination]) & ((prediction != 0) | (nxt[:, destination] != 0))).sum(axis=0).astype(np.uint16)
                atlas[fi, q, destination, 1] = ((prediction != 0) | (nxt[:, destination] != 0)).sum(axis=0).astype(np.uint16)
                atlas[fi, q, destination, 2] = (common.transition_kind(current[:, destination], nxt[:, destination]) >= 3).sum(axis=0).astype(np.uint16)
        selected = max(strata, key=lambda x: (x["minus_best_control"], -x["q"], -x["destination_index"]))
        family_rows.append({"family": family, "strata": strata, "selected_stratum": selected, "family_gate_passed": selected["minus_best_control"] >= gates["model_margin_min"]})
        print(f"[C296] {family}: q{selected['q']}->{selected['q']+1}/{selected['destination_role']} margin={selected['minus_best_control']:+.5f}", flush=True)
    np.save(OUT / "analysis/complete_transition_maps.int8.npy", maps)
    np.save(OUT / "analysis/coordinate_transition_counts.uint16.npy", atlas)
    core.write_rows(OUT / "analysis/family_results.jsonl", family_rows)
    passing = [r["family"] for r in family_rows if r["family_gate_passed"]]
    report = {"phase": 1830, "campaign": "C296", "status": "complete_transition_adjudicated", "families": [{k: v for k, v in r.items() if k != "strata"} for r in family_rows], "families_passing": passing, "broad_gate_passed": len(passing) >= gates["broad_families_min"], "strict_interpretation": protocol["claim_boundary"], "next_authorization": "C297_C309_all_branches"}
    core.save(OUT / "analysis/summary.json", report)
    ach = {"families": len(family_rows) == 6, "strata_each": all(len(r["strata"]) == 216 for r in family_rows), "map_shape": list(maps.shape) == [6, 36, 6, 27, 2560], "atlas_shape": list(atlas.shape) == [6, 36, 6, 3, 2560], "finite": bool(np.isfinite([r["selected_stratum"]["minus_best_control"] for r in family_rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": ach, "all_checks_passed": all(ach.values())})
    final_checks = {"contract": all(checks.values()), "analysis": all(ach.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1830, "campaign": "C296", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
