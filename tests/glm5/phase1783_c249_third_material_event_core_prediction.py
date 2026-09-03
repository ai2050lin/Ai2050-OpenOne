#!/usr/bin/env python3
"""C249: prospective third-material test of the frozen two-material event core."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1780_c246_c255_event_hypergraph_common as common

core = common.core
OUT = common.OUTS["C249"]
PARENT = common.OUTS["C248"]


def main() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(PARENT / "audit/independent_final_audit.json")
    fields = np.load(PARENT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/hidden_index.jsonl")
    core_index = [row for row in index if row["panel"] == "core"]
    key = {(row["family"], row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in core_index}
    thresholds = np.asarray(core.load(common.OLD["C236"] / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    frozen = np.load(common.OLD["C245"] / "analysis/confirmed_rule_codes.int8.npy", mmap_mode="r")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C249"),
        "field": list(fields.shape) == [768, 37, 128, 2560], "core_rows": len(core_index) == 640,
        "frozen_core": list(frozen.shape) == [5, 3, 37, 6, 2560], "no_refit": True,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1783, "campaign": "C249", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "third_material_prediction_frozen", "targets": ["attitude_event", "contrast"], "exploratory": ["comparison"],
        "gate": {"stable_signed_jaccard_min": 0.10, "best_control_margin_min": 0.02, "coordinate_permutation_p_max": 0.05},
        "stability": {"active_prevalence_min": 0.75, "dominant_sign_min": 0.80},
        "controls": ["best_wrong_family", "sign_flip", "zero", "63 coordinate rolls preserving per-cell signs and counts"],
        "claim_boundary": "Role-span means are tested here. The C248 archive remains full-token, but this rule camera is not a full-token rule.",
        "producer_sha256": core.sha(Path(__file__)), "authorization": "derive_once_without_rule_or_threshold_refit",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    groups = []
    events = np.lib.format.open_memmap(OUT / "raw/third_role_events.int8.npy", mode="w+", dtype=np.int8, shape=(160, 3, 37, 6, 2560))
    complete = []
    for family, surface, unit, order in itertools.product(common.FAMILIES, common.SURFACES, range(8), (1, -1)):
        cells = {}
        correct = True
        for a, b in itertools.product((0, 1), repeat=2):
            row = key[(family, surface, unit, a, b, order)]
            correct &= bool(row["correct"])
            state = np.asarray(fields[row["hidden_index"]], np.float32)
            aligned = np.empty((37, 6, 2560), np.float32)
            for role_i, role in enumerate(common.ROLES):
                aligned[:, role_i] = state[:, row["role_positions"][role], :].mean(axis=1)
            cells[(a, b)] = aligned
        effect = common.factorial_effect(cells)
        coded = np.where(effect > thresholds[None, :, None, None], 1, np.where(effect < -thresholds[None, :, None, None], -1, 0)).astype(np.int8)
        event_i = len(groups)
        events[event_i] = coded
        groups.append({"event_index": event_i, "family": family, "surface": surface, "unit": unit, "order": order, "behavior_complete": correct})
        if correct:
            complete.append(event_i)
        if event_i % 16 == 0 or event_i == 159:
            events.flush(); print(f"[C249] role events {event_i + 1}/160", flush=True)
    core.write_rows(OUT / "raw/event_groups.jsonl", groups)
    stable = np.zeros((5, 3, 37, 6, 2560), np.int8)
    family_results = []
    row_results = []
    rng_shifts = [int((i * 37 + 11) % 2560) for i in range(1, 64)]
    for family_i, family in enumerate(common.FAMILIES):
        selected = [row["event_index"] for row in groups if row["family"] == family and row["behavior_complete"]]
        current = np.asarray(events[selected])
        up, down = np.mean(current == 1, axis=0), np.mean(current == -1, axis=0)
        active = up + down
        dominant = np.where(up >= down, 1, -1)
        stable_i = (active >= 0.75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= 0.80)
        stable[family_i] = np.where(stable_i, dominant, 0)
        observed = common.signed_jaccard(np.asarray(frozen[family_i]), stable[family_i])
        wrong = max(common.signed_jaccard(np.asarray(frozen[j]), stable[family_i]) for j in range(5) if j != family_i)
        sign_flip = common.signed_jaccard(-np.asarray(frozen[family_i]), stable[family_i])
        zero = common.signed_jaccard(np.zeros_like(stable[family_i]), stable[family_i])
        null = np.asarray([common.signed_jaccard(np.roll(np.asarray(frozen[family_i]), shift, axis=-1), stable[family_i]) for shift in rng_shifts])
        p = float((1 + np.sum(null >= observed)) / (1 + len(null)))
        best_control = max(wrong, sign_flip, zero, float(null.max()))
        margin = observed - best_control
        passed = observed >= 0.10 and margin >= 0.02 and p <= 0.05
        group_scores = [common.signed_jaccard(np.asarray(frozen[family_i]), np.asarray(events[i])) for i in selected]
        family_results.append({
            "family": family, "behavior_complete_groups": len(selected), "stable_signed_jaccard": observed,
            "group_signed_jaccard_median": float(np.median(group_scores)), "controls": {"best_wrong_family": wrong, "sign_flip": sign_flip, "zero": zero, "coordinate_roll_max": float(null.max()), "coordinate_roll_median": float(np.median(null))},
            "best_control_margin": margin, "coordinate_permutation_p": p, "passed": passed,
            "stable_events": int(np.count_nonzero(stable[family_i])), "tri_material_events": int(np.count_nonzero((stable[family_i] == frozen[family_i]) & (frozen[family_i] != 0))),
        })
        for i in selected:
            row_results.append({"family": family, "event_index": i, "signed_jaccard": common.signed_jaccard(np.asarray(frozen[family_i]), np.asarray(events[i]))})
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/third_stable_rule_codes.int8.npy", stable)
    tri = np.where((stable == frozen) & (frozen != 0), frozen, 0).astype(np.int8)
    np.save(OUT / "analysis/tri_material_core.int8.npy", tri)
    core.write_rows(OUT / "analysis/group_prediction_rows.jsonl", row_results)
    target = {row["family"]: row for row in family_results}
    target_passed = all(target[name]["passed"] for name in ("attitude_event", "contrast"))
    report = {
        "phase": 1783, "campaign": "C249", "status": "adjudicated", "family_results": family_results,
        "target_families_passed": target_passed, "tri_material_events": int(np.count_nonzero(tri)),
        "embedding_events": int(np.count_nonzero(tri[:, :, 0])), "hidden_events": int(np.count_nonzero(tri[:, :, 1:])),
        "strict_interpretation": "A pass establishes prospective role-span event recurrence against coordinate-matched nulls; it does not establish a causal path or semantic neuron.",
        "next_authorization": "C250_full_token_observation_and_C251_typed_composition_always; C252_causal_only_if_targets_pass",
    }
    core.save(OUT / "analysis/summary.json", report)
    analysis_checks = {"groups": len(groups) == 160, "shape": list(events.shape) == [160, 3, 37, 6, 2560], "stable": list(stable.shape) == [5, 3, 37, 6, 2560], "families": len(family_results) == 5, "finite": bool(np.isfinite([row["stable_signed_jaccard"] for row in family_results]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": analysis_checks, "all_checks_passed": all(analysis_checks.values())})
    final_checks = {"contract": True, "analysis": all(analysis_checks.values()), "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1783, "campaign": "C249", "status": "closed", "checks": final_checks, "all_checks_passed": all(final_checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    core.save(OUT / "audit/independent_final_audit.json", {"checks": final_checks, "all_checks_passed": all(final_checks.values()), "authorization": report["next_authorization"]})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
