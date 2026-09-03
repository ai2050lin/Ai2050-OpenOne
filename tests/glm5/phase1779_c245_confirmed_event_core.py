#!/usr/bin/env python3
"""C245: extract the full-coordinate event core shared by C237 and independent C244."""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.RESULT / "phase1779_c245_confirmed_event_core"
C236 = common.OUTS["C236"]
C237 = common.OUTS["C237"]
C244 = common.RESULT / "phase1778_c244_independent_event_replication"


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C244 / "audit/independent_final_audit.json")
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C245"), "old_rule": (C237 / "analysis/rule_codes.int8.npy").is_file(), "new_full_field": (C244 / "raw/full_fields.float16.npy").is_file(), "all_coordinates": common.DIM == 2560}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1779, "campaign": "C245", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmed_event_core_observation_frozen",
        "operation": "intersect C237 discovery rule with independently stable same-sign C244 events",
        "independent_stability": {"active_prevalence_min": 0.75, "dominant_sign_min": 0.80},
        "axes": ["family", "effect", "checkpoint", "role", "all_2560_physical_coordinates"],
        "outputs": ["confirmed_rule_codes", "formation_clock", "cross_family_overlap", "checkpoint_persistence"],
        "claim_boundary": "This is a two-material observational intersection, not a minimal, causal, or context-free code.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "derive_once_without_new_model_or_refit",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def derive_new_events() -> tuple[np.memmap, list[dict]]:
    fields = np.load(C244 / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(C244 / "raw/hidden_index.jsonl")
    key = {(row["family"], row["surface"], row["unit"], row["factor_a"], row["factor_b"], row["order"]): row for row in index}
    thresholds = np.asarray(core.load(C236 / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    groups = []
    for family, surface, unit, order in itertools.product(common.FAMILIES, ("field_memo", "public_hearing"), range(6), (1, -1)):
        if (surface == "field_memo") != (unit < 3):
            continue
        groups.append({"event_index": len(groups), "family": family, "surface": surface, "unit": unit, "order": order})
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    events = np.lib.format.open_memmap(OUT / "raw/c244_role_events.int8.npy", mode="w+", dtype=np.int8, shape=(60, 3, 37, 6, 2560))
    for group in groups:
        cells = {}
        for a, b in itertools.product((0, 1), repeat=2):
            row = key[(group["family"], group["surface"], group["unit"], a, b, group["order"])]
            state = np.asarray(fields[row["hidden_index"]], np.float32)
            aligned = np.empty((37, 6, 2560), np.float32)
            for role_i, role in enumerate(common.ROLES):
                aligned[:, role_i] = state[:, row["role_positions"][role], :].mean(axis=1)
            cells[(a, b)] = aligned
        effects = common.factorial_effect(cells)
        events[group["event_index"]] = np.where(effects > thresholds[None, :, None, None], 1, np.where(effects < -thresholds[None, :, None, None], -1, 0)).astype(np.int8)
        if group["event_index"] % 10 == 0 or group["event_index"] == 59:
            events.flush(); print(f"[C245] C244 events {group['event_index'] + 1}/60", flush=True)
    core.write_rows(OUT / "raw/event_groups.jsonl", groups)
    return events, groups


def analyze() -> None:
    if (OUT / "analysis/confirmed_rule_codes.int8.npy").exists():
        raise RuntimeError("already analyzed")
    events, groups = derive_new_events()
    old = np.load(C237 / "analysis/rule_codes.int8.npy", mmap_mode="r")
    confirmed = np.zeros_like(old, dtype=np.int8)
    family_rows = []
    clock_rows = []
    for family_i, family in enumerate(common.FAMILIES):
        selected = [row["event_index"] for row in groups if row["family"] == family]
        current = np.asarray(events[selected])
        for effect_i, effect in enumerate(common.EFFECTS):
            up = np.mean(current[:, effect_i] == 1, axis=0)
            down = np.mean(current[:, effect_i] == -1, axis=0)
            active = up + down
            dominant_sign = np.where(up >= down, 1, -1)
            stable = (active >= 0.75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= 0.80)
            same = stable & (old[family_i, effect_i] == dominant_sign) & (old[family_i, effect_i] != 0)
            confirmed[family_i, effect_i] = np.where(same, dominant_sign, 0)
            old_count = int(np.count_nonzero(old[family_i, effect_i]))
            count = int(np.count_nonzero(same))
            family_rows.append({"family": family, "effect": effect, "old_rule_events": old_count, "confirmed_events": count, "retention": count / max(old_count, 1), "embedding_events": int(np.count_nonzero(same[0])), "hidden_events": int(np.count_nonzero(same[1:]))})
            for role_i, role in enumerate(common.ROLES):
                counts = np.count_nonzero(same[:, role_i], axis=-1)
                hits = np.flatnonzero(counts)
                clock_rows.append({"family": family, "effect": effect, "role": role, "first_checkpoint": int(hits[0]) if hits.size else -1, "peak_checkpoint": int(np.argmax(counts)) if hits.size else -1, "peak_coordinates": int(counts.max()), "total_events": int(counts.sum())})
        print(f"[C245] confirmed {family}", flush=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "analysis/confirmed_rule_codes.int8.npy", confirmed)
    core.write_rows(OUT / "analysis/family_effect_summary.jsonl", family_rows)
    core.write_rows(OUT / "analysis/formation_clock.jsonl", clock_rows)
    overlaps = []
    flattened = {family: confirmed[i].reshape(-1) for i, family in enumerate(common.FAMILIES)}
    for i, left in enumerate(common.FAMILIES):
        for right in common.FAMILIES[i + 1:]:
            a, b = flattened[left], flattened[right]
            union = (a != 0) | (b != 0)
            overlaps.append({"families": [left, right], "signed_jaccard": float(np.mean(a[union] == b[union])) if union.any() else 1.0, "same_signed_events": int(np.count_nonzero((a == b) & (a != 0)))})
    core.write_rows(OUT / "analysis/cross_family_overlap.jsonl", overlaps)
    persistence = []
    for family_i, family in enumerate(common.FAMILIES):
        for effect_i, effect in enumerate(common.EFFECTS):
            for q in range(36):
                left, right = confirmed[family_i, effect_i, q], confirmed[family_i, effect_i, q + 1]
                source = left != 0
                persistence.append({"family": family, "effect": effect, "from_checkpoint": q, "to_checkpoint": q + 1, "source_events": int(source.sum()), "same_coordinate_sign_persistence": float(np.mean(right[source] == left[source])) if source.any() else None})
    core.write_rows(OUT / "analysis/checkpoint_persistence.jsonl", persistence)
    total = int(np.count_nonzero(confirmed))
    report = {
        "phase": 1779, "campaign": "C245", "status": "confirmed_event_core_observed",
        "confirmed_events": total,
        "old_rule_events": int(np.count_nonzero(old)),
        "overall_retention": total / max(int(np.count_nonzero(old)), 1),
        "embedding_events": int(np.count_nonzero(confirmed[:, :, 0])),
        "hidden_events": int(np.count_nonzero(confirmed[:, :, 1:])),
        "family_effect_summary": family_rows,
        "cross_family_signed_jaccard_median": float(np.median([row["signed_jaccard"] for row in overlaps])),
        "same_checkpoint_persistence_median": float(np.median([row["same_coordinate_sign_persistence"] for row in persistence if row["same_coordinate_sign_persistence"] is not None])),
        "strict_interpretation": "A confirmed event survived discovery stability and one independent controlled-material system with the same sign. It is still context- and threshold-indexed.",
        "next_authorization": "C246 third genuinely natural paraphrase panel with human naturalness audit before prospective reveal",
    }
    core.save(OUT / "analysis/summary.json", report)
    checks = {"events": events.shape == (60, 3, 37, 6, 2560), "confirmed": confirmed.shape == (5, 3, 37, 6, 2560), "family_rows": len(family_rows) == 15, "clock_rows": len(clock_rows) == 90, "overlaps": len(overlaps) == 10, "persistence": len(persistence) == 540, "alphabet": set(np.unique(confirmed).tolist()) <= {-1, 0, 1}}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1779, "campaign": "C245", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
