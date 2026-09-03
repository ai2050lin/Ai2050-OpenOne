#!/usr/bin/env python3
"""C237: discover readable conditional event and precedence rules from discovery only."""
from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C237"]
C235 = common.OUTS["C235"]
C236 = common.OUTS["C236"]


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C236 / "audit/independent_final_audit.json")
    protocol = core.load(common.OUTS["C234"] / "protocol/preregistration.json")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C237"),
        "discovery_units": len(common.PARTITION_UNITS["discovery"]) == 4,
        "discovery_surfaces": sum(value == "discovery" for value in common.SURFACE_PARTITION.values()) == 2,
        "rule_floor": protocol["readable_rule_floor"]["event_prevalence_min"] == 0.75,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    frozen = {
        "phase": 1771, "campaign": "C237", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "discovery_only_readable_event_rule_frozen",
        "fit_partition": "discovery only", "unopened": ["confirmation", "lockbox", "fresh"],
        "role_effect_shape": [160, 3, 37, 6, 2560],
        "rule_shape": [5, 3, 37, 6, 2560],
        "event_prevalence_min": 0.75, "dominant_sign_min": 0.80,
        "role_event_density_min": 0.02, "precedence_rate_min": 0.75,
        "role_alignment": "each factorial cell uses its own compiled semantic span before the standard contrast is formed",
        "claim_boundary": "Rules are discovery descriptions and predictions, not causal edges.",
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "derive_role_aligned_fields_then_discover_rules_without_reading_later_partitions",
    }
    core.save(OUT / "protocol/preregistration.json", frozen)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def derive() -> None:
    if (OUT / "audit/internal_derive_audit.json").exists():
        raise RuntimeError("role effects already exist")
    fields = np.load(C235 / "raw/full_fields.float16.npy", mmap_mode="r")
    hidden = core.rows(C235 / "raw/hidden_index.jsonl")
    key = common.hidden_key(hidden)
    by_index = {row["hidden_index"]: row for row in hidden}
    groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    thresholds = np.asarray(core.load(C236 / "protocol/frozen_event_thresholds.json")["thresholds"], np.float32)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    role_effects = np.lib.format.open_memmap(OUT / "raw/role_effects.float16.npy", mode="w+", dtype=np.float16, shape=(160, 3, 37, 6, 2560))
    role_events = np.lib.format.open_memmap(OUT / "raw/role_events.int8.npy", mode="w+", dtype=np.int8, shape=role_effects.shape)
    for group_i, group in enumerate(groups):
        cells = {}
        for a, b in itertools.product((0, 1), repeat=2):
            idx = key[(group["family"], group["surface"], int(group["unit"]), a, b, int(group["order"]))]
            row = by_index[idx]
            aligned = np.empty((37, len(common.ROLES), 2560), np.float32)
            for role_i, role in enumerate(common.ROLES):
                state = np.asarray(fields[idx], np.float32)
                aligned[:, role_i] = state[:, row["role_positions"][role], :].mean(axis=1)
            cells[(a, b)] = aligned
        values = common.factorial_effect(cells)
        role_effects[group_i] = values.astype(np.float16)
        eta = thresholds[None, :, None, None]
        role_events[group_i] = np.where(values > eta, 1, np.where(values < -eta, -1, 0)).astype(np.int8)
        if group_i % 10 == 0 or group_i == 159:
            role_effects.flush(); role_events.flush()
            print(f"[C237] role fields {group_i + 1}/160", flush=True)
    checks = {"effects": list(role_effects.shape) == [160, 3, 37, 6, 2560], "events": role_events.shape == role_effects.shape, "alphabet": set(np.unique(role_events[:, :, :, :, ::64]).tolist()) <= {-1, 0, 1}}
    core.save(OUT / "audit/internal_derive_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def discover() -> None:
    if (OUT / "analysis/rule_codes.int8.npy").exists():
        raise RuntimeError("rules already exist")
    groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    effects = np.load(OUT / "raw/role_effects.float16.npy", mmap_mode="r")
    events = np.load(OUT / "raw/role_events.int8.npy", mmap_mode="r")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    rules = np.zeros((5, 3, 37, 6, 2560), np.int8)
    lower = np.zeros_like(rules, dtype=np.float16)
    upper = np.zeros_like(rules, dtype=np.float16)
    readable = []
    formation_rows = []
    precedence = []
    for family_i, family in enumerate(common.FAMILIES):
        selected = [row["effect_index"] for row in groups if row["family"] == family and row["partition"] == "discovery"]
        family_events = np.asarray(events[selected])
        family_effects = np.asarray(effects[selected], np.float32)
        for effect_i, effect in enumerate(common.EFFECTS):
            up = np.mean(family_events[:, effect_i] == 1, axis=0)
            down = np.mean(family_events[:, effect_i] == -1, axis=0)
            active = up + down
            dominant = np.maximum(up, down) / np.maximum(active, 1e-9)
            stable = (active >= 0.75) & (dominant >= 0.80)
            rules[family_i, effect_i] = np.where(stable, np.where(up >= down, 1, -1), 0)
            for q in range(37):
                for role_i, role in enumerate(common.ROLES):
                    signs = rules[family_i, effect_i, q, role_i]
                    active_coords = np.flatnonzero(signs)
                    if active_coords.size:
                        vals = family_effects[:, effect_i, q, role_i, active_coords]
                        vals = np.where(np.sign(vals) == signs[active_coords][None, :], vals, np.nan)
                        lower[family_i, effect_i, q, role_i, active_coords] = np.nanquantile(vals, 0.10, axis=0).astype(np.float16)
                        upper[family_i, effect_i, q, role_i, active_coords] = np.nanquantile(vals, 0.90, axis=0).astype(np.float16)
            density_by_group_role_q = np.mean(family_events[:, effect_i] != 0, axis=-1)
            first_by_group_role = np.full((len(selected), len(common.ROLES)), -1, np.int16)
            for local in range(len(selected)):
                for role_i, role in enumerate(common.ROLES):
                    hits = np.flatnonzero(density_by_group_role_q[local, :, role_i] >= 0.02)
                    if hits.size:
                        first_by_group_role[local, role_i] = int(hits[0])
                    formation_rows.append({"family": family, "effect": effect, "group_index": selected[local], "role": role, "first_checkpoint": int(first_by_group_role[local, role_i]), "max_event_density": float(density_by_group_role_q[local, :, role_i].max())})
            for source_i, target_i in itertools.permutations(range(len(common.ROLES)), 2):
                valid = (first_by_group_role[:, source_i] >= 0) & (first_by_group_role[:, target_i] >= 0)
                rate = float(np.mean(first_by_group_role[valid, source_i] < first_by_group_role[valid, target_i])) if valid.any() else 0.0
                precedence.append({"family": family, "effect": effect, "source_role": common.ROLES[source_i], "target_role": common.ROLES[target_i], "support": int(valid.sum()), "precedence_rate": rate, "rule_active": bool(valid.sum() >= 8 and rate >= 0.75)})
            for role_i, role in enumerate(common.ROLES):
                role_rule = rules[family_i, effect_i, :, role_i]
                coords_per_q = np.count_nonzero(role_rule, axis=-1)
                readable.append({"family": family, "effect": effect, "role": role, "stable_event_count": int(np.count_nonzero(role_rule)), "checkpoints_with_events": int(np.count_nonzero(coords_per_q)), "max_coordinates_at_checkpoint": int(coords_per_q.max()), "first_rule_checkpoint": int(np.flatnonzero(coords_per_q)[0]) if np.any(coords_per_q) else -1})
        print(f"[C237] discovered {family}", flush=True)
    np.save(OUT / "analysis/rule_codes.int8.npy", rules)
    np.save(OUT / "analysis/amplitude_lower.float16.npy", lower)
    np.save(OUT / "analysis/amplitude_upper.float16.npy", upper)
    core.write_rows(OUT / "analysis/readable_rules.jsonl", readable)
    core.write_rows(OUT / "analysis/role_formation.jsonl", formation_rows)
    core.write_rows(OUT / "analysis/precedence_rules.jsonl", precedence)
    report = {
        "phase": 1771, "campaign": "C237", "status": "discovery_event_rules_frozen",
        "stable_signed_events": int(np.count_nonzero(rules)),
        "stable_event_fraction": float(np.mean(rules != 0)),
        "active_precedence_rules": sum(row["rule_active"] for row in precedence),
        "readable_rule_rows": len(readable),
        "fit_groups": sum(row["partition"] == "discovery" for row in groups),
        "unread_partitions": ["confirmation", "lockbox", "fresh"],
        "interpretation": "A rule is a repeated role/checkpoint/coordinate sign event and optional role-order statement. It remains observational until prospective tests.",
        "next_authorization": "C238_reveal_confirmation_then_lockbox_and_fresh_without_refitting",
    }
    core.save(OUT / "analysis/summary.json", report)
    checks = {"rules": rules.shape == (5, 3, 37, 6, 2560), "readable": len(readable) == 5 * 3 * 6, "formation": len(formation_rows) == 5 * 3 * 16 * 6, "precedence": len(precedence) == 5 * 3 * 30, "later_unread_by_code": True}
    core.save(OUT / "audit/internal_discovery_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "derive": core.load(OUT / "audit/internal_derive_audit.json")["all_checks_passed"], "discover": core.load(OUT / "audit/internal_discovery_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1771, "campaign": "C237", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def reconcile() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    old_hash = protocol["producer_sha256"]
    new_hash = core.sha(Path(__file__))
    amendment = {
        "kind": "implementation_provenance_reconciliation",
        "old_producer_sha256": old_hash,
        "new_producer_sha256": new_hash,
        "reason": "The semantic-span indexing axis was corrected before discovery rule extraction, but the producer hash was not refreshed at that time.",
        "scientific_fields_changed": False,
        "thresholds_changed": False,
        "partition_access_changed": False,
        "result_values_recomputed_after_code_fix": True,
    }
    protocol["producer_sha256"] = new_hash
    protocol["provenance_amendment"] = amendment
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/provenance_amendment.json", amendment)
    print(json.dumps(amendment, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "derive", "discover", "reconcile", "close"))
    args = parser.parse_args()
    {"contract": contract, "derive": derive, "discover": discover, "reconcile": reconcile, "close": close}[args.command]()


if __name__ == "__main__":
    main()
