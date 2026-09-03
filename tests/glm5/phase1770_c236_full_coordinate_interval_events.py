#!/usr/bin/env python3
"""C236: derive full-coordinate factorial effects and interval events."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C236"]
PARENT = common.OUTS["C235"]


def groups() -> list[dict]:
    values = []
    for surface in common.SURFACES:
        partition = common.SURFACE_PARTITION[surface]
        for family, unit, order in __import__("itertools").product(common.FAMILIES, common.PARTITION_UNITS[partition], (1, -1)):
            values.append({"effect_index": len(values), "family": family, "surface": surface, "partition": partition, "unit": unit, "order": order})
    return values


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(PARENT / "audit/independent_final_audit.json")
    values = groups()
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C236"),
        "groups": len(values) == 160,
        "partitions": {row["partition"] for row in values} == set(common.PARTITIONS),
        "orders": {row["order"] for row in values} == {1, -1},
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "protocol/effect_groups.jsonl", values)
    protocol = {
        "phase": 1770,
        "campaign": "C236",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_coordinate_interval_event_extraction_frozen",
        "effect_shape": [160, 3, 37, 128, 2560],
        "effect_dtype": "float16",
        "event_shape": [160, 3, 37, 128, 2560],
        "event_dtype": "int8 {-1,0,+1}",
        "threshold_formula": "eta_q=max(4*duplicate_max,0.25*q75(abs(nonzero discovery effects at q)),1e-6)",
        "formation": "first nonzero event checkpoint and count of nonzero checkpoints for every group/effect/token/coordinate",
        "no_coordinate_selection": True,
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "extract_effects_then_freeze_thresholds_then_extract_events",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def extract() -> None:
    if (OUT / "raw/effects.float16.npy").exists():
        raise RuntimeError("effects already exist")
    fields = np.load(PARENT / "raw/full_fields.float16.npy", mmap_mode="r")
    index = core.rows(PARENT / "raw/hidden_index.jsonl")
    key = common.hidden_key(index)
    values = core.rows(OUT / "protocol/effect_groups.jsonl")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    effects = np.lib.format.open_memmap(OUT / "raw/effects.float16.npy", mode="w+", dtype=np.float16, shape=(160, 3, 37, 128, 2560))
    for i, group in enumerate(values):
        cells = {}
        for a in (0, 1):
            for b in (0, 1):
                idx = key[(group["family"], group["surface"], int(group["unit"]), a, b, int(group["order"]))]
                cells[(a, b)] = np.asarray(fields[idx], np.float32)
        effects[i] = common.factorial_effect(cells).astype(np.float16)
        if i % 10 == 0 or i + 1 == len(values):
            effects.flush()
            print(f"[C236] effects {i + 1}/{len(values)}", flush=True)
    effects.flush()
    checks = {"shape": list(effects.shape) == [160, 3, 37, 128, 2560], "finite_sample": bool(np.isfinite(effects[:, :, :, ::16, ::64]).all())}
    core.save(OUT / "audit/internal_extract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def events() -> None:
    if (OUT / "raw/events.int8.npy").exists():
        raise RuntimeError("events already exist")
    effects = np.load(OUT / "raw/effects.float16.npy", mmap_mode="r")
    groups_index = core.rows(OUT / "protocol/effect_groups.jsonl")
    discovery = [row["effect_index"] for row in groups_index if row["partition"] == "discovery"]
    repeat = core.load(PARENT / "raw/run_metadata.json")["numerical_repeat"]
    numerical = 4.0 * max(float(repeat["repeat_max_abs"]), float(repeat["float16_roundtrip_max_abs"]))
    thresholds = []
    q75s = []
    for q in range(37):
        values = np.abs(np.asarray(effects[discovery, :, q], np.float16)).reshape(-1)
        positive = values[values > numerical]
        q75 = float(np.quantile(positive, 0.75)) if positive.size else 0.0
        eta = max(numerical, 0.25 * q75, 1e-6)
        q75s.append(q75)
        thresholds.append(eta)
        print(f"[C236] threshold q{q:02d}={eta:.8g} from {positive.size} values", flush=True)
    core.save(OUT / "protocol/frozen_event_thresholds.json", {"numerical_floor": numerical, "discovery_abs_q75": q75s, "thresholds": thresholds})

    event_map = np.lib.format.open_memmap(OUT / "raw/events.int8.npy", mode="w+", dtype=np.int8, shape=effects.shape)
    first = np.lib.format.open_memmap(OUT / "raw/first_formation.int8.npy", mode="w+", dtype=np.int8, shape=(160, 3, 128, 2560))
    persistence = np.lib.format.open_memmap(OUT / "raw/persistence.uint8.npy", mode="w+", dtype=np.uint8, shape=(160, 3, 128, 2560))
    eta = np.asarray(thresholds, np.float32)[None, :, None, None]
    for i in range(160):
        values = np.asarray(effects[i], np.float32)
        coded = np.where(values > eta, 1, np.where(values < -eta, -1, 0)).astype(np.int8)
        event_map[i] = coded
        active = coded != 0
        any_active = active.any(axis=1)
        first_i = np.argmax(active, axis=1).astype(np.int8)
        first_i[~any_active] = -1
        first[i] = first_i
        persistence[i] = active.sum(axis=1).astype(np.uint8)
        if i % 10 == 0 or i == 159:
            event_map.flush(); first.flush(); persistence.flush()
            print(f"[C236] events {i + 1}/160", flush=True)
    checks = {
        "events": list(event_map.shape) == [160, 3, 37, 128, 2560],
        "alphabet": set(np.unique(event_map[:, :, :, ::16, ::64]).tolist()) <= {-1, 0, 1},
        "first": list(first.shape) == [160, 3, 128, 2560],
        "persistence": list(persistence.shape) == [160, 3, 128, 2560],
        "thresholds": len(thresholds) == 37 and min(thresholds) > 0,
    }
    core.save(OUT / "audit/internal_event_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def analyze() -> None:
    events = np.load(OUT / "raw/events.int8.npy", mmap_mode="r")
    groups_index = core.rows(OUT / "protocol/effect_groups.jsonl")
    density = []
    for family in common.FAMILIES:
        for partition in common.PARTITIONS:
            selected = [row["effect_index"] for row in groups_index if row["family"] == family and row["partition"] == partition]
            for effect_i, effect in enumerate(common.EFFECTS):
                for q in range(37):
                    values = np.asarray(events[selected, effect_i, q])
                    density.append({
                        "family": family, "partition": partition, "effect": effect, "checkpoint": q,
                        "support_groups": len(selected), "active_density": float(np.mean(values != 0)),
                        "up_fraction": float(np.mean(values == 1)), "down_fraction": float(np.mean(values == -1)),
                    })
    core.write_rows(OUT / "analysis/event_density.jsonl", density)
    order_agreement = []
    for family in common.FAMILIES:
        for surface in common.SURFACES:
            for unit in common.PARTITION_UNITS[common.SURFACE_PARTITION[surface]]:
                left = next(row["effect_index"] for row in groups_index if row["family"] == family and row["surface"] == surface and row["unit"] == unit and row["order"] == 1)
                right = next(row["effect_index"] for row in groups_index if row["family"] == family and row["surface"] == surface and row["unit"] == unit and row["order"] == -1)
                a = np.asarray(events[left])
                b = np.asarray(events[right])
                active = (a != 0) | (b != 0)
                order_agreement.append({"family": family, "surface": surface, "unit": unit, "signed_agreement_on_union": float(np.mean(a[active] == b[active])) if active.any() else 1.0, "active_union": int(active.sum())})
    core.write_rows(OUT / "analysis/candidate_order_agreement.jsonl", order_agreement)
    report = {
        "phase": 1770,
        "campaign": "C236",
        "status": "full_coordinate_interval_events_extracted",
        "effect_values": int(np.prod(events.shape)),
        "event_values": int(np.prod(events.shape)),
        "mean_active_density": float(np.mean([row["active_density"] for row in density])),
        "median_candidate_order_signed_agreement": float(np.median([row["signed_agreement_on_union"] for row in order_agreement])),
        "thresholds": core.load(OUT / "protocol/frozen_event_thresholds.json")["thresholds"],
        "claim_boundary": "Events are thresholded descriptive changes at physical coordinates. They are not causal edges or semantic neurons.",
        "next_authorization": "C237_discovery_only_readable_conditional_event_rules",
    }
    core.save(OUT / "analysis/summary.json", report)
    checks = {"density_rows": len(density) == 5 * 4 * 3 * 37, "order_rows": len(order_agreement) == 80, "finite": bool(np.isfinite([row["active_density"] for row in density] + [row["signed_agreement_on_union"] for row in order_agreement]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "extract": core.load(OUT / "audit/internal_extract_audit.json")["all_checks_passed"], "events": core.load(OUT / "audit/internal_event_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1770, "campaign": "C236", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "extract", "events", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "extract": extract, "events": events, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
