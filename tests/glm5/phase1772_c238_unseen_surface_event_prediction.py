#!/usr/bin/env python3
"""C238: prospectively test frozen event rules on unseen surfaces and vocabularies."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C238"]
C235 = common.OUTS["C235"]
C236 = common.OUTS["C236"]
C237 = common.OUTS["C237"]


def signed_jaccard(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred != 0) | (truth != 0)
    return float(np.mean(pred[union] == truth[union])) if union.any() else 1.0


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C237 / "audit/independent_final_audit.json")
    gate = core.load(common.OUTS["C234"] / "protocol/preregistration.json")["unseen_event_gate"]
    checks = {"authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C238"), "gate": gate == {"correct_signed_jaccard_min": 0.15, "all_control_margin_min": 0.02, "families_min": 3}}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1772, "campaign": "C238", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "unseen_surface_event_prediction_frozen",
        "frozen_rule": "C237 rule_codes and amplitude intervals",
        "reveal_order": ["confirmation", "lockbox", "fresh"],
        "controls": ["best_wrong_family", "surface_only_generic", "relation_role_only", "nearest_length_discovery_group", "zero"],
        "primary_metric": "signed event Jaccard over every role/checkpoint/coordinate",
        "secondary_metrics": ["amplitude interval coverage", "frozen role-precedence accuracy"],
        "gate": gate,
        "no_refit": True,
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "reveal_confirmation_then_lockbox_then_fresh_once",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def evaluate() -> None:
    if (OUT / "analysis/prediction_rows.jsonl").exists():
        raise RuntimeError("already evaluated")
    groups = core.rows(C236 / "protocol/effect_groups.jsonl")
    rules = np.load(C237 / "analysis/rule_codes.int8.npy", mmap_mode="r")
    lower = np.load(C237 / "analysis/amplitude_lower.float16.npy", mmap_mode="r")
    upper = np.load(C237 / "analysis/amplitude_upper.float16.npy", mmap_mode="r")
    events = np.load(C237 / "raw/role_events.int8.npy", mmap_mode="r")
    effects = np.load(C237 / "raw/role_effects.float16.npy", mmap_mode="r")
    precedence = core.rows(C237 / "analysis/precedence_rules.jsonl")
    hidden = core.rows(C235 / "raw/hidden_index.jsonl")
    lengths = {}
    for group in groups:
        selected = [row["length"] for row in hidden if row["family"] == group["family"] and row["surface"] == group["surface"] and row["unit"] == group["unit"] and row["order"] == group["order"]]
        lengths[group["effect_index"]] = float(np.mean(selected))
    discovery = [row["effect_index"] for row in groups if row["partition"] == "discovery"]
    disc_events = np.asarray(events[discovery])
    up = np.mean(disc_events == 1, axis=0)
    down = np.mean(disc_events == -1, axis=0)
    active = up + down
    generic = np.where((active >= 0.75) & (np.maximum(up, down) / np.maximum(active, 1e-9) >= 0.80), np.where(up >= down, 1, -1), 0).astype(np.int8)
    rows = []
    for partition in ("confirmation", "lockbox", "fresh"):
        for group in [row for row in groups if row["partition"] == partition]:
            gi = int(group["effect_index"])
            family_i = common.FAMILIES.index(group["family"])
            nearest = min((idx for idx in discovery if groups[idx]["order"] == group["order"]), key=lambda idx: abs(lengths[idx] - lengths[gi]))
            for effect_i, effect in enumerate(common.EFFECTS):
                truth = np.asarray(events[gi, effect_i])
                magnitude = np.asarray(effects[gi, effect_i], np.float32)
                correct = np.asarray(rules[family_i, effect_i])
                wrong_scores = [signed_jaccard(np.asarray(rules[wrong_i, effect_i]), truth) for wrong_i in range(5) if wrong_i != family_i]
                relation_only = np.zeros_like(correct)
                relation_only[:, common.ROLES.index("relation")] = correct[:, common.ROLES.index("relation")]
                pred_active = correct != 0
                lo = np.asarray(lower[family_i, effect_i], np.float32)
                hi = np.asarray(upper[family_i, effect_i], np.float32)
                interval_coverage = float(np.mean((magnitude[pred_active] >= lo[pred_active]) & (magnitude[pred_active] <= hi[pred_active]))) if pred_active.any() else 0.0
                density = np.mean(truth != 0, axis=-1)
                first = np.full(len(common.ROLES), -1, np.int16)
                for role_i in range(len(common.ROLES)):
                    hits = np.flatnonzero(density[:, role_i] >= 0.02)
                    if hits.size:
                        first[role_i] = int(hits[0])
                edge_rules = [row for row in precedence if row["family"] == group["family"] and row["effect"] == effect and row["rule_active"]]
                edge_hits = []
                for edge in edge_rules:
                    a = first[common.ROLES.index(edge["source_role"])]
                    b = first[common.ROLES.index(edge["target_role"])]
                    edge_hits.append(a >= 0 and b >= 0 and a < b)
                rows.append({
                    "partition": partition, "family": group["family"], "surface": group["surface"], "unit": group["unit"], "order": group["order"], "effect": effect,
                    "correct_signed_jaccard": signed_jaccard(correct, truth),
                    "best_wrong_family_signed_jaccard": max(wrong_scores),
                    "surface_only_signed_jaccard": signed_jaccard(generic[effect_i], truth),
                    "relation_only_signed_jaccard": signed_jaccard(relation_only, truth),
                    "length_only_signed_jaccard": signed_jaccard(np.asarray(events[nearest, effect_i]), truth),
                    "zero_signed_jaccard": signed_jaccard(np.zeros_like(correct), truth),
                    "amplitude_interval_coverage": interval_coverage,
                    "precedence_accuracy": float(np.mean(edge_hits)) if edge_hits else None,
                    "precedence_rules": len(edge_hits),
                })
        print(f"[C238] revealed {partition}", flush=True)
    core.write_rows(OUT / "analysis/prediction_rows.jsonl", rows)
    controls = ("best_wrong_family_signed_jaccard", "surface_only_signed_jaccard", "relation_only_signed_jaccard", "length_only_signed_jaccard", "zero_signed_jaccard")
    family_results = []
    for family in common.FAMILIES:
        for partition in ("confirmation", "lockbox", "fresh", "final"):
            chosen = [row for row in rows if row["family"] == family and (row["partition"] == partition if partition != "final" else row["partition"] in ("lockbox", "fresh"))]
            correct = float(np.median([row["correct_signed_jaccard"] for row in chosen]))
            control_values = {control: float(np.median([row[control] for row in chosen])) for control in controls}
            margin = correct - max(control_values.values())
            family_results.append({"family": family, "partition": partition, "support": len(chosen), "correct_signed_jaccard": correct, "controls": control_values, "all_control_margin": margin, "interval_coverage": float(np.median([row["amplitude_interval_coverage"] for row in chosen])), "precedence_accuracy": float(np.mean([row["precedence_accuracy"] for row in chosen if row["precedence_accuracy"] is not None])) if any(row["precedence_accuracy"] is not None for row in chosen) else None, "passed": bool(partition == "final" and correct >= 0.15 and margin >= 0.02)})
    core.write_rows(OUT / "analysis/family_results.jsonl", family_results)
    finals = [row for row in family_results if row["partition"] == "final"]
    passed = sum(row["passed"] for row in finals) >= 3
    report = {
        "phase": 1772, "campaign": "C238", "status": "unseen_surface_event_prediction_adjudicated",
        "family_final_results": finals, "families_passed": sum(row["passed"] for row in finals), "campaign_passed": passed,
        "strict_interpretation": "Passing requires a frozen family rule to identify signed role/checkpoint/coordinate events on chronicle and dispatch better than every registered control.",
        "next_authorization": "C239_five_flagship_observation_regardless_of_gate_then_C240_composition_prediction",
    }
    core.save(OUT / "analysis/summary.json", report)
    checks = {"rows": len(rows) == (30 + 20 + 30) * 3, "families": len(finals) == 5, "finite": bool(np.isfinite([row[key] for row in rows for key in ("correct_signed_jaccard",) + controls + ("amplitude_interval_coverage",)]).all()), "no_refit": True}
    core.save(OUT / "audit/internal_evaluation_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "evaluation": core.load(OUT / "audit/internal_evaluation_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1772, "campaign": "C238", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "evaluate", "close"))
    args = parser.parse_args()
    {"contract": contract, "evaluate": evaluate, "close": close}[args.command]()


if __name__ == "__main__":
    main()
