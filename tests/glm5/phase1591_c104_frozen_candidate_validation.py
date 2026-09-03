#!/usr/bin/env python3
"""Phase1591 / C104: validate frozen upstream activation barcodes on fresh materials."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1589_c104_upstream_candidate_validation"
C103 = TESTS / "result/phase1588_c103_code_residualized_role_state_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1577_c101_dual_arm_analysis as c101_analysis

PHASE = 1591
CAMPAIGN = "C104"
FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
ROLES = c101_analysis.BREADTH_ROLES
STATES = 37
DIM = 2560
EFFECTS = ("truth", "code", "truth_x_code")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def prepare() -> None:
    capture = core.load(OUT / "analysis/qwen_full_field_capture_summary.json")
    audit = core.load(OUT / "audit/independent_full_field_capture_audit.json")
    contract = core.load(OUT / "protocol/preregistration.json")
    if capture["authorization"] != "run_phase1591_c104_frozen_candidate_validation" or not audit["all_checks_passed"]:
        raise RuntimeError("C104 validation authorization missing")
    adapter = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "frozen_candidate_staged_validation_adapter",
        "producer_sha256": core.sha(Path(__file__)),
        "raw_sha256": capture["raw_sha256"],
        "index_sha256": capture["index_sha256"],
        "barcode_sha256": contract["barcode_sha256"],
        "predictions": contract["predictions"],
        "effects": list(EFFECTS),
        "roles": list(ROLES),
        "null": "permute all 2560 activation-coordinate correspondences; 2000 draws",
        "formal_partitions": ["confirmation", "lockbox"],
        "response_discovery": "descriptive only; no candidate or threshold modification",
        "authorization": "compute_all_unit_role_state_effects_before_reveal",
    }
    core.save(OUT / "protocol/frozen_candidate_validation_adapter.json", adapter)
    print(json.dumps(adapter, indent=2))


def role_vector(field: np.ndarray, row: dict[str, Any], role: str) -> np.ndarray:
    positions = np.asarray(row["role_positions"][role], dtype=np.int64) + int(row["token_start"])
    return decode_bf16(np.asarray(field[:, positions, :], dtype=np.uint16)).mean(axis=1, dtype=np.float64).astype(np.float32)


def coefficients() -> None:
    adapter = core.load(OUT / "protocol/frozen_candidate_validation_adapter.json")
    capture = core.load(OUT / "analysis/qwen_full_field_capture_summary.json")
    if adapter["authorization"] != "compute_all_unit_role_state_effects_before_reveal" or adapter["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("C104 coefficient computation not authorized")
    raw_path = OUT / "raw/qwen3_all_token_state_coordinate_field.uint16.npy"
    index_path = OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl"
    if core.sha(raw_path) != capture["raw_sha256"] or core.sha(index_path) != capture["index_sha256"]:
        raise RuntimeError("C104 capture hash mismatch")
    field = np.load(raw_path, mmap_mode="r")
    rows = core.rows(index_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["unit_id"]].append(row)
    units = []
    for unit_id, unit_rows in grouped.items():
        first = unit_rows[0]
        units.append({key: first[key] for key in ("unit_id", "family", "world", "partition", "surface")})
    units.sort(key=lambda row: row["unit_id"])
    path = OUT / "raw/qwen3_breadth_three_effect_coefficients.float32.npy"
    coeff = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), 3, STATES, len(ROLES), DIM))
    for unit_index, unit in enumerate(units):
        unit_rows = grouped[unit["unit_id"]]
        if len(unit_rows) != 16:
            raise RuntimeError((unit["unit_id"], len(unit_rows)))
        values = np.stack([np.stack([role_vector(field, row, role) for role in ROLES], axis=1) for row in unit_rows])
        signs = np.asarray([[row["truth_factor"], row["code"], row["truth_factor"] * row["code"]] for row in unit_rows], dtype=np.float32)
        coeff[unit_index] = np.einsum("ce,csrd->esrd", signs, values, optimize=True) / 16.0
        if (unit_index + 1) % 6 == 0:
            coeff.flush()
            print(f"[phase1591] coefficients {unit_index + 1}/{len(units)}", flush=True)
    coeff.flush()
    del coeff
    unit_index_path = OUT / "raw/qwen3_breadth_three_effect_index.jsonl"
    core.write_rows(unit_index_path, [{"row_index": index, **row} for index, row in enumerate(units)])
    values = np.load(path, mmap_mode="r")
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "all_unit_all_role_all_state_effects_computed_before_partition_reveal",
        "shape": list(values.shape),
        "sha256": core.sha(path),
        "index_sha256": core.sha(unit_index_path),
        "finite": bool(np.isfinite(values).all()),
        "authorization": "reveal_response_discovery_descriptively",
    }
    core.save(OUT / "analysis/role_effect_coefficient_summary.json", report)
    print(json.dumps(report, indent=2))


def partition_vector(coeff: np.ndarray, units: list[dict[str, Any]], family: str, partition: str,
                     effect: int, state: int, role: int) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family and row["partition"] == partition]
    if len(selected) != 3:
        raise RuntimeError((family, partition, selected))
    return np.asarray(coeff[selected, effect, state, role], dtype=np.float64).mean(axis=0)


def permutation_test(source: np.ndarray, target: np.ndarray, draws: int, seed: int) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    observed = cosine(source, target)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        null[draw] = cosine(source, target[rng.permutation(DIM)])
    q99 = float(np.quantile(null, 0.99))
    return {
        "observed_cosine": observed,
        "null_q99": q99,
        "beats_q99": observed > q99,
        "upper_p": float((1 + np.count_nonzero(null >= observed)) / (draws + 1)),
        "sign_agreement": float(np.mean(np.sign(source) == np.sign(target))),
        "source_norm": float(np.linalg.norm(source)),
        "target_norm": float(np.linalg.norm(target)),
    }


def evaluate(partition: str, seed_base: int) -> list[dict[str, Any]]:
    contract = core.load(OUT / "protocol/preregistration.json")
    coeff = np.load(OUT / "raw/qwen3_breadth_three_effect_coefficients.float32.npy", mmap_mode="r")
    units = core.rows(OUT / "raw/qwen3_breadth_three_effect_index.jsonl")
    source = np.load(ROOT / contract["barcode_path"], mmap_mode="r")
    predictions = {row["family"]: row for row in contract["predictions"]}
    output = []
    for family_index, family in enumerate(FAMILIES):
        prediction = predictions[family]
        role_index = int(prediction["role_index"])
        state = int(prediction["state"])
        targets = [partition_vector(coeff, units, family, partition, effect, state, role_index) for effect in range(3)]
        primary = permutation_test(source[family_index], targets[0], 2000, seed_base + family_index)
        controls = {EFFECTS[effect]: cosine(source[family_index], targets[effect]) for effect in (1, 2)}
        output.append({
            "partition": partition,
            "family": family,
            "role": prediction["role"],
            "role_index": role_index,
            "state": state,
            "coordinates": DIM,
            "primary": primary,
            "control_cosines": controls,
            "primary_specific_over_controls": primary["observed_cosine"] > max(controls.values()),
        })
    return output


def reveal(partition: str) -> None:
    if partition == "response_discovery":
        summary = core.load(OUT / "analysis/role_effect_coefficient_summary.json")
        audit = core.load(OUT / "audit/independent_coefficient_audit.json")
        expected = "reveal_response_discovery_descriptively"
        if summary["authorization"] != expected or not audit["all_checks_passed"]:
            raise RuntimeError("discovery reveal not authorized")
        rows = evaluate(partition, 15910)
        path = OUT / "analysis/response_discovery_frozen_candidate_results.jsonl"
        next_authorization = "reveal_confirmation_without_modification"
    elif partition == "confirmation":
        discovery = core.load(OUT / "analysis/response_discovery_reveal_summary.json")
        audit = core.load(OUT / "audit/independent_response_discovery_audit.json")
        if discovery["authorization"] != "reveal_confirmation_without_modification" or not audit["all_checks_passed"]:
            raise RuntimeError("confirmation reveal not authorized")
        rows = evaluate(partition, 1591)
        path = OUT / "analysis/confirmation_frozen_candidate_results.jsonl"
        next_authorization = "reveal_lockbox_once_without_modification"
    else:
        confirmation = core.load(OUT / "analysis/confirmation_reveal_summary.json")
        audit = core.load(OUT / "audit/independent_confirmation_audit.json")
        if confirmation["authorization"] != "reveal_lockbox_once_without_modification" or not audit["all_checks_passed"]:
            raise RuntimeError("lockbox reveal not authorized")
        rows = evaluate(partition, 1592)
        path = OUT / "analysis/lockbox_frozen_candidate_results.jsonl"
        next_authorization = "finalize_fresh_validation"
    core.write_rows(path, rows)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "partition": partition,
        "status": f"{partition}_revealed_with_unchanged_frozen_candidates",
        "rows_sha256": core.sha(path),
        "passed_q99": sum(row["primary"]["beats_q99"] for row in rows),
        "specific_over_controls": sum(row["primary_specific_over_controls"] for row in rows),
        "total": len(rows),
        "authorization": next_authorization,
    }
    name = "response_discovery_reveal_summary.json" if partition == "response_discovery" else f"{partition}_reveal_summary.json"
    core.save(OUT / f"analysis/{name}", result)
    print(json.dumps({"summary": result, "rows": rows}, indent=2))


def finalize() -> None:
    lockbox_summary = core.load(OUT / "analysis/lockbox_reveal_summary.json")
    audit = core.load(OUT / "audit/independent_lockbox_audit.json")
    if lockbox_summary["authorization"] != "finalize_fresh_validation" or not audit["all_checks_passed"]:
        raise RuntimeError("C104 finalization not authorized")
    discovery = {row["family"]: row for row in core.rows(OUT / "analysis/response_discovery_frozen_candidate_results.jsonl")}
    confirmation = {row["family"]: row for row in core.rows(OUT / "analysis/confirmation_frozen_candidate_results.jsonl")}
    lockbox = {row["family"]: row for row in core.rows(OUT / "analysis/lockbox_frozen_candidate_results.jsonl")}
    rows = []
    authorized = []
    for family in FAMILIES:
        formal_pass = confirmation[family]["primary"]["beats_q99"] and lockbox[family]["primary"]["beats_q99"]
        control_pass = confirmation[family]["primary_specific_over_controls"] and lockbox[family]["primary_specific_over_controls"]
        rows.append({
            "family": family,
            "role": confirmation[family]["role"],
            "state": confirmation[family]["state"],
            "discovery_cosine": discovery[family]["primary"]["observed_cosine"],
            "confirmation_cosine": confirmation[family]["primary"]["observed_cosine"],
            "confirmation_q99": confirmation[family]["primary"]["null_q99"],
            "lockbox_cosine": lockbox[family]["primary"]["observed_cosine"],
            "lockbox_q99": lockbox[family]["primary"]["null_q99"],
            "formal_replication_pass": bool(formal_pass),
            "specific_over_controls_both": bool(control_pass),
        })
        if formal_pass:
            authorized.append(family)
    core.write_rows(OUT / "analysis/fresh_validation_family_summary.jsonl", rows)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "fresh_frozen_upstream_candidate_validation_complete",
        "formal_replication_families": authorized,
        "formal_replication_passed": len(authorized),
        "specific_over_controls_both": sum(row["specific_over_controls_both"] for row in rows),
        "total_families": 4,
        "family_summary_sha256": core.sha(OUT / "analysis/fresh_validation_family_summary.jsonl"),
        "behavior_scope": core.load(OUT / "analysis/qwen_full_field_capture_summary.json")["behavior"],
        "interpretation": "fresh full-coordinate upstream task-response barcode validation; not yet causal, semantic, weight-level, or cross-model",
        "authorization": "run_phase1592_c104_upstream_role_intervention" if authorized else "close_c104_without_intervention",
    }
    core.save(OUT / "analysis/frozen_candidate_validation_final.json", final)
    print(json.dumps({"final": final, "rows": rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "coefficients", "response_discovery", "confirmation", "lockbox", "finalize"))
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "coefficients": coefficients()
    elif args.action == "finalize": finalize()
    else: reveal(args.action)


if __name__ == "__main__":
    main()
