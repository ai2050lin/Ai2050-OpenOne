#!/usr/bin/env python3
"""Phase1584 / C102: staged full-coordinate barcode analysis and reveal."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1577_c101_dual_arm_analysis as c101_analysis

PHASE = 1584
CAMPAIGN = "C102"
STATES = 37
DIM = 2560
GRAPH_FAMILIES = ("taxonomy", "containment", "comparison", "precedence")
BREADTH_FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
GRAPH_EFFECTS = ("primary", "code", "primary_x_code")
BREADTH_EFFECTS = GRAPH_EFFECTS


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def decode_bf16(bits: np.ndarray) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def prepare() -> None:
    capture = core.load(OUT / "analysis/qwen_full_field_capture_summary.json")
    audit = core.load(OUT / "audit/independent_full_field_capture_audit.json")
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    if capture["authorization"] != "run_phase1584_c102_staged_barcode_analysis" or not audit["all_checks_passed"]:
        raise RuntimeError("C102 analysis authorization missing")
    adapter = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "staged_reveal_adapter_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "raw_sha256": capture["raw_sha256"],
        "index_sha256": capture["index_sha256"],
        "prediction_sha256": core.sha(OUT / "protocol/frozen_coordinate_barcode_predictions.json"),
        "effects": {"graph": list(GRAPH_EFFECTS), "breadth": list(BREADTH_EFFECTS)},
        "roles": {"graph": list(c101_analysis.CONF_ROLES), "breadth": list(c101_analysis.BREADTH_ROLES)},
        "stages": ["coefficients", "response_discovery", "confirmation", "lockbox"],
        "nested_k": frozen["validation"]["nested_k"],
        "null": {"method": "permute activation-coordinate correspondence within the frozen nested support", "draws": frozen["validation"]["null_draws"]},
        "authorization": "compute_c102_role_effect_coefficients",
    }
    core.save(OUT / "protocol/staged_analysis_adapter.json", adapter)
    print(json.dumps(adapter, indent=2))


def role_vector(field: np.ndarray, row: dict[str, Any], role: str) -> np.ndarray:
    positions = np.asarray(row["role_positions"][role], dtype=np.int64) + int(row["token_start"])
    return decode_bf16(np.asarray(field[:, positions, :], dtype=np.uint16)).mean(axis=1, dtype=np.float64).astype(np.float32)


def compute_arm(
    field: np.ndarray,
    rows: list[dict[str, Any]],
    arm: str,
    roles: tuple[str, ...],
) -> tuple[Path, list[dict[str, Any]]]:
    arm_rows = [row for row in rows if row["arm"] == arm]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in arm_rows:
        grouped[row["unit_id"]].append(row)
    units: list[dict[str, Any]] = []
    for unit_id, unit_rows in grouped.items():
        first = unit_rows[0]
        units.append({key: first[key] for key in ("unit_id", "arm", "family", "world", "partition", "surface")})
    units.sort(key=lambda row: row["unit_id"])
    path = OUT / f"raw/qwen3_{arm}_three_effect_coefficients.float32.npy"
    coeff = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), 3, STATES, len(roles), DIM))
    for unit_index, unit in enumerate(units):
        unit_rows = grouped[unit["unit_id"]]
        if len(unit_rows) != 16:
            raise RuntimeError((unit["unit_id"], len(unit_rows)))
        values = np.stack([np.stack([role_vector(field, row, role) for role in roles], axis=1) for row in unit_rows], axis=0)
        if arm == "graph":
            sign_rows = [[row["x"] * row["y"], row["code"], row["x"] * row["y"] * row["code"]] for row in unit_rows]
        else:
            sign_rows = [[row["truth_factor"], row["code"], row["truth_factor"] * row["code"]] for row in unit_rows]
        signs = np.asarray(sign_rows, dtype=np.float32)
        coeff[unit_index] = np.einsum("ce,csrd->esrd", signs, values, optimize=True) / 16.0
        if (unit_index + 1) % 6 == 0:
            coeff.flush()
            print(f"[phase1584] {arm} coefficients {unit_index + 1}/{len(units)}", flush=True)
    coeff.flush()
    del coeff
    index = [{"row_index": index, **row} for index, row in enumerate(units)]
    index_path = OUT / f"raw/qwen3_{arm}_three_effect_index.jsonl"
    core.write_rows(index_path, index)
    return path, index


def coefficients() -> None:
    adapter = core.load(OUT / "protocol/staged_analysis_adapter.json")
    capture = core.load(OUT / "analysis/qwen_full_field_capture_summary.json")
    if adapter["authorization"] != "compute_c102_role_effect_coefficients" or adapter["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("coefficient computation not authorized")
    raw_path = OUT / "raw/qwen3_all_token_state_coordinate_field.uint16.npy"
    index_path = OUT / "raw/qwen3_all_token_state_coordinate_index.jsonl"
    if core.sha(raw_path) != capture["raw_sha256"] or core.sha(index_path) != capture["index_sha256"]:
        raise RuntimeError("capture hash mismatch")
    field = np.load(raw_path, mmap_mode="r")
    rows = core.rows(index_path)
    graph_path, graph_units = compute_arm(field, rows, "graph", c101_analysis.CONF_ROLES)
    breadth_path, breadth_units = compute_arm(field, rows, "breadth", c101_analysis.BREADTH_ROLES)
    graph = np.load(graph_path, mmap_mode="r")
    breadth = np.load(breadth_path, mmap_mode="r")
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "three_effect_role_coefficients_complete_before_partition_reveal",
        "graph": {"shape": list(graph.shape), "sha256": core.sha(graph_path), "index_sha256": core.sha(OUT / "raw/qwen3_graph_three_effect_index.jsonl")},
        "breadth": {"shape": list(breadth.shape), "sha256": core.sha(breadth_path), "index_sha256": core.sha(OUT / "raw/qwen3_breadth_three_effect_index.jsonl")},
        "finite": bool(np.isfinite(graph).all() and np.isfinite(breadth).all()),
        "units": {"graph": len(graph_units), "breadth": len(breadth_units)},
        "authorization": "reveal_response_discovery_only",
    }
    core.save(OUT / "analysis/role_effect_coefficient_summary.json", report)
    print(json.dumps(report, indent=2))


def load_arm(arm: str) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coeff = np.load(OUT / f"raw/qwen3_{arm}_three_effect_coefficients.float32.npy", mmap_mode="r")
    units = core.rows(OUT / f"raw/qwen3_{arm}_three_effect_index.jsonl")
    return coeff, units


def partition_vector(
    coeff: np.ndarray,
    units: list[dict[str, Any]],
    family: str,
    partition: str,
    effect: int,
    state: int,
    role: int,
) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family and row["partition"] == partition]
    if not selected:
        raise RuntimeError((family, partition))
    return np.asarray(coeff[selected, effect, state, role], dtype=np.float64).mean(axis=0)


def permutation_test(source: np.ndarray, target: np.ndarray, coordinates: list[int], draws: int, seed: int) -> dict[str, Any]:
    source_values = np.asarray(source[coordinates], dtype=np.float64)
    target_values = np.asarray(target[coordinates], dtype=np.float64)
    observed = cosine(source_values, target_values)
    rng = np.random.default_rng(seed)
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        null[draw] = cosine(source_values, target_values[rng.permutation(len(target_values))])
    q99 = float(np.quantile(null, 0.99))
    sign_agreement = float(np.mean(np.sign(source_values) == np.sign(target_values)))
    return {
        "observed_cosine": observed,
        "null_q99": q99,
        "beats_q99": observed > q99,
        "upper_p": float((1 + np.count_nonzero(null >= observed)) / (draws + 1)),
        "sign_agreement": sign_agreement,
        "source_norm": float(np.linalg.norm(source_values)),
        "target_norm": float(np.linalg.norm(target_values)),
    }


def source_barcode(arm: str, family: str) -> np.ndarray:
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    families = GRAPH_FAMILIES if arm == "graph" else BREADTH_FAMILIES
    path = ROOT / frozen["barcodes"][f"{arm}_path"]
    values = np.load(path, mmap_mode="r")
    return np.asarray(values[families.index(family)], dtype=np.float64)


def family_selector(family: str) -> dict[str, Any]:
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    return next(row for row in frozen["selectors"] if row["family"] == family)


def evaluate_partition(partition: str, selected_k: dict[str, int] | None, seed_base: int) -> list[dict[str, Any]]:
    frozen = core.load(OUT / "protocol/frozen_coordinate_barcode_predictions.json")
    draws = int(frozen["validation"]["null_draws"])
    nested_k = frozen["validation"]["nested_k"]
    output: list[dict[str, Any]] = []
    for family_index, family in enumerate((*GRAPH_FAMILIES, *BREADTH_FAMILIES)):
        arm = "graph" if family in GRAPH_FAMILIES else "breadth"
        coeff, units = load_arm(arm)
        selector = family_selector(family)
        state = selector["selector"]["state"]
        role_index = selector["selector"]["role_index"]
        source = source_barcode(arm, family)
        targets = [partition_vector(coeff, units, family, partition, effect, state, role_index) for effect in range(3)]
        ks = [selected_k[family]] if selected_k is not None else nested_k
        for k_index, k in enumerate(ks):
            coordinates = selector["coordinate_rank"][:k]
            primary = permutation_test(source[0], targets[0], coordinates, draws, seed_base + family_index * 100 + k_index)
            output.append(
                {
                    "partition": partition,
                    "arm": arm,
                    "family": family,
                    "role": selector["selector"]["role"],
                    "state": state,
                    "k": k,
                    "primary": primary,
                    "control_cosines": {"code": cosine(source[0][coordinates], targets[1][coordinates]), "primary_x_code": cosine(source[0][coordinates], targets[2][coordinates])},
                    "primary_specific_over_controls": primary["observed_cosine"] > max(cosine(source[0][coordinates], targets[1][coordinates]), cosine(source[0][coordinates], targets[2][coordinates])),
                }
            )
    return output


def discover() -> None:
    summary = core.load(OUT / "analysis/role_effect_coefficient_summary.json")
    audit = core.load(OUT / "audit/independent_coefficient_audit.json")
    if summary["authorization"] != "reveal_response_discovery_only" or not audit["all_checks_passed"]:
        raise RuntimeError("response discovery not authorized")
    rows = evaluate_partition("response_discovery", None, 15840)
    selected: dict[str, dict[str, Any]] = {}
    for family in (*GRAPH_FAMILIES, *BREADTH_FAMILIES):
        candidates = [row for row in rows if row["family"] == family]
        choice = max(candidates, key=lambda row: (row["primary"]["observed_cosine"] - row["primary"]["null_q99"], row["primary"]["observed_cosine"], -row["k"]))
        selected[family] = {
            "k": choice["k"],
            "discovery_beats_q99": choice["primary"]["beats_q99"],
            "discovery_cosine": choice["primary"]["observed_cosine"],
            "discovery_null_q99": choice["primary"]["null_q99"],
            "role": choice["role"],
            "state": choice["state"],
        }
    core.write_rows(OUT / "analysis/response_discovery_nested_k.jsonl", rows)
    selection = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "coordinate_coalition_size_selected_from_response_discovery_and_frozen",
        "selection": selected,
        "source_rows_sha256": core.sha(OUT / "analysis/response_discovery_nested_k.jsonl"),
        "rule": "maximize observed cosine minus coordinate-permutation q99; ties favor cosine then smaller K",
        "authorization": "reveal_confirmation_with_frozen_selection",
    }
    core.save(OUT / "protocol/response_discovery_selection.json", selection)
    print(json.dumps(selection, indent=2))


def confirm() -> None:
    selection = core.load(OUT / "protocol/response_discovery_selection.json")
    audit = core.load(OUT / "audit/independent_response_discovery_audit.json")
    if selection["authorization"] != "reveal_confirmation_with_frozen_selection" or not audit["all_checks_passed"]:
        raise RuntimeError("confirmation reveal not authorized")
    chosen = {family: row["k"] for family, row in selection["selection"].items()}
    rows = evaluate_partition("confirmation", chosen, 1584)
    for row in rows:
        row["discovery_pass"] = bool(selection["selection"][row["family"]]["discovery_beats_q99"])
        row["confirmation_pass"] = bool(row["primary"]["beats_q99"])
    path = OUT / "analysis/confirmation_barcode_results.jsonl"
    core.write_rows(path, rows)
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "confirmation_revealed_with_frozen_selector_and_k",
        "selection_sha256": core.sha(OUT / "protocol/response_discovery_selection.json"),
        "rows_sha256": core.sha(path),
        "passed": sum(row["discovery_pass"] and row["confirmation_pass"] for row in rows),
        "total": len(rows),
        "authorization": "reveal_lockbox_once_with_unchanged_selection",
    }
    core.save(OUT / "analysis/confirmation_reveal_summary.json", result)
    print(json.dumps(result, indent=2))


def lockbox() -> None:
    selection = core.load(OUT / "protocol/response_discovery_selection.json")
    confirmation = core.load(OUT / "analysis/confirmation_reveal_summary.json")
    audit = core.load(OUT / "audit/independent_confirmation_audit.json")
    if confirmation["authorization"] != "reveal_lockbox_once_with_unchanged_selection" or not audit["all_checks_passed"]:
        raise RuntimeError("lockbox reveal not authorized")
    if confirmation["selection_sha256"] != core.sha(OUT / "protocol/response_discovery_selection.json"):
        raise RuntimeError("selection changed after confirmation")
    chosen = {family: row["k"] for family, row in selection["selection"].items()}
    rows = evaluate_partition("lockbox", chosen, 1585)
    confirmation_rows = {row["family"]: row for row in core.rows(OUT / "analysis/confirmation_barcode_results.jsonl")}
    authorized = []
    for row in rows:
        family = row["family"]
        row["discovery_pass"] = bool(selection["selection"][family]["discovery_beats_q99"])
        row["confirmation_pass"] = bool(confirmation_rows[family]["confirmation_pass"])
        row["lockbox_pass"] = bool(row["primary"]["beats_q99"])
        row["three_stage_pass"] = row["discovery_pass"] and row["confirmation_pass"] and row["lockbox_pass"]
        if row["three_stage_pass"]:
            authorized.append(family)
    path = OUT / "analysis/lockbox_barcode_results.jsonl"
    core.write_rows(path, rows)
    formation = []
    for family in (*GRAPH_FAMILIES, *BREADTH_FAMILIES):
        arm = "graph" if family in GRAPH_FAMILIES else "breadth"
        coeff, units = load_arm(arm)
        selector = family_selector(family)
        role_index = selector["selector"]["role_index"]
        source_norms = np.asarray(selector["trajectory"]["norms"], dtype=np.float64)
        for partition in ("response_discovery", "confirmation", "lockbox"):
            fresh_norms = np.asarray([np.linalg.norm(partition_vector(coeff, units, family, partition, 0, state, role_index)) for state in range(STATES)], dtype=np.float64)
            formation.append({"family": family, "arm": arm, "partition": partition, "role": selector["selector"]["role"], "source_fresh_norm_cosine": cosine(source_norms, fresh_norms), "source_peak_state": int(np.argmax(source_norms)), "fresh_peak_state": int(np.argmax(fresh_norms)), "fresh_norms": fresh_norms.tolist()})
    core.write_rows(OUT / "analysis/formation_trajectory_validation.jsonl", formation)
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "staged_barcode_reveal_complete",
        "selection_sha256": core.sha(OUT / "protocol/response_discovery_selection.json"),
        "confirmation_sha256": core.sha(OUT / "analysis/confirmation_barcode_results.jsonl"),
        "lockbox_sha256": core.sha(path),
        "formation_sha256": core.sha(OUT / "analysis/formation_trajectory_validation.jsonl"),
        "authorized_intervention_families": authorized,
        "three_stage_passed": len(authorized),
        "total_families": 8,
        "behavior_scope": core.load(OUT / "analysis/qwen_full_field_capture_summary.json")["behavior"],
        "interpretation": "barcode repetition is an activation-coordinate task-response regularity; chance behavior and code asymmetry forbid a natural-semantic-mechanism claim",
        "authorization": "run_phase1585_conditional_coordinate_intervention" if authorized else "skip_intervention_and_export_observational_heatmap",
    }
    core.save(OUT / "analysis/staged_barcode_final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "coefficients", "discover", "confirm", "lockbox"))
    args = parser.parse_args()
    {"prepare": prepare, "coefficients": coefficients, "discover": discover, "confirm": confirm, "lockbox": lockbox}[args.action]()


if __name__ == "__main__":
    main()
