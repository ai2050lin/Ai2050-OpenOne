#!/usr/bin/env python3
"""Phase1582 / C102: freeze full-coordinate predictions from the revealed C101 field."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C101 = TESTS / "result/phase1575_c101_dual_arm"
OUT = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1577_c101_dual_arm_analysis as c101_analysis

PHASE = 1582
CAMPAIGN = "C102"
DIM = 2560
STATES = 37
PARTITIONS = ("response_discovery", "confirmation", "lockbox")
GRAPH_FAMILIES = ("taxonomy", "containment", "comparison", "precedence")
BREADTH_FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
NESTED_K = (16, 32, 64, 128, 256, 512, 1024, 2560)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def load_units(path: Path) -> list[dict[str, Any]]:
    rows = core.rows(path)
    if not all(row["row_index"] == index for index, row in enumerate(rows)):
        raise RuntimeError(f"noncanonical row indices: {path}")
    return rows


def partition_mean(
    coeff: np.ndarray,
    units: list[dict[str, Any]],
    family: str,
    partition: str,
    effect_index: int,
    state: int,
    role_index: int,
) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family and row["partition"] == partition]
    if not selected:
        raise RuntimeError((family, partition))
    return np.asarray(coeff[selected, effect_index, state, role_index], dtype=np.float64).mean(axis=0)


def all_mean(
    coeff: np.ndarray,
    units: list[dict[str, Any]],
    family: str,
    effect_index: int,
    state: int,
    role_index: int,
) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family]
    return np.asarray(coeff[selected, effect_index, state, role_index], dtype=np.float64).mean(axis=0)


def choose_selector(
    coeff: np.ndarray,
    units: list[dict[str, Any]],
    family: str,
    effect_index: int,
    roles: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for role_index, role in enumerate(roles):
        for state in range(STATES):
            vectors = [partition_mean(coeff, units, family, partition, effect_index, state, role_index) for partition in PARTITIONS]
            pair_cosines = [cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2]), cosine(vectors[1], vectors[2])]
            mean_vector = np.mean(vectors, axis=0)
            candidates.append(
                {
                    "role": role,
                    "role_index": role_index,
                    "state": state,
                    "minimum_partition_cosine": min(pair_cosines),
                    "median_partition_cosine": float(np.median(pair_cosines)),
                    "mean_norm": float(np.linalg.norm(mean_vector)),
                }
            )
    candidates.sort(key=lambda row: (row["minimum_partition_cosine"], row["median_partition_cosine"], row["mean_norm"]), reverse=True)
    return candidates[0], candidates


def trajectory(
    coeff: np.ndarray,
    units: list[dict[str, Any]],
    family: str,
    effect_index: int,
    role_index: int,
) -> dict[str, Any]:
    norms = np.asarray(
        [np.linalg.norm(all_mean(coeff, units, family, effect_index, state, role_index)) for state in range(STATES)],
        dtype=np.float64,
    )
    increments = np.diff(norms, prepend=norms[0])
    return {
        "norms": norms.tolist(),
        "increments": increments.tolist(),
        "peak_norm_state": int(np.argmax(norms)),
        "peak_positive_increment_state": int(np.argmax(increments)),
    }


def process_arm(
    arm: str,
    families: tuple[str, ...],
    coeff: np.ndarray,
    units: list[dict[str, Any]],
    roles: tuple[str, ...],
    effects: tuple[str, ...],
    primary_effect: str,
    interaction_effect: str,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    primary_index = effects.index(primary_effect)
    code_index = effects.index("code")
    interaction_index = effects.index(interaction_effect)
    barcodes = np.zeros((len(families), 3, DIM), dtype=np.float32)
    selectors: list[dict[str, Any]] = []
    atlas: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        selector, candidates = choose_selector(coeff, units, family, primary_index, roles)
        role_index = selector["role_index"]
        state = selector["state"]
        effect_indices = (primary_index, code_index, interaction_index)
        for slot, effect_index in enumerate(effect_indices):
            barcodes[family_index, slot] = all_mean(coeff, units, family, effect_index, state, role_index).astype(np.float32)
        primary = np.asarray(barcodes[family_index, 0], dtype=np.float64)
        coordinate_rank = np.argsort(-np.abs(primary), kind="stable").astype(int).tolist()
        total_energy = float(np.dot(primary, primary))
        energy = []
        for k in NESTED_K:
            selected = coordinate_rank[:k]
            captured = float(np.dot(primary[selected], primary[selected]))
            energy.append({"k": k, "energy_fraction": captured / total_energy if total_energy else 0.0})
        item = {
            "arm": arm,
            "family": family,
            "selector": selector,
            "primary_effect": primary_effect,
            "control_effects": ["code", interaction_effect],
            "coordinate_rank": coordinate_rank,
            "nested_energy": energy,
            "trajectory": trajectory(coeff, units, family, primary_index, role_index),
            "top_selector_candidates": candidates[:10],
        }
        selectors.append(item)
        for candidate in candidates:
            atlas.append({"arm": arm, "family": family, **candidate})
    return barcodes, selectors, atlas


def main() -> None:
    contract = core.load(OUT / "protocol/preregistration.json")
    contract_audit = core.load(OUT / "audit/independent_contract_audit.json")
    if contract["authorization"] != "run_phase1582_c102_c101_field_discovery" or not contract_audit["all_checks_passed"]:
        raise RuntimeError("C102 discovery authorization missing")
    if (OUT / "raw/qwen3_all_token_state_coordinate_field.uint16.npy").exists():
        raise RuntimeError("C102 raw field already exists; discovery must precede model capture")

    c101_final = core.load(C101 / "analysis/final.json")
    if not c101_final["all_checks_passed"]:
        raise RuntimeError("C101 source is not complete")
    graph_coeff_path = C101 / "raw/qwen3_confirmation_walsh_coefficients_v2.float32.npy"
    breadth_coeff_path = C101 / "raw/qwen3_breadth_walsh_coefficients_v2.float32.npy"
    graph_index_path = C101 / "raw/qwen3_confirmation_walsh_index_v2.jsonl"
    breadth_index_path = C101 / "raw/qwen3_breadth_walsh_index_v2.jsonl"
    graph_coeff = np.load(graph_coeff_path, mmap_mode="r")
    breadth_coeff = np.load(breadth_coeff_path, mmap_mode="r")
    graph_units = load_units(graph_index_path)
    breadth_units = load_units(breadth_index_path)

    graph_barcodes, graph_selectors, graph_atlas = process_arm(
        "graph",
        GRAPH_FAMILIES,
        graph_coeff,
        graph_units,
        c101_analysis.CONF_ROLES,
        c101_analysis.GRAPH_EFFECTS,
        "xy",
        "xycode",
    )
    breadth_barcodes, breadth_selectors, breadth_atlas = process_arm(
        "breadth",
        BREADTH_FAMILIES,
        breadth_coeff,
        breadth_units,
        c101_analysis.BREADTH_ROLES,
        c101_analysis.BREADTH_EFFECTS,
        "truth",
        "truth:code",
    )
    graph_path = OUT / "raw/c101_graph_frozen_barcodes.float32.npy"
    breadth_path = OUT / "raw/c101_breadth_frozen_barcodes.float32.npy"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(graph_path, graph_barcodes)
    np.save(breadth_path, breadth_barcodes)
    selectors = [*graph_selectors, *breadth_selectors]
    prediction = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "full_coordinate_barcode_and_formation_predictions_frozen_before_c102_capture",
        "source": {
            "campaign": "C101",
            "all_revealed_data_are_discovery_for_c102": True,
            "graph_coeff_sha256": core.sha(graph_coeff_path),
            "breadth_coeff_sha256": core.sha(breadth_coeff_path),
            "graph_index_sha256": core.sha(graph_index_path),
            "breadth_index_sha256": core.sha(breadth_index_path),
        },
        "barcodes": {
            "graph_path": str(graph_path.relative_to(ROOT)).replace("\\", "/"),
            "graph_sha256": core.sha(graph_path),
            "graph_shape": list(graph_barcodes.shape),
            "breadth_path": str(breadth_path.relative_to(ROOT)).replace("\\", "/"),
            "breadth_sha256": core.sha(breadth_path),
            "breadth_shape": list(breadth_barcodes.shape),
            "effect_slots": ["primary", "code", "primary_x_code"],
        },
        "selectors": selectors,
        "validation": {
            "response_discovery": "report all selectors and nested K; choose one K per family by maximum cosine above a coordinate-permutation q99 null",
            "confirmation": "freeze the response-discovery K and require cosine above its coordinate-permutation q99 null",
            "lockbox": "test the same selector and K once without modification against an independently seeded coordinate-permutation q99 null",
            "nested_k": list(NESTED_K),
            "null_draws": 2000,
            "confirmation_seed": 1584,
            "lockbox_seed": 1585,
            "failure_scope": "a failed family barcode is missing for that family; it does not erase descriptive full-field observations",
        },
        "intervention_authorization": "only families passing both frozen confirmation and lockbox barcode tests may enter natural-donor coordinate-coalition intervention",
        "claim_boundary": {
            "allowed": "Qwen activation-coordinate effect barcodes and layerwise formation trajectories",
            "forbidden": ["weight parameters", "semantic neurons", "attention/MLP mechanism", "cross-model law", "new mathematics"],
        },
        "authorization": "run_phase1583_c102_qwen_full_field_capture",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/frozen_coordinate_barcode_predictions.json", prediction)
    core.write_rows(OUT / "analysis/c101_selector_atlas.jsonl", [*graph_atlas, *breadth_atlas])
    summary = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "families": len(selectors),
        "graph_shape": list(graph_barcodes.shape),
        "breadth_shape": list(breadth_barcodes.shape),
        "selected": [{"arm": row["arm"], "family": row["family"], **row["selector"]} for row in selectors],
        "authorization": prediction["authorization"],
    }
    core.save(OUT / "analysis/c101_field_discovery_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
