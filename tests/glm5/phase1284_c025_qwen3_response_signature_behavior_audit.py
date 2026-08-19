#!/usr/bin/env python3
"""Independent pre/final audit for Phase1284."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "tests/glm5/result/phase1283_c025_response_signature_contract"
OUT = ROOT / "tests/glm5/result/phase1284_c025_qwen3_response_signature_behavior"
PROTOCOL = OUT / "protocol/preregistration.json"
MATERIAL = INPUT / "material/frozen_response_worlds.jsonl"
RAW = OUT / "raw/response_library_scores.jsonl"
GENERATION = OUT / "raw/confirmation_generations.jsonl"
FINAL = OUT / "analysis/final.json"
PREAUDIT = OUT / "audit/independent_preaudit.json"
FINAL_AUDIT = OUT / "audit/independent_final_audit.json"
ROLE_ORDER = ("expected_0", "expected_1", "opposite_0", "opposite_1", "control_0", "control_1")
PARTITION_SURFACES = {
    "discovery": ("test_confirmation", "forecast_agreement"),
    "selection": ("evidence_support", "outcome_match"),
    "confirmation": ("measurement_validation", "finding_consistency"),
}
TEMPLATE = np.asarray([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def check(name: str, passed: bool, detail: Any = None) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def unit(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1.0e-12 else np.zeros_like(value)


def preaudit() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    parent = json.loads((INPUT / "analysis/final.json").read_text(encoding="utf-8"))
    parent_audit = json.loads((INPUT / "audit/independent_final_audit.json").read_text(encoding="utf-8"))
    dependencies = {
        "phase1283_protocol": INPUT / "protocol/preregistration.json",
        "phase1283_material": INPUT / "material/frozen_response_worlds.jsonl",
        "phase1283_final": INPUT / "analysis/final.json",
        "phase1283_audit": INPUT / "audit/independent_final_audit.json",
    }
    checks = [
        check("phase_and_campaign", protocol["phase"] == 1284 and protocol["campaign"] == "C025"),
        check("parent_authorization", parent["authorization"] == "phase1284_qwen3_response_signature_behavior" and parent_audit["all_checks_passed"]),
        check("dependency_hashes", all(protocol["dependencies"][key] == file_sha256(path) for key, path in dependencies.items())),
        check("dimensions", protocol["row_count"] == 192 and protocol["context_count"] == 2304 and protocol["scored_sequence_count"] == 13824),
        check("roles_and_surfaces", tuple(protocol["role_order"]) == ROLE_ORDER and {key: tuple(value) for key, value in protocol["partition_surfaces"].items()} == PARTITION_SURFACES),
        check("axis_primary", any("Axis-level minima" in value for value in protocol["hard_stops"])),
        check("generation_frozen", protocol["generation"]["partition"] == "confirmation" and protocol["generation"]["max_new_tokens"] == 12 and protocol["generation"]["do_sample"] is False),
        check("single_run_and_stop", protocol["formal_run_budget"] == 1 and any("stops C025" in value for value in protocol["hard_stops"])),
    ]
    result = {
        "phase": 1284, "audit_type": "independent_preaudit", "checks": checks,
        "passed_count": sum(value["passed"] for value in checks), "check_count": len(checks),
        "all_checks_passed": all(value["passed"] for value in checks),
    }
    atomic_json(PREAUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


def response(keys: dict[tuple[str, str, str], dict[str, Any]], row_id: str, surface: str, right: str, left: str) -> np.ndarray:
    rv = np.asarray([keys[(row_id, surface, right)]["log_prob"][role] for role in ROLE_ORDER], dtype=np.float64)
    lv = np.asarray([keys[(row_id, surface, left)]["log_prob"][role] for role in ROLE_ORDER], dtype=np.float64)
    value = rv - lv
    return value - value.mean()


def recompute_behavior(raw: list[dict[str, Any]], material: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    keys = {(value["row_id"], value["surface"], value["panel"]): value for value in raw}
    signatures = {}
    for row in material:
        for surface in row["contexts"]:
            active = response(keys, row["row_id"], surface, "reversal", "consistency")
            lexical = response(keys, row["row_id"], surface, "lexical_consistency", "carrier_consistency")
            role = response(keys, row["row_id"], surface, "role_reversal", "role_consistency")
            target_scale = max(float(np.mean(np.abs(active[:4]))), 1.0e-12)
            signatures[(row["row_id"], surface)] = {
                "active": active,
                "effect": float(np.mean(active[2:4]) - np.mean(active[0:2])),
                "template_cosine": cosine(active, TEMPLATE),
                "lexical_ratio": float(np.linalg.norm(lexical) / max(np.linalg.norm(active), 1.0e-12)),
                "role_ratio": float(np.linalg.norm(role) / max(np.linalg.norm(active), 1.0e-12)),
                "control_leakage": float(np.mean(np.abs(active[4:6])) / target_scale),
            }
    axis_cells = {}
    for partition, surfaces in PARTITION_SURFACES.items():
        axes = sorted({row["axis"] for row in material if row["partition"] == partition})
        for surface in surfaces:
            for axis_name in axes:
                ids = [row["row_id"] for row in material if row["partition"] == partition and row["axis"] == axis_name]
                values = [signatures[(row_id, surface)] for row_id in ids]
                axis_cells[f"{partition}.{surface}.{axis_name}"] = {
                    "n_worlds": len(ids),
                    "positive_fraction": float(np.mean([value["effect"] > 0 for value in values])),
                    "median_effect": float(np.median([value["effect"] for value in values])),
                    "median_template_cosine": float(np.median([value["template_cosine"] for value in values])),
                    "median_lexical_ratio": float(np.median([value["lexical_ratio"] for value in values])),
                    "median_role_ratio": float(np.median([value["role_ratio"] for value in values])),
                    "median_control_leakage": float(np.median([value["control_leakage"] for value in values])),
                }
    paired = {}
    for partition, surfaces in PARTITION_SURFACES.items():
        axes = sorted({row["axis"] for row in material if row["partition"] == partition})
        for axis_name in axes:
            ids = [row["row_id"] for row in material if row["partition"] == partition and row["axis"] == axis_name]
            values = [cosine(signatures[(row_id, surfaces[0])]["active"], signatures[(row_id, surfaces[1])]["active"]) for row_id in ids]
            paired[f"{partition}.{axis_name}"] = {"n_worlds": len(ids), "median_cosine": float(np.median(values)), "minimum": float(np.min(values))}
    centroid = unit(np.mean([
        unit(signatures[(row["row_id"], surface)]["active"])
        for row in material if row["partition"] == "discovery" for surface in PARTITION_SURFACES["discovery"]
    ], axis=0))
    holdout = {}
    for partition in ("selection", "confirmation"):
        axes = sorted({row["axis"] for row in material if row["partition"] == partition})
        for axis_name in axes:
            values = [
                cosine(signatures[(row["row_id"], surface)]["active"], centroid)
                for row in material if row["partition"] == partition and row["axis"] == axis_name for surface in PARTITION_SURFACES[partition]
            ]
            holdout[f"{partition}.{axis_name}"] = {
                "n_signatures": len(values), "median_cosine": float(np.median(values)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0)), "minimum": float(np.min(values)),
            }
    gates = {
        "finite": float(np.mean([value["finite"] for value in raw])) >= thresholds["finite_fraction_min"],
        "active_positive": min(value["positive_fraction"] for value in axis_cells.values()) >= thresholds["active_positive_fraction_min"],
        "active_axis_median": min(value["median_effect"] for value in axis_cells.values()) >= thresholds["active_axis_median_min"],
        "template_cosine": min(value["median_template_cosine"] for value in axis_cells.values()) >= thresholds["template_cosine_axis_median_min"],
        "paired_surface": min(value["median_cosine"] for value in paired.values()) >= thresholds["paired_surface_cosine_median_min"],
        "holdout_centroid_cosine": min(value["median_cosine"] for value in holdout.values()) >= thresholds["holdout_centroid_cosine_axis_median_min"],
        "holdout_centroid_positive": min(value["positive_fraction"] for value in holdout.values()) >= thresholds["holdout_centroid_positive_fraction_min"],
        "lexical_null": max(value["median_lexical_ratio"] for value in axis_cells.values()) <= thresholds["lexical_null_norm_ratio_max"],
        "role_null": max(value["median_role_ratio"] for value in axis_cells.values()) <= thresholds["role_null_norm_ratio_max"],
        "control_leakage": max(value["median_control_leakage"] for value in axis_cells.values()) <= thresholds["control_leakage_ratio_max"],
    }
    extrema = {
        "active_positive_min": min(value["positive_fraction"] for value in axis_cells.values()),
        "active_effect_min": min(value["median_effect"] for value in axis_cells.values()),
        "template_cosine_min": min(value["median_template_cosine"] for value in axis_cells.values()),
        "paired_surface_cosine_min": min(value["median_cosine"] for value in paired.values()),
        "holdout_cosine_min": min(value["median_cosine"] for value in holdout.values()),
        "holdout_positive_min": min(value["positive_fraction"] for value in holdout.values()),
        "lexical_ratio_max": max(value["median_lexical_ratio"] for value in axis_cells.values()),
        "role_ratio_max": max(value["median_role_ratio"] for value in axis_cells.values()),
        "control_leakage_max": max(value["median_control_leakage"] for value in axis_cells.values()),
    }
    return {"axis_cells": axis_cells, "paired_surface": paired, "holdout": holdout, "centroid": centroid, "gates": gates, "extrema": extrema}


def final_audit() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines() if line.strip()]
    generations = [json.loads(line) for line in GENERATION.read_text(encoding="utf-8").splitlines() if line.strip()]
    material = [json.loads(line) for line in MATERIAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    recomputed = recompute_behavior(raw, material, protocol["thresholds"])
    generation_cells = {}
    for surface in PARTITION_SURFACES["confirmation"]:
        for panel in ("consistency", "reversal"):
            subset = [value for value in generations if value["surface"] == surface and value["panel"] == panel]
            parsed = [value for value in subset if value["parsed"]]
            generation_cells[f"{surface}.{panel}"] = {
                "n": len(subset), "coverage": float(np.mean([value["parsed"] for value in subset])),
                "accuracy_given_parsed": float(np.mean([value["correct"] for value in parsed])) if parsed else 0.0,
            }
    generation_gates = {
        "coverage": min(value["coverage"] for value in generation_cells.values()) >= protocol["thresholds"]["generation_coverage_min"],
        "accuracy": min(value["accuracy_given_parsed"] for value in generation_cells.values()) >= protocol["thresholds"]["generation_accuracy_min"],
    }
    expected_pass = all(recomputed["gates"].values()) and all(generation_gates.values())
    stored = final["behavior"]
    precision = final["precision_audit"]
    checks = [
        check("raw_and_generation_counts", len(raw) == 2304 and len(generations) == 256, {"raw": len(raw), "generation": len(generations)}),
        check("raw_keys_unique", len({(value["row_id"], value["surface"], value["panel"]) for value in raw}) == len(raw)),
        check("all_finite", all(value["finite"] for value in raw)),
        check("axis_cells", recomputed["axis_cells"] == stored["axis_cells"]),
        check("paired_surface", recomputed["paired_surface"] == stored["paired_surface"]),
        check("holdout", recomputed["holdout"] == stored["holdout"]),
        check("centroid", np.allclose(recomputed["centroid"], np.asarray(stored["discovery_centroid"], dtype=np.float64), atol=1.0e-12, rtol=0.0)),
        check("gate_extrema", recomputed["extrema"] == stored["gate_extrema"]),
        check("behavior_gates", recomputed["gates"] == stored["gates"]),
        check("generation_cells", generation_cells == final["generation"]["cells"]),
        check("generation_gates", generation_gates == final["generation"]["gates"]),
        check("verdict_and_authorization", (final["authorization"] == "phase1285_qwen3_typed_multievent_response_causality") == expected_pass),
        check("fp16", set(precision["parameter_dtypes"]) == {"float16"} and not precision["has_quantized_modules"] and not precision["has_bf16_parameters"]),
    ]
    result = {
        "phase": 1284, "audit_type": "independent_final_audit", "checks": checks,
        "passed_count": sum(value["passed"] for value in checks), "check_count": len(checks),
        "all_checks_passed": all(value["passed"] for value in checks), "scientific_gate_passed": expected_pass,
    }
    atomic_json(FINAL_AUDIT, result)
    print(canonical_json(result))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("pre", "final"))
    args = parser.parse_args()
    preaudit() if args.action == "pre" else final_audit()
