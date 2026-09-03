#!/usr/bin/env python3
"""Phase1577 / C101: preregistered confirmation plus breadth observation."""
from __future__ import annotations

import argparse
import itertools
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
RESULT = TESTS / "result"
OUT = RESULT / "phase1575_c101_dual_arm"
C100 = RESULT / "phase1573_c100_graph_field_analysis_adapter"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1571_c098_observation_first_graph_campaign as graph_base

PHASE = 1577
CAMPAIGN = "C101"
STATES = 37
DIM = 2560
CONF_ROLES = ("target_pre", "target_post", "query_target", "query_endpoint", "code_instruction", "boundary")
BREADTH_ROLES = ("focus_pre", "focus_record", "focus_post", "query_focus", "query_anchor", "code_instruction", "boundary")
GRAPH_EFFECTS = graph_base.EFFECTS
BREADTH_FACTORS = ("truth", "surface", "distractor", "code")
BREADTH_EFFECT_MASKS = tuple(tuple(i for i in range(4) if mask & (1 << i)) for mask in range(1, 16))
BREADTH_EFFECTS = tuple(":".join(BREADTH_FACTORS[i] for i in mask) for mask in BREADTH_EFFECT_MASKS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    return float(np.dot(a64, b64) / denom) if denom else 0.0


def role_vector(field: np.ndarray, row: dict[str, Any], role: str) -> np.ndarray:
    left, right = row["role_offsets"][role]
    return np.asarray(field[:, left:right, :], dtype=np.float32).mean(axis=1, dtype=np.float64).astype(np.float32)


def prepare() -> None:
    parent = core.load(OUT / "analysis/qwen_capture_summary.json")
    capture_audit = core.load(OUT / "audit/independent_capture_audit.json")
    protocol = core.load(OUT / "protocol/preregistration.json")
    if parent["authorization"] != "run_phase1577_c101_analysis" or not capture_audit["all_checks_passed"]:
        raise RuntimeError("analysis authorization missing")
    adapter = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "analysis_adapter_frozen",
        "producer_sha256": core.sha(Path(__file__)),
        "source_raw_sha256": parent["raw_sha256"],
        "source_index_sha256": parent["index_sha256"],
        "primary": protocol["confirmation"]["primary"],
        "null": {"method": "within-unit balanced effect-sign permutation preserving raw cells", "draws": 1000, "seed": 1577, "quantile": 0.99},
        "breadth": {"effect": "truth", "status": "exploratory", "no_universal_gate": True},
        "authorization": "execute_c101_analysis",
    }
    core.save(OUT / "protocol/analysis_adapter.json", adapter)
    print(json.dumps(adapter, indent=2))


def compute_coefficients(field: np.ndarray, index: list[dict[str, Any]], arm: str, roles: tuple[str, ...], effects: tuple[str, ...], masks: tuple[tuple[int, ...], ...] | None) -> tuple[Path, list[dict[str, Any]]]:
    rows = [row for row in index if row["arm"] == arm]
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[row["unit_id"]].append(row)
    units = []
    for unit_id, unit_rows in by_unit.items():
        first = unit_rows[0]
        units.append({key: first[key] for key in ("unit_id", "unit_index", "arm", "family", "world", "partition", "surface")})
    units.sort(key=lambda row: row["unit_id"])
    path = OUT / f"raw/qwen3_{arm}_walsh_coefficients.float32.npy"
    coeff = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(units), len(effects), STATES, len(roles), DIM))
    for unit_index, unit in enumerate(units):
        unit_rows = by_unit[unit["unit_id"]]
        values = np.stack([np.stack([role_vector(field, row, role) for role in roles], axis=1) for row in unit_rows], axis=0)
        for effect_index, effect in enumerate(effects):
            if arm == "confirmation":
                signs = np.asarray([graph_base.effect_sign(row, effect) for row in unit_rows], dtype=np.float32)
            else:
                assert masks is not None
                names = [BREADTH_FACTORS[i] for i in masks[effect_index]]
                signs = np.asarray([
                    np.prod([row[{"truth": "truth_factor", "surface": "surface_factor", "distractor": "distractor_factor", "code": "code"}[name]] for name in names])
                    for row in unit_rows
                ], dtype=np.float32)
            coeff[unit_index, effect_index] = np.einsum("c,csrd->srd", signs, values, optimize=True) / 16.0
        if (unit_index + 1) % 12 == 0:
            print(f"[phase1577] {arm} coefficients {unit_index + 1}/{len(units)}", flush=True)
    coeff.flush()
    del coeff
    core.write_rows(OUT / f"raw/qwen3_{arm}_walsh_index.jsonl", [{"row_index": i, **row} for i, row in enumerate(units)])
    return path, units


def mean_coeff(coeff: np.ndarray, units: list[dict[str, Any]], effect_index: int, state: int, role_index: int, **filters: str) -> np.ndarray:
    selected = [row["row_index"] for row in units if all(row[key] == value for key, value in filters.items())]
    if not selected:
        raise RuntimeError(("empty coeff selection", filters))
    return np.asarray(coeff[selected, effect_index, state, role_index], dtype=np.float64).mean(axis=0)


def support_metrics(reference: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    fixed = np.argsort(np.abs(reference))[-64:]
    dynamic = np.argsort(np.abs(target))[-64:]
    return {
        "full_cosine": cosine(reference, target),
        "reference_norm": float(np.linalg.norm(reference)),
        "target_norm": float(np.linalg.norm(target)),
        "fixed_top64_cosine": cosine(reference[fixed], target[fixed]),
        "fixed_top64_sign_agreement": float(np.mean(np.sign(reference[fixed]) == np.sign(target[fixed]))),
        "dynamic_top64_jaccard": float(len(set(fixed) & set(dynamic)) / len(set(fixed) | set(dynamic))),
        "fixed_coordinates": [int(v) for v in fixed],
    }


def c100_reference() -> tuple[np.ndarray, list[dict[str, Any]]]:
    coeff = np.load(C100 / "raw/focus_role_walsh_coefficients.float32.npy", mmap_mode="r")
    units = core.rows(C100 / "raw/focus_role_walsh_index.jsonl")
    return coeff, units


def confirmation_validation(coeff: np.ndarray, units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    old_coeff, old_units = c100_reference()
    xy_old = GRAPH_EFFECTS.index("xy")
    xy_new = GRAPH_EFFECTS.index("xy")
    old_boundary = graph_base.FOCUS_ROLES.index("boundary")
    new_boundary = CONF_ROLES.index("boundary")
    rows = []
    for state in range(STATES):
        for world in graph_base.WORLDS:
            for family in graph_base.FAMILIES:
                reference = mean_coeff(old_coeff, old_units, xy_old, state, old_boundary, partition="response_discovery", world=world, family=family)
                for partition in ("confirmation", "lockbox"):
                    target = mean_coeff(coeff, units, xy_new, state, new_boundary, partition=partition, world=world, family=family)
                    rows.append({"state": state, "world": world, "family": family, "partition": partition, **support_metrics(reference, target)})
    primary_rows = [row for row in rows if row["state"] == 24]
    primary_pass_count = sum(row["full_cosine"] >= 0.50 for row in primary_rows)
    primary = {
        "required": 24,
        "passed": primary_pass_count,
        "all_passed": primary_pass_count == 24,
        "threshold": 0.50,
        "median_cosine": float(np.median([row["full_cosine"] for row in primary_rows])),
        "minimum_cosine": float(min(row["full_cosine"] for row in primary_rows)),
        "maximum_cosine": float(max(row["full_cosine"] for row in primary_rows)),
    }
    secondary = []
    for state in (24, 31, 32):
        selected = [row for row in rows if row["state"] == state]
        secondary.append({
            "state": state,
            "median_cosine": float(np.median([row["full_cosine"] for row in selected])),
            "minimum_cosine": float(min(row["full_cosine"] for row in selected)),
            "count_ge_050": sum(row["full_cosine"] >= 0.50 for row in selected),
            "median_fixed_sign": float(np.median([row["fixed_top64_sign_agreement"] for row in selected])),
            "median_dynamic_jaccard": float(np.median([row["dynamic_top64_jaccard"] for row in selected])),
            "secondary_median_ge_070": float(np.median([row["full_cosine"] for row in selected])) >= 0.70 if state in (31, 32) else None,
        })
    del old_coeff
    return rows, primary, secondary


def geometry(coeff: np.ndarray, units: list[dict[str, Any]], effect_index: int, role_index: int, states: tuple[int, ...], worlds: tuple[str, ...] | None) -> list[dict[str, Any]]:
    output = []
    for partition in graph_base.PARTITIONS:
        for state in states:
            if worlds is not None:
                world_cos = []
                family_cos = []
                for family in graph_base.FAMILIES:
                    vectors = {world: mean_coeff(coeff, units, effect_index, state, role_index, partition=partition, world=world, family=family) for world in worlds}
                    world_cos.extend(cosine(vectors[a], vectors[b]) for a, b in itertools.combinations(worlds, 2))
                for world in worlds:
                    vectors = {family: mean_coeff(coeff, units, effect_index, state, role_index, partition=partition, world=world, family=family) for family in graph_base.FAMILIES}
                    family_cos.extend(cosine(vectors[a], vectors[b]) for a, b in itertools.combinations(graph_base.FAMILIES, 2))
                output.append({"partition": partition, "state": state, "minimum_world_cosine": min(world_cos), "median_world_cosine": float(np.median(world_cos)), "minimum_family_cosine": min(family_cos), "median_family_cosine": float(np.median(family_cos)), "world_min_exceeds_family_min": min(world_cos) > min(family_cos)})
            else:
                vectors = {family: mean_coeff(coeff, units, effect_index, state, role_index, partition=partition, family=family) for family in sorted({row["family"] for row in units})}
                pairwise = [cosine(vectors[a], vectors[b]) for a, b in itertools.combinations(vectors, 2)]
                output.append({"partition": partition, "state": state, "minimum_family_cosine": min(pairwise), "median_family_cosine": float(np.median(pairwise)), "maximum_family_cosine": max(pairwise)})
    return output


def effect_atlas(coeff: np.ndarray, units: list[dict[str, Any]], effects: tuple[str, ...], roles: tuple[str, ...], states: tuple[int, ...]) -> list[dict[str, Any]]:
    output = []
    for partition in graph_base.PARTITIONS:
        selected = [row["row_index"] for row in units if row["partition"] == partition]
        for state in states:
            for role_index, role in enumerate(roles):
                for effect_index, effect in enumerate(effects):
                    vector = np.asarray(coeff[selected, effect_index, state, role_index], dtype=np.float64).mean(axis=0)
                    output.append({"partition": partition, "state": state, "role": role, "effect": effect, "norm": float(np.linalg.norm(vector)), "max_abs": float(np.max(np.abs(vector)))})
    return output


def breadth_validation(coeff: np.ndarray, units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    effect_index = BREADTH_EFFECTS.index("truth")
    rows = []
    families = sorted({row["family"] for row in units})
    for family in families:
        for state in range(STATES):
            for role_index, role in enumerate(BREADTH_ROLES):
                reference = mean_coeff(coeff, units, effect_index, state, role_index, partition="response_discovery", family=family)
                for partition in ("confirmation", "lockbox"):
                    target = mean_coeff(coeff, units, effect_index, state, role_index, partition=partition, family=family)
                    rows.append({"family": family, "state": state, "role": role, "partition": partition, **support_metrics(reference, target)})
    best = {}
    for family in families:
        candidates = []
        for state in range(STATES):
            for role in BREADTH_ROLES:
                selected = [row for row in rows if row["family"] == family and row["state"] == state and row["role"] == role]
                candidates.append((min(row["full_cosine"] for row in selected), state, role, selected))
        score, state, role, selected = max(candidates, key=lambda value: value[0])
        best[family] = {"state": state, "role": role, "minimum_holdout_cosine": score, "median_holdout_cosine": float(np.median([row["full_cosine"] for row in selected])), "post_hoc": True}
    boundary_states = {}
    for state in (16, 24, 31, 32, 35, 36):
        selected = [row for row in rows if row["state"] == state and row["role"] == "boundary"]
        boundary_states[str(state)] = {"median_cosine": float(np.median([row["full_cosine"] for row in selected])), "minimum_cosine": float(min(row["full_cosine"] for row in selected)), "count_ge_050": sum(row["full_cosine"] >= 0.50 for row in selected), "total": len(selected)}
    return rows, {"post_hoc_best": best, "boundary_states": boundary_states}


def design_null(field: np.ndarray, index: list[dict[str, Any]], old_coeff: np.ndarray, old_units: list[dict[str, Any]], conf: bool, draws: int = 1000) -> list[dict[str, Any]]:
    rng = np.random.default_rng(1577 if conf else 1578)
    by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index:
        if (row["arm"] == "confirmation") == conf:
            by_unit[row["unit_id"]].append(row)
    output = []
    role = "boundary"
    state = 24
    groups = sorted({(row["world"], row["family"], row["partition"]) if conf else (row["family"], row["partition"]) for row in index if (row["arm"] == "confirmation") == conf and row["partition"] in ("confirmation", "lockbox")})
    for group in groups:
        if conf:
            world, family, partition = group
            unit_ids = sorted({row["unit_id"] for row in index if row["arm"] == "confirmation" and row["world"] == world and row["family"] == family and row["partition"] == partition})
            reference = mean_coeff(old_coeff, old_units, GRAPH_EFFECTS.index("xy"), state, graph_base.FOCUS_ROLES.index("boundary"), partition="response_discovery", world=world, family=family)
        else:
            family, partition = group
            unit_ids = sorted({row["unit_id"] for row in index if row["arm"] == "breadth" and row["family"] == family and row["partition"] == partition})
            discovery_ids = sorted({row["unit_id"] for row in index if row["arm"] == "breadth" and row["family"] == family and row["partition"] == "response_discovery"})
            vectors = []
            for unit_id in discovery_ids:
                unit_rows = by_unit[unit_id]
                values = np.stack([role_vector(field, row, role)[state] for row in unit_rows])
                signs = np.asarray([row["truth_factor"] for row in unit_rows], dtype=np.float64)
                vectors.append(signs @ values / 16.0)
            reference = np.mean(vectors, axis=0)
        observed_vectors = []
        null_sum = np.zeros((draws, DIM), dtype=np.float32)
        for unit_id in unit_ids:
            unit_rows = by_unit[unit_id]
            values = np.stack([role_vector(field, row, role)[state] for row in unit_rows]).astype(np.float32)
            signs = np.asarray([row["x"] * row["y"] if conf else row["truth_factor"] for row in unit_rows], dtype=np.float32)
            observed_vectors.append(signs @ values / 16.0)
            permuted = np.stack([rng.permutation(signs) for _ in range(draws)], axis=0)
            null_sum += permuted @ values / 16.0
        observed = np.mean(observed_vectors, axis=0)
        null_vectors = null_sum / len(unit_ids)
        ref_norm = np.linalg.norm(reference)
        null_norms = np.linalg.norm(null_vectors, axis=1)
        null_cos = (null_vectors @ reference) / np.maximum(null_norms * ref_norm, 1e-30)
        observed_cos = cosine(reference, observed)
        q99 = float(np.quantile(null_cos, 0.99))
        output.append({"arm": "confirmation" if conf else "breadth", "group": list(group), "state": state, "role": role, "observed_cosine": observed_cos, "null_q99": q99, "beats_q99": observed_cos > q99, "upper_p": float((1 + np.sum(null_cos >= observed_cos)) / (draws + 1)), "draws": draws})
    return output


def analyze() -> None:
    adapter = core.load(OUT / "protocol/analysis_adapter.json")
    capture = core.load(OUT / "analysis/qwen_capture_summary.json")
    if adapter["authorization"] != "execute_c101_analysis" or adapter["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("analysis producer changed or unauthorized")
    raw_path = OUT / "raw/qwen3_registered_role_field.float16.npy"
    index_path = OUT / "raw/qwen3_registered_role_index.jsonl"
    if core.sha(raw_path) != adapter["source_raw_sha256"] or core.sha(index_path) != adapter["source_index_sha256"]:
        raise RuntimeError("source field changed")
    field = np.load(raw_path, mmap_mode="r")
    index = core.rows(index_path)
    conf_path, conf_units0 = compute_coefficients(field, index, "confirmation", CONF_ROLES, GRAPH_EFFECTS, None)
    breadth_path, breadth_units0 = compute_coefficients(field, index, "breadth", BREADTH_ROLES, BREADTH_EFFECTS, BREADTH_EFFECT_MASKS)
    conf_coeff = np.load(conf_path, mmap_mode="r")
    breadth_coeff = np.load(breadth_path, mmap_mode="r")
    conf_units = [{"row_index": i, **row} for i, row in enumerate(conf_units0)]
    breadth_units = [{"row_index": i, **row} for i, row in enumerate(breadth_units0)]
    validation, primary, secondary = confirmation_validation(conf_coeff, conf_units)
    conf_geometry = geometry(conf_coeff, conf_units, GRAPH_EFFECTS.index("xy"), CONF_ROLES.index("boundary"), (24, 31, 32), graph_base.WORLDS)
    conf_effects = effect_atlas(conf_coeff, conf_units, GRAPH_EFFECTS, CONF_ROLES, (0, 8, 16, 24, 31, 32, 35, 36))
    breadth_validation_rows, breadth_summary = breadth_validation(breadth_coeff, breadth_units)
    breadth_geometry = geometry(breadth_coeff, breadth_units, BREADTH_EFFECTS.index("truth"), BREADTH_ROLES.index("boundary"), tuple(range(STATES)), None)
    breadth_effects = effect_atlas(breadth_coeff, breadth_units, BREADTH_EFFECTS, BREADTH_ROLES, (0, 8, 16, 24, 31, 32, 35, 36))
    old_coeff, old_units = c100_reference()
    conf_null = design_null(field, index, old_coeff, old_units, True, adapter["null"]["draws"])
    breadth_null = design_null(field, index, old_coeff, old_units, False, adapter["null"]["draws"])
    del old_coeff
    core.write_rows(OUT / "analysis/c101_confirmation_validation.jsonl", validation)
    core.write_rows(OUT / "analysis/c101_confirmation_geometry.jsonl", conf_geometry)
    core.write_rows(OUT / "analysis/c101_confirmation_effect_atlas.jsonl", conf_effects)
    core.write_rows(OUT / "analysis/c101_breadth_validation.jsonl", breadth_validation_rows)
    core.write_rows(OUT / "analysis/c101_breadth_geometry.jsonl", breadth_geometry)
    core.write_rows(OUT / "analysis/c101_breadth_effect_atlas.jsonl", breadth_effects)
    core.write_rows(OUT / "analysis/c101_design_preserving_null.jsonl", [*conf_null, *breadth_null])
    important = primary["all_passed"] or any(value["minimum_holdout_cosine"] >= 0.50 for value in breadth_summary["post_hoc_best"].values())
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "c101_dual_arm_analysis_complete",
        "behavior": capture["behavior"],
        "c099_behavior_correction": capture["c099_recalibration"],
        "confirmation": {"primary": primary, "secondary": secondary, "geometry": conf_geometry, "design_null": conf_null},
        "breadth": {**breadth_summary, "state24_design_null": breadth_null},
        "coefficients": {
            "confirmation": {"shape": list(conf_coeff.shape), "sha256": core.sha(conf_path), "effects": list(GRAPH_EFFECTS), "roles": list(CONF_ROLES)},
            "breadth": {"shape": list(breadth_coeff.shape), "sha256": core.sha(breadth_path), "effects": list(BREADTH_EFFECTS), "roles": list(BREADTH_ROLES)},
        },
        "flags": {"important_visualization": important, "behavior_missingness": "M_BEHAVIOR", "human_naturalness_missingness": "M_HUMAN_NATURALNESS"},
        "claim_boundary": {
            "allowed": "preregistered fresh Qwen state24 activation-field confirmation and exploratory four-pattern breadth atlas",
            "forbidden": ["correct reasoning mechanism", "weight parameters", "semantic neurons", "causal necessity/sufficiency", "cross-model invariant", "new mathematics"],
        },
        "authorization": "export_c101_parameter_level_heatmap" if important else "close_c101_without_visualization",
        "finished_at_utc": now(),
    }
    checks = {
        "source": list(field.shape) == capture["shape"],
        "confirmation_coeff": list(conf_coeff.shape) == [72, 15, 37, 6, 2560],
        "breadth_coeff": list(breadth_coeff.shape) == [48, 15, 37, 7, 2560],
        "validation": len(validation) == 888,
        "breadth_validation": len(breadth_validation_rows) == 2072,
        "null": len(conf_null) == 24 and len(breadth_null) == 8,
        "finite": all(math.isfinite(v) for row in validation for k, v in row.items() if isinstance(v, float)),
        "behavior_corrected": capture["c099_recalibration"]["corrected_accuracy"] == 0.5,
        "scope": result["flags"]["behavior_missingness"] == "M_BEHAVIOR",
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    final = {"checks": checks, "passed": sum(checks.values()), "total": len(checks), "all_checks_passed": all(checks.values()), "result": result}
    core.save(OUT / "analysis/c101_analysis_summary.json", result)
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "analyze"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else analyze()


if __name__ == "__main__":
    main()
