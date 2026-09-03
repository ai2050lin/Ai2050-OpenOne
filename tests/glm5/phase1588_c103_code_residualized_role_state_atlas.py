#!/usr/bin/env python3
"""Phase1588 / C103: existing-data code-residualized role/state observation atlas."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
C101 = TESTS / "result/phase1575_c101_dual_arm"
C102 = TESTS / "result/phase1581_c102_typed_relation_coordinate_campaign"
OUT = TESTS / "result/phase1588_c103_code_residualized_role_state_atlas"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
import phase1577_c101_dual_arm_analysis as c101_analysis

PHASE = 1588
CAMPAIGN = "C103"
DIM = 2560
STATES = 37
GRAPH_FAMILIES = ("taxonomy", "containment", "comparison", "precedence")
BREADTH_FAMILIES = ("attribute_binding", "agent_patient", "negation_scope", "whole_part_exception")
DATASETS = ("c101_revealed", "c102_confirmation", "c102_lockbox")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def residualize(primary: np.ndarray, code: np.ndarray, interaction: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    primary = np.asarray(primary, dtype=np.float64)
    controls = np.stack([np.asarray(code, dtype=np.float64), np.asarray(interaction, dtype=np.float64)], axis=1)
    coefficients, _, rank, singular = np.linalg.lstsq(controls, primary, rcond=None)
    residual = primary - controls @ coefficients
    primary_norm = float(np.linalg.norm(primary))
    return residual.astype(np.float32), {
        "control_rank": int(rank),
        "control_singular_values": singular.tolist(),
        "coefficients": coefficients.tolist(),
        "primary_norm": primary_norm,
        "residual_norm": float(np.linalg.norm(residual)),
        "retained_fraction": float(np.linalg.norm(residual) / primary_norm) if primary_norm else 0.0,
    }


def prepare() -> None:
    final = core.load(C102 / "analysis/final.json")
    audit = core.load(C102 / "audit/independent_final_audit.json")
    if final["authorization"] != "append_phase1587_c102_memo_then_run_c103_existing_data_observation" or not audit["all_checks_passed"]:
        raise RuntimeError("C103 authorization missing")
    paths = {
        "c101_graph": C101 / "raw/qwen3_confirmation_walsh_coefficients_v2.float32.npy",
        "c101_breadth": C101 / "raw/qwen3_breadth_walsh_coefficients_v2.float32.npy",
        "c102_graph": C102 / "raw/qwen3_graph_three_effect_coefficients.float32.npy",
        "c102_breadth": C102 / "raw/qwen3_breadth_three_effect_coefficients.float32.npy",
    }
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "existing_data_observation_contract_frozen",
        "source_hashes": {name: core.sha(path) for name, path in paths.items()},
        "scope": "all eight families, every registered role, all 37 states and all 2560 activation coordinates",
        "datasets": list(DATASETS),
        "decomposition": "least-squares subtract primary projection onto span(code, primary_x_code)",
        "upstream_definition": "roles other than boundary/code_instruction at states 0 through 24",
        "outputs": ["full residual vectors", "retained norm", "three cross-dataset cosines", "control rank", "descriptive Pareto frontier"],
        "no_gate": True,
        "claim_boundary": [
            "all sources are already revealed and are discovery data",
            "linear residualization is descriptive, not causal purification",
            "a high residual cosine is a future prediction candidate, not confirmation",
            "activation coordinates are not weight parameters",
        ],
        "authorization": "execute_c103_existing_data_atlas",
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    print(json.dumps(protocol, indent=2))


def load_units(path: Path) -> list[dict[str, Any]]:
    return core.rows(path)


def mean_effect(coeff: np.ndarray, units: list[dict[str, Any]], family: str, effect: int, state: int, role: int, partition: str | None = None) -> np.ndarray:
    selected = [row["row_index"] for row in units if row["family"] == family and (partition is None or row["partition"] == partition)]
    return np.asarray(coeff[selected, effect, state, role], dtype=np.float64).mean(axis=0)


def pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = any(
            other is not row
            and other["minimum_residual_cosine"] >= row["minimum_residual_cosine"]
            and other["minimum_retained_fraction"] >= row["minimum_retained_fraction"]
            and (other["minimum_residual_cosine"] > row["minimum_residual_cosine"] or other["minimum_retained_fraction"] > row["minimum_retained_fraction"])
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (row["minimum_residual_cosine"], row["minimum_retained_fraction"]), reverse=True)


def process_arm(
    arm: str,
    families: tuple[str, ...],
    roles: tuple[str, ...],
    old_coeff: np.ndarray,
    old_units: list[dict[str, Any]],
    old_effects: tuple[str, ...],
    old_names: tuple[str, str, str],
    fresh_coeff: np.ndarray,
    fresh_units: list[dict[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    residuals = np.lib.format.open_memmap(OUT / f"raw/{arm}_residual_vectors.float32.npy", mode="w+", dtype=np.float32, shape=(len(families), len(roles), STATES, len(DATASETS), DIM))
    atlas = []
    selections = []
    old_indices = [old_effects.index(name) for name in old_names]
    for family_index, family in enumerate(families):
        family_rows = []
        for role_index, role in enumerate(roles):
            for state in range(STATES):
                vectors = [
                    [mean_effect(old_coeff, old_units, family, effect, state, role_index) for effect in old_indices],
                    [mean_effect(fresh_coeff, fresh_units, family, effect, state, role_index, "confirmation") for effect in range(3)],
                    [mean_effect(fresh_coeff, fresh_units, family, effect, state, role_index, "lockbox") for effect in range(3)],
                ]
                metrics = []
                raw_primary = []
                for dataset_index, (primary, code, interaction) in enumerate(vectors):
                    residual, detail = residualize(primary, code, interaction)
                    residuals[family_index, role_index, state, dataset_index] = residual
                    metrics.append(detail)
                    raw_primary.append(primary)
                residual_values = [np.asarray(residuals[family_index, role_index, state, index], dtype=np.float64) for index in range(3)]
                residual_cosines = [cosine(residual_values[0], residual_values[1]), cosine(residual_values[0], residual_values[2]), cosine(residual_values[1], residual_values[2])]
                row = {
                    "arm": arm,
                    "family": family,
                    "role": role,
                    "role_index": role_index,
                    "state": state,
                    "upstream": role not in ("boundary", "code_instruction") and state <= 24,
                    "raw_cosines": {"source_confirmation": cosine(raw_primary[0], raw_primary[1]), "source_lockbox": cosine(raw_primary[0], raw_primary[2]), "confirmation_lockbox": cosine(raw_primary[1], raw_primary[2])},
                    "residual_cosines": {"source_confirmation": residual_cosines[0], "source_lockbox": residual_cosines[1], "confirmation_lockbox": residual_cosines[2]},
                    "minimum_residual_cosine": min(residual_cosines),
                    "retained_fractions": {DATASETS[index]: metrics[index]["retained_fraction"] for index in range(3)},
                    "minimum_retained_fraction": min(detail["retained_fraction"] for detail in metrics),
                    "control_ranks": {DATASETS[index]: metrics[index]["control_rank"] for index in range(3)},
                }
                atlas.append(row)
                family_rows.append(row)
        upstream = [row for row in family_rows if row["upstream"]]
        best = max(upstream, key=lambda row: (row["minimum_residual_cosine"], row["minimum_retained_fraction"], -row["state"]))
        frontier = pareto(upstream)
        selections.append({"arm": arm, "family": family, "best_upstream": best, "pareto_frontier": [{key: row[key] for key in ("role", "role_index", "state", "minimum_residual_cosine", "minimum_retained_fraction")} for row in frontier]})
        print(f"[phase1588] {family}: {best['role']} S{best['state']} cos={best['minimum_residual_cosine']:.4f} retained={best['minimum_retained_fraction']:.4f}", flush=True)
    residuals.flush()
    return residuals, atlas, selections


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    if protocol["authorization"] != "execute_c103_existing_data_atlas" or protocol["producer_sha256"] != core.sha(Path(__file__)):
        raise RuntimeError("C103 analysis not authorized")
    c101_graph_path = C101 / "raw/qwen3_confirmation_walsh_coefficients_v2.float32.npy"
    c101_breadth_path = C101 / "raw/qwen3_breadth_walsh_coefficients_v2.float32.npy"
    c102_graph_path = C102 / "raw/qwen3_graph_three_effect_coefficients.float32.npy"
    c102_breadth_path = C102 / "raw/qwen3_breadth_three_effect_coefficients.float32.npy"
    for name, path in (("c101_graph", c101_graph_path), ("c101_breadth", c101_breadth_path), ("c102_graph", c102_graph_path), ("c102_breadth", c102_breadth_path)):
        if core.sha(path) != protocol["source_hashes"][name]:
            raise RuntimeError((name, "source changed"))
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    graph_residuals, graph_atlas, graph_selections = process_arm(
        "graph", GRAPH_FAMILIES, c101_analysis.CONF_ROLES,
        np.load(c101_graph_path, mmap_mode="r"), load_units(C101 / "raw/qwen3_confirmation_walsh_index_v2.jsonl"),
        c101_analysis.GRAPH_EFFECTS, ("xy", "code", "xycode"),
        np.load(c102_graph_path, mmap_mode="r"), load_units(C102 / "raw/qwen3_graph_three_effect_index.jsonl"),
    )
    breadth_residuals, breadth_atlas, breadth_selections = process_arm(
        "breadth", BREADTH_FAMILIES, c101_analysis.BREADTH_ROLES,
        np.load(c101_breadth_path, mmap_mode="r"), load_units(C101 / "raw/qwen3_breadth_walsh_index_v2.jsonl"),
        c101_analysis.BREADTH_EFFECTS, ("truth", "code", "truth:code"),
        np.load(c102_breadth_path, mmap_mode="r"), load_units(C102 / "raw/qwen3_breadth_three_effect_index.jsonl"),
    )
    atlas = [*graph_atlas, *breadth_atlas]
    selections = [*graph_selections, *breadth_selections]
    core.write_rows(OUT / "analysis/role_state_atlas.jsonl", atlas)
    core.write_rows(OUT / "analysis/upstream_candidate_frontiers.jsonl", selections)
    candidates = [row["best_upstream"] | {"arm": row["arm"], "family": row["family"]} for row in selections]
    result = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "code_residualized_role_state_observation_complete",
        "rows": len(atlas),
        "graph_residual_shape": list(graph_residuals.shape),
        "breadth_residual_shape": list(breadth_residuals.shape),
        "graph_residual_sha256": core.sha(OUT / "raw/graph_residual_vectors.float32.npy"),
        "breadth_residual_sha256": core.sha(OUT / "raw/breadth_residual_vectors.float32.npy"),
        "atlas_sha256": core.sha(OUT / "analysis/role_state_atlas.jsonl"),
        "candidates": candidates,
        "candidate_ranges": {"minimum_residual_cosine": [min(row["minimum_residual_cosine"] for row in candidates), max(row["minimum_residual_cosine"] for row in candidates)], "minimum_retained_fraction": [min(row["minimum_retained_fraction"] for row in candidates), max(row["minimum_retained_fraction"] for row in candidates)]},
        "claim_boundary": "post-hoc existing-data observations only; candidates are not fresh confirmations, causal semantic states or new mathematics",
        "authorization": "append_phase1588_c103_memo_and_preregister_fresh_validation_only_if_candidate_is_scientifically_useful",
    }
    core.save(OUT / "analysis/final.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "analyze"))
    args = parser.parse_args()
    prepare() if args.action == "prepare" else analyze()


if __name__ == "__main__":
    main()
