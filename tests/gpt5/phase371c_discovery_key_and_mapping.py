#!/usr/bin/env python3
"""Map sealed blind rows to discovery-only A/B/C/D semantics under frozen gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
GATE = PHASE371 / "phase371c_semantic_discovery_gate.json"
ROWS = PHASE371 / "phase371c_blind_vector_contrast/private/phase371c_blind_route_contrasts.jsonl"
COLLECTOR_CASES = PHASE371 / "phase371c_behavior_analysis/private/phase371c_discovery_collector_cases.jsonl"
FULL_CASES = PHASE371 / "phase371c_case_bank/private/phase371c_execution_cases.jsonl"
OUT = PHASE371 / "phase371c_discovery_mapping"
MODELS = ("qwen3", "glm4", "deepseek7b")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_pass(row: dict[str, Any]) -> bool:
    indices = row["indices"]
    correct = float(indices["adjacent_output_direction_persistence"])
    return (
        float(indices["exact_difference_norm"]) > 0.0
        and float(indices["signed_cosine_to_source_output_difference"]) > 0.0
        and float(indices["child_parent_inner_product_share"]) > 0.0
        and correct > float(indices["wrong_depth_control_cosine"])
        and correct > float(indices["wrong_role_control_cosine"])
        and correct > float(indices["time_shuffle_control_cosine"])
    )


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    collector_ids = {row["blind_case_id"] for row in read_jsonl(COLLECTOR_CASES)}
    discovery_cases = [
        row for row in read_jsonl(FULL_CASES)
        if row["blind_case_id"] in collector_ids and row["phase371c_split"] == "fresh_discovery"
    ]
    if len(discovery_cases) != 264:
        raise RuntimeError(f"Expected 264 discovery condition rows, got {len(discovery_cases)}")
    discovery_key = [
        {
            "blind_case_id": row["blind_case_id"],
            "model": row["private_execution_model"],
            "anonymous_group_id": row["anonymous_group_id"],
            "anonymous_parallel_group_id": row["anonymous_parallel_group_id"],
            "anonymous_condition_slot": row["anonymous_condition_slot"],
            "mechanism_id": row["mechanism_id"],
            "contrast_condition": row["contrast_condition"],
        }
        for row in discovery_cases
    ]
    key_path = OUT / "private/phase371c_discovery_condition_key.jsonl"
    write_jsonl(key_path, discovery_key)
    slot_key = {
        (row["model"], row["anonymous_condition_slot"]): row
        for row in discovery_key
    }
    route_rows: dict[tuple[Any, ...], dict[frozenset[str], dict[str, Any]]] = defaultdict(dict)
    group_pair_metrics: dict[tuple[Any, ...], dict[frozenset[str], float]] = defaultdict(dict)
    for line in ROWS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        left = slot_key[(row["model"], row["anonymous_slot_left"])]
        right = slot_key[(row["model"], row["anonymous_slot_right"])]
        conditions = frozenset((left["contrast_condition"][0], right["contrast_condition"][0]))
        route_key = (
            row["model"], row["anonymous_parallel_group_id"], row["generation_time"],
            row["depth_pair"], row["role"], row["route"],
        )
        route_rows[route_key][conditions] = row
        group_pair_metrics[route_key][conditions] = float(
            row["indices"]["adjacent_output_direction_persistence"]
        )
    required_pairs = {frozenset(("A", "B")), frozenset(("C", "D"))}
    other_pairs = {
        frozenset(("A", "C")), frozenset(("A", "D")),
        frozenset(("B", "C")), frozenset(("B", "D")),
    }
    group_pass_rows = []
    candidate_counts = Counter()
    mechanism_by_parallel = {
        row["anonymous_parallel_group_id"]: row["mechanism_id"] for row in discovery_key
    }
    for route_key, by_pair in route_rows.items():
        if set(by_pair) != required_pairs | other_pairs:
            raise RuntimeError(f"Incomplete semantic six-pair route: {route_key}")
        lexical_pass = all(row_pass(by_pair[pair]) for pair in required_pairs)
        correct_mean = sum(group_pair_metrics[route_key][pair] for pair in required_pairs) / 2
        other_mean = sum(group_pair_metrics[route_key][pair] for pair in other_pairs) / 4
        pairing_control_pass = correct_mean > other_mean
        group_pass = lexical_pass and pairing_control_pass
        model, parallel, generation_time, depth, role, route = route_key
        mechanism = mechanism_by_parallel[parallel]
        canonical = (model, mechanism, generation_time, depth, role, route)
        if group_pass:
            candidate_counts[canonical] += 1
        group_pass_rows.append({
            "model": model,
            "mechanism_id": mechanism,
            "anonymous_parallel_group_id": parallel,
            "generation_time": generation_time,
            "depth_pair": depth,
            "role": role,
            "route": route,
            "lexical_replication_pass": lexical_pass,
            "pairing_control_pass": pairing_control_pass,
            "correct_pair_mean_persistence": correct_mean,
            "other_pair_mean_persistence": other_mean,
            "group_pass": group_pass,
        })
    minimum = int(gate["replication_gate"]["minimum_independent_discovery_groups_per_model_mechanism"])
    model_candidates = [
        {
            "model": model,
            "mechanism_id": mechanism,
            "generation_time": generation_time,
            "depth_pair": depth,
            "role": role,
            "route": route,
            "independent_group_pass_count": count,
            "minimum_group_count": minimum,
            "provisional_model_gate_pass": count >= minimum,
        }
        for (model, mechanism, generation_time, depth, role, route), count in sorted(candidate_counts.items())
        if count >= minimum
    ]
    by_canonical: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in model_candidates:
        canonical = (
            row["mechanism_id"], row["generation_time"], row["depth_pair"],
            row["role"], row["route"],
        )
        by_canonical[canonical].add(row["model"])
    cross_model_rows = []
    for canonical, models in sorted(by_canonical.items()):
        heterogeneous_level2 = "glm4" in models and bool(models & {"qwen3", "deepseek7b"})
        level3 = models == set(MODELS)
        cross_model_rows.append({
            "mechanism_id": canonical[0],
            "generation_time": canonical[1],
            "depth_pair": canonical[2],
            "role": canonical[3],
            "route": canonical[4],
            "models": sorted(models),
            "provisional_heterogeneous_level2": heterogeneous_level2,
            "provisional_level3": level3,
            "history_residual_gate_pass": None,
            "exact_replay_confirmation_pass": None,
            "full_candidate_pass": False,
        })
    provisional_level2 = sum(row["provisional_heterogeneous_level2"] for row in cross_model_rows)
    provisional_level3 = sum(row["provisional_level3"] for row in cross_model_rows)
    summary = {
        "schema_version": "47.20.0",
        "phase_id": "Phase371C-Discovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "map_frozen_blind_indices_to_discovery_only_condition_semantics_under_preregistered_gates",
        "input_hashes": {
            "gate": sha256_file(GATE),
            "blind_rows": sha256_file(ROWS),
            "discovery_condition_key": sha256_file(key_path),
        },
        "denominator": {
            "discovery_case_count": len(discovery_cases),
            "semantic_route_group_count": len(route_rows),
            "group_gate_row_count": len(group_pass_rows),
            "provisional_model_candidate_count": len(model_candidates),
            "canonical_cross_model_route_count": len(cross_model_rows),
        },
        "results": {
            "provisional_heterogeneous_level2_count": provisional_level2,
            "provisional_level3_count": provisional_level3,
            "history_residual_gate_completed": False,
            "exact_replay_confirmation_completed": False,
            "full_candidate_language_path_count": 0,
            "calibration_authorized": False,
            "language_mechanism_claimed": False,
        },
        "claim_boundary": {
            "semantic_key_scope": "fresh_discovery_only",
            "calibration_condition_key_opened": False,
            "physical_condition_key_opened": False,
            "navigation_indices_are_not_terminal_states": True,
        },
        "next_decision": "run_exact_history_residual_and_replay_confirmation_only_on_all_gate_pass_routes",
    }
    write_json(OUT / "phase371c_discovery_mapping_summary.json", summary)
    write_jsonl(OUT / "private/phase371c_group_gate_rows.jsonl", group_pass_rows)
    write_jsonl(OUT / "private/phase371c_provisional_model_candidates.jsonl", model_candidates)
    write_jsonl(OUT / "private/phase371c_cross_model_routes.jsonl", cross_model_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
