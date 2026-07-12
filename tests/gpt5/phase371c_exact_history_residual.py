#!/usr/bin/env python3
"""Evaluate exact t0/t1 route spans against t2 future vectors for provisional routes."""

from __future__ import annotations

import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase365_dynamic_bundle_extraction import load_weight  # noqa: E402
from phase371c_blind_vector_contrast import (  # noqa: E402
    BASE, CASES, DEPTH_NAMES, MODELS, derive_routes, layer_file,
    model_pairs, static_roles,
)


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
PROTOCOL = PHASE371 / "phase371c_history_residual_protocol.json"
DISCOVERY_KEY = PHASE371 / "phase371c_discovery_mapping/private/phase371c_discovery_condition_key.jsonl"
GROUP_GATES = PHASE371 / "phase371c_discovery_mapping/private/phase371c_group_gate_rows.jsonl"
MODEL_CANDIDATES = PHASE371 / "phase371c_discovery_mapping/private/phase371c_provisional_model_candidates.jsonl"
OUT = PHASE371 / "phase371c_exact_history_residual"
ROLE_INDEX = {"source_end": 0, "query_end": 1, "answer_start": 2, "current_generation": 3}


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


def projection_error(target: torch.Tensor, basis: list[torch.Tensor]) -> tuple[float, bool]:
    target = target.float()
    target_norm = torch.linalg.vector_norm(target)
    if float(target_norm.item()) <= 1e-8:
        return 1.0, False
    orthonormal = []
    for vector in basis:
        residual = vector.float().clone()
        for direction in orthonormal:
            residual = residual - torch.dot(residual, direction) * direction
        norm = torch.linalg.vector_norm(residual)
        if float(norm.item()) > 1e-8:
            orthonormal.append(residual / norm)
    if not orthonormal:
        return 1.0, False
    projection = sum((torch.dot(target, direction) * direction for direction in orthonormal), torch.zeros_like(target))
    error = float((torch.linalg.vector_norm(target - projection) / target_norm).item())
    return error, True


def model_case_groups(model: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(CASES):
        if row["private_execution_model"] == model:
            groups[row["anonymous_parallel_group_id"]].append(row)
    return groups


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    candidates = [row for row in read_jsonl(MODEL_CANDIDATES) if row["generation_time"] == 1]
    group_gates = {
        (
            row["model"], row["mechanism_id"], row["anonymous_parallel_group_id"],
            row["generation_time"], row["depth_pair"], row["role"], row["route"],
        ): bool(row["group_pass"])
        for row in read_jsonl(GROUP_GATES)
    }
    condition_key = {
        (row["model"], row["anonymous_condition_slot"]): row["contrast_condition"][0]
        for row in read_jsonl(DISCOVERY_KEY)
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    group_results = []
    model_counts = Counter()
    for model in MODELS:
        model_candidates = [row for row in candidates if row["model"] == model]
        if not model_candidates:
            continue
        candidates_by_depth: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in model_candidates:
            candidates_by_depth[row["depth_pair"]].append(row)
        spec = get_model_spec(model)
        tokenizer = AutoTokenizer.from_pretrained(
            str(spec.local_dir), trust_remote_code=spec.trust_remote_code,
            local_files_only=True, use_fast=False,
        )
        pair_contracts, _ = model_pairs(model)
        pair_by_name = {row["name"]: row for row in pair_contracts}
        weights = {}
        for depth in candidates_by_depth:
            layer = pair_by_name[depth]["source_layer"]
            weights[depth] = (
                load_weight(model, f"model.layers.{layer}.self_attn.o_proj.weight").to(device=device, dtype=torch.float32),
                load_weight(model, f"model.layers.{layer}.mlp.down_proj.weight").to(device=device, dtype=torch.float32),
            )
        for group_index, (parallel, cases) in enumerate(sorted(model_case_groups(model).items()), 1):
            condition_cases = {condition_key[(model, row["anonymous_condition_slot"])]: row for row in cases}
            if set(condition_cases) != {"A", "B", "C", "D"}:
                raise RuntimeError(f"Incomplete discovery condition map: {model}/{parallel}")
            static = {row["blind_case_id"]: static_roles(tokenizer, row)[0] for row in cases}
            for depth, depth_candidates in candidates_by_depth.items():
                relevant = [
                    candidate for candidate in depth_candidates
                    if group_gates.get((
                        model, candidate["mechanism_id"], parallel, 1, depth,
                        candidate["role"], candidate["route"],
                    ), False)
                ]
                if not relevant:
                    continue
                pair = pair_by_name[depth]
                o_weight, down_weight = weights[depth]
                data: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
                for generation_time in range(3):
                    for condition, case in condition_cases.items():
                        case_id = case["blind_case_id"]
                        source = torch.load(
                            layer_file(pair["source_root"], model, case_id, generation_time, pair["source_layer"]),
                            map_location="cpu", weights_only=True,
                        )
                        receiver = torch.load(
                            layer_file(pair["receiver_root"], model, case_id, generation_time, pair["receiver_layer"]),
                            map_location="cpu", weights_only=True,
                        )
                        positions = [*static[case_id], int(source["sequence_length"]) - 1]
                        routes, _errors = derive_routes(source, positions, o_weight, down_weight, device)
                        receiver_output = receiver["component_vectors"]["layer_output_all_positions"][0]
                        data[generation_time][condition] = {
                            "routes": routes,
                            "receiver": receiver_output.index_select(0, torch.tensor(positions)).float(),
                        }
                for candidate in relevant:
                    role_index = ROLE_INDEX[candidate["role"]]
                    lexical_rows = []
                    for left, right in (("A", "B"), ("C", "D")):
                        past = (
                            data[0][left]["routes"][candidate["route"]][role_index]
                            - data[0][right]["routes"][candidate["route"]][role_index]
                        )
                        current = (
                            data[1][left]["routes"][candidate["route"]][role_index]
                            - data[1][right]["routes"][candidate["route"]][role_index]
                        )
                        future = (
                            data[2][left]["receiver"][role_index]
                            - data[2][right]["receiver"][role_index]
                        )
                        past_error, past_valid = projection_error(future, [past])
                        current_error, current_valid = projection_error(future, [current])
                        history_error, history_valid = projection_error(future, [current, past])
                        history_gain = current_error - history_error
                        lexical_pass = (
                            past_valid and current_valid and history_valid
                            and current_error < past_error
                            and history_gain <= float(protocol["lexical_pair_gate"]["history_gain_max"])
                        )
                        lexical_rows.append({
                            "lexical_pair": f"{left}_{right}",
                            "past_error": past_error,
                            "current_error": current_error,
                            "history_error": history_error,
                            "history_gain": history_gain,
                            "pass": lexical_pass,
                        })
                    group_pass = all(row["pass"] for row in lexical_rows)
                    key = (
                        model, candidate["mechanism_id"], depth,
                        candidate["role"], candidate["route"],
                    )
                    if group_pass:
                        model_counts[key] += 1
                    group_results.append({
                        "model": model,
                        "mechanism_id": candidate["mechanism_id"],
                        "anonymous_parallel_group_id": parallel,
                        "generation_time": 1,
                        "depth_pair": depth,
                        "role": candidate["role"],
                        "route": candidate["route"],
                        "lexical_pairs": lexical_rows,
                        "history_group_pass": group_pass,
                    })
            if group_index % 8 == 0 or group_index == 22:
                print(f"[{model}] exact history groups {group_index}/22", flush=True)
        del weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    minimum = int(protocol["replication_gate"]["minimum_independent_groups_per_model_mechanism_route"])
    model_pass_rows = [
        {
            "model": key[0],
            "mechanism_id": key[1],
            "generation_time": 1,
            "depth_pair": key[2],
            "role": key[3],
            "route": key[4],
            "history_group_pass_count": count,
            "minimum_group_count": minimum,
            "history_model_gate_pass": count >= minimum,
        }
        for key, count in sorted(model_counts.items())
        if count >= minimum
    ]
    cross_models: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in model_pass_rows:
        canonical = (
            row["mechanism_id"], row["generation_time"], row["depth_pair"],
            row["role"], row["route"],
        )
        cross_models[canonical].add(row["model"])
    cross_rows = []
    for canonical, models in sorted(cross_models.items()):
        level2 = "glm4" in models and bool(models & {"qwen3", "deepseek7b"})
        level3 = models == set(MODELS)
        cross_rows.append({
            "mechanism_id": canonical[0],
            "generation_time": canonical[1],
            "depth_pair": canonical[2],
            "role": canonical[3],
            "route": canonical[4],
            "models": sorted(models),
            "history_heterogeneous_level2_pass": level2,
            "history_level3_pass": level3,
            "causal_same_graph_intervention_pass": None,
            "full_candidate_pass": False,
        })
    summary = {
        "schema_version": "47.22.0",
        "phase_id": "Phase371C-History",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "evaluate_exact_history_gain_on_t1_provisional_routes",
        "execution": {"device": str(device), "model_execution": False, "calibration_opened": False},
        "denominator": {
            "t1_provisional_model_candidate_count": len(candidates),
            "history_group_result_count": len(group_results),
            "history_model_pass_count": len(model_pass_rows),
            "history_cross_model_route_count": len(cross_rows),
        },
        "results": {
            "history_heterogeneous_level2_count": sum(row["history_heterogeneous_level2_pass"] for row in cross_rows),
            "history_level3_count": sum(row["history_level3_pass"] for row in cross_rows),
            "causal_same_graph_intervention_completed": False,
            "full_candidate_language_path_count": 0,
            "calibration_authorized": False,
            "language_mechanism_claimed": False,
        },
        "claim_boundary": {
            "history_projection_is_not_causal_replay": True,
            "t0_and_t2_candidates_rejected_from_history_gate": True,
            "calibration_and_physical_keys_opened": False,
        },
        "next_decision": "design_small_discovery_only_same_graph_intervention_replay_for_history_pass_routes",
    }
    write_json(OUT / "phase371c_exact_history_residual_summary.json", summary)
    write_jsonl(OUT / "private/phase371c_history_group_rows.jsonl", group_results)
    write_jsonl(OUT / "private/phase371c_history_model_passes.jsonl", model_pass_rows)
    write_jsonl(OUT / "private/phase371c_history_cross_model_routes.jsonl", cross_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
