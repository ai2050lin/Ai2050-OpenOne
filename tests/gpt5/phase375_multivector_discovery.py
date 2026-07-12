#!/usr/bin/env python3
"""Evaluate frozen finite exact state templates on the Phase371C discovery split."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT / "tests/gpt5"))

from model_registry import get_model_spec  # noqa: E402
from phase371c_blind_vector_contrast import (  # noqa: E402
    BASE,
    CASES,
    MODELS,
    ROLE_NAMES,
    layer_file,
    model_pairs,
    static_roles,
)


PHASE371 = ROOT / "tests/gpt5/result/phase371_exact_vector_coactivity"
OUT = ROOT / "tests/gpt5/result/phase375_finite_exact_subgraphs"
PROTOCOL = OUT / "phase375_protocol.json"
BLIND_AUDIT = OUT / "phase375_blind_inventory_audit.json"
CONDITION_KEY = (
    PHASE371
    / "phase371c_discovery_mapping/private/phase371c_discovery_condition_key.jsonl"
)
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}
CYCLIC_ROLE = {
    "source_end": "query_end",
    "query_end": "answer_start",
    "answer_start": "current_generation",
    "current_generation": "source_end",
}
DIRECT_COMPONENTS = {
    "layer_input": "layer_input_all_positions",
    "attention_merge": "attention_output_all_positions",
    "post_attention": "post_attention_state_all_positions",
    "mlp_merge": "mlp_output_all_positions",
    "layer_output": "layer_output_all_positions",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def orthonormal_basis(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for vector in vectors:
        residual = vector.float().reshape(-1).clone()
        for direction in basis:
            residual = residual - torch.dot(residual, direction) * direction
        norm = torch.linalg.vector_norm(residual)
        if float(norm.item()) > 1e-8:
            basis.append(residual / norm)
    return basis


def projection_error(target: torch.Tensor, vectors: list[torch.Tensor]) -> tuple[float, int]:
    target = target.float().reshape(-1)
    norm = torch.linalg.vector_norm(target)
    if float(norm.item()) <= 1e-8:
        return 1.0, 0
    basis = orthonormal_basis(vectors)
    if not basis:
        return 1.0, 0
    projection = sum(
        (torch.dot(target, direction) * direction for direction in basis),
        torch.zeros_like(target),
    )
    return float((torch.linalg.vector_norm(target - projection) / norm).item()), len(basis)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-8:
        return 0.0
    return float((torch.dot(left, right) / denominator).item())


def condition_letter(value: str) -> str:
    letter = value.split("_", 1)[0]
    if letter not in {"A", "B", "C", "D"}:
        raise ValueError(value)
    return letter


def model_groups(model: str) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    key_rows = [row for row in read_jsonl(CONDITION_KEY) if row["model"] == model]
    key = {row["blind_case_id"]: row for row in key_rows}
    groups: dict[str, dict[str, Any]] = defaultdict(dict)
    mechanisms: dict[str, str] = {}
    for case in cases:
        mapping = key[case["blind_case_id"]]
        parallel = case["anonymous_parallel_group_id"]
        groups[parallel][condition_letter(mapping["contrast_condition"])] = case
        mechanisms[parallel] = mapping["mechanism_id"]
    if len(groups) != 22 or any(set(rows) != {"A", "B", "C", "D"} for rows in groups.values()):
        raise RuntimeError(f"Invalid discovery grouping for {model}")
    return dict(groups), mechanisms


def load_mechanism_data(
    model: str,
    mechanism: str,
    tokenizer: Any,
    groups: dict[str, dict[str, Any]],
    mechanisms: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    parallels = sorted(parallel for parallel, value in mechanisms.items() if value == mechanism)
    pair_rows, _manifests = model_pairs(model)
    data: dict[str, Any] = {}
    for group_index, parallel in enumerate(parallels, 1):
        condition_cases = groups[parallel]
        group_data: dict[str, Any] = {"depths": defaultdict(lambda: defaultdict(dict)), "vocab": defaultdict(dict)}
        static = {
            condition: static_roles(tokenizer, case)[0]
            for condition, case in condition_cases.items()
        }
        for pair in pair_rows:
            for generation_time in (0, 1, 2):
                for condition, case in condition_cases.items():
                    case_id = case["blind_case_id"]
                    source = torch.load(
                        layer_file(
                            pair["source_root"], model, case_id, generation_time, pair["source_layer"]
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    receiver = torch.load(
                        layer_file(
                            pair["receiver_root"], model, case_id, generation_time, pair["receiver_layer"]
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    positions = [*static[condition], int(source["sequence_length"]) - 1]
                    index = torch.tensor(positions, dtype=torch.long)
                    routes = {
                        route: source["component_vectors"][pointer][0]
                        .index_select(0, index)
                        .float()
                        for route, pointer in DIRECT_COMPONENTS.items()
                    }
                    receiver_output = (
                        receiver["component_vectors"]["layer_output_all_positions"][0]
                        .index_select(0, index)
                        .float()
                    )
                    group_data["depths"][pair["name"]][generation_time][condition] = {
                        "routes": routes,
                        "receiver": receiver_output,
                    }
        for generation_time in (0, 1, 2):
            for condition, case in condition_cases.items():
                meta = torch.load(
                    BASE
                    / "private/models"
                    / model
                    / case["blind_case_id"]
                    / f"time_{generation_time}/time_meta.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                group_data["vocab"][generation_time][condition] = (
                    meta["full_vocabulary_logits"].float().reshape(-1)
                )
        data[parallel] = group_data
        print(
            f"[{model}/{mechanism}] load groups {group_index}/{len(parallels)}",
            flush=True,
        )
    return data, parallels


def template_basis(
    cell: dict[str, Any],
    left: str,
    right: str,
    members: list[dict[str, str]],
    role_map: dict[str, str] | None = None,
) -> list[torch.Tensor]:
    vectors = []
    for member in members:
        role = role_map.get(member["role"], member["role"]) if role_map else member["role"]
        index = ROLE_INDEX[role]
        route = member["route"]
        vectors.append(
            cell[left]["routes"][route][index] - cell[right]["routes"][route][index]
        )
    return vectors


def evaluate_pair(
    group: dict[str, Any],
    wrong_group: dict[str, Any],
    depth: str,
    wrong_depth: str,
    template: list[dict[str, str]],
    left: str,
    right: str,
    gates: dict[str, Any],
) -> dict[str, Any]:
    target_index = ROLE_INDEX["current_generation"]
    target = (
        group["depths"][depth][2][left]["receiver"][target_index]
        - group["depths"][depth][2][right]["receiver"][target_index]
    )
    current = template_basis(group["depths"][depth][1], left, right, template)
    past = template_basis(group["depths"][depth][0], left, right, template)
    wrong_depth_basis = template_basis(
        group["depths"][wrong_depth][1], left, right, template
    )
    wrong_role_basis = template_basis(
        group["depths"][depth][1], left, right, template, CYCLIC_ROLE
    )
    wrong_group_basis = template_basis(
        wrong_group["depths"][depth][1], left, right, template
    )
    current_error, current_rank = projection_error(target, current)
    past_error, _ = projection_error(target, past)
    wrong_depth_error, _ = projection_error(target, wrong_depth_basis)
    wrong_role_error, _ = projection_error(target, wrong_role_basis)
    wrong_group_error, _ = projection_error(target, wrong_group_basis)
    history_error, history_rank = projection_error(target, [*current, *past])
    single_errors = [projection_error(target, [vector])[0] for vector in current]
    best_single_error = min(single_errors) if single_errors else 1.0
    history_gain = current_error - history_error

    vocab_future = group["vocab"][2][left] - group["vocab"][2][right]
    vocab_current = group["vocab"][1][left] - group["vocab"][1][right]
    vocab_past = group["vocab"][0][left] - group["vocab"][0][right]
    vocab_wrong_group = wrong_group["vocab"][1][left] - wrong_group["vocab"][1][right]
    vocab_current_cosine = cosine(vocab_current, vocab_future)
    vocab_past_cosine = cosine(vocab_past, vocab_future)
    vocab_wrong_group_cosine = cosine(vocab_wrong_group, vocab_future)
    vocab_context_pass = (
        vocab_current_cosine
        >= vocab_past_cosine + float(gates["minimum_vocab_persistence_margin_vs_past"])
        and vocab_current_cosine
        >= vocab_wrong_group_cosine
        + float(gates["minimum_vocab_persistence_margin_vs_wrong_group"])
    )
    component_gates = {
        "rank": current_rank >= int(gates["minimum_exact_basis_rank"]),
        "absolute_error": current_error <= float(gates["maximum_current_projection_error"]),
        "best_single": current_error
        <= best_single_error - float(gates["minimum_error_margin_vs_best_single"]),
        "past": current_error <= past_error - float(gates["minimum_error_margin_vs_past"]),
        "wrong_depth": current_error
        <= wrong_depth_error - float(gates["minimum_error_margin_vs_wrong_depth"]),
        "wrong_role": current_error
        <= wrong_role_error - float(gates["minimum_error_margin_vs_wrong_role"]),
        "wrong_group": current_error
        <= wrong_group_error - float(gates["minimum_error_margin_vs_wrong_group"]),
        "history": history_gain <= float(gates["maximum_history_gain"]),
        "vocab_context": vocab_context_pass,
    }
    return {
        "lexical_pair": f"{left}_{right}",
        "target_norm": float(torch.linalg.vector_norm(target).item()),
        "basis_vector_count": len(current),
        "current_basis_rank": current_rank,
        "history_basis_rank": history_rank,
        "current_error": current_error,
        "best_single_error": best_single_error,
        "past_error": past_error,
        "wrong_depth_error": wrong_depth_error,
        "wrong_role_error": wrong_role_error,
        "wrong_group_error": wrong_group_error,
        "history_error": history_error,
        "history_gain": history_gain,
        "vocab_current_future_cosine": vocab_current_cosine,
        "vocab_past_future_cosine": vocab_past_cosine,
        "vocab_wrong_group_future_cosine": vocab_wrong_group_cosine,
        "component_gates": component_gates,
        "pass": all(component_gates.values()),
    }


def process_model(model: str) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    audit = read_json(BLIND_AUDIT)
    if not audit["authorization"]["run_discovery_subgraph_gate"]:
        raise RuntimeError("Blind audit did not authorize discovery mapping")
    templates = protocol["object_separation"]["state_template_definitions"]
    gates = protocol["frozen_numeric_gates"]
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    groups, mechanisms = model_groups(model)
    group_rows = []
    pass_counts: Counter[tuple[str, str, str]] = Counter()
    failure_counts: Counter[str] = Counter()
    for mechanism in protocol["frozen_scope"]["mechanisms"]:
        data, parallels = load_mechanism_data(
            model, mechanism, tokenizer, groups, mechanisms
        )
        for group_index, parallel in enumerate(parallels):
            wrong_parallel = parallels[(group_index + 1) % len(parallels)]
            for depth_index, depth in enumerate(("early", "middle", "late")):
                wrong_depth = ("early", "middle", "late")[(depth_index + 1) % 3]
                for template_name, template in templates.items():
                    lexical_rows = [
                        evaluate_pair(
                            data[parallel],
                            data[wrong_parallel],
                            depth,
                            wrong_depth,
                            template,
                            left,
                            right,
                            gates,
                        )
                        for left, right in (("A", "B"), ("C", "D"))
                    ]
                    group_pass = all(row["pass"] for row in lexical_rows)
                    if group_pass:
                        pass_counts[(mechanism, depth, template_name)] += 1
                    for lexical in lexical_rows:
                        for gate_name, passed in lexical["component_gates"].items():
                            if not passed:
                                failure_counts[gate_name] += 1
                    group_rows.append(
                        {
                            "schema_version": "48.3.0",
                            "phase_id": "Phase375-Discovery",
                            "model": model,
                            "mechanism_id": mechanism,
                            "anonymous_parallel_group_id": parallel,
                            "wrong_group_control_id": wrong_parallel,
                            "relative_depth": depth,
                            "wrong_depth_control": wrong_depth,
                            "template": template_name,
                            "lexical_pairs": lexical_rows,
                            "group_pass": group_pass,
                            "projection_is_causal": False,
                            "vocab_context_is_subgraph_mediation": False,
                        }
                    )
        del data
        gc.collect()
    minimum = int(gates["minimum_independent_groups_per_model_mechanism_template"])
    model_candidates = [
        {
            "model": model,
            "mechanism_id": key[0],
            "relative_depth": key[1],
            "template": key[2],
            "group_pass_count": count,
            "minimum_group_count": minimum,
            "model_gate_pass": count >= minimum,
        }
        for key, count in sorted(pass_counts.items())
        if count >= minimum
    ]
    model_dir = OUT / "phase375_discovery/models" / model
    write_jsonl(model_dir / "private/phase375_group_rows.jsonl", group_rows)
    write_jsonl(model_dir / "phase375_model_candidates.jsonl", model_candidates)
    summary = {
        "schema_version": "48.3.0",
        "phase_id": "Phase375-Discovery",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "execution": {
            "device": "cpu",
            "model_execution": False,
            "reads_existing_cuda_collected_ledgers": True,
            "semantic_discovery_key_opened": True,
            "calibration_opened": False,
            "physical_opened": False,
        },
        "denominator": {
            "mechanism_count": 2,
            "parallel_group_count": 22,
            "relative_depth_count": 3,
            "state_template_count": len(templates),
            "group_candidate_count": len(group_rows),
            "lexical_evaluation_count": 2 * len(group_rows),
        },
        "results": {
            "model_candidate_count": len(model_candidates),
            "model_candidates": model_candidates,
            "failed_gate_counts": dict(sorted(failure_counts.items())),
        },
        "claim_boundary": {
            "candidate_is_language_path": False,
            "candidate_is_causal": False,
            "vocab_context_is_candidate_specific": False,
        },
    }
    write_json(model_dir / "phase375_model_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def merge_models() -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    summaries = [
        read_json(
            OUT
            / "phase375_discovery/models"
            / model
            / "phase375_model_summary.json"
        )
        for model in MODELS
    ]
    canonical: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for summary in summaries:
        for row in summary["results"]["model_candidates"]:
            canonical[
                (row["mechanism_id"], row["relative_depth"], row["template"])
            ].add(row["model"])
    cross_rows = []
    for key, models in sorted(canonical.items()):
        level2 = "glm4" in models and bool(models & {"qwen3", "deepseek7b"})
        level3 = models == set(MODELS)
        cross_rows.append(
            {
                "mechanism_id": key[0],
                "relative_depth": key[1],
                "template": key[2],
                "models": sorted(models),
                "heterogeneous_level2_pass": level2,
                "level3_pass": level3,
                "causal_replay_completed": False,
                "language_mechanism_claimed": False,
            }
        )
    level2_rows = [row for row in cross_rows if row["heterogeneous_level2_pass"]]
    level3_rows = [row for row in cross_rows if row["level3_pass"]]
    causal_authorized = bool(level2_rows)
    summary = {
        "schema_version": "48.4.0",
        "phase_id": "Phase375-Discovery-Merge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "merge_model_specific_finite_subgraph_gates_without_coordinate_alignment",
        "execution": {
            "model_order": list(MODELS),
            "model_execution": False,
            "calibration_opened": False,
            "physical_opened": False,
        },
        "denominator": {
            "model_count": 3,
            "total_group_candidate_count": sum(
                row["denominator"]["group_candidate_count"] for row in summaries
            ),
            "total_lexical_evaluation_count": sum(
                row["denominator"]["lexical_evaluation_count"] for row in summaries
            ),
            "model_candidate_count": sum(
                row["results"]["model_candidate_count"] for row in summaries
            ),
            "canonical_candidate_count": len(cross_rows),
        },
        "results": {
            "heterogeneous_level2_count": len(level2_rows),
            "level3_count": len(level3_rows),
            "causal_replay_authorized": causal_authorized,
            "language_path_candidate_count": 0,
            "language_mechanism_claimed": False,
        },
        "model_summaries": [
            {
                "model": row["model"],
                "model_candidate_count": row["results"]["model_candidate_count"],
                "failed_gate_counts": row["results"]["failed_gate_counts"],
            }
            for row in summaries
        ],
        "cross_model_rows": cross_rows,
        "authorization": {
            "run_discovery_same_graph_causal_replay": causal_authorized,
            "open_calibration": False,
            "open_physical": False,
            "single_neuron_scan": False,
        },
        "next_decision": (
            "run_preregistered_discovery_same_graph_causal_replay"
            if causal_authorized
            else "stop_and_reject_current_finite_state_templates"
        ),
        "protocol_hash": read_json(OUT / "phase375_discovery_execution_freeze.json")[
            "sealed_hashes"
        ]["protocol"],
    }
    write_jsonl(
        OUT / "phase375_discovery/phase375_cross_model_rows.jsonl", cross_rows
    )
    write_json(OUT / "phase375_discovery/phase375_discovery_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_models()
        return
    if not args.model:
        raise SystemExit("Use --model MODEL or --merge")
    process_model(args.model)


if __name__ == "__main__":
    main()
