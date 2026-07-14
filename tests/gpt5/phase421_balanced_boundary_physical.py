#!/usr/bin/env python3
"""Trace qualified Phase421 source paths and independent MLP geometry.

Pass one uses only discovery groups to select typed source coordinates.  Pass
two replays the pre-registered development panel at those fixed coordinates.
The physical holdout remains sealed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from model_registry import get_model_spec  # noqa: E402
from phase358_multiresolution_component_conservation import (  # noqa: E402
    install_hooks,
    module_attr,
    relative_error,
)
from phase371b_anchor_qk_collection import capture_actual_qkv, repeat_key_value  # noqa: E402
from phase421_balanced_boundary_case_bank import (  # noqa: E402
    HISTORY_RELATIONS,
    MODELS,
    OUT,
    SCHEMA_VERSION,
    current_prompt,
    serialize_prompt,
)


PHASE_ID = "Phase421-BalancedBoundaryPhysical"
REGISTERED = OUT / "phase421_registered_conditions.jsonl"
AUTHORIZATION = OUT / "phase421_physical_development_authorization.json"
DEVELOPMENT_SPLITS = {"discovery", "calibration", "behavior_holdout"}
SOURCE_ROLES = (
    "history_answer",
    "current_evidence",
    "current_query",
    "history_length_control",
    "current_length_control",
    "query_length_control",
)
ROLE_CONTROL = {
    "history_answer": "history_length_control",
    "current_evidence": "current_length_control",
}
LEDGER_THRESHOLD = 0.01
NUMERIC_GEOMETRY_FLOOR = 1e-6


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase421 non-finite physical scalar: {value}")
    return round(float(value), 10)


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def fast_tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Phase421 requires a fast offset tokenizer for {model}")
    return tokenizer


def prompt_and_ids(loaded: Any, row: dict[str, Any]) -> tuple[str, dict[str, torch.Tensor], list[int]]:
    prompt = serialize_prompt(
        loaded.tokenizer,
        row["raw_prompt"],
        row["source_fragment"],
        row["interface"],
        row["history_answer"],
        int(row["current_support_count"]),
        int(row["history_reliability_score"]),
    )
    encoded = loaded.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    if len(ids) != int(row["registered_prompt_token_count"]):
        raise RuntimeError(f"Phase421 prompt contract changed: {row['phase421_condition_id']}")
    return prompt, {key: value.to(loaded.input_device) for key, value in encoded.items()}, ids


def all_char_spans(text: str, fragment: str, start: int, end: int) -> list[tuple[int, int]]:
    output = []
    cursor = start
    while cursor < end:
        found = text.find(fragment, cursor, end)
        if found < 0:
            break
        output.append((found, found + len(fragment)))
        cursor = found + max(1, len(fragment))
    if not output:
        raise RuntimeError(f"Fragment not found in registered interval: {fragment!r}")
    return output


def positions_for_spans(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int]],
) -> list[int]:
    output = []
    for index, (left, right) in enumerate(offsets):
        if right <= left:
            continue
        if any(left < end and right > start for start, end in spans):
            output.append(index)
    output = sorted(set(output))
    if not output:
        raise RuntimeError(f"No token positions for character spans: {spans}")
    return output


def matched_control_positions(
    length: int,
    token_count: int,
    excluded: set[int],
    preferred_start: int,
) -> list[int]:
    possible = list(range(max(0, preferred_start), max(0, token_count - length)))
    possible += list(range(0, max(0, preferred_start)))
    for start in possible:
        positions = list(range(start, start + length))
        if positions and positions[-1] < token_count - 1 and not any(pos in excluded for pos in positions):
            return positions
    available = [pos for pos in range(token_count - 1) if pos not in excluded]
    if not available:
        raise RuntimeError(f"Cannot create a Phase421 control for length {length}")
    # Long repeated evidence can occupy most of a short prompt.  In that case
    # no disjoint control span of equal length exists.  Cycle deterministically
    # through the remaining non-source positions so the write statistic still
    # has the same sample count and is normalized per sampled token.
    return [available[index % len(available)] for index in range(length)]


def register_positions(
    fast_tokenizer: Any,
    prompt: str,
    slow_ids: list[int],
    row: dict[str, Any],
) -> dict[str, Any]:
    fast = fast_tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    fast_ids = [int(value) for value in fast["input_ids"]]
    if fast_ids != slow_ids:
        raise RuntimeError(f"Phase421 fast/slow token mismatch: {row['phase421_condition_id']}")
    offsets = [(int(left), int(right)) for left, right in fast["offset_mapping"]]
    raw_start = prompt.rfind(row["raw_prompt"])
    if raw_start < 0:
        raise RuntimeError(f"Current raw prompt missing: {row['phase421_condition_id']}")
    current_header = f"Current support count: {row['current_support_count']}."
    current_start = prompt.rfind(current_header, 0, raw_start + 1)
    if current_start < 0:
        raise RuntimeError(f"Current support header missing: {row['phase421_condition_id']}")
    history_end = current_start
    history_spans = all_char_spans(prompt, row["history_answer"], 0, history_end)
    evidence_spans = all_char_spans(
        prompt,
        row["source_fragment"],
        current_start,
        len(prompt),
    )
    query_local = row["raw_prompt"].find(row["query_fragment"])
    if query_local < 0:
        raise RuntimeError(f"Current query fragment missing: {row['phase421_condition_id']}")
    query_start = raw_start + query_local
    query_spans = [(query_start, query_start + len(row["query_fragment"]))]
    history = positions_for_spans(offsets, history_spans[-1:])
    evidence = positions_for_spans(offsets, evidence_spans)
    query = positions_for_spans(offsets, query_spans)
    excluded = set(history) | set(evidence) | set(query) | {len(slow_ids) - 1}
    # Each control independently excludes every registered source role.  The
    # controls may overlap one another: forcing three long controls to be
    # mutually disjoint would consume all neutral positions in short prompts.
    history_control = matched_control_positions(len(history), len(slow_ids), excluded, 1)
    current_control = matched_control_positions(
        len(evidence), len(slow_ids), excluded, max(query) + 1
    )
    query_control = matched_control_positions(len(query), len(slow_ids), excluded, 1)
    roles = {
        "history_answer": history,
        "current_evidence": evidence,
        "current_query": query,
        "history_length_control": history_control,
        "current_length_control": current_control,
        "query_length_control": query_control,
    }
    return {
        "condition_id": row["phase421_condition_id"],
        "prompt_token_count": len(slow_ids),
        "prediction_position": len(slow_ids) - 1,
        "roles": roles,
        "all_roles_nonempty": all(roles[role] for role in SOURCE_ROLES),
        "history_control_length_matched": len(history) == len(history_control),
        "current_control_length_matched": len(evidence) == len(current_control),
        "query_control_length_matched": len(query) == len(query_control),
        "current_evidence_occurrence_count": len(evidence_spans),
        "fast_slow_token_ids_exact": True,
    }


def source_writes(
    probabilities: torch.Tensor,
    repeated_value: torch.Tensor,
    output_blocks: torch.Tensor,
    positions: list[int],
    query_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.tensor(positions, dtype=torch.long, device=probabilities.device)
    alpha = probabilities[0, :, query_position].index_select(-1, index).float()
    values = repeated_value[0].index_select(1, index).float()
    weighted = torch.einsum("hs,hsd->hd", alpha, values)
    writes = torch.einsum("hd,ohd->ho", weighted, output_blocks)
    return alpha.sum(dim=-1).detach().cpu(), torch.linalg.vector_norm(writes, dim=-1).detach().cpu()


def independent_geometry(attention: torch.Tensor, mlp: torch.Tensor) -> dict[str, float]:
    eps = 1e-8
    attention_norm = float(torch.linalg.vector_norm(attention).item())
    mlp_norm = float(torch.linalg.vector_norm(mlp).item())
    denominator = attention_norm * attention_norm + eps
    parallel = float(torch.dot(mlp, attention).item() / denominator)
    orthogonal = mlp - parallel * attention
    return {
        "parallel_gain": clean(parallel),
        "orthogonal_rewrite_ratio": clean(
            float(torch.linalg.vector_norm(orthogonal).item()) / (attention_norm + eps)
        ),
        "total_mlp_attention_ratio": clean(mlp_norm / (attention_norm + eps)),
        "delta_attention_norm": clean(attention_norm),
        "delta_mlp_norm": clean(mlp_norm),
        "delta_combined_norm": clean(float(torch.linalg.vector_norm(attention + mlp).item())),
    }


def per_token(state: dict[str, Any], layer: int, role: str, head: int) -> float:
    count = len(state["position_registry"]["roles"][role])
    return float(state["head_write_norm"][layer][role][head].item()) / max(1, count)


def role_specificity(
    state: dict[str, Any],
    layer: int,
    role: str,
    head: int,
) -> float:
    control = ROLE_CONTROL[role]
    return per_token(state, layer, role, head) - per_token(state, layer, control, head)


@torch.inference_mode()
def collect_condition(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    row: dict[str, Any],
    full_ledger: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prompt, encoded, prompt_ids = prompt_and_ids(loaded, row)
    registry = register_positions(fast_tokenizer, prompt, prompt_ids, row)
    captures.clear()
    result = loaded.model(
        **encoded,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    query_position = len(prompt_ids) - 1
    state: dict[str, Any] = {
        "head_mass": [],
        "head_write_norm": [],
        "attention_vectors": [],
        "mlp_vectors": [],
        "position_registry": registry,
    }
    layer_rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        probabilities = captures.get(("attention_probabilities", layer_index))
        if probabilities is None:
            raise RuntimeError(f"Missing Phase421 attention probabilities at layer {layer_index}")
        value = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        repeated_value = repeat_key_value(value, head_count)
        output_projection = module_attr(layer.self_attn, ("o_proj", "dense"))
        head_dim = int(repeated_value.shape[-1])
        output_blocks = output_projection.weight.float().view(
            output_projection.weight.shape[0], head_count, head_dim
        )
        role_mass = {}
        role_norm = {}
        for role in SOURCE_ROLES:
            mass, norm = source_writes(
                probabilities,
                repeated_value,
                output_blocks,
                registry["roles"][role],
                query_position,
            )
            role_mass[role] = mass
            role_norm[role] = norm
        state["head_mass"].append(role_mass)
        state["head_write_norm"].append(role_norm)
        attention_actual = captures[("attention_output", layer_index)].float()
        mlp_actual = captures[("mlp_output", layer_index)].float()
        state["attention_vectors"].append(
            attention_actual[0, query_position].detach().float().cpu()
        )
        state["mlp_vectors"].append(mlp_actual[0, query_position].detach().float().cpu())
        if not full_ledger:
            continue
        weighted_all = torch.matmul(probabilities.float(), repeated_value.float())
        head_writes_all = torch.einsum("bhqd,ohd->bqho", weighted_all, output_blocks)
        attention_replay = head_writes_all.sum(dim=2)
        if output_projection.bias is not None:
            attention_replay = attention_replay + output_projection.bias.float()
        _, attention_error = relative_error(attention_actual, attention_replay)
        down_input = captures[("down_proj_input", layer_index)].float()
        down_projection = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))
        mlp_replay = F.linear(
            down_input,
            down_projection.weight.float(),
            down_projection.bias.float() if down_projection.bias is not None else None,
        )
        _, mlp_error = relative_error(mlp_actual, mlp_replay)
        layer_input = captures[("layer_input", layer_index)].float()
        layer_output = captures[("layer_output", layer_index)].float()
        _, block_error = relative_error(layer_output, layer_input + attention_actual + mlp_actual)
        probability_error = float(
            (probabilities[0, :, query_position].float().sum(dim=-1) - 1.0).abs().max().item()
        )
        layer_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase421-PhysicalConditionLayer",
                "created_at": now(),
                "model": row["model"],
                "condition_id": row["phase421_condition_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "family_id": row["family_id"],
                "mechanism_id": row["mechanism_id"],
                "interface": row["interface"],
                "current_identity": row["current_identity"],
                "current_support_count": row["current_support_count"],
                "history_reliability_score": row["history_reliability_score"],
                "history_relation": row["history_relation"],
                "layer": layer_index,
                "relative_depth": layer_index / max(1, len(layers) - 1),
                "depth_bin": depth_bin(layer_index, len(layers)),
                "head_count": head_count,
                "key_value_head_count": int(value.shape[1]),
                "attention_probability_sum_max_error": clean(probability_error),
                "attention_replay_relative_error": clean(attention_error),
                "mlp_replay_relative_error": clean(mlp_error),
                "block_replay_relative_error": clean(block_error),
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    del result, encoded
    return state, layer_rows, registry


def grouped_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        output[row["group_id"]].append(row)
    return output


def state_cells(
    group_rows: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any]]]:
    return {
        (
            row["interface"],
            row["current_identity"],
            row["current_support_count"],
            row["history_reliability_score"],
            row["history_relation"],
        ): (row, states[row["phase421_condition_id"]])
        for row in group_rows
    }


def update_coordinate_search(
    accumulator: dict[tuple[Any, ...], list[float]],
    group_rows: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    layer_count: int,
) -> None:
    cells = state_cells(group_rows, states)
    family = group_rows[0]["family_id"]
    for interface in ("chat", "completion"):
        for current_identity in ("a", "b"):
            for support in (1, 3):
                for reliability in (1, 3):
                    for relation in ("compatible", "conflict"):
                        relation_state = cells[(interface, current_identity, support, reliability, relation)][1]
                        for role in ("history_answer", "current_evidence"):
                            for layer in range(layer_count):
                                head_count = relation_state["head_write_norm"][layer][role].numel()
                                for head in range(head_count):
                                    accumulator[(family, interface, relation, role, layer, head)].append(
                                        role_specificity(relation_state, layer, role, head)
                                    )


def select_coordinates(
    model: str,
    accumulator: dict[tuple[Any, ...], list[float]],
    layer_count: int,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], tuple[int, int]]]:
    search_rows = []
    selected: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (family, interface, relation, role, layer, head), values in accumulator.items():
        row = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase421-DiscoveryCoordinateSearch",
            "created_at": now(),
            "model": model,
            "family_id": family,
            "interface": interface,
            "history_relation": relation,
            "source_role": role,
            "layer": layer,
            "relative_depth": layer / max(1, layer_count - 1),
            "depth_bin": depth_bin(layer, layer_count),
            "head_index": head,
            "discovery_specificity_median": clean(median(values)),
            "discovery_specificity_positive_rate": clean(
                sum(value > 0 for value in values) / len(values)
            ),
            "discovery_contrast_count": len(values),
            "physical": True,
            "predictive": False,
            "causal": False,
        }
        search_rows.append(row)
        grouped[(family, interface, relation, role)].append(row)
    for key, values in grouped.items():
        choice = max(
            values,
            key=lambda row: (
                row["discovery_specificity_positive_rate"],
                row["discovery_specificity_median"],
                -row["layer"],
                -row["head_index"],
            ),
        )
        selected[key] = (int(choice["layer"]), int(choice["head_index"]))
        choice["selected_coordinate"] = True
    for row in search_rows:
        row.setdefault("selected_coordinate", False)
    return search_rows, selected


def deterministic_control_head(
    group_index: int,
    layer: int,
    selected_head: int,
    head_count: int,
    salt: int,
) -> int:
    candidate = (group_index * 17 + layer * 7 + salt) % head_count
    if candidate == selected_head:
        candidate = (candidate + 1) % head_count
    return candidate


def summarize_group_at_selected_coordinates(
    model: str,
    group_rows: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    selected: dict[tuple[str, str, str, str], tuple[int, int]],
    behavior_effects: dict[tuple[Any, ...], dict[str, Any]],
    layer_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cells = state_cells(group_rows, states)
    first = group_rows[0]
    feature_rows = []
    geometry_rows = []
    for interface in ("chat", "completion"):
        for current_identity in ("a", "b"):
            for support in (1, 3):
                for reliability in (1, 3):
                    irrelevant_row, irrelevant_state = cells[
                        (interface, current_identity, support, reliability, "irrelevant")
                    ]
                    for relation in ("compatible", "conflict"):
                        relation_row, relation_state = cells[
                            (interface, current_identity, support, reliability, relation)
                        ]
                        source_payload: dict[str, Any] = {}
                        for salt, role in enumerate(("history_answer", "current_evidence"), 1):
                            layer, head = selected[(first["family_id"], interface, relation, role)]
                            head_count = int(relation_state["head_write_norm"][layer][role].numel())
                            random_head = deterministic_control_head(
                                first["group_index"], layer, head, head_count, salt
                            )
                            relation_specificity = role_specificity(relation_state, layer, role, head)
                            irrelevant_specificity = role_specificity(irrelevant_state, layer, role, head)
                            relation_head_values = [
                                role_specificity(relation_state, layer, role, index)
                                for index in range(head_count)
                            ]
                            source_payload.update(
                                {
                                    f"{role}_selected_layer": layer,
                                    f"{role}_selected_head": head,
                                    f"{role}_selected_relative_depth": clean(
                                        layer / max(1, layer_count - 1)
                                    ),
                                    f"{role}_relation_specificity": clean(relation_specificity),
                                    f"{role}_irrelevant_specificity": clean(irrelevant_specificity),
                                    f"{role}_specificity_change": clean(
                                        relation_specificity - irrelevant_specificity
                                    ),
                                    f"{role}_random_head_control": clean(
                                        role_specificity(
                                            relation_state, layer, role, random_head
                                        )
                                    ),
                                    f"{role}_same_layer_head_median_control": clean(
                                        median(relation_head_values)
                                    ),
                                }
                            )
                        geometry_by_depth: dict[str, list[dict[str, float]]] = defaultdict(list)
                        for layer in range(layer_count):
                            delta_attention = (
                                relation_state["attention_vectors"][layer]
                                - irrelevant_state["attention_vectors"][layer]
                            )
                            delta_mlp = (
                                relation_state["mlp_vectors"][layer]
                                - irrelevant_state["mlp_vectors"][layer]
                            )
                            geometry = independent_geometry(delta_attention, delta_mlp)
                            depth = depth_bin(layer, layer_count)
                            geometry_by_depth[depth].append(geometry)
                            geometry_rows.append(
                                {
                                    "schema_version": SCHEMA_VERSION,
                                    "phase_id": "Phase421-IndependentMLPGeometry",
                                    "created_at": now(),
                                    "model": model,
                                    "group_id": first["group_id"],
                                    "split": first["split"],
                                    "family_id": first["family_id"],
                                    "mechanism_id": first["mechanism_id"],
                                    "interface": interface,
                                    "current_identity": current_identity,
                                    "current_support_count": support,
                                    "history_reliability_score": reliability,
                                    "history_relation": relation,
                                    "layer": layer,
                                    "relative_depth": layer / max(1, layer_count - 1),
                                    "depth_bin": depth,
                                    **geometry,
                                    "physical": True,
                                    "predictive": False,
                                    "causal": False,
                                }
                            )
                        geometry_payload = {}
                        for depth in ("early", "middle", "late"):
                            for metric in (
                                "parallel_gain",
                                "orthogonal_rewrite_ratio",
                                "total_mlp_attention_ratio",
                                "delta_attention_norm",
                                "delta_mlp_norm",
                            ):
                                geometry_payload[f"{metric}_{depth}_median"] = clean(
                                    median(item[metric] for item in geometry_by_depth[depth])
                                )
                        behavior_key = (
                            model,
                            first["group_id"],
                            current_identity,
                            interface,
                            support,
                            reliability,
                            relation,
                        )
                        behavior = behavior_effects[behavior_key]
                        feature_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "phase_id": "Phase421-FixedPathFeature",
                                "created_at": now(),
                                "model": model,
                                "group_id": first["group_id"],
                                "group_index": first["group_index"],
                                "split": first["split"],
                                "family_id": first["family_id"],
                                "mechanism_id": first["mechanism_id"],
                                "interface": interface,
                                "current_identity": current_identity,
                                "current_support_count": support,
                                "history_reliability_score": reliability,
                                "history_relation": relation,
                                "prompt_token_count": relation_state["position_registry"]["prompt_token_count"],
                                "target_token_count": relation_row["registered_target_token_count"],
                                "relation_margin_effect_vs_irrelevant": behavior[
                                    "relation_margin_effect_vs_irrelevant"
                                ],
                                **source_payload,
                                **geometry_payload,
                                "physical": True,
                                "predictive": False,
                                "causal": False,
                            }
                        )
    return feature_rows, geometry_rows


def noise_repeat_keys(group_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [
        row
        for row in group_rows
        if row["interface"] == "chat"
        and row["current_identity"] == "a"
        and row["current_support_count"] == 1
        and row["history_reliability_score"] == 1
        and row["history_relation"] in {"conflict", "irrelevant"}
    ]
    by_relation = {row["history_relation"]: row for row in selected}
    return by_relation["conflict"], by_relation["irrelevant"]


def geometry_noise_rows(
    model: str,
    original: dict[str, dict[str, Any]],
    repeated: dict[str, dict[str, Any]],
    group_row: dict[str, Any],
    layer_count: int,
) -> list[dict[str, Any]]:
    output = []
    original_relation = original["conflict"]
    original_irrelevant = original["irrelevant"]
    repeat_relation = repeated["conflict"]
    repeat_irrelevant = repeated["irrelevant"]
    for layer in range(layer_count):
        original_geometry = independent_geometry(
            original_relation["attention_vectors"][layer]
            - original_irrelevant["attention_vectors"][layer],
            original_relation["mlp_vectors"][layer]
            - original_irrelevant["mlp_vectors"][layer],
        )
        repeat_geometry = independent_geometry(
            repeat_relation["attention_vectors"][layer]
            - repeat_irrelevant["attention_vectors"][layer],
            repeat_relation["mlp_vectors"][layer]
            - repeat_irrelevant["mlp_vectors"][layer],
        )
        output.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase421-GeometryRepeatNoise",
                "created_at": now(),
                "model": model,
                "group_id": group_row["group_id"],
                "split": group_row["split"],
                "family_id": group_row["family_id"],
                "layer": layer,
                "depth_bin": depth_bin(layer, layer_count),
                "parallel_gain_absolute_repeat_difference": clean(
                    abs(original_geometry["parallel_gain"] - repeat_geometry["parallel_gain"])
                ),
                "orthogonal_rewrite_absolute_repeat_difference": clean(
                    abs(
                        original_geometry["orthogonal_rewrite_ratio"]
                        - repeat_geometry["orthogonal_rewrite_ratio"]
                    )
                ),
                "total_ratio_absolute_repeat_difference": clean(
                    abs(
                        original_geometry["total_mlp_attention_ratio"]
                        - repeat_geometry["total_mlp_attention_ratio"]
                    )
                ),
                "minimum_numeric_floor": NUMERIC_GEOMETRY_FLOOR,
                "causal": False,
            }
        )
    return output


def collect_group_states(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    group_rows: list[dict[str, Any]],
    full_ledger: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    states = {}
    layer_rows = []
    position_rows = []
    for row in group_rows:
        state, rows_for_layers, registry = collect_condition(
            loaded, fast_tokenizer, layers, captures, row, full_ledger
        )
        states[row["phase421_condition_id"]] = state
        layer_rows.extend(rows_for_layers)
        if full_ledger:
            position_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase421-PositionRegistry",
                    "created_at": now(),
                    "model": loaded.key,
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "family_id": row["family_id"],
                    "interface": row["interface"],
                    "current_identity": row["current_identity"],
                    "current_support_count": row["current_support_count"],
                    "history_reliability_score": row["history_reliability_score"],
                    "history_relation": row["history_relation"],
                    **registry,
                }
            )
        captures.clear()
    return states, layer_rows, position_rows


@torch.inference_mode()
def collect_physical(loaded: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    physical_rows = [
        row
        for row in rows
        if row["physical_development_panel"] and row["split"] in DEVELOPMENT_SPLITS
    ]
    if len(physical_rows) != 1_344:
        raise RuntimeError(f"Expected 1344 Phase421 physical rows, found {len(physical_rows)}")
    group_map = grouped_rows(physical_rows)
    discovery_group_ids = sorted(
        group_id
        for group_id, values in group_map.items()
        if values[0]["split"] == "discovery"
    )
    development_group_ids = sorted(
        group_map,
        key=lambda group_id: (
            ("discovery", "calibration", "behavior_holdout").index(
                group_map[group_id][0]["split"]
            ),
            group_map[group_id][0]["group_index"],
        ),
    )
    behavior_effect_rows = read_jsonl(OUT / "phase421_behavior_effect_rows.jsonl")
    behavior_effects = {
        (
            row["model"],
            row["group_id"],
            row["current_identity"],
            row["interface"],
            row["current_support_count"],
            row["history_reliability_score"],
            row["history_relation"],
        ): row
        for row in behavior_effect_rows
        if row["model"] == loaded.key
    }
    layers = get_layers(loaded.model)
    fast_tokenizer = fast_tokenizer_for(loaded.key)
    captures: dict[tuple[str, int], Any] = {}
    handles = install_hooks(layers, captures)
    coordinate_accumulator: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    condition_layer_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    try:
        with capture_actual_qkv(loaded.key, tuple(range(len(layers))), captures):
            for index, group_id in enumerate(discovery_group_ids, 1):
                group_rows = sorted(
                    group_map[group_id], key=lambda row: row["phase421_condition_id"]
                )
                states, _, _ = collect_group_states(
                    loaded, fast_tokenizer, layers, captures, group_rows, False
                )
                update_coordinate_search(
                    coordinate_accumulator, group_rows, states, len(layers)
                )
                del states
                gc.collect()
                print(
                    f"[Phase421:{loaded.key}:search] {index}/{len(discovery_group_ids)}",
                    flush=True,
                )
            search_rows, selected = select_coordinates(
                loaded.key, coordinate_accumulator, len(layers)
            )
            repeated_split_family: set[tuple[str, str]] = set()
            for index, group_id in enumerate(development_group_ids, 1):
                group_rows = sorted(
                    group_map[group_id], key=lambda row: row["phase421_condition_id"]
                )
                states, group_layers, group_positions = collect_group_states(
                    loaded, fast_tokenizer, layers, captures, group_rows, True
                )
                condition_layer_rows.extend(group_layers)
                position_rows.extend(group_positions)
                features, geometries = summarize_group_at_selected_coordinates(
                    loaded.key,
                    group_rows,
                    states,
                    selected,
                    behavior_effects,
                    len(layers),
                )
                feature_rows.extend(features)
                geometry_rows.extend(geometries)
                split_family = (group_rows[0]["split"], group_rows[0]["family_id"])
                if split_family not in repeated_split_family:
                    repeated_split_family.add(split_family)
                    conflict_row, irrelevant_row = noise_repeat_keys(group_rows)
                    original = {
                        "conflict": states[conflict_row["phase421_condition_id"]],
                        "irrelevant": states[irrelevant_row["phase421_condition_id"]],
                    }
                    repeated = {}
                    for relation, repeat_row in (
                        ("conflict", conflict_row),
                        ("irrelevant", irrelevant_row),
                    ):
                        repeat_state, _, _ = collect_condition(
                            loaded,
                            fast_tokenizer,
                            layers,
                            captures,
                            repeat_row,
                            False,
                        )
                        repeated[relation] = repeat_state
                    noise_rows.extend(
                        geometry_noise_rows(
                            loaded.key,
                            original,
                            repeated,
                            group_rows[0],
                            len(layers),
                        )
                    )
                    del repeated
                del states
                gc.collect()
                print(
                    f"[Phase421:{loaded.key}:fixed] {index}/{len(development_group_ids)} "
                    f"features={len(feature_rows)}",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()
    return {
        "condition_layer_rows": condition_layer_rows,
        "position_rows": position_rows,
        "coordinate_search_rows": search_rows,
        "feature_rows": feature_rows,
        "geometry_rows": geometry_rows,
        "noise_rows": noise_rows,
    }


def run_model(model: str) -> dict[str, Any]:
    authorization = read_json(AUTHORIZATION)
    if not authorization["physical_development_collection_authorized"]:
        raise RuntimeError("Phase421 physical development is not authorized")
    rows = [row for row in read_jsonl(REGISTERED) if row["model"] == model]
    loaded = None
    started = time.monotonic()
    try:
        print(f"[Phase421] loading {model}; physical=1344; holdout=sealed", flush=True)
        loaded = load_probe_model(model)
        outputs = collect_physical(loaded, rows)
        model_root = OUT / "models" / model
        write_jsonl(
            model_root / "phase421_physical_condition_layer_rows.jsonl",
            outputs["condition_layer_rows"],
        )
        write_jsonl(model_root / "phase421_position_registry.jsonl", outputs["position_rows"])
        write_jsonl(
            model_root / "phase421_discovery_coordinate_search.jsonl",
            outputs["coordinate_search_rows"],
        )
        write_jsonl(model_root / "phase421_fixed_path_feature_rows.jsonl", outputs["feature_rows"])
        write_jsonl(model_root / "phase421_independent_mlp_geometry.jsonl", outputs["geometry_rows"])
        write_jsonl(model_root / "phase421_geometry_repeat_noise.jsonl", outputs["noise_rows"])
        errors = [
            max(
                row["attention_probability_sum_max_error"],
                row["attention_replay_relative_error"],
                row["mlp_replay_relative_error"],
                row["block_replay_relative_error"],
            )
            for row in outputs["condition_layer_rows"]
        ]
        selected_count = sum(
            row["selected_coordinate"] for row in outputs["coordinate_search_rows"]
        )
        all_pass = bool(
            len(outputs["position_rows"]) == 1_344
            and len(outputs["feature_rows"]) == 896
            and selected_count == 32
            and all(row["all_roles_nonempty"] for row in outputs["position_rows"])
            and all(error <= LEDGER_THRESHOLD for error in errors)
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "execution_dtype": str(next(loaded.model.parameters()).dtype).replace("torch.", ""),
            "physical_development_condition_count": len(outputs["position_rows"]),
            "physical_holdout_condition_count": 0,
            "physical_holdout_remains_sealed": True,
            "physical_condition_layer_row_count": len(outputs["condition_layer_rows"]),
            "coordinate_search_row_count": len(outputs["coordinate_search_rows"]),
            "selected_coordinate_count": selected_count,
            "fixed_path_feature_row_count": len(outputs["feature_rows"]),
            "independent_mlp_geometry_row_count": len(outputs["geometry_rows"]),
            "geometry_repeat_noise_row_count": len(outputs["noise_rows"]),
            "max_physical_ledger_relative_error": max(errors),
            "all_development_rows_pass": all_pass,
            "feature_rows_sha256": hash_rows(outputs["feature_rows"]),
            "elapsed_seconds": time.monotonic() - started,
            "vram_gb": vram_gb(),
            "physical_holdout_collection_authorized": False,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
        }
        write_json(model_root / "phase421_physical_complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    summary = run_model(args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["all_development_rows_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
