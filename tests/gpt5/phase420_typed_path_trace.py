#!/usr/bin/env python3
"""Trace typed history/current source writes and same-layer MLP rewrites.

The collector keeps the physical holdout sealed.  It records behavior for the
full frozen denominator, then traces only discovery, calibration and behavior
holdout groups.  Attention source writes use the model's actual attention
probabilities, actual value states and the corresponding output-projection
head block; no head number is assigned a semantic name.
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
from phase416_real_collector_qualification import (  # noqa: E402
    eos_ids,
    exact_answer,
    neutral_generation_config,
    target_match,
)
from phase420_typed_path_case_bank import (  # noqa: E402
    CURRENT_IDENTITIES,
    HISTORY_IDENTITIES,
    INTERFACES,
    MODELS,
    OUT,
    SCHEMA_VERSION,
    serialize_crossed_prompt,
)


PHASE_ID = "Phase420-TypedNaturalPathTrace"
REGISTERED = OUT / "phase420_registered_conditions.jsonl"
DEVELOPMENT_SPLITS = {"discovery", "calibration", "behavior_holdout"}
SOURCE_ROLES = (
    "history_answer",
    "current_evidence",
    "current_query",
    "history_length_control",
    "current_length_control",
)
# The local models execute these paths in reduced precision.  Phase401 already
# qualified exact component replay at a 1% relative-error gate; Phase420 keeps
# that frozen engineering gate and records the observed error continuously.
LEDGER_THRESHOLD = 0.01
BEHAVIOR_HORIZON = 12
EXTENDED_BEHAVIOR_HORIZON = 24


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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Phase420 non-finite scalar: {value}")
    return round(float(value), 10)


def fast_tokenizer_for(model: str) -> Any:
    spec = get_model_spec(model)
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.local_dir),
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
        use_fast=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Phase420 requires offset mapping for {model}")
    return tokenizer


def prompt_and_ids(loaded: Any, row: dict[str, Any]) -> tuple[str, dict[str, torch.Tensor], list[int]]:
    prompt, _ = serialize_crossed_prompt(
        loaded.tokenizer,
        row["raw_prompt"],
        row["interface"],
        row["history_answer"],
    )
    encoded = loaded.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    return prompt, {key: value.to(loaded.input_device) for key, value in encoded.items()}, ids


def first_token_candidates(tokenizer: Any, text: str) -> list[int]:
    output = set()
    for variant in (text, " " + text, "\n" + text):
        ids = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if ids:
            output.add(int(ids[0]))
    return sorted(output)


def first_step_margin(scores: torch.Tensor, tokenizer: Any, target: str, opposite: str) -> float:
    target_ids = first_token_candidates(tokenizer, target)
    opposite_ids = first_token_candidates(tokenizer, opposite)
    if not target_ids or not opposite_ids:
        raise RuntimeError(f"Missing first-token candidates: {target!r}/{opposite!r}")
    target_score = max(float(scores[token_id].item()) for token_id in target_ids)
    opposite_score = max(float(scores[token_id].item()) for token_id in opposite_ids)
    return target_score - opposite_score


def char_span(prompt: str, row: dict[str, Any], fragment: str, before_current: bool = False) -> tuple[int, int]:
    current_start = prompt.rfind(row["raw_prompt"])
    if current_start < 0:
        raise RuntimeError(f"Current raw prompt not found: {row['phase420_condition_id']}")
    if before_current:
        start = prompt.rfind(fragment, 0, current_start)
    else:
        local = row["raw_prompt"].find(fragment)
        start = current_start + local if local >= 0 else -1
    if start < 0:
        raise RuntimeError(
            f"Registered fragment not found for {row['phase420_condition_id']}: {fragment!r}"
        )
    return start, start + len(fragment)


def token_positions_for_char_span(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    start, end = span
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if not positions:
        raise RuntimeError(f"Empty token span for character interval {span}")
    return positions


def matched_control_positions(
    length: int,
    token_count: int,
    excluded: set[int],
    preferred_start: int,
) -> list[int]:
    starts = list(range(max(0, preferred_start), max(0, token_count - length)))
    starts += list(range(0, max(0, preferred_start)))
    for start in starts:
        positions = list(range(start, start + length))
        if positions and positions[-1] < token_count - 1 and not any(pos in excluded for pos in positions):
            return positions
    available = [pos for pos in range(token_count - 1) if pos not in excluded]
    if len(available) < length:
        raise RuntimeError(f"Cannot register length-{length} source control")
    return available[:length]


def register_positions(
    fast_tokenizer: Any,
    prompt: str,
    slow_ids: list[int],
    row: dict[str, Any],
) -> dict[str, Any]:
    fast = fast_tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    fast_ids = [int(value) for value in fast["input_ids"]]
    if fast_ids != slow_ids:
        raise RuntimeError(f"Fast/slow tokenizer ID mismatch: {row['phase420_condition_id']}")
    offsets = [(int(left), int(right)) for left, right in fast["offset_mapping"]]
    history = token_positions_for_char_span(
        offsets,
        char_span(prompt, row, row["history_answer"], before_current=True),
    )
    evidence = token_positions_for_char_span(
        offsets,
        char_span(prompt, row, row["source_fragment"]),
    )
    query = token_positions_for_char_span(
        offsets,
        char_span(prompt, row, row["query_fragment"]),
    )
    excluded = set(history) | set(evidence) | set(query) | {len(slow_ids) - 1}
    history_control = matched_control_positions(
        len(history), len(slow_ids), excluded, preferred_start=1
    )
    excluded.update(history_control)
    current_control = matched_control_positions(
        len(evidence), len(slow_ids), excluded, preferred_start=max(query) + 1
    )
    roles = {
        "history_answer": history,
        "current_evidence": evidence,
        "current_query": query,
        "history_length_control": history_control,
        "current_length_control": current_control,
    }
    return {
        "condition_id": row["phase420_condition_id"],
        "prompt_token_count": len(slow_ids),
        "prediction_position": len(slow_ids) - 1,
        "roles": roles,
        "all_roles_nonempty": all(roles[role] for role in SOURCE_ROLES),
        "history_control_length_matched": len(history) == len(history_control),
        "current_control_length_matched": len(evidence) == len(current_control),
        "fast_slow_token_ids_exact": True,
    }


@torch.inference_mode()
def collect_behavior(
    loaded: Any,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output_rows = []
    eos = eos_ids(loaded.tokenizer, loaded.model)
    for index, row in enumerate(rows, 1):
        prompt, encoded, prompt_ids = prompt_and_ids(loaded, row)
        if len(prompt_ids) != int(row["registered_prompt_token_count"]):
            raise RuntimeError(f"Registered prompt count changed: {row['phase420_condition_id']}")

        def generate(horizon: int) -> Any:
            return loaded.model.generate(
                **encoded,
                generation_config=neutral_generation_config(loaded),
                max_new_tokens=horizon,
                return_dict_in_generate=True,
                output_scores=True,
            )

        result = generate(BEHAVIOR_HORIZON)
        generated = [int(value) for value in result.sequences[0, len(prompt_ids) :].tolist()]
        initial_right_censored = not any(token in eos for token in generated) and len(generated) >= BEHAVIOR_HORIZON
        extended = False
        if initial_right_censored:
            del result
            result = generate(EXTENDED_BEHAVIOR_HORIZON)
            generated = [int(value) for value in result.sequences[0, len(prompt_ids) :].tolist()]
            extended = True
        text = loaded.tokenizer.decode(generated, skip_special_tokens=True)
        emitted_stop = any(token in eos for token in generated)
        scores = result.scores[0][0].detach().float()
        output_rows.append(
            {
                **row,
                "phase_id": "Phase420-BehaviorTrace",
                "created_at": now(),
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "prompt_token_count": len(prompt_ids),
                "registered_prompt_token_count_pass": True,
                "generated_token_ids": generated,
                "generated_text": text,
                "target_event_match": target_match(text, row["target_aliases"]),
                "opposite_identity_event_match": target_match(text, [row["opposite_identity_target"]]),
                "exact_answer_match": exact_answer(text, row["target_aliases"]),
                "target_first_step_margin": clean(
                    first_step_margin(
                        scores,
                        loaded.tokenizer,
                        row["target"],
                        row["opposite_identity_target"],
                    )
                ),
                "behavior_horizon_initial": BEHAVIOR_HORIZON,
                "behavior_horizon_used": EXTENDED_BEHAVIOR_HORIZON if extended else BEHAVIOR_HORIZON,
                "behavior_horizon_extended": extended,
                "initial_right_censored": initial_right_censored,
                "right_censored": not emitted_stop and len(generated) >= (
                    EXTENDED_BEHAVIOR_HORIZON if extended else BEHAVIOR_HORIZON
                ),
                "emitted_stop": emitted_stop,
                "behavior_trace_pass": bool(result.scores and torch.isfinite(scores).all().item()),
                "causal": False,
            }
        )
        del result, encoded, scores
        if index % 16 == 0 or index == len(rows):
            print(
                f"[Phase420:{loaded.key}:behavior] {index}/{len(rows)} "
                f"pass={sum(item['behavior_trace_pass'] for item in output_rows)}",
                flush=True,
            )
    return output_rows


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) <= 1e-12 or float(right_norm.item()) <= 1e-12:
        return None
    return clean(float(torch.dot(left, right).item() / (left_norm.item() * right_norm.item())))


def mlp_relation(attention: torch.Tensor, mlp: torch.Tensor) -> dict[str, Any]:
    eps = 1e-8
    attention_norm = torch.linalg.vector_norm(attention)
    mlp_norm = torch.linalg.vector_norm(mlp)
    relation_cosine = cosine(attention, mlp)
    cancellation = 1.0 - float(
        torch.linalg.vector_norm(attention + mlp).item()
        / max(float(attention_norm.item() + mlp_norm.item()), eps)
    )
    if float(attention_norm.item()) > eps:
        projection = attention * (torch.dot(mlp, attention) / attention_norm.square())
    else:
        projection = torch.zeros_like(mlp)
    novelty = float(torch.linalg.vector_norm(mlp - projection).item() / max(float(mlp_norm.item()), eps))
    if relation_cosine is not None and relation_cosine >= 0.25 and cancellation < 0.20:
        relation_class = "same_direction_amplification"
    elif (relation_cosine is not None and relation_cosine <= -0.25) or cancellation >= 0.25:
        relation_class = "opposing_cancellation"
    elif novelty >= 0.90:
        relation_class = "orthogonal_rewrite"
    else:
        relation_class = "mixed_rewrite"
    return {
        "attention_mlp_cosine": relation_cosine,
        "cancellation_index": clean(cancellation),
        "rewrite_novelty": clean(novelty),
        "rewrite_class": relation_class,
    }


def source_writes(
    probabilities: torch.Tensor,
    repeated_value: torch.Tensor,
    output_blocks: torch.Tensor,
    positions: list[int],
    query_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.tensor(positions, dtype=torch.long, device=probabilities.device)
    alpha = probabilities[0, :, query_position].index_select(-1, index).float()
    values = repeated_value[0].index_select(1, index).float()
    weighted = torch.einsum("hs,hsd->hd", alpha, values)
    writes = torch.einsum("hd,ohd->ho", weighted, output_blocks)
    mass = alpha.sum(dim=-1)
    norms = torch.linalg.vector_norm(writes, dim=-1)
    return mass.detach().cpu(), norms.detach().cpu(), writes.sum(dim=0).detach().cpu()


def collect_physical_condition(
    loaded: Any,
    fast_tokenizer: Any,
    layers: list[Any],
    captures: dict[tuple[str, int], Any],
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prompt, encoded, prompt_ids = prompt_and_ids(loaded, row)
    if len(prompt_ids) != int(row["registered_prompt_token_count"]):
        raise RuntimeError(f"Physical prompt count changed: {row['phase420_condition_id']}")
    registry = register_positions(fast_tokenizer, prompt, prompt_ids, row)
    captures.clear()
    result = loaded.model(
        **encoded,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    qpos = len(prompt_ids) - 1
    state: dict[str, Any] = {
        "head_mass": [],
        "head_write_norm": [],
        "source_vectors": [],
        "layer_metrics": [],
    }
    layer_rows = []
    for layer_index, layer in enumerate(layers):
        probabilities = captures.get(("attention_probabilities", layer_index))
        if probabilities is None:
            raise RuntimeError(f"Missing actual attention probabilities at layer {layer_index}")
        value = captures[("value", layer_index)]
        head_count = int(probabilities.shape[1])
        repeated_value = repeat_key_value(value, head_count)
        o_proj = module_attr(layer.self_attn, ("o_proj", "dense"))
        head_dim = int(repeated_value.shape[-1])
        output_blocks = o_proj.weight.float().view(o_proj.weight.shape[0], head_count, head_dim)
        role_mass = {}
        role_norm = {}
        role_vectors = {}
        for role in SOURCE_ROLES:
            mass, norms, vector = source_writes(
                probabilities,
                repeated_value,
                output_blocks,
                registry["roles"][role],
                qpos,
            )
            role_mass[role] = mass
            role_norm[role] = norms
            role_vectors[role] = vector

        weighted_all = torch.matmul(probabilities.float(), repeated_value.float())
        head_writes_all = torch.einsum("bhqd,ohd->bqho", weighted_all, output_blocks)
        attention_replay = head_writes_all.sum(dim=2)
        if o_proj.bias is not None:
            attention_replay = attention_replay + o_proj.bias.float()
        attention_actual = captures[("attention_output", layer_index)].float()
        _, attention_error = relative_error(attention_actual, attention_replay)

        down_input = captures[("down_proj_input", layer_index)].float()
        down_proj = module_attr(layer.mlp, ("down_proj", "dense_4h_to_h"))
        mlp_replay = F.linear(
            down_input,
            down_proj.weight.float(),
            down_proj.bias.float() if down_proj.bias is not None else None,
        )
        mlp_actual = captures[("mlp_output", layer_index)].float()
        _, mlp_error = relative_error(mlp_actual, mlp_replay)
        layer_input = captures[("layer_input", layer_index)].float()
        layer_output = captures[("layer_output", layer_index)].float()
        _, block_error = relative_error(layer_output, layer_input + attention_actual + mlp_actual)
        probability_error = float(
            (probabilities[0, :, qpos].float().sum(dim=-1) - 1.0).abs().max().item()
        )
        attn_q = attention_actual[0, qpos].detach().float().cpu()
        mlp_q = mlp_actual[0, qpos].detach().float().cpu()
        input_q = layer_input[0, qpos].detach().float().cpu()
        output_q = layer_output[0, qpos].detach().float().cpu()
        relation = mlp_relation(attn_q, mlp_q)
        metrics = {
            "attention_output_norm": clean(float(torch.linalg.vector_norm(attn_q).item())),
            "mlp_output_norm": clean(float(torch.linalg.vector_norm(mlp_q).item())),
            "layer_input_norm": clean(float(torch.linalg.vector_norm(input_q).item())),
            "post_attention_state_norm": clean(float(torch.linalg.vector_norm(input_q + attn_q).item())),
            "layer_output_norm": clean(float(torch.linalg.vector_norm(output_q).item())),
            **relation,
        }
        layer_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase420-PhysicalConditionLayer",
                "created_at": now(),
                "model": row["model"],
                "condition_id": row["phase420_condition_id"],
                "group_id": row["group_id"],
                "family_id": row["family_id"],
                "mechanism_id": row["mechanism_id"],
                "split": row["split"],
                "interface": row["interface"],
                "current_identity": row["current_identity"],
                "history_identity": row["history_identity"],
                "history_compatible": row["history_compatible"],
                "layer": layer_index,
                "relative_depth": layer_index / max(1, len(layers) - 1),
                "depth_bin": depth_bin(layer_index, len(layers)),
                "head_count": head_count,
                "key_value_head_count": int(value.shape[1]),
                "head_dim": head_dim,
                "attention_probability_sum_max_error": clean(probability_error),
                "attention_replay_relative_error": clean(attention_error),
                "mlp_replay_relative_error": clean(mlp_error),
                "block_replay_relative_error": clean(block_error),
                **metrics,
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
        state["head_mass"].append(role_mass)
        state["head_write_norm"].append(role_norm)
        state["source_vectors"].append(role_vectors)
        state["layer_metrics"].append(metrics)
    del result, encoded
    return state, layer_rows, registry


def scalar_effect(cells: dict[tuple[str, str], Any], getter: Any) -> tuple[float, float, float]:
    aa = float(getter(cells[("a", "a")]))
    ab = float(getter(cells[("a", "b")]))
    ba = float(getter(cells[("b", "a")]))
    bb = float(getter(cells[("b", "b")]))
    term_a = ab - aa
    term_b = ba - bb
    return clean(0.5 * (term_a + term_b)), clean(term_a), clean(term_b)


def vector_effect(cells: dict[tuple[str, str], Any], getter: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aa = getter(cells[("a", "a")]).float()
    ab = getter(cells[("a", "b")]).float()
    ba = getter(cells[("b", "a")]).float()
    bb = getter(cells[("b", "b")]).float()
    term_a = ab - aa
    term_b = ba - bb
    return 0.5 * (term_a + term_b), term_a, term_b


def summarize_crossed_cells(
    model: str,
    rows: list[dict[str, Any]],
    cells: dict[tuple[str, str], dict[str, Any]],
    layer_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    first = rows[0]
    head_rows = []
    path_rows = []
    rewrite_rows = []
    for layer in range(layer_count):
        head_count = int(cells[("a", "a")]["head_mass"][layer]["history_answer"].numel())
        for head in range(head_count):
            payload: dict[str, Any] = {}
            for role in SOURCE_ROLES:
                effect, term_a, term_b = scalar_effect(
                    cells,
                    lambda state, role=role: state["head_write_norm"][layer][role][head].item()
                    / max(1, len(state["position_registry"]["roles"][role])),
                )
                mass_effect, _, _ = scalar_effect(
                    cells,
                    lambda state, role=role: state["head_mass"][layer][role][head].item()
                    / max(1, len(state["position_registry"]["roles"][role])),
                )
                compatible_mean = 0.5 * (
                    cells[("a", "a")]["head_write_norm"][layer][role][head].item()
                    / len(cells[("a", "a")]["position_registry"]["roles"][role])
                    + cells[("b", "b")]["head_write_norm"][layer][role][head].item()
                    / len(cells[("b", "b")]["position_registry"]["roles"][role])
                )
                conflict_mean = 0.5 * (
                    cells[("a", "b")]["head_write_norm"][layer][role][head].item()
                    / len(cells[("a", "b")]["position_registry"]["roles"][role])
                    + cells[("b", "a")]["head_write_norm"][layer][role][head].item()
                    / len(cells[("b", "a")]["position_registry"]["roles"][role])
                )
                payload[f"{role}_compatible_write_norm_per_token_mean"] = clean(compatible_mean)
                payload[f"{role}_conflict_write_norm_per_token_mean"] = clean(conflict_mean)
                payload[f"{role}_compatibility_effect"] = effect
                payload[f"{role}_compatibility_term_a"] = term_a
                payload[f"{role}_compatibility_term_b"] = term_b
                payload[f"{role}_attention_mass_per_token_effect"] = mass_effect
            payload["history_source_specificity_effect"] = clean(
                payload["history_answer_compatibility_effect"]
                - payload["history_length_control_compatibility_effect"]
            )
            payload["current_source_specificity_effect"] = clean(
                payload["current_evidence_compatibility_effect"]
                - payload["current_length_control_compatibility_effect"]
            )
            payload["history_current_competition_effect"] = clean(
                payload["history_answer_compatibility_effect"]
                - payload["current_evidence_compatibility_effect"]
            )
            head_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase420-CrossedHeadPath",
                    "created_at": now(),
                    "model": model,
                    "group_id": first["group_id"],
                    "family_id": first["family_id"],
                    "mechanism_id": first["mechanism_id"],
                    "split": first["split"],
                    "interface": first["interface"],
                    "layer": layer,
                    "relative_depth": layer / max(1, layer_count - 1),
                    "depth_bin": depth_bin(layer, layer_count),
                    "head_index": head,
                    **payload,
                    "physical": True,
                    "predictive": False,
                    "causal": False,
                }
            )
        for role in SOURCE_ROLES:
            effect_vector, term_a, term_b = vector_effect(
                cells,
                lambda state, role=role: state["source_vectors"][layer][role],
            )
            path_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": "Phase420-CrossedSourcePath",
                    "created_at": now(),
                    "model": model,
                    "group_id": first["group_id"],
                    "family_id": first["family_id"],
                    "mechanism_id": first["mechanism_id"],
                    "split": first["split"],
                    "interface": first["interface"],
                    "layer": layer,
                    "relative_depth": layer / max(1, layer_count - 1),
                    "depth_bin": depth_bin(layer, layer_count),
                    "source_role": role,
                    "compatibility_effect_vector_norm": clean(
                        float(torch.linalg.vector_norm(effect_vector).item())
                    ),
                    "crossed_term_a_vector_norm": clean(float(torch.linalg.vector_norm(term_a).item())),
                    "crossed_term_b_vector_norm": clean(float(torch.linalg.vector_norm(term_b).item())),
                    "crossed_term_direction_cosine": cosine(term_a, term_b),
                    "physical": True,
                    "predictive": False,
                    "causal": False,
                }
            )
        metric_effects = {}
        for metric in (
            "attention_output_norm",
            "mlp_output_norm",
            "layer_output_norm",
            "cancellation_index",
            "rewrite_novelty",
        ):
            metric_effects[f"{metric}_compatibility_effect"], _, _ = scalar_effect(
                cells,
                lambda state, metric=metric: state["layer_metrics"][layer][metric],
            )
        cosine_values = [
            state["layer_metrics"][layer]["attention_mlp_cosine"]
            for state in cells.values()
            if state["layer_metrics"][layer]["attention_mlp_cosine"] is not None
        ]
        classes = [state["layer_metrics"][layer]["rewrite_class"] for state in cells.values()]
        class_counts = {label: classes.count(label) for label in sorted(set(classes))}
        rewrite_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase420-AttentionMLPRewrite",
                "created_at": now(),
                "model": model,
                "group_id": first["group_id"],
                "family_id": first["family_id"],
                "mechanism_id": first["mechanism_id"],
                "split": first["split"],
                "interface": first["interface"],
                "layer": layer,
                "relative_depth": layer / max(1, layer_count - 1),
                "depth_bin": depth_bin(layer, layer_count),
                **metric_effects,
                "mean_attention_mlp_cosine": clean(sum(cosine_values) / len(cosine_values)),
                "rewrite_class_counts": class_counts,
                "rewrite_class_stable_across_four_cells": len(set(classes)) == 1,
                "stable_rewrite_class": classes[0] if len(set(classes)) == 1 else "mixed_across_cells",
                "physical": True,
                "predictive": False,
                "causal": False,
            }
        )
    return head_rows, path_rows, rewrite_rows


@torch.inference_mode()
def collect_development_physical(
    loaded: Any,
    fast_tokenizer: Any,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    development = [row for row in rows if row["split"] in DEVELOPMENT_SPLITS]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in development:
        groups[(row["group_id"], row["interface"])].append(row)
    layers = get_layers(loaded.model)
    captures: dict[tuple[str, int], Any] = {}
    handles = install_hooks(layers, captures)
    condition_layer_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    rewrite_rows: list[dict[str, Any]] = []
    try:
        with capture_actual_qkv(loaded.key, tuple(range(len(layers))), captures):
            for group_index, ((group_id, interface), cell_rows) in enumerate(sorted(groups.items()), 1):
                if len(cell_rows) != 4:
                    raise RuntimeError(f"Expected four crossed cells: {group_id}/{interface}")
                cells = {}
                ordered = sorted(
                    cell_rows,
                    key=lambda row: (
                        CURRENT_IDENTITIES.index(row["current_identity"]),
                        HISTORY_IDENTITIES.index(row["history_identity"]),
                    ),
                )
                for row in ordered:
                    state, layers_for_condition, registry = collect_physical_condition(
                        loaded, fast_tokenizer, layers, captures, row
                    )
                    state["position_registry"] = registry
                    cells[(row["current_identity"], row["history_identity"])] = state
                    condition_layer_rows.extend(layers_for_condition)
                    position_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "phase_id": "Phase420-PositionRegistry",
                            "created_at": now(),
                            "model": loaded.key,
                            "group_id": row["group_id"],
                            "family_id": row["family_id"],
                            "mechanism_id": row["mechanism_id"],
                            "split": row["split"],
                            "interface": row["interface"],
                            "current_identity": row["current_identity"],
                            "history_identity": row["history_identity"],
                            **registry,
                        }
                    )
                    captures.clear()
                heads, paths, rewrites = summarize_crossed_cells(
                    loaded.key, ordered, cells, len(layers)
                )
                head_rows.extend(heads)
                path_rows.extend(paths)
                rewrite_rows.extend(rewrites)
                del cells
                gc.collect()
                if group_index % 4 == 0 or group_index == len(groups):
                    print(
                        f"[Phase420:{loaded.key}:physical] crossed={group_index}/{len(groups)} "
                        f"head_rows={len(head_rows)}",
                        flush=True,
                    )
    finally:
        for handle in handles:
            handle.remove()
    return condition_layer_rows, position_rows, head_rows, path_rows, rewrite_rows


def run_model(model: str) -> dict[str, Any]:
    qualification = read_json(OUT / "phase420_denominator_qualification.json")
    if not qualification["valid"] or not qualification["development_physical_collection_authorized"]:
        raise RuntimeError("Phase420 denominator is not authorized")
    rows = [row for row in read_jsonl(REGISTERED) if row["model"] == model]
    if len(rows) != 264:
        raise RuntimeError(f"Expected 264 Phase420 rows for {model}, found {len(rows)}")
    loaded = None
    started = time.monotonic()
    try:
        print(f"[Phase420] loading {model}; behavior={len(rows)}", flush=True)
        loaded = load_probe_model(model)
        fast_tokenizer = fast_tokenizer_for(model)
        behavior = collect_behavior(loaded, rows)
        physical = collect_development_physical(loaded, fast_tokenizer, rows)
        condition_layers, positions, heads, paths, rewrites = physical
        model_root = OUT / "models" / model
        write_jsonl(model_root / "phase420_behavior_rows.jsonl", behavior)
        write_jsonl(model_root / "phase420_physical_condition_layer_rows.jsonl", condition_layers)
        write_jsonl(model_root / "phase420_position_registry.jsonl", positions)
        write_jsonl(model_root / "phase420_head_path_rows.jsonl", heads)
        write_jsonl(model_root / "phase420_source_path_rows.jsonl", paths)
        write_jsonl(model_root / "phase420_mlp_rewrite_rows.jsonl", rewrites)
        errors = [
            max(
                row["attention_probability_sum_max_error"],
                row["attention_replay_relative_error"],
                row["mlp_replay_relative_error"],
                row["block_replay_relative_error"],
            )
            for row in condition_layers
        ]
        all_pass = bool(
            len(behavior) == 264
            and all(row["behavior_trace_pass"] for row in behavior)
            and len(positions) == 216
            and all(row["all_roles_nonempty"] for row in positions)
            and all(error <= LEDGER_THRESHOLD for error in errors)
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model,
            "behavior_condition_count": len(behavior),
            "behavior_pass_count": sum(row["behavior_trace_pass"] for row in behavior),
            "behavior_target_event_count": sum(row["target_event_match"] for row in behavior),
            "behavior_exact_answer_count": sum(row["exact_answer_match"] for row in behavior),
            "behavior_initial_right_censored_count": sum(row["initial_right_censored"] for row in behavior),
            "behavior_final_right_censored_count": sum(row["right_censored"] for row in behavior),
            "behavior_extended_horizon_count": sum(row["behavior_horizon_extended"] for row in behavior),
            "development_physical_condition_count": len(positions),
            "physical_holdout_condition_count": 0,
            "physical_holdout_remains_sealed": True,
            "physical_condition_layer_row_count": len(condition_layers),
            "position_registry_row_count": len(positions),
            "head_path_row_count": len(heads),
            "source_path_row_count": len(paths),
            "mlp_rewrite_row_count": len(rewrites),
            "max_physical_ledger_relative_error": max(errors),
            "all_development_rows_pass": all_pass,
            "behavior_rows_sha256": hash_rows(behavior),
            "head_path_rows_sha256": hash_rows(heads),
            "elapsed_seconds": time.monotonic() - started,
            "vram_gb": vram_gb(),
            "physical_holdout_collection_authorized": False,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
            "claim_boundary": "typed_natural_source_write_and_mlp_rewrite_observation_before_prediction_gate",
        }
        write_json(model_root / "phase420_trace_complete.json", summary)
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
