#!/usr/bin/env python3
"""Phase480 component margin ledger precheck.

Records coarse attention/MLP readout changes at label_instruction and
terminal_token positions. This is a component readout ledger, not a causal
ablation, head attribution, or neuron-level mechanism claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model  # noqa: E402
from phase451_glm4_v2_pilot_behavior import load_jsonl, prompt_for, write_jsonl  # noqa: E402
from phase475_mapping_position_scalar_precheck import (  # noqa: E402
    GEN_PATH,
    MAX_PAIR_INDEX,
    SAMPLES_PATH,
    SELECTED_TRANSFORM,
    build_eval_rows,
    locate_role_positions,
)
from phase477_candidate_margin_readout import (  # noqa: E402
    final_norm,
    logit_lens_scores,
    mapping_sign,
    single_token_id,
    truth_sign,
)


OUT_DIR = ROOT / "tests" / "gpt5" / "result" / "phase480_component_margin_ledger"
PROTOCOL_PATH = OUT_DIR / "phase480_component_margin_ledger_protocol.json"
ROWS_PATH = OUT_DIR / "phase480_component_margin_ledger_rows.jsonl"
SUMMARY_PATH = OUT_DIR / "phase480_component_margin_ledger_summary.json"

LEDGER_ROLES = ("label_instruction", "terminal_token")
STATES = ("layer_pre", "attn_post", "mlp_post")
COMPONENTS = ("attention_readout_delta", "mlp_readout_delta", "layer_readout_delta")


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, tuple):
        return value[0]
    return value


def state_scores(
    model: Any,
    norm: Any,
    vec: torch.Tensor,
    a_id: int,
    b_id: int,
    s_mu: int,
    s_truth: int,
) -> dict[str, float]:
    scores = logit_lens_scores(model, norm, vec, a_id, b_id)
    return {
        **scores,
        "margin_true": s_mu * scores["margin_ab"],
        "margin_correct": s_truth * s_mu * scores["margin_ab"],
    }


def margin_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        "delta_margin_ab": after["margin_ab"] - before["margin_ab"],
        "delta_margin_true": after["margin_true"] - before["margin_true"],
        "delta_margin_correct": after["margin_correct"] - before["margin_correct"],
    }


def register_hooks(layers: list[Any], role_positions_ref: dict[str, list[int]], cache: dict[int, dict[str, dict[str, torch.Tensor]]]) -> list[Any]:
    handles = []

    def save_means(layer_idx: int, state: str, tensor: torch.Tensor) -> None:
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise RuntimeError(f"Unexpected tensor shape for layer {layer_idx} {state}: {tuple(tensor.shape)}")
        cache.setdefault(layer_idx, {}).setdefault(state, {})
        for role in LEDGER_ROLES:
            positions = role_positions_ref[role]
            cache[layer_idx][state][role] = tensor[0, positions].mean(dim=0).detach()

    for layer_idx, layer in enumerate(layers):
        def layer_pre_hook(module: Any, inputs: tuple[Any, ...], idx: int = layer_idx) -> None:
            save_means(idx, "layer_pre", first_tensor(inputs[0]))

        def attn_hook(module: Any, inputs: tuple[Any, ...], output: Any, idx: int = layer_idx) -> None:
            save_means(idx, "attn_delta", first_tensor(output))

        def mlp_hook(module: Any, inputs: tuple[Any, ...], output: Any, idx: int = layer_idx) -> None:
            save_means(idx, "mlp_delta", first_tensor(output))

        def layer_post_hook(module: Any, inputs: tuple[Any, ...], output: Any, idx: int = layer_idx) -> None:
            save_means(idx, "mlp_post", first_tensor(output))

        handles.append(layer.register_forward_pre_hook(layer_pre_hook))
        handles.append(layer.self_attn.register_forward_hook(attn_hook))
        handles.append(layer.mlp.register_forward_hook(mlp_hook))
        handles.append(layer.register_forward_hook(layer_post_hook))
    return handles


def cache_to_rows(
    model: Any,
    norm: Any,
    cache: dict[int, dict[str, dict[str, torch.Tensor]]],
    row: dict[str, Any],
    a_id: int,
    b_id: int,
) -> list[dict[str, Any]]:
    out = []
    s_mu = mapping_sign(row["label_mapping"])
    s_truth = truth_sign(bool(row["truth_value"]))
    for layer_index, layer_cache in sorted(cache.items()):
        for role in LEDGER_ROLES:
            layer_pre = layer_cache["layer_pre"][role]
            attn_post = layer_pre + layer_cache["attn_delta"][role]
            mlp_post = layer_cache["mlp_post"][role]
            state_vectors = {
                "layer_pre": layer_pre,
                "attn_post": attn_post,
                "mlp_post": mlp_post,
            }
            scores = {
                state: state_scores(model, norm, vec, a_id, b_id, s_mu, s_truth)
                for state, vec in state_vectors.items()
            }
            components = {
                "attention_readout_delta": margin_delta(scores["attn_post"], scores["layer_pre"]),
                "mlp_readout_delta": margin_delta(scores["mlp_post"], scores["attn_post"]),
                "layer_readout_delta": margin_delta(scores["mlp_post"], scores["layer_pre"]),
            }
            for state in STATES:
                out.append({
                    "model": "glm4",
                    "phase": "phase480",
                    "row_type": "state_margin",
                    "ledger_scope": "coarse_component_readout_not_causal_effect",
                    "sample_id": row["sample_id"],
                    "source_sample_id": row["source_sample_id"],
                    "source_pair_id": row["source_pair_id"],
                    "pair_index": row["pair_index"],
                    "pair_role": row["pair_role"],
                    "transform": row["transform"],
                    "label_mapping": row["label_mapping"],
                    "role": role,
                    "state": state,
                    "truth_value": row["truth_value"],
                    "classification": row["classification"],
                    "normalized_generated": row["normalized_generated"],
                    "behavior_truth": row["behavior_truth"],
                    "layer_index": layer_index,
                    **scores[state],
                })
            for component in COMPONENTS:
                out.append({
                    "model": "glm4",
                    "phase": "phase480",
                    "row_type": "component_delta",
                    "ledger_scope": "coarse_component_readout_not_causal_effect",
                    "sample_id": row["sample_id"],
                    "source_sample_id": row["source_sample_id"],
                    "source_pair_id": row["source_pair_id"],
                    "pair_index": row["pair_index"],
                    "pair_role": row["pair_role"],
                    "transform": row["transform"],
                    "label_mapping": row["label_mapping"],
                    "role": role,
                    "component": component,
                    "truth_value": row["truth_value"],
                    "classification": row["classification"],
                    "normalized_generated": row["normalized_generated"],
                    "behavior_truth": row["behavior_truth"],
                    "layer_index": layer_index,
                    **components[component],
                })
    return out


def trace_rows(model: Any, tokenizer: Any, device: torch.device, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer.padding_side = "left"
    a_id = single_token_id(tokenizer, "A")
    b_id = single_token_id(tokenizer, "B")
    norm = final_norm(model)
    layers = get_layers(model)
    out = []
    role_positions_ref: dict[str, list[int]] = {}
    cache: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    handles = register_hooks(layers, role_positions_ref, cache)
    try:
        for idx, row in enumerate(rows, start=1):
            cache.clear()
            prompt = prompt_for(row["eval_text"])
            all_positions = locate_role_positions(tokenizer, prompt, row["eval_text"])
            role_positions_ref.clear()
            role_positions_ref.update({role: all_positions[role] for role in LEDGER_ROLES})
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, add_special_tokens=False)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                model(**encoded, output_hidden_states=False, use_cache=False)
            out.extend(cache_to_rows(model, norm, cache, row, a_id, b_id))
            if idx % 12 == 0:
                print(f"[phase480] traced {idx}/{len(rows)} prompts", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_rows = [
        row for row in rows
        if row["row_type"] == "state_margin" and row["state"] == "layer_pre" and row["role"] == "terminal_token" and row["layer_index"] == 0
    ]
    prompts = {(row["sample_id"], row["label_mapping"]) for row in prompt_rows}
    behavior_counts = Counter(
        (row["label_mapping"], row["truth_value"], row["classification"], row["behavior_truth"])
        for row in prompt_rows
    )
    row_types = Counter(row["row_type"] for row in rows)
    return {
        "schema_version": "phase480_component_margin_ledger.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "component_margin_ledger_complete",
        "model": "glm4",
        "transform": SELECTED_TRANSFORM,
        "roles": list(LEDGER_ROLES),
        "states": list(STATES),
        "components": list(COMPONENTS),
        "prompt_count": len(prompts),
        "trace_row_count": len(rows),
        "row_type_counts": dict(row_types),
        "behavior_counts": {str(key): value for key, value in behavior_counts.items()},
        "authorization": {
            "component_readout_ledger_authorized": True,
            "causal_effect_authorized": False,
            "head_or_neuron_claim_authorized": False,
            "next_step": "phase481_component_margin_ledger_analysis",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-8bit", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_eval_rows(load_jsonl(SAMPLES_PATH), load_jsonl(GEN_PATH))
    protocol = {
        "schema_version": "phase480_component_margin_ledger_protocol.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_component_margin_ledger_precheck",
        "model": "glm4",
        "transform": SELECTED_TRANSFORM,
        "roles": list(LEDGER_ROLES),
        "states": list(STATES),
        "components": list(COMPONENTS),
        "pair_index_range": [0, MAX_PAIR_INDEX],
        "prompt_count": len(rows),
        "scope": "coarse attention/MLP readout ledger only; no head, neuron, ablation, or causal claim",
    }
    PROTOCOL_PATH.write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model, tokenizer, device = load_model("glm4", use_8bit=args.use_8bit)
    try:
        traced = trace_rows(model, tokenizer, device, rows)
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(ROWS_PATH, traced)
    SUMMARY_PATH.write_text(json.dumps(summarize(traced), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(ROWS_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
