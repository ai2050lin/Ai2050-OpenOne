#!/usr/bin/env python3
"""Map Phase1017 contextual branch interactions in real components."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, release_model
from phase1008_global_response_atlas_scan import StateCapture
from phase1014_bf16_precision_confirmation import load_bf16
from phase1015_query_surface_scan import (
    MultiPositionHeadCapture,
)
from phase1016_query_factorial_scan import safe_cosine
from phase1017_semantic_niche_protocol import (
    CAPTURE_ROLES,
    MODELS,
    OUT_ROOT,
    PHASE,
    PROTOCOL_REVISION,
    STATES,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


STATE_INDEX = {state: index for index, state in enumerate(STATES)}
ANALYSIS_CONTRASTS = (
    "BA",
    "BN",
    "BT",
    "BT_L0",
    "BT_L1",
    "LA",
    "LN",
    "T_B0",
    "T_B1",
    "I",
)
CONTRAST_INDEX = {
    name: index for index, name in enumerate(ANALYSIS_CONTRASTS)
}
DIRECTION_CONTRASTS = ("BA", "BN", "BT", "BT_L0", "BT_L1")
DIRECTION_INDEX = {
    name: index for index, name in enumerate(DIRECTION_CONTRASTS)
}
ROLE_INDEX = {role: index for index, role in enumerate(CAPTURE_ROLES)}
KEY_DIRECTION_ROLES = (
    "target_word",
    "query_operator",
    "answer_boundary",
)
EPSILON = 1e-12


def direction_consistency(
    sums: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    result = np.full(sums.shape[:-1], np.nan, dtype=np.float32)
    squared = np.einsum(
        "...d,...d->...",
        sums.astype(np.float64, copy=False),
        sums.astype(np.float64, copy=False),
    )
    valid = counts >= 2
    result[valid] = (
        (squared[valid] - counts[valid])
        / (counts[valid] * (counts[valid] - 1.0))
    ).astype(np.float32)
    return result


def event_definitions(
    n_layers: int,
    head_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[tuple[str, int]],
    list[tuple[int, int]],
]:
    events = []
    whole_keys: list[tuple[str, int]] = [("residual", 0)]
    for depth in range(1, n_layers + 1):
        whole_keys.extend((
            ("residual", depth),
            ("attention_output", depth),
            ("mlp_output", depth),
        ))
    for component, depth in whole_keys:
        events.append({
            "schema_version": "phase1017_semantic_niche_event.v1",
            "phase": PHASE,
            "event_index": len(events),
            "event_id": f"{component}.d{depth:02d}",
            "component": component,
            "depth": int(depth),
            "relative_depth": float(depth / max(n_layers, 1)),
            "head": None,
            "vector_space": "model_width",
            "claim": "contextual_response_only",
        })
    head_keys = []
    for depth in range(1, n_layers + 1):
        for head in range(head_count):
            head_keys.append((depth, head))
            events.append({
                "schema_version": "phase1017_semantic_niche_event.v1",
                "phase": PHASE,
                "event_index": len(events),
                "event_id": (
                    f"attention_head.d{depth:02d}.h{head:02d}"
                ),
                "component": "attention_head_pre_o_proj",
                "depth": int(depth),
                "relative_depth": float(depth / max(n_layers, 1)),
                "head": int(head),
                "vector_space": "head_width",
                "claim": "physical_head_contextual_response_only",
            })
    return events, whole_keys, head_keys


def mean_scale(
    values: torch.Tensor,
    states: tuple[str, ...],
) -> torch.Tensor:
    norms = torch.stack([
        torch.linalg.vector_norm(values[STATE_INDEX[state]], dim=-1)
        for state in states
    ])
    return norms.mean(dim=0)


def contrast_values(
    values: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    a00 = values[STATE_INDEX["a0_l0"]]
    a10 = values[STATE_INDEX["a1_l0"]]
    a01 = values[STATE_INDEX["a0_l1"]]
    a11 = values[STATE_INDEX["a1_l1"]]
    n00 = values[STATE_INDEX["n0_l0"]]
    n10 = values[STATE_INDEX["n1_l0"]]
    n01 = values[STATE_INDEX["n0_l1"]]
    n11 = values[STATE_INDEX["n1_l1"]]
    identity = values[STATE_INDEX["identity"]]

    ba_l0 = a10 - a00
    ba_l1 = a11 - a01
    bn_l0 = n10 - n00
    bn_l1 = n11 - n01
    bt_l0 = ba_l0 - bn_l0
    bt_l1 = ba_l1 - bn_l1
    deltas = {
        "BA": 0.5 * (ba_l0 + ba_l1),
        "BN": 0.5 * (bn_l0 + bn_l1),
        "BT": 0.5 * (bt_l0 + bt_l1),
        "BT_L0": bt_l0,
        "BT_L1": bt_l1,
        "LA": 0.5 * ((a01 - a00) + (a11 - a10)),
        "LN": 0.5 * ((n01 - n00) + (n11 - n10)),
        "T_B0": 0.5 * ((a00 - n00) + (a01 - n01)),
        "T_B1": 0.5 * ((a10 - n10) + (a11 - n11)),
        "I": identity - a00,
    }
    ambiguous_states = ("a0_l0", "a1_l0", "a0_l1", "a1_l1")
    neutral_states = ("n0_l0", "n1_l0", "n0_l1", "n1_l1")
    all_factorial = ambiguous_states + neutral_states
    scales = {
        "BA": mean_scale(values, ambiguous_states),
        "BN": mean_scale(values, neutral_states),
        "BT": mean_scale(values, all_factorial),
        "BT_L0": mean_scale(
            values,
            ("a0_l0", "a1_l0", "n0_l0", "n1_l0"),
        ),
        "BT_L1": mean_scale(
            values,
            ("a0_l1", "a1_l1", "n0_l1", "n1_l1"),
        ),
        "LA": mean_scale(values, ambiguous_states),
        "LN": mean_scale(values, neutral_states),
        "T_B0": mean_scale(
            values,
            ("a0_l0", "a0_l1", "n0_l0", "n0_l1"),
        ),
        "T_B1": mean_scale(
            values,
            ("a1_l0", "a1_l1", "n1_l0", "n1_l1"),
        ),
        "I": mean_scale(values, ("identity", "a0_l0")),
    }
    return deltas, scales


def prediction(
    logits: torch.Tensor,
    case: dict[str, Any],
) -> dict[str, Any]:
    gold_id = int(case["candidate_token_ids"][case["gold"]])
    foil_id = int(case["candidate_token_ids"][case["foil"]])
    pair = torch.tensor(
        [gold_id, foil_id],
        dtype=torch.long,
        device=logits.device,
    )
    pair_prediction = int(
        pair[logits.index_select(0, pair).argmax()].item()
    )
    full_prediction = int(logits.argmax().item())
    return {
        "gold": case["gold"],
        "foil": case["foil"],
        "gold_id": gold_id,
        "foil_id": foil_id,
        "candidate_margin": float(
            logits[gold_id].item() - logits[foil_id].item()
        ),
        "candidate_hit": bool(pair_prediction == gold_id),
        "full_vocabulary_prediction_id": full_prediction,
        "full_vocabulary_hit": bool(full_prediction == gold_id),
    }


def add_direction(
    *,
    whole_delta: torch.Tensor,
    head_delta: torch.Tensor,
    target_index: int,
    whole_sums: np.ndarray,
    head_sums: np.ndarray,
    whole_counts: np.ndarray,
    head_counts: np.ndarray,
) -> None:
    whole_norm = torch.linalg.vector_norm(whole_delta, dim=-1)
    head_norm = torch.linalg.vector_norm(head_delta, dim=-1)
    whole_valid = whole_norm > EPSILON
    head_valid = head_norm > EPSILON
    whole_unit = torch.zeros_like(whole_delta)
    head_unit = torch.zeros_like(head_delta)
    whole_unit[whole_valid] = (
        whole_delta[whole_valid] / whole_norm[whole_valid, None]
    )
    head_unit[head_valid] = (
        head_delta[head_valid] / head_norm[head_valid, None]
    )
    whole_sums[target_index] += whole_unit.numpy()
    head_sums[target_index] += head_unit.numpy()
    whole_counts[target_index] += whole_valid.numpy().astype(np.int32)
    head_counts[target_index] += head_valid.numpy().astype(np.int32)


def run_panel(
    *,
    model,
    layers,
    info,
    head_count: int,
    device,
    model_name: str,
    prompt_mode: str,
    word: str,
    split: str,
    panel_units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
    natural_by_id: dict[str, dict[str, Any]],
    output_root: Path,
    events: list[dict[str, Any]],
    whole_keys: list[tuple[str, int]],
    head_keys: list[tuple[int, int]],
    state_capture: StateCapture,
    head_capture: MultiPositionHeadCapture,
) -> dict[str, Any]:
    unit_count = len(panel_units)
    role_count = len(CAPTURE_ROLES)
    event_count = len(events)
    whole_count = len(whole_keys)
    head_event_count = len(head_keys)
    d_model = int(info.d_model)
    attention_width = int(layers[0].self_attn.o_proj.in_features)
    if attention_width % head_count:
        raise RuntimeError("pre-o_proj attention width is not head aligned")
    head_width = attention_width // head_count

    normalized_magnitude = np.full(
        (
            unit_count,
            len(ANALYSIS_CONTRASTS),
            role_count,
            event_count,
        ),
        np.nan,
        dtype=np.float32,
    )
    whole_sums = np.zeros(
        (
            len(DIRECTION_CONTRASTS),
            role_count,
            whole_count,
            d_model,
        ),
        dtype=np.float32,
    )
    head_sums = np.zeros(
        (
            len(DIRECTION_CONTRASTS),
            role_count,
            head_event_count,
            head_width,
        ),
        dtype=np.float32,
    )
    whole_counts = np.zeros(
        (
            len(DIRECTION_CONTRASTS),
            role_count,
            whole_count,
        ),
        dtype=np.int32,
    )
    head_counts = np.zeros(
        (
            len(DIRECTION_CONTRASTS),
            role_count,
            head_event_count,
        ),
        dtype=np.int32,
    )
    unit_rows = []
    identity_maximum = 0.0
    interaction_cue_maximum = 0.0
    target_embedding_interaction_maximum = 0.0
    started = time.time()

    for unit_index, unit in enumerate(panel_units):
        cases = [
            case_by_id[unit["record_ids"][state]]
            for state in STATES
        ]
        state_whole = []
        state_head = []
        state_behavior = {}
        for state, case in zip(STATES, cases):
            input_ids = torch.tensor(
                [case["input_ids"]],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            role_positions = [
                int(case["role_positions"][role])
                for role in CAPTURE_ROLES
            ]
            positions = torch.tensor(
                [role_positions],
                dtype=torch.long,
                device=device,
            )
            head_positions = torch.tensor(
                role_positions,
                dtype=torch.long,
                device=device,
            )
            state_capture.begin(positions)
            head_capture.begin(head_positions)
            try:
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                state_capture.validate()
                head_capture.validate()
                whole = torch.stack([
                    state_capture.captured[key][0].float().cpu()
                    for key in whole_keys
                ]).permute(1, 0, 2).contiguous()
                heads = torch.stack([
                    head_capture.values[depth][0, :, head].float().cpu()
                    for depth, head in head_keys
                ]).permute(1, 0, 2).contiguous()
                state_whole.append(whole)
                state_head.append(heads)
                state_behavior[state] = prediction(
                    output.logits[0, -1].float(),
                    case,
                )
                natural = natural_by_id.get(case["record_id"])
                if natural is not None:
                    state_behavior[state]["generation_first_word_hit"] = bool(
                        natural["generation_first_word_hit"]
                    )
                    state_behavior[state]["generated_first_word"] = (
                        natural["generated_first_word"]
                    )
                del output, whole, heads
            finally:
                state_capture.captured = {}
                head_capture.values = {}
                del (
                    input_ids,
                    attention_mask,
                    positions,
                    head_positions,
                )

        whole_values = torch.stack(state_whole)
        head_values = torch.stack(state_head)
        whole_deltas, whole_scales = contrast_values(whole_values)
        head_deltas, head_scales = contrast_values(head_values)
        ambiguous_behavior = [
            state_behavior[state]
            for state in ("a0_l0", "a1_l0", "a0_l1", "a1_l1")
        ]
        neutral_behavior = [
            state_behavior[state]
            for state in ("n0_l0", "n1_l0", "n0_l1", "n1_l1")
        ]
        natural_available = all(
            "generation_first_word_hit" in row
            for row in ambiguous_behavior
        )
        unit_rows.append({
            "schema_version": "phase1017_semantic_niche_scan_unit.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "prompt_mode": prompt_mode,
            "word": word,
            "split": split,
            "template": int(unit["template"]),
            "world": int(unit["world"]),
            "unit_index": int(unit_index),
            "unit_id": unit["unit_id"],
            "state_behavior": state_behavior,
            "ambiguous_candidate_all_hit": bool(all(
                row["candidate_hit"] for row in ambiguous_behavior
            )),
            "ambiguous_candidate_hit_count": int(sum(
                row["candidate_hit"] for row in ambiguous_behavior
            )),
            "ambiguous_generation_all_hit": (
                bool(all(
                    row["generation_first_word_hit"]
                    for row in ambiguous_behavior
                ))
                if natural_available else None
            ),
            "ambiguous_generation_hit_count": (
                int(sum(
                    row["generation_first_word_hit"]
                    for row in ambiguous_behavior
                ))
                if natural_available else None
            ),
            "neutral_candidate_all_hit": bool(all(
                row["candidate_hit"] for row in neutral_behavior
            )),
            "neutral_candidate_hit_count": int(sum(
                row["candidate_hit"] for row in neutral_behavior
            )),
        })

        for name in ANALYSIS_CONTRASTS:
            contrast_index = CONTRAST_INDEX[name]
            whole_raw = torch.linalg.vector_norm(
                whole_deltas[name],
                dim=-1,
            )
            head_raw = torch.linalg.vector_norm(
                head_deltas[name],
                dim=-1,
            )
            whole_norm = whole_raw / torch.clamp(
                whole_scales[name],
                min=EPSILON,
            )
            head_norm = head_raw / torch.clamp(
                head_scales[name],
                min=EPSILON,
            )
            normalized_magnitude[
                unit_index,
                contrast_index,
                :,
                :whole_count,
            ] = whole_norm.numpy()
            normalized_magnitude[
                unit_index,
                contrast_index,
                :,
                whole_count:,
            ] = head_norm.numpy()
            if name == "I":
                identity_maximum = max(
                    identity_maximum,
                    float(whole_raw.max().item()),
                    float(head_raw.max().item()),
                )
            if name == "BT":
                cue_index = ROLE_INDEX["cue"]
                interaction_cue_maximum = max(
                    interaction_cue_maximum,
                    float(whole_raw[cue_index].max().item()),
                    float(head_raw[cue_index].max().item()),
                )
                target_index = ROLE_INDEX["target_word"]
                target_embedding_interaction_maximum = max(
                    target_embedding_interaction_maximum,
                    float(whole_raw[target_index, 0].item()),
                )

        for name in DIRECTION_CONTRASTS:
            add_direction(
                whole_delta=whole_deltas[name],
                head_delta=head_deltas[name],
                target_index=DIRECTION_INDEX[name],
                whole_sums=whole_sums,
                head_sums=head_sums,
                whole_counts=whole_counts,
                head_counts=head_counts,
            )

        del (
            state_whole,
            state_head,
            whole_values,
            head_values,
            whole_deltas,
            whole_scales,
            head_deltas,
            head_scales,
        )
        if (unit_index + 1) % 4 == 0:
            print(
                f"[scan] {model_name}/{word}/{split} "
                f"{unit_index + 1}/{unit_count}",
                flush=True,
            )

    whole_consistency = direction_consistency(
        whole_sums,
        whole_counts,
    )
    head_consistency = direction_consistency(
        head_sums,
        head_counts,
    )
    bt_l0 = DIRECTION_INDEX["BT_L0"]
    bt_l1 = DIRECTION_INDEX["BT_L1"]
    ba = DIRECTION_INDEX["BA"]
    bn = DIRECTION_INDEX["BN"]
    whole_lexical_alignment = safe_cosine(
        whole_sums[bt_l0],
        whole_sums[bt_l1],
    )
    head_lexical_alignment = safe_cosine(
        head_sums[bt_l0],
        head_sums[bt_l1],
    )
    whole_ambiguous_neutral_alignment = safe_cosine(
        whole_sums[ba],
        whole_sums[bn],
    )
    head_ambiguous_neutral_alignment = safe_cosine(
        head_sums[ba],
        head_sums[bn],
    )

    panel_root = output_root / model_name / word / split
    panel_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        panel_root / "response_scalars.npz",
        normalized_magnitude=normalized_magnitude,
        contrast_names=np.asarray(ANALYSIS_CONTRASTS),
        role_names=np.asarray(CAPTURE_ROLES),
    )
    np.savez_compressed(
        panel_root / "direction_metrics.npz",
        whole_consistency=whole_consistency,
        head_consistency=head_consistency,
        whole_count=whole_counts,
        head_count=head_counts,
        whole_lexical_alignment=whole_lexical_alignment,
        head_lexical_alignment=head_lexical_alignment,
        whole_ambiguous_neutral_alignment=(
            whole_ambiguous_neutral_alignment
        ),
        head_ambiguous_neutral_alignment=(
            head_ambiguous_neutral_alignment
        ),
        direction_contrast_names=np.asarray(DIRECTION_CONTRASTS),
        role_names=np.asarray(CAPTURE_ROLES),
    )
    key_indices = [ROLE_INDEX[role] for role in KEY_DIRECTION_ROLES]
    np.savez_compressed(
        panel_root / "key_direction_sums.npz",
        whole_sums=whole_sums[:, key_indices],
        head_sums=head_sums[:, key_indices],
        whole_count=whole_counts[:, key_indices],
        head_count=head_counts[:, key_indices],
        direction_contrast_names=np.asarray(DIRECTION_CONTRASTS),
        role_names=np.asarray(KEY_DIRECTION_ROLES),
    )
    write_jsonl(panel_root / "units.jsonl", unit_rows)
    summary = {
        "schema_version": "phase1017_semantic_niche_scan_panel.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "word": word,
        "split": split,
        "unit_count": unit_count,
        "event_count": event_count,
        "whole_event_count": whole_count,
        "head_event_count": head_event_count,
        "role_count": role_count,
        "singleton_forward_count": unit_count * len(STATES),
        "identity_maximum": identity_maximum,
        "interaction_cue_maximum": interaction_cue_maximum,
        "target_embedding_interaction_maximum": (
            target_embedding_interaction_maximum
        ),
        "ambiguous_candidate_all_hit_count": int(sum(
            row["ambiguous_candidate_all_hit"] for row in unit_rows
        )),
        "ambiguous_generation_all_hit_count": int(sum(
            row["ambiguous_generation_all_hit"] is True
            for row in unit_rows
        )),
        "neutral_candidate_all_hit_count": int(sum(
            row["neutral_candidate_all_hit"] for row in unit_rows
        )),
        "elapsed_seconds": time.time() - started,
        "claim_limits": [
            "BT is a factorial interaction, not a semantic mechanism equation",
            "stable interaction is not a causal edge",
            "fixed weights only test contextual state reconstruction",
        ],
    }
    write_json(panel_root / "summary.json", summary)
    return summary


def run_model(
    model_name: str,
    *,
    output_namespace: str,
    max_panels: int | None,
    resume: bool,
) -> dict[str, Any]:
    prereg = read_json(OUT_ROOT / "protocol" / "preregistration.json")
    if int(prereg["protocol_revision"]) != PROTOCOL_REVISION:
        raise RuntimeError("Phase1017 protocol revision drift")
    selection = read_json(
        OUT_ROOT / "behavior" / model_name / "selection.json"
    )
    if selection["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("behavior/protocol digest mismatch")
    prompt_mode = selection["selected_prompt_mode"]
    units = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"units.{model_name}.{prompt_mode}.jsonl"
    )
    cases = read_jsonl(
        OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.{prompt_mode}.jsonl"
    )
    natural_rows = read_jsonl(
        OUT_ROOT / "behavior" / model_name / "formal.jsonl"
    )
    case_by_id = {row["record_id"]: row for row in cases}
    natural_by_id = {row["record_id"]: row for row in natural_rows}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[(unit["word"], unit["split"])].append(unit)
    panel_items = sorted(grouped.items())
    if max_panels is not None:
        panel_items = panel_items[:max_panels]

    output_root = OUT_ROOT / output_namespace
    model_root = output_root / model_name
    model_root.mkdir(parents=True, exist_ok=True)
    model = tokenizer = device = None
    state_capture = head_capture = None
    panel_summaries = []
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        info = get_model_info(model, model_name)
        layers = get_layers(model)
        head_count = int(model.config.num_attention_heads)
        events, whole_keys, head_keys = event_definitions(
            int(info.n_layers),
            head_count,
        )
        write_jsonl(model_root / "events.jsonl", events)
        state_capture = StateCapture(model, layers)
        head_capture = MultiPositionHeadCapture(layers, head_count)
        state_capture.register()
        head_capture.register()
        for (word, split), panel_units in panel_items:
            panel_root = model_root / word / split
            summary_path = panel_root / "summary.json"
            required = (
                panel_root / "response_scalars.npz",
                panel_root / "direction_metrics.npz",
                panel_root / "key_direction_sums.npz",
                panel_root / "units.jsonl",
            )
            if (
                resume
                and summary_path.exists()
                and all(path.exists() for path in required)
            ):
                existing = read_json(summary_path)
                if (
                    int(existing["protocol_revision"])
                    == PROTOCOL_REVISION
                    and existing["model"] == model_name
                    and existing["word"] == word
                    and existing["split"] == split
                    and int(existing["unit_count"]) == len(panel_units)
                ):
                    panel_summaries.append(existing)
                    print(
                        f"[resume] {model_name}/{word}/{split}",
                        flush=True,
                    )
                    continue
            panel_summaries.append(run_panel(
                model=model,
                layers=layers,
                info=info,
                head_count=head_count,
                device=device,
                model_name=model_name,
                prompt_mode=prompt_mode,
                word=word,
                split=split,
                panel_units=panel_units,
                case_by_id=case_by_id,
                natural_by_id=natural_by_id,
                output_root=output_root,
                events=events,
                whole_keys=whole_keys,
                head_keys=head_keys,
                state_capture=state_capture,
                head_capture=head_capture,
            ))
        summary = {
            "schema_version": "phase1017_semantic_niche_scan_model.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "prompt_mode": prompt_mode,
            "output_namespace": output_namespace,
            "precision": "bf16",
            "placement": placement,
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": head_count,
                "head_width": int(
                    layers[0].self_attn.o_proj.in_features // head_count
                ),
            },
            "panel_count": len(panel_summaries),
            "unit_count": int(sum(
                row["unit_count"] for row in panel_summaries
            )),
            "singleton_forward_count": int(sum(
                row["singleton_forward_count"]
                for row in panel_summaries
            )),
            "identity_maximum": float(max(
                row["identity_maximum"] for row in panel_summaries
            )),
            "interaction_cue_maximum": float(max(
                row["interaction_cue_maximum"]
                for row in panel_summaries
            )),
            "target_embedding_interaction_maximum": float(max(
                row["target_embedding_interaction_maximum"]
                for row in panel_summaries
            )),
            "ambiguous_candidate_all_hit_count": int(sum(
                row["ambiguous_candidate_all_hit_count"]
                for row in panel_summaries
            )),
            "ambiguous_generation_all_hit_count": int(sum(
                row["ambiguous_generation_all_hit_count"]
                for row in panel_summaries
            )),
            "elapsed_seconds": time.time() - started,
        }
        write_json(model_root / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return summary
    finally:
        if head_capture is not None:
            head_capture.close()
        if state_capture is not None:
            state_capture.close()
        if model is not None:
            release_model(model)
        del model, tokenizer, device, state_capture, head_capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--output-namespace",
        default="formal_scan",
    )
    parser.add_argument("--max-panels", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_model(
        args.model,
        output_namespace=args.output_namespace,
        max_panels=args.max_panels,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
