#!/usr/bin/env python3
"""Scan Phase1016 factorial query responses without persisting hidden tensors."""

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
    consistency_from_sums,
)
from phase1016_query_factorial_protocol import (
    CAPTURE_ROLES,
    FACTORIAL_STATES,
    FAMILIES,
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
    "S",
    "L",
    "SL",
    "O",
    "E",
    "I",
    "S_L0",
    "S_L1",
    "L_S0",
    "L_S1",
)
CONTRAST_INDEX = {
    name: index for index, name in enumerate(ANALYSIS_CONTRASTS)
}
DIRECTION_CONTRASTS = ("S", "S_L0", "S_L1")
DIRECTION_INDEX = {
    name: index for index, name in enumerate(DIRECTION_CONTRASTS)
}
DIRECTION_MODES = ("raw", "canonical")
ROLE_INDEX = {role: index for index, role in enumerate(CAPTURE_ROLES)}
SEMANTIC_PREFIX_ROLES = (
    "focal_source",
    "focal_relation",
    "focal_target",
    "background_source",
    "background_relation",
    "background_target",
    "query_anchor",
)
KEY_DIRECTION_ROLES = ("query_operator", "answer_boundary")
EPSILON = 1e-12


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
            "schema_version": "phase1016_factorial_event.v1",
            "phase": PHASE,
            "event_index": len(events),
            "event_id": f"{component}.d{depth:02d}",
            "component": component,
            "depth": int(depth),
            "relative_depth": float(depth / max(n_layers, 1)),
            "head": None,
            "vector_space": "model_width",
            "claim": "role_conditioned_response_only",
        })
    head_keys = []
    for depth in range(1, n_layers + 1):
        for head in range(head_count):
            head_keys.append((depth, head))
            events.append({
                "schema_version": "phase1016_factorial_event.v1",
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
                "claim": "physical_head_role_response_only",
            })
    return events, whole_keys, head_keys


def contrast_values(
    values: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    h00 = values[STATE_INDEX["s0_l0"]]
    h10 = values[STATE_INDEX["s1_l0"]]
    h01 = values[STATE_INDEX["s0_l1"]]
    h11 = values[STATE_INDEX["s1_l1"]]
    order = values[STATE_INDEX["order_control"]]
    entity = values[STATE_INDEX["entity_control"]]
    identity = values[STATE_INDEX["identity"]]
    s_l0 = h10 - h00
    s_l1 = h11 - h01
    l_s0 = h01 - h00
    l_s1 = h11 - h10
    deltas = {
        "S": 0.5 * (s_l0 + s_l1),
        "L": 0.5 * (l_s0 + l_s1),
        "SL": h11 - h10 - h01 + h00,
        "O": order - h00,
        "E": entity - h00,
        "I": identity - h00,
        "S_L0": s_l0,
        "S_L1": s_l1,
        "L_S0": l_s0,
        "L_S1": l_s1,
    }
    norms = torch.linalg.vector_norm(values, dim=-1)
    factorial_scale = 0.25 * (
        norms[STATE_INDEX["s0_l0"]]
        + norms[STATE_INDEX["s1_l0"]]
        + norms[STATE_INDEX["s0_l1"]]
        + norms[STATE_INDEX["s1_l1"]]
    )
    scales = {
        name: factorial_scale
        for name in ("S", "L", "SL")
    }
    scales.update({
        "O": 0.5 * (
            norms[STATE_INDEX["order_control"]]
            + norms[STATE_INDEX["s0_l0"]]
        ),
        "E": 0.5 * (
            norms[STATE_INDEX["entity_control"]]
            + norms[STATE_INDEX["s0_l0"]]
        ),
        "I": 0.5 * (
            norms[STATE_INDEX["identity"]]
            + norms[STATE_INDEX["s0_l0"]]
        ),
        "S_L0": 0.5 * (
            norms[STATE_INDEX["s1_l0"]]
            + norms[STATE_INDEX["s0_l0"]]
        ),
        "S_L1": 0.5 * (
            norms[STATE_INDEX["s1_l1"]]
            + norms[STATE_INDEX["s0_l1"]]
        ),
        "L_S0": 0.5 * (
            norms[STATE_INDEX["s0_l1"]]
            + norms[STATE_INDEX["s0_l0"]]
        ),
        "L_S1": 0.5 * (
            norms[STATE_INDEX["s1_l1"]]
            + norms[STATE_INDEX["s1_l0"]]
        ),
    })
    return deltas, scales


def safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.einsum(
        "...d,...d->...",
        left.astype(np.float64, copy=False),
        right.astype(np.float64, copy=False),
    )
    left_norm = np.sqrt(np.einsum(
        "...d,...d->...",
        left.astype(np.float64, copy=False),
        left.astype(np.float64, copy=False),
    ))
    right_norm = np.sqrt(np.einsum(
        "...d,...d->...",
        right.astype(np.float64, copy=False),
        right.astype(np.float64, copy=False),
    ))
    denominator = left_norm * right_norm
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = denominator > EPSILON
    result[valid] = (numerator[valid] / denominator[valid]).astype(
        np.float32
    )
    return result


def prediction(
    logits: torch.Tensor,
    case: dict[str, Any],
) -> dict[str, Any]:
    gold_id = int(case["candidate_token_ids"][case["gold"]])
    foil_id = int(case["candidate_token_ids"][case["foil"]])
    candidates = torch.tensor(
        [gold_id, foil_id],
        dtype=torch.long,
        device=logits.device,
    )
    pair_id = int(
        candidates[logits.index_select(0, candidates).argmax()].item()
    )
    full_id = int(logits.argmax().item())
    return {
        "gold_id": gold_id,
        "foil_id": foil_id,
        "candidate_prediction_id": pair_id,
        "full_vocabulary_prediction_id": full_id,
        "candidate_hit": bool(pair_id == gold_id),
        "full_vocabulary_hit": bool(full_id == gold_id),
        "candidate_margin": float(
            logits[gold_id].item() - logits[foil_id].item()
        ),
    }


def add_direction(
    *,
    whole_delta: torch.Tensor,
    head_delta: torch.Tensor,
    target_index: int,
    sign: int,
    whole_sums: np.ndarray,
    head_sums: np.ndarray,
    whole_counts: np.ndarray,
    head_counts: np.ndarray,
) -> None:
    whole_norm = torch.linalg.vector_norm(whole_delta, dim=-1)
    head_norm = torch.linalg.vector_norm(head_delta, dim=-1)
    whole_direction = (
        whole_delta
        / torch.clamp(whole_norm[..., None], min=EPSILON)
    ).numpy()
    head_direction = (
        head_delta
        / torch.clamp(head_norm[..., None], min=EPSILON)
    ).numpy()
    whole_valid = whole_norm.numpy() > EPSILON
    head_valid = head_norm.numpy() > EPSILON
    whole_sums[0, target_index] += whole_direction
    head_sums[0, target_index] += head_direction
    whole_sums[1, target_index] += sign * whole_direction
    head_sums[1, target_index] += sign * head_direction
    whole_counts[target_index] += whole_valid.astype(np.int32)
    head_counts[target_index] += head_valid.astype(np.int32)


def run_panel(
    *,
    model,
    layers,
    info,
    head_count: int,
    device,
    model_name: str,
    prompt_mode: str,
    family: str,
    template: int,
    panel_units: list[dict[str, Any]],
    case_by_id: dict[str, dict[str, Any]],
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
            len(DIRECTION_MODES),
            len(DIRECTION_CONTRASTS),
            role_count,
            whole_count,
            d_model,
        ),
        dtype=np.float32,
    )
    head_sums = np.zeros(
        (
            len(DIRECTION_MODES),
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
    semantic_prefix_maximum = 0.0
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
        factorial_behavior = [
            state_behavior[state] for state in FACTORIAL_STATES
        ]
        unit_rows.append({
            "schema_version": "phase1016_factorial_scan_unit.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "model": model_name,
            "prompt_mode": prompt_mode,
            "family": family,
            "template": int(template),
            "split": unit["split"],
            "name_pool": int(unit["name_pool"]),
            "world_index": int(unit["world_index"]),
            "unit_index": int(unit_index),
            "unit_id": unit["unit_id"],
            "canonical_semantic_sign": int(
                unit["canonical_semantic_sign"]
            ),
            "state_behavior": state_behavior,
            "factorial_candidate_all_hit": bool(all(
                row["candidate_hit"] for row in factorial_behavior
            )),
            "factorial_full_vocab_all_hit": bool(all(
                row["full_vocabulary_hit"]
                for row in factorial_behavior
            )),
            "factorial_candidate_hit_count": int(sum(
                row["candidate_hit"] for row in factorial_behavior
            )),
            "semantic_switch_pair_hit_l0": bool(
                state_behavior["s0_l0"]["candidate_hit"]
                and state_behavior["s1_l0"]["candidate_hit"]
            ),
            "semantic_switch_pair_hit_l1": bool(
                state_behavior["s0_l1"]["candidate_hit"]
                and state_behavior["s1_l1"]["candidate_hit"]
            ),
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
            if name == "S":
                prefix = [
                    ROLE_INDEX[role] for role in SEMANTIC_PREFIX_ROLES
                ]
                semantic_prefix_maximum = max(
                    semantic_prefix_maximum,
                    float(whole_raw[prefix].max().item()),
                    float(head_raw[prefix].max().item()),
                )

        sign = int(unit["canonical_semantic_sign"])
        for name in DIRECTION_CONTRASTS:
            add_direction(
                whole_delta=whole_deltas[name],
                head_delta=head_deltas[name],
                target_index=DIRECTION_INDEX[name],
                sign=sign,
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
        if (unit_index + 1) % 8 == 0:
            print(
                f"[scan] {model_name}/{family}/t{template} "
                f"{unit_index + 1}/{unit_count}",
                flush=True,
            )

    whole_consistency = consistency_from_sums(
        whole_sums,
        whole_counts,
    )
    head_consistency = consistency_from_sums(
        head_sums,
        head_counts,
    )
    l0_index = DIRECTION_INDEX["S_L0"]
    l1_index = DIRECTION_INDEX["S_L1"]
    whole_lexical_alignment = safe_cosine(
        whole_sums[1, l0_index],
        whole_sums[1, l1_index],
    )
    head_lexical_alignment = safe_cosine(
        head_sums[1, l0_index],
        head_sums[1, l1_index],
    )
    panel_root = output_root / model_name / family / f"template_{template}"
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
        direction_mode_names=np.asarray(DIRECTION_MODES),
        direction_contrast_names=np.asarray(DIRECTION_CONTRASTS),
        role_names=np.asarray(CAPTURE_ROLES),
    )
    key_indices = [
        ROLE_INDEX[role] for role in KEY_DIRECTION_ROLES
    ]
    np.savez_compressed(
        panel_root / "key_direction_sums.npz",
        whole_canonical=whole_sums[1][:, key_indices],
        head_canonical=head_sums[1][:, key_indices],
        whole_count=whole_counts[:, key_indices],
        head_count=head_counts[:, key_indices],
        direction_contrast_names=np.asarray(DIRECTION_CONTRASTS),
        role_names=np.asarray(KEY_DIRECTION_ROLES),
    )
    write_jsonl(panel_root / "units.jsonl", unit_rows)
    summary = {
        "schema_version": "phase1016_factorial_scan_panel.v1",
        "phase": PHASE,
        "protocol_revision": PROTOCOL_REVISION,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "family": family,
        "template": int(template),
        "split": panel_units[0]["split"],
        "unit_count": unit_count,
        "event_count": event_count,
        "whole_event_count": whole_count,
        "head_event_count": head_event_count,
        "role_count": role_count,
        "singleton_forward_count": unit_count * len(STATES),
        "identity_maximum": identity_maximum,
        "semantic_causal_prefix_maximum": semantic_prefix_maximum,
        "factorial_candidate_all_hit_count": int(sum(
            row["factorial_candidate_all_hit"] for row in unit_rows
        )),
        "factorial_full_vocab_all_hit_count": int(sum(
            row["factorial_full_vocab_all_hit"] for row in unit_rows
        )),
        "mean_factorial_candidate_hit_count": float(np.mean([
            row["factorial_candidate_hit_count"] for row in unit_rows
        ])),
        "elapsed_seconds": time.time() - started,
        "claim_limits": [
            "factorial contrasts are measurement operators, not a mechanism",
            "ordered co-response is not a causal edge",
            "behavior-failed units remain failure-computation controls",
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
        raise RuntimeError("Phase1016 protocol revision drift")
    selection = read_json(
        OUT_ROOT
        / "behavior_calibration"
        / model_name
        / "selection.json"
    )
    if selection["protocol_digest"] != prereg["protocol_digest"]:
        raise RuntimeError("behavior selection protocol digest drift")
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
    case_by_id = {row["record_id"]: row for row in cases}
    grouped = defaultdict(list)
    for unit in units:
        grouped[(unit["family"], int(unit["template"]))].append(unit)
    panel_items = sorted(grouped.items())
    if max_panels is not None:
        panel_items = panel_items[:max_panels]
    output_root = OUT_ROOT / output_namespace
    output_root.mkdir(parents=True, exist_ok=True)
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
        model_root = output_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(model_root / "events.jsonl", events)
        state_capture = StateCapture(model, layers)
        head_capture = MultiPositionHeadCapture(layers, head_count)
        state_capture.register()
        head_capture.register()
        for (family, template), panel_units in panel_items:
            panel_root = (
                model_root / family / f"template_{template}"
            )
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
                    and int(existing["unit_count"]) == len(panel_units)
                ):
                    panel_summaries.append(existing)
                    print(
                        f"[resume] {model_name}/{family}/t{template}",
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
                family=family,
                template=template,
                panel_units=panel_units,
                case_by_id=case_by_id,
                output_root=output_root,
                events=events,
                whole_keys=whole_keys,
                head_keys=head_keys,
                state_capture=state_capture,
                head_capture=head_capture,
            ))
        summary = {
            "schema_version": "phase1016_factorial_scan_model.v1",
            "phase": PHASE,
            "protocol_revision": PROTOCOL_REVISION,
            "protocol_digest": prereg["protocol_digest"],
            "model": model_name,
            "prompt_mode": prompt_mode,
            "output_namespace": output_namespace,
            "precision": "bf16",
            "model_info": {
                "n_layers": int(info.n_layers),
                "d_model": int(info.d_model),
                "head_count": head_count,
                "head_width": int(
                    layers[0].self_attn.o_proj.in_features
                    // head_count
                ),
                "model_class": info.model_class,
                "placement": placement,
            },
            "panel_count": len(panel_summaries),
            "event_count": len(events),
            "whole_event_count": len(whole_keys),
            "head_event_count": len(head_keys),
            "role_count": len(CAPTURE_ROLES),
            "singleton_forward_count": sum(
                row["singleton_forward_count"]
                for row in panel_summaries
            ),
            "identity_maximum": max(
                row["identity_maximum"]
                for row in panel_summaries
            ),
            "semantic_causal_prefix_maximum": max(
                row["semantic_causal_prefix_maximum"]
                for row in panel_summaries
            ),
            "factorial_candidate_all_hit_count": sum(
                row["factorial_candidate_all_hit_count"]
                for row in panel_summaries
            ),
            "factorial_full_vocab_all_hit_count": sum(
                row["factorial_full_vocab_all_hit_count"]
                for row in panel_summaries
            ),
            "panel_summaries": panel_summaries,
            "elapsed_seconds": time.time() - started,
        }
        write_json(model_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        if head_capture is not None:
            head_capture.close()
        if state_capture is not None:
            state_capture.close()
        if model is not None:
            release_model(model)
        del model, tokenizer, state_capture, head_capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--output-namespace", default="formal_scan")
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
