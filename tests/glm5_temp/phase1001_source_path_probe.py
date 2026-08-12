#!/usr/bin/env python3
"""Temporary exact source-position reconstruction probe for Phase 1001."""
from __future__ import annotations

import gc
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_protocol import COLORS, MODEL
from phase1000_scpg_discovery import (
    batches_by_template,
    capture_residuals,
    source_patch_spec,
)
from phase1001_attention_head_discovery import (
    HEAD_COUNT,
    HEAD_DIM,
    RESULT_ROOT,
    SOURCE_DEPTH,
    TARGET_LAYERS,
    selected_phase1000_inputs,
)


def capture_physical(model, layers, device, rows, source_patch=None):
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    values = {}
    weights = {}
    head_outputs = {}
    counts = defaultdict(int)
    handles = []
    source_handle = None
    try:
        if source_patch is not None:
            from phase1000_scpg_discovery import register_source_patch

            source_handle, source_count = register_source_patch(
                layers, source_patch, full_width=None
            )
        for layer_number in TARGET_LAYERS:
            layer = layers[layer_number - 1]
            positions = torch.tensor(
                [row["role_positions"]["answer_boundary"] for row in rows],
                dtype=torch.long,
                device=device,
            )

            def make_v(number):
                def hook(module, args, output):
                    values[number] = (
                        output.detach()
                        .reshape(output.shape[0], output.shape[1], 8, HEAD_DIM)
                    )
                    counts[f"v/{number}"] += 1

                return hook

            def make_o(number, pos):
                def hook(module, args):
                    value = args[0]
                    batch_index = torch.arange(value.shape[0], device=value.device)
                    head_outputs[number] = (
                        value[batch_index, pos, :]
                        .reshape(value.shape[0], HEAD_COUNT, HEAD_DIM)
                        .detach()
                    )
                    counts[f"o/{number}"] += 1

                return hook

            def make_attn(number, pos):
                def hook(module, args, output):
                    batch_index = torch.arange(output[1].shape[0], device=device)
                    weights[number] = output[1][
                        batch_index, :, pos, :
                    ].detach()
                    counts[f"a/{number}"] += 1

                return hook

            handles.append(
                layer.self_attn.v_proj.register_forward_hook(
                    make_v(layer_number)
                )
            )
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(
                    make_o(layer_number, positions)
                )
            )
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_attn(layer_number, positions)
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        if source_patch is not None and source_count[0] != 1:
            raise RuntimeError("source patch count drift")
        del output
        return values, weights, head_outputs
    finally:
        for handle in reversed(handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def role_groups(row):
    query_slot = int(row["query_slot"])
    other_slot = 1 - query_slot
    named = {
        "record_queried_entity": row["role_positions"][
            f"slot{query_slot}_entity"
        ],
        "record_queried_value": row["role_positions"][
            f"slot{query_slot}_color"
        ],
        "record_alternative_entity": row["role_positions"][
            f"slot{other_slot}_entity"
        ],
        "record_alternative_value": row["role_positions"][
            f"slot{other_slot}_color"
        ],
        "query_name": row["role_positions"]["query_name"],
        "answer_boundary": row["role_positions"]["answer_boundary"],
    }
    used = set(named.values())
    result = {key: [value] for key, value in named.items()}
    result["other_context"] = [
        index for index in range(len(row["input_ids"])) if index not in used
    ]
    return result


def main():
    _, _, _, directional, _ = selected_phase1000_inputs("formal")
    batch = next(iter(batches_by_template(directional, 16)))
    source_cases = [row["source"] for row in batch]
    target_cases = [row["target"] for row in batch]
    protocol = json.loads(
        (
            ROOT
            / "tests/glm5/result/phase1000_factorial_binding_scpg/protocol/protocol.json"
        ).read_text(encoding="utf-8")
    )
    candidate_ids = {
        color: int(protocol["candidate_token_ids"][color]) for color in COLORS
    }
    frozen = json.loads(
        (RESULT_ROOT / "head_discovery/frozen_spec.json").read_text(
            encoding="utf-8"
        )
    )
    model = tokenizer = None
    try:
        model, tokenizer, device = load_model(
            MODEL, dtype=torch.bfloat16, use_8bit=False
        )
        layers = get_layers(model)
        _, source_residuals = capture_residuals(
            model, device, source_cases, (SOURCE_DEPTH,), candidate_ids
        )
        patch = source_patch_spec(
            SOURCE_DEPTH,
            target_cases,
            source_residuals[SOURCE_DEPTH],
            "joint",
        )
        target_v, target_a, target_z = capture_physical(
            model, layers, device, target_cases
        )
        do_v, do_a, do_z = capture_physical(
            model, layers, device, target_cases, source_patch=patch
        )
        rows = []
        for event_id in frozen["frozen_joint_event_ids"]:
            layer_number = int(event_id[1:3])
            head_index = int(event_id[-2:])
            kv_head = head_index // 4
            batch_index = torch.arange(len(batch), device=device)
            target_contrib = (
                target_a[layer_number][:, head_index, :, None]
                * target_v[layer_number][:, :, kv_head, :]
            )
            do_contrib = (
                do_a[layer_number][:, head_index, :, None]
                * do_v[layer_number][:, :, kv_head, :]
            )
            target_rebuilt = target_contrib.sum(dim=1)
            do_rebuilt = do_contrib.sum(dim=1)
            target_error = (
                target_rebuilt.float()
                - target_z[layer_number][:, head_index, :].float()
            ).abs()
            do_error = (
                do_rebuilt.float()
                - do_z[layer_number][:, head_index, :].float()
            ).abs()
            role_delta_norm = defaultdict(list)
            qk_norm = defaultdict(list)
            v_norm = defaultdict(list)
            interaction_norm = defaultdict(list)
            for index, item in enumerate(batch):
                groups = role_groups(item["target"])
                delta_a = (
                    do_a[layer_number][index, head_index].float()
                    - target_a[layer_number][index, head_index].float()
                )
                target_value = target_v[layer_number][
                    index, :, kv_head, :
                ].float()
                delta_value = (
                    do_v[layer_number][index, :, kv_head, :].float()
                    - target_value
                )
                for role, positions in groups.items():
                    role_delta_norm[role].append(
                        float(
                            (
                                do_contrib[index, positions].float().sum(0)
                                - target_contrib[index, positions].float().sum(0)
                            ).norm()
                        )
                    )
                    qk_norm[role].append(
                        float(
                            (
                                delta_a[positions, None]
                                * target_value[positions]
                            ).sum(0).norm()
                        )
                    )
                    v_norm[role].append(
                        float(
                            (
                                target_a[layer_number][
                                    index, head_index, positions, None
                                ].float()
                                * delta_value[positions]
                            ).sum(0).norm()
                        )
                    )
                    interaction_norm[role].append(
                        float(
                            (
                                delta_a[positions, None]
                                * delta_value[positions]
                            ).sum(0).norm()
                        )
                    )
            rows.append(
                {
                    "event_id": event_id,
                    "target_max_abs_reconstruction_error": float(
                        target_error.max()
                    ),
                    "do_max_abs_reconstruction_error": float(do_error.max()),
                    "mean_role_delta_norm": {
                        key: sum(values) / len(values)
                        for key, values in role_delta_norm.items()
                    },
                    "mean_qk_norm": {
                        key: sum(values) / len(values)
                        for key, values in qk_norm.items()
                    },
                    "mean_v_norm": {
                        key: sum(values) / len(values)
                        for key, values in v_norm.items()
                    },
                    "mean_interaction_norm": {
                        key: sum(values) / len(values)
                        for key, values in interaction_norm.items()
                    },
                }
            )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
