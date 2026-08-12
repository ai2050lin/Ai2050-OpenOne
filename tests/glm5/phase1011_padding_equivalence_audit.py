#!/usr/bin/env python3
"""Compare right-padded response capture with singleton forward passes."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1008_global_response_atlas_scan import StateCapture
from phase1011_native_semantic_protocol import (
    MODELS,
    OUT_ROOT,
    PHASE,
    read_jsonl,
    write_json,
)
from phase1011_native_semantic_scan import case_tensors, stage_case


STATES = ("base", "F", "Q", "FQ", "E", "O", "N", "S")
EPSILON = 1e-12


def fixed_width_repeated_tensors(
    case: dict[str, Any],
    *,
    batch_size: int,
    width: int,
    pad_token_id: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    length = len(case["input_ids"])
    if length > width:
        raise RuntimeError("fixed width shorter than case")
    input_ids = torch.full(
        (batch_size, width),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    values = torch.tensor(
        case["input_ids"], dtype=torch.long, device=device
    )
    input_ids[:, :length] = values[None, :]
    attention[:, :length] = 1
    lengths = torch.full(
        (batch_size,), length, dtype=torch.long, device=device
    )
    return input_ids, attention, lengths


def response_comparison(
    reference: dict[tuple[str, int], torch.Tensor],
    comparison: dict[tuple[str, int], torch.Tensor],
) -> dict[str, float]:
    hidden_relative_errors = []
    response_absolute_errors = []
    response_relative_errors = []
    response_cosines = []
    for key, reference_values in reference.items():
        comparison_values = comparison[key]
        differences = torch.linalg.vector_norm(
            reference_values - comparison_values, dim=-1
        )
        scale = torch.clamp(
            torch.linalg.vector_norm(comparison_values, dim=-1),
            min=EPSILON,
        )
        hidden_relative_errors.extend(
            (differences / scale).flatten().tolist()
        )
        reference_delta = (
            reference_values[1:] - reference_values[0]
        )
        comparison_delta = (
            comparison_values[1:] - comparison_values[0]
        )
        reference_norm = torch.linalg.vector_norm(
            reference_delta, dim=-1
        )
        comparison_norm = torch.linalg.vector_norm(
            comparison_delta, dim=-1
        )
        absolute = torch.abs(reference_norm - comparison_norm)
        response_absolute_errors.extend(absolute.flatten().tolist())
        valid = comparison_norm > EPSILON
        if torch.any(valid):
            response_relative_errors.extend(
                (
                    absolute[valid]
                    / torch.clamp(
                        comparison_norm[valid], min=EPSILON
                    )
                ).flatten().tolist()
            )
            cosine = torch.nn.functional.cosine_similarity(
                reference_delta[valid],
                comparison_delta[valid],
                dim=-1,
            )
            response_cosines.extend(cosine.flatten().tolist())
    return {
        "maximum_hidden_relative_error": float(
            max(hidden_relative_errors)
        ),
        "median_hidden_relative_error": float(
            np.median(hidden_relative_errors)
        ),
        "maximum_response_magnitude_error": float(
            max(response_absolute_errors)
        ),
        "median_response_magnitude_error": float(
            np.median(response_absolute_errors)
        ),
        "maximum_response_relative_error": float(
            max(response_relative_errors)
        ),
        "median_response_relative_error": float(
            np.median(response_relative_errors)
        ),
        "minimum_response_direction_cosine": float(
            min(response_cosines)
        ),
        "median_response_direction_cosine": float(
            np.median(response_cosines)
        ),
    }


def run_model(model_name: str) -> dict[str, Any]:
    cases = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    units = read_jsonl(
        OUT_ROOT / "protocol" / model_name / "units.jsonl"
    )
    unit = next(
        row for row in units
        if row["family"] == "comparison"
        and row["output_mode"] == "entity"
        and row["split"] == "discovery"
        and row["template"] == 0
        and row["name_pool"] == 0
        and row["world_index"] == 0
    )
    case_by_id = {case["record_id"]: case for case in cases}
    selected_cases = [
        case_by_id[unit["case_ids"][state]] for state in STATES
    ]
    staged = [stage_case(case, "prompt") for case in selected_cases]
    role_names = list(staged[0]["role_classes"])
    started = time.time()
    model = tokenizer = device = capture = None
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=True)
        layers = get_layers(model)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = (
                tokenizer.eos_token_id
                if tokenizer.eos_token_id is not None
                else 0
            )
        capture = StateCapture(model, layers)
        capture.register()
        positions = torch.tensor(
            [
                [
                    int(case["scan_role_positions"][role])
                    for role in role_names
                ]
                for case in staged
            ],
            dtype=torch.long,
            device=device,
        )
        input_ids, attention, lengths = case_tensors(
            staged, device, int(pad_token_id)
        )
        capture.begin(positions)
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            )
        capture.validate()
        batch = torch.arange(len(staged), device=output.logits.device)
        batched_logits = output.logits[
            batch, lengths - 1, :
        ].detach().float().cpu()
        batched_capture = {
            key: value.detach().float().cpu()
            for key, value in capture.captured.items()
        }
        del output, input_ids, attention, lengths, positions, batch
        capture.captured = {}

        singleton_logits = []
        singleton_capture: dict[
            tuple[str, int], list[torch.Tensor]
        ] = {key: [] for key in batched_capture}
        for case in staged:
            one_position = torch.tensor(
                [[
                    int(case["scan_role_positions"][role])
                    for role in role_names
                ]],
                dtype=torch.long,
                device=device,
            )
            one_ids, one_attention, one_lengths = case_tensors(
                [case], device, int(pad_token_id)
            )
            capture.begin(one_position)
            with torch.inference_mode():
                one_output = model(
                    input_ids=one_ids,
                    attention_mask=one_attention,
                    use_cache=False,
                    return_dict=True,
                )
            capture.validate()
            singleton_logits.append(
                one_output.logits[
                    0, int(one_lengths[0].item()) - 1, :
                ].detach().float().cpu()
            )
            for key, value in capture.captured.items():
                singleton_capture[key].append(
                    value[0].detach().float().cpu()
                )
            del (
                one_output,
                one_ids,
                one_attention,
                one_lengths,
                one_position,
            )
            capture.captured = {}
        singleton_logits_tensor = torch.stack(singleton_logits)
        singleton_capture_tensor = {
            key: torch.stack(values)
            for key, values in singleton_capture.items()
        }

        homogeneous_logits = []
        homogeneous_capture: dict[
            tuple[str, int], list[torch.Tensor]
        ] = {key: [] for key in batched_capture}
        fixed_width = max(len(case["input_ids"]) for case in staged)
        for case in staged:
            repeated_positions = torch.tensor(
                [
                    [
                        int(case["scan_role_positions"][role])
                        for role in role_names
                    ]
                    for _ in staged
                ],
                dtype=torch.long,
                device=device,
            )
            repeated_ids, repeated_attention, repeated_lengths = (
                fixed_width_repeated_tensors(
                    case,
                    batch_size=len(staged),
                    width=fixed_width,
                    pad_token_id=int(pad_token_id),
                    device=device,
                )
            )
            capture.begin(repeated_positions)
            with torch.inference_mode():
                repeated_output = model(
                    input_ids=repeated_ids,
                    attention_mask=repeated_attention,
                    use_cache=False,
                    return_dict=True,
                )
            capture.validate()
            homogeneous_logits.append(
                repeated_output.logits[
                    0,
                    int(repeated_lengths[0].item()) - 1,
                    :,
                ].detach().float().cpu()
            )
            for key, value in capture.captured.items():
                homogeneous_capture[key].append(
                    value[0].detach().float().cpu()
                )
            del (
                repeated_output,
                repeated_ids,
                repeated_attention,
                repeated_lengths,
                repeated_positions,
            )
            capture.captured = {}
        homogeneous_logits_tensor = torch.stack(homogeneous_logits)
        homogeneous_capture_tensor = {
            key: torch.stack(values)
            for key, values in homogeneous_capture.items()
        }

        candidate_agreement = []
        candidate_logit_errors = []
        homogeneous_candidate_agreement = []
        homogeneous_candidate_logit_errors = []
        for index, case in enumerate(staged):
            candidate_ids = torch.tensor(
                [
                    int(value)
                    for value in case["candidate_token_ids"].values()
                ],
                dtype=torch.long,
            )
            batch_panel = batched_logits[
                index
            ].index_select(0, candidate_ids)
            single_panel = singleton_logits_tensor[
                index
            ].index_select(0, candidate_ids)
            candidate_agreement.append(
                int(batch_panel.argmax().item())
                == int(single_panel.argmax().item())
            )
            candidate_logit_errors.append(float(
                torch.max(torch.abs(batch_panel - single_panel)).item()
            ))
            homogeneous_panel = homogeneous_logits_tensor[
                index
            ].index_select(0, candidate_ids)
            homogeneous_candidate_agreement.append(
                int(batch_panel.argmax().item())
                == int(homogeneous_panel.argmax().item())
            )
            homogeneous_candidate_logit_errors.append(float(
                torch.max(
                    torch.abs(batch_panel - homogeneous_panel)
                ).item()
            ))

        singleton_comparison = response_comparison(
            batched_capture, singleton_capture_tensor
        )
        homogeneous_comparison = response_comparison(
            batched_capture, homogeneous_capture_tensor
        )

        result = {
            "schema_version": "phase1011_padding_equivalence.v2",
            "phase": PHASE,
            "model": model_name,
            "unit_id": unit["unit_id"],
            "state_count": len(staged),
            "input_lengths": [
                len(case["input_ids"]) for case in staged
            ],
            "mixed_batch_vs_singleton": {
                "candidate_panel_prediction_agreement_rate": float(
                    np.mean(candidate_agreement)
                ),
                "maximum_candidate_logit_error": float(
                    max(candidate_logit_errors)
                ),
                "median_candidate_logit_error": float(
                    np.median(candidate_logit_errors)
                ),
                **singleton_comparison,
                "scope": (
                    "includes batch-shape and matrix-kernel drift"
                ),
            },
            "mixed_batch_vs_same_shape_homogeneous": {
                "candidate_panel_prediction_agreement_rate": float(
                    np.mean(homogeneous_candidate_agreement)
                ),
                "maximum_candidate_logit_error": float(
                    max(homogeneous_candidate_logit_errors)
                ),
                "median_candidate_logit_error": float(
                    np.median(homogeneous_candidate_logit_errors)
                ),
                **homogeneous_comparison,
                "scope": (
                    "holds batch size and padded width fixed; isolates "
                    "mixed-state batch composition"
                ),
            },
            "elapsed_seconds": time.time() - started,
            "interpretation": (
                "quantifies batching/padding instrumentation drift; it "
                "does not test a language mechanism"
            ),
        }
        output_root = OUT_ROOT / "padding_audit" / model_name
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        if capture is not None:
            capture.close()
        if model is not None:
            release_model(model)
        model = tokenizer = device = capture = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    run_model(args.model)


if __name__ == "__main__":
    main()
