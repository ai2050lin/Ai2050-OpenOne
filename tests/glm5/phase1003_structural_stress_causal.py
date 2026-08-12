#!/usr/bin/env python3
"""Causal stress test for non-isomorphic Phase1003 language structures.

Only tasks that pass the independently frozen behavior gate are tested.
Each intervention condition is evaluated in a separate forward pass so
8-bit batch composition cannot masquerade as a causal effect.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import DynamicCache


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_crossparadigm_protocol import (
    MODELS,
    PHASE,
    digest,
    read_json,
    read_jsonl,
    stable_order,
    write_json,
    write_jsonl,
)
from phase1003_structural_stress_protocol import STRESS_ROOT, TASKS


CACHE_CONDITIONS = (
    "target_cache",
    "all_source_cache",
    "source_keys_only",
    "source_values_only",
)


def semantic_case(case: dict[str, Any]) -> dict[str, Any]:
    step = int(case["semantic_step"])
    result = dict(case)
    result["input_ids"] = (
        list(case["input_ids"])
        + list(case["answer_token_ids"][:step])
    )
    result["input_token_count"] = len(result["input_ids"])
    result["role_positions"] = dict(case["role_positions"])
    result["role_positions"]["answer_boundary"] = (
        result["input_token_count"] - 1
    )
    return result


def step_case(case: dict[str, Any], step: int) -> dict[str, Any]:
    result = dict(case)
    result["input_ids"] = (
        list(case["input_ids"])
        + list(case["answer_token_ids"][:step])
    )
    result["input_token_count"] = len(result["input_ids"])
    result["role_positions"] = dict(case["role_positions"])
    result["role_positions"]["answer_boundary"] = (
        result["input_token_count"] - 1
    )
    return result


def case_tensors(
    cases: list[dict[str, Any]], device
) -> tuple[torch.Tensor, torch.Tensor]:
    widths = {len(case["input_ids"]) for case in cases}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift: {widths}")
    input_ids = torch.tensor(
        [case["input_ids"] for case in cases],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def candidate_logits(
    logits: torch.Tensor, candidate_ids: dict[str, int]
) -> torch.Tensor:
    ids = torch.tensor(
        list(candidate_ids.values()),
        dtype=torch.long,
        device=logits.device,
    )
    return logits.index_select(-1, ids).float().detach()


def predictions(
    logits: torch.Tensor, candidate_ids: dict[str, int]
) -> list[str]:
    labels = list(candidate_ids)
    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    return [labels[int(index)] for index in indices]


def contrast_margin(
    logits: torch.Tensor,
    labels: list[str],
    donors: list[dict[str, Any]],
    targets: list[dict[str, Any]],
) -> torch.Tensor:
    label_index = {
        label: index for index, label in enumerate(labels)
    }
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    donor_index = torch.tensor(
        [label_index[case["gold"]] for case in donors],
        dtype=torch.long,
        device=logits.device,
    )
    target_index = torch.tensor(
        [label_index[case["gold"]] for case in targets],
        dtype=torch.long,
        device=logits.device,
    )
    return (
        logits[batch_index, donor_index]
        - logits[batch_index, target_index]
    )


def choose_donors(
    cases: list[dict[str, Any]],
    model_name: str,
    task: str,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usage: Counter[str] = Counter()
    donors = []
    for recipient_index, target in enumerate(cases):
        eligible = [
            candidate
            for candidate in cases
            if candidate["world_id"] != target["world_id"]
            and candidate["gold"] != target["gold"]
            and candidate["template"] == target["template"]
            and candidate["input_token_count"]
            == target["input_token_count"]
            and candidate["candidate_labels"]
            == target["candidate_labels"]
        ]
        if not eligible:
            raise RuntimeError(
                f"{model_name}/{task}/{split}: no donor for "
                f"{target['record_id']}"
            )
        eligible.sort(key=lambda candidate: (
            usage[candidate["record_id"]],
            stable_order(
                candidate["record_id"],
                f"stress-donor:{model_name}:{task}:{split}:"
                f"{recipient_index}:{target['record_id']}",
            ),
        ))
        donor = eligible[0]
        usage[donor["record_id"]] += 1
        donors.append(donor)
    audit = {
        "recipient_count": len(cases),
        "unique_donor_count": len(usage),
        "unique_donor_fraction": len(usage) / max(len(cases), 1),
        "maximum_donor_reuse": max(usage.values()),
        "minimum_used_donor_reuse": min(usage.values()),
        "all_cross_world": all(
            donor["world_id"] != target["world_id"]
            for target, donor in zip(cases, donors)
        ),
        "all_donor_answers_differ_from_target": all(
            donor["gold"] != target["gold"]
            for target, donor in zip(cases, donors)
        ),
        "assignment_digest": digest([
            {
                "recipient": target["record_id"],
                "donor": donor["record_id"],
            }
            for target, donor in zip(cases, donors)
        ]),
    }
    return donors, audit


def paired_batches(
    cases: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
):
    groups: dict[
        tuple[int, int],
        list[tuple[dict[str, Any], dict[str, Any]]],
    ] = defaultdict(list)
    for case, donor in zip(cases, donors):
        groups[(
            int(case["template"]),
            int(case["input_token_count"]),
        )].append((case, donor))
    for _, values in sorted(groups.items()):
        values.sort(key=lambda item: item[0]["record_id"])
        for start in range(0, len(values), batch_size):
            chunk = values[start : start + batch_size]
            yield (
                [item[0] for item in chunk],
                [item[1] for item in chunk],
            )


def capture_depth(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention = case_tensors(cases, device)
    captured: list[torch.Tensor] = []

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach())

    handle = layers[depth - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        if len(captured) != 1:
            raise RuntimeError(f"capture count {len(captured)}")
        return (
            candidate_logits(
                output.logits[:, -1, :],
                cases[0]["candidate_token_ids"],
            ),
            captured[0],
        )
    finally:
        handle.remove()
        del input_ids, attention


def patch_spec(
    depth: int,
    roles: list[str],
    donor_roles: Iterable[str],
    target_cases: list[dict[str, Any]],
    target_hidden: torch.Tensor,
    donor_cases: list[dict[str, Any]],
    donor_hidden: torch.Tensor,
) -> dict[str, Any]:
    donor_set = set(donor_roles)
    batch_index = torch.arange(
        len(target_cases), device=target_hidden.device
    )
    positions = {}
    vectors = {}
    for role in roles:
        target_positions = torch.tensor(
            [
                int(case["role_positions"][role])
                for case in target_cases
            ],
            dtype=torch.long,
            device=target_hidden.device,
        )
        donor_positions = torch.tensor(
            [
                int(case["role_positions"][role])
                for case in donor_cases
            ],
            dtype=torch.long,
            device=donor_hidden.device,
        )
        positions[role] = target_positions
        if role in donor_set:
            vectors[role] = donor_hidden[
                batch_index, donor_positions, :
            ]
        else:
            vectors[role] = target_hidden[
                batch_index, target_positions, :
            ]
    return {
        "depth": depth,
        "roles": roles,
        "positions": positions,
        "vectors": vectors,
    }


def register_patch(
    layers,
    patch: dict[str, Any] | None,
    full_width: int,
):
    if patch is None:
        return None, [0]
    count = [0]

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        if value.shape[1] != full_width:
            return output
        patched = value.clone()
        batch_index = torch.arange(
            patched.shape[0], device=patched.device
        )
        for role in patch["roles"]:
            patched[
                batch_index,
                patch["positions"][role],
                :,
            ] = patch["vectors"][role].to(
                device=patched.device, dtype=patched.dtype
            )
        count[0] += 1
        return (
            (patched,) + output[1:]
            if isinstance(output, tuple)
            else patched
        )

    handle = layers[patch["depth"] - 1].register_forward_hook(hook)
    return handle, count


def patched_forward(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    patch: dict[str, Any] | None,
) -> torch.Tensor:
    input_ids, attention = case_tensors(cases, device)
    handle, count = register_patch(
        layers, patch, input_ids.shape[1]
    )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        if patch is not None and count[0] != 1:
            raise RuntimeError(f"forward patch count {count[0]}")
        return candidate_logits(
            output.logits[:, -1, :],
            cases[0]["candidate_token_ids"],
        )
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention


def teacher_causal(
    model,
    layers,
    device,
    model_name: str,
    task: str,
    source_depth: int,
    cases: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_rows = []
    roles = list(cases[0]["anchor_roles"])
    conditions = [
        ("target_noop", []),
        ("full_source", roles),
    ] + [
        (
            f"leave_out_{role}",
            [candidate for candidate in roles if candidate != role],
        )
        for role in roles
    ]
    all_batches = list(paired_batches(cases, donors, batch_size))
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        target_cases = [semantic_case(case) for case in batch]
        donor_cases = [semantic_case(case) for case in donor_batch]
        clean_logits, target_hidden = capture_depth(
            model, layers, device, target_cases, source_depth
        )
        _, donor_hidden = capture_depth(
            model, layers, device, donor_cases, source_depth
        )
        condition_logits = {}
        for condition, donor_roles in conditions:
            patch = patch_spec(
                source_depth,
                roles,
                donor_roles,
                target_cases,
                target_hidden,
                donor_cases,
                donor_hidden,
            )
            condition_logits[condition] = patched_forward(
                model, layers, device, target_cases, patch
            )
        labels = list(target_cases[0]["candidate_token_ids"])
        clean_margin = contrast_margin(
            clean_logits, labels, donor_cases, target_cases
        )
        full_margin = contrast_margin(
            condition_logits["full_source"],
            labels,
            donor_cases,
            target_cases,
        )
        clean_predictions = predictions(
            clean_logits, target_cases[0]["candidate_token_ids"]
        )
        for condition, _ in conditions:
            logits = condition_logits[condition]
            condition_predictions = predictions(
                logits, target_cases[0]["candidate_token_ids"]
            )
            margins = contrast_margin(
                logits, labels, donor_cases, target_cases
            )
            for index, target in enumerate(target_cases):
                span = max(
                    abs(float(
                        full_margin[index] - clean_margin[index]
                    )),
                    1e-8,
                )
                result_rows.append({
                    "schema_version": (
                        "phase1003_structural_teacher_causal_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "task": task,
                    "split": target["split"],
                    "record_id": target["record_id"],
                    "donor_record_id": donor_cases[index]["record_id"],
                    "condition": condition,
                    "target_gold": target["gold"],
                    "donor_gold": donor_cases[index]["gold"],
                    "prediction": condition_predictions[index],
                    "target_prediction": (
                        condition_predictions[index] == target["gold"]
                    ),
                    "donor_prediction": (
                        condition_predictions[index]
                        == donor_cases[index]["gold"]
                    ),
                    "margin": float(margins[index]),
                    "normalized_full_transfer": float(
                        (
                            margins[index] - clean_margin[index]
                        ) / span
                    ),
                    "noop_prediction_agreement": (
                        condition_predictions[index]
                        == clean_predictions[index]
                        if condition == "target_noop"
                        else None
                    ),
                    "noop_candidate_max_abs_difference": (
                        float(torch.max(torch.abs(
                            logits[index] - clean_logits[index]
                        )))
                        if condition == "target_noop"
                        else None
                    ),
                })
        del (
            clean_logits,
            target_hidden,
            donor_hidden,
            condition_logits,
        )
        print(
            f"[{model_name}/{task}/{cases[0]['split']}/teacher] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )

    by_condition = {}
    for condition, _ in conditions:
        values = [
            row for row in result_rows
            if row["condition"] == condition
        ]
        item = {
            "n": len(values),
            "target_rate": float(np.mean([
                row["target_prediction"] for row in values
            ])),
            "donor_rate": float(np.mean([
                row["donor_prediction"] for row in values
            ])),
            "median_margin": float(np.median([
                row["margin"] for row in values
            ])),
            "median_normalized_full_transfer": float(np.median([
                row["normalized_full_transfer"] for row in values
            ])),
        }
        if condition == "target_noop":
            item["prediction_agreement"] = float(np.mean([
                row["noop_prediction_agreement"] for row in values
            ]))
            item["maximum_candidate_logit_difference"] = float(max(
                row["noop_candidate_max_abs_difference"]
                for row in values
            ))
        by_condition[condition] = item
    full = by_condition["full_source"]
    leaveout_effects = {
        role: {
            "donor_rate_drop_from_full": (
                full["donor_rate"]
                - by_condition[f"leave_out_{role}"]["donor_rate"]
            ),
            "median_margin_drop_from_full": (
                full["median_margin"]
                - by_condition[f"leave_out_{role}"]["median_margin"]
            ),
            "leaveout_donor_rate": by_condition[
                f"leave_out_{role}"
            ]["donor_rate"],
        }
        for role in roles
    }
    prereg = read_json(
        STRESS_ROOT / "preregistered_protocol.json"
    )
    thresholds = prereg["thresholds"]
    summary = {
        "roles": roles,
        "conditions": by_condition,
        "leaveout_effects": leaveout_effects,
        "full_anchor_teacher_pass": (
            by_condition["target_noop"]["prediction_agreement"]
            >= thresholds["noop_agreement"]
            and full["donor_rate"]
            >= thresholds["full_anchor_donor_rate"]
        ),
    }
    return result_rows, summary


def generate(
    model,
    tokenizer,
    layers,
    device,
    cases: list[dict[str, Any]],
    effective_eos: list[int],
    patch: dict[str, Any] | None,
) -> list[list[int]]:
    input_ids, attention = case_tensors(cases, device)
    prompt_width = input_ids.shape[1]
    handle, count = register_patch(layers, patch, prompt_width)
    try:
        answer_len = max(
            len(case["answer_token_ids"]) for case in cases
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=answer_len + 3,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        if patch is not None and count[0] != 1:
            raise RuntimeError(f"generation patch count {count[0]}")
        return [
            [int(value) for value in row]
            for row in generated.sequences[:, prompt_width:]
            .detach()
            .cpu()
            .tolist()
        ]
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention


def strip_eos(
    values: list[int], eos_set: set[int]
) -> list[int]:
    for index, token_id in enumerate(values):
        if token_id in eos_set:
            return values[:index]
    return values


def semantic_label(
    ids: list[int], case: dict[str, Any]
) -> str | None:
    step = int(case["semantic_step"])
    if len(ids) <= step:
        return None
    lookup = {
        int(token_id): label
        for label, token_id in case["candidate_token_ids"].items()
    }
    return lookup.get(ids[step])


def natural_confirmation(
    model,
    tokenizer,
    layers,
    device,
    model_name: str,
    task: str,
    source_depth: int,
    cases: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
    effective_eos: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_rows = []
    roles = list(cases[0]["anchor_roles"])
    eos_set = set(effective_eos)
    all_batches = list(paired_batches(cases, donors, batch_size))
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        _, target_hidden = capture_depth(
            model, layers, device, batch, source_depth
        )
        _, donor_hidden = capture_depth(
            model, layers, device, donor_batch, source_depth
        )
        clean = generate(
            model,
            tokenizer,
            layers,
            device,
            batch,
            effective_eos,
            None,
        )
        patches = {
            "target_noop": patch_spec(
                source_depth,
                roles,
                [],
                batch,
                target_hidden,
                donor_batch,
                donor_hidden,
            ),
            "full_source": patch_spec(
                source_depth,
                roles,
                roles,
                batch,
                target_hidden,
                donor_batch,
                donor_hidden,
            ),
        }
        generated = {
            condition: generate(
                model,
                tokenizer,
                layers,
                device,
                batch,
                effective_eos,
                patch,
            )
            for condition, patch in patches.items()
        }
        for condition, suffixes in generated.items():
            for index, target in enumerate(batch):
                clean_ids = strip_eos(clean[index], eos_set)
                ids = strip_eos(suffixes[index], eos_set)
                label = semantic_label(ids, target)
                result_rows.append({
                    "schema_version": (
                        "phase1003_structural_natural_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "task": task,
                    "split": "confirmation",
                    "record_id": target["record_id"],
                    "donor_record_id": (
                        donor_batch[index]["record_id"]
                    ),
                    "condition": condition,
                    "generated_ids": ids,
                    "generated_text": tokenizer.decode(
                        ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "semantic_prediction": label,
                    "target_semantic": label == target["gold"],
                    "donor_semantic": (
                        label == donor_batch[index]["gold"]
                    ),
                    "target_exact": (
                        ids == target["answer_token_ids"]
                    ),
                    "donor_exact": (
                        ids == donor_batch[index]["answer_token_ids"]
                    ),
                    "noop_sequence_agreement": (
                        ids == clean_ids
                        if condition == "target_noop"
                        else None
                    ),
                })
        del target_hidden, donor_hidden
        print(
            f"[{model_name}/{task}/confirmation/natural] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    condition_summary = {}
    for condition in ("target_noop", "full_source"):
        values = [
            row for row in result_rows
            if row["condition"] == condition
        ]
        item = {
            "n": len(values),
            "target_semantic_rate": float(np.mean([
                row["target_semantic"] for row in values
            ])),
            "donor_semantic_rate": float(np.mean([
                row["donor_semantic"] for row in values
            ])),
            "target_exact_rate": float(np.mean([
                row["target_exact"] for row in values
            ])),
            "donor_exact_rate": float(np.mean([
                row["donor_exact"] for row in values
            ])),
        }
        if condition == "target_noop":
            item["noop_sequence_agreement"] = float(np.mean([
                row["noop_sequence_agreement"] for row in values
            ]))
        condition_summary[condition] = item
    thresholds = read_json(
        STRESS_ROOT / "preregistered_protocol.json"
    )["thresholds"]
    full = condition_summary["full_source"]
    summary = {
        "conditions": condition_summary,
        "natural_confirmation_pass": (
            condition_summary["target_noop"][
                "noop_sequence_agreement"
            ] >= thresholds["noop_agreement"]
            and full["donor_semantic_rate"]
            >= thresholds["full_anchor_donor_rate"]
            and full["donor_exact_rate"]
            >= thresholds["full_anchor_donor_rate"]
        ),
    }
    return result_rows, summary


def build_cache(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    patch: dict[str, Any] | None,
):
    input_ids, attention = case_tensors(cases, device)
    handle, count = register_patch(
        layers, patch, input_ids.shape[1]
    )
    try:
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        if patch is not None and count[0] != 1:
            raise RuntimeError(f"cache patch count {count[0]}")
        return output.past_key_values
    finally:
        if handle is not None:
            handle.remove()
        del input_ids, attention


def clone_cache_mix(
    target_cache,
    source_cache,
    source_key_layers: set[int],
    source_value_layers: set[int],
    model_config,
) -> DynamicCache:
    if len(target_cache.layers) != len(source_cache.layers):
        raise RuntimeError("cache layer count drift")
    data = []
    for layer_index, (target_layer, source_layer) in enumerate(
        zip(target_cache.layers, source_cache.layers)
    ):
        keys = (
            source_layer.keys
            if layer_index in source_key_layers
            else target_layer.keys
        )
        values = (
            source_layer.values
            if layer_index in source_value_layers
            else target_layer.values
        )
        data.append((
            keys.detach().clone(),
            values.detach().clone(),
        ))
    return DynamicCache(data, config=model_config)


def continue_cache(
    model,
    device,
    current_ids: list[int],
    prefix_length: int,
    cache,
    candidate_ids: dict[str, int],
) -> torch.Tensor:
    input_ids = torch.tensor(
        [[token_id] for token_id in current_ids],
        dtype=torch.long,
        device=device,
    )
    attention = torch.ones(
        (len(current_ids), prefix_length + 1),
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    return candidate_logits(
        output.logits[:, -1, :], candidate_ids
    )


def cache_causal(
    model,
    layers,
    device,
    model_name: str,
    task: str,
    source_depth: int,
    cases: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layer_count = len(layers)
    all_layers = set(range(layer_count))
    result_rows = []
    roles = list(cases[0]["anchor_roles"])
    all_batches = list(paired_batches(cases, donors, batch_size))
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        semantic_step = int(batch[0]["semantic_step"])
        if any(
            int(case["semantic_step"]) != semantic_step
            for case in batch
        ):
            raise RuntimeError("semantic step drift")
        prefix_step = semantic_step - 1
        if prefix_step < 0:
            raise RuntimeError("semantic step lacks prefix token")
        prefix_cases = [
            step_case(case, prefix_step) for case in batch
        ]
        semantic_cases = [
            step_case(case, semantic_step) for case in batch
        ]
        _, donor_hidden = capture_depth(
            model, layers, device, donor_batch, source_depth
        )
        _, target_prefix_hidden = capture_depth(
            model, layers, device, prefix_cases, source_depth
        )
        _, target_semantic_hidden = capture_depth(
            model, layers, device, semantic_cases, source_depth
        )
        patch_prefix = patch_spec(
            source_depth,
            roles,
            roles,
            prefix_cases,
            target_prefix_hidden,
            donor_batch,
            donor_hidden,
        )
        patch_semantic = patch_spec(
            source_depth,
            roles,
            roles,
            semantic_cases,
            target_semantic_hidden,
            donor_batch,
            donor_hidden,
        )
        target_cache = build_cache(
            model, layers, device, prefix_cases, None
        )
        source_cache = build_cache(
            model, layers, device, prefix_cases, patch_prefix
        )
        current_ids = [
            int(case["answer_token_ids"][prefix_step])
            for case in batch
        ]
        prefix_length = len(prefix_cases[0]["input_ids"])
        candidate_ids = batch[0]["candidate_token_ids"]
        layer_mixes = {
            "target_cache": (set(), set()),
            "all_source_cache": (all_layers, all_layers),
            "source_keys_only": (all_layers, set()),
            "source_values_only": (set(), all_layers),
        }
        logits_by_condition = {
            condition: continue_cache(
                model,
                device,
                current_ids,
                prefix_length,
                clone_cache_mix(
                    target_cache,
                    source_cache,
                    key_layers,
                    value_layers,
                    model.config,
                ),
                candidate_ids,
            )
            for condition, (
                key_layers, value_layers
            ) in layer_mixes.items()
        }
        target_full = patched_forward(
            model, layers, device, semantic_cases, None
        )
        source_full = patched_forward(
            model,
            layers,
            device,
            semantic_cases,
            patch_semantic,
        )
        labels = list(candidate_ids)
        margins = {
            condition: contrast_margin(
                logits, labels, donor_batch, batch
            )
            for condition, logits in logits_by_condition.items()
        }
        target_margin = margins["target_cache"]
        source_margin = margins["all_source_cache"]
        condition_predictions = {
            condition: predictions(logits, candidate_ids)
            for condition, logits in logits_by_condition.items()
        }
        target_full_predictions = predictions(
            target_full, candidate_ids
        )
        source_full_predictions = predictions(
            source_full, candidate_ids
        )
        for condition in CACHE_CONDITIONS:
            logits = logits_by_condition[condition]
            for index, target in enumerate(batch):
                span = max(
                    abs(float(
                        source_margin[index] - target_margin[index]
                    )),
                    1e-8,
                )
                prediction = condition_predictions[condition][index]
                result_rows.append({
                    "schema_version": (
                        "phase1003_structural_cache_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "task": task,
                    "split": target["split"],
                    "record_id": target["record_id"],
                    "donor_record_id": (
                        donor_batch[index]["record_id"]
                    ),
                    "condition": condition,
                    "prediction": prediction,
                    "target_prediction": (
                        prediction == target["gold"]
                    ),
                    "donor_prediction": (
                        prediction == donor_batch[index]["gold"]
                    ),
                    "margin": float(margins[condition][index]),
                    "normalized_all_cache_transfer": float(
                        (
                            margins[condition][index]
                            - target_margin[index]
                        ) / span
                    ),
                    "full_prediction_agreement": (
                        prediction == target_full_predictions[index]
                        if condition == "target_cache"
                        else (
                            prediction
                            == source_full_predictions[index]
                            if condition == "all_source_cache"
                            else None
                        )
                    ),
                    "candidate_max_abs_difference_from_full": (
                        float(torch.max(torch.abs(
                            logits[index] - target_full[index]
                        )))
                        if condition == "target_cache"
                        else (
                            float(torch.max(torch.abs(
                                logits[index] - source_full[index]
                            )))
                            if condition == "all_source_cache"
                            else None
                        )
                    ),
                })
        del (
            donor_hidden,
            target_prefix_hidden,
            target_semantic_hidden,
            target_cache,
            source_cache,
            logits_by_condition,
            target_full,
            source_full,
        )
        print(
            f"[{model_name}/{task}/{cases[0]['split']}/cache] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    condition_summary = {}
    for condition in CACHE_CONDITIONS:
        values = [
            row for row in result_rows
            if row["condition"] == condition
        ]
        item = {
            "n": len(values),
            "target_rate": float(np.mean([
                row["target_prediction"] for row in values
            ])),
            "donor_rate": float(np.mean([
                row["donor_prediction"] for row in values
            ])),
            "median_margin": float(np.median([
                row["margin"] for row in values
            ])),
            "median_normalized_all_cache_transfer": float(
                np.median([
                    row["normalized_all_cache_transfer"]
                    for row in values
                ])
            ),
        }
        if condition in ("target_cache", "all_source_cache"):
            item["full_prediction_agreement"] = float(np.mean([
                row["full_prediction_agreement"] for row in values
            ]))
            item["maximum_candidate_logit_difference"] = float(max(
                row["candidate_max_abs_difference_from_full"]
                for row in values
            ))
        condition_summary[condition] = item
    thresholds = read_json(
        STRESS_ROOT / "preregistered_protocol.json"
    )["thresholds"]
    summary = {
        "conditions": condition_summary,
        "source_parent_gate": (
            condition_summary["all_source_cache"]["donor_rate"]
            >= thresholds["full_anchor_donor_rate"]
        ),
        "value_transport_gate": (
            condition_summary["source_values_only"]["donor_rate"]
            >= thresholds["cache_value_donor_rate"]
        ),
        "target_cache_instrument": (
            condition_summary["target_cache"][
                "full_prediction_agreement"
            ] >= thresholds["noop_agreement"]
        ),
        "source_cache_instrument": (
            condition_summary["all_source_cache"][
                "full_prediction_agreement"
            ] >= thresholds["noop_agreement"]
        ),
        "key_value_difference_is_descriptive": True,
    }
    summary["cache_causal_pass"] = (
        summary["source_parent_gate"]
        and summary["value_transport_gate"]
        and summary["target_cache_instrument"]
        and summary["source_cache_instrument"]
    )
    return result_rows, summary


def run_task(
    model,
    tokenizer,
    layers,
    device,
    model_name: str,
    task: str,
    source_depth: int,
    batch_size: int,
    effective_eos: list[int],
) -> dict[str, Any]:
    all_cases = read_jsonl(
        STRESS_ROOT / "protocol" / model_name / "cases.jsonl"
    )
    task_cases = [
        case for case in all_cases if case["task"] == task
    ]
    root = STRESS_ROOT / "causal" / model_name / task
    donor_audits = {}
    teacher_rows = []
    cache_rows = []
    teacher_summaries = {}
    cache_summaries = {}
    confirmation_cases = []
    confirmation_donors = []
    for split in ("discovery", "confirmation"):
        cases = [
            case for case in task_cases if case["split"] == split
        ]
        donors, donor_audit = choose_donors(
            cases, model_name, task, split
        )
        donor_audits[split] = donor_audit
        rows, summary = teacher_causal(
            model,
            layers,
            device,
            model_name,
            task,
            source_depth,
            cases,
            donors,
            batch_size,
        )
        teacher_rows.extend(rows)
        teacher_summaries[split] = summary
        rows, summary = cache_causal(
            model,
            layers,
            device,
            model_name,
            task,
            source_depth,
            cases,
            donors,
            batch_size,
        )
        cache_rows.extend(rows)
        cache_summaries[split] = summary
        if split == "confirmation":
            confirmation_cases = cases
            confirmation_donors = donors
    natural_rows, natural_summary = natural_confirmation(
        model,
        tokenizer,
        layers,
        device,
        model_name,
        task,
        source_depth,
        confirmation_cases,
        confirmation_donors,
        batch_size,
        effective_eos,
    )
    role_effects = {
        split: teacher_summaries[split]["leaveout_effects"]
        for split in ("discovery", "confirmation")
    }
    repeated_positive_leaveout_roles = [
        role
        for role in confirmation_cases[0]["anchor_roles"]
        if all(
            role_effects[split][role][
                "donor_rate_drop_from_full"
            ] > 0
            for split in ("discovery", "confirmation")
        )
    ]
    summary = {
        "schema_version": (
            "phase1003_structural_stress_causal_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "task": task,
        "status": "complete",
        "source_depth": source_depth,
        "roles": confirmation_cases[0]["anchor_roles"],
        "case_count": len(task_cases),
        "donor_audits": donor_audits,
        "teacher": teacher_summaries,
        "natural_confirmation": natural_summary,
        "cache": cache_summaries,
        "repeated_positive_leaveout_roles_descriptive": (
            repeated_positive_leaveout_roles
        ),
        "structural_transport_pass": (
            all(
                teacher_summaries[split][
                    "full_anchor_teacher_pass"
                ]
                for split in ("discovery", "confirmation")
            )
            and natural_summary["natural_confirmation_pass"]
            and all(
                cache_summaries[split]["cache_causal_pass"]
                for split in ("discovery", "confirmation")
            )
        ),
        "claim_boundary": (
            "The task-specific role set is an experimental coordinate "
            "system, not a discovered language ontology. Positive "
            "leave-one-out effects are descriptive because no magnitude "
            "threshold was preregistered."
        ),
    }
    write_jsonl(root / "teacher_rows.jsonl", teacher_rows)
    write_jsonl(root / "natural_rows.jsonl", natural_rows)
    write_jsonl(root / "cache_rows.jsonl", cache_rows)
    write_json(root / "summary.json", summary)
    return summary


def run_model(
    model_name: str, batch_size: int
) -> dict[str, Any]:
    behavior = read_json(
        STRESS_ROOT / "behavior" / model_name / "summary.json"
    )
    tasks = [
        task
        for task in TASKS
        if behavior["task_gates"][task]
    ]
    prereg = read_json(
        STRESS_ROOT / "preregistered_protocol.json"
    )
    source_depth = int(prereg["source_depths"][model_name])
    model = tokenizer = None
    summaries = {}
    started = time.time()
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        for task in tasks:
            summaries[task] = run_task(
                model,
                tokenizer,
                layers,
                device,
                model_name,
                task,
                source_depth,
                batch_size,
                effective_eos,
            )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": (
            "phase1003_structural_stress_causal_model.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "behavior_passing_tasks": tasks,
        "tasks": summaries,
        "pass_count": sum(
            summary["structural_transport_pass"]
            for summary in summaries.values()
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        STRESS_ROOT / "causal" / model_name / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            STRESS_ROOT / "causal" / model_name / "summary.json"
        )
        if path.exists():
            summaries[model_name] = read_json(path)
    cross_task = {}
    for task in TASKS:
        available = [
            summary["tasks"][task]
            for summary in summaries.values()
            if task in summary["tasks"]
        ]
        cross_task[task] = {
            "behavior_qualified_model_count": len(available),
            "structural_transport_pass_count": sum(
                item["structural_transport_pass"]
                for item in available
            ),
            "models": {
                item["model"]: {
                    "pass": item["structural_transport_pass"],
                    "teacher_discovery_full_donor_rate": (
                        item["teacher"]["discovery"]["conditions"][
                            "full_source"
                        ]["donor_rate"]
                    ),
                    "teacher_confirmation_full_donor_rate": (
                        item["teacher"]["confirmation"]["conditions"][
                            "full_source"
                        ]["donor_rate"]
                    ),
                    "natural_full_donor_rate": (
                        item["natural_confirmation"]["conditions"][
                            "full_source"
                        ]["donor_semantic_rate"]
                    ),
                    "cache_confirmation_key_donor_rate": (
                        item["cache"]["confirmation"]["conditions"][
                            "source_keys_only"
                        ]["donor_rate"]
                    ),
                    "cache_confirmation_value_donor_rate": (
                        item["cache"]["confirmation"]["conditions"][
                            "source_values_only"
                        ]["donor_rate"]
                    ),
                }
                for item in available
            },
        }
    minimum = read_json(
        STRESS_ROOT / "preregistered_protocol.json"
    )["thresholds"]["cross_model_minimum"]
    payload = {
        "schema_version": (
            "phase1003_structural_stress_causal_aggregate.v1"
        ),
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_task": cross_task,
        "cross_model_structural_gates": {
            task: (
                item["structural_transport_pass_count"] >= minimum
            )
            for task, item in cross_task.items()
        },
    }
    write_json(STRESS_ROOT / "causal" / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        run_model(args.model, args.batch_size)
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
