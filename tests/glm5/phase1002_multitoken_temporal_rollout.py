#!/usr/bin/env python3
"""Trace and intervene on the frozen topology during autoregressive rollout.

The source intervention is applied once while the prompt cache is built. Later
effects therefore have to travel through the model's own cached autoregressive
state. Receiver restoration is aligned to the current generated-token boundary.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1002_multitoken_frozen_topology import (
    event_from_id,
    read_json,
    read_jsonl,
)
from phase1002_multitoken_protocol import (
    COLORS,
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)


PHASE = 1002
ROLLOUT_PAIRS_PER_STRATUM = 2
CACHE_AUDIT_PAIRS_PER_STRATUM = 1


def selected_directional_rows(
    model_name: str,
    split: str,
    pairs_per_stratum: int,
) -> list[dict[str, Any]]:
    protocol_root = OUT_ROOT / "protocol" / model_name
    cases = {
        row["record_id"]: row
        for row in read_jsonl(protocol_root / "cases.jsonl")
    }
    pairs = read_jsonl(
        protocol_root / f"{split}_selected_pairs.jsonl"
    )
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for pair in pairs:
        key = (
            int(pair["template"]),
            int(pair["display_order"]),
            int(pair["value_swap"]),
            int(pair["query_role"]),
        )
        groups[key].append(pair)
    chosen = []
    for key in sorted(groups):
        chosen.extend(
            sorted(groups[key], key=lambda row: row["pair_id"])[
                :pairs_per_stratum
            ]
        )
    rows = []
    for pair in chosen:
        arm0 = cases[pair["arm0_record_id"]]
        arm1 = cases[pair["arm1_record_id"]]
        for direction, source, target in (
            ("arm0_to_arm1", arm0, arm1),
            ("arm1_to_arm0", arm1, arm0),
        ):
            rows.append({
                "split": split,
                "pair_id": pair["pair_id"],
                "direction": direction,
                "template": int(pair["template"]),
                "source": source,
                "target": target,
            })
    expected = 32 * pairs_per_stratum * 2
    if len(rows) != expected:
        raise RuntimeError(
            f"{model_name}/{split}: rollout size {len(rows)} != {expected}"
        )
    return sorted(
        rows,
        key=lambda row: (
            row["template"], row["pair_id"], row["direction"]
        ),
    )


def batches(rows: list[dict[str, Any]], batch_size: int):
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["template"]),
            int(row["target"]["input_token_count"]),
        )
        groups[key].append(row)
    for key in sorted(groups):
        values = groups[key]
        for start in range(0, len(values), batch_size):
            yield values[start:start + batch_size]


def step_case(case: dict[str, Any], step: int) -> dict[str, Any]:
    result = dict(case)
    result["input_ids"] = (
        list(case["input_ids"]) + list(case["answer_token_ids"][:step])
    )
    result["input_token_count"] = len(result["input_ids"])
    result["role_positions"] = dict(case["role_positions"])
    result["role_positions"]["answer_boundary"] = (
        result["input_token_count"] - 1
    )
    return result


def strip_eos(
    values: list[int], eos_set: set[int]
) -> tuple[list[int], int | None]:
    for index, value in enumerate(values):
        if value in eos_set:
            return values[:index], index
    return values, None


def register_dynamic_receivers(
    layers,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    donor_by_step: list[dict[str, torch.Tensor]],
    active_steps: set[int],
    device,
):
    handles = []
    counters: dict[str, list[int]] = {}
    for event in events:
        event_id = event["event_id"]
        counter = [0]
        counters[event_id] = counter

        def make_hook(
            frozen_event: dict[str, Any],
            frozen_id: str,
            frozen_counter: list[int],
        ):
            def hook(module, args, output):
                step = frozen_counter[0]
                frozen_counter[0] += 1
                if (
                    step not in active_steps
                    or step >= len(donor_by_step)
                ):
                    return output
                value = output[0] if isinstance(output, tuple) else output
                if frozen_event["role"] == "query_name":
                    if step != 0:
                        return output
                    positions = torch.tensor(
                        [
                            row["target"]["role_positions"]["query_name"]
                            for row in rows
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    positions = torch.full(
                        (len(rows),),
                        value.shape[1] - 1,
                        dtype=torch.long,
                        device=device,
                    )
                return scpg.replace_positions(
                    output,
                    {"receiver": positions},
                    {"receiver": donor_by_step[step][frozen_id]},
                )

            return hook

        handles.append(
            scpg.component_module(layers, event).register_forward_hook(
                make_hook(event, event_id, counter)
            )
        )
    return handles, counters


def cached_generate(
    model,
    layers,
    tokenizer,
    device,
    rows: list[dict[str, Any]],
    source_patch: dict[str, Any],
    events: list[dict[str, Any]],
    donor_by_step: list[dict[str, torch.Tensor]],
    active_steps: set[int],
    effective_eos: list[int],
    budget: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    target_cases = [row["target"] for row in rows]
    input_ids, attention = scpg.case_tensors(target_cases, device)
    full_width = input_ids.shape[1]
    source_handle = None
    receiver_handles = []
    try:
        source_handle, source_count = scpg.register_source_patch(
            layers, source_patch, full_width=full_width
        )
        receiver_handles, counters = register_dynamic_receivers(
            layers,
            rows,
            events,
            donor_by_step,
            active_steps,
            device,
        )
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                max_new_tokens=budget,
                eos_token_id=effective_eos,
                pad_token_id=int(tokenizer.pad_token_id),
                return_dict_in_generate=True,
            )
        if source_count[0] != 1:
            raise RuntimeError(
                f"cached source hook count drift: {source_count[0]}"
            )
        suffixes = generated.sequences[:, full_width:].detach().cpu().tolist()
        eos_set = set(effective_eos)
        result = []
        for suffix in suffixes:
            suffix = [int(value) for value in suffix]
            before_eos, eos_position = strip_eos(suffix, eos_set)
            result.append({
                "suffix_ids": suffix,
                "before_eos_ids": before_eos,
                "eos_position": eos_position,
                "text": tokenizer.decode(
                    before_eos,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
            })
        return result, {
            event_id: counter[0]
            for event_id, counter in counters.items()
        }
    finally:
        for handle in reversed(receiver_handles):
            handle.remove()
        if source_handle is not None:
            source_handle.remove()


def full_recompute_source_do(
    model,
    layers,
    device,
    rows: list[dict[str, Any]],
    source_vectors: dict[str, torch.Tensor],
    source_depth: int,
    effective_eos: list[int],
    budget: int,
) -> list[list[int]]:
    current = [
        list(row["target"]["input_ids"])
        for row in rows
    ]
    eos_set = set(effective_eos)
    suffixes = [[] for _ in rows]
    for _ in range(budget):
        current_cases = []
        for row, ids in zip(rows, current):
            case = dict(row["target"])
            case["input_ids"] = ids
            case["input_token_count"] = len(ids)
            case["role_positions"] = dict(case["role_positions"])
            case["role_positions"]["answer_boundary"] = len(ids) - 1
            current_cases.append(case)
        patch = scpg.source_patch_spec(
            source_depth,
            current_cases,
            source_vectors,
            "joint",
        )
        input_ids, attention = scpg.case_tensors(current_cases, device)
        handle = None
        try:
            handle, count = scpg.register_source_patch(
                layers, patch, full_width=None
            )
            with torch.inference_mode():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention,
                    use_cache=False,
                    return_dict=True,
                )
            next_ids = (
                torch.argmax(output.logits[:, -1, :], dim=-1)
                .detach()
                .cpu()
                .tolist()
            )
            if count[0] != 1:
                raise RuntimeError(
                    f"full recompute hook count drift: {count[0]}"
                )
        finally:
            if handle is not None:
                handle.remove()
        for index, token_id in enumerate(next_ids):
            token_id = int(token_id)
            current[index].append(token_id)
            suffixes[index].append(token_id)
        if all(
            any(token_id in eos_set for token_id in suffix)
            for suffix in suffixes
        ):
            break
    return suffixes


def temporal_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["event_id"], int(row["step"]))].append(row)
    result = {}
    for (event_id, step), values in sorted(groups.items()):
        result.setdefault(event_id, {})[str(step)] = {
            "step_role": values[0]["step_role"],
            "n": len(values),
            "median_relative_response": float(np.median([
                row["relative_response"] for row in values
            ])),
            "mean_relative_response": float(np.mean([
                row["relative_response"] for row in values
            ])),
            "median_response_norm": float(np.median([
                row["response_norm"] for row in values
            ])),
        }
    return result


def rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["condition"])].append(row)
    result = {}
    for (split, condition), values in sorted(groups.items()):
        result.setdefault(split, {})[condition] = {
            "n": len(values),
            "source_exact_rate": float(np.mean([
                row["source_exact"] for row in values
            ])),
            "target_exact_rate": float(np.mean([
                row["target_exact"] for row in values
            ])),
            "source_semantic_rate": float(np.mean([
                row["semantic_prediction"] == row["source_gold"]
                for row in values
            ])),
            "target_semantic_rate": float(np.mean([
                row["semantic_prediction"] == row["target_gold"]
                for row in values
            ])),
            "eos_rate": float(np.mean([
                row["eos_position"] is not None for row in values
            ])),
        }
    return result


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    topology_summary = read_json(
        OUT_ROOT / "frozen_topology" / model_name / "summary.json"
    )
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    frozen = prereg["frozen_phase1001_topology"][model_name]
    variant_name = (
        "preregistered_k2_secondary"
        if model_name == "qwen3"
        else "primary"
    )
    events = [
        event_from_id(event_id)
        for event_id in frozen["variants"][variant_name]
    ]
    source_depth = int(frozen["source_depth"])
    thresholds = prereg["primary_thresholds"]
    receiver_gate_checks = {}
    for split in ("discovery", "confirmation"):
        metrics = topology_summary["split_summary"][split][variant_name]
        receiver_gate_checks[split] = {
            "source_do": (
                metrics["source_do_source_rate"]
                >= thresholds["source_do_semantic_flip_rate"]
            ),
            "restore": (
                metrics["restore_target_rate"]
                >= thresholds["frozen_topology_semantic_restore_rate"]
            ),
            "mediation": (
                metrics["median_mediation_fraction"]
                >= thresholds["semantic_mediation_median"]
            ),
        }
    receiver_gate_pass = all(
        all(checks.values()) for checks in receiver_gate_checks.values()
    )
    model = tokenizer = None
    started = time.time()
    state_rows = []
    generation_rows = []
    cache_rows = []
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        for split in ("discovery", "confirmation"):
            split_rows = selected_directional_rows(
                model_name,
                split,
                ROLLOUT_PAIRS_PER_STRATUM,
            )
            split_batches = list(batches(split_rows, batch_size))
            for batch_number, batch in enumerate(split_batches, 1):
                source_cases = [row["source"] for row in batch]
                target_cases = [row["target"] for row in batch]
                candidate_ids = target_cases[0]["candidate_token_ids"]
                answer_len = len(target_cases[0]["answer_token_ids"])
                semantic_step = int(target_cases[0]["semantic_step"])
                _, source_residuals = scpg.capture_residuals(
                    model,
                    device,
                    source_cases,
                    (source_depth,),
                    candidate_ids,
                )
                source_vectors = source_residuals[source_depth]
                source_patch = scpg.source_patch_spec(
                    source_depth,
                    target_cases,
                    source_vectors,
                    "joint",
                )

                clean_donors = []
                for step in range(answer_len + 1):
                    target_step_cases = [
                        step_case(row["target"], step) for row in batch
                    ]
                    target_logits, target_components = (
                        scpg.capture_components(
                            model,
                            layers,
                            device,
                            target_step_cases,
                            events,
                            candidate_ids,
                        )
                    )
                    step_patch = scpg.source_patch_spec(
                        source_depth,
                        target_step_cases,
                        source_vectors,
                        "joint",
                    )
                    do_logits, do_components = scpg.capture_components(
                        model,
                        layers,
                        device,
                        target_step_cases,
                        events,
                        candidate_ids,
                        source_patch=step_patch,
                    )
                    clean_donors.append(target_components)
                    for event in events:
                        event_id = event["event_id"]
                        delta = (
                            do_components[event_id]
                            - target_components[event_id]
                        ).float()
                        response_norm = torch.linalg.vector_norm(
                            delta, dim=-1
                        )
                        base_norm = torch.linalg.vector_norm(
                            target_components[event_id].float(), dim=-1
                        )
                        for index, item in enumerate(batch):
                            state_rows.append({
                                "schema_version": (
                                    "phase1002_temporal_state_row.v1"
                                ),
                                "phase": PHASE,
                                "model": model_name,
                                "split": split,
                                "pair_id": item["pair_id"],
                                "direction": item["direction"],
                                "template": item["template"],
                                "event_id": event_id,
                                "step": step,
                                "step_role": (
                                    target_cases[0][
                                        "answer_step_roles"
                                    ][step]
                                    if step < answer_len
                                    else "eos"
                                ),
                                "response_norm": float(
                                    response_norm[index]
                                ),
                                "base_norm": float(base_norm[index]),
                                "relative_response": float(
                                    response_norm[index]
                                    / max(float(base_norm[index]), 1e-8)
                                ),
                                "target_color_candidates": [
                                    float(value)
                                    for value in target_logits[index]
                                ],
                                "source_do_color_candidates": [
                                    float(value)
                                    for value in do_logits[index]
                                ],
                            })
                    del (
                        target_logits,
                        do_logits,
                        do_components,
                    )

                conditions = {
                    "source_do": set(),
                    "restore_all_steps": set(range(answer_len + 1)),
                    "restore_semantic_step_only": {semantic_step},
                    "restore_nonsemantic_steps": (
                        set(range(answer_len + 1)) - {semantic_step}
                    ),
                }
                cached_by_condition = {}
                for condition, active_steps in conditions.items():
                    generated, hook_counts = cached_generate(
                        model,
                        layers,
                        tokenizer,
                        device,
                        batch,
                        source_patch,
                        events,
                        clean_donors,
                        active_steps,
                        effective_eos,
                        answer_len + 2,
                    )
                    cached_by_condition[condition] = generated
                    for index, (item, output) in enumerate(
                        zip(batch, generated)
                    ):
                        before = output["before_eos_ids"]
                        target_answer = item["target"]["answer_token_ids"]
                        source_answer = item["source"]["answer_token_ids"]
                        predicted = None
                        if len(before) > semantic_step:
                            token_id = before[semantic_step]
                            for color in COLORS:
                                if (
                                    token_id
                                    == item["target"][
                                        "candidate_token_ids"
                                    ][color]
                                ):
                                    predicted = color
                                    break
                        generation_rows.append({
                            "schema_version": (
                                "phase1002_temporal_rollout_row.v1"
                            ),
                            "phase": PHASE,
                            "model": model_name,
                            "split": split,
                            "pair_id": item["pair_id"],
                            "direction": item["direction"],
                            "template": item["template"],
                            "condition": condition,
                            "event_ids": [
                                event["event_id"] for event in events
                            ],
                            "source_gold": item["source"]["gold"],
                            "target_gold": item["target"]["gold"],
                            "semantic_prediction": predicted,
                            "source_exact": before == source_answer,
                            "target_exact": before == target_answer,
                            "eos_position": output["eos_position"],
                            "before_eos_ids": before,
                            "generated_text": output["text"],
                            "receiver_hook_counts": hook_counts,
                        })

                if batch_number % ROLLOUT_PAIRS_PER_STRATUM == 1:
                    audit_rows = batch
                    full_suffixes = full_recompute_source_do(
                        model,
                        layers,
                        device,
                        audit_rows,
                        source_vectors,
                        source_depth,
                        effective_eos,
                        answer_len + 2,
                    )
                    cached_outputs = cached_by_condition["source_do"]
                    eos_set = set(effective_eos)
                    for item, full_suffix, cached_output in zip(
                        audit_rows, full_suffixes, cached_outputs
                    ):
                        full_before, _ = strip_eos(
                            full_suffix, eos_set
                        )
                        cache_rows.append({
                            "schema_version": (
                                "phase1002_cache_recompute_audit_row.v1"
                            ),
                            "phase": PHASE,
                            "model": model_name,
                            "split": split,
                            "pair_id": item["pair_id"],
                            "direction": item["direction"],
                            "cached_before_eos_ids": (
                                cached_output["before_eos_ids"]
                            ),
                            "full_recompute_before_eos_ids": full_before,
                            "token_sequence_agreement": (
                                cached_output["before_eos_ids"]
                                == full_before
                            ),
                        })
                del source_residuals, clean_donors
                print(
                    f"[{model_name}/{split}] "
                    f"{batch_number}/{len(split_batches)}",
                    flush=True,
                )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    model_root = OUT_ROOT / "temporal_rollout" / model_name
    write_jsonl(model_root / "state_rows.jsonl", state_rows)
    write_jsonl(model_root / "generation_rows.jsonl", generation_rows)
    write_jsonl(model_root / "cache_audit_rows.jsonl", cache_rows)
    generation_summary = rollout_summary(generation_rows)
    cache_agreement = float(np.mean([
        row["token_sequence_agreement"] for row in cache_rows
    ]))
    wrong_step_rates = [
        generation_summary[split]["restore_nonsemantic_steps"][
            "target_semantic_rate"
        ]
        for split in ("discovery", "confirmation")
    ]
    summary = {
        "schema_version": "phase1002_temporal_rollout_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "source_applied_once_at_prompt_cache_build": True,
        "topology_variant": variant_name,
        "receiver_gate_checks": receiver_gate_checks,
        "receiver_gate_pass_before_rollout": receiver_gate_pass,
        "frozen_event_ids": [
            event["event_id"] for event in events
        ],
        "rollout_directions_per_split": (
            32 * ROLLOUT_PAIRS_PER_STRATUM * 2
        ),
        "generation_summary": generation_summary,
        "temporal_state_summary": temporal_summary(state_rows),
        "cache_full_recompute_audit": {
            "n": len(cache_rows),
            "token_sequence_agreement": cache_agreement,
        },
        "thresholds": {
            "semantic_restore_rate": (
                prereg["primary_thresholds"][
                    "frozen_topology_semantic_restore_rate"
                ]
            ),
            "wrong_step_restore_target_max": (
                prereg["primary_thresholds"][
                    "wrong_step_restore_target_max"
                ]
            ),
            "cache_full_recompute_token_agreement": (
                prereg["primary_thresholds"][
                    "cache_full_recompute_token_agreement"
                ]
            ),
        },
        "checks": {
            "semantic_step_restore": all(
                generation_summary[split][
                    "restore_semantic_step_only"
                ]["target_semantic_rate"]
                >= prereg["primary_thresholds"][
                    "frozen_topology_semantic_restore_rate"
                ]
                for split in ("discovery", "confirmation")
            ),
            "wrong_step_control": all(
                rate
                <= prereg["primary_thresholds"][
                    "wrong_step_restore_target_max"
                ]
                for rate in wrong_step_rates
            ),
            "cache_recompute": (
                cache_agreement
                >= prereg["primary_thresholds"][
                    "cache_full_recompute_token_agreement"
                ]
            ),
        },
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "This establishes a cache-carried causal effect if it passes. "
            "It does not by itself identify whether keys, values, or both "
            "carry that effect. A rollout from a receiver variant that "
            "failed its semantic-step gate is diagnostic only."
        ),
    }
    summary["temporal_rollout_pass"] = (
        receiver_gate_pass and all(summary["checks"].values())
    )
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT / "temporal_rollout" / model_name / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "temporal_rollout"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": "phase1002_temporal_rollout_cross_model.v1",
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["temporal_rollout_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["temporal_rollout_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(OUT_ROOT / "temporal_rollout" / "summary.json", payload)
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
