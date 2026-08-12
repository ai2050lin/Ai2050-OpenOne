#!/usr/bin/env python3
"""Test compositional multi-token translation with cached K/V swaps."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers
from phase1023_fp16_utils import (
    load_fp16,
    quantization_audit,
    release_fp16,
)
import phase1052_full_vocab_kv_bridge_scan as bridge
import phase1054_joint_kv_rollout_scan as eos_tools
import phase1057_translation_trajectory_scan as trajectory
import phase1058_multitoken_translation_protocol as protocol


PAIR_BATCH_SIZE = bridge.PAIR_BATCH_SIZE


def evenly_spaced(rows: list[Any], count: int) -> list[Any]:
    if len(rows) <= count:
        return list(rows)
    return [
        rows[(index * len(rows)) // count] for index in range(count)
    ]


def content_tokens(
    values: list[int],
    eos_ids: set[int],
) -> list[int]:
    output = []
    for value in values:
        if int(value) in eos_ids:
            break
        output.append(int(value))
    return output


def censored_tokens(
    values: list[int],
    eos_ids: set[int],
) -> list[int]:
    output = []
    for value in values:
        output.append(int(value))
        if int(value) in eos_ids:
            break
    return output


def terminated(values: list[int], eos_ids: set[int]) -> bool:
    return any(int(value) in eos_ids for value in values)


def trim_finished(
    generated: list[list[int]],
    eos_ids: set[int],
) -> list[list[int]]:
    return [censored_tokens(row, eos_ids) for row in generated]


def generate_case_outputs(
    model,
    device: torch.device,
    rows: list[dict[str, Any]],
    *,
    eos_ids: set[int],
    batch_size: int,
    steps: int,
) -> dict[int, list[int]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_length[len(row["input_ids"])].append(row)
    outputs: dict[int, list[int]] = {}
    for length in sorted(by_length):
        panel = by_length[length]
        for start in range(0, len(panel), 2 * batch_size):
            batch = panel[start:start + 2 * batch_size]
            input_ids = torch.tensor(
                [row["input_ids"] for row in batch],
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.ones_like(input_ids)
            generated = [[] for _ in batch]
            finished = torch.zeros(
                len(batch), dtype=torch.bool, device=device
            )
            past = None
            with torch.inference_mode():
                for step in range(steps):
                    if step == 0:
                        output = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=True,
                            return_dict=True,
                        )
                    else:
                        output = model(
                            input_ids=next_token[:, None],
                            attention_mask=attention_mask,
                            past_key_values=past,
                            use_cache=True,
                            return_dict=True,
                        )
                    logits = output.logits[:, -1, :].float()
                    next_token = torch.argmax(logits, dim=-1)
                    if eos_ids:
                        fallback = min(eos_ids)
                        next_token = torch.where(
                            finished,
                            torch.full_like(next_token, fallback),
                            next_token,
                        )
                    for slot, token in enumerate(
                        next_token.detach().cpu().tolist()
                    ):
                        if not bool(finished[slot]):
                            generated[slot].append(int(token))
                    if eos_ids:
                        is_eos = torch.zeros_like(finished)
                        for eos_id in eos_ids:
                            is_eos |= next_token == int(eos_id)
                        finished |= is_eos
                    past = output.past_key_values
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (len(batch), 1),
                                dtype=attention_mask.dtype,
                                device=device,
                            ),
                        ),
                        dim=1,
                    )
                    del output, logits
                    if bool(finished.all()):
                        break
            for row, values in zip(batch, generated):
                outputs[int(row["semantic_case_index"])] = list(values)
            del input_ids, attention_mask, next_token, past
    return outputs


def pair_batch(
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    site: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = []
    for target in target_rows:
        rows.extend((
            cases[int(target["target_case_index"])],
            cases[int(target["cross_case_index"])],
        ))
    lengths = {len(row["input_ids"]) for row in rows}
    if len(lengths) != 1:
        raise RuntimeError(f"cached pair length drift: {lengths}")
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    positions = torch.zeros(
        (len(rows), protocol.MAX_ROLE_SPAN),
        dtype=torch.long,
        device=device,
    )
    masks = torch.zeros_like(positions, dtype=torch.bool)
    for slot, row in enumerate(rows):
        role_start, role_end = row["role_spans"][site]
        span = list(range(int(role_start), int(role_end) + 1))
        if len(span) > protocol.MAX_ROLE_SPAN:
            raise RuntimeError(f"role span too wide: {site} {span}")
        positions[slot, :len(span)] = torch.tensor(
            span, dtype=torch.long, device=device
        )
        masks[slot, :len(span)] = True
    return input_ids, attention_mask, positions, masks


def make_swap(
    layers: list[Any],
    condition: dict[str, Any],
    head_dim: int,
):
    channels = [str(value) for value in condition["channels"]]
    if set(channels) == {"k", "v"}:
        return bridge.OnlineKVSwap(
            layers,
            [int(value) for value in condition["depths"]],
            [int(value) for value in condition["groups"]],
            head_dim,
        )
    return trajectory.OnlineChannelSwap(
        layers,
        [int(value) for value in condition["depths"]],
        [int(value) for value in condition["groups"]],
        channels,
        head_dim,
    )


def generate_paired_cached(
    model,
    device: torch.device,
    layers: list[Any],
    target_rows: list[dict[str, Any]],
    cases: dict[int, dict[str, Any]],
    condition: dict[str, Any],
    *,
    head_dim: int,
    eos_ids: set[int],
    pair_batch_size: int,
    steps: int,
) -> dict[int, dict[str, list[int]]]:
    by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for target in target_rows:
        left = cases[int(target["target_case_index"])]
        right = cases[int(target["cross_case_index"])]
        if len(left["input_ids"]) != len(right["input_ids"]):
            raise RuntimeError("paired prompt lengths differ")
        by_length[len(left["input_ids"])].append(target)
    records = {}
    for length in sorted(by_length):
        panel = by_length[length]
        for start in range(0, len(panel), pair_batch_size):
            batch = panel[start:start + pair_batch_size]
            (
                input_ids,
                attention_mask,
                positions,
                masks,
            ) = pair_batch(
                batch,
                cases,
                str(condition["site"]),
                device=device,
            )
            generated = [[] for _ in range(input_ids.shape[0])]
            finished = torch.zeros(
                input_ids.shape[0], dtype=torch.bool, device=device
            )
            swap = make_swap(layers, condition, head_dim)
            swap.register()
            try:
                swap.begin(positions, masks)
                with torch.inference_mode():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                swap.end()
            finally:
                swap.close()
            logits = output.logits[:, -1, :].float()
            next_token = torch.argmax(logits, dim=-1)
            past = output.past_key_values
            del output, logits
            with torch.inference_mode():
                for step in range(steps):
                    if step > 0:
                        output = model(
                            input_ids=next_token[:, None],
                            attention_mask=attention_mask,
                            past_key_values=past,
                            use_cache=True,
                            return_dict=True,
                        )
                        logits = output.logits[:, -1, :].float()
                        next_token = torch.argmax(logits, dim=-1)
                        past = output.past_key_values
                        del output, logits
                    if eos_ids:
                        fallback = min(eos_ids)
                        next_token = torch.where(
                            finished,
                            torch.full_like(next_token, fallback),
                            next_token,
                        )
                    for slot, token in enumerate(
                        next_token.detach().cpu().tolist()
                    ):
                        if not bool(finished[slot]):
                            generated[slot].append(int(token))
                    if eos_ids:
                        is_eos = torch.zeros_like(finished)
                        for eos_id in eos_ids:
                            is_eos |= next_token == int(eos_id)
                        finished |= is_eos
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.ones(
                                (input_ids.shape[0], 1),
                                dtype=attention_mask.dtype,
                                device=device,
                            ),
                        ),
                        dim=1,
                    )
                    if bool(finished.all()):
                        break
            generated = trim_finished(generated, eos_ids)
            for pair_slot, target in enumerate(batch):
                records[int(target["target_index"])] = {
                    "target": generated[2 * pair_slot],
                    "cross": generated[2 * pair_slot + 1],
                }
            del input_ids, attention_mask, positions, masks, next_token, past
    return records


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return index
    return None


def prefix_fraction(candidate: list[int], donor: list[int]) -> float:
    if not donor:
        return 1.0 if not candidate else 0.0
    count = 0
    for left, right in zip(candidate, donor):
        if left != right:
            break
        count += 1
    return count / len(donor)


def evaluate_condition(
    target_rows: list[dict[str, Any]],
    clean_outputs: dict[int, list[int]],
    patched: dict[int, dict[str, list[int]]],
    eos_ids: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    step_matches: dict[int, list[bool]] = defaultdict(list)
    for target in target_rows:
        target_index = int(target["target_index"])
        left_index = int(target["target_case_index"])
        right_index = int(target["cross_case_index"])
        clean_left = censored_tokens(
            clean_outputs[left_index], eos_ids
        )
        clean_right = censored_tokens(
            clean_outputs[right_index], eos_ids
        )
        patch_left = censored_tokens(
            patched[target_index]["target"], eos_ids
        )
        patch_right = censored_tokens(
            patched[target_index]["cross"], eos_ids
        )
        content_clean_left = content_tokens(clean_left, eos_ids)
        content_clean_right = content_tokens(clean_right, eos_ids)
        content_patch_left = content_tokens(patch_left, eos_ids)
        content_patch_right = content_tokens(patch_right, eos_ids)
        left_diff = first_difference(
            content_clean_left, content_clean_right
        )
        right_diff = first_difference(
            content_clean_right, content_clean_left
        )

        def distinguishing_match(
            candidate: list[int],
            donor: list[int],
            index: int | None,
        ) -> bool:
            return (
                index is not None
                and index < len(candidate)
                and index < len(donor)
                and candidate[index] == donor[index]
            )

        left_eos = patch_left == clean_right
        right_eos = patch_right == clean_left
        left_content = content_patch_left == content_clean_right
        right_content = content_patch_right == content_clean_left
        max_steps = max(len(content_clean_left), len(content_clean_right))
        for step in range(max_steps):
            if step < len(content_clean_right):
                step_matches[step].append(
                    step < len(content_patch_left)
                    and content_patch_left[step]
                    == content_clean_right[step]
                )
            if step < len(content_clean_left):
                step_matches[step].append(
                    step < len(content_patch_right)
                    and content_patch_right[step]
                    == content_clean_left[step]
                )
        records.append({
            "target_index": target_index,
            "pair_family": str(target["pair_family"]),
            "target_matches_other_clean_eos": left_eos,
            "cross_matches_other_clean_eos": right_eos,
            "both_match_other_clean_eos": left_eos and right_eos,
            "target_matches_other_clean_content": left_content,
            "cross_matches_other_clean_content": right_content,
            "both_match_other_clean_content": (
                left_content and right_content
            ),
            "target_first_distinguishing_token_matches": (
                distinguishing_match(
                    content_patch_left, content_clean_right, left_diff
                )
            ),
            "cross_first_distinguishing_token_matches": (
                distinguishing_match(
                    content_patch_right, content_clean_left, right_diff
                )
            ),
            "target_donor_prefix_fraction": prefix_fraction(
                content_patch_left, content_clean_right
            ),
            "cross_donor_prefix_fraction": prefix_fraction(
                content_patch_right, content_clean_left
            ),
            "target_terminated": terminated(patch_left, eos_ids),
            "cross_terminated": terminated(patch_right, eos_ids),
            "termination_steps_match": (
                len(patch_left) == len(clean_right)
                and len(patch_right) == len(clean_left)
            ),
            "clean_target": clean_left,
            "clean_cross": clean_right,
            "patched_target": patch_left,
            "patched_cross": patch_right,
        })
    count = len(records)

    def mean(key: str) -> float:
        return (
            sum(float(row[key]) for row in records) / count
            if count else 0.0
        )

    prefix_values = [
        float(row[key])
        for row in records
        for key in (
            "target_donor_prefix_fraction",
            "cross_donor_prefix_fraction",
        )
    ]
    first_values = [
        bool(row[key])
        for row in records
        for key in (
            "target_first_distinguishing_token_matches",
            "cross_first_distinguishing_token_matches",
        )
    ]
    metrics = {
        "pair_count": count,
        "target_matches_other_clean_eos_rate": mean(
            "target_matches_other_clean_eos"
        ),
        "cross_matches_other_clean_eos_rate": mean(
            "cross_matches_other_clean_eos"
        ),
        "both_match_other_clean_eos_count": sum(
            bool(row["both_match_other_clean_eos"]) for row in records
        ),
        "both_match_other_clean_eos_rate": mean(
            "both_match_other_clean_eos"
        ),
        "both_match_other_clean_content_rate": mean(
            "both_match_other_clean_content"
        ),
        "first_distinguishing_token_match_rate": (
            sum(first_values) / len(first_values)
            if first_values else 0.0
        ),
        "donor_prefix_fraction_mean": (
            float(np.mean(prefix_values)) if prefix_values else 0.0
        ),
        "patched_termination_rate": (
            sum(
                bool(row["target_terminated"])
                + bool(row["cross_terminated"])
                for row in records
            )
            / (2 * count) if count else 0.0
        ),
        "termination_steps_match_rate": mean(
            "termination_steps_match"
        ),
        "donor_token_match_rate_by_step": {
            str(step): sum(values) / len(values)
            for step, values in sorted(step_matches.items())
        },
    }
    return metrics, records


def clean_behavior(
    cases: dict[int, dict[str, Any]],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> tuple[dict[str, Any], set[int]]:
    exact_indices = set()
    split_counts = Counter()
    termination_counts = Counter()
    examples = []
    for index, row in cases.items():
        generated = clean_outputs[index]
        split = str(row["split"])
        if terminated(generated, eos_ids):
            termination_counts[split] += 1
        exact = (
            content_tokens(generated, eos_ids)
            == [int(value) for value in row["expected_token_ids"]]
            and terminated(generated, eos_ids)
        )
        if exact:
            exact_indices.add(index)
            split_counts[split] += 1
        elif len(examples) < 12:
            examples.append({
                "case_key": row["case_key"],
                "expected": row["expected_token_ids"],
                "generated": generated,
            })
    total_by_split = Counter(str(row["split"]) for row in cases.values())
    return {
        "exact_case_counts": dict(split_counts),
        "total_case_counts": dict(total_by_split),
        "termination_counts": dict(termination_counts),
        "exact_case_rates": {
            split: split_counts[split] / total_by_split[split]
            for split in total_by_split
        },
        "mismatch_examples": examples,
    }, exact_indices


def valid_targets(
    rows: list[dict[str, Any]],
    exact_indices: set[int],
    clean_outputs: dict[int, list[int]],
    eos_ids: set[int],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        left = int(row["target_case_index"])
        right = int(row["cross_case_index"])
        if left not in exact_indices or right not in exact_indices:
            continue
        if (
            content_tokens(clean_outputs[left], eos_ids)
            == content_tokens(clean_outputs[right], eos_ids)
        ):
            continue
        output.append(row)
    return output


def condition_specifications(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    all_groups = [int(value) for value in plan["all_groups"]]
    early = [int(value) for value in plan["early_depths"]]
    post = [int(value) for value in plan["postsource_depths"]]
    all_depths = [int(value) for value in plan["all_layers"]]
    frozen_groups = [int(value) for value in plan["frozen_groups"]]
    frozen_depths = [int(value) for value in plan["frozen_depths"]]
    return {
        "phrase_post_kv": {
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "color_post_kv": {
            "pair_family": "color",
            "site": "source_color",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "noun_post_kv": {
            "pair_family": "noun",
            "site": "source_noun",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "phrase_early_kv": {
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": early,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "phrase_all_kv": {
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": all_depths,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "phrase_post_k_only": {
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["k"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "phrase_post_v_only": {
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["v"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "phrase_frozen_rectangle": {
            "pair_family": "phrase",
            "site": "source_phrase",
            "channels": ["k", "v"],
            "groups": frozen_groups,
            "depths": frozen_depths,
            "pair_limit": protocol.PAIR_LIMIT,
        },
        "operator_post_kv": {
            "pair_family": "phrase",
            "site": "operator",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
        "target_language_post_kv": {
            "pair_family": "phrase",
            "site": "target_language",
            "channels": ["k", "v"],
            "groups": all_groups,
            "depths": post,
            "pair_limit": protocol.CONTROL_PAIR_LIMIT,
        },
    }


def run(model_name: str) -> None:
    prereg = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "preregistration.json"
    )
    audit = protocol.read_json(
        protocol.OUT_ROOT / "protocol" / "audit.json"
    )
    if not audit["all_checks_passed"]:
        raise RuntimeError("Phase1058 protocol audit failed")
    case_rows = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"cases.{model_name}.jsonl"
    )
    targets = protocol.read_jsonl(
        protocol.OUT_ROOT
        / "protocol"
        / f"targets.{model_name}.jsonl"
    )
    cases = {
        int(row["semantic_case_index"]): row for row in case_rows
    }
    plan = prereg["model_plans"][model_name]
    started = time.time()
    model = tokenizer = None
    try:
        model, tokenizer, device, placement = load_fp16(model_name)
        precision = quantization_audit(model)
        if (
            precision["has_quantized_modules"]
            or precision["has_bf16_parameters"]
            or not precision["has_fp16_parameters"]
        ):
            raise RuntimeError("FP16/no-quantization audit failed")
        layers = list(get_layers(model))
        width = bridge.projection_width(layers[0].self_attn.k_proj)
        n_kv_heads = int(plan["n_kv_heads"])
        if width % n_kv_heads:
            raise RuntimeError("KV projection geometry drift")
        head_dim = width // n_kv_heads
        eos_ids = set(eos_tools.eos_token_ids(model, tokenizer))
        if not eos_ids:
            raise RuntimeError("no EOS token ids discovered")
        batch_size = PAIR_BATCH_SIZE[model_name]
        clean_outputs = generate_case_outputs(
            model,
            device,
            case_rows,
            eos_ids=eos_ids,
            batch_size=batch_size,
            steps=int(prereg["generation_steps"]),
        )
        behavior_summary, exact_indices = clean_behavior(
            cases, clean_outputs, eos_ids
        )
        valid_by_split_family = {}
        for split in ("discovery", "confirmation"):
            for family in protocol.PAIR_FAMILIES:
                valid_by_split_family[(split, family)] = valid_targets(
                    [
                        row for row in targets
                        if row["split"] == split
                        and row["pair_family"] == family
                    ],
                    exact_indices,
                    clean_outputs,
                    eos_ids,
                )

        gates = prereg["gates"]
        behavior_gate = (
            behavior_summary["exact_case_counts"].get(
                "discovery", 0
            ) >= gates["discovery_exact_case_count_min"]
            and behavior_summary["exact_case_counts"].get(
                "confirmation", 0
            ) >= gates["confirmation_exact_case_count_min"]
            and all(
                len(valid_by_split_family[("confirmation", family)])
                >= gates["confirmation_pair_count_per_family_min"]
                for family in protocol.PAIR_FAMILIES
            )
        )
        specs = condition_specifications(plan)
        condition_results = {}
        condition_records = {}
        selected_targets = {}
        for name, spec in specs.items():
            family = str(spec["pair_family"])
            selected = evenly_spaced(
                valid_by_split_family[("confirmation", family)],
                int(spec["pair_limit"]),
            )
            selected_targets[name] = [
                int(row["target_index"]) for row in selected
            ]
            if not selected:
                condition_results[name] = {
                    "pair_count": 0,
                    "both_match_other_clean_eos_rate": 0.0,
                }
                condition_records[name] = []
                continue
            patched = generate_paired_cached(
                model,
                device,
                layers,
                selected,
                cases,
                spec,
                head_dim=head_dim,
                eos_ids=eos_ids,
                pair_batch_size=batch_size,
                steps=int(prereg["generation_steps"]),
            )
            metrics, records = evaluate_condition(
                selected, clean_outputs, patched, eos_ids
            )
            condition_results[name] = metrics
            condition_records[name] = records
            print(json.dumps({
                "model": model_name,
                "condition": name,
                "pairs": metrics["pair_count"],
                "eos_exact": metrics[
                    "both_match_other_clean_eos_rate"
                ],
                "content_exact": metrics[
                    "both_match_other_clean_content_rate"
                ],
                "first_distinguishing": metrics[
                    "first_distinguishing_token_match_rate"
                ],
            }), flush=True)

        # Cache parity is an instrument audit: compare the cached result to
        # the established full-recomputation implementation on a small,
        # predeclared subset using first-token outputs.
        phrase_rows = evenly_spaced(
            valid_by_split_family[("confirmation", "phrase")],
            int(prereg["cache_parity_pair_limit"]),
        )
        parity_spec = specs["phrase_post_kv"]
        cached_records = condition_records["phrase_post_kv"]
        cached_by_index = {
            int(row["target_index"]): row for row in cached_records
        }
        parity_rows = [
            row for row in phrase_rows
            if int(row["target_index"]) in cached_by_index
        ]
        recompute = trajectory.run_output_condition(
            model,
            device,
            layers,
            parity_rows,
            cases,
            {
                key: value for key, value in parity_spec.items()
                if key not in ("pair_family", "pair_limit")
            },
            head_dim=head_dim,
            pad_token_id=(
                int(tokenizer.pad_token_id)
                if tokenizer.pad_token_id is not None
                else min(eos_ids)
            ),
            pair_batch_size=batch_size,
        )
        parity_arm_values = []
        for pair_slot, row in enumerate(parity_rows):
            record = cached_by_index[int(row["target_index"])]
            cached_left = record["patched_target"]
            cached_right = record["patched_cross"]
            parity_arm_values.extend((
                bool(cached_left)
                and int(cached_left[0])
                == int(recompute["top1"][pair_slot, 0]),
                bool(cached_right)
                and int(cached_right[0])
                == int(recompute["top1"][pair_slot, 1]),
            ))
        cache_parity = {
            "pair_count": len(parity_rows),
            "arm_count": len(parity_arm_values),
            "first_token_match_rate": (
                sum(parity_arm_values) / len(parity_arm_values)
                if parity_arm_values else 0.0
            ),
        }
        cache_parity_passed = (
            cache_parity["first_token_match_rate"]
            >= gates["cache_parity_rate_min"]
        )

        phrase_rate = condition_results["phrase_post_kv"][
            "both_match_other_clean_eos_rate"
        ]
        color_rate = condition_results["color_post_kv"][
            "both_match_other_clean_eos_rate"
        ]
        noun_rate = condition_results["noun_post_kv"][
            "both_match_other_clean_eos_rate"
        ]
        control_rate = max(
            condition_results["operator_post_kv"][
                "both_match_other_clean_eos_rate"
            ],
            condition_results["target_language_post_kv"][
                "both_match_other_clean_eos_rate"
            ],
        )
        composition_gate = (
            behavior_gate
            and cache_parity_passed
            and phrase_rate
            >= gates["phrase_post_eos_exact_rate_min"]
            and color_rate
            >= gates["component_post_eos_exact_rate_min"]
            and noun_rate
            >= gates["component_post_eos_exact_rate_min"]
            and phrase_rate - control_rate
            >= gates["source_minus_control_rate_min"]
        )
        phase_rates = {
            key: condition_results[key][
                "both_match_other_clean_eos_rate"
            ]
            for key in (
                "phrase_early_kv",
                "phrase_post_kv",
                "phrase_all_kv",
            )
        }
        if (
            phase_rates["phrase_early_kv"] >= 0.30
            and phase_rates["phrase_post_kv"] >= 0.50
            and phase_rates["phrase_all_kv"] <= 0.10
        ):
            phase_class = "early_post_conflict"
        elif (
            phase_rates["phrase_early_kv"] <= 0.10
            and phase_rates["phrase_post_kv"] >= 0.50
            and phase_rates["phrase_all_kv"] >= 0.50
        ):
            phase_class = "late_dominant"
        else:
            phase_class = "mixed_or_unresolved"
        summary = {
            "schema_version": "phase1058_model_summary.v1",
            "phase": protocol.PHASE,
            "model": model_name,
            "protocol_digest": prereg["protocol_digest"],
            "precision": precision,
            "placement": placement,
            "eos_token_ids": sorted(eos_ids),
            "clean_behavior": behavior_summary,
            "valid_pair_counts": {
                f"{split}.{family}": len(rows)
                for (split, family), rows in sorted(
                    valid_by_split_family.items()
                )
            },
            "behavior_gate_passed": behavior_gate,
            "condition_results": condition_results,
            "condition_records": condition_records,
            "selected_target_indices": selected_targets,
            "cache_parity": cache_parity,
            "cache_parity_passed": cache_parity_passed,
            "component_rates": {
                "phrase": phrase_rate,
                "color": color_rate,
                "noun": noun_rate,
            },
            "phase_rates": phase_rates,
            "phase_class": phase_class,
            "channel_rates": {
                "k_only": condition_results["phrase_post_k_only"][
                    "both_match_other_clean_eos_rate"
                ],
                "v_only": condition_results["phrase_post_v_only"][
                    "both_match_other_clean_eos_rate"
                ],
                "kv": phrase_rate,
            },
            "frozen_rectangle_rate": condition_results[
                "phrase_frozen_rectangle"
            ]["both_match_other_clean_eos_rate"],
            "maximum_role_control_rate": control_rate,
            "composition_gate_passed": composition_gate,
            "elapsed_seconds": float(time.time() - started),
        }
        protocol.write_json(
            protocol.OUT_ROOT / "atlas" / model_name / "summary.json",
            summary,
        )
        print(json.dumps({
            "model": model_name,
            "behavior_gate": behavior_gate,
            "exact_cases": behavior_summary["exact_case_counts"],
            "component_rates": summary["component_rates"],
            "phase_rates": phase_rates,
            "phase_class": phase_class,
            "channel_rates": summary["channel_rates"],
            "frozen": summary["frozen_rectangle_rate"],
            "control": control_rate,
            "cache_parity": cache_parity,
            "composition_gate": composition_gate,
            "elapsed_seconds": summary["elapsed_seconds"],
        }), flush=True)
    finally:
        if model is not None:
            release_fp16(model)
        del tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=protocol.MODELS
    )
    args = parser.parse_args()
    run(args.model)


if __name__ == "__main__":
    main()
