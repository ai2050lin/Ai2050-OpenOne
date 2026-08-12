#!/usr/bin/env python3
"""Natural-generation confirmation of discovery-frozen anchor subsets."""
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

from model_utils import get_layers, load_model, release_model
from phase1000_factorial_binding_behavior import eos_ids
from phase1003_anchor_subset_exhaustive import choose_donors
from phase1003_crossparadigm_protocol import (
    ANCHOR_ROLES,
    DOMAINS,
    MODELS,
    OUT_ROOT,
    PHASE,
    read_json,
    selected_directional_rows,
    write_json,
    write_jsonl,
)


def case_tensors(rows: list[dict[str, Any]], device):
    widths = {len(row["input_ids"]) for row in rows}
    if len(widths) != 1:
        raise RuntimeError(f"input width drift {widths}")
    input_ids = torch.tensor(
        [row["input_ids"] for row in rows],
        dtype=torch.long,
        device=device,
    )
    return input_ids, torch.ones_like(input_ids)


def capture_prompt_depth(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    depth: int,
) -> torch.Tensor:
    input_ids, attention = case_tensors(cases, device)
    captured = []

    def hook(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach())

    handle = layers[depth - 1].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        if len(captured) != 1:
            raise RuntimeError(f"capture count {len(captured)}")
        return captured[0]
    finally:
        handle.remove()
        del input_ids, attention


def generation_patch(
    depth: int,
    target_cases: list[dict[str, Any]],
    target_hidden: torch.Tensor,
    donor_cases: list[dict[str, Any]],
    donor_hidden: torch.Tensor,
    roles: list[str],
) -> dict[str, Any]:
    role_set = set(roles)
    batch_index = torch.arange(
        len(target_cases), device=target_hidden.device
    )
    positions = {}
    vectors = {}
    for role in ANCHOR_ROLES:
        target_positions = torch.tensor(
            [
                int(row["role_positions"][role])
                for row in target_cases
            ],
            dtype=torch.long,
            device=target_hidden.device,
        )
        donor_positions = torch.tensor(
            [
                int(row["role_positions"][role])
                for row in donor_cases
            ],
            dtype=torch.long,
            device=donor_hidden.device,
        )
        positions[role] = target_positions
        vectors[role] = (
            donor_hidden[batch_index, donor_positions, :]
            if role in role_set
            else target_hidden[batch_index, target_positions, :]
        )
    return {
        "depth": depth,
        "prompt_width": len(target_cases[0]["input_ids"]),
        "positions": positions,
        "vectors": vectors,
    }


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
    handle = None
    count = [0]
    if patch is not None:
        def hook(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.shape[1] != patch["prompt_width"]:
                return output
            patched = value.clone()
            batch_index = torch.arange(
                patched.shape[0], device=patched.device
            )
            for role in ANCHOR_ROLES:
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
    try:
        answer_len = max(
            len(row["answer_token_ids"]) for row in cases
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
) -> tuple[list[int], int | None]:
    for index, token_id in enumerate(values):
        if token_id in eos_set:
            return values[:index], index
    return values, None


def batches(
    rows: list[dict[str, Any]],
    donors: list[dict[str, Any]],
    batch_size: int,
):
    groups = defaultdict(list)
    for row, donor in zip(rows, donors):
        groups[int(row["template"])].append((row, donor))
    for _, values in sorted(groups.items()):
        values.sort(key=lambda item: (
            item[0]["pair_id"], item[0]["direction"]
        ))
        for start in range(0, len(values), batch_size):
            chunk = values[start : start + batch_size]
            yield (
                [item[0] for item in chunk],
                [item[1] for item in chunk],
            )


def semantic_label(
    ids: list[int],
    case: dict[str, Any],
) -> str | None:
    step = int(case["semantic_step"])
    if len(ids) <= step:
        return None
    return {
        int(token_id): label
        for label, token_id in case["candidate_token_ids"].items()
    }.get(ids[step])


def run_domain(
    model,
    tokenizer,
    layers,
    device,
    model_name: str,
    domain: str,
    source_depth: int,
    batch_size: int,
    effective_eos: list[int],
) -> dict[str, Any]:
    anchor_summary = read_json(
        OUT_ROOT
        / "anchor_subsets"
        / model_name
        / domain
        / "summary.json"
    )
    frozen = anchor_summary["discovery_selection"][
        "selected_subsets"
    ]
    if not frozen:
        raise RuntimeError(f"{model_name}/{domain}: no frozen subset")
    rows = selected_directional_rows(
        model_name, domain, "confirmation"
    )
    donors, donor_audit = choose_donors(
        rows, model_name, domain, "confirmation"
    )
    eos_set = set(effective_eos)
    result_rows = []
    all_batches = list(batches(rows, donors, batch_size))
    for batch_number, (batch, donor_batch) in enumerate(
        all_batches, 1
    ):
        target_cases = [row["target"] for row in batch]
        target_hidden = capture_prompt_depth(
            model, layers, device, target_cases, source_depth
        )
        donor_hidden = capture_prompt_depth(
            model, layers, device, donor_batch, source_depth
        )
        clean_suffixes = generate(
            model,
            tokenizer,
            layers,
            device,
            target_cases,
            effective_eos,
            None,
        )
        noop_patch = generation_patch(
            source_depth,
            target_cases,
            target_hidden,
            donor_batch,
            donor_hidden,
            [],
        )
        noop_suffixes = generate(
            model,
            tokenizer,
            layers,
            device,
            target_cases,
            effective_eos,
            noop_patch,
        )
        conditions = [
            ("target_noop", [], noop_suffixes)
        ]
        for roles in frozen:
            patch = generation_patch(
                source_depth,
                target_cases,
                target_hidden,
                donor_batch,
                donor_hidden,
                roles,
            )
            suffixes = generate(
                model,
                tokenizer,
                layers,
                device,
                target_cases,
                effective_eos,
                patch,
            )
            conditions.append((
                "frozen_" + "_".join(roles),
                roles,
                suffixes,
            ))
        for condition, roles, suffixes in conditions:
            for index, row in enumerate(batch):
                clean_ids, _ = strip_eos(
                    clean_suffixes[index], eos_set
                )
                generated_ids, eos_position = strip_eos(
                    suffixes[index], eos_set
                )
                expected_target = [
                    int(value)
                    for value in row["target"]["answer_token_ids"]
                ]
                expected_donor = [
                    int(value)
                    for value in donor_batch[index]["answer_token_ids"]
                ]
                prediction = semantic_label(
                    generated_ids, row["target"]
                )
                result_rows.append({
                    "schema_version": (
                        "phase1003_anchor_natural_row.v1"
                    ),
                    "phase": PHASE,
                    "model": model_name,
                    "domain": domain,
                    "split": "confirmation",
                    "pair_id": row["pair_id"],
                    "direction": row["direction"],
                    "template": row["template"],
                    "condition": condition,
                    "roles": list(roles),
                    "target_gold": row["target"]["gold"],
                    "donor_gold": donor_batch[index]["gold"],
                    "generated_ids": generated_ids,
                    "generated_text": tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "semantic_prediction": prediction,
                    "target_semantic": (
                        prediction == row["target"]["gold"]
                    ),
                    "donor_semantic": (
                        prediction == donor_batch[index]["gold"]
                    ),
                    "target_exact": generated_ids == expected_target,
                    "donor_exact": generated_ids == expected_donor,
                    "eos_position": eos_position,
                    "noop_sequence_agreement": (
                        generated_ids == clean_ids
                        if condition == "target_noop"
                        else None
                    ),
                })
        print(
            f"[{model_name}/{domain}] "
            f"{batch_number}/{len(all_batches)}",
            flush=True,
        )
    conditions = sorted({
        row["condition"] for row in result_rows
    })
    condition_summary = {}
    for condition in conditions:
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
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    frozen_conditions = [
        item for name, item in condition_summary.items()
        if name != "target_noop"
    ]
    summary = {
        "schema_version": "phase1003_anchor_natural_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "domain": domain,
        "status": "complete",
        "source_depth": source_depth,
        "frozen_subsets": frozen,
        "direction_count": len(rows),
        "donor_audit": donor_audit,
        "condition_summary": condition_summary,
        "natural_confirmation_pass": (
            condition_summary["target_noop"][
                "noop_sequence_agreement"
            ] >= thresholds["noop_prediction_agreement"]
            and all(
                item["donor_semantic_rate"]
                >= thresholds["full_anchor_donor_rate"]
                and item["donor_exact_rate"]
                >= thresholds["full_anchor_donor_rate"]
                for item in frozen_conditions
            )
        ),
        "claim_boundary": (
            "Natural confirmation tests the frozen short-answer protocol. "
            "It does not establish open-ended or multi-sentence rollout."
        ),
    }
    root = (
        OUT_ROOT / "anchor_natural" / model_name / domain
    )
    write_jsonl(root / "rows.jsonl", result_rows)
    write_json(root / "summary.json", summary)
    return summary


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    anchor_model = read_json(
        OUT_ROOT
        / "anchor_subsets"
        / model_name
        / "summary.json"
    )
    domains = [
        domain
        for domain, summary in anchor_model["domains"].items()
        if summary["parent_instrument_pass"]
        and summary["frozen_subset_confirmation_pass"]
    ]
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][model_name])
    model = tokenizer = None
    started = time.time()
    summaries = {}
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        tokenizer.padding_side = "left"
        layers = get_layers(model)
        effective_eos = eos_ids(model, tokenizer)
        for domain in domains:
            summaries[domain] = run_domain(
                model,
                tokenizer,
                layers,
                device,
                model_name,
                domain,
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
        "schema_version": "phase1003_anchor_natural_model.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "domains": summaries,
        "pass_count": sum(
            summary["natural_confirmation_pass"]
            for summary in summaries.values()
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        OUT_ROOT / "anchor_natural" / model_name / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            OUT_ROOT
            / "anchor_natural"
            / model_name
            / "summary.json"
        )
        if path.exists():
            summaries[model_name] = read_json(path)
    cross_domain = {}
    for domain in DOMAINS:
        available = [
            summary["domains"][domain]
            for summary in summaries.values()
            if domain in summary["domains"]
        ]
        cross_domain[domain] = {
            "tested_model_count": len(available),
            "pass_count": sum(
                item["natural_confirmation_pass"]
                for item in available
            ),
        }
    payload = {
        "schema_version": "phase1003_anchor_natural_aggregate.v1",
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_domain": cross_domain,
    }
    write_json(
        OUT_ROOT / "anchor_natural" / "summary.json", payload
    )
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
