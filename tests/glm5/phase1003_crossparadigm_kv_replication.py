#!/usr/bin/env python3
"""Replicate value-dominant cache transport across Phase 1003 paradigms."""
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
from transformers import DynamicCache


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase1003_anchor_natural_confirmation import capture_prompt_depth
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


CONDITIONS = (
    "target_cache",
    "all_source_cache",
    "source_keys_only",
    "source_values_only",
    "frozen_value_layers_only",
    "restore_frozen_value_layers",
)


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


def candidate_logits(
    logits: torch.Tensor,
    candidate_ids: dict[str, int],
) -> torch.Tensor:
    ids = torch.tensor(
        list(candidate_ids.values()),
        dtype=torch.long,
        device=logits.device,
    )
    return logits.index_select(-1, ids).float().detach()


def predictions(
    logits: torch.Tensor,
    candidate_ids: dict[str, int],
) -> list[str]:
    labels = list(candidate_ids)
    indices = logits.argmax(dim=-1).detach().cpu().tolist()
    return [labels[int(index)] for index in indices]


def patch_spec(
    depth: int,
    target_cases: list[dict[str, Any]],
    donor_cases: list[dict[str, Any]],
    donor_hidden: torch.Tensor,
) -> dict[str, Any]:
    batch_index = torch.arange(
        len(target_cases), device=donor_hidden.device
    )
    positions = {}
    vectors = {}
    for role in ANCHOR_ROLES:
        positions[role] = torch.tensor(
            [
                int(row["role_positions"][role])
                for row in target_cases
            ],
            dtype=torch.long,
            device=donor_hidden.device,
        )
        donor_positions = torch.tensor(
            [
                int(row["role_positions"][role])
                for row in donor_cases
            ],
            dtype=torch.long,
            device=donor_hidden.device,
        )
        vectors[role] = donor_hidden[
            batch_index, donor_positions, :
        ]
    return {
        "depth": depth,
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
        for role in ANCHOR_ROLES:
            patched[
                batch_index,
                patch["positions"][role],
                :,
            ] = patch["vectors"][role].to(
                device=patched.device, dtype=patched.dtype
            )
        count[0] += 1
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    handle = layers[patch["depth"] - 1].register_forward_hook(hook)
    return handle, count


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


def full_forward(
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
            raise RuntimeError(f"full patch count {count[0]}")
        return candidate_logits(
            output.logits[:, -1, :],
            cases[0]["candidate_token_ids"],
        )
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
        [label_index[row["gold"]] for row in donors],
        dtype=torch.long,
        device=logits.device,
    )
    target_index = torch.tensor(
        [label_index[row["gold"]] for row in targets],
        dtype=torch.long,
        device=logits.device,
    )
    return (
        logits[batch_index, donor_index]
        - logits[batch_index, target_index]
    )


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


def run_domain(
    model,
    layers,
    device,
    model_name: str,
    domain: str,
    source_depth: int,
    frozen_layer_numbers: list[int],
    batch_size: int,
) -> dict[str, Any]:
    layer_count = len(layers)
    frozen_layers = {
        int(number) - 1 for number in frozen_layer_numbers
    }
    all_layers = set(range(layer_count))
    result_rows = []
    donor_audits = {}
    for split in ("discovery", "confirmation"):
        rows = selected_directional_rows(
            model_name, domain, split
        )
        donors, donor_audit = choose_donors(
            rows, model_name, domain, split
        )
        donor_audits[split] = donor_audit
        all_batches = list(batches(rows, donors, batch_size))
        for batch_number, (batch, donor_batch) in enumerate(
            all_batches, 1
        ):
            target_cases = [row["target"] for row in batch]
            donor_hidden = capture_prompt_depth(
                model,
                layers,
                device,
                donor_batch,
                source_depth,
            )
            semantic_step = int(target_cases[0]["semantic_step"])
            prefix_step = semantic_step - 1
            if prefix_step < 0:
                raise RuntimeError("semantic step lacks generated prefix")
            prefix_cases = [
                step_case(case, prefix_step)
                for case in target_cases
            ]
            semantic_cases = [
                step_case(case, semantic_step)
                for case in target_cases
            ]
            patch_prefix = patch_spec(
                source_depth,
                prefix_cases,
                donor_batch,
                donor_hidden,
            )
            patch_semantic = patch_spec(
                source_depth,
                semantic_cases,
                donor_batch,
                donor_hidden,
            )
            target_cache = build_cache(
                model, layers, device, prefix_cases, None
            )
            source_cache = build_cache(
                model,
                layers,
                device,
                prefix_cases,
                patch_prefix,
            )
            current_ids = [
                int(case["answer_token_ids"][prefix_step])
                for case in target_cases
            ]
            prefix_length = len(prefix_cases[0]["input_ids"])
            candidate_ids = target_cases[0]["candidate_token_ids"]
            cache_conditions = {
                "target_cache": (set(), set()),
                "all_source_cache": (all_layers, all_layers),
                "source_keys_only": (all_layers, set()),
                "source_values_only": (set(), all_layers),
                "frozen_value_layers_only": (
                    set(), frozen_layers
                ),
                "restore_frozen_value_layers": (
                    set(), all_layers - frozen_layers
                ),
            }
            logits_by_condition = {}
            for condition, (
                key_layers,
                value_layers,
            ) in cache_conditions.items():
                logits_by_condition[condition] = continue_cache(
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
            target_full = full_forward(
                model, layers, device, semantic_cases, None
            )
            source_full = full_forward(
                model,
                layers,
                device,
                semantic_cases,
                patch_semantic,
            )
            labels = list(candidate_ids)
            margins = {
                condition: contrast_margin(
                    logits,
                    labels,
                    donor_batch,
                    target_cases,
                )
                for condition, logits in logits_by_condition.items()
            }
            target_margin = margins["target_cache"]
            source_margin = margins["all_source_cache"]
            value_margin = margins["source_values_only"]
            predictions_by_condition = {
                condition: predictions(logits, candidate_ids)
                for condition, logits in logits_by_condition.items()
            }
            target_full_predictions = predictions(
                target_full, candidate_ids
            )
            source_full_predictions = predictions(
                source_full, candidate_ids
            )
            for condition in CONDITIONS:
                logits = logits_by_condition[condition]
                condition_predictions = predictions_by_condition[
                    condition
                ]
                for index, row in enumerate(batch):
                    all_span = max(
                        abs(float(
                            source_margin[index]
                            - target_margin[index]
                        )),
                        1e-8,
                    )
                    value_span = max(
                        abs(float(
                            value_margin[index]
                            - target_margin[index]
                        )),
                        1e-8,
                    )
                    result_rows.append({
                        "schema_version": (
                            "phase1003_crossparadigm_kv_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "domain": domain,
                        "split": split,
                        "pair_id": row["pair_id"],
                        "direction": row["direction"],
                        "template": row["template"],
                        "condition": condition,
                        "target_gold": row["target"]["gold"],
                        "donor_gold": donor_batch[index]["gold"],
                        "prediction": condition_predictions[index],
                        "donor_prediction": (
                            condition_predictions[index]
                            == donor_batch[index]["gold"]
                        ),
                        "target_prediction": (
                            condition_predictions[index]
                            == row["target"]["gold"]
                        ),
                        "margin": float(margins[condition][index]),
                        "normalized_all_cache_transfer": float(
                            (
                                margins[condition][index]
                                - target_margin[index]
                            )
                            / all_span
                        ),
                        "normalized_value_transfer": float(
                            (
                                margins[condition][index]
                                - target_margin[index]
                            )
                            / value_span
                        ),
                        "target_cache_full_prediction_agreement": (
                            condition_predictions[index]
                            == target_full_predictions[index]
                            if condition == "target_cache"
                            else None
                        ),
                        "source_cache_full_prediction_agreement": (
                            condition_predictions[index]
                            == source_full_predictions[index]
                            if condition == "all_source_cache"
                            else None
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
                target_cache,
                source_cache,
                logits_by_condition,
                target_full,
                source_full,
            )
            print(
                f"[{model_name}/{domain}/{split}] "
                f"{batch_number}/{len(all_batches)}",
                flush=True,
            )

    split_summary = {}
    for split in ("discovery", "confirmation"):
        split_summary[split] = {}
        for condition in CONDITIONS:
            values = [
                row
                for row in result_rows
                if row["split"] == split
                and row["condition"] == condition
            ]
            item = {
                "n": len(values),
                "donor_rate": float(np.mean([
                    row["donor_prediction"] for row in values
                ])),
                "target_rate": float(np.mean([
                    row["target_prediction"] for row in values
                ])),
                "median_normalized_all_cache_transfer": float(
                    np.median([
                        row["normalized_all_cache_transfer"]
                        for row in values
                    ])
                ),
                "median_normalized_value_transfer": float(
                    np.median([
                        row["normalized_value_transfer"]
                        for row in values
                    ])
                ),
            }
            if condition == "target_cache":
                item["full_prediction_agreement"] = float(np.mean([
                    row["target_cache_full_prediction_agreement"]
                    for row in values
                ]))
                item["maximum_candidate_logit_difference"] = float(max(
                    row["candidate_max_abs_difference_from_full"]
                    for row in values
                ))
            elif condition == "all_source_cache":
                item["full_prediction_agreement"] = float(np.mean([
                    row["source_cache_full_prediction_agreement"]
                    for row in values
                ]))
                item["maximum_candidate_logit_difference"] = float(max(
                    row["candidate_max_abs_difference_from_full"]
                    for row in values
                ))
            split_summary[split][condition] = item
        value_only = [
            row
            for row in result_rows
            if row["split"] == split
            and row["condition"] == "source_values_only"
        ]
        restore = {
            (row["pair_id"], row["direction"]): row
            for row in result_rows
            if row["split"] == split
            and row["condition"] == "restore_frozen_value_layers"
        }
        target = {
            (row["pair_id"], row["direction"]): row
            for row in result_rows
            if row["split"] == split
            and row["condition"] == "target_cache"
        }
        mediations = []
        for row in value_only:
            key = (row["pair_id"], row["direction"])
            mediations.append(
                (row["margin"] - restore[key]["margin"])
                / max(
                    abs(row["margin"] - target[key]["margin"]),
                    1e-8,
                )
            )
        split_summary[split][
            "frozen_value_layer_median_restoration_mediation"
        ] = float(np.median(mediations))

    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    thresholds = prereg["primary_thresholds"]
    checks = {}
    for split in ("discovery", "confirmation"):
        values = split_summary[split]
        checks[split] = {
            "source_cache_parent": (
                values["all_source_cache"]["donor_rate"]
                >= thresholds["full_anchor_donor_rate"]
            ),
            "value_dominant": (
                values["source_values_only"]["donor_rate"]
                >= thresholds["cache_value_donor_rate"]
            ),
            "frozen_layer_sufficiency": (
                values["frozen_value_layers_only"]["donor_rate"]
                >= thresholds[
                    "frozen_value_layer_sufficiency_rate"
                ]
            ),
            "frozen_layer_restoration": (
                values["restore_frozen_value_layers"]["target_rate"]
                >= thresholds["frozen_value_layer_restore_rate"]
            ),
            "target_cache_instrument": (
                values["target_cache"]["full_prediction_agreement"]
                >= 0.99
            ),
            "source_cache_instrument": (
                values["all_source_cache"]["full_prediction_agreement"]
                >= 0.99
            ),
        }
    summary = {
        "schema_version": (
            "phase1003_crossparadigm_kv_summary.v1"
        ),
        "phase": PHASE,
        "model": model_name,
        "domain": domain,
        "status": "complete",
        "source_depth": source_depth,
        "frozen_phase1002_value_layer_numbers": (
            frozen_layer_numbers
        ),
        "direction_count_per_split": 64,
        "donor_audits": donor_audits,
        "split_summary": split_summary,
        "checks": checks,
        "crossparadigm_kv_pass": all(
            all(values.values()) for values in checks.values()
        ),
        "claim_boundary": (
            "This tests whether Phase1002 cache roles and frozen value "
            "layers repeat in controlled attribute tasks. It does not "
            "identify KV heads or value channels."
        ),
    }
    root = OUT_ROOT / "kv_replication" / model_name / domain
    write_jsonl(root / "rows.jsonl", result_rows)
    write_json(root / "summary.json", summary)
    return summary


def run_model(model_name: str, batch_size: int) -> dict[str, Any]:
    natural = read_json(
        OUT_ROOT
        / "anchor_natural"
        / model_name
        / "summary.json"
    )
    domains = [
        domain
        for domain, summary in natural["domains"].items()
        if summary["natural_confirmation_pass"]
    ]
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(prereg["source_depths"][model_name])
    frozen_layers = prereg["frozen_phase1002_value_layers"][
        model_name
    ]["layer_numbers"]
    model = tokenizer = None
    started = time.time()
    summaries = {}
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=True
        )
        layers = get_layers(model)
        for domain in domains:
            summaries[domain] = run_domain(
                model,
                layers,
                device,
                model_name,
                domain,
                source_depth,
                frozen_layers,
                batch_size,
            )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": "phase1003_kv_replication_model.v1",
        "phase": PHASE,
        "model": model_name,
        "status": "complete",
        "domains": summaries,
        "pass_count": sum(
            summary["crossparadigm_kv_pass"]
            for summary in summaries.values()
        ),
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        OUT_ROOT / "kv_replication" / model_name / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def aggregate() -> dict[str, Any]:
    summaries = {}
    for model_name in MODELS:
        path = (
            OUT_ROOT
            / "kv_replication"
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
                item["crossparadigm_kv_pass"]
                for item in available
            ),
        }
    payload = {
        "schema_version": "phase1003_kv_replication_aggregate.v1",
        "phase": PHASE,
        "models": summaries,
        "all_models_complete": len(summaries) == len(MODELS),
        "cross_domain": cross_domain,
    }
    write_json(
        OUT_ROOT / "kv_replication" / "summary.json", payload
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
