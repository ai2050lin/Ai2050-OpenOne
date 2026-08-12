#!/usr/bin/env python3
"""Decompose the cache-carried source effect into global key and value arms."""
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
from transformers import DynamicCache


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

import phase1000_scpg_discovery as scpg
from model_utils import get_layers, load_model, release_model
from phase1002_multitoken_frozen_topology import read_json
from phase1002_multitoken_protocol import (
    COLORS,
    MODELS,
    OUT_ROOT,
    write_json,
    write_jsonl,
)
from phase1002_multitoken_temporal_rollout import (
    batches,
    selected_directional_rows,
    step_case,
)


PHASE = 1002
PAIRS_PER_STRATUM = 4


def candidate_logits(
    logits: torch.Tensor, candidate_ids: dict[str, int]
) -> torch.Tensor:
    ids = torch.tensor(
        [candidate_ids[color] for color in COLORS],
        dtype=torch.long,
        device=logits.device,
    )
    return logits[:, ids].float().detach()


def predictions(logits: torch.Tensor) -> list[str]:
    indices = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [COLORS[int(index)] for index in indices]


def margin(
    logits: torch.Tensor, rows: list[dict[str, Any]]
) -> torch.Tensor:
    color_index = {color: index for index, color in enumerate(COLORS)}
    batch_index = torch.arange(len(rows), device=logits.device)
    source_index = torch.tensor(
        [color_index[row["source"]["gold"]] for row in rows],
        dtype=torch.long,
        device=logits.device,
    )
    target_index = torch.tensor(
        [color_index[row["target"]["gold"]] for row in rows],
        dtype=torch.long,
        device=logits.device,
    )
    return (
        logits[batch_index, source_index]
        - logits[batch_index, target_index]
    )


def clone_mixed_cache(
    key_cache,
    value_cache,
    model_config,
) -> DynamicCache:
    if not hasattr(key_cache, "layers") or not hasattr(
        value_cache, "layers"
    ):
        raise RuntimeError(
            f"unsupported cache classes: "
            f"{type(key_cache).__name__}/{type(value_cache).__name__}"
        )
    if len(key_cache.layers) != len(value_cache.layers):
        raise RuntimeError("cache layer count drift")
    data = []
    for key_layer, value_layer in zip(
        key_cache.layers, value_cache.layers
    ):
        data.append((
            key_layer.keys.detach().clone(),
            value_layer.values.detach().clone(),
        ))
    return DynamicCache(data, config=model_config)


def build_cache(
    model,
    layers,
    device,
    cases: list[dict[str, Any]],
    source_patch: dict[str, Any] | None,
):
    input_ids, attention = scpg.case_tensors(cases, device)
    handle = None
    try:
        handle, count = scpg.register_source_patch(
            layers, source_patch, full_width=input_ids.shape[1]
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=True,
                return_dict=True,
            )
        if source_patch is not None and count[0] != 1:
            raise RuntimeError(
                f"cache build source hook drift: {count[0]}"
            )
        return output.past_key_values
    finally:
        if handle is not None:
            handle.remove()


def continue_from_cache(
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
            return_dict=True,
        )
    return candidate_logits(output.logits[:, -1, :], candidate_ids)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for split in ("discovery", "confirmation"):
        values = [row for row in rows if row["split"] == split]
        result[split] = {
            "n": len(values),
            "target_cached_target_rate": float(np.mean([
                row["target_cached_prediction"] == row["target_gold"]
                for row in values
            ])),
            "source_do_cached_source_rate": float(np.mean([
                row["source_do_cached_prediction"] == row["source_gold"]
                for row in values
            ])),
            "source_keys_target_values_source_rate": float(np.mean([
                (
                    row["source_keys_target_values_prediction"]
                    == row["source_gold"]
                )
                for row in values
            ])),
            "target_keys_source_values_source_rate": float(np.mean([
                (
                    row["target_keys_source_values_prediction"]
                    == row["source_gold"]
                )
                for row in values
            ])),
            "median_total_cache_transfer": float(np.median([
                row["total_cache_transfer"] for row in values
            ])),
            "median_key_only_transfer": float(np.median([
                row["key_only_transfer"] for row in values
            ])),
            "median_value_only_transfer": float(np.median([
                row["value_only_transfer"] for row in values
            ])),
            "median_key_restore_mediation": float(np.median([
                row["key_restore_mediation"] for row in values
            ])),
            "median_value_restore_mediation": float(np.median([
                row["value_restore_mediation"] for row in values
            ])),
            "median_factorial_interaction": float(np.median([
                row["factorial_interaction"] for row in values
            ])),
            "target_cache_full_prediction_agreement": float(np.mean([
                row["target_cache_full_prediction_agreement"]
                for row in values
            ])),
            "source_do_cache_full_prediction_agreement": float(np.mean([
                row["source_do_cache_full_prediction_agreement"]
                for row in values
            ])),
            "target_cache_full_max_abs_difference": float(max(
                row["target_cache_full_max_abs_difference"]
                for row in values
            )),
            "source_do_cache_full_max_abs_difference": float(max(
                row["source_do_cache_full_max_abs_difference"]
                for row in values
            )),
        }
    return result


def run_model(
    model_name: str,
    batch_size: int,
    use_8bit: bool = True,
    pairs_per_stratum: int = PAIRS_PER_STRATUM,
) -> dict[str, Any]:
    behavior = read_json(
        OUT_ROOT / "behavior" / model_name / "summary.json"
    )
    if not behavior["behavior_gate_pass"]:
        raise RuntimeError(f"{model_name}: behavior gate failed")
    prereg = read_json(OUT_ROOT / "preregistered_protocol.json")
    source_depth = int(
        prereg["frozen_phase1001_topology"][model_name]["source_depth"]
    )
    model = tokenizer = None
    started = time.time()
    result_rows = []
    try:
        model, tokenizer, device = load_model(
            model_name, dtype=torch.bfloat16, use_8bit=use_8bit
        )
        layers = get_layers(model)
        for split in ("discovery", "confirmation"):
            split_rows = selected_directional_rows(
                model_name, split, pairs_per_stratum
            )
            split_batches = list(batches(split_rows, batch_size))
            for batch_number, batch in enumerate(split_batches, 1):
                source_cases = [row["source"] for row in batch]
                target_cases = [row["target"] for row in batch]
                candidate_ids = target_cases[0]["candidate_token_ids"]
                semantic_step = int(target_cases[0]["semantic_step"])
                cache_prefix_step = semantic_step - 1
                if cache_prefix_step < 0:
                    raise RuntimeError("semantic step has no current token")
                target_prefix_cases = [
                    step_case(row["target"], cache_prefix_step)
                    for row in batch
                ]
                semantic_cases = [
                    step_case(row["target"], semantic_step)
                    for row in batch
                ]
                source_semantic_cases = [
                    step_case(row["source"], semantic_step)
                    for row in batch
                ]
                current_ids = [
                    int(row["target"]["answer_token_ids"][
                        cache_prefix_step
                    ])
                    for row in batch
                ]
                _, source_residuals = scpg.capture_residuals(
                    model,
                    device,
                    source_cases,
                    (source_depth,),
                    candidate_ids,
                )
                source_vectors = source_residuals[source_depth]
                prefix_patch = scpg.source_patch_spec(
                    source_depth,
                    target_prefix_cases,
                    source_vectors,
                    "joint",
                )
                semantic_patch = scpg.source_patch_spec(
                    source_depth,
                    semantic_cases,
                    source_vectors,
                    "joint",
                )

                target_cache = build_cache(
                    model,
                    layers,
                    device,
                    target_prefix_cases,
                    None,
                )
                source_do_cache = build_cache(
                    model,
                    layers,
                    device,
                    target_prefix_cases,
                    prefix_patch,
                )
                prefix_length = int(
                    target_prefix_cases[0]["input_token_count"]
                )
                target_cached = continue_from_cache(
                    model,
                    device,
                    current_ids,
                    prefix_length,
                    clone_mixed_cache(
                        target_cache, target_cache, model.config
                    ),
                    candidate_ids,
                )
                source_do_cached = continue_from_cache(
                    model,
                    device,
                    current_ids,
                    prefix_length,
                    clone_mixed_cache(
                        source_do_cache, source_do_cache, model.config
                    ),
                    candidate_ids,
                )
                source_keys_target_values = continue_from_cache(
                    model,
                    device,
                    current_ids,
                    prefix_length,
                    clone_mixed_cache(
                        source_do_cache, target_cache, model.config
                    ),
                    candidate_ids,
                )
                target_keys_source_values = continue_from_cache(
                    model,
                    device,
                    current_ids,
                    prefix_length,
                    clone_mixed_cache(
                        target_cache, source_do_cache, model.config
                    ),
                    candidate_ids,
                )

                target_full = scpg.forward_candidate(
                    model,
                    layers,
                    device,
                    semantic_cases,
                    candidate_ids,
                )
                source_do_full = scpg.forward_candidate(
                    model,
                    layers,
                    device,
                    semantic_cases,
                    candidate_ids,
                    source_patch=semantic_patch,
                )
                source_full = scpg.forward_candidate(
                    model,
                    layers,
                    device,
                    source_semantic_cases,
                    candidate_ids,
                )

                source_margin = margin(source_full, batch)
                target_margin = margin(target_cached, batch)
                do_margin = margin(source_do_cached, batch)
                key_margin = margin(source_keys_target_values, batch)
                value_margin = margin(target_keys_source_values, batch)
                target_predictions = predictions(target_cached)
                do_predictions = predictions(source_do_cached)
                key_predictions = predictions(
                    source_keys_target_values
                )
                value_predictions = predictions(
                    target_keys_source_values
                )
                target_full_predictions = predictions(target_full)
                do_full_predictions = predictions(source_do_full)

                for index, item in enumerate(batch):
                    source_effect = float(
                        do_margin[index] - target_margin[index]
                    )
                    key_effect = float(
                        key_margin[index] - target_margin[index]
                    )
                    value_effect = float(
                        value_margin[index] - target_margin[index]
                    )
                    clean_span = float(
                        source_margin[index] - target_margin[index]
                    )
                    result_rows.append({
                        "schema_version": (
                            "phase1002_kv_cache_decomposition_row.v1"
                        ),
                        "phase": PHASE,
                        "model": model_name,
                        "split": split,
                        "pair_id": item["pair_id"],
                        "direction": item["direction"],
                        "template": item["template"],
                        "source_gold": item["source"]["gold"],
                        "target_gold": item["target"]["gold"],
                        "target_cached_prediction": (
                            target_predictions[index]
                        ),
                        "source_do_cached_prediction": (
                            do_predictions[index]
                        ),
                        "source_keys_target_values_prediction": (
                            key_predictions[index]
                        ),
                        "target_keys_source_values_prediction": (
                            value_predictions[index]
                        ),
                        "target_full_prediction": (
                            target_full_predictions[index]
                        ),
                        "source_do_full_prediction": (
                            do_full_predictions[index]
                        ),
                        "target_margin": float(target_margin[index]),
                        "source_do_margin": float(do_margin[index]),
                        "source_keys_target_values_margin": float(
                            key_margin[index]
                        ),
                        "target_keys_source_values_margin": float(
                            value_margin[index]
                        ),
                        "total_cache_transfer": (
                            source_effect / max(abs(clean_span), 1e-8)
                        ),
                        "key_only_transfer": (
                            key_effect / max(abs(clean_span), 1e-8)
                        ),
                        "value_only_transfer": (
                            value_effect / max(abs(clean_span), 1e-8)
                        ),
                        "key_restore_mediation": (
                            (source_effect - value_effect)
                            / max(abs(source_effect), 1e-8)
                        ),
                        "value_restore_mediation": (
                            (source_effect - key_effect)
                            / max(abs(source_effect), 1e-8)
                        ),
                        "factorial_interaction": (
                            source_effect - key_effect - value_effect
                        ),
                        "target_cache_full_prediction_agreement": (
                            target_predictions[index]
                            == target_full_predictions[index]
                        ),
                        "source_do_cache_full_prediction_agreement": (
                            do_predictions[index]
                            == do_full_predictions[index]
                        ),
                        "target_cache_full_max_abs_difference": float(
                            torch.max(torch.abs(
                                target_cached[index] - target_full[index]
                            ))
                        ),
                        "source_do_cache_full_max_abs_difference": float(
                            torch.max(torch.abs(
                                source_do_cached[index]
                                - source_do_full[index]
                            ))
                        ),
                    })
                del (
                    source_residuals,
                    target_cache,
                    source_do_cache,
                    target_cached,
                    source_do_cached,
                    source_keys_target_values,
                    target_keys_source_values,
                    target_full,
                    source_do_full,
                    source_full,
                )
                print(
                    f"[{model_name}/{split}] "
                    f"{batch_number}/{len(split_batches)}",
                    flush=True,
                )
    finally:
        if model is not None:
            release_model(model)
        gc.collect()

    result_name = (
        model_name if use_8bit else f"{model_name}_bf16"
    )
    model_root = OUT_ROOT / "kv_cache_decomposition" / result_name
    write_jsonl(model_root / "rows.jsonl", result_rows)
    split_summary = summarize(result_rows)
    checks = {
        split: {
            "source_do": (
                split_summary[split]["source_do_cached_source_rate"]
                >= prereg["primary_thresholds"][
                    "source_do_semantic_flip_rate"
                ]
            ),
            "target_cache_instrument": (
                split_summary[split][
                    "target_cache_full_prediction_agreement"
                ] >= 0.99
            ),
            "source_do_cache_instrument": (
                split_summary[split][
                    "source_do_cache_full_prediction_agreement"
                ] >= 0.99
            ),
        }
        for split in ("discovery", "confirmation")
    }
    summary = {
        "schema_version": "phase1002_kv_cache_decomposition_summary.v1",
        "phase": PHASE,
        "model": model_name,
        "precision": "8bit" if use_8bit else "bf16",
        "pairs_per_stratum": pairs_per_stratum,
        "status": "complete",
        "source_depth": source_depth,
        "semantic_step": prereg["protocol_audits"][model_name][
            "semantic_step"
        ],
        "split_summary": split_summary,
        "checks": checks,
        "cache_transport_pass": all(
            all(values.values()) for values in checks.values()
        ),
        "elapsed_seconds": time.time() - started,
        "claim_boundary": (
            "The key/value mix is global across all cache layers. It "
            "identifies key-arm, value-arm, and interaction contributions "
            "to this local task, not a per-head language mechanism."
        ),
    }
    write_json(model_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def aggregate() -> dict[str, Any]:
    summaries = {
        model_name: read_json(
            OUT_ROOT
            / "kv_cache_decomposition"
            / model_name
            / "summary.json"
        )
        for model_name in MODELS
        if (
            OUT_ROOT
            / "kv_cache_decomposition"
            / model_name
            / "summary.json"
        ).exists()
    }
    payload = {
        "schema_version": "phase1002_kv_cache_cross_model.v1",
        "phase": PHASE,
        "models": summaries,
        "pass_count": sum(
            summary["cache_transport_pass"]
            for summary in summaries.values()
        ),
        "cross_model_pass": (
            len(summaries) == len(MODELS)
            and sum(
                summary["cache_transport_pass"]
                for summary in summaries.values()
            ) >= 2
        ),
    }
    write_json(
        OUT_ROOT / "kv_cache_decomposition" / "summary.json",
        payload,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--pairs-per-stratum",
        type=int,
        choices=(1, 2, 3, 4),
        default=PAIRS_PER_STRATUM,
    )
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
    elif args.model:
        if args.bf16 and args.model != "qwen3":
            raise SystemExit("bf16 audit is limited to qwen3 on this GPU")
        run_model(
            args.model,
            args.batch_size,
            use_8bit=not args.bf16,
            pairs_per_stratum=args.pairs_per_stratum,
        )
    else:
        raise SystemExit("provide --model or --aggregate")


if __name__ == "__main__":
    main()
