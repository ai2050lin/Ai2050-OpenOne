#!/usr/bin/env python3
"""Run full short generation for confirmed Phase579 local causal branches."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase578_natural_trace import projected_states, span_positions  # noqa: E402
from phase579_option_routing_causal import (  # noqa: E402
    attention_mask_row,
    normalized_weights,
    replace_attention_output,
    swap_group_means,
    swap_group_value_means,
    swap_group_weight_mass,
)
import phase577_natural_choice_behavior as behavior  # noqa: E402
import phase577_natural_choice_protocol as choice  # noqa: E402
import phase578_choice_world_protocol as source  # noqa: E402
import phase579_option_routing_generation_protocol as generation_protocol  # noqa: E402


PATCH_CONDITIONS = {
    "option_score_swap",
    "object_relation_score_swap_control",
    "option_weight_swap",
    "option_value_swap_positive_control",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths(model: str) -> dict[str, Path]:
    stem = source.OUT_DIR / f"phase579_{model}_option_routing_generation"
    return {
        "rows": stem.with_name(stem.name + "_rows.jsonl.gz"),
        "summary": stem.with_name(stem.name + "_summary.json"),
        "contract": stem.with_name(stem.name + "_contract.json"),
    }


def load_worlds(model: str) -> list[tuple[str, list[dict[str, Any]]]]:
    frozen = read_json(generation_protocol.PROTOCOL_PATH)
    selected = list(frozen["world_ids_by_model"][model])
    allowed = set(frozen["confirmed_relations_by_model"][model])
    selected_set = set(selected)
    bank: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(source.SOURCE_CASES_PATH):
        if row["world_id"] not in selected_set:
            continue
        if row["split"] != "causal_confirmation" or row["sealed"]:
            raise RuntimeError("Phase579 generation attempted forbidden rows")
        if row["relation"] not in allowed:
            raise RuntimeError("Phase579 generation relation drift")
        bank[row["world_id"]][row["variant"]] = row
    worlds = []
    for world_id in selected:
        variants = bank.get(world_id, {})
        if set(variants) != set(choice.VARIANTS):
            raise RuntimeError(f"Phase579 incomplete generation world: {world_id}")
        worlds.append(
            (world_id, [variants[variant] for variant in choice.VARIANTS])
        )
    return worlds


def prepare_left_batch(
    loaded: Any,
    model: str,
    worlds: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    rows = [row for _, variants in worlds for row in variants]
    prompts = [render_chat(loaded.tokenizer, model, row["raw_prompt"]) for row in rows]
    unpadded = [
        span_positions(loaded.tokenizer, prompt, row)
        for prompt, row in zip(prompts, rows, strict=True)
    ]
    loaded.tokenizer.padding_side = "left"
    encoded = loaded.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    width = int(encoded["input_ids"].shape[1])
    metadata = []
    for batch_index, (row, item) in enumerate(zip(rows, unpadded, strict=True)):
        active = encoded["input_ids"][batch_index][
            encoded["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active] != item["ids"]:
            raise RuntimeError("Phase579 generation tokenization drift")
        offset = width - len(item["ids"])
        positions = {
            key: [int(position) + offset for position in values]
            for key, values in item.items()
            if key != "ids"
        }
        metadata.append(
            {
                "batch_index": batch_index,
                "row": row,
                "positions": positions,
            }
        )
    position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    encoded["position_ids"] = position_ids
    return encoded, metadata


def generate_condition(
    loaded: Any,
    layers: list[Any],
    encoded_cpu: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    selected_layers: dict[str, int],
    condition: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    patch_calls = 0
    intervention_audit: dict[str, Any] = {
        "matched_case_count": 0,
        "modified_case_count": 0,
        "weight_changed_case_count": 0,
        "projected_output_changed_case_count": 0,
        "max_weight_delta": 0.0,
        "min_weight_delta": None,
        "max_projected_output_delta": 0.0,
        "min_projected_output_delta": None,
    }

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            nonlocal patch_calls
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or int(hidden.shape[1]) <= 1:
                return None
            position_embeddings = kwargs.get("position_embeddings")
            attention_mask = kwargs.get("attention_mask")
            if (
                position_embeddings is None
                or not isinstance(output, tuple)
                or output[1] is None
            ):
                raise RuntimeError("Phase579 generation requires eager attention")
            query, key, value = projected_states(module, hidden, position_embeddings)
            raw_scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling
            primary = output[0].clone()
            weights = output[1].clone()
            values = (
                value.clone()
                if condition == "option_value_swap_positive_control"
                else value
            )
            modified = condition in PATCH_CONDITIONS
            local_count = 0
            for item in metadata:
                row = item["row"]
                if selected_layers[row["relation"]] != layer_index:
                    continue
                batch_index = int(item["batch_index"])
                positions = item["positions"]
                receiver = int(positions["answer_boundary"][-1])
                target_positions = positions["target_option"]
                foil_positions = positions["foil_option"]
                score_row = raw_scores[batch_index, :, receiver, :].clone()
                weight_row = weights[batch_index, :, receiver, :].clone()
                natural_weight_row = weight_row.clone()
                natural_primary = primary[batch_index, receiver, :].clone()
                value_row = values[batch_index]
                if condition == "option_score_swap":
                    score_row = swap_group_means(
                        score_row, target_positions, foil_positions
                    )
                    weight_row = normalized_weights(
                        score_row,
                        attention_mask_row(attention_mask, batch_index, receiver),
                    )
                elif condition == "object_relation_score_swap_control":
                    score_row = swap_group_means(
                        score_row, positions["object"], positions["relation"]
                    )
                    weight_row = normalized_weights(
                        score_row,
                        attention_mask_row(attention_mask, batch_index, receiver),
                    )
                elif condition == "option_weight_swap":
                    weight_row = swap_group_weight_mass(
                        weight_row, target_positions, foil_positions
                    )
                elif condition == "option_value_swap_positive_control":
                    value_row = swap_group_value_means(
                        value_row, target_positions, foil_positions
                    )
                    values[batch_index] = value_row
                elif condition not in {
                    "natural_baseline",
                    "option_score_swap_restore",
                }:
                    raise RuntimeError(
                        f"Unknown Phase579 generation condition: {condition}"
                    )
                if modified:
                    head_output = torch.einsum("hs,hsd->hd", weight_row, value_row)
                    projected = module.o_proj(head_output.reshape(1, -1)).squeeze(0)
                    weight_delta = float(
                        (weight_row - natural_weight_row).abs().max().item()
                    )
                    projected_output_delta = float(
                        (projected - natural_primary).abs().max().item()
                    )
                    intervention_audit["modified_case_count"] += 1
                    intervention_audit["weight_changed_case_count"] += int(
                        weight_delta > 1e-7
                    )
                    intervention_audit[
                        "projected_output_changed_case_count"
                    ] += int(projected_output_delta > 1e-7)
                    intervention_audit["max_weight_delta"] = max(
                        intervention_audit["max_weight_delta"], weight_delta
                    )
                    intervention_audit["max_projected_output_delta"] = max(
                        intervention_audit["max_projected_output_delta"],
                        projected_output_delta,
                    )
                    for key, value in (
                        ("min_weight_delta", weight_delta),
                        ("min_projected_output_delta", projected_output_delta),
                    ):
                        current = intervention_audit[key]
                        intervention_audit[key] = (
                            value if current is None else min(current, value)
                        )
                    primary[batch_index, receiver, :] = projected
                    weights[batch_index, :, receiver, :] = weight_row
                local_count += 1
            patch_calls += local_count
            intervention_audit["matched_case_count"] += local_count
            if modified:
                return replace_attention_output(output, primary, weights)
            return None

        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index in sorted(set(selected_layers.values()))
    ]
    try:
        with torch.inference_mode():
            generated = loaded.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=False,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    if patch_calls != len(metadata):
        raise RuntimeError(
            f"Phase579 generation patch count drift for {condition}: "
            f"{patch_calls}/{len(metadata)}"
        )
    outputs = []
    for item in metadata:
        batch_index = int(item["batch_index"])
        row = item["row"]
        text = loaded.tokenizer.decode(
            generated[batch_index, prompt_width:], skip_special_tokens=True
        )
        outputs.append({**row, **behavior.classify(row, text)})
    del encoded, generated
    return outputs, intervention_audit


def run(model: str, restart: bool) -> Path:
    frozen = read_json(generation_protocol.PROTOCOL_PATH)
    if model not in frozen["confirmed_relations_by_model"]:
        raise RuntimeError(f"Phase579 generation is not authorized for {model}")
    if frozen["source_cases_sha256"] != sha256_file(source.SOURCE_CASES_PATH):
        raise RuntimeError("Phase579 generation source hash drift")
    if not torch.cuda.is_available():
        raise RuntimeError("Phase579 full generation requires CUDA")
    output = paths(model)
    if restart:
        for path in output.values():
            path.unlink(missing_ok=True)
    worlds = load_worlds(model)
    selected_layers = {
        relation: int(item["layer"])
        for relation, item in frozen[
            "selected_coordinates_by_model_and_relation"
        ][model].items()
    }
    write_json(
        output["contract"],
        {
            "schema_version": "phase579_option_routing_generation_contract.v1",
            "phase_id": frozen["phase_id"],
            "created_at": now(),
            "model": model,
            "world_count": len(worlds),
            "world_ids": [world_id for world_id, _ in worlds],
            "relations": frozen["confirmed_relations_by_model"][model],
            "selected_layers_by_relation": selected_layers,
            "conditions": frozen["conditions"],
            "repeats": frozen["repeats"],
            "protocol_sha256": sha256_file(generation_protocol.PROTOCOL_PATH),
            "sealed_split_read": False,
        },
    )
    loaded = None
    rows_out: list[dict[str, Any]] = []
    intervention_audits: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    started = time.monotonic()
    try:
        loaded = load_probe_model(model)
        if loaded.input_device.type != "cuda":
            raise RuntimeError(
                f"Phase579 generation requires CUDA, got {loaded.input_device}"
            )
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != frozen["execution"]["torch_dtype"]:
            raise RuntimeError(f"Phase579 generation dtype drift: {dtype}")
        loaded.model.config._attn_implementation = "eager"
        layers = get_layers(loaded.model)
        batch_size = int(frozen["execution"]["batch_size"])
        max_new_tokens = int(frozen["execution"]["max_new_tokens"])
        for repeat in frozen["repeats"]:
            for condition in frozen["conditions"]:
                for start in range(0, len(worlds), batch_size):
                    batch = worlds[start : start + batch_size]
                    encoded, metadata = prepare_left_batch(loaded, model, batch)
                    outputs, batch_audit = generate_condition(
                        loaded,
                        layers,
                        encoded,
                        metadata,
                        selected_layers,
                        condition,
                        max_new_tokens,
                    )
                    aggregate = intervention_audits[repeat].setdefault(
                        condition,
                        {
                            "matched_case_count": 0,
                            "modified_case_count": 0,
                            "weight_changed_case_count": 0,
                            "projected_output_changed_case_count": 0,
                            "max_weight_delta": 0.0,
                            "min_weight_delta": None,
                            "max_projected_output_delta": 0.0,
                            "min_projected_output_delta": None,
                        },
                    )
                    for key in (
                        "matched_case_count",
                        "modified_case_count",
                        "weight_changed_case_count",
                        "projected_output_changed_case_count",
                    ):
                        aggregate[key] += int(batch_audit[key])
                    for key in ("max_weight_delta", "max_projected_output_delta"):
                        aggregate[key] = max(
                            float(aggregate[key]), float(batch_audit[key])
                        )
                    for key in ("min_weight_delta", "min_projected_output_delta"):
                        value = batch_audit[key]
                        if value is None:
                            continue
                        current = aggregate[key]
                        aggregate[key] = (
                            float(value)
                            if current is None
                            else min(float(current), float(value))
                        )
                    for row in outputs:
                        rows_out.append(
                            {
                                **row,
                                "schema_version": (
                                    "phase579_option_routing_generation_row.v1"
                                ),
                                "phase_id": frozen["phase_id"],
                                "created_at": now(),
                                "model": model,
                                "condition": condition,
                                "execution_repeat": repeat,
                                "selected_layer": selected_layers[row["relation"]],
                                "causal": condition in PATCH_CONDITIONS,
                                "direct_restore": condition
                                == "option_score_swap_restore",
                                "sealed": False,
                            }
                        )
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase579 generation "
                    f"{repeat}/{condition} {len(worlds)}/{len(worlds)}",
                    flush=True,
                )
        write_jsonl(output["rows"], rows_out)
        summary = {
            "schema_version": "phase579_option_routing_generation_summary.v2",
            "phase_id": frozen["phase_id"],
            "created_at": now(),
            "status": "complete",
            "model": model,
            "device_type": loaded.input_device.type,
            "torch_dtype": dtype,
            "world_count": len(worlds),
            "case_count_per_condition_repeat": len(worlds) * len(choice.VARIANTS),
            "condition_count": len(frozen["conditions"]),
            "repeat_count": len(frozen["repeats"]),
            "row_count": len(rows_out),
            "semantic_event_counts": dict(
                Counter(row["semantic_event"] for row in rows_out)
            ),
            "intervention_audits": intervention_audits,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(output["rows"]),
            "sealed_split_read": False,
        }
        write_json(output["summary"], summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return output["summary"]
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=source.MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.restart)


if __name__ == "__main__":
    main()
