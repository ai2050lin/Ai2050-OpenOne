#!/usr/bin/env python3
"""Qualify two real-model collection paths on 55 sealed formal cases.

For every case this script compares:

1. a direct full forward with native hidden-state output;
2. an independently hooked full forward with a component ledger;
3. a two-chunk KV-cache continuation against the full prompt;
4. layer-checkpoint replay of the unchanged complete all-position state;
5. native greedy generation against an explicit incremental greedy loop.

The stage is an instrument gate.  Correct target behavior is recorded on a
separate axis and never participates in collector equivalence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import GenerationConfig


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase333_dynamic_survey import continuation_ids  # noqa: E402
from phase416_dual_track_case_bank import (  # noqa: E402
    MODELS,
    OUT,
    PHASE_ID,
    SCHEMA_VERSION,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)


CASES = OUT / "phase416_registered_cases.jsonl"
PROTOCOL = OUT / "phase416_dual_track_protocol.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def component_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Unsupported component output: {type(value).__name__}")


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().item())


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(left.detach().float() - right.detach().float())
    scale = torch.linalg.vector_norm(left.detach().float()).clamp_min(1e-8)
    return float((difference / scale).item())


def js_divergence(left_logits: torch.Tensor, right_logits: torch.Tensor) -> float:
    left = torch.softmax(left_logits.detach().float(), dim=-1)
    right = torch.softmax(right_logits.detach().float(), dim=-1)
    midpoint = (left + right) * 0.5
    epsilon = 1e-12
    value = 0.5 * (
        torch.sum(left * (torch.log(left + epsilon) - torch.log(midpoint + epsilon)))
        + torch.sum(right * (torch.log(right + epsilon) - torch.log(midpoint + epsilon)))
    )
    return float(max(0.0, value.item()))


def normalize_text(value: str) -> str:
    value = value.lower().replace("</think>", " ")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def target_match(text: str, aliases: list[str]) -> bool:
    normalized = f" {normalize_text(text)} "
    return any(f" {normalize_text(alias)} " in normalized for alias in aliases if normalize_text(alias))


def exact_answer(text: str, aliases: list[str]) -> bool:
    normalized = normalize_text(text)
    return any(normalized == normalize_text(alias) for alias in aliases)


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    values: list[Any] = [tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)]
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            result.add(int(value))
        else:
            result.update(int(item) for item in value)
    return result


def encode_case(loaded: Any, case: dict[str, Any]) -> dict[str, torch.Tensor]:
    encoded = loaded.tokenizer(
        case["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=256,
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
    )
    return {key: value.to(loaded.input_device) for key, value in encoded.items()}


def cache_tensors(cache: Any) -> list[tuple[str, torch.Tensor]]:
    rows: list[tuple[str, torch.Tensor]] = []
    if cache is None:
        return rows
    if hasattr(cache, "layers"):
        for index, layer in enumerate(cache.layers):
            for name in ("keys", "values"):
                value = getattr(layer, name, None)
                if torch.is_tensor(value):
                    rows.append((f"L{index}:{name}", value))
        return rows
    if isinstance(cache, (tuple, list)):
        for layer_index, layer in enumerate(cache):
            if isinstance(layer, (tuple, list)):
                for tensor_index, value in enumerate(layer):
                    if torch.is_tensor(value):
                        rows.append((f"L{layer_index}:T{tensor_index}", value))
    return rows


def compare_caches(left: Any, right: Any) -> dict[str, Any]:
    left_rows = cache_tensors(left)
    right_rows = cache_tensors(right)
    if len(left_rows) != len(right_rows) or not left_rows:
        return {
            "tensor_count": min(len(left_rows), len(right_rows)),
            "shape_exact": False,
            "max_abs": math.inf,
            "max_relative_error": math.inf,
        }
    shape_exact = True
    max_difference = 0.0
    max_relative = 0.0
    for (left_name, left_value), (right_name, right_value) in zip(left_rows, right_rows, strict=True):
        shape_exact = shape_exact and left_name == right_name and left_value.shape == right_value.shape
        if left_value.shape != right_value.shape:
            continue
        max_difference = max(max_difference, max_abs(left_value, right_value))
        max_relative = max(max_relative, relative_error(left_value, right_value))
    return {
        "tensor_count": len(left_rows),
        "shape_exact": shape_exact,
        "max_abs": max_difference,
        "max_relative_error": max_relative,
    }


def capture_direct_and_replays(loaded: Any, encoded: dict[str, torch.Tensor]) -> tuple[Any, list[dict[str, Any]]]:
    layers = get_layers(loaded.model)
    first_kwargs: dict[str, Any] = {}

    def capture_kwargs(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if not first_kwargs:
            first_kwargs.update(kwargs)

    handle = layers[0].register_forward_pre_hook(capture_kwargs, with_kwargs=True)
    try:
        direct = loaded.model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        handle.remove()

    replay_kwargs = {
        key: value for key, value in first_kwargs.items() if key not in {"output_hidden_states"}
    }
    layer_count = len(layers)
    checkpoints = sorted({0, layer_count // 3, (2 * layer_count) // 3, layer_count})
    direct_last = direct.logits[0, -1]
    replay_rows: list[dict[str, Any]] = []
    for completed_layer_count in checkpoints:
        if completed_layer_count == layer_count:
            hidden = direct.hidden_states[-1]
        else:
            hidden = direct.hidden_states[completed_layer_count]
            for layer in layers[completed_layer_count:]:
                hidden = component_tensor(layer(hidden, **replay_kwargs))
            hidden = loaded.model.model.norm(hidden)
        replay_logits = loaded.model.lm_head(hidden)[0, -1]
        replay_rows.append(
            {
                "completed_layer_count": completed_layer_count,
                "terminal_logit_max_abs": max_abs(direct_last, replay_logits),
                "terminal_js": js_divergence(direct_last, replay_logits),
            }
        )
    return direct, replay_rows


def hooked_forward(loaded: Any, encoded: dict[str, torch.Tensor], reference_hidden: tuple[torch.Tensor, ...]) -> tuple[Any, list[dict[str, Any]]]:
    layers = get_layers(loaded.model)
    transient: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    rows: list[dict[str, Any]] = []
    handles: list[Any] = []

    for layer_index, layer in enumerate(layers):
        def layer_pre(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                transient[index]["input"] = inputs[0]

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
            transient[index]["attention"] = component_tensor(output)

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
            transient[index]["mlp"] = component_tensor(output)

        def layer_post(_module: Any, _inputs: tuple[Any, ...], output: Any, index: int = layer_index) -> None:
            layer_output = component_tensor(output)
            values = transient[index]
            reconstructed = values["input"] + values["attention"] + values["mlp"]
            if index == len(layers) - 1:
                observed = loaded.model.model.norm(layer_output)
                expected = reference_hidden[-1]
            else:
                observed = layer_output
                expected = reference_hidden[index + 1]
            rows.append(
                {
                    "layer": index,
                    "hidden_max_abs": max_abs(expected, observed),
                    "hidden_relative_error": relative_error(expected, observed),
                    "component_ledger_relative_error": relative_error(layer_output, reconstructed),
                }
            )
            transient.pop(index, None)

        handles.extend(
            [
                layer.register_forward_pre_hook(layer_pre),
                layer.self_attn.register_forward_hook(attention_post),
                layer.mlp.register_forward_hook(mlp_post),
                layer.register_forward_hook(layer_post),
            ]
        )
    try:
        output = loaded.model(**encoded, use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    rows.sort(key=lambda row: row["layer"])
    return output, rows


def chunked_cache_forward(loaded: Any, encoded: dict[str, torch.Tensor]) -> Any:
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    width = int(input_ids.shape[1])
    split = max(1, width // 2)
    first = loaded.model(
        input_ids=input_ids[:, :split],
        attention_mask=attention_mask[:, :split],
        use_cache=True,
        return_dict=True,
    )
    second = loaded.model(
        input_ids=input_ids[:, split:],
        attention_mask=attention_mask,
        past_key_values=first.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    return second


def neutral_generation_config(loaded: Any) -> GenerationConfig:
    config = GenerationConfig.from_model_config(loaded.model.config)
    config.do_sample = False
    config.pad_token_id = loaded.tokenizer.pad_token_id
    config.eos_token_id = loaded.tokenizer.eos_token_id
    config.bos_token_id = loaded.tokenizer.bos_token_id
    return config


def manual_greedy(loaded: Any, encoded: dict[str, torch.Tensor], max_new_tokens: int) -> tuple[list[int], list[torch.Tensor]]:
    current_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(current_ids))
    past = None
    generated: list[int] = []
    scores: list[torch.Tensor] = []
    stops = eos_ids(loaded.tokenizer, loaded.model)
    for _step in range(max_new_tokens):
        output = loaded.model(
            input_ids=current_ids,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        score = output.logits[0, -1].detach()
        token = int(torch.argmax(score).item())
        generated.append(token)
        scores.append(score)
        past = output.past_key_values
        if token in stops:
            break
        current_ids = torch.tensor([[token]], dtype=encoded["input_ids"].dtype, device=loaded.input_device)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
    return generated, scores


def compare_generation(loaded: Any, encoded: dict[str, torch.Tensor], case: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    reference = loaded.model.generate(
        **encoded,
        generation_config=neutral_generation_config(loaded),
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
    )
    prompt_length = int(encoded["input_ids"].shape[1])
    reference_ids = [int(value) for value in reference.sequences[0, prompt_length:].tolist()]
    manual_ids, manual_scores = manual_greedy(loaded, encoded, max_new_tokens)
    compared_steps = min(len(reference.scores), len(manual_scores))
    score_max_abs = max(
        (max_abs(reference.scores[index][0], manual_scores[index]) for index in range(compared_steps)),
        default=0.0,
    )
    text = loaded.tokenizer.decode(reference_ids, skip_special_tokens=True)
    stops = eos_ids(loaded.tokenizer, loaded.model)
    emitted_stop = any(token in stops for token in reference_ids)
    finite = all(torch.isfinite(score).all().item() for score in manual_scores)
    events = {
        "semantic_category": "target" if target_match(text, case["target_aliases"]) else "other",
        "format_complete": exact_answer(text, case["target_aliases"]),
        "sentence_boundary": emitted_stop or bool(re.search(r"[.!?]\s*$", text.strip())),
        "stop": emitted_stop,
        "numerical_validity": bool(finite),
        "response_role": "current_response",
        "right_censored": not emitted_stop and len(reference_ids) >= max_new_tokens,
    }
    return {
        "reference_token_ids": reference_ids,
        "manual_token_ids": manual_ids,
        "token_exact": reference_ids == manual_ids,
        "compared_score_step_count": compared_steps,
        "score_max_abs": score_max_abs,
        "generated_text": text,
        "events": events,
        "target_event_match": target_match(text, case["target_aliases"]),
        "exact_answer_match": exact_answer(text, case["target_aliases"]),
    }


def tokenizer_contract(loaded: Any, cases: list[dict[str, Any]]) -> dict[str, Any]:
    files = []
    digest = hashlib.sha256()
    for path in sorted(loaded.spec.local_dir.iterdir()):
        if not path.is_file() or path.suffix in {".safetensors", ".bin"}:
            continue
        if "token" not in path.name and path.name not in {
            "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json"
        }:
            continue
        value = path.read_bytes()
        file_hash = hashlib.sha256(value).hexdigest()
        digest.update(path.name.encode("utf-8"))
        digest.update(value)
        files.append({"filename": path.name, "sha256": file_hash})
    event_rows = []
    for case in cases:
        values = [case["target"], *case["distractors"]]
        event_rows.append(
            {
                "case_id": case["case_id"],
                "semantic_case_id": case["semantic_case_id"],
                "events": [
                    {"text": value, "token_ids": continuation_ids(loaded, case, value)}
                    for value in values
                ],
            }
        )
    return {
        "schema_version": "phase416_tokenizer_contract.v1",
        "phase_id": PHASE_ID,
        "created_at": now(),
        "model": loaded.key,
        "tokenizer_sha256": digest.hexdigest(),
        "files": files,
        "bos_token_id": loaded.tokenizer.bos_token_id,
        "eos_token_ids": sorted(eos_ids(loaded.tokenizer, loaded.model)),
        "pad_token_id": loaded.tokenizer.pad_token_id,
        "semantic_event_tokenizations": event_rows,
        "cross_model_token_id_identity_required": False,
    }


@torch.inference_mode()
def run_model(model_key: str, max_new_tokens: int, max_cases: int | None = None) -> dict[str, Any]:
    protocol = read_json(PROTOCOL)
    thresholds = protocol["collector_gates"]
    cases = [row for row in read_jsonl(CASES) if row["model"] == model_key]
    if max_cases is not None:
        cases = cases[:max_cases]
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        print(f"[Phase416] loading {model_key} for {len(cases)} collector cases", flush=True)
        loaded = load_probe_model(model_key)
        model_root = OUT / "models" / model_key
        write_json(model_root / "phase416_tokenizer_contract.json", tokenizer_contract(loaded, cases))
        for case_index, case in enumerate(cases, start=1):
            encoded = encode_case(loaded, case)
            direct, replay_rows = capture_direct_and_replays(loaded, encoded)
            hooked, layer_rows = hooked_forward(loaded, encoded, direct.hidden_states)
            direct_logits = direct.logits[0, -1]
            hooked_logits = hooked.logits[0, -1]
            chunked = chunked_cache_forward(loaded, encoded)
            cache_comparison = compare_caches(hooked.past_key_values, chunked.past_key_values)
            generation = compare_generation(loaded, encoded, case, max_new_tokens)

            direct_hook_logit = max_abs(direct_logits, hooked_logits)
            direct_hook_js = js_divergence(direct_logits, hooked_logits)
            layer_max_abs = max((row["hidden_max_abs"] for row in layer_rows), default=math.inf)
            ledger_max_relative = max(
                (row["component_ledger_relative_error"] for row in layer_rows), default=math.inf
            )
            chunk_logit_max_abs = max_abs(hooked_logits, chunked.logits[0, -1])
            chunk_terminal_js = js_divergence(hooked_logits, chunked.logits[0, -1])
            chunk_top1_exact = bool(
                torch.argmax(hooked_logits).item() == torch.argmax(chunked.logits[0, -1]).item()
            )
            replay_logit_max_abs = max(row["terminal_logit_max_abs"] for row in replay_rows)
            replay_js = max(row["terminal_js"] for row in replay_rows)
            gates = {
                "direct_hook_logit_pass": direct_hook_logit <= thresholds["direct_vs_hook_terminal_logit_max_abs"],
                "direct_hook_js_pass": direct_hook_js <= thresholds["direct_vs_hook_terminal_js"],
                "layer_output_pass": layer_max_abs <= thresholds["layer_output_max_abs"],
                "component_ledger_pass": ledger_max_relative <= thresholds["component_ledger_relative_error"],
                "chunked_cache_logit_pass": chunk_logit_max_abs <= thresholds["chunked_cache_terminal_logit_max_abs"],
                "chunked_cache_js_pass": chunk_terminal_js <= thresholds["chunked_cache_terminal_js"],
                "chunked_cache_top1_pass": chunk_top1_exact,
                "chunked_cache_shape_pass": cache_comparison["shape_exact"],
                "chunked_cache_value_pass": cache_comparison["max_relative_error"] <= thresholds["chunked_cache_relative_error"],
                "checkpoint_replay_logit_pass": replay_logit_max_abs <= thresholds["checkpoint_replay_terminal_logit_max_abs"],
                "checkpoint_replay_js_pass": replay_js <= thresholds["checkpoint_replay_terminal_js"],
                "greedy_token_pass": generation["token_exact"],
                "greedy_score_pass": generation["score_max_abs"] <= thresholds["greedy_generation_score_max_abs"],
                "finite_generation_pass": generation["events"]["numerical_validity"],
            }
            row = {
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE_ID,
                "created_at": now(),
                "model": model_key,
                "case_id": case["case_id"],
                "semantic_case_id": case["semantic_case_id"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "split": case["split"],
                "template_id": case["template_id"],
                "prompt_sha256": case["prompt_sha256"],
                "prompt_token_count": int(encoded["input_ids"].shape[1]),
                "layer_count": len(layer_rows),
                "direct_hook_terminal_logit_max_abs": direct_hook_logit,
                "direct_hook_terminal_js": direct_hook_js,
                "layer_output_max_abs": layer_max_abs,
                "component_ledger_max_relative_error": ledger_max_relative,
                "chunked_cache_terminal_logit_max_abs": chunk_logit_max_abs,
                "chunked_cache_terminal_js": chunk_terminal_js,
                "chunked_cache_terminal_top1_exact": chunk_top1_exact,
                "chunked_cache": cache_comparison,
                "checkpoint_replays": replay_rows,
                "checkpoint_replay_terminal_logit_max_abs": replay_logit_max_abs,
                "checkpoint_replay_terminal_js": replay_js,
                "generation": generation,
                "gates": gates,
                "collector_case_pass": all(gates.values()),
                "target_behavior_pass": generation["target_event_match"],
                "exact_answer_pass": generation["exact_answer_match"],
                "instrument_result_not_mechanism_evidence": True,
                "causal": False,
            }
            rows.append(row)
            del direct, hooked, chunked, encoded
            gc.collect()
            if case_index % 5 == 0 or case_index == len(cases):
                print(
                    f"[Phase416:{model_key}] {case_index}/{len(cases)} "
                    f"collector_pass={sum(item['collector_case_pass'] for item in rows)}",
                    flush=True,
                )

        family_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            family_buckets[row["family_id"]].append(row)
        behavior_cells = []
        behavior_gate = protocol["behavior_gate"]
        for family, values in sorted(family_buckets.items()):
            target_rate = sum(row["target_behavior_pass"] for row in values) / len(values)
            exact_rate = sum(row["exact_answer_pass"] for row in values) / len(values)
            behavior_cells.append(
                {
                    "model": model_key,
                    "family_id": family,
                    "case_count": len(values),
                    "collector_pass_count": sum(row["collector_case_pass"] for row in values),
                    "target_event_match_rate": target_rate,
                    "exact_answer_rate": exact_rate,
                    "formal_behavior_qualified": bool(
                        len(values) >= behavior_gate["minimum_family_case_count"]
                        and target_rate >= behavior_gate["minimum_target_event_match_rate"]
                    ),
                    "natural_language_external_validity": False,
                }
            )

        expected = 55 if max_cases is None else len(cases)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": "Phase416-RealCollectorQualification",
            "created_at": now(),
            "model": model_key,
            "case_count": len(rows),
            "required_case_count": expected,
            "collector_case_pass_count": sum(row["collector_case_pass"] for row in rows),
            "collector_case_failure_count": sum(not row["collector_case_pass"] for row in rows),
            "collector_qualification_pass": bool(
                len(rows) == expected and all(row["collector_case_pass"] for row in rows)
            ),
            "target_behavior_pass_count": sum(row["target_behavior_pass"] for row in rows),
            "exact_answer_pass_count": sum(row["exact_answer_pass"] for row in rows),
            "behavior_cells": behavior_cells,
            "qualified_formal_family_count": sum(row["formal_behavior_qualified"] for row in behavior_cells),
            "max_observed_errors": {
                "direct_hook_terminal_logit_max_abs": max(row["direct_hook_terminal_logit_max_abs"] for row in rows),
                "direct_hook_terminal_js": max(row["direct_hook_terminal_js"] for row in rows),
                "layer_output_max_abs": max(row["layer_output_max_abs"] for row in rows),
                "component_ledger_relative_error": max(row["component_ledger_max_relative_error"] for row in rows),
                "chunked_cache_terminal_logit_max_abs": max(row["chunked_cache_terminal_logit_max_abs"] for row in rows),
                "chunked_cache_terminal_js": max(row["chunked_cache_terminal_js"] for row in rows),
                "chunked_cache_relative_error": max(row["chunked_cache"]["max_relative_error"] for row in rows),
                "checkpoint_replay_terminal_logit_max_abs": max(row["checkpoint_replay_terminal_logit_max_abs"] for row in rows),
                "checkpoint_replay_terminal_js": max(row["checkpoint_replay_terminal_js"] for row in rows),
                "greedy_generation_score_max_abs": max(row["generation"]["score_max_abs"] for row in rows),
            },
            "vram_gb": vram_gb(),
            "raw_physical_collection_authorized": bool(
                len(rows) == expected and all(row["collector_case_pass"] for row in rows)
            ),
            "functional_labels_authorized_only_for_qualified_formal_families": True,
            "causal_intervention_authorized": False,
            "neuron_scan_authorized": False,
            "claim_boundary": "real_instrument_and_formal_behavior_qualification_not_language_mechanism_closure",
        }
        model_root = OUT / "models" / model_key
        write_jsonl(model_root / "phase416_collector_case_rows.jsonl", rows)
        write_json(model_root / "phase416_collector_complete.json", summary)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    summary = run_model(args.model, args.max_new_tokens, args.max_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["collector_qualification_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
