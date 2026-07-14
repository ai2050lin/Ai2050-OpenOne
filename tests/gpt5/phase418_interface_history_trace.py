#!/usr/bin/env python3
"""Collect Phase418 paired interface/history behavior and prefill vectors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded, vram_gb  # noqa: E402
from phase416_dual_track_case_bank import read_jsonl, write_json, write_jsonl  # noqa: E402
from phase416_real_collector_qualification import (  # noqa: E402
    component_tensor,
    eos_ids,
    exact_answer,
    neutral_generation_config,
    target_match,
)
from phase418_interface_history_case_bank import (  # noqa: E402
    HISTORIES,
    INTERFACES,
    MODELS,
    OUT,
    SCHEMA_VERSION,
)


PHASE_ID = "Phase418-InterfaceHistoryPhysicalTrace"
REGISTERED = OUT / "phase418_registered_conditions.jsonl"
SHARED_SUFFIX = "\nFinal answer:"
SHARED_TERMINAL_LITERAL = "Final answer:"
CORE_COMPONENTS = (
    "layer_input",
    "attention_output",
    "mlp_output",
    "residual_increment",
    "layer_output",
)
LEDGER_THRESHOLD = 1e-5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_rows(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm(left.detach().float() - right.detach().float())
    scale = torch.linalg.vector_norm(left.detach().float()).clamp_min(1e-8)
    return float((difference / scale).item())


def history_exchange(row: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    history = row["history_condition"]
    current = row["raw_prompt"]
    if history == "none":
        return [], current
    if history == "compatible":
        exchange = [
            {"role": "user", "content": "Earlier draft answer for this same task:"},
            {"role": "assistant", "content": row["target"]},
        ]
    elif history == "irrelevant":
        exchange = [
            {"role": "user", "content": "Earlier note for a separate calibration task:"},
            {"role": "assistant", "content": row["distractors"][-1]},
        ]
    else:
        exchange = [
            {"role": "user", "content": "Earlier draft answer for this same task:"},
            {"role": "assistant", "content": row["distractors"][0]},
        ]
    if history == "override":
        current = "The earlier draft is obsolete. Follow the current task only.\n" + current
    return exchange, current


def serialize_prompt(tokenizer: Any, row: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    history, current = history_exchange(row)
    messages = [*history, {"role": "user", "content": current}]
    if row["interface"] == "chat":
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        chunks = []
        if history:
            chunks.extend(
                [
                    "Previous exchange:",
                    f"Question: {history[0]['content']}",
                    f"Answer: {history[1]['content']}",
                    "Current task:",
                    current,
                ]
            )
        else:
            chunks.extend(["Current task:", current])
        prompt = "\n".join(chunks)
    return prompt.rstrip() + SHARED_SUFFIX, messages


def encode_prompt(loaded: Any, prompt: str) -> dict[str, torch.Tensor]:
    encoded = loaded.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    return {key: value.to(loaded.input_device) for key, value in encoded.items()}


class PromptVectorCollector:
    def __init__(self, loaded: Any) -> None:
        self.loaded = loaded
        self.layers = get_layers(loaded.model)
        self.handles: list[Any] = []
        self.active = False
        self.call_index = -1
        self.call_widths: list[int] = []
        self.transient: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
        self.vectors: dict[tuple[int, str], torch.Tensor] = {}
        self.ledger_errors: list[float] = []

    def begin(self) -> None:
        self.active = True
        self.call_index = -1
        self.call_widths = []
        self.transient.clear()
        self.vectors = {}
        self.ledger_errors = []

    def end(self) -> None:
        self.active = False
        self.transient.clear()

    def save(self, layer: int, component: str, value: Any) -> None:
        if not self.active or self.call_index != 0:
            return
        tensor = component_tensor(value)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            return
        self.vectors[(layer, component)] = tensor[0, -1].detach().float().cpu()

    def install(self) -> None:
        for layer_index, layer in enumerate(self.layers):
            def layer_pre(_module: Any, inputs: tuple[Any, ...], index: int = layer_index) -> None:
                if not self.active or not inputs or not torch.is_tensor(inputs[0]):
                    return
                if index == 0:
                    self.call_index += 1
                    self.call_widths.append(int(inputs[0].shape[1]))
                if self.call_index == 0:
                    self.transient[index]["input"] = inputs[0]
                    self.save(index, "layer_input", inputs[0])

            def attention_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                if self.active and self.call_index == 0:
                    tensor = component_tensor(output)
                    self.transient[index]["attention"] = tensor
                    self.save(index, "attention_output", tensor)

            def mlp_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                if self.active and self.call_index == 0:
                    tensor = component_tensor(output)
                    self.transient[index]["mlp"] = tensor
                    self.save(index, "mlp_output", tensor)

            def layer_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                index: int = layer_index,
            ) -> None:
                if not self.active or self.call_index != 0:
                    return
                tensor = component_tensor(output)
                values = self.transient.pop(index)
                before = values["input"]
                reconstructed = before + values["attention"] + values["mlp"]
                self.ledger_errors.append(relative_error(tensor, reconstructed))
                self.save(index, "residual_increment", tensor - before)
                self.save(index, "layer_output", tensor)

            self.handles.extend(
                [
                    layer.register_forward_pre_hook(layer_pre),
                    layer.self_attn.register_forward_hook(attention_post),
                    layer.mlp.register_forward_hook(mlp_post),
                    layer.register_forward_hook(layer_post),
                ]
            )

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def physical_rows(
    row: dict[str, Any],
    vectors: dict[tuple[int, str], torch.Tensor],
    layer_count: int,
) -> list[dict[str, Any]]:
    result = []
    for layer in range(layer_count):
        input_norm = float(torch.linalg.vector_norm(vectors[(layer, "layer_input")]).item())
        for component in CORE_COMPONENTS:
            vector = vectors[(layer, component)]
            norm = float(torch.linalg.vector_norm(vector).item())
            rms = float(torch.sqrt(torch.mean(vector.square())).item())
            signed_mean = float(torch.mean(vector).item())
            max_abs = float(torch.max(vector.abs()).item())
            finite = bool(torch.isfinite(vector).all().item())
            if not finite:
                norm = rms = signed_mean = max_abs = None
            result.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE_ID,
                    "created_at": now(),
                    "model": row["model"],
                    "condition_id": row["phase418_condition_id"],
                    "semantic_case_id": row["semantic_case_id"],
                    "family_id": row["family_id"],
                    "mechanism_id": row["mechanism_id"],
                    "split": row["split"],
                    "template_id": row["template_id"],
                    "item_index": int(row["item_index"]),
                    "interface": row["interface"],
                    "history_condition": row["history_condition"],
                    "composite_override_condition": row["composite_override_condition"],
                    "execution_phase": "prompt_prefill",
                    "position_role": "current_prediction_input",
                    "layer": layer,
                    "relative_depth": layer / max(1, layer_count - 1),
                    "depth_bin": depth_bin(layer, layer_count),
                    "component": component,
                    "vector_width": int(vector.numel()),
                    "l2_norm": norm,
                    "rms": rms,
                    "signed_mean": signed_mean,
                    "max_abs": max_abs,
                    "relative_write_rate": norm / max(input_norm, 1e-8) if norm is not None and math.isfinite(input_norm) else None,
                    "numerically_finite": finite,
                    "physical": True,
                    "reduced_measurement": True,
                    "causal": False,
                    "single_neuron_causal": False,
                }
            )
    return result


def length_stratum(delta: int) -> str:
    absolute = abs(delta)
    if absolute == 0:
        return "exact"
    if absolute <= 2:
        return "near_2"
    if absolute <= 8:
        return "near_8"
    return "far"


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right).div(denominator).item())


def aggregate_contrast(
    source: dict[tuple[int, str], torch.Tensor] | None,
    target: dict[tuple[int, str], torch.Tensor] | None,
    terms: list[tuple[float, dict[tuple[int, str], torch.Tensor]]],
    component: str,
    depth: str,
    layer_count: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    layers = [layer for layer in range(layer_count) if depth_bin(layer, layer_count) == depth]
    layer_metrics = []
    layer_deltas = []
    for layer in layers:
        delta = sum(weight * vectors[(layer, component)] for weight, vectors in terms)
        layer_deltas.append(delta)
        denominator = sum(
            abs(weight) * float(torch.linalg.vector_norm(vectors[(layer, component)]).item())
            for weight, vectors in terms
        ) / max(1.0, sum(abs(weight) for weight, _vectors in terms))
        item = {
            "layer": layer,
            "delta_l2": float(torch.linalg.vector_norm(delta).item()),
            "relative_delta": float(torch.linalg.vector_norm(delta).item()) / max(denominator, 1e-8),
            "delta_signed_mean": float(torch.mean(delta).item()),
        }
        if source is not None and target is not None:
            item["endpoint_cosine"] = cosine(
                source[(layer, component)], target[(layer, component)]
            )
        layer_metrics.append(item)
    dominant = max(layer_metrics, key=lambda item: item["relative_delta"])
    mean_delta = torch.stack(layer_deltas).mean(dim=0)
    return (
        {
            "layer_count": len(layers),
            "mean_delta_l2": sum(item["delta_l2"] for item in layer_metrics) / len(layer_metrics),
            "mean_relative_delta": sum(item["relative_delta"] for item in layer_metrics) / len(layer_metrics),
            "mean_delta_signed_mean": sum(item["delta_signed_mean"] for item in layer_metrics) / len(layer_metrics),
            "mean_endpoint_cosine": (
                sum(item.get("endpoint_cosine", 0.0) for item in layer_metrics) / len(layer_metrics)
                if source is not None and target is not None
                else None
            ),
            "dominant_layer": dominant["layer"],
            "dominant_layer_relative_delta": dominant["relative_delta"],
        },
        mean_delta,
    )


def build_contrasts(
    semantic_rows: list[dict[str, Any]],
    vectors: dict[tuple[str, str], dict[tuple[int, str], torch.Tensor]],
    layer_count: int,
    direction_sums: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    row_index = {(row["interface"], row["history_condition"]): row for row in semantic_rows}
    contrasts: list[tuple[str, str, str | None, str | None, list[tuple[float, tuple[str, str]]]]] = []
    for history in HISTORIES:
        contrasts.append(
            ("interface", history, "chat", "completion", [(-1.0, ("chat", history)), (1.0, ("completion", history))])
        )
    for interface in INTERFACES:
        for history in HISTORIES[1:]:
            contrasts.append(
                ("history", history, "none", history, [(-1.0, (interface, "none")), (1.0, (interface, history))])
            )
    for history in HISTORIES[1:]:
        contrasts.append(
            (
                "interaction",
                history,
                None,
                None,
                [
                    (1.0, ("completion", history)),
                    (-1.0, ("completion", "none")),
                    (-1.0, ("chat", history)),
                    (1.0, ("chat", "none")),
                ],
            )
        )

    first = semantic_rows[0]
    output: list[dict[str, Any]] = []
    for contrast_type, contrast_name, source_label, target_label, term_specs in contrasts:
        term_vectors = [(weight, vectors[key]) for weight, key in term_specs]
        pair_interface = term_specs[-1][1][0] if contrast_type == "history" else None
        if contrast_type == "interface":
            source_key, target_key = ("chat", contrast_name), ("completion", contrast_name)
        elif contrast_type == "history":
            source_key, target_key = (pair_interface, "none"), (pair_interface, contrast_name)
        else:
            source_key = target_key = None
        source_vectors = vectors[source_key] if source_key else None
        target_vectors = vectors[target_key] if target_key else None
        source_tokens = row_index[source_key]["prompt_token_count"] if source_key else None
        target_tokens = row_index[target_key]["prompt_token_count"] if target_key else None
        token_delta = target_tokens - source_tokens if source_tokens is not None else None
        for component in CORE_COMPONENTS:
            for depth in ("early", "middle", "late"):
                metrics, mean_delta = aggregate_contrast(
                    source_vectors,
                    target_vectors,
                    term_vectors,
                    component,
                    depth,
                    layer_count,
                )
                direction_norm = torch.linalg.vector_norm(mean_delta)
                direction_key = (
                    first["model"],
                    first["family_id"],
                    "discovery" if first["split"] == "discovery" else "non_discovery",
                    contrast_type,
                    contrast_name,
                    pair_interface,
                    component,
                    depth,
                )
                accumulator = direction_sums.setdefault(
                    direction_key,
                    {"sum": torch.zeros_like(mean_delta), "count": 0},
                )
                if float(direction_norm.item()) > 1e-12:
                    accumulator["sum"] += mean_delta / direction_norm
                    accumulator["count"] += 1
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": "Phase418-RegisteredVectorContrasts",
                        "created_at": now(),
                        "model": first["model"],
                        "semantic_case_id": first["semantic_case_id"],
                        "family_id": first["family_id"],
                        "mechanism_id": first["mechanism_id"],
                        "split": first["split"],
                        "template_id": first["template_id"],
                        "contrast_type": contrast_type,
                        "contrast_name": contrast_name,
                        "history_interface": pair_interface,
                        "source_label": source_label,
                        "target_label": target_label,
                        "component": component,
                        "depth_bin": depth,
                        "prompt_token_count_delta": token_delta,
                        "token_length_stratum": length_stratum(token_delta) if token_delta is not None else "interaction",
                        "composite_override_contrast": contrast_name == "override",
                        **metrics,
                        "physical": True,
                        "predictive": False,
                        "causal": False,
                    }
                )
    return output


def direction_rows(direction_sums: dict[tuple[Any, ...], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for key, value in sorted(direction_sums.items()):
        model, family, partition, contrast_type, contrast_name, interface, component, depth = key
        count = int(value["count"])
        sum_norm_sq = float(torch.sum(value["sum"].square()).item())
        concentration = math.sqrt(sum_norm_sq) / count if count else 0.0
        pairwise = (sum_norm_sq - count) / (count * (count - 1)) if count > 1 else None
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "phase_id": "Phase418-DirectionConsistency",
                "created_at": now(),
                "model": model,
                "family_id": family,
                "validation_partition": partition,
                "contrast_type": contrast_type,
                "contrast_name": contrast_name,
                "history_interface": interface,
                "component": component,
                "depth_bin": depth,
                "unit_direction_count": count,
                "resultant_direction_concentration": concentration,
                "mean_pairwise_direction_cosine": pairwise,
                "cross_model_hidden_space_aligned": False,
                "physical": True,
                "causal": False,
            }
        )
    return rows


@torch.inference_mode()
def run_model(model_key: str, max_new_tokens: int, max_semantic_cases: int | None) -> dict[str, Any]:
    registered = [row for row in read_jsonl(REGISTERED) if row["model"] == model_key]
    semantic_ids = sorted({row["semantic_case_id"] for row in registered})
    if max_semantic_cases is not None:
        semantic_ids = semantic_ids[:max_semantic_cases]
        registered = [row for row in registered if row["semantic_case_id"] in set(semantic_ids)]
    expected = len(semantic_ids) * len(INTERFACES) * len(HISTORIES)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in registered:
        groups[row["semantic_case_id"]].append(row)

    loaded = None
    collector = None
    started = time.monotonic()
    case_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    direction_sums: dict[tuple[Any, ...], dict[str, Any]] = {}
    anchors: dict[str, np.ndarray] = {}
    anchor_manifest: list[dict[str, Any]] = []
    try:
        print(f"[Phase418] loading {model_key}; semantic={len(semantic_ids)} conditions={expected}", flush=True)
        loaded = load_probe_model(model_key)
        collector = PromptVectorCollector(loaded)
        collector.install()
        layers = collector.layers
        # The leading newline can merge with an interface-specific role marker.
        # Freeze the literal terminal chain, then audit the preceding role tokens
        # separately through the complete prompt hash and interface label.
        suffix_ids = loaded.tokenizer(SHARED_TERMINAL_LITERAL, add_special_tokens=False)["input_ids"]
        if not suffix_ids:
            raise RuntimeError("Shared suffix produced no token IDs")
        eos = eos_ids(loaded.tokenizer, loaded.model)

        for semantic_index, semantic_id in enumerate(semantic_ids, start=1):
            semantic_rows = sorted(
                groups[semantic_id],
                key=lambda row: (INTERFACES.index(row["interface"]), HISTORIES.index(row["history_condition"])),
            )
            vector_bank: dict[tuple[str, str], dict[tuple[int, str], torch.Tensor]] = {}
            for row in semantic_rows:
                prompt, messages = serialize_prompt(loaded.tokenizer, row)
                encoded = encode_prompt(loaded, prompt)
                ids = encoded["input_ids"][0].tolist()
                suffix_aligned = prompt.endswith(SHARED_TERMINAL_LITERAL) and ids[-len(suffix_ids):] == suffix_ids
                if not suffix_aligned:
                    raise RuntimeError(f"Terminal suffix mismatch: {row['phase418_condition_id']}")
                collector.begin()
                output = loaded.model.generate(
                    **encoded,
                    generation_config=neutral_generation_config(loaded),
                    max_new_tokens=max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                collector.active = False
                generated_ids = [int(value) for value in output.sequences[0, len(ids):].tolist()]
                generated_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
                emitted_stop = any(token in eos for token in generated_ids)
                ledger_max = max(collector.ledger_errors, default=math.inf)
                vector_count = len(collector.vectors)
                finite = all(torch.isfinite(vector).all().item() for vector in collector.vectors.values())
                condition_pass = bool(
                    suffix_aligned
                    and vector_count == len(layers) * len(CORE_COMPONENTS)
                    and finite
                    and ledger_max <= LEDGER_THRESHOLD
                )
                executable = {
                    **row,
                    "created_at": now(),
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": len(ids),
                    "shared_suffix_token_ids": suffix_ids,
                    "terminal_suffix_token_alignment_pass": suffix_aligned,
                    "message_count": len(messages),
                    "generated_token_ids": generated_ids,
                    "generated_text": generated_text,
                    "target_event_match": target_match(generated_text, row["target_aliases"]),
                    "exact_answer_match": exact_answer(generated_text, row["target_aliases"]),
                    "emitted_stop": emitted_stop,
                    "right_censored": not emitted_stop and len(generated_ids) >= max_new_tokens,
                    "native_generation_call_count": collector.call_index + 1,
                    "native_generation_call_widths": collector.call_widths,
                    "prompt_vector_count": vector_count,
                    "component_ledger_max_relative_error": ledger_max,
                    "physical_finite": finite,
                    "condition_pass": condition_pass,
                    "instrument_qualified_by_phase417": True,
                    "causal": False,
                }
                case_rows.append(executable)
                scalar_rows.extend(physical_rows(executable, collector.vectors, len(layers)))
                vector_bank[(row["interface"], row["history_condition"])] = {
                    key: value.clone() for key, value in collector.vectors.items()
                }

                family_first = min(
                    item["semantic_case_id"]
                    for item in registered
                    if item["family_id"] == row["family_id"]
                )
                is_anchor = semantic_id == family_first and (
                    (row["interface"], row["history_condition"]) in (("chat", "none"), ("completion", "conflict"))
                )
                if is_anchor:
                    anchor_prefix = f"{row['family_id']}__{row['interface']}__{row['history_condition']}"
                    for (layer, component), vector in collector.vectors.items():
                        key = f"{anchor_prefix}__layer_{layer:02d}__{component}"
                        anchors[key] = vector.numpy().astype(np.float16)
                    anchor_manifest.append(
                        {
                            "condition_id": row["phase418_condition_id"],
                            "semantic_case_id": semantic_id,
                            "family_id": row["family_id"],
                            "interface": row["interface"],
                            "history_condition": row["history_condition"],
                            "vector_count": len(collector.vectors),
                        }
                    )
                collector.end()
                del output, encoded

            semantic_case_rows = case_rows[-10:]
            if all(item["condition_pass"] for item in semantic_case_rows):
                contrast_rows.extend(
                    build_contrasts(
                        semantic_rows=semantic_case_rows,
                        vectors=vector_bank,
                        layer_count=len(layers),
                        direction_sums=direction_sums,
                    )
                )
            del vector_bank
            gc.collect()
            if semantic_index % 2 == 0 or semantic_index == len(semantic_ids):
                passed = sum(row["condition_pass"] for row in case_rows)
                print(
                    f"[Phase418:{model_key}] semantic={semantic_index}/{len(semantic_ids)} "
                    f"conditions={len(case_rows)} pass={passed}",
                    flush=True,
                )

        model_root = OUT / "models" / model_key
        direction = direction_rows(direction_sums)
        write_jsonl(model_root / "phase418_condition_rows.jsonl", case_rows)
        write_jsonl(model_root / "phase418_prefill_physical_rows.jsonl", scalar_rows)
        write_jsonl(model_root / "phase418_vector_contrast_rows.jsonl", contrast_rows)
        write_jsonl(model_root / "phase418_direction_consistency_rows.jsonl", direction)
        model_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(model_root / "phase418_anchor_vectors.npz", **anchors)
        write_json(model_root / "phase418_anchor_vector_manifest.json", anchor_manifest)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE_ID,
            "created_at": now(),
            "model": model_key,
            "semantic_case_count": len(semantic_ids),
            "condition_count": len(case_rows),
            "required_condition_count": expected,
            "condition_pass_count": sum(row["condition_pass"] for row in case_rows),
            "all_conditions_pass": bool(len(case_rows) == expected and all(row["condition_pass"] for row in case_rows)),
            "target_event_match_count": sum(row["target_event_match"] for row in case_rows),
            "exact_answer_match_count": sum(row["exact_answer_match"] for row in case_rows),
            "right_censored_count": sum(row["right_censored"] for row in case_rows),
            "physical_row_count": len(scalar_rows),
            "vector_contrast_row_count": len(contrast_rows),
            "direction_consistency_row_count": len(direction),
            "lossless_anchor_condition_count": len(anchor_manifest),
            "lossless_anchor_vector_count": len(anchors),
            "max_prompt_token_count": max(row["prompt_token_count"] for row in case_rows),
            "max_component_ledger_relative_error": max(row["component_ledger_max_relative_error"] for row in case_rows),
            "terminal_suffix_alignment_pass_count": sum(row["terminal_suffix_token_alignment_pass"] for row in case_rows),
            "condition_rows_sha256": hash_rows(case_rows),
            "physical_rows_sha256": hash_rows(scalar_rows),
            "contrast_rows_sha256": hash_rows(contrast_rows),
            "elapsed_seconds": time.monotonic() - started,
            "vram_gb": vram_gb(),
            "functional_names_authorized": False,
            "causal_intervention_authorized": False,
            "single_neuron_scan_authorized": False,
            "claim_boundary": "paired_interface_history_behavior_and_prompt_prefill_reduced_physical_differences_only",
        }
        write_json(model_root / "phase418_trace_complete.json", summary)
        return summary
    finally:
        if collector is not None:
            collector.remove()
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-semantic-cases", type=int)
    args = parser.parse_args()
    summary = run_model(args.model, args.max_new_tokens, args.max_semantic_cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    if not summary["all_conditions_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
