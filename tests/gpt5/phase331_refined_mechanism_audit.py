#!/usr/bin/env python3
"""Run Phase331 dual-interface refinement, controls, and compensation tracing."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase330_nine_family_case_bank import MODELS  # noqa: E402
import phase330_global_atlas_survey as survey  # noqa: E402
import phase326_distributed_carrier_atlas as phase326  # noqa: E402
from phase330_registered_causal_audit import module_for, spec_key  # noqa: E402
from phase331_refined_mechanism_case_bank import (  # noqa: E402
    CONTROL_CONDITIONS,
    GENERATION_CONDITIONS,
    POSITIVE_CONDITIONS,
    ROUND_DEFAULT,
)
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase331"
SCHEMA_VERSION = "9.0.0"
SOURCE = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas/nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase331_refined_mechanism_audit"
ROLES = ("source", "query", "last")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd", row_group_size=32768)


def ordered_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    component_order = {"attention_head_input": 0, "mlp_product_group": 1}
    return sorted(
        rows,
        key=lambda row: (
            component_order[row["component_type"]], int(row["set_rank"]),
            int(row["component_layer"]), int(row["component_index"]),
        ),
    )


def encode_prompt(loaded: Any, case: dict[str, Any]) -> dict[str, torch.Tensor]:
    encoded = loaded.tokenizer(
        case["prompt"], return_tensors="pt", truncation=True, max_length=192,
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
    )
    return {key: value.to(loaded.input_device) for key, value in encoded.items()}


def token_ids(loaded: Any, text: str, interface: str) -> list[int]:
    candidate = (" " + text) if interface == "raw_completion" else text
    ids = loaded.tokenizer(candidate, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Cannot tokenize continuation {text!r}")
    return [int(value) for value in ids]


def locate_span(tokenizer: Any, case: dict[str, Any], fragment: str, seq_len: int) -> tuple[int, int]:
    prompt = case["prompt"]
    start_char = prompt.index(fragment)
    end_char = start_char + len(fragment)
    add_special = bool(case["tokenization_add_special_tokens"])
    prefix = tokenizer(prompt[:start_char], add_special_tokens=add_special)["input_ids"]
    endpoint = tokenizer(prompt[:end_char], add_special_tokens=add_special)["input_ids"]
    start = min(seq_len - 1, max(0, len(prefix) - 1))
    end = min(seq_len - 1, max(start, len(endpoint) - 1))
    return start, end


def role_spans(tokenizer: Any, case: dict[str, Any], seq_len: int) -> dict[str, tuple[int, int]]:
    prompt = case["prompt"]
    first = prompt.index(case["source_fragments"][0])
    last_fragment = case["source_fragments"][-1]
    source = prompt[first : prompt.index(last_fragment, first) + len(last_fragment)]
    return {
        "source": locate_span(tokenizer, case, source, seq_len),
        "query": locate_span(tokenizer, case, case["query_fragment"], seq_len),
        "last": (seq_len - 1, seq_len - 1),
    }


def donor_case(case: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        **case,
        "prompt": case[f"{prefix}_prompt"],
        "source_fragments": case[f"{prefix}_source_fragments"],
        "query_fragment": case[f"{prefix}_query_fragment"],
    }


def same_target_donor_case(loaded: Any, case: dict[str, Any]) -> dict[str, Any]:
    source = f"The exact answer token for this identity control is {case['target']}."
    query = "Return that token exactly."
    raw = f"{source}\n{query}\nAnswer:"
    if case["interface"] == "chat_template":
        prompt = loaded.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}], tokenize=False, add_generation_prompt=True,
        )
        add_special = False
    else:
        prompt = raw
        add_special = True
    return {
        **case,
        "prompt": prompt,
        "raw_prompt": raw,
        "source_fragments": [source],
        "query_fragment": query,
        "tokenization_add_special_tokens": add_special,
    }


@torch.inference_mode()
def capture_values(loaded: Any, case: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    encoded = encode_prompt(loaded, case)
    seq_len = int(encoded["attention_mask"].sum().item())
    spans = role_spans(loaded.tokenizer, case, seq_len)
    by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    modules: dict[int, Any] = {}
    for spec in specs:
        module = module_for(loaded.model, spec)
        by_module[id(module)].append(spec)
        modules[id(module)] = module
    values: dict[str, torch.Tensor] = {}
    handles = []
    for key, module in modules.items():
        selected = by_module[key]

        def hook(_module: Any, inputs: tuple[Any, ...], selected: list[dict[str, Any]] = selected) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            tensor = inputs[0]
            for spec in selected:
                pos_start, pos_end = spans[spec["position_role"]]
                vector = tensor[0, pos_start : pos_end + 1, int(spec["component_start"]):int(spec["component_end"])]
                values[spec_key(spec)] = vector.detach().mean(dim=0).clone()

        handles.append(module.register_forward_pre_hook(hook))
    try:
        loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if len(values) != len(specs):
        raise RuntimeError(f"Captured {len(values)} of {len(specs)} donor values")
    return values


def install_intervention_hooks(
    loaded: Any,
    case: dict[str, Any],
    encoded: dict[str, torch.Tensor],
    zero_specs: list[dict[str, Any]],
    transplant_specs: list[dict[str, Any]],
    values: dict[str, torch.Tensor],
) -> list[Any]:
    seq_len = int(encoded["attention_mask"].sum().item())
    spans = role_spans(loaded.tokenizer, case, seq_len)
    zero_by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transplant_by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    modules: dict[int, Any] = {}
    for bucket, specs in ((zero_by_module, zero_specs), (transplant_by_module, transplant_specs)):
        for spec in specs:
            module = module_for(loaded.model, spec)
            bucket[id(module)].append(spec)
            modules[id(module)] = module
    handles = []
    for key, module in modules.items():
        zero = zero_by_module.get(key, [])
        transplant = transplant_by_module.get(key, [])

        def hook(
            _module: Any, inputs: tuple[Any, ...], zero: list[dict[str, Any]] = zero,
            transplant: list[dict[str, Any]] = transplant,
        ) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            changed = inputs[0].clone()
            for spec in zero:
                pos_start, pos_end = spans[spec["position_role"]]
                if pos_start >= changed.shape[1]:
                    continue
                positions = slice(pos_start, min(pos_end + 1, changed.shape[1]))
                changed[0, positions, int(spec["component_start"]):int(spec["component_end"])] = 0
            for spec in transplant:
                pos_start, pos_end = spans[spec["position_role"]]
                if pos_start >= changed.shape[1]:
                    continue
                positions = slice(pos_start, min(pos_end + 1, changed.shape[1]))
                value = values[spec_key(spec)].to(changed.device, changed.dtype)
                changed[0, positions, int(spec["component_start"]):int(spec["component_end"])] = value
            return (changed, *inputs[1:])

        handles.append(module.register_forward_pre_hook(hook))
    return handles


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported component output {type(output).__name__}")


def install_trace_hooks(
    loaded: Any,
    case: dict[str, Any],
    encoded: dict[str, torch.Tensor],
    direction: torch.Tensor,
    carrier_specs: list[dict[str, Any]],
    zero_specs: list[dict[str, Any]],
    transplant_specs: list[dict[str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    seq_len = int(encoded["attention_mask"].sum().item())
    spans = role_spans(loaded.tokenizer, case, seq_len)
    layers = get_layers(loaded.model)
    direction = direction.detach().float()
    capture: dict[str, Any] = {
        "residual_vectors": {},
        "component_values": {},
        "unit_inputs": {},
        "spans": spans,
    }
    handles: list[Any] = []

    def pooled(tensor: torch.Tensor, role: str) -> torch.Tensor:
        start, end = spans[role]
        if start >= tensor.shape[1]:
            start = tensor.shape[1] - 1
            end = start
        return tensor[0, start : min(end + 1, tensor.shape[1])].detach().float().mean(dim=0)

    for layer_index, layer in enumerate(layers):
        def residual_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                for role in ROLES:
                    capture["residual_vectors"][(idx, role)] = pooled(inputs[0], role).cpu()

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            tensor = component_tensor(output)
            for role in ROLES:
                vector = pooled(tensor, role)
                capture["component_values"][("attention", idx, role)] = (
                    float(torch.dot(vector, direction.to(vector.device)).item()),
                    float(torch.linalg.vector_norm(vector).item()),
                )

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            tensor = component_tensor(output)
            for role in ROLES:
                vector = pooled(tensor, role)
                capture["component_values"][("mlp", idx, role)] = (
                    float(torch.dot(vector, direction.to(vector.device)).item()),
                    float(torch.linalg.vector_norm(vector).item()),
                )

        handles.append(layer.register_forward_pre_hook(residual_pre))
        handles.append(layer.self_attn.register_forward_hook(attention_post))
        handles.append(layer.mlp.register_forward_hook(mlp_post))

    unit_modules: dict[tuple[str, int, str], Any] = {}
    for spec in carrier_specs:
        key = (spec["component_type"], int(spec["component_layer"]), spec["position_role"])
        unit_modules[key] = module_for(loaded.model, spec)
    for key, module in unit_modules.items():
        component_type, layer_index, role = key

        def unit_pre(
            _module: Any, inputs: tuple[Any, ...], key: tuple[str, int, str] = key,
            role: str = role,
        ) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                capture["unit_inputs"][key] = pooled(inputs[0], role).cpu()

        handles.append(module.register_forward_pre_hook(unit_pre))
    capture["carrier_keys"] = {spec_key(row) for row in carrier_specs}
    capture["zero_keys"] = {spec_key(row) for row in zero_specs}
    capture["transplant_keys"] = {spec_key(row) for row in transplant_specs}
    return handles, capture


def trace_rows(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    capture: dict[str, Any],
    direction: torch.Tensor,
    carrier_specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    created_at = now()
    path_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    layers = get_layers(loaded.model)
    residual = capture["residual_vectors"]
    for layer_index in range(len(layers)):
        for role in ROLES:
            vector = residual[(layer_index, role)].float()
            path_rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": created_at,
                "audit_case_id": case["audit_case_id"],
                "model": case["model"],
                "cohort": case["cohort"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "interface": case["interface"],
                "item_index": case["item_index"],
                "template_id": case["template_id"],
                "condition": condition,
                "layer": layer_index,
                "position_role": role,
                "component_type": "residual_cumulative_state",
                "projection": round(float(torch.dot(vector, direction.cpu()).item()), 7),
                "activation_norm": round(float(torch.linalg.vector_norm(vector).item()), 7),
                "incremental": False,
                "evidence_level": "L2_intervention_trace",
            })
            for component in ("attention", "mlp"):
                projection, norm = capture["component_values"][(component, layer_index, role)]
                path_rows.append({
                    **{key: path_rows[-1][key] for key in (
                        "schema_version", "phase_id", "created_at", "audit_case_id", "model", "cohort",
                        "family_id", "mechanism_id", "interface", "item_index", "template_id", "condition",
                        "layer", "position_role", "evidence_level",
                    )},
                    "component_type": f"{component}_increment",
                    "projection": round(projection, 7),
                    "activation_norm": round(norm, 7),
                    "incremental": True,
                })
            if layer_index + 1 < len(layers):
                delta = residual[(layer_index + 1, role)].float() - vector
                path_rows.append({
                    **{key: path_rows[-1][key] for key in (
                        "schema_version", "phase_id", "created_at", "audit_case_id", "model", "cohort",
                        "family_id", "mechanism_id", "interface", "item_index", "template_id", "condition",
                        "layer", "position_role", "evidence_level",
                    )},
                    "component_type": "residual_layer_increment",
                    "projection": round(float(torch.dot(delta, direction.cpu()).item()), 7),
                    "activation_norm": round(float(torch.linalg.vector_norm(delta).item()), 7),
                    "incremental": True,
                })

    specs_by_identity = {
        (row["component_type"], int(row["component_layer"]), row["position_role"], int(row["component_index"])): row
        for row in carrier_specs
    }
    direction_device = direction.to(loaded.input_device).float()
    for key, cpu_input in capture["unit_inputs"].items():
        component_type, layer_index, role = key
        vector = cpu_input.to(loaded.input_device).float()
        if component_type == "attention_head_input":
            projection, unit_count, unit_width = head_meta(loaded.model, layer_index)
            input_direction = direction_device @ projection.weight.detach().float()
            ranges = [(index * unit_width, (index + 1) * unit_width) for index in range(unit_count)]
            unit_kind = "attention_head"
        else:
            down_proj = phase326.get_down_proj(layers[layer_index])
            if down_proj is None:
                raise TypeError(f"No MLP down projection at layer {layer_index}")
            input_direction = direction_device @ down_proj.weight.detach().float()
            ranges = phase326.group_ranges(int(down_proj.in_features))
            unit_kind = "mlp_product_group"
        for unit_index, (start, end) in enumerate(ranges):
            contribution = float(torch.dot(vector[start:end], input_direction[start:end]).item())
            identity = (component_type, layer_index, role, unit_index)
            selected_spec = specs_by_identity.get(identity)
            selected = selected_spec is not None
            selected_key = spec_key(selected_spec) if selected_spec is not None else None
            unit_rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": created_at,
                "audit_case_id": case["audit_case_id"],
                "model": case["model"],
                "cohort": case["cohort"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "interface": case["interface"],
                "item_index": case["item_index"],
                "template_id": case["template_id"],
                "condition": condition,
                "component_type": component_type,
                "unit_kind": unit_kind,
                "layer": layer_index,
                "position_role": role,
                "unit_index": unit_index,
                "unit_start": start,
                "unit_end": end,
                "approx_target_readout_contribution": round(contribution, 7),
                "selected_carrier_member": selected,
                "intervened": bool(selected_key and selected_key in capture["zero_keys"]),
                "transplanted": bool(selected_key and selected_key in capture["transplant_keys"]),
                "evidence_level": "L3_intervention_component_response",
                "single_unit_causal": False,
            })
    return path_rows, unit_rows


def target_match(text: str, aliases: list[str]) -> bool:
    normalized = survey.normalize_text(text)
    for alias in aliases:
        value = survey.normalize_text(alias)
        if not value:
            continue
        if re.search(rf"(?<!\w){re.escape(value)}(?!\w)", normalized):
            return True
    return False


def answer_segment(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    return text.strip()


def eos_ids(tokenizer: Any) -> list[int]:
    value = tokenizer.eos_token_id
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(item) for item in value]


def phrase_logprob(
    loaded: Any,
    encoded: dict[str, torch.Tensor],
    answer_ids: list[int],
) -> tuple[float, list[float]]:
    if len(answer_ids) == 1:
        output = loaded.model(**encoded, use_cache=False, return_dict=True)
        seq_len = int(encoded["attention_mask"].sum().item())
        value = float(torch.log_softmax(output.logits[0, seq_len - 1].float(), dim=-1)[answer_ids[0]].item())
        return value, [value]
    prefix_ids = encoded["input_ids"]
    append = torch.tensor([answer_ids[:-1]], dtype=prefix_ids.dtype, device=prefix_ids.device)
    input_ids = torch.cat([prefix_ids, append], dim=1)
    attention_mask = torch.cat([
        encoded["attention_mask"],
        torch.ones_like(append, dtype=encoded["attention_mask"].dtype),
    ], dim=1)
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    output = loaded.model(**kwargs, use_cache=False, return_dict=True)
    prompt_length = prefix_ids.shape[1]
    log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
    values = [
        float(log_probs[prompt_length + offset - 1, token_id].item())
        for offset, token_id in enumerate(answer_ids)
    ]
    return sum(values), values


@torch.inference_mode()
def run_condition(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    carrier_specs: list[dict[str, Any]],
    zero_specs: list[dict[str, Any]],
    transplant_specs: list[dict[str, Any]],
    values: dict[str, torch.Tensor],
    max_new_tokens: int,
    trace: bool,
    force_generation: bool | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    encoded = encode_prompt(loaded, case)
    answer_ids = token_ids(loaded, case["target"], case["interface"])
    distractor_ids = [token_ids(loaded, value, case["interface"])[0] for value in case["distractors"]]
    alias_ids = list(dict.fromkeys(
        token_ids(loaded, value, case["interface"])[0]
        for value in case["target_aliases"]
    ))
    output_weight = loaded.model.get_output_embeddings().weight.detach()
    direction = output_weight[answer_ids[0]].float() - output_weight[distractor_ids].float().mean(dim=0)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
    intervention_handles = install_intervention_hooks(
        loaded, case, encoded, zero_specs, transplant_specs, values,
    )
    trace_handles: list[Any] = []
    capture: dict[str, Any] | None = None
    if trace:
        trace_handles, capture = install_trace_hooks(
            loaded, case, encoded, direction, carrier_specs, zero_specs, transplant_specs,
        )
    try:
        output = loaded.model(**encoded, use_cache=False, return_dict=True)
        seq_len = int(encoded["attention_mask"].sum().item())
        logits = output.logits[0, seq_len - 1].detach().float()
        for handle in trace_handles:
            handle.remove()
        trace_handles = []
        path_rows: list[dict[str, Any]] = []
        unit_rows: list[dict[str, Any]] = []
        if capture is not None:
            path_rows, unit_rows = trace_rows(loaded, case, condition, capture, direction, carrier_specs)

        target_logit = float(logits[answer_ids[0]].item())
        distractor_logits = [float(logits[token_id].item()) for token_id in distractor_ids]
        best_distractor = max(distractor_logits)
        rank = 1 + int((logits > target_logit).sum().item())
        top_values, top_indices = torch.topk(logits, k=50)
        top50 = []
        eos_set = set(eos_ids(loaded.tokenizer))
        for rank_index, (token_id, value) in enumerate(zip(top_indices.tolist(), top_values.tolist(), strict=True), 1):
            token_text = loaded.tokenizer.decode([int(token_id)])
            if int(token_id) == answer_ids[0]:
                blocker_type = "target"
            elif int(token_id) in distractor_ids:
                blocker_type = "registered_distractor"
            elif int(token_id) in eos_set:
                blocker_type = "eos"
            elif token_text.strip() in {"", ".", ",", ":", ";", "-"}:
                blocker_type = "protocol_or_punctuation"
            else:
                blocker_type = "other_vocabulary"
            top50.append({
                "rank": rank_index,
                "token_id": int(token_id),
                "token_text": token_text,
                "logit": round(float(value), 7),
                "blocker_type": blocker_type,
            })

        phrase_total, phrase_values = phrase_logprob(loaded, encoded, answer_ids)
        generation_executed = (
            condition in GENERATION_CONDITIONS if force_generation is None else force_generation
        )
        generated_text: str | None = None
        generated_ids: list[int] | None = None
        target_anywhere: bool | None = None
        target_answer: bool | None = None
        protocol_full: bool | None = None
        protocol_answer: bool | None = None
        behavior_success: bool | None = None
        eos_emitted: bool | None = None
        if generation_executed:
            generated = loaded.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
            )
            suffix = generated[0, encoded["input_ids"].shape[1] :]
            generated_ids = [int(value) for value in suffix.tolist()]
            generated_text = loaded.tokenizer.decode(generated_ids, skip_special_tokens=True)
            segment = answer_segment(generated_text)
            target_anywhere = target_match(generated_text, case["target_aliases"])
            target_answer = target_match(segment, case["target_aliases"])
            protocol_full = survey.protocol_ok(case, generated_text)
            protocol_answer = survey.protocol_ok(case, segment)
            behavior_success = bool(target_answer and protocol_answer)
            eos_emitted = any(token_id in eos_set for token_id in generated_ids)

        continue_candidates = []
        for value in ("\n", " and", ".", ",", ":"):
            ids = loaded.tokenizer(value, add_special_tokens=False)["input_ids"]
            if ids:
                continue_candidates.append(int(ids[0]))
        continue_logit = max(float(logits[token_id].item()) for token_id in continue_candidates)
        eos_logit = max((float(logits[token_id].item()) for token_id in eos_set), default=float("-inf"))
        alias_values = [float(logits[token_id].item()) for token_id in alias_ids]
        result = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "audit_case_id": case["audit_case_id"],
            "model": case["model"],
            "cohort": case["cohort"],
            "family_id": case["family_id"],
            "mechanism_id": case["mechanism_id"],
            "paired_mechanism_id": case["paired_mechanism_id"],
            "item_index": case["item_index"],
            "template_id": case["template_id"],
            "interface": case["interface"],
            "target": case["target"],
            "target_bucket": case["target_bucket"],
            "target_class": case["target_class"],
            "condition": condition,
            "sequence_length": seq_len,
            "target_first_token_id": answer_ids[0],
            "target_token_ids": json.dumps(answer_ids),
            "target_token_count": len(answer_ids),
            "target_logit": round(target_logit, 7),
            "best_distractor_logit": round(best_distractor, 7),
            "target_margin": round(target_logit - best_distractor, 7),
            "target_full_vocabulary_rank": rank,
            "target_in_top50": rank <= 50,
            "candidate_winner_is_target": target_logit >= best_distractor,
            "alias_subspace_max_logit": round(max(alias_values), 7),
            "alias_subspace_mean_logit": round(mean(alias_values), 7),
            "target_phrase_logprob": round(phrase_total, 7),
            "target_phrase_mean_logprob": round(mean(phrase_values), 7),
            "target_token_logprobs": json.dumps([round(value, 7) for value in phrase_values]),
            "top50": json.dumps(top50, ensure_ascii=False),
            "top1_token_id": int(top_indices[0].item()),
            "top1_token_text": loaded.tokenizer.decode([int(top_indices[0].item())]),
            "eos_logit": None if not math.isfinite(eos_logit) else round(eos_logit, 7),
            "continue_logit": round(continue_logit, 7),
            "target_minus_eos_logit": None if not math.isfinite(eos_logit) else round(target_logit - eos_logit, 7),
            "target_minus_continue_logit": round(target_logit - continue_logit, 7),
            "generation_executed": generation_executed,
            "generated_text": generated_text,
            "generated_token_ids": None if generated_ids is None else json.dumps(generated_ids),
            "target_anywhere_match": target_anywhere,
            "target_answer_segment_match": target_answer,
            "protocol_success_full_output": protocol_full,
            "protocol_success_answer_segment": protocol_answer,
            "behavior_success": behavior_success,
            "eos_emitted": eos_emitted,
            "zero_component_count": len(zero_specs),
            "transplant_component_count": len(transplant_specs),
            "trace_executed": trace,
            "single_unit_causal": False,
            "evidence_level": "L4_expanded_registered_set_intervention",
        }
        return result, path_rows, unit_rows
    finally:
        for handle in trace_handles:
            handle.remove()
        for handle in intervention_handles:
            handle.remove()


def build_condition_plan(
    loaded: Any,
    case: dict[str, Any],
    specs: list[dict[str, Any]],
    paired_specs: list[dict[str, Any]],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]]:
    joint = ordered_specs(specs)
    attention = [row for row in joint if row["component_type"] == "attention_head_input"]
    mlp = [row for row in joint if row["component_type"] == "mlp_product_group"]
    plan: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]] = {
        "baseline": ([], [], {}),
        "joint_set_zero": (joint, [], {}),
        "attention_set_zero": (attention, [], {}),
        "mlp_set_zero": (mlp, [], {}),
        "matched_random_joint_zero": (phase326.randomize_specs(loaded.model, joint), [], {}),
        "wrong_layer_joint_zero": (phase326.wrong_layer_specs(loaded.model, joint), [], {}),
    }
    for index, spec in enumerate(joint):
        plan[f"single_member_{index}_zero"] = ([spec], [], {})
        plan[f"set_without_member_{index}_zero"] = ([row for offset, row in enumerate(joint) if offset != index], [], {})
    paired_name = "paired_control_joint_zero" if case["cohort"] == "positive" else "paired_positive_joint_zero"
    plan[paired_name] = (ordered_specs(paired_specs), [], {})
    if case["cohort"] == "positive":
        correct = donor_case(case, "correct_donor")
        wrong = donor_case(case, "wrong_donor")
        random_donor = donor_case(case, "matched_random_donor")
        same_target = same_target_donor_case(loaded, case)
        correct_values = capture_values(loaded, correct, joint)
        wrong_values = capture_values(loaded, wrong, joint)
        random_values = capture_values(loaded, random_donor, joint)
        same_values = capture_values(loaded, same_target, joint)
        wrong_layer_specs = phase326.wrong_layer_specs(loaded.model, joint)
        wrong_layer_values = capture_values(loaded, correct, wrong_layer_specs)
        plan.update({
            "correct_donor_transplant": ([], joint, correct_values),
            "wrong_donor_transplant": ([], joint, wrong_values),
            "same_target_donor_transplant": ([], joint, same_values),
            "matched_random_donor_transplant": ([], joint, random_values),
            "wrong_layer_donor_transplant": ([], wrong_layer_specs, wrong_layer_values),
            "correct_donor_restoration": (joint, joint, correct_values),
        })
    return plan


def run_model(model: str, round_name: str, max_new_tokens: int, max_cases: int = 0) -> dict[str, Any]:
    root = OUT / round_name
    output_dir = root / "model_runs" / model
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and max_cases == 0:
        return read_json(complete_path)
    cases = [row for row in read_jsonl(root / "phase331_registered_cases.jsonl") if row["model"] == model]
    if max_cases:
        cases = cases[:max_cases]
    carriers = [row for row in read_jsonl(SOURCE / "carrier_sets.jsonl") if row["model"] == model]
    carrier_map: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in carriers:
        carrier_map[(row["family_id"], row["mechanism_id"])].append(row)
    loaded = None
    condition_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for case_index, case in enumerate(cases, 1):
            specs = carrier_map[(case["family_id"], case["mechanism_id"])]
            paired_specs = carrier_map[(case["family_id"], case["paired_mechanism_id"])]
            if len(specs) != 4 or len(paired_specs) != 4:
                raise RuntimeError(f"Expected four frozen members for {case['audit_case_id']}")
            plan = build_condition_plan(loaded, case, specs, paired_specs)
            conditions = POSITIVE_CONDITIONS if case["cohort"] == "positive" else CONTROL_CONDITIONS
            case_results = []
            for condition in conditions:
                zero, transplant, values = plan[condition]
                trace = case["template_id"] == "template_c"
                row, new_paths, new_units = run_condition(
                    loaded, case, condition, ordered_specs(specs), zero, transplant, values,
                    max_new_tokens, trace,
                )
                case_results.append(row)
                path_rows.extend(new_paths)
                unit_rows.extend(new_units)
            baseline = next(row for row in case_results if row["condition"] == "baseline")
            for row in case_results:
                row["delta_target_margin_vs_baseline"] = round(row["target_margin"] - baseline["target_margin"], 7)
                row["delta_phrase_logprob_vs_baseline"] = round(
                    row["target_phrase_logprob"] - baseline["target_phrase_logprob"], 7
                )
                row["target_rank_change_vs_baseline"] = int(
                    baseline["target_full_vocabulary_rank"] - row["target_full_vocabulary_rank"]
                )
                row["top1_changed_vs_baseline"] = row["top1_token_id"] != baseline["top1_token_id"]
                if row["generation_executed"]:
                    row["generation_changed_vs_baseline"] = row["generated_token_ids"] != baseline["generated_token_ids"]
                    row["behavior_changed_vs_baseline"] = row["behavior_success"] != baseline["behavior_success"]
                    row["behavior_lost_vs_baseline"] = bool(
                        baseline["behavior_success"] and not row["behavior_success"]
                    )
                    row["behavior_gained_vs_baseline"] = bool(
                        not baseline["behavior_success"] and row["behavior_success"]
                    )
                    row["protocol_changed_vs_baseline"] = (
                        row["protocol_success_answer_segment"] != baseline["protocol_success_answer_segment"]
                    )
                    row["protocol_lost_vs_baseline"] = bool(
                        baseline["protocol_success_answer_segment"]
                        and not row["protocol_success_answer_segment"]
                    )
                else:
                    row["generation_changed_vs_baseline"] = None
                    row["behavior_changed_vs_baseline"] = None
                    row["behavior_lost_vs_baseline"] = None
                    row["behavior_gained_vs_baseline"] = None
                    row["protocol_changed_vs_baseline"] = None
                    row["protocol_lost_vs_baseline"] = None
            condition_rows.extend(case_results)
            if case_index % 8 == 0:
                print(json.dumps({
                    "quality_only": True,
                    "model": model,
                    "completed_interface_cases": case_index,
                    "total_interface_cases": len(cases),
                    "condition_rows": len(condition_rows),
                    "path_rows": len(path_rows),
                }), flush=True)
        write_parquet(output_dir / "condition_rows.parquet", condition_rows)
        write_jsonl(output_dir / "condition_rows.jsonl", condition_rows)
        write_parquet(output_dir / "compensation_path_rows.parquet", path_rows)
        write_parquet(output_dir / "component_response_rows.parquet", unit_rows)
        expected_cases = 240 if not max_cases else len(cases)
        expected_conditions = 3120 if not max_cases else len(condition_rows)
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "interface_case_count": len(cases),
            "condition_row_count": len(condition_rows),
            "generation_row_count": sum(row["generation_executed"] for row in condition_rows),
            "trace_condition_count": sum(row["trace_executed"] for row in condition_rows),
            "compensation_path_row_count": len(path_rows),
            "component_response_row_count": len(unit_rows),
            "expected_interface_case_count": expected_cases,
            "expected_condition_row_count": expected_conditions,
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(cases) == expected_cases and len(condition_rows) == expected_conditions,
            "smoke": bool(max_cases),
        }
        write_json(complete_path if not max_cases else output_dir / "smoke_complete.json", quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def collect(round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    condition_rows: list[dict[str, Any]] = []
    path_tables = []
    unit_tables = []
    qualities = []
    for model in MODELS:
        model_dir = root / "model_runs" / model
        condition_rows.extend(read_jsonl(model_dir / "condition_rows.jsonl"))
        path_tables.append(pq.read_table(model_dir / "compensation_path_rows.parquet"))
        unit_tables.append(pq.read_table(model_dir / "component_response_rows.parquet"))
        qualities.append(read_json(model_dir / "complete.json"))
    write_jsonl(root / "phase331_condition_rows.jsonl", condition_rows)
    write_parquet(root / "phase331_condition_rows.parquet", condition_rows)
    pq.write_table(pa.concat_tables(path_tables), root / "phase331_compensation_path_rows.parquet", compression="zstd")
    pq.write_table(pa.concat_tables(unit_tables), root / "phase331_component_response_rows.parquet", compression="zstd")
    quality = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model_count": len(qualities),
        "interface_case_count": sum(row["interface_case_count"] for row in qualities),
        "condition_row_count": len(condition_rows),
        "generation_row_count": sum(row["generation_executed"] for row in condition_rows),
        "trace_condition_count": sum(row["trace_executed"] for row in condition_rows),
        "compensation_path_row_count": sum(row["compensation_path_row_count"] for row in qualities),
        "component_response_row_count": sum(row["component_response_row_count"] for row in qualities),
        "all_model_runs_valid": all(row["valid"] for row in qualities),
        "expected_interface_case_count": 720,
        "expected_condition_row_count": 9360,
        "selection_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
    }
    quality["valid"] = (
        quality["model_count"] == 3
        and quality["interface_case_count"] == 720
        and quality["condition_row_count"] == 9360
        and quality["all_model_runs_valid"]
    )
    write_json(root / "phase331_execution_quality.json", quality)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()
    if args.model:
        result = run_model(args.model, args.round, args.max_new_tokens, args.max_cases)
    elif args.collect:
        result = collect(args.round)
    else:
        raise SystemExit("Use --model MODEL or --collect")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
