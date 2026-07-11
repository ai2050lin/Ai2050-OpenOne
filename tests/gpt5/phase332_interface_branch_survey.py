#!/usr/bin/env python3
"""Map natural all-layer interface paths and freeze variable Phase332 member sets."""

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
import phase326_distributed_carrier_atlas as phase326  # noqa: E402
import phase330_global_atlas_survey as phase330_survey  # noqa: E402
from phase331_refined_mechanism_audit import answer_segment, target_match  # noqa: E402
from phase332_interface_branch_case_bank import INTERFACES, ROUND_DEFAULT  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase332"
SCHEMA_VERSION = "10.0.0"
OUT = ROOT / "tests/gpt5/result/phase332_interface_branch_atlas"
ROLES = ("source", "query", "answer_start")


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


class ParquetSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        normalized = [
            {
                **row,
                "interface_equivalent_to": row.get("interface_equivalent_to") or "",
            }
            for row in rows
        ]
        table = pa.Table.from_pylist(normalized)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table, row_group_size=32768)
        self.row_count += len(rows)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def continuation_ids(loaded: Any, case: dict[str, Any], text: str) -> list[int]:
    value = (" " + text) if case["interface"] == "raw_completion" else text
    ids = loaded.tokenizer(value, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Cannot tokenize {text!r}")
    return [int(item) for item in ids]


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
    return {
        "source": locate_span(tokenizer, case, case["source_fragments"][0], seq_len),
        "query": locate_span(tokenizer, case, case["query_fragment"], seq_len),
        "answer_start": (seq_len - 1, seq_len - 1),
    }


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def output_base(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "model": case["model"],
        "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "item_id": case["item_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "cohort": case["cohort"],
        "item_index": case["item_index"],
        "split": case["split"],
        "template_id": case["template_id"],
        "interface": case["interface"],
        "interface_equivalent_to": case["interface_equivalent_to"],
        "answer_phase": case["answer_phase"],
        "target_class": case["target_class"],
    }


def masks_for(
    tokenizer: Any, cases: list[dict[str, Any]], lengths: list[int], width: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, list[int]]]]:
    masks = {role: torch.zeros((len(cases), width), device=device) for role in ROLES}
    span_rows = []
    for index, (case, seq_len) in enumerate(zip(cases, lengths, strict=True)):
        spans = role_spans(tokenizer, case, seq_len)
        span_rows.append({role: [start, end] for role, (start, end) in spans.items()})
        for role, (start, end) in spans.items():
            masks[role][index, start : end + 1] = 1
    return masks, span_rows


def directions_for(loaded: Any, cases: list[dict[str, Any]]) -> tuple[torch.Tensor, list[list[int]], list[list[int]]]:
    weight = loaded.model.get_output_embeddings().weight.detach().float()
    directions = []
    targets = []
    distractors = []
    for case in cases:
        target = continuation_ids(loaded, case, case["target"])
        wrong = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
        direction = weight[target[0]] - weight[wrong].mean(dim=0)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
        directions.append(direction)
        targets.append(target)
        distractors.append(wrong)
    return torch.stack(directions).to(loaded.input_device), targets, distractors


def accumulate_candidate(
    accumulator: dict[tuple[Any, ...], dict[str, Any]], row: dict[str, Any]
) -> None:
    if row["split"] != "discovery":
        return
    key = (
        row["family_id"], row["mechanism_id"], row["interface"], row["component_type"],
        row["component_layer"], row["position_role"], row["component_index"],
        row["component_start"], row["component_end"],
    )
    bucket = accumulator.setdefault(key, {"values": [], "items": defaultdict(list), "templates": defaultdict(list)})
    value = float(row["approx_target_readout_contribution"])
    bucket["values"].append(value)
    bucket["items"][int(row["item_index"])].append(value)
    bucket["templates"][row["template_id"]].append(value)


@torch.inference_mode()
def trace_batch(
    loaded: Any,
    cases: list[dict[str, Any]],
    path_sink: ParquetSink,
    unit_sink: ParquetSink,
    accumulator: dict[tuple[Any, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "right"
    add_special = bool(cases[0]["tokenization_add_special_tokens"])
    encoded = tokenizer(
        [case["prompt"] for case in cases], return_tensors="pt", padding=True,
        truncation=True, max_length=256, add_special_tokens=add_special,
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    masks, span_rows = masks_for(
        tokenizer, cases, lengths, int(encoded["input_ids"].shape[1]), loaded.input_device
    )
    directions, targets, distractors = directions_for(loaded, cases)
    layers = get_layers(loaded.model)
    captures: dict[tuple[str, int], torch.Tensor] = {}
    component_outputs: dict[tuple[str, int], torch.Tensor] = {}
    residual_inputs: dict[int, torch.Tensor] = {}
    handles = []

    for layer_index, layer in enumerate(layers):
        o_proj, _heads, _head_dim = head_meta(loaded.model, layer_index)
        down_proj = phase326.get_down_proj(layer)
        if down_proj is None:
            raise TypeError(f"Missing MLP down projection at layer {layer_index}")

        def residual_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                residual_inputs[idx] = inputs[0].detach()

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            component_outputs[("attention", idx)] = component_tensor(output).detach()

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            component_outputs[("mlp", idx)] = component_tensor(output).detach()

        def attention_input(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                captures[("attention_head_input", idx)] = inputs[0].detach()

        def mlp_input(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                captures[("mlp_product_group", idx)] = inputs[0].detach()

        handles.extend([
            layer.register_forward_pre_hook(residual_pre),
            layer.self_attn.register_forward_hook(attention_post),
            layer.mlp.register_forward_hook(mlp_post),
            o_proj.register_forward_pre_hook(attention_input),
            down_proj.register_forward_pre_hook(mlp_input),
        ])
    try:
        output = loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    created_at = now()
    path_rows = []
    unit_rows = []

    def pool(tensor: torch.Tensor, role: str) -> torch.Tensor:
        mask = masks[role][:, : tensor.shape[1]]
        value = torch.einsum("bth,bt->bh", tensor.detach().float(), mask)
        return value / mask.sum(dim=1, keepdim=True).clamp_min(1)

    for layer_index, layer in enumerate(layers):
        for role in ROLES:
            residual = pool(residual_inputs[layer_index], role)
            attention = pool(component_outputs[("attention", layer_index)], role)
            mlp = pool(component_outputs[("mlp", layer_index)], role)
            values = (
                ("residual_cumulative_state", residual, False),
                ("attention_increment", attention, True),
                ("mlp_increment", mlp, True),
            )
            if layer_index + 1 < len(layers):
                delta = pool(residual_inputs[layer_index + 1], role) - residual
                values = (*values, ("residual_layer_increment", delta, True))
            for component_type, tensor, incremental in values:
                projections = (tensor * directions.to(tensor.device)).sum(dim=1)
                norms = torch.linalg.vector_norm(tensor, dim=1)
                for case_index, case in enumerate(cases):
                    path_rows.append({
                        **output_base(case),
                        "created_at": created_at,
                        "component_type": component_type,
                        "component_layer": layer_index,
                        "position_role": role,
                        "projection": round(float(projections[case_index].item()), 7),
                        "activation_norm": round(float(norms[case_index].item()), 7),
                        "incremental": incremental,
                        "evidence_level": "L2_natural_interface_path",
                    })

        o_proj, n_heads, head_dim = head_meta(loaded.model, layer_index)
        down_proj = phase326.get_down_proj(layer)
        ranges = phase326.group_ranges(int(down_proj.in_features))
        for role in ROLES:
            attn_input = pool(captures[("attention_head_input", layer_index)], role)
            attn_direction = directions.float() @ o_proj.weight.detach().float()
            attn_scores = (
                attn_input.view(len(cases), n_heads, head_dim)
                * attn_direction.view(len(cases), n_heads, head_dim)
            ).sum(dim=2)
            mlp_input = pool(captures[("mlp_product_group", layer_index)], role)
            mlp_direction = directions.float() @ down_proj.weight.detach().float()
            mlp_scores = torch.stack([
                (mlp_input[:, start:end] * mlp_direction[:, start:end]).sum(dim=1)
                for start, end in ranges
            ], dim=1)
            for case_index, case in enumerate(cases):
                base = output_base(case)
                for head in range(n_heads):
                    row = {
                        **base,
                        "created_at": created_at,
                        "component_type": "attention_head_input",
                        "component_layer": layer_index,
                        "position_role": role,
                        "component_index": head,
                        "component_start": head * head_dim,
                        "component_end": (head + 1) * head_dim,
                        "approx_target_readout_contribution": round(float(attn_scores[case_index, head].item()), 7),
                        "evidence_level": "L3_natural_interface_component_candidate",
                        "single_unit_causal": False,
                    }
                    unit_rows.append(row)
                    accumulate_candidate(accumulator, row)
                for group_index, (start, end) in enumerate(ranges):
                    row = {
                        **base,
                        "created_at": created_at,
                        "component_type": "mlp_product_group",
                        "component_layer": layer_index,
                        "position_role": role,
                        "component_index": group_index,
                        "component_start": start,
                        "component_end": end,
                        "approx_target_readout_contribution": round(float(mlp_scores[case_index, group_index].item()), 7),
                        "evidence_level": "L3_natural_interface_component_candidate",
                        "single_unit_causal": False,
                    }
                    unit_rows.append(row)
                    accumulate_candidate(accumulator, row)
    path_sink.write(path_rows)
    unit_sink.write(unit_rows)

    logits = torch.stack([
        output.logits[index, seq_len - 1].detach().float().cpu()
        for index, seq_len in enumerate(lengths)
    ])
    readout_rows = []
    for index, case in enumerate(cases):
        target_logit = float(logits[index, targets[index][0]].item())
        best_wrong = max(float(logits[index, token].item()) for token in distractors[index])
        rank = 1 + int((logits[index] > target_logit).sum().item())
        readout_rows.append({
            **output_base(case),
            "created_at": created_at,
            "sequence_length": lengths[index],
            "role_spans": json.dumps(span_rows[index], sort_keys=True),
            "target_first_token_id": targets[index][0],
            "target_logit": round(target_logit, 7),
            "best_distractor_logit": round(best_wrong, 7),
            "target_margin": round(target_logit - best_wrong, 7),
            "target_full_vocabulary_rank": rank,
            "target_in_top50": rank <= 50,
            "candidate_winner_is_target": target_logit >= best_wrong,
        })
    del output, logits, captures, component_outputs, residual_inputs, encoded
    return readout_rows


@torch.inference_mode()
def phrase_rows(loaded: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        encoded = loaded.tokenizer(
            case["prompt"], return_tensors="pt", truncation=True, max_length=256,
            add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        )
        encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
        answer = continuation_ids(loaded, case, case["target"])
        prompt_length = int(encoded["input_ids"].shape[1])
        append = torch.tensor([answer], dtype=encoded["input_ids"].dtype, device=loaded.input_device)
        input_ids = torch.cat([encoded["input_ids"], append], dim=1)
        attention_mask = torch.cat([
            encoded["attention_mask"], torch.ones_like(append, dtype=encoded["attention_mask"].dtype)
        ], dim=1)
        output = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
        values = [
            float(log_probs[prompt_length + offset - 1, token].item())
            for offset, token in enumerate(answer)
        ]
        rows.append({
            "case_id": case["case_id"],
            "target_token_ids": json.dumps(answer),
            "target_token_count": len(answer),
            "target_phrase_logprob": round(sum(values), 7),
            "target_phrase_mean_logprob": round(mean(values), 7),
        })
    return rows


@torch.inference_mode()
def generation_rows(loaded: Any, cases: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    rows = []
    tokenizer = loaded.tokenizer
    for case in cases:
        encoded = tokenizer(
            case["prompt"], return_tensors="pt", truncation=True, max_length=256,
            add_special_tokens=bool(case["tokenization_add_special_tokens"]),
        )
        encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
        generated = loaded.model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        suffix = generated[0, encoded["input_ids"].shape[1] :]
        ids = [int(value) for value in suffix.tolist()]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        segment = answer_segment(text)
        think_closed = "</think>" in text or case["answer_phase"] != "think_start"
        rows.append({
            "case_id": case["case_id"],
            "generated_text": text,
            "generated_token_ids": json.dumps(ids),
            "generated_token_count": len(ids),
            "answer_phase_reached": think_closed,
            "target_anywhere_match": target_match(text, case["target_aliases"]),
            "target_answer_segment_match": target_match(segment, case["target_aliases"]),
            "protocol_success_answer_segment": phase330_survey.protocol_ok(case, segment),
            "behavior_success": target_match(segment, case["target_aliases"]) and phase330_survey.protocol_ok(case, segment),
            "eos_emitted": tokenizer.eos_token_id in ids,
        })
    return rows


def freeze_member_sets(
    model: str,
    accumulator: dict[tuple[Any, ...], dict[str, Any]],
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = protocol["component_selection"]
    candidate_rows = []
    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for key, bucket in accumulator.items():
        family, mechanism, interface, component, layer, role, index, start, end = key
        values = bucket["values"]
        mean_value = mean(values)
        sign = 1 if mean_value >= 0 else -1
        item_consistency = sum(
            (mean(item_values) >= 0) == (sign > 0)
            for item_values in bucket["items"].values()
        ) / len(bucket["items"])
        template_consistency = sum(
            (mean(template_values) >= 0) == (sign > 0)
            for template_values in bucket["templates"].values()
        ) / len(bucket["templates"])
        row = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "interface": interface,
            "component_type": component,
            "component_layer": int(layer),
            "position_role": role,
            "component_index": int(index),
            "component_start": int(start),
            "component_end": int(end),
            "discovery_case_count": len(values),
            "mean_contribution": round(mean_value, 7),
            "item_sign_consistency": round(item_consistency, 7),
            "template_sign_consistency": round(template_consistency, 7),
            "selection_score": round(abs(mean_value) * item_consistency * template_consistency, 7),
            "stable": item_consistency >= settings["item_sign_consistency_min"],
            "selection_split": "discovery_only",
            "single_unit_causal": False,
        }
        candidate_rows.append(row)
        grouped[(family, mechanism, interface, component, role)].append(row)
    selected = []
    selected_keys: dict[tuple[str, str, str], set[tuple[Any, ...]]] = defaultdict(set)
    for group_key, values in grouped.items():
        stable = sorted(
            (row for row in values if row["stable"]),
            key=lambda row: row["selection_score"], reverse=True,
        )
        count = min(
            settings["max_members_per_component_role"],
            max(1, math.ceil(len(stable) * settings["top_fraction_per_component_role"])),
        ) if stable else 0
        for rank, row in enumerate(stable[:count]):
            member = dict(row)
            member.update(set_type="interface_top", set_rank=rank, heldout_used=False)
            selected.append(member)
            identity = (
                row["component_type"], row["component_layer"], row["position_role"], row["component_index"],
                row["component_start"], row["component_end"],
            )
            selected_keys[(row["family_id"], row["mechanism_id"], row["interface"])].add(identity)

    result = [*selected]
    mechanisms = sorted({(row["family_id"], row["mechanism_id"]) for row in selected})
    for family, mechanism in mechanisms:
        unique_interfaces = ["raw_completion", "native_chat", "chat_no_think", "answer_aligned_chat"]
        if model == "glm4":
            unique_interfaces.remove("chat_no_think")
        sets = [selected_keys[(family, mechanism, interface)] for interface in unique_interfaces]
        shared = set.intersection(*sets) if sets else set()
        for identity in sorted(shared):
            source = next(
                row for row in selected
                if row["family_id"] == family and row["mechanism_id"] == mechanism
                and row["interface"] == unique_interfaces[0]
                and (
                    row["component_type"], row["component_layer"], row["position_role"], row["component_index"],
                    row["component_start"], row["component_end"],
                ) == identity
            )
            result.append({**source, "interface": "shared_all_unique_interfaces", "set_type": "shared_skeleton"})
        for interface in unique_interfaces:
            others = set().union(*[
                selected_keys[(family, mechanism, other)]
                for other in unique_interfaces if other != interface
            ])
            branch = selected_keys[(family, mechanism, interface)] - others
            for identity in sorted(branch):
                source = next(
                    row for row in selected
                    if row["family_id"] == family and row["mechanism_id"] == mechanism
                    and row["interface"] == interface
                    and (
                        row["component_type"], row["component_layer"], row["position_role"], row["component_index"],
                        row["component_start"], row["component_end"],
                    ) == identity
                )
                result.append({**source, "set_type": "interface_branch"})
    return result


def run_model(model: str, round_name: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "survey" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    cases = [row for row in read_jsonl(root / "phase332_registered_cases.jsonl") if row["model"] == model]
    protocol = read_json(root / "phase332_registered_protocol.json")
    path_sink = ParquetSink(model_dir / "natural_path_rows.parquet")
    unit_sink = ParquetSink(model_dir / "natural_unit_rows.parquet")
    accumulator: dict[tuple[Any, ...], dict[str, Any]] = {}
    readout_rows: list[dict[str, Any]] = []
    phrase_output: list[dict[str, Any]] = []
    generation_output: list[dict[str, Any]] = []
    loaded = None
    try:
        loaded = load_probe_model(model)
        ordered = []
        for interface in INTERFACES:
            ordered.extend(row for row in cases if row["interface"] == interface)
        for batch_index, batch in enumerate(chunks(ordered, batch_size), 1):
            readout_rows.extend(trace_batch(loaded, batch, path_sink, unit_sink, accumulator))
            phrase_output.extend(phrase_rows(loaded, batch))
            if batch_index % 8 == 0:
                print(json.dumps({
                    "quality_only": True, "model": model,
                    "survey_cases": min(batch_index * batch_size, len(ordered)),
                    "total_cases": len(ordered), "path_rows": path_sink.row_count,
                    "unit_rows": unit_sink.row_count,
                }), flush=True)
        heldout = [row for row in ordered if row["split"] == "heldout"]
        for index, case in enumerate(heldout, 1):
            generation_output.extend(generation_rows(loaded, [case], max_new_tokens))
            if index % 16 == 0:
                print(json.dumps({
                    "quality_only": True, "model": model,
                    "heldout_generations": index, "total_generations": len(heldout),
                }), flush=True)
        phrase_map = {row["case_id"]: row for row in phrase_output}
        generation_map = {row["case_id"]: row for row in generation_output}
        baseline_rows = []
        for row in readout_rows:
            combined = {**row, **phrase_map[row["case_id"]]}
            combined.update(generation_map.get(row["case_id"], {
                "generated_text": None, "generated_token_ids": None, "generated_token_count": None,
                "answer_phase_reached": None, "target_anywhere_match": None,
                "target_answer_segment_match": None, "protocol_success_answer_segment": None,
                "behavior_success": None, "eos_emitted": None,
            }))
            baseline_rows.append(combined)
        member_sets = freeze_member_sets(model, accumulator, protocol)
        write_jsonl(model_dir / "baseline_rows.jsonl", baseline_rows)
        write_parquet(model_dir / "baseline_rows.parquet", baseline_rows)
        write_jsonl(model_dir / "member_sets.jsonl", member_sets)
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "case_count": len(cases),
            "readout_row_count": len(readout_rows),
            "phrase_row_count": len(phrase_output),
            "generation_row_count": len(generation_output),
            "natural_path_row_count": path_sink.row_count,
            "natural_unit_row_count": unit_sink.row_count,
            "member_set_row_count": len(member_sets),
            "shared_skeleton_member_count": sum(row["set_type"] == "shared_skeleton" for row in member_sets),
            "interface_branch_member_count": sum(row["set_type"] == "interface_branch" for row in member_sets),
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(cases) == 384 and len(readout_rows) == 384 and len(generation_output) == 192,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        path_sink.close()
        unit_sink.close()
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(
        run_model(args.model, args.round, args.batch_size, args.max_new_tokens),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
