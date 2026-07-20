#!/usr/bin/env python3
"""Test whether query-terminal condition messages cause late fact selection."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_relation_competition_behavior import classify  # noqa: E402
from phase569_role_position_utils import role_positions  # noqa: E402
from phase573_coarse_message_causal import (  # noqa: E402
    edge_contribution,
    reconstructed_receiver,
    replace_primary,
)
from phase573_natural_transition_behavior import balanced_worlds, stable_expected  # noqa: E402
from phase574_query_source_behavior import generate_batch  # noqa: E402
import phase574_query_source_protocol as protocol  # noqa: E402
import phase574_query_source_causal_protocol as causal_protocol  # noqa: E402
import phase574_query_source_trace as trace  # noqa: E402


MODEL = causal_protocol.MODEL
OUT_DIR = protocol.OUT_DIR
VARIANTS = ("base", "relation_swap", "object_swap", "order_swap")
VARIANT_INDEX = {variant: index for index, variant in enumerate(VARIANTS)}
TARGET_ROLES = (
    "target_fact_object", "target_fact_relation", "target_fact_value",
)
OTHER_ROLES = (
    "other_fact_object", "other_fact_relation", "other_fact_value",
)
BEHAVIOR_ROWS_PATH = OUT_DIR / "phase574_qwen3_causal_behavior_rows.jsonl.gz"
CAUSAL_ROWS_PATH = OUT_DIR / "phase574_qwen3_query_source_causal_rows.jsonl.gz"
GENERATION_ROWS_PATH = OUT_DIR / "phase574_qwen3_query_source_generation_rows.jsonl.gz"
REGISTRY_PATH = OUT_DIR / "phase574_qwen3_causal_registry.json"
SUMMARY_PATH = OUT_DIR / "phase574_qwen3_query_source_causal_summary.json"
DECISION_PATH = OUT_DIR / "phase574_query_source_causal_decision.json"
CONTRACT_PATH = OUT_DIR / "phase574_qwen3_query_source_causal_contract.json"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def mean(values: list[float]) -> float:
    return finite(sum(values) / max(1, len(values)))


def deterministic_roll(key: str, hidden: int) -> int:
    value = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    return 1 + value % max(1, hidden - 1)


def candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate["candidate_id"])


def load_case_bank() -> dict[str, dict[str, Any]]:
    bank = {
        row["case_id"]: row
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["split"] in protocol.CAUSAL_SPLITS
    }
    expected = len(protocol.CAUSAL_SPLITS) * 1024 * len(VARIANTS)
    if len(bank) != expected or any(row["sealed"] for row in bank.values()):
        raise RuntimeError(f"Phase574 causal denominator drift: {len(bank)}/{expected}")
    return bank


def run_behavior_stage(
    loaded: Any,
    case_bank: dict[str, dict[str, Any]],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    loaded.tokenizer.padding_side = "left"
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)["causal_behavior"]
    by_world_variant = {
        (row["base_case_id"], row["variant"]): row for row in case_bank.values()
    }
    base_rows = {
        row["base_case_id"]: row
        for row in case_bank.values() if row["variant"] == "base"
    }
    output: list[dict[str, Any]] = []
    selected: dict[str, list[str]] = {}
    diagnostics: dict[str, Any] = {}
    for split in protocol.CAUSAL_SPLITS:
        relation_rows = sorted(
            [
                row for row in case_bank.values()
                if row["split"] == split
                and row["variant"] in ("base", "relation_swap")
            ],
            key=lambda row: row["case_id"],
        )
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(relation_rows), frozen["batch_size"]):
                output.extend(generate_batch(
                    loaded, MODEL, relation_rows[start:start + frozen["batch_size"]],
                    repeat, max_new_tokens,
                ))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase574 {split} "
                f"relation/{repeat} {len(relation_rows)}/{len(relation_rows)}",
                flush=True,
            )
        by_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output
        }
        world_ids = sorted({row["base_case_id"] for row in relation_rows})
        relation_eligible = [
            base_id for base_id in world_ids
            if stable_expected(by_repeat, f"{base_id}_base")
            and stable_expected(by_repeat, f"{base_id}_relation_swap")
        ]
        if len(relation_eligible) < frozen["relation_minimum_each_split"]:
            raise RuntimeError(
                f"Phase574 causal behavior relation gate failed: "
                f"{split}/{len(relation_eligible)}"
            )
        controls_selected = balanced_worlds(
            base_rows, relation_eligible, frozen["control_screen_cap_each_split"]
        )
        controls = sorted(
            [
                by_world_variant[(base_id, variant)]
                for base_id in controls_selected
                for variant in ("object_swap", "order_swap")
            ],
            key=lambda row: row["case_id"],
        )
        for repeat in ("noop1", "noop2"):
            for start in range(0, len(controls), frozen["batch_size"]):
                output.extend(generate_batch(
                    loaded, MODEL, controls[start:start + frozen["batch_size"]],
                    repeat, max_new_tokens,
                ))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase574 {split} "
                f"controls/{repeat} {len(controls)}/{len(controls)}",
                flush=True,
            )
        by_repeat = {
            (row["case_id"], row["execution_repeat"]): row for row in output
        }
        all_axis = [
            base_id for base_id in controls_selected
            if stable_expected(by_repeat, f"{base_id}_object_swap")
            and stable_expected(by_repeat, f"{base_id}_order_swap")
        ]
        selected[split] = balanced_worlds(
            base_rows, all_axis, frozen["all_axis_worlds_selected_each_split"]
        )
        if len(selected[split]) != frozen["all_axis_worlds_selected_each_split"]:
            raise RuntimeError(
                f"Phase574 causal behavior all-axis gate failed: "
                f"{split}/{len(selected[split])}"
            )
        diagnostics[split] = {
            "relation_qualified_world_count": len(relation_eligible),
            "control_screen_world_count": len(controls_selected),
            "all_axis_qualified_world_count": len(all_axis),
            "selected_world_count": len(selected[split]),
        }
    return output, selected, diagnostics


def selected_worlds(
    case_bank: dict[str, dict[str, Any]], selected: dict[str, list[str]], split: str
) -> list[list[dict[str, Any]]]:
    by_world: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in case_bank.values():
        if row["split"] == split and row["base_case_id"] in set(selected[split]):
            by_world[row["base_case_id"]][row["variant"]] = row
    worlds = []
    for base_id in selected[split]:
        variants = by_world.get(base_id, {})
        if set(variants) != set(VARIANTS):
            raise RuntimeError(f"Phase574 incomplete causal world: {base_id}")
        worlds.append([variants[variant] for variant in VARIANTS])
    return worlds


def prepare_flat_batch(
    tokenizer: Any, worlds: list[list[dict[str, Any]]], padding_side: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]:
    tokenizer.padding_side = padding_side
    rows = [row for world in worlds for row in world]
    prompts = [render_chat(tokenizer, MODEL, row["raw_prompt"]) for row in rows]
    individual = [
        role_positions(tokenizer, prompt, row)
        for prompt, row in zip(prompts, rows)
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    width = int(encoded["input_ids"].shape[1])
    positions = []
    for batch_index, (ids, roles) in enumerate(individual):
        active = encoded["input_ids"][batch_index][
            encoded["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active] != ids:
            raise RuntimeError("Phase574 causal tokenization drift")
        offset = width - len(ids) if padding_side == "left" else 0
        positions.append({
            "offset": offset,
            "roles_unpadded": roles,
            "query_terminal": roles["query_terminal"][-1] + offset,
            "answer_boundary": roles["answer_boundary"][-1] + offset,
            "query_relation": [pos + offset for pos in roles["query_relation"]],
        })
    position_ids = encoded["attention_mask"].long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    encoded["position_ids"] = position_ids

    meta = []
    for world_index, world in enumerate(worlds):
        indices = {
            variant: world_index * len(VARIANTS) + VARIANT_INDEX[variant]
            for variant in VARIANTS
        }
        base_index = indices["base"]
        base_offset = int(positions[base_index]["offset"])

        def source_for(variant: str) -> list[int]:
            donor_roles = positions[indices[variant]]["roles_unpadded"]
            return sorted(
                pos + base_offset
                for role in TARGET_ROLES
                for pos in donor_roles[role]
            )

        meta.append({
            "world_index": world_index,
            "base_case_id": world[0]["base_case_id"],
            "indices": indices,
            "query_terminal": positions[base_index]["query_terminal"],
            "answer_boundary": positions[base_index]["answer_boundary"],
            "recipient_source": source_for("base"),
            "relation_source": source_for("relation_swap"),
            "object_source": source_for("object_swap"),
            "recipient_target": world[0]["target"],
            "relation_target": world[1]["target"],
            "object_target": world[2]["target"],
            "order_target": world[3]["target"],
        })
    return rows, positions, meta, encoded


def source_message(
    module: Any,
    weights: torch.Tensor,
    values: torch.Tensor,
    batch_index: int,
    receiver: int,
    source_positions: list[int],
) -> torch.Tensor:
    return edge_contribution(
        module, weights, values, batch_index, receiver, source_positions
    )


def downstream_capture(
    module: Any,
    hidden: torch.Tensor,
    weights: torch.Tensor,
    meta: list[dict[str, Any]],
    batch_indices: list[int],
) -> list[dict[str, Any]]:
    batch, sequence, _ = hidden.shape
    values = module.v_proj(hidden).view(
        batch, sequence, -1, module.head_dim
    ).transpose(1, 2)
    values = values.repeat_interleave(module.num_key_value_groups, dim=1)
    output = []
    for local, model_index in enumerate(batch_indices):
        item = meta[local]
        receiver = int(item["answer_boundary"])

        def mass(source: list[int]) -> float:
            return finite(float(
                weights[model_index, :, receiver, source]
                .float().sum(dim=-1).mean().item()
            ))

        messages = {
            name: source_message(
                module, weights, values, model_index, receiver, item[f"{name}_source"]
            ).detach()
            for name in ("recipient", "relation", "object")
        }
        output.append({
            "recipient_route_mass": mass(item["recipient_source"]),
            "relation_route_mass": mass(item["relation_source"]),
            "object_route_mass": mass(item["object_source"]),
            "recipient_message": messages["recipient"],
            "relation_message": messages["relation"],
            "object_message": messages["object"],
        })
    return output


def outcome_rows(
    result: Any,
    downstream: list[dict[str, Any]],
    worlds: list[list[dict[str, Any]]],
    meta: list[dict[str, Any]],
    model_indices: list[int],
) -> list[dict[str, Any]]:
    output = []
    for local, model_index in enumerate(model_indices):
        row = worlds[local][0]
        boundary = int(meta[local]["answer_boundary"])
        logits = result.logits[model_index, boundary].float()
        scores = {
            value: finite(float(logits[token_ids[0]].item()))
            for value, token_ids in row["candidate_token_ids_by_model"][MODEL].items()
        }
        item = downstream[local]
        output.append({
            "candidate_scores": scores,
            "candidate_winner": max(scores, key=scores.get),
            "recipient_route_mass": item["recipient_route_mass"],
            "relation_route_mass": item["relation_route_mass"],
            "object_route_mass": item["object_route_mass"],
            "relation_route_switch_margin": (
                item["relation_route_mass"] - item["recipient_route_mass"]
            ),
            "object_route_switch_margin": (
                item["object_route_mass"] - item["recipient_route_mass"]
            ),
            "recipient_message_norm": finite(float(
                item["recipient_message"].float().norm().item()
            )),
            "relation_message_norm": finite(float(
                item["relation_message"].float().norm().item()
            )),
            "object_message_norm": finite(float(
                item["object_message"].float().norm().item()
            )),
        })
    return output


def capture_natural(
    loaded: Any,
    layers: list[Any],
    worlds: list[list[dict[str, Any]]],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor],
    dict[int, dict[str, torch.Tensor]], list[dict[str, Any]], float,
]:
    rows, positions, meta, encoded_cpu = prepare_flat_batch(
        loaded.tokenizer, worlds, "right"
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    captures: dict[int, dict[str, torch.Tensor]] = {}
    downstream: list[dict[str, Any]] = []
    errors: list[float] = []

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            nonlocal downstream
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or not isinstance(output, tuple) or output[1] is None:
                raise RuntimeError("Phase574 natural causal capture requires attention weights")
            primary, weights = output[0], output[1]
            batch, sequence, _ = hidden.shape
            values = module.v_proj(hidden).view(
                batch, sequence, -1, module.head_dim
            ).transpose(1, 2)
            values = values.repeat_interleave(module.num_key_value_groups, dim=1)
            relation_messages = []
            terminal_outputs = []
            for batch_index in range(batch):
                receiver = int(positions[batch_index]["query_terminal"])
                relation_messages.append(source_message(
                    module, weights, values, batch_index, receiver,
                    positions[batch_index]["query_relation"],
                ))
                terminal_outputs.append(primary[batch_index, receiver])
                reconstructed = reconstructed_receiver(
                    module, weights, values, batch_index, receiver
                )
                actual = primary[batch_index, receiver]
                errors.append(float(
                    (reconstructed.float() - actual.float()).norm().item()
                    / max(actual.float().norm().item(), 1e-8)
                ))
            captures[layer_index] = {
                "query_relation_value_message": torch.stack(
                    relation_messages
                ).detach(),
                "query_terminal_attention_output": torch.stack(
                    terminal_outputs
                ).detach(),
            }
            if layer_index == 24:
                base_indices = [
                    item["indices"]["base"] for item in meta
                ]
                downstream = downstream_capture(
                    module, hidden, weights, meta, base_indices
                )
        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index in range(5, 25)
    ]
    with torch.inference_mode():
        result = loaded.model(
            **encoded, use_cache=False, output_attentions=True, return_dict=True
        )
    for handle in handles:
        handle.remove()
    if set(captures) != set(range(5, 25)) or len(downstream) != len(worlds):
        raise RuntimeError("Phase574 natural causal capture drift")
    base_indices = [item["indices"]["base"] for item in meta]
    outcomes = outcome_rows(result, downstream, worlds, meta, base_indices)
    del result, encoded
    return rows, positions, encoded_cpu, captures, outcomes, max(errors)


def candidate_deltas(
    captures: dict[int, dict[str, torch.Tensor]],
    meta: list[dict[str, Any]],
    candidate: dict[str, Any],
    condition: str,
) -> dict[int, torch.Tensor]:
    component = candidate["component"]
    mapping = read_json(causal_protocol.CAUSAL_PROTOCOL)[
        "wrong_depth_layer_mapping"
    ]
    deltas: dict[int, torch.Tensor] = {}
    for layer_index in candidate["patch_layers"]:
        vectors = captures[layer_index][component]
        rows = []
        for item in meta:
            indices = item["indices"]
            recipient = vectors[indices["base"]]
            relation = vectors[indices["relation_swap"]]
            obj = vectors[indices["object_swap"]]
            order = vectors[indices["order_swap"]]
            if condition in ("recipient_remove", "recipient_remove_restore"):
                delta = -recipient
            elif condition == "relation_donor_replace":
                delta = relation - recipient
            elif condition == "object_donor_replace":
                delta = obj - recipient
            elif condition == "order_donor_replace":
                delta = order - recipient
            elif condition == "wrong_depth_relation_replace":
                wrong_layer = int(mapping[str(layer_index)])
                wrong_relation = captures[wrong_layer][component][
                    indices["relation_swap"]
                ]
                delta = wrong_relation - recipient
            elif condition == "wrong_position_relation_replace":
                delta = relation - recipient
            elif condition == "channel_roll_relation_replace":
                shift = deterministic_roll(
                    f"{item['base_case_id']}|{candidate_id(candidate)}|{layer_index}",
                    int(relation.shape[-1]),
                )
                delta = torch.roll(relation, shift, dims=-1) - recipient
            else:
                raise RuntimeError(f"Unknown Phase574 condition: {condition}")
            rows.append(delta)
        deltas[layer_index] = torch.stack(rows).to(
            dtype=next(iter(captures[layer_index].values())).dtype
        )
    return deltas


def run_patched(
    loaded: Any,
    layers: list[Any],
    worlds: list[list[dict[str, Any]]],
    encoded_cpu: dict[str, torch.Tensor],
    captures: dict[int, dict[str, torch.Tensor]],
    meta: list[dict[str, Any]],
    candidate: dict[str, Any],
    condition: str,
) -> list[dict[str, Any]]:
    encoded = {
        key: value.to(loaded.input_device) for key, value in encoded_cpu.items()
    }
    deltas = candidate_deltas(captures, meta, candidate, condition)
    normal_receiver = condition != "wrong_position_relation_replace"
    patch_positions = torch.tensor(
        [
            item["query_terminal"] if normal_receiver else item["answer_boundary"]
            for item in meta
        ],
        dtype=torch.long,
        device=loaded.input_device,
    )
    batch_indices = torch.tensor(
        [item["indices"]["base"] for item in meta],
        dtype=torch.long,
        device=loaded.input_device,
    )
    handles: list[Any] = []

    def make_patch(layer_index: int, sign: float):
        delta = deltas[layer_index].to(
            device=loaded.input_device,
            dtype=next(loaded.model.parameters()).dtype,
        ) * sign

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            primary = output[0].clone() if isinstance(output, tuple) else output.clone()
            primary[batch_indices, patch_positions, :] += delta
            return replace_primary(output, primary)
        return hook

    for layer_index in candidate["patch_layers"]:
        handles.append(
            layers[layer_index].self_attn.register_forward_hook(
                make_patch(layer_index, 1.0)
            )
        )
        if condition == "recipient_remove_restore":
            # The second hook restores exactly what the first hook removed.
            handles.append(
                layers[layer_index].self_attn.register_forward_hook(
                    make_patch(layer_index, -1.0)
                )
            )

    downstream: list[dict[str, Any]] = []

    def outcome_hook(
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        nonlocal downstream
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or not isinstance(output, tuple) or output[1] is None:
            raise RuntimeError("Phase574 patched outcome requires attention weights")
        downstream = downstream_capture(
            module, hidden, output[1], meta,
            [item["indices"]["base"] for item in meta],
        )

    handles.append(
        layers[24].self_attn.register_forward_hook(outcome_hook, with_kwargs=True)
    )
    with torch.inference_mode():
        result = loaded.model(
            **encoded, use_cache=False, output_attentions=True, return_dict=True
        )
    for handle in handles:
        handle.remove()
    if len(downstream) != len(worlds):
        raise RuntimeError("Phase574 patched downstream capture drift")
    outcomes = outcome_rows(
        result, downstream, worlds, meta,
        [item["indices"]["base"] for item in meta],
    )
    del result, encoded, deltas
    return outcomes


def causal_row(
    split: str,
    world: list[dict[str, Any]],
    candidate: dict[str, Any],
    condition: str,
    baseline: dict[str, Any],
    outcome: dict[str, Any],
) -> dict[str, Any]:
    recipient_target = world[0]["target"]
    relation_target = world[1]["target"]
    object_target = world[2]["target"]
    baseline_scores = baseline["candidate_scores"]
    scores = outcome["candidate_scores"]
    baseline_recipient_margin = (
        baseline_scores[recipient_target] - baseline_scores[relation_target]
    )
    recipient_margin = scores[recipient_target] - scores[relation_target]
    return {
        "schema_version": "phase574_query_source_causal_row.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "split": split,
        "base_case_id": world[0]["base_case_id"],
        "candidate_id": candidate_id(candidate),
        "component": candidate["component"],
        "band_start": candidate["band_start"],
        "band_end": candidate["band_end"],
        "condition": condition,
        "recipient_target": recipient_target,
        "relation_donor_target": relation_target,
        "object_donor_target": object_target,
        "baseline_candidate_scores": baseline_scores,
        "intervention_candidate_scores": scores,
        "baseline_candidate_winner": baseline["candidate_winner"],
        "intervention_candidate_winner": outcome["candidate_winner"],
        "baseline_relation_route_switch_margin": baseline[
            "relation_route_switch_margin"
        ],
        "intervention_relation_route_switch_margin": outcome[
            "relation_route_switch_margin"
        ],
        "relation_route_switch_effect": (
            outcome["relation_route_switch_margin"]
            - baseline["relation_route_switch_margin"]
        ),
        "baseline_object_route_switch_margin": baseline[
            "object_route_switch_margin"
        ],
        "intervention_object_route_switch_margin": outcome[
            "object_route_switch_margin"
        ],
        "object_route_switch_effect": (
            outcome["object_route_switch_margin"]
            - baseline["object_route_switch_margin"]
        ),
        "baseline_relation_logit_switch_margin": (
            baseline_scores[relation_target] - baseline_scores[recipient_target]
        ),
        "intervention_relation_logit_switch_margin": (
            scores[relation_target] - scores[recipient_target]
        ),
        "relation_logit_switch_effect": (
            scores[relation_target] - scores[recipient_target]
            - baseline_scores[relation_target] + baseline_scores[recipient_target]
        ),
        "baseline_object_logit_switch_margin": (
            baseline_scores[object_target] - baseline_scores[recipient_target]
        ),
        "intervention_object_logit_switch_margin": (
            scores[object_target] - scores[recipient_target]
        ),
        "object_logit_switch_effect": (
            scores[object_target] - scores[recipient_target]
            - baseline_scores[object_target] + baseline_scores[recipient_target]
        ),
        "recipient_margin_damage": baseline_recipient_margin - recipient_margin,
        "maximum_candidate_logit_delta": max(
            abs(scores[value] - baseline_scores[value]) for value in scores
        ),
        "baseline_recipient_message_norm": baseline["recipient_message_norm"],
        "intervention_recipient_message_norm": outcome["recipient_message_norm"],
        "baseline_relation_message_norm": baseline["relation_message_norm"],
        "intervention_relation_message_norm": outcome["relation_message_norm"],
        "relation_donor_target_wins": outcome["candidate_winner"] == relation_target,
        "object_donor_target_wins": outcome["candidate_winner"] == object_target,
        "post_softmax_value_message_measured": True,
        "query_condition_intervention": condition != "natural_baseline",
        "output_embedding_direction_used": False,
        "head_channel_parameter_neuron_scan_executed": False,
        "sealed": False,
    }


def run_causal_split(
    loaded: Any,
    layers: list[Any],
    split: str,
    worlds: list[list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    world_batch_size = int(frozen["causal_world_batch_size"])
    conditions = [
        condition for condition in frozen["conditions"]
        if condition != "natural_baseline"
    ]
    output: list[dict[str, Any]] = []
    reconstruction_max = 0.0
    for start in range(0, len(worlds), world_batch_size):
        batch_worlds = worlds[start:start + world_batch_size]
        rows, positions, encoded_cpu, captures, baseline, error = capture_natural(
            loaded, layers, batch_worlds
        )
        reconstruction_max = max(reconstruction_max, error)
        _, _, meta, _ = prepare_flat_batch(
            loaded.tokenizer, batch_worlds, "right"
        )
        for candidate in candidates:
            for local, world in enumerate(batch_worlds):
                output.append(causal_row(
                    split, world, candidate, "natural_baseline",
                    baseline[local], baseline[local],
                ))
            for condition in conditions:
                outcomes = run_patched(
                    loaded, layers, batch_worlds, encoded_cpu, captures, meta,
                    candidate, condition,
                )
                for local, world in enumerate(batch_worlds):
                    output.append(causal_row(
                        split, world, candidate, condition,
                        baseline[local], outcomes[local],
                    ))
        del rows, positions, encoded_cpu, captures, baseline
        print(
            f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase574 {split} causal "
            f"{min(start + world_batch_size, len(worlds))}/{len(worlds)} "
            f"candidates={len(candidates)}",
            flush=True,
        )
    return output, reconstruction_max


def sign_flip_audit(
    values: dict[str, float], permutations: int, namespace: str
) -> dict[str, Any]:
    ordered = sorted(values)
    observed = mean([values[key] for key in ordered])
    at_least = 0
    for permutation in range(permutations):
        permuted = []
        for key in ordered:
            digest = hashlib.sha256(
                f"Phase574|{namespace}|{permutation}|{key}".encode("utf-8")
            ).digest()
            permuted.append(values[key] if digest[0] & 1 else -values[key])
        at_least += int(mean(permuted) >= observed)
    return {
        "observed_mean": observed,
        "permutation_count": permutations,
        "count_at_least_observed": at_least,
        "smoothed_tail_fraction": (at_least + 1) / (permutations + 1),
    }


def candidate_metrics(
    rows: list[dict[str, Any]], split: str, candidate: str
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["split"] == split and row["candidate_id"] == candidate
    ]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_condition[row["condition"]].append(row)
    relation_rows = by_condition["relation_donor_replace"]
    relation_effects = [row["relation_route_switch_effect"] for row in relation_rows]
    control_names = (
        "object_donor_replace", "order_donor_replace",
        "wrong_depth_relation_replace", "wrong_position_relation_replace",
        "channel_roll_relation_replace",
    )
    control_means = {
        name: mean([
            row["relation_route_switch_effect"] for row in by_condition[name]
        ])
        for name in control_names
    }
    relation_mean = mean(relation_effects)
    gaps = {
        name: relation_mean - abs(value) for name, value in control_means.items()
    }
    remove_damage = mean([
        row["recipient_margin_damage"] for row in by_condition["recipient_remove"]
    ])
    restore_max = max(
        row["maximum_candidate_logit_delta"]
        for row in by_condition["recipient_remove_restore"]
    )
    per_world: dict[str, float] = {}
    controls_by_world = {
        name: {
            row["base_case_id"]: row["relation_route_switch_effect"]
            for row in by_condition[name]
        }
        for name in control_names
    }
    for row in relation_rows:
        base_id = row["base_case_id"]
        per_world[base_id] = row["relation_route_switch_effect"] - max(
            abs(controls_by_world[name][base_id]) for name in control_names
        )
    rule = read_json(causal_protocol.CAUSAL_PROTOCOL)["selection_rule"]
    eligible = (
        relation_mean >= rule["eligible_candidate_requires_relation_effect_mean_minimum"]
        and sum(value > 0 for value in relation_effects) / max(1, len(relation_effects))
        >= rule["eligible_candidate_requires_relation_effect_positive_rate"]
        and min(gaps.values())
        >= rule["eligible_candidate_requires_relation_vs_each_control_gap_minimum"]
        and remove_damage
        >= rule["eligible_candidate_requires_remove_recipient_margin_damage_mean"]
        and restore_max
        <= rule["eligible_candidate_requires_restore_max_candidate_logit_delta"]
    )
    return {
        "case_count": len(relation_rows),
        "relation_route_switch_effect_mean": relation_mean,
        "relation_route_switch_effect_positive_rate": sum(
            value > 0 for value in relation_effects
        ) / max(1, len(relation_effects)),
        "relation_logit_switch_effect_mean": mean([
            row["relation_logit_switch_effect"] for row in relation_rows
        ]),
        "relation_donor_target_win_rate": sum(
            row["relation_donor_target_wins"] for row in relation_rows
        ) / max(1, len(relation_rows)),
        "control_relation_route_effect_means": control_means,
        "relation_vs_control_gaps": gaps,
        "recipient_remove_margin_damage_mean": remove_damage,
        "restore_maximum_candidate_logit_delta": restore_max,
        "eligible": eligible,
        "world_level_relation_minus_strongest_control": per_world,
    }


def familywise_pipeline_audit(
    metrics: dict[str, dict[str, Any]], permutations: int
) -> dict[str, Any]:
    candidate_ids = sorted(metrics)
    world_ids = sorted(next(iter(metrics.values()))[
        "world_level_relation_minus_strongest_control"
    ])
    observed_by_candidate = {
        candidate: mean(list(metrics[candidate][
            "world_level_relation_minus_strongest_control"
        ].values()))
        for candidate in candidate_ids
    }
    observed_max = max(observed_by_candidate.values())
    at_least = 0
    for permutation in range(permutations):
        permuted_means = []
        for candidate in candidate_ids:
            values = metrics[candidate][
                "world_level_relation_minus_strongest_control"
            ]
            signed = []
            for world_id in world_ids:
                digest = hashlib.sha256(
                    f"Phase574|pipeline|{permutation}|{world_id}".encode("utf-8")
                ).digest()
                signed.append(values[world_id] if digest[0] & 1 else -values[world_id])
            permuted_means.append(mean(signed))
        at_least += int(max(permuted_means) >= observed_max)
    return {
        "candidate_count": len(candidate_ids),
        "world_count": len(world_ids),
        "observed_mean_by_candidate": observed_by_candidate,
        "observed_maximum_mean": observed_max,
        "permutation_count": permutations,
        "count_at_least_observed": at_least,
        "smoothed_tail_fraction": (at_least + 1) / (permutations + 1),
    }


def capture_generation_vectors(
    loaded: Any,
    layers: list[Any],
    worlds: list[list[dict[str, Any]]],
    candidate: dict[str, Any],
) -> tuple[dict[int, dict[str, torch.Tensor]], list[dict[str, Any]]]:
    rows, positions, meta, encoded_cpu = prepare_flat_batch(
        loaded.tokenizer, worlds, "left"
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    component = candidate["component"]
    captures: dict[int, dict[str, torch.Tensor]] = {}

    def hook_for(layer_index: int):
        def hook(
            module: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden is None or not isinstance(output, tuple) or output[1] is None:
                raise RuntimeError("Phase574 generation capture requires attention weights")
            primary, weights = output[0], output[1]
            batch, sequence, _ = hidden.shape
            if component == "query_relation_value_message":
                values = module.v_proj(hidden).view(
                    batch, sequence, -1, module.head_dim
                ).transpose(1, 2)
                values = values.repeat_interleave(
                    module.num_key_value_groups, dim=1
                )
                vectors = torch.stack([
                    source_message(
                        module, weights, values, batch_index,
                        int(positions[batch_index]["query_terminal"]),
                        positions[batch_index]["query_relation"],
                    )
                    for batch_index in range(batch)
                ])
            else:
                vectors = torch.stack([
                    primary[batch_index, int(positions[batch_index]["query_terminal"])]
                    for batch_index in range(batch)
                ])
            captures[layer_index] = {component: vectors.detach()}
        return hook

    handles = [
        layers[layer_index].self_attn.register_forward_hook(
            hook_for(layer_index), with_kwargs=True
        )
        for layer_index in candidate["patch_layers"]
    ]
    with torch.inference_mode():
        result = loaded.model(
            **encoded, use_cache=False, output_attentions=True, return_dict=True
        )
    for handle in handles:
        handle.remove()
    if set(captures) != set(candidate["patch_layers"]):
        raise RuntimeError("Phase574 generation vector capture drift")
    del result, encoded, encoded_cpu, rows, positions
    return captures, meta


def generate_intervention_batch(
    loaded: Any,
    layers: list[Any],
    worlds: list[list[dict[str, Any]]],
    candidate: dict[str, Any],
    condition: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    captures, capture_meta = capture_generation_vectors(
        loaded, layers, worlds, candidate
    )
    base_worlds = [[world[0]] for world in worlds]
    loaded.tokenizer.padding_side = "left"
    base_rows = [world[0] for world in worlds]
    prompts = [
        render_chat(loaded.tokenizer, MODEL, row["raw_prompt"]) for row in base_rows
    ]
    individual = [
        role_positions(loaded.tokenizer, prompt, row)
        for prompt, row in zip(prompts, base_rows)
    ]
    encoded_cpu = loaded.tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=False
    )
    width = int(encoded_cpu["input_ids"].shape[1])
    patch_positions = []
    for batch_index, (ids, roles) in enumerate(individual):
        active = encoded_cpu["input_ids"][batch_index][
            encoded_cpu["attention_mask"][batch_index].bool()
        ].tolist()
        if [int(value) for value in active] != ids:
            raise RuntimeError("Phase574 generation tokenization drift")
        patch_positions.append(width - len(ids) + roles["query_terminal"][-1])
    encoded = {key: value.to(loaded.input_device) for key, value in encoded_cpu.items()}
    batch_indices = torch.arange(len(worlds), device=loaded.input_device)
    position_tensor = torch.tensor(
        patch_positions, dtype=torch.long, device=loaded.input_device
    )
    deltas = (
        candidate_deltas(captures, capture_meta, candidate, condition)
        if condition != "natural_baseline" else {}
    )
    handles: list[Any] = []

    def make_hook(layer_index: int, sign: float):
        delta = deltas[layer_index].to(
            device=loaded.input_device,
            dtype=next(loaded.model.parameters()).dtype,
        ) * sign

        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
            primary = output[0].clone() if isinstance(output, tuple) else output.clone()
            if primary.shape[1] > 1:
                primary[batch_indices, position_tensor, :] += delta
            return replace_primary(output, primary)
        return hook

    if condition != "natural_baseline":
        for layer_index in candidate["patch_layers"]:
            handles.append(
                layers[layer_index].self_attn.register_forward_hook(
                    make_hook(layer_index, 1.0)
                )
            )
            if condition == "recipient_remove_restore":
                handles.append(
                    layers[layer_index].self_attn.register_forward_hook(
                        make_hook(layer_index, -1.0)
                    )
                )
    with torch.inference_mode():
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
    for handle in handles:
        handle.remove()
    output = []
    for index, row in enumerate(base_rows):
        text = loaded.tokenizer.decode(
            generated[index, width:], skip_special_tokens=True
        )
        classified = classify({
            **row,
            "candidate_token_ids": row["candidate_token_ids_by_model"][MODEL],
        }, text)
        output.append({
            "schema_version": "phase574_query_source_generation_row.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "split": row["split"],
            "base_case_id": row["base_case_id"],
            "candidate_id": candidate_id(candidate),
            "condition": condition,
            "recipient_target": row["target"],
            "relation_donor_target": worlds[index][1]["target"],
            "object_donor_target": worlds[index][2]["target"],
            **classified,
            "query_condition_intervention": condition != "natural_baseline",
            "full_short_generation": True,
            "sealed": False,
        })
    del encoded, encoded_cpu, generated, captures, deltas, base_worlds
    return output


def run_generation_stage(
    loaded: Any,
    layers: list[Any],
    selected_worlds_by_split: dict[str, list[list[dict[str, Any]]]],
    candidate: dict[str, Any],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    conditions = frozen["full_generation_gate"]["conditions"]
    batch_size = int(frozen["generation_world_batch_size"])
    output: list[dict[str, Any]] = []
    for split in protocol.CAUSAL_SPLITS:
        worlds = selected_worlds_by_split[split]
        for start in range(0, len(worlds), batch_size):
            batch = worlds[start:start + batch_size]
            for condition in conditions:
                output.extend(generate_intervention_batch(
                    loaded, layers, batch, candidate, condition, max_new_tokens
                ))
            print(
                f"[{time.strftime('%H:%M:%S')}] {MODEL} Phase574 {split} generation "
                f"{min(start + batch_size, len(worlds))}/{len(worlds)}",
                flush=True,
            )
    metrics = {}
    gates = frozen["full_generation_gate"]
    all_pass = True
    for split in protocol.CAUSAL_SPLITS:
        split_rows = [row for row in output if row["split"] == split]
        by_condition: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in split_rows:
            by_condition[row["condition"]][row["base_case_id"]] = row
        world_ids = sorted(by_condition["natural_baseline"])

        def target_rate(condition: str, target_field: str) -> float:
            return sum(
                by_condition[condition][world_id]["selected_candidate"]
                == by_condition[condition][world_id][target_field]
                for world_id in world_ids
            ) / max(1, len(world_ids))

        baseline_relation = target_rate(
            "natural_baseline", "relation_donor_target"
        )
        relation_rate = target_rate(
            "relation_donor_replace", "relation_donor_target"
        )
        object_relation_rate = target_rate(
            "object_donor_replace", "relation_donor_target"
        )
        order_relation_rate = target_rate(
            "order_donor_replace", "relation_donor_target"
        )
        relation_gain = relation_rate - baseline_relation
        object_gain = object_relation_rate - baseline_relation
        order_gain = order_relation_rate - baseline_relation
        restore_mismatch = sum(
            (
                by_condition["recipient_remove_restore"][world_id][
                    "normalized_generated"
                ]
                != by_condition["natural_baseline"][world_id][
                    "normalized_generated"
                ]
                or by_condition["recipient_remove_restore"][world_id][
                    "selected_candidate"
                ]
                != by_condition["natural_baseline"][world_id][
                    "selected_candidate"
                ]
            )
            for world_id in world_ids
        )
        split_pass = (
            relation_gain
            >= gates["relation_donor_target_win_rate_gain_minimum"]
            and relation_gain > object_gain
            and relation_gain > order_gain
            and restore_mismatch
            <= gates["restore_exact_semantic_mismatch_maximum"]
        )
        all_pass = all_pass and split_pass
        metrics[split] = {
            "world_count": len(world_ids),
            "natural_recipient_target_win_rate": target_rate(
                "natural_baseline", "recipient_target"
            ),
            "natural_relation_target_win_rate": baseline_relation,
            "relation_replace_relation_target_win_rate": relation_rate,
            "object_replace_relation_target_win_rate": object_relation_rate,
            "order_replace_relation_target_win_rate": order_relation_rate,
            "relation_donor_target_win_rate_gain": relation_gain,
            "object_control_relation_target_win_rate_gain": object_gain,
            "order_control_relation_target_win_rate_gain": order_gain,
            "remove_recipient_target_win_rate": target_rate(
                "recipient_remove", "recipient_target"
            ),
            "restore_exact_semantic_mismatch_count": restore_mismatch,
            "generation_gate_pass": split_pass,
        }
    return output, metrics, all_pass


def prepare_contract(restart: bool) -> None:
    paths = (
        BEHAVIOR_ROWS_PATH, CAUSAL_ROWS_PATH, GENERATION_ROWS_PATH,
        REGISTRY_PATH, SUMMARY_PATH, DECISION_PATH, CONTRACT_PATH,
    )
    if restart:
        for path in paths:
            path.unlink(missing_ok=True)
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    payload = {
        "schema_version": "phase574_query_source_causal_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "causal_protocol_sha256": sha256_file(causal_protocol.CAUSAL_PROTOCOL),
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "trace_decision_sha256": sha256_file(trace.DECISION_PATH),
        "candidate_ids": [
            candidate_id(candidate) for candidate in frozen["candidate_rows"]
        ],
        "conditions": frozen["conditions"],
        "causal_splits": frozen["causal_splits"],
        "recipient_variant": frozen["recipient_variant"],
        "sealed_split_read": False,
    }
    if CONTRACT_PATH.exists():
        existing = read_json(CONTRACT_PATH)
        for key, value in payload.items():
            if key != "created_at" and existing[key] != value:
                raise RuntimeError(f"Phase574 causal contract drift: {key}")
    else:
        write_json(CONTRACT_PATH, payload)


def run(max_new_tokens: int, restart: bool) -> Path:
    if not read_json(trace.DECISION_PATH)["coarse_query_source_causal_authorized"]:
        raise RuntimeError("Phase574 causal execution is not authorized")
    prepare_contract(restart)
    if SUMMARY_PATH.exists() and DECISION_PATH.exists() and not restart:
        return SUMMARY_PATH
    frozen = read_json(causal_protocol.CAUSAL_PROTOCOL)
    candidates = frozen["candidate_rows"]
    case_bank = load_case_bank()
    loaded = None
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        layers = get_layers(loaded.model)
        dtype = str(next(loaded.model.parameters()).dtype)
        if dtype != "torch.bfloat16" or len(layers) != 36:
            raise RuntimeError(f"Phase574 causal model drift: {dtype}/{len(layers)}")
        loaded.model.config._attn_implementation = "eager"

        loaded.tokenizer.padding_side = "left"
        behavior_rows, selected, behavior_diagnostics = run_behavior_stage(
            loaded, case_bank, max_new_tokens
        )
        write_jsonl(BEHAVIOR_ROWS_PATH, behavior_rows)
        worlds_by_split = {
            split: selected_worlds(case_bank, selected, split)
            for split in protocol.CAUSAL_SPLITS
        }
        registry = {
            "schema_version": "phase574_qwen3_causal_registry.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "selected_base_case_ids_by_split": selected,
            "selected_world_count_by_split": {
                split: len(worlds) for split, worlds in worlds_by_split.items()
            },
            "behavior_diagnostics_by_split": behavior_diagnostics,
            "selection_uses_internal_state": False,
            "sealed_split_read": False,
        }
        write_json(REGISTRY_PATH, registry)

        discovery_split = "causal_discovery"
        discovery_rows, reconstruction_max = run_causal_split(
            loaded, layers, discovery_split, worlds_by_split[discovery_split],
            candidates,
        )
        discovery_metrics = {
            candidate_id(candidate): candidate_metrics(
                discovery_rows, discovery_split, candidate_id(candidate)
            )
            for candidate in candidates
        }
        permutation_count = int(
            frozen["selection_rule"]["pipeline_permutation_count"]
        )
        familywise = familywise_pipeline_audit(
            discovery_metrics, permutation_count
        )
        eligible = [
            candidate for candidate in candidates
            if discovery_metrics[candidate_id(candidate)]["eligible"]
        ]
        selected_candidate = None
        if eligible:
            selected_candidate = sorted(
                eligible,
                key=lambda candidate: (
                    -discovery_metrics[candidate_id(candidate)][
                        "relation_route_switch_effect_mean"
                    ],
                    candidate_id(candidate),
                ),
            )[0]
        familywise_pass = (
            familywise["smoothed_tail_fraction"]
            <= frozen["selection_rule"]["maximum_smoothed_tail_fraction"]
        )
        discovery_gate = selected_candidate is not None and familywise_pass

        confirmation_rows: list[dict[str, Any]] = []
        confirmation_metrics: dict[str, Any] | None = None
        confirmation_audit: dict[str, Any] | None = None
        confirmation_gate = False
        if discovery_gate and selected_candidate is not None:
            confirmation_split = "causal_confirmation"
            confirmation_rows, confirmation_reconstruction = run_causal_split(
                loaded, layers, confirmation_split,
                worlds_by_split[confirmation_split], [selected_candidate],
            )
            reconstruction_max = max(
                reconstruction_max, confirmation_reconstruction
            )
            confirmation_metrics = candidate_metrics(
                confirmation_rows, confirmation_split,
                candidate_id(selected_candidate),
            )
            confirmation_audit = sign_flip_audit(
                confirmation_metrics[
                    "world_level_relation_minus_strongest_control"
                ],
                permutation_count,
                "confirmation",
            )
            confirmation_gate = (
                confirmation_metrics["eligible"]
                and confirmation_audit["smoothed_tail_fraction"]
                <= frozen["selection_rule"]["maximum_smoothed_tail_fraction"]
            )

        causal_rows = discovery_rows + confirmation_rows
        write_jsonl(CAUSAL_ROWS_PATH, causal_rows)

        generation_rows: list[dict[str, Any]] = []
        generation_metrics: dict[str, Any] = {}
        generation_gate = False
        if confirmation_gate and selected_candidate is not None:
            generation_rows, generation_metrics, generation_gate = (
                run_generation_stage(
                    loaded, layers, worlds_by_split, selected_candidate,
                    max_new_tokens,
                )
            )
        write_jsonl(GENERATION_ROWS_PATH, generation_rows)

        open_causal_gate = bool(
            discovery_gate and confirmation_gate and generation_gate
        )
        summary = {
            "schema_version": "phase574_query_source_causal_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": dtype,
            "behavior_diagnostics_by_split": behavior_diagnostics,
            "behavior_row_count": len(behavior_rows),
            "selected_world_count_by_split": {
                split: len(worlds) for split, worlds in worlds_by_split.items()
            },
            "discovery_candidate_metrics": discovery_metrics,
            "discovery_familywise_pipeline_audit": familywise,
            "eligible_discovery_candidate_ids": [
                candidate_id(candidate) for candidate in eligible
            ],
            "selected_candidate": selected_candidate,
            "discovery_gate_pass": discovery_gate,
            "confirmation_metrics": confirmation_metrics,
            "confirmation_sign_flip_audit": confirmation_audit,
            "confirmation_gate_pass": confirmation_gate,
            "full_generation_metrics_by_split": generation_metrics,
            "full_generation_gate_pass": generation_gate,
            "open_query_source_causal_gate_pass": open_causal_gate,
            "maximum_attention_reconstruction_relative_error": reconstruction_max,
            "causal_row_count": len(causal_rows),
            "generation_row_count": len(generation_rows),
            "behavior_rows_sha256": sha256_file(BEHAVIOR_ROWS_PATH),
            "causal_rows_sha256": sha256_file(CAUSAL_ROWS_PATH),
            "generation_rows_sha256": sha256_file(GENERATION_ROWS_PATH),
            "runtime_seconds": time.monotonic() - started,
            "output_embedding_direction_used": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "causal_splits_read": True,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        decision = {
            "schema_version": "phase574_query_source_causal_decision.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "model": MODEL,
            "selected_candidate_id": (
                candidate_id(selected_candidate)
                if selected_candidate is not None else None
            ),
            "discovery_gate_pass": discovery_gate,
            "confirmation_gate_pass": confirmation_gate,
            "full_generation_gate_pass": generation_gate,
            "open_query_source_causal_gate_pass": open_causal_gate,
            "new_sealed_validation_authorized": open_causal_gate,
            "reason": (
                "One frozen coarse query-condition candidate passed discovery, "
                "confirmation, controls, familywise permutation, and full generation."
                if open_causal_gate else
                "The query-condition to late source-selection chain failed at least one open gate."
            ),
            "head_channel_parameter_neuron_scan_authorized": False,
            "causal_splits_read": True,
            "sealed_split_read": False,
        }
        write_json(DECISION_PATH, decision)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
