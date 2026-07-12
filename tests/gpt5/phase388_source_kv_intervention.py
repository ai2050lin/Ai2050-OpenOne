#!/usr/bin/env python3
"""Run Phase388 source K/V transport interventions on one model and split."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase358_multiresolution_component_conservation import module_attr  # noqa: E402
from phase371c_behavior_qualification import answer_head, contains_alias  # noqa: E402


P388 = ROOT / "tests/gpt5/result/phase388_source_kv_transport"
CASE_FILE = P388 / "protocol/private/phase388_frozen_intervention_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("instrument_audit", "causal_test")
CONDITIONS = (
    "no_intervention",
    "identity_source_kv",
    "donor_source_k_only",
    "donor_source_v_only",
    "donor_source_kv",
    "donor_wrong_source_kv",
    "donor_source_kv_at_terminal_control_depth",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sequence_axis(tensor: torch.Tensor) -> int:
    if tensor.ndim != 3:
        raise RuntimeError(f"Expected rank-3 projection tensor, got {tuple(tensor.shape)}")
    if tensor.shape[0] == 1:
        return 1
    if tensor.shape[1] == 1:
        return 0
    raise RuntimeError(f"Cannot identify batch axis in {tuple(tensor.shape)}")


def sequence_length(tensor: torch.Tensor) -> int:
    return int(tensor.shape[sequence_axis(tensor)])


def vector_at(tensor: torch.Tensor, position: int) -> torch.Tensor:
    axis = sequence_axis(tensor)
    value = tensor[0, position] if axis == 1 else tensor[position, 0]
    return value.detach().float().cpu()


def replace_at(
    tensor: torch.Tensor, position: int, vector: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    axis = sequence_axis(tensor)
    if position >= sequence_length(tensor):
        return tensor, 0.0, 0.0
    patched = tensor.clone()
    source = vector.to(device=tensor.device, dtype=tensor.dtype)
    if axis == 1:
        before = patched[0, position].clone()
        patched[0, position] = source
        outside = torch.cat((patched[0, :position] - tensor[0, :position], patched[0, position + 1 :] - tensor[0, position + 1 :]), dim=0)
        after = patched[0, position]
    else:
        before = patched[position, 0].clone()
        patched[position, 0] = source
        outside = torch.cat((patched[:position, 0] - tensor[:position, 0], patched[position + 1 :, 0] - tensor[position + 1 :, 0]), dim=0)
        after = patched[position, 0]
    patch_error = float((after.float() - source.float()).abs().max().item())
    outside_error = float(outside.float().abs().max().item()) if outside.numel() else 0.0
    if torch.equal(before, after) and not torch.equal(before, source):
        raise RuntimeError("Projection patch did not change the selected vector")
    return patched, patch_error, outside_error


def projection_metrics(
    recipient: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor
) -> dict[str, float]:
    direction = donor.float() - recipient.float()
    shift = patched.float() - recipient.float()
    denom = float(torch.dot(direction, direction).item())
    projection = float(torch.dot(shift, direction).item()) / max(denom, 1e-12)
    direction_norm = float(torch.linalg.vector_norm(direction).item())
    shift_norm = float(torch.linalg.vector_norm(shift).item())
    cosine = float(torch.dot(shift, direction).item()) / max(
        shift_norm * direction_norm, 1e-12
    )
    residual = shift - projection * direction
    off_axis = float(torch.linalg.vector_norm(residual).item()) / max(direction_norm, 1e-12)
    return {
        "query_projection_toward_donor": projection,
        "query_shift_cosine_to_donor_direction": cosine,
        "query_off_axis_ratio": off_axis,
        "query_shift_norm": shift_norm,
        "natural_query_contrast_norm": direction_norm,
    }


def patch_output(
    output: Any,
    position: int,
    vector: torch.Tensor | None,
    audit: dict[str, Any],
    name: str,
) -> Any:
    if vector is None:
        return output
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(tensor) or sequence_length(tensor) <= position:
        return output
    patched, patch_error, outside_error = replace_at(tensor, position, vector)
    audit[f"{name}_patch_call_count"] += 1
    audit[f"{name}_max_patch_error"] = max(
        audit[f"{name}_max_patch_error"], patch_error
    )
    audit[f"{name}_max_outside_error"] = max(
        audit[f"{name}_max_outside_error"], outside_error
    )
    if isinstance(output, tuple):
        return (patched, *output[1:])
    if isinstance(output, list):
        return [patched, *output[1:]]
    return patched


@torch.inference_mode()
def run_path(
    loaded: Any,
    case: dict[str, Any],
    *,
    capture_layers: tuple[int, ...],
    patch_layer: int | None = None,
    patch_k: torch.Tensor | None = None,
    patch_v: torch.Tensor | None = None,
) -> dict[str, Any]:
    layers = get_layers(loaded.model)
    candidate_layer = int(case["candidate_layer"])
    capture: dict[str, Any] = {}
    handles = []
    audit = {
        "key_patch_call_count": 0,
        "value_patch_call_count": 0,
        "key_max_patch_error": 0.0,
        "value_max_patch_error": 0.0,
        "key_max_outside_error": 0.0,
        "value_max_outside_error": 0.0,
    }
    source_position = int(case["source_position_private"])
    wrong_position = int(case["wrong_source_position_private"])
    query_position = int(case["query_position_private"])
    try:
        for layer_index in capture_layers:
            attention = layers[layer_index].self_attn
            key_module = module_attr(attention, ("k_proj", "key"))
            value_module = module_attr(attention, ("v_proj", "value"))

            def capture_key(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                idx: int = layer_index,
            ) -> Any:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if sequence_length(tensor) > source_position:
                    capture[f"L{idx}_key_source"] = vector_at(tensor, source_position)
                    capture[f"L{idx}_key_wrong"] = vector_at(tensor, wrong_position)
                if patch_layer == idx:
                    return patch_output(
                        output, source_position, patch_k, audit, "key"
                    )
                return output

            def capture_value(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                idx: int = layer_index,
            ) -> Any:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                if sequence_length(tensor) > source_position:
                    capture[f"L{idx}_value_source"] = vector_at(tensor, source_position)
                    capture[f"L{idx}_value_wrong"] = vector_at(tensor, wrong_position)
                if patch_layer == idx:
                    return patch_output(
                        output, source_position, patch_v, audit, "value"
                    )
                return output

            handles.extend(
                [
                    key_module.register_forward_hook(capture_key),
                    value_module.register_forward_hook(capture_value),
                ]
            )

        candidate_o = module_attr(
            layers[candidate_layer].self_attn, ("o_proj", "dense")
        )

        def capture_query(_module: Any, inputs: tuple[Any, ...]) -> None:
            tensor = inputs[0]
            if sequence_length(tensor) > query_position:
                capture["query_attention_head_state"] = vector_at(
                    tensor, query_position
                )

        handles.append(candidate_o.register_forward_pre_hook(capture_query))

        prompt_ids = [int(item) for item in case["prompt_token_ids_private"]]
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=loaded.input_device
        )
        attention_mask = torch.ones_like(input_ids)
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        logits = output.logits[0, -1].detach().float().cpu()
        past = output.past_key_values
        prefix_replay: list[bool] = []
        prefix_ids = [
            int(item) for item in case["target_decision_prefix_token_ids_private"]
        ]
        total_length = len(prompt_ids)
        for token_id in prefix_ids:
            prefix_replay.append(int(torch.argmax(logits).item()) == token_id)
            total_length += 1
            token = torch.tensor(
                [[token_id]], dtype=torch.long, device=loaded.input_device
            )
            mask = torch.ones(
                (1, total_length), dtype=torch.long, device=loaded.input_device
            )
            output = loaded.model(
                input_ids=token,
                attention_mask=mask,
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            logits = output.logits[0, -1].detach().float().cpu()
            past = output.past_key_values
        expected = int(case["target_first_token_id_private"])
        return {
            **capture,
            "decision_logits": logits,
            "prefix_replay_matches": prefix_replay,
            "all_prefix_transitions_match": all(prefix_replay),
            "target_decision_argmax_match": int(torch.argmax(logits).item())
            == expected,
            "target_decision_argmax_token_id": int(torch.argmax(logits).item()),
            "target_first_token_id": expected,
            "model_call_count": 1 + len(prefix_ids),
            "audit": audit,
        }
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def generate_with_patch(
    loaded: Any,
    case: dict[str, Any],
    layer_index: int,
    patch_k: torch.Tensor,
    patch_v: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    layer = get_layers(loaded.model)[layer_index]
    key_module = module_attr(layer.self_attn, ("k_proj", "key"))
    value_module = module_attr(layer.self_attn, ("v_proj", "value"))
    source_position = int(case["source_position_private"])
    audit = {
        "key_patch_call_count": 0,
        "value_patch_call_count": 0,
        "key_max_patch_error": 0.0,
        "value_max_patch_error": 0.0,
        "key_max_outside_error": 0.0,
        "value_max_outside_error": 0.0,
    }

    def key_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return patch_output(output, source_position, patch_k, audit, "key")

    def value_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        return patch_output(output, source_position, patch_v, audit, "value")

    handles = [
        key_module.register_forward_hook(key_hook),
        value_module.register_forward_hook(value_hook),
    ]
    try:
        prompt_ids = [int(item) for item in case["prompt_token_ids_private"]]
        input_ids = torch.tensor(
            [prompt_ids], dtype=torch.long, device=loaded.input_device
        )
        generated = loaded.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
        suffix = [int(item) for item in generated[0, len(prompt_ids) :].tolist()]
        text = loaded.tokenizer.decode(suffix, skip_special_tokens=True)
        head = answer_head(text)
        return {
            "generated_token_ids": suffix,
            "generated_text": text,
            "answer_head_text": head,
            "donor_target_present": contains_alias(head, case["donor_target_aliases_private"]),
            "recipient_target_present": contains_alias(head, case["target_aliases"]),
            "donor_target_strict_switch": (
                contains_alias(head, case["donor_target_aliases_private"])
                and not contains_alias(head, case["target_aliases"])
            ),
            "audit": audit,
        }
    finally:
        for handle in handles:
            handle.remove()


def scalar_metrics(
    logits: torch.Tensor,
    donor_token: int,
    recipient_token: int,
    recipient_margin: float,
    donor_margin: float,
) -> dict[str, float]:
    margin = float(logits[donor_token].item() - logits[recipient_token].item())
    shift = margin - recipient_margin
    return {
        "donor_vs_recipient_logit_margin": margin,
        "donor_direction_margin_shift": shift,
        "normalized_margin_mediation": shift
        / max(abs(donor_margin - recipient_margin), 1e-8),
    }


def scenario_spec(
    condition: str,
    recipient_case: dict[str, Any],
    recipient: dict[str, Any],
    donor: dict[str, Any],
) -> tuple[int | None, torch.Tensor | None, torch.Tensor | None]:
    candidate_layer = int(recipient_case["candidate_layer"])
    control_layer = int(recipient_case["terminal_control_layer"])
    if condition == "no_intervention":
        return None, None, None
    if condition == "identity_source_kv":
        return (
            candidate_layer,
            recipient[f"L{candidate_layer}_key_source"],
            recipient[f"L{candidate_layer}_value_source"],
        )
    if condition == "donor_source_k_only":
        return candidate_layer, donor[f"L{candidate_layer}_key_source"], None
    if condition == "donor_source_v_only":
        return candidate_layer, None, donor[f"L{candidate_layer}_value_source"]
    if condition == "donor_source_kv":
        return (
            candidate_layer,
            donor[f"L{candidate_layer}_key_source"],
            donor[f"L{candidate_layer}_value_source"],
        )
    if condition == "donor_wrong_source_kv":
        return (
            candidate_layer,
            donor[f"L{candidate_layer}_key_wrong"],
            donor[f"L{candidate_layer}_value_wrong"],
        )
    if condition == "donor_source_kv_at_terminal_control_depth":
        return (
            control_layer,
            donor[f"L{control_layer}_key_source"],
            donor[f"L{control_layer}_value_source"],
        )
    raise KeyError(condition)


@torch.inference_mode()
def run(model: str, split: str, max_new_tokens: int) -> dict[str, Any]:
    if split == "causal_test":
        gate = P388 / "phase388_instrument_audit_summary.json"
        if not gate.is_file() or not read_json(gate)["authorization"]["causal_test"]:
            raise RuntimeError("Phase388 causal test is not authorized by instrument audit")
    cases = [
        row
        for row in read_jsonl(CASE_FILE)
        if row["private_execution_model"] == model and row["phase388_split"] == split
    ]
    expected = 4 if split == "instrument_audit" else 32
    if len(cases) != expected:
        raise RuntimeError(f"Expected {expected} {model}/{split} cases, found {len(cases)}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["parallel_group_id"]][case["condition"]] = case

    loaded = None
    direction_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, pair) in enumerate(sorted(grouped.items()), 1):
            natural: dict[str, dict[str, Any]] = {}
            for condition, case in pair.items():
                natural[condition] = run_path(
                    loaded,
                    case,
                    capture_layers=(
                        int(case["candidate_layer"]),
                        int(case["terminal_control_layer"]),
                    ),
                )
                if not natural[condition]["all_prefix_transitions_match"]:
                    raise RuntimeError(
                        f"Natural target prefix replay failed for {case['blind_case_id']}"
                    )
                if not natural[condition]["target_decision_argmax_match"]:
                    raise RuntimeError(
                        f"Natural target decision failed for {case['blind_case_id']}"
                    )

            for recipient_name, donor_name in (
                ("mapping_a", "mapping_b"),
                ("mapping_b", "mapping_a"),
            ):
                recipient_case = dict(pair[recipient_name])
                donor_case = pair[donor_name]
                recipient_case["donor_target_aliases_private"] = donor_case[
                    "target_aliases"
                ]
                recipient_natural = natural[recipient_name]
                donor_natural = natural[donor_name]
                donor_token = int(donor_case["target_first_token_id_private"])
                recipient_token = int(recipient_case["target_first_token_id_private"])
                if donor_token == recipient_token:
                    raise RuntimeError("Donor and recipient target first tokens are identical")
                recipient_margin = float(
                    recipient_natural["decision_logits"][donor_token].item()
                    - recipient_natural["decision_logits"][recipient_token].item()
                )
                donor_margin = float(
                    donor_natural["decision_logits"][donor_token].item()
                    - donor_natural["decision_logits"][recipient_token].item()
                )
                scenario_rows: list[dict[str, Any]] = []
                for intervention in CONDITIONS:
                    if intervention == "no_intervention":
                        outcome = recipient_natural
                    else:
                        layer_index, patch_k, patch_v = scenario_spec(
                            intervention,
                            recipient_case,
                            recipient_natural,
                            donor_natural,
                        )
                        outcome = run_path(
                            loaded,
                            recipient_case,
                            capture_layers=(
                                int(recipient_case["candidate_layer"]),
                                int(recipient_case["terminal_control_layer"]),
                            ),
                            patch_layer=layer_index,
                            patch_k=patch_k,
                            patch_v=patch_v,
                        )
                    query = projection_metrics(
                        recipient_natural["query_attention_head_state"],
                        donor_natural["query_attention_head_state"],
                        outcome["query_attention_head_state"],
                    )
                    scalar = scalar_metrics(
                        outcome["decision_logits"],
                        donor_token,
                        recipient_token,
                        recipient_margin,
                        donor_margin,
                    )
                    scenario_rows.append(
                        {
                            "intervention": intervention,
                            **query,
                            **scalar,
                            "target_decision_argmax_token_id": outcome[
                                "target_decision_argmax_token_id"
                            ],
                            "target_decision_argmax_is_donor": outcome[
                                "target_decision_argmax_token_id"
                            ]
                            == donor_token,
                            "target_decision_argmax_is_recipient": outcome[
                                "target_decision_argmax_token_id"
                            ]
                            == recipient_token,
                            "model_call_count": outcome["model_call_count"],
                            "patch_audit": outcome["audit"],
                        }
                    )
                candidate_layer = int(recipient_case["candidate_layer"])
                main_generation = generate_with_patch(
                    loaded,
                    recipient_case,
                    candidate_layer,
                    donor_natural[f"L{candidate_layer}_key_source"],
                    donor_natural[f"L{candidate_layer}_value_source"],
                    max_new_tokens,
                )
                identity = next(
                    row for row in scenario_rows if row["intervention"] == "identity_source_kv"
                )
                direction_rows.append(
                    {
                        "schema_version": "62.3.0",
                        "phase_id": "Phase388-SourceKVIntervention",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "split": split,
                        "parallel_group_id": group_id,
                        "direction_id": f"{group_id}:{recipient_name}<-{donor_name}",
                        "recipient_condition": recipient_name,
                        "donor_condition": donor_name,
                        "recipient_target": recipient_case["target"],
                        "donor_target": donor_case["target"],
                        "candidate_layer": candidate_layer,
                        "terminal_control_layer": int(
                            recipient_case["terminal_control_layer"]
                        ),
                        "source_position": int(recipient_case["source_position_private"]),
                        "query_position": int(recipient_case["query_position_private"]),
                        "wrong_source_position": int(
                            recipient_case["wrong_source_position_private"]
                        ),
                        "recipient_natural_margin": recipient_margin,
                        "donor_natural_margin": donor_margin,
                        "natural_margin_separation": donor_margin - recipient_margin,
                        "natural_query_contrast_norm": float(
                            torch.linalg.vector_norm(
                                donor_natural["query_attention_head_state"]
                                - recipient_natural["query_attention_head_state"]
                            ).item()
                        ),
                        "identity_query_max_abs_error": float(
                            abs(identity["query_shift_norm"])
                        ),
                        "identity_margin_shift_abs_error": float(
                            abs(identity["donor_direction_margin_shift"])
                        ),
                        "scenario_rows": scenario_rows,
                        "main_generation": main_generation,
                        "causal_claim": False,
                        "single_neuron_claim": False,
                    }
                )
            print(
                f"[{model}] Phase388 {split} groups {group_index}/{len(grouped)} "
                f"directions={len(direction_rows)}",
                flush=True,
            )

        summary = {
            "schema_version": "62.3.0",
            "phase_id": "Phase388-SourceKVIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "split": split,
            "group_count": len(grouped),
            "direction_count": len(direction_rows),
            "scenario_count": len(direction_rows) * len(CONDITIONS),
            "main_generation_count": len(direction_rows),
            "valid": len(direction_rows) == len(grouped) * 2,
        }
        root = P388 / "collection" / split / model
        write_jsonl(root / "direction_rows.jsonl", direction_rows)
        write_json(root / "complete.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    run(args.model, args.split, args.max_new_tokens)
