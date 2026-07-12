#!/usr/bin/env python3
"""Run graph-consistent Phase392 layer-input parent-boundary interventions."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import answer_head, contains_alias  # noqa: E402
from phase371c_blind_vector_contrast import static_roles  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase392_parent_boundary_replay"
CASES = OUT / "protocol/private/phase392_frozen_intervention_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SPLITS = ("instrument_audit", "causal_test")
SEMANTIC_ROLES = ("entities", "attributes_items", "relations", "query_keywords", "query_window")
SCENARIOS = (
    "no_intervention",
    "identity_semantic_joint",
    "donor_semantic_joint",
    "donor_attributes_only",
    "donor_fixed_best_role",
    "donor_frozen_structure_roles",
    "donor_same_count_random_parent_positions",
    "donor_semantic_joint_wrong_depth",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def component_tensor(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, (tuple, list)) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor layer output, got {type(value).__name__}")
    return value


def positions_for(case: dict[str, Any], roles: tuple[str, ...]) -> list[int]:
    return sorted(
        {
            int(position)
            for role in roles
            for position in case["role_positions_private"][role]
        }
    )


def projection_metrics(recipient: torch.Tensor, donor: torch.Tensor, patched: torch.Tensor) -> dict[str, float]:
    direction = donor.float() - recipient.float()
    shift = patched.float() - recipient.float()
    denominator = float(torch.dot(direction, direction).item())
    projection = float(torch.dot(shift, direction).item()) / max(denominator, 1e-12)
    return {
        "query_projection_toward_donor": projection,
        "query_shift_norm": float(torch.linalg.vector_norm(shift).item()),
        "natural_query_contrast_norm": float(torch.linalg.vector_norm(direction).item()),
    }


@torch.inference_mode()
def run_path(
    loaded: Any,
    case: dict[str, Any],
    capture_layers: tuple[int, ...],
    *,
    patch_layer: int | None = None,
    recipient_positions: list[int] | None = None,
    patch_vectors: torch.Tensor | None = None,
) -> dict[str, Any]:
    layers = get_layers(loaded.model)
    captures: dict[int, torch.Tensor] = {}
    query_outputs: dict[int, torch.Tensor] = {}
    handles = []
    audit = {
        "patch_call_count": 0,
        "max_patch_error": 0.0,
        "max_outside_error": 0.0,
    }
    query_position = static_roles(loaded.tokenizer, case)[0][1]
    prompt_length = len(case["prompt_token_ids_private"])
    try:
        for layer_index in capture_layers:
            layer = layers[layer_index]

            def layer_pre(
                _module: Any,
                inputs: tuple[Any, ...],
                idx: int = layer_index,
            ) -> tuple[Any, ...] | None:
                hidden = inputs[0]
                if hidden.shape[1] == prompt_length:
                    captures[idx] = hidden.detach().float().cpu()
                if (
                    patch_layer != idx
                    or recipient_positions is None
                    or patch_vectors is None
                    or hidden.shape[1] != prompt_length
                ):
                    return None
                positions = torch.tensor(
                    recipient_positions, dtype=torch.long, device=hidden.device
                )
                source = patch_vectors.to(device=hidden.device, dtype=hidden.dtype)
                if source.shape != (len(recipient_positions), hidden.shape[-1]):
                    raise RuntimeError("Phase392 patch vector shape mismatch")
                patched = hidden.clone()
                before = patched.index_select(1, positions).clone()
                patched[0, positions] = source
                after = patched.index_select(1, positions)
                outside_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
                outside_mask[positions] = False
                outside = (patched[:, outside_mask] - hidden[:, outside_mask]).float()
                audit["patch_call_count"] += 1
                audit["max_patch_error"] = max(
                    audit["max_patch_error"],
                    float((after[0].float() - source.float()).abs().max().item()),
                )
                audit["max_outside_error"] = max(
                    audit["max_outside_error"],
                    float(outside.abs().max().item()) if outside.numel() else 0.0,
                )
                if torch.equal(before, after) and not torch.equal(before[0], source):
                    raise RuntimeError("Phase392 patch did not change selected states")
                return (patched, *inputs[1:])

            def layer_post(
                _module: Any,
                _inputs: tuple[Any, ...],
                output: Any,
                idx: int = layer_index,
            ) -> None:
                tensor = component_tensor(output)
                if tensor.shape[1] == prompt_length:
                    query_outputs[idx] = tensor[0, query_position].detach().float().cpu()

            handles.extend(
                [
                    layer.register_forward_pre_hook(layer_pre),
                    layer.register_forward_hook(layer_post),
                ]
            )
        ids = torch.tensor(
            [case["prompt_token_ids_private"]],
            dtype=torch.long,
            device=loaded.input_device,
        )
        output = loaded.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
        logits = output.logits[0, -1].detach().float().cpu()
        past = output.past_key_values
        prefix_matches: list[bool] = []
        total_length = ids.shape[1]
        for token_id in case["target_decision_prefix_token_ids_private"]:
            prefix_matches.append(int(torch.argmax(logits).item()) == int(token_id))
            total_length += 1
            token = torch.tensor([[int(token_id)]], dtype=torch.long, device=loaded.input_device)
            output = loaded.model(
                input_ids=token,
                attention_mask=torch.ones((1, total_length), dtype=torch.long, device=loaded.input_device),
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            logits = output.logits[0, -1].detach().float().cpu()
            past = output.past_key_values
        return {
            "layer_inputs": captures,
            "query_outputs": query_outputs,
            "decision_logits": logits,
            "prefix_matches": prefix_matches,
            "all_prefix_transitions_match": all(prefix_matches),
            "target_decision_argmax_match": int(torch.argmax(logits).item())
            == int(case["target_first_token_id_private"]),
            "audit": audit,
        }
    finally:
        for handle in handles:
            handle.remove()


def patch_spec(
    scenario: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient: dict[str, Any],
    donor: dict[str, Any],
) -> tuple[int | None, list[int] | None, torch.Tensor | None]:
    layer = int(recipient_case["candidate_layer"])
    wrong = int(recipient_case["wrong_depth_layer"])
    if scenario == "no_intervention":
        return None, None, None
    if scenario == "identity_semantic_joint":
        roles = SEMANTIC_ROLES
        source = recipient
        source_case = recipient_case
        selected_layer = layer
    elif scenario == "donor_semantic_joint":
        roles = SEMANTIC_ROLES
        source, source_case, selected_layer = donor, donor_case, layer
    elif scenario == "donor_attributes_only":
        roles = ("attributes_items",)
        source, source_case, selected_layer = donor, donor_case, layer
    elif scenario == "donor_fixed_best_role":
        roles = (recipient_case["fixed_best_role"],)
        source, source_case, selected_layer = donor, donor_case, layer
    elif scenario == "donor_frozen_structure_roles":
        roles = tuple(recipient_case["frozen_structure_roles"])
        source, source_case, selected_layer = donor, donor_case, layer
    elif scenario == "donor_same_count_random_parent_positions":
        semantic_count = len(positions_for(recipient_case, SEMANTIC_ROLES))
        recipient_other = positions_for(recipient_case, ("other_causal_prefix",))[:semantic_count]
        donor_other = positions_for(donor_case, ("other_causal_prefix",))[:semantic_count]
        if len(recipient_other) != semantic_count or len(donor_other) != semantic_count:
            raise RuntimeError("Insufficient Phase392 random control positions")
        return layer, recipient_other, donor["layer_inputs"][layer][0, donor_other]
    elif scenario == "donor_semantic_joint_wrong_depth":
        roles = SEMANTIC_ROLES
        source, source_case, selected_layer = donor, donor_case, wrong
    else:
        raise KeyError(scenario)
    recipient_positions = positions_for(recipient_case, roles)
    donor_positions = positions_for(source_case, roles)
    if len(recipient_positions) != len(donor_positions):
        raise RuntimeError(f"Phase392 role count mismatch for {scenario}")
    return (
        selected_layer,
        recipient_positions,
        source["layer_inputs"][selected_layer][0, donor_positions],
    )


@torch.inference_mode()
def generate_joint(
    loaded: Any,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    donor: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    layer_index = int(recipient_case["candidate_layer"])
    layer = get_layers(loaded.model)[layer_index]
    recipient_positions = positions_for(recipient_case, SEMANTIC_ROLES)
    donor_positions = positions_for(donor_case, SEMANTIC_ROLES)
    vectors = donor["layer_inputs"][layer_index][0, donor_positions]
    audit = {"patch_call_count": 0, "max_patch_error": 0.0, "max_outside_error": 0.0}
    prompt_length = len(recipient_case["prompt_token_ids_private"])

    def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        hidden = inputs[0]
        if hidden.shape[1] != prompt_length:
            return None
        positions = torch.tensor(recipient_positions, dtype=torch.long, device=hidden.device)
        source = vectors.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        patched[0, positions] = source
        outside_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        outside_mask[positions] = False
        outside = (patched[:, outside_mask] - hidden[:, outside_mask]).float()
        audit["patch_call_count"] += 1
        audit["max_patch_error"] = max(
            audit["max_patch_error"],
            float((patched[0, positions].float() - source.float()).abs().max().item()),
        )
        audit["max_outside_error"] = max(
            audit["max_outside_error"],
            float(outside.abs().max().item()) if outside.numel() else 0.0,
        )
        return (patched, *inputs[1:])

    handle = layer.register_forward_pre_hook(hook)
    try:
        ids = torch.tensor(
            [recipient_case["prompt_token_ids_private"]],
            dtype=torch.long,
            device=loaded.input_device,
        )
        generated = loaded.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
        suffix = [int(value) for value in generated[0, ids.shape[1] :].tolist()]
        text = loaded.tokenizer.decode(suffix, skip_special_tokens=True)
        head = answer_head(text)
        donor_present = contains_alias(head, donor_case["target_aliases"])
        recipient_present = contains_alias(head, recipient_case["target_aliases"])
        return {
            "generated_token_ids": suffix,
            "generated_text": text,
            "answer_head_text": head,
            "donor_target_present": donor_present,
            "recipient_target_present": recipient_present,
            "strict_donor_target_switch": donor_present and not recipient_present,
            "audit": audit,
        }
    finally:
        handle.remove()


def run(model: str, split: str, max_new_tokens: int) -> dict[str, Any]:
    if split == "causal_test":
        gate = OUT / "phase392_instrument_audit_summary.json"
        if not gate.is_file() or not read_json(gate)["authorization"]["causal_test"]:
            raise RuntimeError("Phase392 causal test is not authorized")
    cases = [
        row
        for row in read_jsonl(CASES)
        if row["private_execution_model"] == model and row["phase392_split"] == split
    ]
    expected = 4 if split == "instrument_audit" else 48
    if len(cases) != expected:
        raise RuntimeError(f"Expected {expected} Phase392 cases for {model}/{split}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["parallel_group_id"]][case["condition"]] = case
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, conditions) in enumerate(sorted(grouped.items()), 1):
            capture_layers = tuple(
                sorted(
                    {
                        int(next(iter(conditions.values()))["candidate_layer"]),
                        int(next(iter(conditions.values()))["wrong_depth_layer"]),
                    }
                )
            )
            natural = {
                condition: run_path(loaded, case, capture_layers)
                for condition, case in conditions.items()
            }
            for condition, outcome in natural.items():
                if not outcome["all_prefix_transitions_match"] or not outcome["target_decision_argmax_match"]:
                    raise RuntimeError(f"Phase392 natural replay failed for {model}/{group_id}/{condition}")
            for recipient_name, donor_name in (("mapping_x", "mapping_y"), ("mapping_y", "mapping_x")):
                recipient_case = conditions[recipient_name]
                donor_case = conditions[donor_name]
                recipient = natural[recipient_name]
                donor = natural[donor_name]
                recipient_token = int(recipient_case["target_first_token_id_private"])
                donor_token = int(donor_case["target_first_token_id_private"])
                recipient_margin = float(recipient["decision_logits"][donor_token] - recipient["decision_logits"][recipient_token])
                donor_margin = float(donor["decision_logits"][donor_token] - donor["decision_logits"][recipient_token])
                scenario_rows = []
                for scenario in SCENARIOS:
                    if scenario == "no_intervention":
                        outcome = recipient
                    else:
                        patch_layer, positions, vectors = patch_spec(
                            scenario, recipient_case, donor_case, recipient, donor
                        )
                        outcome = run_path(
                            loaded,
                            recipient_case,
                            capture_layers,
                            patch_layer=patch_layer,
                            recipient_positions=positions,
                            patch_vectors=vectors,
                        )
                    margin = float(outcome["decision_logits"][donor_token] - outcome["decision_logits"][recipient_token])
                    query = projection_metrics(
                        recipient["query_outputs"][int(recipient_case["candidate_layer"])],
                        donor["query_outputs"][int(recipient_case["candidate_layer"])],
                        outcome["query_outputs"][int(recipient_case["candidate_layer"])],
                    )
                    scenario_rows.append(
                        {
                            "scenario": scenario,
                            **query,
                            "donor_vs_recipient_margin": margin,
                            "donor_margin_shift": margin - recipient_margin,
                            "normalized_margin_mediation": (margin - recipient_margin)
                            / max(abs(donor_margin - recipient_margin), 1e-8),
                            "argmax_token_id": int(torch.argmax(outcome["decision_logits"]).item()),
                            "argmax_is_donor": int(torch.argmax(outcome["decision_logits"]).item()) == donor_token,
                            "patch_audit": outcome["audit"],
                        }
                    )
                generation = generate_joint(
                    loaded, recipient_case, donor_case, donor, max_new_tokens
                )
                rows.append(
                    {
                        "schema_version": "66.3.0",
                        "phase_id": "Phase392-ParentBoundaryIntervention",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "split": split,
                        "parallel_group_id": group_id,
                        "direction_id": f"{group_id}:{recipient_name}<-{donor_name}",
                        "recipient_condition": recipient_name,
                        "donor_condition": donor_name,
                        "candidate_layer": int(recipient_case["candidate_layer"]),
                        "wrong_depth_layer": int(recipient_case["wrong_depth_layer"]),
                        "recipient_natural_margin": recipient_margin,
                        "donor_natural_margin": donor_margin,
                        "natural_margin_separation": donor_margin - recipient_margin,
                        "scenario_rows": scenario_rows,
                        "joint_generation": generation,
                        "causal_path_claim": False,
                        "single_neuron_claim": False,
                    }
                )
            print(f"[{model}/{split}] Phase392 groups {group_index}/{len(grouped)} directions={len(rows)}", flush=True)
        root = OUT / "collection" / split / model
        write_jsonl(root / "direction_rows.jsonl", rows)
        summary = {
            "schema_version": "66.3.0",
            "phase_id": "Phase392-ParentBoundaryIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "split": split,
            "group_count": len(grouped),
            "direction_count": len(rows),
            "scenario_count": len(rows) * len(SCENARIOS),
            "joint_generation_count": len(rows),
            "valid": len(rows) == len(grouped) * 2,
        }
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
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    run(args.model, args.split, args.max_new_tokens)
