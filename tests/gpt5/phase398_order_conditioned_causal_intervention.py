#!/usr/bin/env python3
"""Run finite query-end parent-boundary interventions for Phase398."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase371c_behavior_qualification import answer_head, contains_alias  # noqa: E402
from phase398_order_conditioned_causal_protocol import MODELS, SCENARIOS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase398_joint_binding"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


@torch.inference_mode()
def capture_parent_states(loaded: Any, case: dict[str, Any]) -> dict[str, torch.Tensor]:
    layers = get_layers(loaded.model)
    candidate = int(case["candidate_parent_layer_private"])
    wrong = int(case["wrong_depth_parent_layer_private"])
    prompt_length = len(case["prompt_token_ids_private"])
    query_position = int(case["query_end_position_private"])
    answer_position = int(case["answer_anchor_position_private"])
    captures: dict[str, torch.Tensor] = {}
    handles = []
    try:
        for label, layer_index in (("candidate", candidate), ("wrong", wrong)):
            def hook(_module: Any, inputs: tuple[Any, ...], name: str = label) -> None:
                hidden = inputs[0]
                if hidden.shape[1] == prompt_length:
                    captures[f"{name}_query"] = hidden[0, query_position].detach().float().cpu()
                    captures[f"{name}_answer"] = hidden[0, answer_position].detach().float().cpu()

            handles.append(layers[layer_index].register_forward_pre_hook(hook))
        ids = torch.tensor([case["prompt_token_ids_private"]], dtype=torch.long, device=loaded.input_device)
        loaded.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        if set(captures) != {"candidate_query", "candidate_answer", "wrong_query", "wrong_answer"}:
            raise RuntimeError("Phase398 parent-state capture is incomplete")
        return captures
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def generate_patch(
    loaded: Any,
    recipient: dict[str, Any],
    donor: dict[str, Any],
    *,
    patch_layer: int | None,
    recipient_position: int | None,
    vector: torch.Tensor | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    handles = []
    audit = {"patch_call_count": 0, "max_patch_error": 0.0, "max_outside_error": 0.0}
    prompt_length = len(recipient["prompt_token_ids_private"])
    if patch_layer is not None:
        if recipient_position is None or vector is None:
            raise RuntimeError("Phase398 incomplete patch specification")
        layer = get_layers(loaded.model)[patch_layer]

        def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            hidden = inputs[0]
            if hidden.shape[1] != prompt_length:
                return None
            source = vector.to(device=hidden.device, dtype=hidden.dtype)
            patched = hidden.clone()
            patched[0, recipient_position] = source
            outside_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
            outside_mask[recipient_position] = False
            outside = (patched[:, outside_mask] - hidden[:, outside_mask]).float()
            audit["patch_call_count"] += 1
            audit["max_patch_error"] = max(audit["max_patch_error"], float((patched[0, recipient_position].float() - source.float()).abs().max().item()))
            audit["max_outside_error"] = max(audit["max_outside_error"], float(outside.abs().max().item()) if outside.numel() else 0.0)
            return (patched, *inputs[1:])

        handles.append(layer.register_forward_pre_hook(hook))
    try:
        ids = torch.tensor([recipient["prompt_token_ids_private"]], dtype=torch.long, device=loaded.input_device)
        generated = loaded.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
        suffix = [int(value) for value in generated[0, ids.shape[1]:].tolist()]
        text = loaded.tokenizer.decode(suffix, skip_special_tokens=True)
        head = answer_head(text)
        recipient_present = contains_alias(head, recipient["target_aliases"])
        donor_present = contains_alias(head, donor["target_aliases"])
        return {
            "generated_token_ids": suffix,
            "generated_text": text,
            "answer_head_text": head,
            "recipient_target_present": recipient_present,
            "donor_target_present": donor_present,
            "strict_recipient_answer": recipient_present and not donor_present,
            "strict_donor_answer_switch": donor_present and not recipient_present,
            "patch_audit": audit,
        }
    finally:
        for handle in handles:
            handle.remove()


def scenario_spec(
    scenario: str,
    recipient: dict[str, Any],
    donor: dict[str, Any],
    wrong_order: dict[str, Any],
    states: dict[str, dict[str, torch.Tensor]],
) -> tuple[int | None, int | None, torch.Tensor | None, dict[str, Any]]:
    recipient_id = recipient["blind_case_id"]
    donor_id = donor["blind_case_id"]
    wrong_id = wrong_order["blind_case_id"]
    candidate = int(recipient["candidate_parent_layer_private"])
    wrong_depth = int(recipient["wrong_depth_parent_layer_private"])
    if scenario == "no_intervention":
        return None, None, None, recipient
    if scenario == "identity_candidate_parent":
        return candidate, int(recipient["query_end_position_private"]), states[recipient_id]["candidate_query"], recipient
    if scenario == "same_order_joint_donor_candidate":
        return candidate, int(recipient["query_end_position_private"]), states[donor_id]["candidate_query"], donor
    if scenario == "wrong_order_joint_donor_candidate":
        return candidate, int(recipient["query_end_position_private"]), states[wrong_id]["candidate_query"], wrong_order
    if scenario == "same_order_joint_donor_wrong_depth":
        return wrong_depth, int(recipient["query_end_position_private"]), states[donor_id]["wrong_query"], donor
    if scenario == "same_order_donor_answer_anchor_control":
        return candidate, int(recipient["answer_anchor_position_private"]), states[donor_id]["candidate_answer"], donor
    raise KeyError(scenario)


def run(model: str, split: str, max_new_tokens: int) -> dict[str, Any]:
    protocol = read_json(OUT / "phase398_order_conditioned_causal_protocol.json")
    if split == "instrument" and not protocol["authorization"]["run_instrument"]:
        raise RuntimeError("Phase398 causal instrument is not authorized")
    if split == "causal_test":
        audit = read_json(OUT / "phase398_order_conditioned_causal_instrument_audit.json")
        if not audit["authorization"]["run_causal_test"]:
            raise RuntimeError("Phase398 causal test is not authorized")
    source = OUT / f"causal/protocol/private/phase398_{split}_causal_cases.jsonl"
    cases = [row for row in read_jsonl(source) if row["private_execution_model"] == model]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[case["anonymous_parallel_group_id"]].append(case)
    expected_groups = 3 if split == "instrument" else 9
    if len(grouped) != expected_groups or any(len(items) != 16 for items in grouped.values()):
        raise RuntimeError(f"Invalid Phase398 {split} causal denominator for {model}")
    loaded = None
    direction_rows = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, items) in enumerate(sorted(grouped.items()), 1):
            by_key = {
                (row["axis_private"], int(row["relation_level_private"]), int(row["order_level_private"]), int(row["query_level_private"])): row
                for row in items
            }
            states = {row["blind_case_id"]: capture_parent_states(loaded, row) for row in items}
            for key, recipient in sorted(by_key.items()):
                axis, relation, order, query = key
                donor = by_key[(axis, relation, order, 1 - query)]
                wrong_order = by_key[(axis, relation, 1 - order, 1 - query)]
                if donor["target"] != wrong_order["target"] or donor["target"] == recipient["target"]:
                    raise RuntimeError("Phase398 causal donor target contract failed")
                scenario_rows = []
                for scenario in SCENARIOS:
                    patch_layer, position, vector, evaluation_donor = scenario_spec(
                        scenario, recipient, donor, wrong_order, states
                    )
                    outcome = generate_patch(
                        loaded,
                        recipient,
                        evaluation_donor,
                        patch_layer=patch_layer,
                        recipient_position=position,
                        vector=vector,
                        max_new_tokens=max_new_tokens,
                    )
                    scenario_rows.append({"scenario": scenario, **outcome})
                direction_rows.append({
                    "schema_version": "72.11.0",
                    "phase_id": "Phase398-OrderConditionedCausalIntervention",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "split": split,
                    "task_surface": recipient["task_surface_private"],
                    "public_parallel_group_id": recipient["phase398_public_parallel_group_id"],
                    "direction_id": f"{recipient['phase398_public_parallel_group_id']}:{axis}:R{relation}:O{order}:Q{query}to{1-query}",
                    "axis": axis,
                    "relation_level": relation,
                    "order_level": order,
                    "recipient_query_level": query,
                    "candidate_output_layer": recipient["candidate_output_layer_private"],
                    "candidate_parent_layer": recipient["candidate_parent_layer_private"],
                    "wrong_depth_parent_layer": recipient["wrong_depth_parent_layer_private"],
                    "scenario_rows": scenario_rows,
                    "single_neuron_claim": False,
                })
            print(f"[{model}/{split}] group {group_index}/{len(grouped)} directions={len(direction_rows)}", flush=True)
        root = OUT / f"causal/{split}/private/models/{model}"
        write_jsonl(root / "direction_rows.jsonl", direction_rows)
        summary = {
            "schema_version": "72.11.0",
            "phase_id": "Phase398-OrderConditionedCausalIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "split": split,
            "group_count": len(grouped),
            "direction_count": len(direction_rows),
            "scenario_count": len(direction_rows) * len(SCENARIOS),
            "valid": len(direction_rows) == expected_groups * 16,
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
    parser.add_argument("--split", choices=("instrument", "causal_test"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()
    run(args.model, args.split, args.max_new_tokens)
