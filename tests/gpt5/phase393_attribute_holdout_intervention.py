#!/usr/bin/env python3
"""Run independent Phase393 attribute/structure/depth parent-boundary holdout."""

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
from phase392_parent_boundary_intervention import (  # noqa: E402
    positions_for,
    projection_metrics,
    run_path,
)


OUT = ROOT / "tests/gpt5/result/phase393_attribute_content_holdout"
CASES = OUT / "protocol/private/phase393_holdout_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SCENARIOS = (
    "no_intervention",
    "identity_attributes",
    "donor_attributes_candidate_depth",
    "donor_structure_candidate_depth",
    "donor_random_candidate_depth",
    "donor_attributes_wrong_depth",
)


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


def scenario_spec(
    scenario: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient: dict[str, Any],
    donor: dict[str, Any],
) -> tuple[int | None, list[int] | None, torch.Tensor | None]:
    candidate = int(recipient_case["candidate_layer"])
    wrong = int(recipient_case["wrong_depth_layer"])
    if scenario == "no_intervention":
        return None, None, None
    if scenario == "identity_attributes":
        layer, roles, source, source_case = candidate, ("attributes_items",), recipient, recipient_case
    elif scenario == "donor_attributes_candidate_depth":
        layer, roles, source, source_case = candidate, ("attributes_items",), donor, donor_case
    elif scenario == "donor_structure_candidate_depth":
        layer = candidate
        roles = tuple(recipient_case["frozen_structure_roles"])
        source, source_case = donor, donor_case
    elif scenario == "donor_random_candidate_depth":
        count = len(positions_for(recipient_case, ("attributes_items",)))
        recipient_positions = positions_for(recipient_case, ("other_causal_prefix",))[:count]
        donor_positions = positions_for(donor_case, ("other_causal_prefix",))[:count]
        return candidate, recipient_positions, donor["layer_inputs"][candidate][0, donor_positions]
    elif scenario == "donor_attributes_wrong_depth":
        layer, roles, source, source_case = wrong, ("attributes_items",), donor, donor_case
    else:
        raise KeyError(scenario)
    recipient_positions = positions_for(recipient_case, roles)
    donor_positions = positions_for(source_case, roles)
    if len(recipient_positions) != len(donor_positions):
        raise RuntimeError(f"Phase393 role mismatch for {scenario}")
    return layer, recipient_positions, source["layer_inputs"][layer][0, donor_positions]


@torch.inference_mode()
def generate_patch(
    loaded: Any,
    case: dict[str, Any],
    donor_case: dict[str, Any],
    donor: dict[str, Any],
    scenario: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    layer_index, positions, vectors = scenario_spec(
        scenario, case, donor_case, {}, donor
    )
    if layer_index is None or positions is None or vectors is None:
        raise RuntimeError("Phase393 generation requires an active patch")
    layer = get_layers(loaded.model)[layer_index]
    prompt_length = len(case["prompt_token_ids_private"])
    audit = {"patch_call_count": 0, "max_patch_error": 0.0, "max_outside_error": 0.0}

    def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        hidden = inputs[0]
        if hidden.shape[1] != prompt_length:
            return None
        index = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        source = vectors.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        patched[0, index] = source
        outside_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        outside_mask[index] = False
        outside = (patched[:, outside_mask] - hidden[:, outside_mask]).float()
        audit["patch_call_count"] += 1
        audit["max_patch_error"] = max(
            audit["max_patch_error"],
            float((patched[0, index].float() - source.float()).abs().max().item()),
        )
        audit["max_outside_error"] = max(
            audit["max_outside_error"],
            float(outside.abs().max().item()) if outside.numel() else 0.0,
        )
        return (patched, *inputs[1:])

    handle = layer.register_forward_pre_hook(hook)
    try:
        ids = torch.tensor(
            [case["prompt_token_ids_private"]], dtype=torch.long, device=loaded.input_device
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
        recipient_present = contains_alias(head, case["target_aliases"])
        return {
            "generated_text": text,
            "answer_head_text": head,
            "donor_target_present": donor_present,
            "recipient_target_present": recipient_present,
            "strict_donor_target_switch": donor_present and not recipient_present,
            "audit": audit,
        }
    finally:
        handle.remove()


def run(model: str, max_new_tokens: int) -> dict[str, Any]:
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    if len(cases) != 24:
        raise RuntimeError(f"Expected 24 Phase393 cases for {model}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["parallel_group_id"]][case["condition"]] = case
    loaded = None
    rows = []
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
            natural = {name: run_path(loaded, case, capture_layers) for name, case in conditions.items()}
            for name, outcome in natural.items():
                if not outcome["all_prefix_transitions_match"] or not outcome["target_decision_argmax_match"]:
                    raise RuntimeError(f"Phase393 natural replay failed for {model}/{group_id}/{name}")
            for recipient_name, donor_name in (("mapping_x", "mapping_y"), ("mapping_y", "mapping_x")):
                recipient_case, donor_case = conditions[recipient_name], conditions[donor_name]
                recipient, donor = natural[recipient_name], natural[donor_name]
                recipient_token = int(recipient_case["target_first_token_id_private"])
                donor_token = int(donor_case["target_first_token_id_private"])
                recipient_margin = float(recipient["decision_logits"][donor_token] - recipient["decision_logits"][recipient_token])
                donor_margin = float(donor["decision_logits"][donor_token] - donor["decision_logits"][recipient_token])
                scenario_rows = []
                generation_rows = []
                for scenario in SCENARIOS:
                    if scenario == "no_intervention":
                        outcome = recipient
                    else:
                        layer, positions, vectors = scenario_spec(
                            scenario, recipient_case, donor_case, recipient, donor
                        )
                        outcome = run_path(
                            loaded,
                            recipient_case,
                            capture_layers,
                            patch_layer=layer,
                            recipient_positions=positions,
                            patch_vectors=vectors,
                        )
                    margin = float(outcome["decision_logits"][donor_token] - outcome["decision_logits"][recipient_token])
                    scenario_rows.append(
                        {
                            "scenario": scenario,
                            "normalized_margin_mediation": (margin - recipient_margin)
                            / max(abs(donor_margin - recipient_margin), 1e-8),
                            "donor_margin_shift": margin - recipient_margin,
                            "query_projection_toward_donor": projection_metrics(
                                recipient["query_outputs"][int(recipient_case["candidate_layer"])],
                                donor["query_outputs"][int(recipient_case["candidate_layer"])],
                                outcome["query_outputs"][int(recipient_case["candidate_layer"])],
                            )["query_projection_toward_donor"],
                            "patch_audit": outcome["audit"],
                        }
                    )
                    if scenario in {
                        "donor_attributes_candidate_depth",
                        "donor_structure_candidate_depth",
                        "donor_random_candidate_depth",
                        "donor_attributes_wrong_depth",
                    }:
                        generation_rows.append(
                            {
                                "scenario": scenario,
                                **generate_patch(
                                    loaded,
                                    recipient_case,
                                    donor_case,
                                    donor,
                                    scenario,
                                    max_new_tokens,
                                ),
                            }
                        )
                rows.append(
                    {
                        "schema_version": "67.1.0",
                        "phase_id": "Phase393-AttributeHoldoutIntervention",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "parallel_group_id": group_id,
                        "direction_id": f"{group_id}:{recipient_name}<-{donor_name}",
                        "scenario_rows": scenario_rows,
                        "generation_rows": generation_rows,
                    }
                )
            print(f"[{model}] Phase393 groups {group_index}/{len(grouped)} directions={len(rows)}", flush=True)
        root = OUT / "collection" / model
        write_jsonl(root / "direction_rows.jsonl", rows)
        summary = {
            "schema_version": "67.1.0",
            "phase_id": "Phase393-AttributeHoldoutIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "group_count": len(grouped),
            "direction_count": len(rows),
            "scenario_count": len(rows) * len(SCENARIOS),
            "generation_count": len(rows) * 4,
            "valid": len(rows) == 24,
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
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()
    run(args.model, args.max_new_tokens)
