#!/usr/bin/env python3
"""Run graph-consistent Phase395 binding/content/control interventions."""

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
from phase392_parent_boundary_intervention import projection_metrics, run_path  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase395_natural_binding"
CASES = OUT / "protocol/private/phase395_calibration_causal_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")
SCENARIOS = (
    "no_intervention",
    "identity_same_literal_candidate",
    "donor_same_literal_candidate",
    "donor_same_position_candidate",
    "donor_source_entities_candidate",
    "donor_same_count_random_candidate",
    "donor_query_candidate",
    "donor_full_source_candidate",
    "donor_same_literal_wrong_depth",
)
GENERATION_SCENARIOS = set(SCENARIOS) - {
    "no_intervention",
    "identity_same_literal_candidate",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def flatten_map(case: dict[str, Any], field: str, keys: tuple[str, ...]) -> list[int]:
    return [int(position) for key in keys for position in case[field][key]]


def scenario_spec(
    scenario: str,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient: dict[str, Any],
    donor: dict[str, Any],
) -> tuple[int | None, list[int] | None, torch.Tensor | None]:
    candidate = int(recipient_case["candidate_layer"])
    wrong = int(recipient_case["wrong_depth_layer"])
    literal_keys = ("value_a", "value_b")
    entity_keys = ("entity_a", "entity_b")
    if scenario == "no_intervention":
        return None, None, None
    if scenario == "identity_same_literal_candidate":
        layer = candidate
        recipient_positions = flatten_map(recipient_case, "literal_value_positions_private", literal_keys)
        donor_positions = recipient_positions
        source = recipient
    elif scenario == "donor_same_literal_candidate":
        layer = candidate
        recipient_positions = flatten_map(recipient_case, "literal_value_positions_private", literal_keys)
        donor_positions = flatten_map(donor_case, "literal_value_positions_private", literal_keys)
        source = donor
    elif scenario == "donor_same_position_candidate":
        layer = candidate
        recipient_positions = sorted(flatten_map(recipient_case, "literal_value_positions_private", literal_keys))
        donor_positions = sorted(flatten_map(donor_case, "literal_value_positions_private", literal_keys))
        source = donor
    elif scenario == "donor_source_entities_candidate":
        layer = candidate
        recipient_positions = flatten_map(recipient_case, "source_entity_positions_private", entity_keys)
        donor_positions = flatten_map(donor_case, "source_entity_positions_private", entity_keys)
        source = donor
    elif scenario == "donor_same_count_random_candidate":
        layer = candidate
        recipient_positions = recipient_case["random_control_positions_private"]
        donor_positions = donor_case["random_control_positions_private"]
        source = donor
    elif scenario == "donor_query_candidate":
        layer = candidate
        recipient_positions = recipient_case["query_positions_private"]
        donor_positions = donor_case["query_positions_private"]
        source = donor
    elif scenario == "donor_full_source_candidate":
        layer = candidate
        recipient_positions = recipient_case["source_positions_private"]
        donor_positions = donor_case["source_positions_private"]
        source = donor
    elif scenario == "donor_same_literal_wrong_depth":
        layer = wrong
        recipient_positions = flatten_map(recipient_case, "literal_value_positions_private", literal_keys)
        donor_positions = flatten_map(donor_case, "literal_value_positions_private", literal_keys)
        source = donor
    else:
        raise KeyError(scenario)
    if len(recipient_positions) != len(donor_positions):
        raise RuntimeError(f"Phase395 position count mismatch for {scenario}")
    vectors = source["layer_inputs"][layer][0, donor_positions]
    return layer, [int(value) for value in recipient_positions], vectors


@torch.inference_mode()
def generate_patch(
    loaded: Any,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    recipient: dict[str, Any],
    donor: dict[str, Any],
    scenario: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    layer_index, positions, vectors = scenario_spec(
        scenario, recipient_case, donor_case, recipient, donor
    )
    if layer_index is None or positions is None or vectors is None:
        raise RuntimeError("Phase395 generation requires an active patch")
    layer = get_layers(loaded.model)[layer_index]
    prompt_length = len(recipient_case["prompt_token_ids_private"])
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
            "patch_audit": audit,
        }
    finally:
        handle.remove()


def selected_groups(cases: list[dict[str, Any]], mode: str) -> set[str]:
    all_groups: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        group = case["phase395_public_parallel_group_id"]
        if group not in all_groups[case["task_surface_private"]]:
            all_groups[case["task_surface_private"]].append(group)
    if mode == "calibration":
        return {group for groups in all_groups.values() for group in groups}
    return {sorted(groups)[0] for groups in all_groups.values()}


def run(model: str, mode: str, max_new_tokens: int) -> dict[str, Any]:
    if mode == "calibration":
        gate = OUT / "phase395_causal_instrument_audit.json"
        if not gate.is_file() or not read_json(gate)["authorization"]["full_calibration"]:
            raise RuntimeError("Phase395 full calibration is not authorized")
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    allowed = selected_groups(cases, mode)
    cases = [row for row in cases if row["phase395_public_parallel_group_id"] in allowed]
    expected_cases = 8 if mode == "instrument_audit" else 48
    if len(cases) != expected_cases:
        raise RuntimeError(f"Expected {expected_cases} Phase395 {mode} cases for {model}, got {len(cases)}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["phase395_public_parallel_group_id"]][case["condition_code_private"]] = case
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, conditions) in enumerate(sorted(grouped.items()), 1):
            surface = next(iter(conditions.values()))["task_surface_private"]
            capture_layers = tuple(sorted({
                int(next(iter(conditions.values()))["candidate_layer"]),
                int(next(iter(conditions.values()))["wrong_depth_layer"]),
            }))
            natural = {
                name: run_path(loaded, case, capture_layers)
                for name, case in conditions.items()
            }
            for name, outcome in natural.items():
                if not outcome["all_prefix_transitions_match"] or not outcome["target_decision_argmax_match"]:
                    raise RuntimeError(f"Natural replay failed for {model}/{group_id}/{name}")
            for recipient_name, donor_name in (("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")):
                recipient_case, donor_case = conditions[recipient_name], conditions[donor_name]
                recipient, donor = natural[recipient_name], natural[donor_name]
                recipient_token = int(recipient_case["target_first_token_id_private"])
                donor_token = int(donor_case["target_first_token_id_private"])
                recipient_margin = float(
                    recipient["decision_logits"][donor_token]
                    - recipient["decision_logits"][recipient_token]
                )
                donor_margin = float(
                    donor["decision_logits"][donor_token]
                    - donor["decision_logits"][recipient_token]
                )
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
                    margin = float(
                        outcome["decision_logits"][donor_token]
                        - outcome["decision_logits"][recipient_token]
                    )
                    projection = projection_metrics(
                        recipient["query_outputs"][int(recipient_case["candidate_layer"])],
                        donor["query_outputs"][int(recipient_case["candidate_layer"])],
                        outcome["query_outputs"][int(recipient_case["candidate_layer"])],
                    )
                    scenario_rows.append({
                        "scenario": scenario,
                        **projection,
                        "donor_vs_recipient_margin": margin,
                        "donor_margin_shift": margin - recipient_margin,
                        "normalized_margin_mediation": (margin - recipient_margin)
                        / max(abs(donor_margin - recipient_margin), 1e-8),
                        "argmax_token_id": int(torch.argmax(outcome["decision_logits"]).item()),
                        "argmax_is_donor": int(torch.argmax(outcome["decision_logits"]).item()) == donor_token,
                        "patch_audit": outcome["audit"],
                    })
                    if scenario in GENERATION_SCENARIOS:
                        generation_rows.append({
                            "scenario": scenario,
                            **generate_patch(
                                loaded,
                                recipient_case,
                                donor_case,
                                recipient,
                                donor,
                                scenario,
                                max_new_tokens,
                            ),
                        })
                rows.append({
                    "schema_version": "69.8.0",
                    "phase_id": "Phase395-BindingIntervention",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model,
                    "mode": mode,
                    "task_surface": surface,
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
                    "generation_rows": generation_rows,
                    "causal_binding_claim": False,
                    "single_neuron_claim": False,
                })
            print(
                f"[{model}/{mode}] groups {group_index}/{len(grouped)} directions={len(rows)}",
                flush=True,
            )
        root = OUT / "causal" / mode / model
        write_jsonl(root / "direction_rows.jsonl", rows)
        expected_directions = len(grouped) * 4
        summary = {
            "schema_version": "69.8.0",
            "phase_id": "Phase395-BindingIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "mode": mode,
            "group_count": len(grouped),
            "direction_count": len(rows),
            "scenario_count": len(rows) * len(SCENARIOS),
            "generation_count": len(rows) * len(GENERATION_SCENARIOS),
            "valid": len(rows) == expected_directions,
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
    parser.add_argument("--mode", choices=("instrument_audit", "calibration"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    run(args.model, args.mode, args.max_new_tokens)
