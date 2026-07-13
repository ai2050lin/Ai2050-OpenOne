#!/usr/bin/env python3
"""Run Phase397 aggregate value-state causal factor separation."""

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
from phase392_parent_boundary_intervention import projection_metrics, run_path  # noqa: E402
from phase397_causal_protocol import SCENARIOS  # noqa: E402


OUT = ROOT / "tests/gpt5/result/phase397_multitask_binding"
CASES = OUT / "factor_trace/protocol/private/phase397_discovery_trace_cases.jsonl"
MODELS = ("qwen3", "glm4", "deepseek7b")


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


def flatten(case: dict[str, Any], field: str, keys: tuple[str, ...], *, sorted_positions: bool = False) -> list[int]:
    values = [int(position) for key in keys for position in case[field][key]]
    return sorted(values) if sorted_positions else values


def scenario_spec(
    scenario: str,
    recipient_code: str,
    relation_code: str,
    controls: dict[str, str],
    cases: dict[str, dict[str, Any]],
    natural: dict[str, dict[str, Any]],
) -> tuple[int | None, list[int] | None, torch.Tensor | None]:
    recipient_case = cases[recipient_code]
    candidate = int(recipient_case["candidate_layer"])
    wrong = int(recipient_case["wrong_depth_layer"])
    value_keys = ("value_a", "value_b")
    entity_keys = ("entity_a", "entity_b")
    if scenario == "no_intervention":
        return None, None, None
    if scenario == "identity_relation_candidate":
        donor_code, layer, mapping, field = recipient_code, candidate, "literal", "literal_value_positions_private"
    elif scenario == "donor_relation_candidate":
        donor_code, layer, mapping, field = relation_code, candidate, "literal", "literal_value_positions_private"
    elif scenario == "donor_content_candidate":
        donor_code, layer, mapping, field = controls["content"], candidate, "position", "literal_value_positions_private"
    elif scenario == "donor_order_candidate":
        donor_code, layer, mapping, field = controls["order"], candidate, "literal", "literal_value_positions_private"
    elif scenario == "donor_syntax_candidate":
        donor_code, layer, mapping, field = controls["syntax"], candidate, "literal", "literal_value_positions_private"
    elif scenario == "donor_query_source_candidate":
        donor_code, layer, mapping, field = controls["query"], candidate, "literal", "literal_value_positions_private"
    elif scenario == "donor_entities_candidate":
        donor_code, layer, mapping, field = relation_code, candidate, "literal", "source_entity_positions_private"
    elif scenario == "donor_random_candidate":
        donor_code, layer, mapping, field = relation_code, candidate, "random", "random_control_positions_private"
    elif scenario == "donor_relation_wrong_depth":
        donor_code, layer, mapping, field = relation_code, wrong, "literal", "literal_value_positions_private"
    elif scenario == "donor_full_source_candidate":
        donor_code, layer, mapping, field = relation_code, candidate, "source", "source_positions_private"
    else:
        raise KeyError(scenario)

    donor_case = cases[donor_code]
    if field == "literal_value_positions_private":
        recipient_positions = flatten(recipient_case, field, value_keys, sorted_positions=mapping == "position")
        donor_positions = flatten(donor_case, field, value_keys, sorted_positions=mapping == "position")
    elif field == "source_entity_positions_private":
        recipient_positions = flatten(recipient_case, field, entity_keys)
        donor_positions = flatten(donor_case, field, entity_keys)
    else:
        recipient_positions = [int(value) for value in recipient_case[field]]
        donor_positions = [int(value) for value in donor_case[field]]
    if len(recipient_positions) != len(donor_positions):
        raise RuntimeError(f"Phase397 position count mismatch for {scenario}")
    return layer, recipient_positions, natural[donor_code]["layer_inputs"][layer][0, donor_positions]


@torch.inference_mode()
def generate_relation_patch(
    loaded: Any,
    recipient_case: dict[str, Any],
    donor_case: dict[str, Any],
    layer_index: int,
    recipient_positions: list[int],
    vectors: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    layer = get_layers(loaded.model)[layer_index]
    prompt_length = len(recipient_case["prompt_token_ids_private"])
    audit = {"patch_call_count": 0, "max_patch_error": 0.0, "max_outside_error": 0.0}

    def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
        hidden = inputs[0]
        if hidden.shape[1] != prompt_length:
            return None
        index = torch.tensor(recipient_positions, dtype=torch.long, device=hidden.device)
        source = vectors.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        patched[0, index] = source
        outside_mask = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        outside_mask[index] = False
        outside = (patched[:, outside_mask] - hidden[:, outside_mask]).float()
        audit["patch_call_count"] += 1
        audit["max_patch_error"] = max(audit["max_patch_error"], float((patched[0, index].float() - source.float()).abs().max().item()))
        audit["max_outside_error"] = max(audit["max_outside_error"], float(outside.abs().max().item()) if outside.numel() else 0.0)
        return (patched, *inputs[1:])

    handle = layer.register_forward_pre_hook(hook)
    try:
        ids = torch.tensor([recipient_case["prompt_token_ids_private"]], dtype=torch.long, device=loaded.input_device)
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


def run(model: str, max_new_tokens: int) -> dict[str, Any]:
    protocol = read_json(OUT / "phase397_causal_protocol.json")
    if not protocol["authorization"]["run_discovery_causal_three_models_sequentially"]:
        raise RuntimeError("Phase397 causal intervention is not authorized")
    cases = [row for row in read_jsonl(CASES) if row["private_execution_model"] == model]
    if len(cases) != 240:
        raise RuntimeError(f"Expected 240 discovery cases for {model}, got {len(cases)}")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for case in cases:
        grouped[case["phase397_public_parallel_group_id"]][case["condition_code_private"]] = case
    if len(grouped) != 24 or any(set(items) != set("ABCDEFGHIJ") for items in grouped.values()):
        raise RuntimeError(f"Invalid Phase397 causal groups for {model}")
    loaded = None
    rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for group_index, (group_id, conditions) in enumerate(sorted(grouped.items()), 1):
            layers = tuple(sorted({int(conditions["A"]["candidate_layer"]), int(conditions["A"]["wrong_depth_layer"])}))
            natural = {code: run_path(loaded, case, layers) for code, case in conditions.items()}
            if any(not outcome["all_prefix_transitions_match"] or not outcome["target_decision_argmax_match"] for outcome in natural.values()):
                raise RuntimeError(f"Phase397 causal natural replay failed {model}/{group_id}")
            directions = (
                ("A", "B", {"content": "F", "order": "C", "syntax": "D", "query": "E"}),
                ("F", "G", {"content": "A", "order": "H", "syntax": "I", "query": "J"}),
            )
            for recipient_code, relation_code, controls in directions:
                recipient_case = conditions[recipient_code]
                relation_case = conditions[relation_code]
                recipient = natural[recipient_code]
                relation = natural[relation_code]
                recipient_token = int(recipient_case["target_first_token_id_private"])
                relation_token = int(relation_case["target_first_token_id_private"])
                recipient_margin = float(recipient["decision_logits"][relation_token] - recipient["decision_logits"][recipient_token])
                donor_margin = float(relation["decision_logits"][relation_token] - relation["decision_logits"][recipient_token])
                scenario_rows = []
                generation_row = None
                for scenario in SCENARIOS:
                    if scenario == "no_intervention":
                        outcome = recipient
                        layer_index = None
                        recipient_positions = None
                        vectors = None
                    else:
                        layer_index, recipient_positions, vectors = scenario_spec(
                            scenario, recipient_code, relation_code, controls, conditions, natural
                        )
                        outcome = run_path(
                            loaded,
                            recipient_case,
                            layers,
                            patch_layer=layer_index,
                            recipient_positions=recipient_positions,
                            patch_vectors=vectors,
                        )
                    margin = float(outcome["decision_logits"][relation_token] - outcome["decision_logits"][recipient_token])
                    scenario_rows.append(
                        {
                            "scenario": scenario,
                            **projection_metrics(
                                recipient["query_outputs"][int(recipient_case["candidate_layer"])],
                                relation["query_outputs"][int(recipient_case["candidate_layer"])],
                                outcome["query_outputs"][int(recipient_case["candidate_layer"])],
                            ),
                            "relation_vs_recipient_margin": margin,
                            "relation_margin_shift": margin - recipient_margin,
                            "normalized_relation_margin_mediation": (margin - recipient_margin) / max(abs(donor_margin - recipient_margin), 1e-8),
                            "argmax_token_id": int(torch.argmax(outcome["decision_logits"]).item()),
                            "argmax_is_relation_donor": int(torch.argmax(outcome["decision_logits"]).item()) == relation_token,
                            "patch_audit": outcome["audit"],
                        }
                    )
                    if scenario == "donor_relation_candidate":
                        if layer_index is None or recipient_positions is None or vectors is None:
                            raise RuntimeError("Missing relation generation patch")
                        generation_row = generate_relation_patch(
                            loaded, recipient_case, relation_case, layer_index, recipient_positions, vectors, max_new_tokens
                        )
                rows.append(
                    {
                        "schema_version": "71.10.0",
                        "phase_id": "Phase397-CausalIntervention",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model": model,
                        "task_surface": recipient_case["task_surface_private"],
                        "parallel_group_id": group_id,
                        "direction_id": f"{group_id}:{recipient_code}<-{relation_code}",
                        "recipient_condition": recipient_code,
                        "relation_donor_condition": relation_code,
                        "candidate_layer": int(recipient_case["candidate_layer"]),
                        "wrong_depth_layer": int(recipient_case["wrong_depth_layer"]),
                        "recipient_natural_margin": recipient_margin,
                        "relation_donor_natural_margin": donor_margin,
                        "natural_margin_separation": donor_margin - recipient_margin,
                        "scenario_rows": scenario_rows,
                        "relation_generation_row": generation_row,
                        "causal_relation_binding_claim": False,
                        "single_neuron_claim": False,
                    }
                )
            print(f"[{model}/causal] groups {group_index}/{len(grouped)} directions={len(rows)}", flush=True)
        root = OUT / "causal/discovery" / model
        write_jsonl(root / "direction_rows.jsonl", rows)
        summary = {
            "schema_version": "71.10.0",
            "phase_id": "Phase397-CausalIntervention",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "group_count": len(grouped),
            "direction_count": len(rows),
            "scenario_count": len(rows) * len(SCENARIOS),
            "generation_count": len(rows),
            "valid": len(rows) == 48,
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
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()
    run(args.model, args.max_new_tokens)
