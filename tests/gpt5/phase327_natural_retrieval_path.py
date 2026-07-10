#!/usr/bin/env python3
"""Phase327 natural object-to-carrier-to-answer path audit.

The Phase326 carrier selections are frozen.  This phase does not discover new
components; it tests natural identity structure, position-separated necessity,
natural-state transplantation, and complete generated answers.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

import phase326_distributed_carrier_atlas as phase326  # noqa: E402
import phase327_natural_retrieval_case_bank as case_bank  # noqa: E402
from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase327"
SCHEMA_VERSION = "6.0.0"
MODELS = ("qwen3", "glm4", "deepseek7b")
ROUND_DEFAULT = "natural_retrieval_path"
OUT = ROOT / "tests/gpt5/result/phase327_natural_retrieval_path"
PHASE326 = ROOT / "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas"
ROLES = ("source", "query", "last")
POSITION_CONDITIONS = (
    "baseline",
    "source_zero",
    "query_zero",
    "last_zero",
    "source_query_zero",
    "query_last_zero",
    "joint_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
)
TRANSPLANT_CONDITIONS = (
    "recipient_baseline",
    "correct_donor_transplant",
    "same_target_donor_transplant",
    "same_semantic_wrong_donor_transplant",
    "unrelated_donor_transplant",
    "correct_donor_wrong_layer_transplant",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_mean(values: list[float]) -> float:
    return round(mean(values), 6) if values else 0.0


def ratio(numerator: float, denominator: float) -> float:
    return round(numerator / max(abs(denominator), 1e-8), 6)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def target_token_ids(tokenizer: Any, target: str) -> list[int]:
    for text in (" " + target, target):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            return [int(value) for value in ids]
    raise ValueError(f"Cannot tokenize target {target!r}")


def spec_key(spec: dict[str, Any]) -> str:
    return ":".join(str(value) for value in (
        spec["component_type"], spec["component_layer"], spec["position_role"],
        spec["component_start"], spec["component_end"],
    ))


def variant_case(case: dict[str, Any], variant: str) -> dict[str, Any]:
    row = case["variants"][variant]
    return {
        **case,
        "prompt": row["prompt"],
        "source_fragments": [row["subject"]],
        "subject": row["subject"],
        "semantic_group": row["semantic_group"],
        "natural_target": row["natural_target"],
        "variant": variant,
    }


def selections_for_case(selections: list[dict[str, Any]], case: dict[str, Any]) -> list[dict[str, Any]]:
    return phase326.selections_for(selections, case)


def role_specs(specs: list[dict[str, Any]], roles: tuple[str, ...]) -> list[dict[str, Any]]:
    return [spec for spec in specs if spec["position_role"] in roles]


def module_for(model_obj: Any, spec: dict[str, Any]) -> Any:
    layer = int(spec["component_layer"])
    if spec["component_type"] == "attention_head_input":
        return head_meta(model_obj, layer)[0]
    module = phase326.get_down_proj(get_layers(model_obj)[layer])
    if module is None:
        raise TypeError(f"Missing MLP down projection at layer {layer}")
    return module


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return round(float(torch.dot(a, b).item() / denominator.item()), 6)


def rms_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return round(float(torch.sqrt(torch.mean((a - b) ** 2)).item()), 6)


@torch.inference_mode()
def forward_probe(
    loaded: Any,
    case: dict[str, Any],
    target: str,
    distractors: list[str],
    *,
    mutate_specs: list[dict[str, Any]] | None = None,
    capture_specs: list[dict[str, Any]] | None = None,
    transplant_specs: list[dict[str, Any]] | None = None,
    transplant_values: dict[str, torch.Tensor] | None = None,
    capture_hidden: bool = False,
    capture_hidden_tokens: bool = False,
    residual_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mutate_specs = mutate_specs or []
    capture_specs = capture_specs or []
    transplant_specs = transplant_specs or []
    transplant_values = transplant_values or {}
    prompt_batch = loaded.tokenizer(
        case["prompt"], return_tensors="pt", truncation=True, max_length=96
    )
    prompt_ids = prompt_batch["input_ids"].to(loaded.input_device)
    prompt_mask = prompt_batch["attention_mask"].to(loaded.input_device)
    prompt_len = int(prompt_mask.sum().item())
    answer_ids = target_token_ids(loaded.tokenizer, target)
    answer = torch.tensor([answer_ids], dtype=prompt_ids.dtype, device=prompt_ids.device)
    input_ids = torch.cat((prompt_ids, answer), dim=1)
    attention_mask = torch.cat((prompt_mask, torch.ones_like(answer)), dim=1)
    spans = phase326.role_spans(loaded.tokenizer, case["prompt"], case, prompt_len)

    mutate_by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    capture_by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    transplant_by_module: dict[int, list[dict[str, Any]]] = defaultdict(list)
    modules: dict[int, Any] = {}
    for bucket, specs in (
        (mutate_by_module, mutate_specs),
        (capture_by_module, capture_specs),
        (transplant_by_module, transplant_specs),
    ):
        for spec in specs:
            module = module_for(loaded.model, spec)
            key = id(module)
            modules[key] = module
            bucket[key].append(spec)

    captures: dict[str, torch.Tensor] = {}
    energies: dict[str, list[float]] = defaultdict(list)
    handles = []
    if residual_patch:
        residual_layer = int(residual_patch["layer"])
        residual_role = str(residual_patch["position_role"])
        residual_value = residual_patch["value"]

        def residual_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 3:
                return None
            tensor = inputs[0]
            positions = [
                position
                for position in phase326.role_positions(spans[residual_role])
                if position < min(prompt_len, tensor.shape[1])
            ]
            if not positions:
                return None
            changed = tensor.clone()
            value = residual_value.to(changed.device, changed.dtype)
            if value.ndim == 1:
                changed[0, positions, :] = value.view(1, -1)
            elif value.ndim == 2 and value.shape[0] == len(positions):
                changed[0, positions, :] = value
            else:
                raise ValueError(
                    "Residual patch must be [hidden] or [role_tokens, hidden]; "
                    f"got {tuple(value.shape)} for {len(positions)} positions"
                )
            return (changed, *inputs[1:])

        handles.append(get_layers(loaded.model)[residual_layer].register_forward_pre_hook(residual_hook))
    for key, module in modules.items():
        observed = capture_by_module.get(key, [])
        zeroed = mutate_by_module.get(key, [])
        transplanted = transplant_by_module.get(key, [])

        def hook(
            _module: Any,
            inputs: tuple[Any, ...],
            observed: list[dict[str, Any]] = observed,
            zeroed: list[dict[str, Any]] = zeroed,
            transplanted: list[dict[str, Any]] = transplanted,
        ) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 3:
                return None
            tensor = inputs[0]

            def positions(spec: dict[str, Any]) -> list[int]:
                return [
                    position for position in phase326.role_positions(spans[str(spec["position_role"])])
                    if position < min(prompt_len, tensor.shape[1])
                ]

            for spec in observed:
                pos = positions(spec)
                if not pos:
                    continue
                start, end = int(spec["component_start"]), int(spec["component_end"])
                value = tensor[0, pos, start:end].detach().float().mean(dim=0).cpu()
                captures[spec_key(spec)] = value
                energies[str(spec["component_type"])].append(
                    float(torch.linalg.vector_norm(value).item() / math.sqrt(max(1, value.numel())))
                )
            if not zeroed and not transplanted:
                return None
            changed = tensor.clone()
            for spec in zeroed:
                pos = positions(spec)
                if pos:
                    changed[0, pos, int(spec["component_start"]):int(spec["component_end"])] = 0
            for spec in transplanted:
                pos = positions(spec)
                value = transplant_values.get(spec_key(spec))
                if pos and value is not None:
                    start, end = int(spec["component_start"]), int(spec["component_end"])
                    changed[0, pos, start:end] = value.to(changed.device, changed.dtype).view(1, -1)
            return (changed, *inputs[1:])

        handles.append(module.register_forward_pre_hook(hook))
    try:
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=capture_hidden or capture_hidden_tokens,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    first_logits = output.logits[0, prompt_len - 1].detach().float().cpu()
    target_id = phase326.answer_token_id(loaded.tokenizer, target)
    distractor_ids = [phase326.answer_token_id(loaded.tokenizer, value) for value in distractors]
    token_log_probs = []
    for offset, token_id in enumerate(answer_ids):
        logits = output.logits[0, prompt_len - 1 + offset].detach().float()
        token_log_probs.append(float(torch.log_softmax(logits, dim=-1)[token_id].item()))
    hidden_vectors: dict[str, torch.Tensor] = {}
    hidden_token_vectors: dict[str, torch.Tensor] = {}
    if capture_hidden or capture_hidden_tokens:
        for layer, hidden in enumerate(output.hidden_states[1:]):
            for role, span in spans.items():
                key = f"L{layer}:{role}"
                if capture_hidden:
                    hidden_vectors[key] = phase326.pool_role(hidden, span)
                if capture_hidden_tokens:
                    start, end = span
                    hidden_token_vectors[key] = hidden[0, start : end + 1].detach().float().cpu()
    del output
    return {
        "first_logits": first_logits,
        "metrics": phase326.output_metrics(first_logits, target_id, distractor_ids),
        "phrase_logprob_sum": round(sum(token_log_probs), 6),
        "phrase_logprob_mean": safe_mean(token_log_probs),
        "target_token_count": len(answer_ids),
        "captures": captures,
        "hidden_vectors": hidden_vectors,
        "hidden_token_vectors": hidden_token_vectors,
        "attention_energy": safe_mean(energies.get("attention_head_input", [])),
        "mlp_energy": safe_mean(energies.get("mlp_product_group", [])),
        "joint_energy": safe_mean([
            *energies.get("attention_head_input", []),
            *energies.get("mlp_product_group", []),
        ]),
        "prompt_token_count": prompt_len,
        "role_spans": {role: list(span) for role, span in spans.items()},
    }


def natural_target_metrics(loaded: Any, logits: torch.Tensor, case: dict[str, Any], natural_target: str) -> dict[str, Any]:
    target_values = sorted({row["target"] for row in case_bank.OBJECTS[case["mechanism_id"]]})
    target_id = phase326.answer_token_id(loaded.tokenizer, natural_target)
    distractors = [
        phase326.answer_token_id(loaded.tokenizer, value)
        for value in target_values if value != natural_target
    ]
    return phase326.output_metrics(logits, target_id, distractors)


def batch_a_natural_identity(
    loaded: Any,
    cases: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    natural_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        specs = selections_for_case(selections, case)
        observations: dict[str, dict[str, Any]] = {}
        for variant in case_bank.VARIANTS:
            probe_case = variant_case(case, variant)
            result = forward_probe(
                loaded, probe_case, case["target"], case["distractors"],
                capture_specs=specs, capture_hidden=True,
            )
            natural_metrics = natural_target_metrics(
                loaded, result["first_logits"], case, probe_case["natural_target"]
            )
            observations[variant] = result
            natural_rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": loaded.key,
                "batch": "A_natural_identity",
                "case_id": case["case_id"],
                "base_case_id": case["base_case_id"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "split": case["split"],
                "template_id": case["template_id"],
                "variant": variant,
                "subject": probe_case["subject"],
                "evaluation_target": case["target"],
                "natural_target": probe_case["natural_target"],
                **result["metrics"],
                "phrase_logprob_sum": result["phrase_logprob_sum"],
                "phrase_logprob_mean": result["phrase_logprob_mean"],
                "natural_target_margin": natural_metrics["target_margin"],
                "natural_target_winner": natural_metrics["candidate_winner_is_target"],
                "attention_energy": result["attention_energy"],
                "mlp_energy": result["mlp_energy"],
                "joint_energy": result["joint_energy"],
                "carrier_member_count": len(specs),
                "selection_frozen_from": "Phase326",
                "causal": False,
            })
        correct = observations["correct_object"]
        for variant in case_bank.VARIANTS[1:]:
            other = observations[variant]
            carrier_keys = sorted(set(correct["captures"]) & set(other["captures"]))
            carrier_cosines = [
                cosine(correct["captures"][key], other["captures"][key]) for key in carrier_keys
            ]
            carrier_deltas = [
                rms_delta(correct["captures"][key], other["captures"][key]) for key in carrier_keys
            ]
            hidden_keys = sorted(set(correct["hidden_vectors"]) & set(other["hidden_vectors"]))
            by_layer: dict[int, dict[str, float]] = defaultdict(dict)
            for key in hidden_keys:
                layer_text, role = key.split(":", 1)
                layer = int(layer_text[1:])
                by_layer[layer][role] = rms_delta(
                    correct["hidden_vectors"][key], other["hidden_vectors"][key]
                )
                delta_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    "model": loaded.key,
                    "batch": "A_residual_path",
                    "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"],
                    "family_id": case["family_id"],
                    "mechanism_id": case["mechanism_id"],
                    "split": case["split"],
                    "template_id": case["template_id"],
                    "comparison_variant": variant,
                    "layer": layer,
                    "position_role": role,
                    "residual_rms_delta": by_layer[layer][role],
                    "residual_cosine": cosine(
                        correct["hidden_vectors"][key], other["hidden_vectors"][key]
                    ),
                    "carrier_cosine_to_correct": safe_mean(carrier_cosines),
                    "carrier_rms_delta_to_correct": safe_mean(carrier_deltas),
                    "causal": False,
                })
            for row in delta_rows[-len(hidden_keys):]:
                values = by_layer[int(row["layer"])]
                row["source_to_query_delta_ratio"] = ratio(
                    values.get("query", 0.0), values.get("source", 0.0)
                )
                row["source_to_last_delta_ratio"] = ratio(
                    values.get("last", 0.0), values.get("source", 0.0)
                )
        if number % 6 == 0 or number == len(cases):
            print(f"[{loaded.key}] batch A {number}/{len(cases)}", flush=True)
    return natural_rows, delta_rows


def position_conditions(model_obj: Any, specs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "baseline": [],
        "source_zero": role_specs(specs, ("source",)),
        "query_zero": role_specs(specs, ("query",)),
        "last_zero": role_specs(specs, ("last",)),
        "source_query_zero": role_specs(specs, ("source", "query")),
        "query_last_zero": role_specs(specs, ("query", "last")),
        "joint_zero": specs,
        "matched_random_joint_zero": phase326.randomize_specs(model_obj, specs),
        "wrong_layer_joint_zero": phase326.wrong_layer_specs(model_obj, specs),
    }


def batch_b_position_necessity(
    loaded: Any,
    cases: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, case in enumerate(cases, start=1):
        specs = selections_for_case(selections, case)
        conditions = position_conditions(loaded.model, specs)
        baseline = forward_probe(loaded, case, case["target"], case["distractors"])
        for condition in POSITION_CONDITIONS:
            result = baseline if condition == "baseline" else forward_probe(
                loaded, case, case["target"], case["distractors"],
                mutate_specs=conditions[condition],
            )
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": loaded.key,
                "batch": "B_position_necessity",
                "case_id": case["case_id"],
                "base_case_id": case["base_case_id"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "split": case["split"],
                "template_id": case["template_id"],
                "condition": condition,
                "selected_component_count": len(conditions[condition]),
                "baseline_target_margin": baseline["metrics"]["target_margin"],
                "baseline_phrase_logprob_sum": baseline["phrase_logprob_sum"],
                **result["metrics"],
                "phrase_logprob_sum": result["phrase_logprob_sum"],
                "phrase_logprob_mean": result["phrase_logprob_mean"],
                "target_token_count": result["target_token_count"],
                "target_margin_drop": round(
                    baseline["metrics"]["target_margin"] - result["metrics"]["target_margin"], 6
                ),
                "phrase_logprob_drop": round(
                    baseline["phrase_logprob_sum"] - result["phrase_logprob_sum"], 6
                ),
                "js_divergence_from_baseline": 0.0 if condition == "baseline" else phase326.js_divergence(
                    baseline["first_logits"], result["first_logits"]
                ),
                "selection_frozen_from": "Phase326",
                "causal_scope": "distributed_component_set" if condition != "baseline" else "none",
                "single_unit_causal": False,
            })
        if number % 9 == 0 or number == len(cases):
            print(f"[{loaded.key}] batch B {number}/{len(cases)}", flush=True)
    return rows


def capture_donor(
    loaded: Any,
    case: dict[str, Any],
    variant: str,
    specs: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    probe_case = variant_case(case, variant)
    return forward_probe(
        loaded, probe_case, case["target"], case["distractors"], capture_specs=specs
    )["captures"]


def batch_c_natural_transplant(
    loaded: Any,
    cases: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    generation_states: dict[str, dict[str, Any]] = {}
    for number, case in enumerate(cases, start=1):
        specs = selections_for_case(selections, case)
        wrong_specs = phase326.wrong_layer_specs(loaded.model, specs)
        recipient = variant_case(case, "same_semantic_wrong_target")
        correct_values = capture_donor(loaded, case, "correct_object", specs)
        same_target_values = capture_donor(loaded, case, "same_target_object", specs)
        same_semantic_values = capture_donor(loaded, case, "same_semantic_wrong_target", specs)
        unrelated_values = capture_donor(loaded, case, "unrelated_wrong_target", specs)
        wrong_layer_values = capture_donor(loaded, case, "correct_object", wrong_specs)
        settings = {
            "recipient_baseline": ([], {}),
            "correct_donor_transplant": (specs, correct_values),
            "same_target_donor_transplant": (specs, same_target_values),
            "same_semantic_wrong_donor_transplant": (specs, same_semantic_values),
            "unrelated_donor_transplant": (specs, unrelated_values),
            "correct_donor_wrong_layer_transplant": (wrong_specs, wrong_layer_values),
        }
        baseline = forward_probe(loaded, recipient, case["target"], case["distractors"])
        for condition in TRANSPLANT_CONDITIONS:
            transplant_specs, values = settings[condition]
            result = baseline if condition == "recipient_baseline" else forward_probe(
                loaded, recipient, case["target"], case["distractors"],
                transplant_specs=transplant_specs, transplant_values=values,
            )
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": loaded.key,
                "batch": "C_natural_state_transplant",
                "case_id": case["case_id"],
                "base_case_id": case["base_case_id"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "split": case["split"],
                "template_id": case["template_id"],
                "condition": condition,
                "recipient_subject": recipient["subject"],
                "recipient_natural_target": recipient["natural_target"],
                "evaluation_target": case["target"],
                "selected_component_count": len(transplant_specs),
                "recipient_baseline_target_margin": baseline["metrics"]["target_margin"],
                "recipient_baseline_phrase_logprob_sum": baseline["phrase_logprob_sum"],
                **result["metrics"],
                "phrase_logprob_sum": result["phrase_logprob_sum"],
                "phrase_logprob_mean": result["phrase_logprob_mean"],
                "target_margin_gain": round(
                    result["metrics"]["target_margin"] - baseline["metrics"]["target_margin"], 6
                ),
                "phrase_logprob_gain": round(
                    result["phrase_logprob_sum"] - baseline["phrase_logprob_sum"], 6
                ),
                "js_divergence_from_recipient": 0.0 if condition == "recipient_baseline" else phase326.js_divergence(
                    baseline["first_logits"], result["first_logits"]
                ),
                "donor_is_natural_prompt_state": condition != "recipient_baseline",
                "selection_frozen_from": "Phase326",
                "causal_scope": "distributed_component_state_transplant" if condition != "recipient_baseline" else "none",
                "single_unit_causal": False,
            })
        if case["template_id"] == "template_g":
            generation_states[case["case_id"]] = {
                "recipient": recipient,
                "specs": specs,
                "correct_values": correct_values,
            }
        if number % 9 == 0 or number == len(cases):
            print(f"[{loaded.key}] batch C {number}/{len(cases)}", flush=True)
    return rows, generation_states


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


@torch.inference_mode()
def greedy_generate(
    loaded: Any,
    case: dict[str, Any],
    *,
    mutate_specs: list[dict[str, Any]] | None = None,
    transplant_specs: list[dict[str, Any]] | None = None,
    transplant_values: dict[str, torch.Tensor] | None = None,
    residual_patch: dict[str, Any] | None = None,
    max_new_tokens: int = 4,
) -> str:
    batch = loaded.tokenizer(case["prompt"], return_tensors="pt", truncation=True, max_length=96)
    input_ids = batch["input_ids"].to(loaded.input_device)
    generated: list[int] = []
    eos = loaded.tokenizer.eos_token_id
    # Reuse forward_probe's intervention contract while scoring a throwaway
    # target, then take the actual argmax from the returned prompt logits.
    for _step in range(max_new_tokens):
        current = loaded.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        dynamic_case = {**case, "prompt": current}
        # Keep original source/query fragments. Generated tokens are not allowed
        # to redefine the registered last-position intervention.
        result = _generation_forward(
            loaded, dynamic_case, case,
            mutate_specs or [], transplant_specs or [], transplant_values or {},
            input_ids, residual_patch,
        )
        next_id = int(torch.argmax(result).item())
        generated.append(next_id)
        input_ids = torch.cat((input_ids, torch.tensor([[next_id]], device=input_ids.device)), dim=1)
        if eos is not None and next_id == eos:
            break
    return loaded.tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def _generation_forward(
    loaded: Any,
    dynamic_case: dict[str, Any],
    registered_case: dict[str, Any],
    mutate_specs: list[dict[str, Any]],
    transplant_specs: list[dict[str, Any]],
    transplant_values: dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    residual_patch: dict[str, Any] | None = None,
) -> torch.Tensor:
    original_prompt_ids = loaded.tokenizer(registered_case["prompt"], return_tensors="pt")["input_ids"]
    prompt_len = int(original_prompt_ids.shape[1])
    spans = phase326.role_spans(
        loaded.tokenizer, registered_case["prompt"], registered_case, prompt_len
    )
    by_module: dict[int, dict[str, Any]] = {}
    for mode, specs in (("zero", mutate_specs), ("transplant", transplant_specs)):
        for spec in specs:
            module = module_for(loaded.model, spec)
            entry = by_module.setdefault(id(module), {"module": module, "zero": [], "transplant": []})
            entry[mode].append(spec)
    handles = []
    if residual_patch:
        residual_layer = int(residual_patch["layer"])
        residual_role = str(residual_patch["position_role"])
        residual_value = residual_patch["value"]

        def residual_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 3:
                return None
            changed = inputs[0].clone()
            positions = [
                position
                for position in phase326.role_positions(spans[residual_role])
                if position < min(prompt_len, changed.shape[1])
            ]
            if not positions:
                return None
            value = residual_value.to(changed.device, changed.dtype)
            if value.ndim == 1:
                changed[0, positions, :] = value.view(1, -1)
            elif value.ndim == 2 and value.shape[0] == len(positions):
                changed[0, positions, :] = value
            else:
                raise ValueError(
                    "Residual generation patch must be [hidden] or [role_tokens, hidden]; "
                    f"got {tuple(value.shape)} for {len(positions)} positions"
                )
            return (changed, *inputs[1:])

        handles.append(
            get_layers(loaded.model)[residual_layer].register_forward_pre_hook(residual_hook)
        )
    for entry in by_module.values():
        zeroed = entry["zero"]
        transplanted = entry["transplant"]

        def hook(
            _module: Any, inputs: tuple[Any, ...], zeroed: list[dict[str, Any]] = zeroed,
            transplanted: list[dict[str, Any]] = transplanted,
        ) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 3:
                return None
            changed = inputs[0].clone()
            for spec in zeroed:
                positions = [
                    p for p in phase326.role_positions(spans[str(spec["position_role"])])
                    if p < min(prompt_len, changed.shape[1])
                ]
                if positions:
                    changed[0, positions, int(spec["component_start"]):int(spec["component_end"])] = 0
            for spec in transplanted:
                positions = [
                    p for p in phase326.role_positions(spans[str(spec["position_role"])])
                    if p < min(prompt_len, changed.shape[1])
                ]
                value = transplant_values.get(spec_key(spec))
                if positions and value is not None:
                    start, end = int(spec["component_start"]), int(spec["component_end"])
                    changed[0, positions, start:end] = value.to(changed.device, changed.dtype).view(1, -1)
            return (changed, *inputs[1:])

        handles.append(entry["module"].register_forward_pre_hook(hook))
    try:
        output = loaded.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            return_dict=True,
        )
        return output.logits[0, -1].detach().float()
    finally:
        for handle in handles:
            handle.remove()


def batch_d_generation(
    loaded: Any,
    cases: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    generation_states: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generation_cases = [case for case in cases if case["template_id"] == "template_g"]
    for number, case in enumerate(generation_cases, start=1):
        specs = selections_for_case(selections, case)
        state = generation_states[case["case_id"]]
        conditions = {
            "correct_baseline": (case, [], [], {}),
            "correct_joint_zero": (case, specs, [], {}),
            "recipient_baseline": (state["recipient"], [], [], {}),
            "recipient_correct_donor_transplant": (
                state["recipient"], [], state["specs"], state["correct_values"]
            ),
        }
        for condition, (prompt_case, zeroed, transplanted, values) in conditions.items():
            generated = greedy_generate(
                loaded, prompt_case, mutate_specs=zeroed,
                transplant_specs=transplanted, transplant_values=values,
            )
            words = normalized_words(generated)
            target_words = normalized_words(case["target"])
            exact = words == target_words
            first_match = bool(words and target_words and words[0] == target_words[0])
            rows.append({
                "schema_version": SCHEMA_VERSION,
                "phase_id": PHASE,
                "created_at": now(),
                "model": loaded.key,
                "batch": "D_complete_generation",
                "case_id": case["case_id"],
                "base_case_id": case["base_case_id"],
                "family_id": case["family_id"],
                "mechanism_id": case["mechanism_id"],
                "split": case["split"],
                "template_id": case["template_id"],
                "condition": condition,
                "target": case["target"],
                "generated_text": generated,
                "generated_words": words,
                "exact_target_match": exact,
                "first_word_target_match": first_match,
                "one_word_protocol_compliant": len(words) == 1,
                "causal_scope": "distributed_component_set" if condition != "correct_baseline" else "none",
                "single_unit_causal": False,
            })
        if number % 9 == 0 or number == len(generation_cases):
            print(f"[{loaded.key}] batch D {number}/{len(generation_cases)}", flush=True)
    return rows


def rows_for(rows: list[dict[str, Any]], mechanism: str, condition: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["mechanism_id"] == mechanism and row.get("condition") == condition
    ]


def mechanism_audits(
    model: str,
    natural_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    transplant_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audits = []
    for mechanism in case_bank.OBJECTS:
        natural = [row for row in natural_rows if row["mechanism_id"] == mechanism]
        correct = [row for row in natural if row["variant"] == "correct_object"]
        same_target_delta = [
            row for row in delta_rows
            if row["mechanism_id"] == mechanism
            and row["comparison_variant"] == "same_target_object"
            and row["position_role"] == "last"
        ]
        wrong_delta = [
            row for row in delta_rows
            if row["mechanism_id"] == mechanism
            and row["comparison_variant"] in {
                "same_semantic_wrong_target", "token_length_wrong_target", "unrelated_wrong_target"
            }
            and row["position_role"] == "last"
        ]
        same_carrier_cos = safe_mean([row["carrier_cosine_to_correct"] for row in same_target_delta])
        wrong_carrier_cos = safe_mean([row["carrier_cosine_to_correct"] for row in wrong_delta])
        same_residual_delta = safe_mean([row["residual_rms_delta"] for row in same_target_delta])
        wrong_residual_delta = safe_mean([row["residual_rms_delta"] for row in wrong_delta])
        natural_specificity = round(same_carrier_cos - wrong_carrier_cos, 6)
        residual_specificity = round(wrong_residual_delta - same_residual_delta, 6)
        baseline_accuracy = round(
            sum(row["candidate_winner_is_target"] for row in correct) / max(1, len(correct)), 6
        )
        natural_gate_pass = (
            len(correct) == 36 and baseline_accuracy >= 0.60
            and natural_specificity > 0.0 and residual_specificity > 0.0
        )

        baseline_b = rows_for(position_rows, mechanism, "baseline")
        eligible = {row["case_id"] for row in baseline_b if row["candidate_winner_is_target"]}

        def eligible_values(condition: str, field: str) -> list[float]:
            return [
                float(row[field]) for row in rows_for(position_rows, mechanism, condition)
                if row["case_id"] in eligible
            ]

        joint_margin = eligible_values("joint_zero", "target_margin_drop")
        joint_phrase = eligible_values("joint_zero", "phrase_logprob_drop")
        random_phrase = eligible_values("matched_random_joint_zero", "phrase_logprob_drop")
        wrong_layer_phrase = eligible_values("wrong_layer_joint_zero", "phrase_logprob_drop")
        role_phrase = {
            role: safe_mean(eligible_values(f"{role}_zero", "phrase_logprob_drop"))
            for role in ROLES
        }
        joint_phrase_mean = safe_mean(joint_phrase)
        position_specificity = round(
            joint_phrase_mean - max(safe_mean(random_phrase), safe_mean(wrong_layer_phrase)), 6
        )
        position_consistency = round(
            sum(value > 0 for value in joint_phrase) / max(1, len(joint_phrase)), 6
        )
        position_pass = (
            len(eligible) >= 18 and safe_mean(joint_margin) > 0.0 and joint_phrase_mean > 0.0
            and position_specificity > 0.0 and position_consistency >= 0.65
            and any(value > 0.0 for value in role_phrase.values())
        )

        baseline_c = rows_for(transplant_rows, mechanism, "recipient_baseline")
        base_by_id = {row["case_id"]: row for row in baseline_c}

        def transplant_values(condition: str, field: str) -> list[float]:
            return [
                float(row[field]) for row in rows_for(transplant_rows, mechanism, condition)
                if row["case_id"] in base_by_id
            ]

        correct_gain = transplant_values("correct_donor_transplant", "phrase_logprob_gain")
        same_target_gain = transplant_values("same_target_donor_transplant", "phrase_logprob_gain")
        wrong_donor_gain = transplant_values("same_semantic_wrong_donor_transplant", "phrase_logprob_gain")
        unrelated_gain = transplant_values("unrelated_donor_transplant", "phrase_logprob_gain")
        wrong_layer_gain = transplant_values("correct_donor_wrong_layer_transplant", "phrase_logprob_gain")
        positive_gain = min(safe_mean(correct_gain), safe_mean(same_target_gain))
        donor_specificity = round(
            positive_gain - max(safe_mean(wrong_donor_gain), safe_mean(unrelated_gain)), 6
        )
        donor_layer_specificity = round(safe_mean(correct_gain) - safe_mean(wrong_layer_gain), 6)
        transplant_consistency = round(
            sum(value > 0 for value in correct_gain) / max(1, len(correct_gain)), 6
        )
        transplant_pass = (
            len(correct_gain) == 36 and positive_gain > 0.0 and donor_specificity > 0.0
            and donor_layer_specificity > 0.0 and transplant_consistency >= 0.65
        )

        generation = [row for row in generation_rows if row["mechanism_id"] == mechanism]

        def generation_rate(condition: str, field: str) -> float:
            values = [bool(row[field]) for row in generation if row["condition"] == condition]
            return round(sum(values) / max(1, len(values)), 6)

        correct_generation = generation_rate("correct_baseline", "first_word_target_match")
        zero_generation = generation_rate("correct_joint_zero", "first_word_target_match")
        recipient_generation = generation_rate("recipient_baseline", "first_word_target_match")
        transplant_generation = generation_rate(
            "recipient_correct_donor_transplant", "first_word_target_match"
        )
        protocol_rate = generation_rate("correct_baseline", "one_word_protocol_compliant")
        generation_pass = (
            correct_generation >= 0.60 and correct_generation > zero_generation
            and transplant_generation > recipient_generation
        )
        full_chain = natural_gate_pass and position_pass and transplant_pass and generation_pass
        audits.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "registered_prompt_count": len(correct),
            "registered_independent_object_count": len({row["base_case_id"] for row in correct}),
            "baseline_accuracy": baseline_accuracy,
            "same_target_carrier_cosine": same_carrier_cos,
            "wrong_target_carrier_cosine": wrong_carrier_cos,
            "natural_carrier_identity_specificity": natural_specificity,
            "natural_residual_identity_specificity": residual_specificity,
            "natural_gate_observational_pass": natural_gate_pass,
            "position_eligible_count": len(eligible),
            "joint_margin_drop": safe_mean(joint_margin),
            "joint_phrase_logprob_drop": joint_phrase_mean,
            "random_phrase_logprob_drop": safe_mean(random_phrase),
            "wrong_layer_phrase_logprob_drop": safe_mean(wrong_layer_phrase),
            "position_specificity": position_specificity,
            "position_positive_consistency": position_consistency,
            "role_phrase_logprob_drops": role_phrase,
            "position_necessity_pass": position_pass,
            "correct_donor_phrase_gain": safe_mean(correct_gain),
            "same_target_donor_phrase_gain": safe_mean(same_target_gain),
            "wrong_donor_phrase_gain": safe_mean(wrong_donor_gain),
            "unrelated_donor_phrase_gain": safe_mean(unrelated_gain),
            "wrong_layer_donor_phrase_gain": safe_mean(wrong_layer_gain),
            "natural_donor_specificity": donor_specificity,
            "natural_donor_layer_specificity": donor_layer_specificity,
            "natural_transplant_consistency": transplant_consistency,
            "natural_state_transplant_pass": transplant_pass,
            "correct_generation_rate": correct_generation,
            "joint_zero_generation_rate": zero_generation,
            "recipient_generation_rate": recipient_generation,
            "recipient_transplant_generation_rate": transplant_generation,
            "one_word_protocol_rate": protocol_rate,
            "complete_generation_pass": generation_pass,
            "full_chain_pass": full_chain,
            "l5_promoted": full_chain,
            "single_unit_causal": False,
            "evidence_boundary": (
                "distributed carrier-set chain candidate; no single-unit causality"
                if full_chain else
                "one or more natural gate, position, transplant, or generation criteria failed"
            ),
        })
    return audits


def run_model(model_key: str, round_name: str, max_cases: int = 0) -> dict[str, Any]:
    output = OUT / round_name
    cases = case_bank.build_cases()
    if max_cases:
        cases = cases[:max_cases]
    selections = read_jsonl(PHASE326 / f"phase326_{model_key}_carrier_sets.jsonl")
    if not selections:
        raise FileNotFoundError(f"Missing frozen Phase326 selections for {model_key}")
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        natural_rows, delta_rows = batch_a_natural_identity(loaded, cases, selections)
        write_jsonl(output / f"phase327_{model_key}_natural_identity_rows.jsonl", natural_rows)
        write_jsonl(output / f"phase327_{model_key}_residual_path_rows.jsonl", delta_rows)
        position_rows = batch_b_position_necessity(loaded, cases, selections)
        write_jsonl(output / f"phase327_{model_key}_position_necessity_rows.jsonl", position_rows)
        transplant_rows, generation_states = batch_c_natural_transplant(loaded, cases, selections)
        write_jsonl(output / f"phase327_{model_key}_natural_transplant_rows.jsonl", transplant_rows)
        generation_rows = batch_d_generation(loaded, cases, selections, generation_states)
        write_jsonl(output / f"phase327_{model_key}_generation_rows.jsonl", generation_rows)
        audits = mechanism_audits(
            model_key, natural_rows, delta_rows, position_rows, transplant_rows, generation_rows
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model_key,
            "model_name_or_path": str(loaded.spec.local_dir),
            "model_revision": "local_unknown",
            "registered_prompt_count": len(cases),
            "registered_independent_object_count": len({case["base_case_id"] for case in cases}),
            "frozen_carrier_member_count": len(selections),
            "natural_variant_row_count": len(natural_rows),
            "residual_path_row_count": len(delta_rows),
            "position_intervention_row_count": len(position_rows),
            "natural_transplant_row_count": len(transplant_rows),
            "generation_row_count": len(generation_rows),
            "mechanism_audits": audits,
            "natural_gate_pass_count": sum(row["natural_gate_observational_pass"] for row in audits),
            "position_necessity_pass_count": sum(row["position_necessity_pass"] for row in audits),
            "natural_state_transplant_pass_count": sum(row["natural_state_transplant_pass"] for row in audits),
            "complete_generation_pass_count": sum(row["complete_generation_pass"] for row in audits),
            "full_chain_pass_count": sum(row["full_chain_pass"] for row in audits),
            "single_unit_causal_count": 0,
        }
        write_jsonl(output / f"phase327_{model_key}_mechanism_audits.jsonl", audits)
        write_json(output / f"phase327_{model_key}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_cd_chunk(model_key: str, round_name: str, case_start: int, case_end: int) -> dict[str, int]:
    """Run the transplant/generation batches in a restartable CUDA process."""
    output = OUT / round_name
    cases = case_bank.build_cases()[case_start:case_end]
    selections = read_jsonl(PHASE326 / f"phase326_{model_key}_carrier_sets.jsonl")
    if not cases or not selections:
        raise ValueError("C/D chunk requires nonempty registered cases and Phase326 selections")
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        transplant_rows, generation_states = batch_c_natural_transplant(loaded, cases, selections)
        generation_rows = batch_d_generation(loaded, cases, selections, generation_states)
        suffix = f"part_{case_start:03d}_{case_end:03d}"
        write_jsonl(
            output / f"phase327_{model_key}_natural_transplant_rows_{suffix}.jsonl",
            transplant_rows,
        )
        write_jsonl(
            output / f"phase327_{model_key}_generation_rows_{suffix}.jsonl",
            generation_rows,
        )
        result = {
            "case_count": len(cases),
            "transplant_row_count": len(transplant_rows),
            "generation_row_count": len(generation_rows),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_ab_model(model_key: str, round_name: str) -> dict[str, int]:
    """Run natural observation and position necessity without C/D state retention."""
    output = OUT / round_name
    cases = case_bank.build_cases()
    selections = read_jsonl(PHASE326 / f"phase326_{model_key}_carrier_sets.jsonl")
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        natural_rows, delta_rows = batch_a_natural_identity(loaded, cases, selections)
        write_jsonl(output / f"phase327_{model_key}_natural_identity_rows.jsonl", natural_rows)
        write_jsonl(output / f"phase327_{model_key}_residual_path_rows.jsonl", delta_rows)
        position_rows = batch_b_position_necessity(loaded, cases, selections)
        write_jsonl(output / f"phase327_{model_key}_position_necessity_rows.jsonl", position_rows)
        result = {
            "natural_row_count": len(natural_rows),
            "residual_row_count": len(delta_rows),
            "position_row_count": len(position_rows),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def finalize_chunked_model(model_key: str, round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    natural_rows = read_jsonl(output / f"phase327_{model_key}_natural_identity_rows.jsonl")
    delta_rows = read_jsonl(output / f"phase327_{model_key}_residual_path_rows.jsonl")
    position_rows = read_jsonl(output / f"phase327_{model_key}_position_necessity_rows.jsonl")
    part_transplant_rows = [
        row
        for path in sorted(output.glob(f"phase327_{model_key}_natural_transplant_rows_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    part_generation_rows = [
        row
        for path in sorted(output.glob(f"phase327_{model_key}_generation_rows_part_*.jsonl"))
        for row in read_jsonl(path)
    ]
    transplant_rows = part_transplant_rows or read_jsonl(
        output / f"phase327_{model_key}_natural_transplant_rows.jsonl"
    )
    generation_rows = part_generation_rows or read_jsonl(
        output / f"phase327_{model_key}_generation_rows.jsonl"
    )
    expected = {
        "natural": 540,
        "position": 972,
        "transplant": 648,
        "generation": 216,
    }
    actual = {
        "natural": len(natural_rows),
        "position": len(position_rows),
        "transplant": len(transplant_rows),
        "generation": len(generation_rows),
    }
    if actual != expected:
        raise ValueError(f"Incomplete chunked model rows: expected={expected}, actual={actual}")
    write_jsonl(output / f"phase327_{model_key}_natural_transplant_rows.jsonl", transplant_rows)
    write_jsonl(output / f"phase327_{model_key}_generation_rows.jsonl", generation_rows)
    audits = mechanism_audits(
        model_key, natural_rows, delta_rows, position_rows, transplant_rows, generation_rows
    )
    phase326_summary = read_json(PHASE326 / f"phase326_{model_key}_summary.json")
    selections = read_jsonl(PHASE326 / f"phase326_{model_key}_carrier_sets.jsonl")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": model_key,
        "model_name_or_path": phase326_summary["model_name_or_path"],
        "model_revision": phase326_summary.get("model_revision", "local_unknown"),
        "registered_prompt_count": 108,
        "registered_independent_object_count": 54,
        "frozen_carrier_member_count": len(selections),
        "natural_variant_row_count": len(natural_rows),
        "residual_path_row_count": len(delta_rows),
        "position_intervention_row_count": len(position_rows),
        "natural_transplant_row_count": len(transplant_rows),
        "generation_row_count": len(generation_rows),
        "mechanism_audits": audits,
        "natural_gate_pass_count": sum(row["natural_gate_observational_pass"] for row in audits),
        "position_necessity_pass_count": sum(row["position_necessity_pass"] for row in audits),
        "natural_state_transplant_pass_count": sum(row["natural_state_transplant_pass"] for row in audits),
        "complete_generation_pass_count": sum(row["complete_generation_pass"] for row in audits),
        "full_chain_pass_count": sum(row["full_chain_pass"] for row in audits),
        "single_unit_causal_count": 0,
        "execution_note": (
            "C/D used restartable 36-prompt CUDA chunks after a libcudart process fault; rows were count-validated before merge."
            if part_transplant_rows else
            "Summary recomputed from count-validated canonical raw rows without rerunning the model."
        ),
    }
    write_jsonl(output / f"phase327_{model_key}_mechanism_audits.jsonl", audits)
    write_json(output / f"phase327_{model_key}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def residual_layer_summaries(delta_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], dict[str, list[float]]] = defaultdict(
        lambda: {"same": [], "wrong": []}
    )
    for row in delta_rows:
        key = (row["model"], row["mechanism_id"], int(row["layer"]), row["position_role"])
        bucket = "same" if row["comparison_variant"] == "same_target_object" else "wrong"
        grouped[key][bucket].append(float(row["residual_rms_delta"]))
    result = []
    for (model, mechanism, layer, role), values in sorted(grouped.items()):
        same = safe_mean(values["same"])
        wrong = safe_mean(values["wrong"])
        result.append({
            "schema_version": "residual_identity_layer_summary.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "layer": layer,
            "position_role": role,
            "same_target_rms_delta": same,
            "wrong_target_rms_delta": wrong,
            "target_identity_specificity": round(wrong - same, 6),
            "same_target_observation_count": len(values["same"]),
            "wrong_target_observation_count": len(values["wrong"]),
            "causal": False,
        })
    return result


def atlas_paths(
    cross_model: dict[str, Any],
    all_audits: list[dict[str, Any]],
    layer_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for audit in all_audits:
        mechanism = audit["mechanism_id"]
        model = audit["model"]
        selections = read_jsonl(PHASE326 / f"phase326_{model}_carrier_sets.jsonl")
        members = [
            {
                "component_type": row["component_type"],
                "layer": row["component_layer"],
                "position_role": row["position_role"],
                "component_index": row["component_index"],
                "component_start": row["component_start"],
                "component_end": row["component_end"],
            }
            for row in selections if row["mechanism_id"] == mechanism
        ]
        layer_rows = [
            row for row in layer_summaries
            if row["model"] == model and row["mechanism_id"] == mechanism
        ]
        max_layer = max((int(row["layer"]) for row in layer_rows), default=0)
        early_mid_limit = round(max_layer * 0.70)
        residual_peaks = []
        for role in ROLES:
            candidates = [
                row for row in layer_rows
                if row["position_role"] == role and int(row["layer"]) <= early_mid_limit
            ]
            if candidates:
                residual_peaks.append(max(candidates, key=lambda row: row["target_identity_specificity"]))
        rows.append({
            "schema_version": "pattern_family_physical_path.v1",
            "phase_id": PHASE,
            "created_at": now(),
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "model": model,
            "path_id": f"content_knowledge:{mechanism}:{model}:natural_retrieval",
            "stages": [
                {"stage": "natural_object_source", "evidence_level": "L2"},
                {"stage": "early_mid_residual_identity", "evidence_level": "L2"},
                {"stage": "frozen_late_carrier_set", "evidence_level": "L3+L4"},
                {"stage": "target_phrase_readout", "evidence_level": "L2"},
                {"stage": "greedy_generation", "evidence_level": "L2"},
            ],
            "carrier_members": members,
            "early_mid_residual_identity_peaks": residual_peaks,
            "early_mid_residual_identity_peak_rule": (
                "highest target-identity specificity within the first 70% of layers; boundary maxima are not local peaks"
            ),
            "natural_gate_observational_pass": audit["natural_gate_observational_pass"],
            "position_necessity_pass": audit["position_necessity_pass"],
            "natural_state_transplant_pass": audit["natural_state_transplant_pass"],
            "complete_generation_pass": audit["complete_generation_pass"],
            "full_chain_pass": audit["full_chain_pass"],
            "causal": False,
            "causal_boundary": "No upstream intervention was shown to alter the downstream frozen set; ordered stages remain noncausal.",
            "single_unit_causal": False,
            "source_artifacts": [
                f"tests/gpt5/result/phase327_natural_retrieval_path/{ROUND_DEFAULT}/phase327_{model}_mechanism_audits.jsonl",
                "tests/gpt5/result/phase326_distributed_carrier_atlas/distributed_carrier_atlas/phase326_carrier_sets.jsonl",
            ],
        })
    return rows


def collect(round_name: str) -> dict[str, Any]:
    output = OUT / round_name
    summaries = [read_json(output / f"phase327_{model}_summary.json") for model in MODELS]
    all_audits = [
        row for model in MODELS
        for row in read_jsonl(output / f"phase327_{model}_mechanism_audits.jsonl")
    ]
    mechanism_results = []
    for mechanism in case_bank.OBJECTS:
        rows = [row for row in all_audits if row["mechanism_id"] == mechanism]
        full_models = [row["model"] for row in rows if row["full_chain_pass"]]
        mechanism_results.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "family_id": "content_knowledge",
            "mechanism_id": mechanism,
            "models_tested": list(MODELS),
            "natural_gate_pass_models": [row["model"] for row in rows if row["natural_gate_observational_pass"]],
            "position_necessity_pass_models": [row["model"] for row in rows if row["position_necessity_pass"]],
            "natural_state_transplant_pass_models": [row["model"] for row in rows if row["natural_state_transplant_pass"]],
            "complete_generation_pass_models": [row["model"] for row in rows if row["complete_generation_pass"]],
            "full_chain_pass_models": full_models,
            "cross_model_full_chain_replicated": len(full_models) >= 2,
            "l5_promoted": len(full_models) >= 2,
            "single_unit_causal": False,
        })
    cross_model = {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "models": list(MODELS),
        "registered_prompt_cases": sum(row["registered_prompt_count"] for row in summaries),
        "registered_independent_model_objects": sum(row["registered_independent_object_count"] for row in summaries),
        "natural_variant_rows": sum(row["natural_variant_row_count"] for row in summaries),
        "position_intervention_rows": sum(row["position_intervention_row_count"] for row in summaries),
        "natural_transplant_rows": sum(row["natural_transplant_row_count"] for row in summaries),
        "generation_rows": sum(row["generation_row_count"] for row in summaries),
        "mechanism_results": mechanism_results,
        "cross_model_full_chain_count": sum(row["cross_model_full_chain_replicated"] for row in mechanism_results),
        "single_unit_causal_count": 0,
        "physical_candidate_family_count": 2,
        "strict_natural_chain_family_count": int(any(row["cross_model_full_chain_replicated"] for row in mechanism_results)),
        "small_model_scope_warning": "Results may differ materially in larger models; no architecture-invariant claim is made.",
    }
    merged_files = {
        "natural_identity_rows": "phase327_natural_identity_rows.jsonl",
        "residual_path_rows": "phase327_residual_path_rows.jsonl",
        "position_necessity_rows": "phase327_position_necessity_rows.jsonl",
        "natural_transplant_rows": "phase327_natural_transplant_rows.jsonl",
        "generation_rows": "phase327_generation_rows.jsonl",
        "mechanism_audits": "phase327_mechanism_audits.jsonl",
    }
    for key, filename in merged_files.items():
        source_name = key
        merged = []
        for model in MODELS:
            model_filename = f"phase327_{model}_{source_name}.jsonl"
            merged.extend(read_jsonl(output / model_filename))
        write_jsonl(output / filename, merged)
    all_delta_rows = read_jsonl(output / "phase327_residual_path_rows.jsonl")
    layer_summaries = residual_layer_summaries(all_delta_rows)
    write_jsonl(output / "phase327_residual_layer_summaries.jsonl", layer_summaries)
    write_json(output / "phase327_cross_model_summary.json", cross_model)
    write_jsonl(
        output / "phase327_atlas_paths.jsonl",
        atlas_paths(cross_model, all_audits, layer_summaries),
    )
    print(json.dumps(cross_model, ensure_ascii=False, indent=2))
    return cross_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--stage", choices=("full", "ab", "cd-chunk", "finalize"), default="full")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--case-end", type=int, default=0)
    args = parser.parse_args()
    if args.collect:
        collect(args.round)
        return
    if not args.model:
        parser.error("--model is required unless --collect is used")
    if args.stage == "ab":
        run_ab_model(args.model, args.round)
        return
    if args.stage == "cd-chunk":
        if args.case_end <= args.case_start:
            parser.error("--case-end must be greater than --case-start for cd-chunk")
        run_cd_chunk(args.model, args.round, args.case_start, args.case_end)
        return
    if args.stage == "finalize":
        finalize_chunked_model(args.model, args.round)
        return
    run_model(args.model, args.round, args.max_cases)


if __name__ == "__main__":
    main()
