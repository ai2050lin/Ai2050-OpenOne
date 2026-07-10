#!/usr/bin/env python3
"""Registered heldout causal audit for the frozen Phase330 carrier sets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase330_nine_family_case_bank import FAMILY_MECHANISMS, MODELS  # noqa: E402
import phase330_global_atlas_survey as survey  # noqa: E402
import phase326_distributed_carrier_atlas as phase326  # noqa: E402
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402


PHASE = "Phase330"
SCHEMA_VERSION = "8.0.0"
ROUND_DEFAULT = "nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas"
CONDITIONS = (
    "baseline",
    "matched_natural_state",
    "single_attention_zero",
    "attention_set_zero",
    "single_mlp_zero",
    "mlp_set_zero",
    "joint_set_zero",
    "matched_random_joint_zero",
    "wrong_layer_joint_zero",
    "wrong_donor_transplant",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def spec_key(spec: dict[str, Any]) -> str:
    return ":".join(str(spec[key]) for key in (
        "component_type", "component_layer", "position_role", "component_start", "component_end",
    ))


def module_for(model_obj: Any, spec: dict[str, Any]) -> Any:
    layer = int(spec["component_layer"])
    if spec["component_type"] == "attention_head_input":
        return head_meta(model_obj, layer)[0]
    module = phase326.get_down_proj(get_layers(model_obj)[layer])
    if module is None:
        raise TypeError(f"No MLP down projection at layer {layer}")
    return module


def register_cases(round_name: str) -> list[dict[str, Any]]:
    root = OUT / round_name
    cases = read_jsonl(root / "phase330_case_bank.jsonl")
    by_key = {
        (row["family_id"], row["mechanism_id"], row["item_index"], row["template_id"]): row
        for row in cases
    }
    result = []
    for model in MODELS:
        for family, mechanisms in FAMILY_MECHANISMS.items():
            for mechanism in mechanisms:
                # Use separated heldout endpoints.  Adjacent 18/19 concentrate
                # wrapper families on the same 7/moon target pair.
                for item_index in (18, 23):
                    recipient = by_key[(family, mechanism, item_index, "template_c")]
                    correct_donor = by_key[(family, mechanism, item_index, "template_a")]
                    wrong_donor = by_key[(family, mechanism, 20 if item_index == 18 else 22, "template_a")]
                    result.append({
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "causal_case_id": f"phase330_causal_{model}_{family}_{mechanism}_{item_index:02d}",
                        "model": model,
                        "family_id": family,
                        "mechanism_id": mechanism,
                        "recipient_case_id": recipient["case_id"],
                        "correct_donor_case_id": correct_donor["case_id"],
                        "wrong_donor_case_id": wrong_donor["case_id"],
                        "item_index": item_index,
                        "split": "heldout",
                        "recipient_template": "template_c",
                        "donor_template": "template_a",
                        "condition_count": len(CONDITIONS),
                        "registration_rule": "separated_heldout_endpoints_18_23_after_denominator_concentration_audit",
                        "selection_updates_allowed": False,
                    })
    if len(result) != 432:
        raise RuntimeError(f"Expected 432 registered cases, got {len(result)}")
    write_jsonl(root / "registered_causal_cases.jsonl", result)
    write_json(root / "registered_causal_protocol.json", {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "registered_case_count": len(result),
        "registered_condition_count": len(CONDITIONS),
        "registered_row_count": len(result) * len(CONDITIONS),
        "conditions": list(CONDITIONS),
        "heldout_item_indices": [18, 23],
        "supersedes_invalid_registration": "adjacent_indices_18_19_target_concentration",
        "selection_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
    })
    return result


def case_lookup(round_name: str) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in read_jsonl(OUT / round_name / "phase330_case_bank.jsonl")}


@torch.inference_mode()
def capture_values(loaded: Any, case: dict[str, Any], specs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    tokenizer = loaded.tokenizer
    encoded = tokenizer(case["prompt"], return_tensors="pt", truncation=True, max_length=128)
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    seq_len = int(encoded["attention_mask"].sum().item())
    spans = survey.role_spans(tokenizer, case, seq_len)
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

        def pre_hook(_module: Any, inputs: tuple[Any, ...], selected: list[dict[str, Any]] = selected) -> None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return
            tensor = inputs[0]
            for spec in selected:
                start_pos, end_pos = spans[spec["position_role"]]
                start = int(spec["component_start"])
                end = int(spec["component_end"])
                values[spec_key(spec)] = tensor[0, start_pos : end_pos + 1, start:end].detach().mean(dim=0).clone()

        handles.append(module.register_forward_pre_hook(pre_hook))
    try:
        loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return values


def install_hooks(
    loaded: Any,
    case: dict[str, Any],
    zero_specs: list[dict[str, Any]],
    transplant_specs: list[dict[str, Any]],
    transplant_values: dict[str, torch.Tensor],
) -> tuple[list[Any], dict[str, torch.Tensor]]:
    tokenizer = loaded.tokenizer
    prompt = tokenizer(case["prompt"], return_tensors="pt", truncation=True, max_length=128)
    prompt = {key: value.to(loaded.input_device) for key, value in prompt.items()}
    seq_len = int(prompt["attention_mask"].sum().item())
    spans = survey.role_spans(tokenizer, case, seq_len)
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

        def pre_hook(
            _module: Any, inputs: tuple[Any, ...], zero: list[dict[str, Any]] = zero,
            transplant: list[dict[str, Any]] = transplant,
        ) -> tuple[Any, ...] | None:
            if not inputs or not torch.is_tensor(inputs[0]):
                return None
            tensor = inputs[0]
            changed = tensor.clone()
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
                value = transplant_values[spec_key(spec)].to(changed.device, changed.dtype)
                changed[0, positions, int(spec["component_start"]):int(spec["component_end"])] = value
            return (changed, *inputs[1:])

        handles.append(module.register_forward_pre_hook(pre_hook))
    return handles, prompt


@torch.inference_mode()
def run_condition(
    loaded: Any,
    case: dict[str, Any],
    condition: str,
    zero_specs: list[dict[str, Any]],
    transplant_specs: list[dict[str, Any]],
    transplant_values: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> dict[str, Any]:
    handles, encoded = install_hooks(loaded, case, zero_specs, transplant_specs, transplant_values)
    try:
        output = loaded.model(**encoded, use_cache=False, return_dict=True)
        seq_len = int(encoded["attention_mask"].sum().item())
        logits = output.logits[0, seq_len - 1].detach().float()
        target_id = survey.target_ids(loaded.tokenizer, case["target"])[0]
        distractor_ids = [survey.target_ids(loaded.tokenizer, value)[0] for value in case["distractors"]]
        target_logit = float(logits[target_id].item())
        best_distractor = max(float(logits[token_id].item()) for token_id in distractor_ids)
        rank = 1 + int((logits > target_logit).sum().item())
        generated = loaded.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            pad_token_id=loaded.tokenizer.pad_token_id,
            eos_token_id=loaded.tokenizer.eos_token_id,
        )
        suffix = generated[0, encoded["input_ids"].shape[1] :]
        ids = [int(value) for value in suffix.tolist()]
        text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
        normalized = survey.normalize_text(text)
        aliases = [survey.normalize_text(value) for value in case["target_aliases"]]
        return {
            "condition": condition,
            "target_logit": round(target_logit, 7),
            "best_distractor_logit": round(best_distractor, 7),
            "target_margin": round(target_logit - best_distractor, 7),
            "target_full_vocabulary_rank": rank,
            "target_top1": rank == 1,
            "candidate_winner_is_target": target_logit >= best_distractor,
            "generated_text": text,
            "generated_token_ids": json.dumps(ids),
            "target_match": any(alias and (normalized.startswith(alias) or alias in normalized) for alias in aliases),
            "protocol_success": survey.protocol_ok(case, text),
            "behavior_success": any(alias and (normalized.startswith(alias) or alias in normalized) for alias in aliases) and survey.protocol_ok(case, text),
            "eos_emitted": loaded.tokenizer.eos_token_id in ids,
            "zero_component_count": len(zero_specs),
            "transplant_component_count": len(transplant_specs),
        }
    finally:
        for handle in handles:
            handle.remove()


def condition_plan(
    model_obj: Any,
    specs: list[dict[str, Any]],
    correct_values: dict[str, torch.Tensor],
    wrong_values: dict[str, torch.Tensor],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, torch.Tensor]]]:
    attention = [row for row in specs if row["component_type"] == "attention_head_input"]
    mlp = [row for row in specs if row["component_type"] == "mlp_product_group"]
    joint = [*attention, *mlp]
    return {
        "baseline": ([], [], {}),
        "matched_natural_state": ([], joint, correct_values),
        "single_attention_zero": (attention[:1], [], {}),
        "attention_set_zero": (attention, [], {}),
        "single_mlp_zero": (mlp[:1], [], {}),
        "mlp_set_zero": (mlp, [], {}),
        "joint_set_zero": (joint, [], {}),
        "matched_random_joint_zero": (phase326.randomize_specs(model_obj, joint), [], {}),
        "wrong_layer_joint_zero": (phase326.wrong_layer_specs(model_obj, joint), [], {}),
        "wrong_donor_transplant": ([], joint, wrong_values),
    }


def run_model(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    output_dir = root / "causal_audit_balanced" / model
    if (output_dir / "complete.json").exists():
        return json.loads((output_dir / "complete.json").read_text(encoding="utf-8"))
    registry = [row for row in read_jsonl(root / "registered_causal_cases.jsonl") if row["model"] == model]
    cases = case_lookup(round_name)
    carrier_rows = [row for row in read_jsonl(root / "carrier_sets.jsonl") if row["model"] == model]
    loaded = None
    rows = []
    try:
        loaded = load_probe_model(model)
        for case_index, registered in enumerate(registry, 1):
            recipient = cases[registered["recipient_case_id"]]
            correct_donor = cases[registered["correct_donor_case_id"]]
            wrong_donor = cases[registered["wrong_donor_case_id"]]
            specs = [
                row for row in carrier_rows
                if row["family_id"] == registered["family_id"] and row["mechanism_id"] == registered["mechanism_id"]
            ]
            correct_values = capture_values(loaded, correct_donor, specs)
            wrong_values = capture_values(loaded, wrong_donor, specs)
            plan = condition_plan(loaded.model, specs, correct_values, wrong_values)
            case_rows = []
            for condition in CONDITIONS:
                zero, transplant, values = plan[condition]
                outcome = run_condition(loaded, recipient, condition, zero, transplant, values, max_new_tokens)
                case_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "phase_id": PHASE,
                    "created_at": now(),
                    **{key: registered[key] for key in (
                        "causal_case_id", "model", "family_id", "mechanism_id", "recipient_case_id",
                        "correct_donor_case_id", "wrong_donor_case_id", "item_index", "split",
                    )},
                    "target": recipient["target"],
                    "target_bucket": recipient["target_bucket"],
                    "target_absent_from_prompt": recipient["target_absent_from_prompt"],
                    **outcome,
                    "selection_updates_allowed": False,
                    "single_unit_causal": False,
                })
            baseline = next(row for row in case_rows if row["condition"] == "baseline")
            for row in case_rows:
                row["delta_target_margin_vs_baseline"] = round(row["target_margin"] - baseline["target_margin"], 7)
                row["target_rank_change_vs_baseline"] = int(baseline["target_full_vocabulary_rank"] - row["target_full_vocabulary_rank"])
                row["behavior_changed_vs_baseline"] = row["behavior_success"] != baseline["behavior_success"]
                row["evidence_level"] = "L4_registered_heldout_set_intervention"
            rows.extend(case_rows)
            if case_index % 16 == 0:
                print(json.dumps({"quality_only": True, "model": model, "completed_cases": case_index, "total_cases": len(registry)}), flush=True)
        expected = len(registry) * len(CONDITIONS)
        if len(rows) != expected:
            raise RuntimeError(f"Expected {expected} causal rows, got {len(rows)}")
        write_parquet(output_dir / "causal_rows.parquet", rows)
        write_jsonl(output_dir / "causal_rows.jsonl", rows)
        quality = {
            "phase_id": PHASE,
            "created_at": now(),
            "quality_only": True,
            "scientific_analysis_permitted": False,
            "model": model,
            "registered_case_count": len(registry),
            "condition_count": len(CONDITIONS),
            "causal_row_count": len(rows),
            "expected_causal_row_count": 1440,
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(registry) == 144 and len(rows) == 1440,
        }
        write_json(output_dir / "complete.json", quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def collect(round_name: str) -> dict[str, Any]:
    root = OUT / round_name
    all_rows = []
    quality_rows = []
    for model in MODELS:
        all_rows.extend(read_jsonl(root / "causal_audit_balanced" / model / "causal_rows.jsonl"))
        quality_rows.append(json.loads((root / "causal_audit_balanced" / model / "complete.json").read_text(encoding="utf-8")))
    write_parquet(root / "causal_rows.parquet", all_rows)
    write_jsonl(root / "causal_rows.jsonl", all_rows)
    quality = {
        "phase_id": PHASE,
        "created_at": now(),
        "model_count": len(quality_rows),
        "registered_case_count": sum(row["registered_case_count"] for row in quality_rows),
        "causal_row_count": len(all_rows),
        "expected_causal_row_count": 4320,
        "all_valid": all(row["valid"] for row in quality_rows),
        "selection_updates_allowed": False,
        "single_unit_intervention_gate_open": False,
    }
    quality["valid"] = quality["model_count"] == 3 and quality["causal_row_count"] == 4320 and quality["all_valid"]
    write_json(root / "causal_audit_quality.json", quality)
    return quality


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.register:
        print(json.dumps({"registered_cases": len(register_cases(args.round))}, indent=2))
    elif args.model:
        print(json.dumps(run_model(args.model, args.round, args.max_new_tokens), indent=2))
    elif args.collect:
        print(json.dumps(collect(args.round), indent=2))
    else:
        raise SystemExit("Use --register, --model MODEL, or --collect")


if __name__ == "__main__":
    main()
