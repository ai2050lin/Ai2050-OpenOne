#!/usr/bin/env python3
"""Map Phase334 natural receiver paths and freeze early/middle/late candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase331_refined_mechanism_audit import answer_segment, target_match  # noqa: E402
import phase330_global_atlas_survey as phase330_survey  # noqa: E402
from phase333_dynamic_survey import continuation_ids  # noqa: E402
from phase334_natural_necessity_case_bank import ROUND_DEFAULT  # noqa: E402


PHASE = "Phase334"
SCHEMA_VERSION = "12.0.0"
OUT = ROOT / "tests/gpt5/result/phase334_natural_necessity_atlas"
COMPONENTS = ("attention_output", "mlp_output", "residual_increment", "residual_output")
SELECTABLE_COMPONENTS = ("attention_output", "mlp_output", "residual_increment")
POSITION_ROLES = ("source", "query", "answer_start")

CONTRAST_SCHEMA = pa.schema([
    pa.field("schema_version", pa.string()), pa.field("phase_id", pa.string()),
    pa.field("created_at", pa.string()), pa.field("model", pa.string()),
    pa.field("case_id", pa.string()), pa.field("semantic_case_id", pa.string()),
    pa.field("family_id", pa.string()), pa.field("mechanism_id", pa.string()),
    pa.field("cohort", pa.string()), pa.field("paired_mechanism_id", pa.string()),
    pa.field("item_index", pa.int64()), pa.field("split", pa.string()),
    pa.field("template_id", pa.string()), pa.field("interface", pa.string()),
    pa.field("component_type", pa.string()), pa.field("component_layer", pa.int64()),
    pa.field("relative_depth", pa.float64()), pa.field("depth_bin", pa.string()),
    pa.field("position_role", pa.string()), pa.field("position_index", pa.int64()),
    pa.field("position_exact", pa.bool_()), pa.field("activation_norm", pa.float64()),
    pa.field("paired_activation_norm", pa.float64()), pa.field("contrast_norm", pa.float64()),
    pa.field("relative_contrast", pa.float64()),
    pa.field("target_projection", pa.float64()),
    pa.field("selection_metric", pa.string()), pa.field("evidence_level", pa.string()),
])


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported output type {type(output).__name__}")


def encoded_prompt(loaded: Any, case: dict[str, Any]) -> dict[str, torch.Tensor]:
    encoded = loaded.tokenizer(
        case["prompt"], return_tensors="pt", truncation=True, max_length=256,
        add_special_tokens=bool(case["tokenization_add_special_tokens"]),
    )
    return {key: value.to(loaded.input_device) for key, value in encoded.items()}


def find_subsequence(sequence: list[int], pattern: list[int], last: bool) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    matches = [
        index for index in range(len(sequence) - len(pattern) + 1)
        if sequence[index : index + len(pattern)] == pattern
    ]
    if not matches:
        return -1
    start = matches[-1] if last else matches[0]
    return start + len(pattern) - 1


def fragment_position(tokenizer: Any, prompt_ids: list[int], fragment: str, last: bool) -> tuple[int, bool]:
    for candidate in (fragment, " " + fragment):
        pattern = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        position = find_subsequence(prompt_ids, [int(value) for value in pattern], last)
        if position >= 0:
            return position, True
    final_word = fragment.rstrip(" .?").split()[-1]
    pattern = tokenizer(" " + final_word, add_special_tokens=False)["input_ids"]
    position = find_subsequence(prompt_ids, [int(value) for value in pattern], last)
    if position >= 0:
        return position, False
    fallback = max(0, min(len(prompt_ids) - 1, len(prompt_ids) // (2 if last else 3)))
    return fallback, False


def role_positions(loaded: Any, case: dict[str, Any], prompt_ids: list[int]) -> dict[str, tuple[int, bool]]:
    source = fragment_position(loaded.tokenizer, prompt_ids, case["source_fragment"], False)
    query = fragment_position(loaded.tokenizer, prompt_ids, case["query_fragment"], True)
    return {
        "source": source,
        "query": query,
        "answer_start": (len(prompt_ids) - 1, True),
    }


def depth_bin(layer: int, layer_count: int) -> str:
    relative = layer / max(1, layer_count - 1)
    if relative < 1 / 3:
        return "early"
    if relative < 2 / 3:
        return "middle"
    return "late"


def target_direction(loaded: Any, case: dict[str, Any]) -> torch.Tensor:
    target_id = continuation_ids(loaded, case, case["target"])[0]
    distractors = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
    weight = loaded.model.get_output_embeddings().weight.detach().float()
    direction = weight[target_id] - weight[distractors].mean(dim=0)
    return direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)


@torch.inference_mode()
def capture_natural(loaded: Any, case: dict[str, Any]) -> tuple[dict[tuple[str, int, str], torch.Tensor], dict[str, tuple[int, bool]]]:
    encoded = encoded_prompt(loaded, case)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    positions = role_positions(loaded, case, prompt_ids)
    layers = get_layers(loaded.model)
    captures: dict[tuple[str, int, str], torch.Tensor] = {}
    residual_inputs: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        def layer_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            residual_inputs[idx] = inputs[0].detach()

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            tensor = component_tensor(output)
            for role, (position, _exact) in positions.items():
                captures[("attention_output", idx, role)] = tensor[0, position].detach().float().cpu()

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            tensor = component_tensor(output)
            for role, (position, _exact) in positions.items():
                captures[("mlp_output", idx, role)] = tensor[0, position].detach().float().cpu()

        def layer_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            tensor = component_tensor(output)
            before = residual_inputs[idx]
            for role, (position, _exact) in positions.items():
                captures[("residual_output", idx, role)] = tensor[0, position].detach().float().cpu()
                captures[("residual_increment", idx, role)] = (
                    tensor[0, position] - before[0, position]
                ).detach().float().cpu()

        handles.extend([
            layer.register_forward_pre_hook(layer_pre),
            layer.self_attn.register_forward_hook(attention_post),
            layer.mlp.register_forward_hook(mlp_post),
            layer.register_forward_hook(layer_post),
        ])
    try:
        loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return captures, positions


@torch.inference_mode()
def phrase_logprob(loaded: Any, case: dict[str, Any]) -> float:
    encoded = encoded_prompt(loaded, case)
    target_ids = continuation_ids(loaded, case, case["target"])
    suffix = torch.tensor([target_ids], dtype=encoded["input_ids"].dtype, device=loaded.input_device)
    input_ids = torch.cat([encoded["input_ids"], suffix], dim=1)
    attention_mask = torch.ones_like(input_ids)
    output = loaded.model(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True,
    )
    prompt_length = int(encoded["input_ids"].shape[1])
    positions = torch.arange(prompt_length - 1, prompt_length - 1 + len(target_ids), device=loaded.input_device)
    logits = output.logits[0, positions].float()
    token_tensor = torch.tensor(target_ids, device=loaded.input_device)
    return float(torch.log_softmax(logits, dim=-1).gather(1, token_tensor[:, None]).sum().item())


@torch.inference_mode()
def baseline_metrics(loaded: Any, case: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    encoded = encoded_prompt(loaded, case)
    generated = loaded.model.generate(
        **encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        return_dict_in_generate=True, output_scores=True,
        pad_token_id=loaded.tokenizer.pad_token_id,
        eos_token_id=loaded.tokenizer.eos_token_id,
    )
    suffix = generated.sequences[0, encoded["input_ids"].shape[1] :]
    ids = [int(value) for value in suffix.tolist()]
    text = loaded.tokenizer.decode(ids, skip_special_tokens=True)
    segment = answer_segment(text)
    logits = generated.scores[0][0].detach().float()
    target_id = continuation_ids(loaded, case, case["target"])[0]
    distractor_ids = [continuation_ids(loaded, case, value)[0] for value in case["distractors"]]
    target_logit = float(logits[target_id].item())
    target_rank = 1 + int((logits > target_logit).sum().item())
    protocol = phase330_survey.protocol_ok(case, segment)
    matched = target_match(segment, case["target_aliases"])
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "created_at": now(),
        "model": case["model"],
        "case_id": case["case_id"],
        "semantic_case_id": case["semantic_case_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "cohort": case["cohort"],
        "paired_mechanism_id": case["paired_mechanism_id"],
        "item_index": case["item_index"],
        "split": case["split"],
        "template_id": case["template_id"],
        "interface": case["interface"],
        "target": case["target"],
        "target_first_token_id": target_id,
        "target_margin": round(target_logit - max(float(logits[idx].item()) for idx in distractor_ids), 7),
        "target_rank": target_rank,
        "target_phrase_logprob": round(phrase_logprob(loaded, case), 7),
        "generated_text": text,
        "generated_token_ids": json.dumps(ids),
        "generated_token_count": len(ids),
        "target_answer_segment_match": matched,
        "protocol_success_answer_segment": protocol,
        "behavior_success": matched and protocol,
        "eos_emitted": loaded.tokenizer.eos_token_id in ids,
        "evidence_level": "L2_natural_receiver_baseline",
        "single_unit_causal": False,
    }


def natural_rows_for_pair(
    loaded: Any,
    left: dict[str, Any], left_capture: dict[tuple[str, int, str], torch.Tensor],
    left_positions: dict[str, tuple[int, bool]],
    right: dict[str, Any], right_capture: dict[tuple[str, int, str], torch.Tensor],
    right_positions: dict[str, tuple[int, bool]],
) -> list[dict[str, Any]]:
    rows = []
    layer_count = len(get_layers(loaded.model))
    directions = {left["case_id"]: target_direction(loaded, left).cpu(), right["case_id"]: target_direction(loaded, right).cpu()}
    for case, capture, positions, paired_capture in (
        (left, left_capture, left_positions, right_capture),
        (right, right_capture, right_positions, left_capture),
    ):
        direction = directions[case["case_id"]]
        for layer in range(layer_count):
            for component in COMPONENTS:
                for role in POSITION_ROLES:
                    value = capture[(component, layer, role)]
                    paired = paired_capture[(component, layer, role)]
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "phase_id": PHASE,
                        "created_at": now(),
                        "model": case["model"],
                        "case_id": case["case_id"],
                        "semantic_case_id": case["semantic_case_id"],
                        "family_id": case["family_id"],
                        "mechanism_id": case["mechanism_id"],
                        "cohort": case["cohort"],
                        "paired_mechanism_id": case["paired_mechanism_id"],
                        "item_index": case["item_index"],
                        "split": case["split"],
                        "template_id": case["template_id"],
                        "interface": case["interface"],
                        "component_type": component,
                        "component_layer": layer,
                        "relative_depth": round(layer / max(1, layer_count - 1), 7),
                        "depth_bin": depth_bin(layer, layer_count),
                        "position_role": role,
                        "position_index": positions[role][0],
                        "position_exact": positions[role][1],
                        "activation_norm": round(float(torch.linalg.vector_norm(value).item()), 7),
                        "paired_activation_norm": round(float(torch.linalg.vector_norm(paired).item()), 7),
                        "contrast_norm": round(float(torch.linalg.vector_norm(value - paired).item()), 7),
                        "relative_contrast": round(float(
                            torch.linalg.vector_norm(value - paired).item()
                            / max(1e-8, 0.5 * (
                                torch.linalg.vector_norm(value).item()
                                + torch.linalg.vector_norm(paired).item()
                            ))
                        ), 7),
                        "target_projection": round(float(value @ direction), 7),
                        "selection_metric": "paired_natural_component_relative_contrast",
                        "evidence_level": "L2_natural_component_contrast",
                    })
    return rows


def freeze_discovery_candidates(model: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "discovery" or row["component_type"] not in SELECTABLE_COMPONENTS:
            continue
        key = (
            row["family_id"], row["mechanism_id"], row["interface"], row["depth_bin"],
            row["component_type"], row["component_layer"], row["position_role"],
        )
        grouped[key].append(row)
    candidates: dict[tuple[str, str, str, str], list[tuple[tuple[Any, ...], list[dict[str, Any]]]]] = defaultdict(list)
    for key, values in grouped.items():
        family, mechanism, interface, bin_name, _component, _layer, _role = key
        candidates[(family, mechanism, interface, bin_name)].append((key, values))
    plans = []
    for (family, mechanism, interface, bin_name), values in sorted(candidates.items()):
        eligible = [entry for entry in values if len(entry[1]) == 18]
        pool = eligible or values
        key, observations = max(
            pool,
            key=lambda entry: (
                median(
                    float(row.get("relative_contrast") or (
                        float(row["contrast_norm"])
                        / max(1e-8, 0.5 * (
                            float(row["activation_norm"]) + float(row["paired_activation_norm"])
                        ))
                    ))
                    for row in entry[1]
                ),
                -int(entry[0][5]),
            ),
        )
        _family, _mechanism, _interface, _bin, component, layer, role = key
        plans.append({
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "family_id": family,
            "mechanism_id": mechanism,
            "interface": interface,
            "depth_bin": bin_name,
            "selected_component": component,
            "selected_layer": int(layer),
            "selected_position_role": role,
            "discovery_observation_count": len(observations),
            "discovery_item_count": len({row["item_index"] for row in observations}),
            "median_contrast_norm": round(median(row["contrast_norm"] for row in observations), 7),
            "median_relative_contrast": round(median(
                float(row.get("relative_contrast") or (
                    float(row["contrast_norm"])
                    / max(1e-8, 0.5 * (
                        float(row["activation_norm"]) + float(row["paired_activation_norm"])
                    ))
                ))
                for row in observations
            ), 7),
            "position_exact_rate": round(sum(row["position_exact"] for row in observations) / len(observations), 7),
            "selection_metric": "maximum_median_paired_natural_component_relative_contrast",
            "selection_uses_target_output_direction": False,
            "selection_split": "discovery_only",
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
        })
    if len(plans) != 54:
        raise RuntimeError(f"Expected 54 discovery candidate plans for {model}, got {len(plans)}")
    return plans


def refreeze_existing(model: str, round_name: str) -> dict[str, Any]:
    model_dir = OUT / round_name / "survey" / model
    rows = pq.read_table(model_dir / "natural_contrast_rows.parquet").to_pylist()
    plans = freeze_discovery_candidates(model, rows)
    write_jsonl(model_dir / "discovery_candidate_plans.jsonl", plans)
    quality = read_json(model_dir / "complete.json")
    quality["candidate_selection_metric"] = "maximum_median_paired_natural_component_relative_contrast"
    quality["selection_scale_calibrated"] = True
    write_json(model_dir / "complete.json", quality)
    return {
        "phase_id": PHASE,
        "model": model,
        "discovery_candidate_plan_count": len(plans),
        "selection_metric": quality["candidate_selection_metric"],
        "valid": len(plans) == 54,
    }


def run_model(model: str, round_name: str, max_new_tokens: int) -> dict[str, Any]:
    root = OUT / round_name
    model_dir = root / "survey" / model
    complete_path = model_dir / "complete.json"
    if complete_path.exists():
        return read_json(complete_path)
    cases = [row for row in read_jsonl(root / "phase334_registered_cases.jsonl") if row["model"] == model]
    lookup = {
        (row["family_id"], row["mechanism_id"], row["item_index"], row["template_id"], row["interface"]): row
        for row in cases
    }
    primary = [row for row in cases if row["cohort"] == "primary"]
    loaded = None
    baseline_rows = []
    contrast_rows = []
    try:
        loaded = load_probe_model(model)
        for pair_index, left in enumerate(primary, 1):
            right = lookup[(
                left["family_id"], left["paired_mechanism_id"], left["item_index"],
                left["template_id"], left["interface"],
            )]
            left_capture, left_positions = capture_natural(loaded, left)
            right_capture, right_positions = capture_natural(loaded, right)
            contrast_rows.extend(natural_rows_for_pair(
                loaded, left, left_capture, left_positions, right, right_capture, right_positions
            ))
            baseline_rows.append(baseline_metrics(loaded, left, max_new_tokens))
            baseline_rows.append(baseline_metrics(loaded, right, max_new_tokens))
            del left_capture, right_capture
            if pair_index % 12 == 0:
                print(json.dumps({
                    "quality_only": True,
                    "model": model,
                    "natural_pairs": pair_index,
                    "total_pairs": len(primary),
                    "baseline_rows": len(baseline_rows),
                    "contrast_rows": len(contrast_rows),
                }), flush=True)
        plans = freeze_discovery_candidates(model, contrast_rows)
        write_jsonl(model_dir / "baseline_rows.jsonl", baseline_rows)
        write_jsonl(model_dir / "discovery_candidate_plans.jsonl", plans)
        pq.write_table(
            pa.Table.from_pylist(json_safe(contrast_rows), schema=CONTRAST_SCHEMA),
            model_dir / "natural_contrast_rows.parquet", compression="zstd", row_group_size=32768,
        )
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "model": model,
            "registered_case_count": len(cases),
            "natural_pair_count": len(primary),
            "baseline_row_count": len(baseline_rows),
            "natural_contrast_row_count": len(contrast_rows),
            "discovery_candidate_plan_count": len(plans),
            "selection_uses_target_output_direction": False,
            "selection_updates_allowed": False,
            "single_unit_intervention_gate_open": False,
            "valid": len(cases) == 648 and len(baseline_rows) == 648 and len(plans) == 54,
        }
        write_json(complete_path, quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--model", choices=("qwen3", "glm4", "deepseek7b"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--refreeze-candidates", action="store_true")
    args = parser.parse_args()
    result = (
        refreeze_existing(args.model, args.round)
        if args.refreeze_candidates
        else run_model(args.model, args.round, args.max_new_tokens)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
