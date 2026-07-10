#!/usr/bin/env python3
"""Run the frozen Phase330 behavior, readout, rollout, and layer-path census.

The scientific denominator is fixed by ``phase330_nine_family_case_bank``.
This runner writes one resumable model/family partition and intentionally does
not make scientific claims while partitions are incomplete.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase330_nine_family_case_bank import FAMILY_MECHANISMS, MODELS  # noqa: E402


PHASE = "Phase330"
SCHEMA_VERSION = "8.0.0"
ROUND_DEFAULT = "nine_family_global_atlas"
OUT = ROOT / "tests/gpt5/result/phase330_nine_family_global_atlas"
ROLES = ("source", "query", "last")
COMPONENTS = ("attention", "mlp", "residual")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd", row_group_size=32768)


def read_cases(round_name: str, family: str) -> list[dict[str, Any]]:
    path = OUT / round_name / "phase330_case_bank.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen case bank: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row["family_id"] == family]
    if len(selected) != 576:
        raise ValueError(f"Expected 576 cases for {family}, found {len(selected)}")
    return selected


def chunks(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def target_ids(tokenizer: Any, text: str) -> list[int]:
    for candidate in (" " + text, text):
        values = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        if values:
            return [int(value) for value in values]
    raise ValueError(f"Cannot tokenize answer {text!r}")


def locate_fragment_span(tokenizer: Any, prompt: str, fragment: str, seq_len: int) -> tuple[int, int]:
    char_start = prompt.index(fragment)
    char_end = char_start + len(fragment)
    prefix_ids = tokenizer(prompt[:char_start], add_special_tokens=True)["input_ids"]
    end_ids = tokenizer(prompt[:char_end], add_special_tokens=True)["input_ids"]
    start = min(seq_len - 1, max(0, len(prefix_ids) - 1))
    end = min(seq_len - 1, max(start, len(end_ids) - 1))
    return start, end


def role_spans(tokenizer: Any, case: dict[str, Any], seq_len: int) -> dict[str, tuple[int, int]]:
    prompt = case["prompt"]
    first = prompt.index(case["source_fragments"][0])
    last_fragment = case["source_fragments"][-1]
    source_text = prompt[first : prompt.index(last_fragment, first) + len(last_fragment)]
    return {
        "source": locate_fragment_span(tokenizer, prompt, source_text, seq_len),
        "query": locate_fragment_span(tokenizer, prompt, case["query_fragment"], seq_len),
        "last": (seq_len - 1, seq_len - 1),
    }


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().replace("<|im_end|>", "").split())


def protocol_ok(case: dict[str, Any], generated: str) -> bool:
    text = generated.strip()
    structure = case["expected_structure"]
    protocol = case["protocol"]
    if not text:
        return False
    if structure == "json" or protocol == "json":
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    if structure == "quoted" or protocol == "quote":
        return len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]
    if structure == "list" or protocol == "list":
        return text.startswith(("-", "*", "1."))
    if structure == "single_sentence" or protocol == "sentence":
        return text.count(".") + text.count("!") + text.count("?") <= 1
    if protocol in {"short", "answer_only", "no_explanation"}:
        return len(text.split()) <= max(8, len(case["target"].split()) + 4)
    return True


def output_row_base(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase_id": PHASE,
        "model": model,
        "case_id": case["case_id"],
        "item_id": case["item_id"],
        "family_id": case["family_id"],
        "mechanism_id": case["mechanism_id"],
        "split": case["split"],
        "template_id": case["template_id"],
        "language": case["language"],
        "target_bucket": case["target_bucket"],
        "target_absent_from_prompt": case["target_absent_from_prompt"],
        "selection_eligible": case["selection_eligible"],
    }


def component_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported component output {type(output).__name__}")


def build_role_masks(
    tokenizer: Any, cases: list[dict[str, Any]], lengths: list[int], width: int, device: torch.device
) -> tuple[dict[str, torch.Tensor], list[dict[str, list[int]]]]:
    masks = {role: torch.zeros((len(cases), width), device=device) for role in ROLES}
    span_rows: list[dict[str, list[int]]] = []
    for batch_index, (case, seq_len) in enumerate(zip(cases, lengths, strict=True)):
        spans = role_spans(tokenizer, case, seq_len)
        span_rows.append({role: [start, end] for role, (start, end) in spans.items()})
        for role, (start, end) in spans.items():
            masks[role][batch_index, start : end + 1] = 1.0
    return masks, span_rows


def make_direction_matrix(loaded: Any, cases: list[dict[str, Any]]) -> tuple[torch.Tensor, list[int], list[list[int]]]:
    first_targets: list[int] = []
    first_distractors: list[list[int]] = []
    weight = loaded.model.get_output_embeddings().weight.detach()
    vectors = []
    for case in cases:
        target_id = target_ids(loaded.tokenizer, case["target"])[0]
        distractor_ids = [target_ids(loaded.tokenizer, value)[0] for value in case["distractors"]]
        distractor_ids = list(dict.fromkeys(distractor_ids))
        direction = weight[target_id].float() - weight[distractor_ids].float().mean(dim=0)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-8)
        vectors.append(direction)
        first_targets.append(target_id)
        first_distractors.append(distractor_ids)
    return torch.stack(vectors).to(loaded.input_device), first_targets, first_distractors


@torch.inference_mode()
def trace_batch(
    loaded: Any,
    cases: list[dict[str, Any]],
    model: str,
    event_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        [case["prompt"] for case in cases], return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    lengths = [int(value) for value in encoded["attention_mask"].sum(dim=1).tolist()]
    role_masks, span_rows = build_role_masks(
        tokenizer, cases, lengths, int(encoded["input_ids"].shape[1]), loaded.input_device
    )
    directions, target_first, distractor_first = make_direction_matrix(loaded, cases)
    layers = get_layers(loaded.model)
    created_at = now()
    handles = []

    def record_component(tensor: torch.Tensor, layer_index: int, component: str) -> None:
        if tensor.ndim != 3:
            return
        tensor = tensor.detach().float()
        for role in ROLES:
            mask = role_masks[role][:, : tensor.shape[1]]
            pooled = torch.einsum("bth,bt->bh", tensor, mask)
            pooled = pooled / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            projection = (pooled * directions.to(pooled.device)).sum(dim=1)
            norm = torch.linalg.vector_norm(pooled, dim=1)
            cosine = projection / norm.clamp_min(1e-8)
            proj_values = projection.cpu().tolist()
            norm_values = norm.cpu().tolist()
            cos_values = cosine.cpu().tolist()
            for index, case in enumerate(cases):
                event_rows.append({
                    **output_row_base(case, model),
                    "created_at": created_at,
                    "component_type": component,
                    "layer": layer_index,
                    "position_role": role,
                    "projection": round(float(proj_values[index]), 7),
                    "activation_norm": round(float(norm_values[index]), 7),
                    "projection_cosine": round(float(cos_values[index]), 7),
                    "evidence_level": "L2_observational_path_event",
                    "single_unit_causal": False,
                })

    for layer_index, layer in enumerate(layers):
        def residual_pre(_module: Any, inputs: tuple[Any, ...], idx: int = layer_index) -> None:
            if inputs and torch.is_tensor(inputs[0]):
                record_component(inputs[0], idx, "residual")

        def attention_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            record_component(component_tensor(output), idx, "attention")

        def mlp_post(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_index) -> None:
            record_component(component_tensor(output), idx, "mlp")

        handles.append(layer.register_forward_pre_hook(residual_pre))
        handles.append(layer.self_attn.register_forward_hook(attention_post))
        handles.append(layer.mlp.register_forward_hook(mlp_post))
    try:
        output = loaded.model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    logits = torch.stack([
        output.logits[index, seq_len - 1].detach().float().cpu()
        for index, seq_len in enumerate(lengths)
    ])
    values, indices = torch.topk(logits, k=50, dim=-1)
    readout_rows: list[dict[str, Any]] = []
    top50_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        target_id = target_first[index]
        target_logit = float(logits[index, target_id].item())
        distractor_logits = [float(logits[index, token_id].item()) for token_id in distractor_first[index]]
        target_rank = 1 + int((logits[index] > target_logit).sum().item())
        readout_rows.append({
            **output_row_base(case, model),
            "created_at": created_at,
            "sequence_length": lengths[index],
            "role_spans": json.dumps(span_rows[index], sort_keys=True),
            "target_first_token_id": target_id,
            "target_first_token": tokenizer.decode([target_id]),
            "target_logit": round(target_logit, 7),
            "best_distractor_logit": round(max(distractor_logits), 7),
            "target_margin": round(target_logit - max(distractor_logits), 7),
            "target_full_vocabulary_rank": target_rank,
            "target_in_top50": target_rank <= 50,
            "candidate_winner_is_target": target_logit >= max(distractor_logits),
        })
        for rank, (token_id, logit) in enumerate(zip(indices[index].tolist(), values[index].tolist(), strict=True), 1):
            top50_rows.append({
                **output_row_base(case, model),
                "created_at": created_at,
                "rank": rank,
                "token_id": int(token_id),
                "token_text": tokenizer.decode([int(token_id)]),
                "logit": round(float(logit), 7),
                "above_target": float(logit) > target_logit,
                "is_target_first_token": int(token_id) == target_id,
                "is_distractor_first_token": int(token_id) in distractor_first[index],
            })
    del output, logits, values, indices, encoded, directions
    return readout_rows, top50_rows


@torch.inference_mode()
def phrase_logprobs(loaded: Any, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tokenizer = loaded.tokenizer
    pad = int(tokenizer.pad_token_id)
    prompt_ids = [tokenizer(case["prompt"], add_special_tokens=True, truncation=True, max_length=128)["input_ids"] for case in cases]
    answer_ids = [target_ids(tokenizer, case["target"]) for case in cases]
    combined = [prompt + answer for prompt, answer in zip(prompt_ids, answer_ids, strict=True)]
    width = max(len(row) for row in combined)
    input_ids = torch.full((len(cases), width), pad, dtype=torch.long, device=loaded.input_device)
    attention_mask = torch.zeros_like(input_ids)
    for index, row in enumerate(combined):
        input_ids[index, : len(row)] = torch.tensor(row, device=loaded.input_device)
        attention_mask[index, : len(row)] = 1
    output = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    log_probs = torch.log_softmax(output.logits.detach().float(), dim=-1)
    rows = []
    for index, (prompt, answer) in enumerate(zip(prompt_ids, answer_ids, strict=True)):
        token_values = [
            float(log_probs[index, len(prompt) + offset - 1, token_id].item())
            for offset, token_id in enumerate(answer)
        ]
        rows.append({
            "target_token_ids": json.dumps(answer),
            "target_token_count": len(answer),
            "target_phrase_logprob": round(sum(token_values), 7),
            "target_phrase_mean_logprob": round(mean(token_values), 7),
            "target_token_logprobs": json.dumps([round(value, 7) for value in token_values]),
        })
    del output, log_probs, input_ids, attention_mask
    return rows


@torch.inference_mode()
def generate_batch(loaded: Any, cases: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    tokenizer = loaded.tokenizer
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [case["prompt"] for case in cases], return_tensors="pt", padding=True,
        truncation=True, max_length=128,
    )
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    generated = loaded.model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # Hooks and model-specific hybrid caches do not expose an equivalent
        # execution surface.  Use the same cache-free path as causal audits.
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    suffix = generated[:, encoded["input_ids"].shape[1] :]
    rows = []
    for index, case in enumerate(cases):
        ids = [int(value) for value in suffix[index].tolist()]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        normalized = normalize_text(text)
        aliases = [normalize_text(alias) for alias in case["target_aliases"]]
        onset = next((offset for offset in range(len(ids)) if normalize_text(tokenizer.decode(ids[: offset + 1], skip_special_tokens=True)).startswith(tuple(aliases))), None)
        rows.append({
            "generated_text": text,
            "generated_token_ids": json.dumps(ids),
            "generated_token_count": len(ids),
            "target_match": any(alias and (normalized.startswith(alias) or alias in normalized) for alias in aliases),
            "target_onset_token": onset,
            "protocol_success": protocol_ok(case, text),
            "eos_emitted": tokenizer.eos_token_id in ids,
            "repetition_detected": len(ids) >= 4 and len(set(ids[-4:])) <= 2,
        })
    del generated, suffix, encoded
    tokenizer.padding_side = "right"
    return rows


def refresh_rollout_partition(
    model: str, family: str, round_name: str, batch_size: int, max_new_tokens: int
) -> dict[str, Any]:
    """Replace only rollout/behavior rows after a generation-runtime correction."""
    cases = read_cases(round_name, family)
    partition = OUT / round_name / "survey" / model / family
    readout_path = partition / "readout.parquet"
    if not readout_path.exists():
        raise FileNotFoundError(f"Survey readout must exist before rollout refresh: {readout_path}")
    readout_rows = pq.read_table(readout_path).to_pylist()
    readout_by_case = {row["case_id"]: row for row in readout_rows}
    loaded = None
    rollout_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        for batch in chunks(cases, batch_size):
            generated = generate_batch(loaded, batch, max_new_tokens)
            for case, rollout in zip(batch, generated, strict=True):
                readout = readout_by_case[case["case_id"]]
                rollout_rows.append({**output_row_base(case, model), "created_at": now(), **rollout})
                behavior_rows.append({
                    **output_row_base(case, model),
                    "created_at": now(),
                    "candidate_winner_is_target": readout["candidate_winner_is_target"],
                    "target_in_top50": readout["target_in_top50"],
                    "target_match": rollout["target_match"],
                    "protocol_success": rollout["protocol_success"],
                    "behavior_success": rollout["target_match"] and rollout["protocol_success"],
                })
        write_parquet(partition / "rollout.parquet", rollout_rows)
        write_parquet(partition / "behavior.parquet", behavior_rows)
        quality = json.loads((partition / "quality.json").read_text(encoding="utf-8"))
        quality.update({
            "created_at": now(),
            "generation_runtime": "cache_free_uniform",
            "rollout_refresh_reason": "model_specific_cache_interface_confound_removed",
            "rollout_refresh_count": len(rollout_rows),
        })
        write_json(partition / "quality.json", quality)
        write_json(partition / "complete.json", quality)
        return quality
    finally:
        release_loaded(loaded)


def path_signatures(event_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in event_rows:
        grouped[(row["case_id"], row["component_type"], row["position_role"])].append(row)
    result = []
    for (_case_id, component, role), rows in grouped.items():
        rows.sort(key=lambda row: row["layer"])
        projections = [float(row["projection"]) for row in rows]
        absolute = [abs(value) for value in projections]
        peak_index = max(range(len(rows)), key=lambda index: absolute[index])
        threshold = absolute[peak_index] * 0.25
        onset_index = next((index for index, value in enumerate(absolute) if value >= threshold), peak_index)
        signs = [1 if value > 0 else -1 if value < 0 else 0 for value in projections]
        flips = sum(left != 0 and right != 0 and left != right for left, right in zip(signs, signs[1:]))
        base = rows[0]
        result.append({
            **{key: base[key] for key in (
                "schema_version", "phase_id", "model", "case_id", "item_id", "family_id",
                "mechanism_id", "split", "template_id", "language", "target_bucket",
                "target_absent_from_prompt", "selection_eligible",
            )},
            "created_at": now(),
            "component_type": component,
            "position_role": role,
            "layer_count": len(rows),
            "onset_layer": int(rows[onset_index]["layer"]),
            "peak_layer": int(rows[peak_index]["layer"]),
            "peak_projection": round(projections[peak_index], 7),
            "peak_absolute_projection": round(absolute[peak_index], 7),
            "mean_projection": round(mean(projections), 7),
            "positive_layer_fraction": round(sum(value > 0 for value in projections) / len(projections), 7),
            "sign_flip_count": flips,
            "post_onset_persistence": round(
                sum(sign == signs[peak_index] for sign in signs[onset_index:]) / max(1, len(signs) - onset_index), 7
            ),
            "path_shape": "single_peak" if flips <= 1 else "oscillatory",
            "evidence_level": "L2_observational_path_signature",
            "single_unit_causal": False,
        })
    return result


def run_partition(model: str, family: str, round_name: str, batch_size: int, max_new_tokens: int) -> dict[str, Any]:
    cases = read_cases(round_name, family)
    partition = OUT / round_name / "survey" / model / family
    complete = partition / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    loaded = None
    behavior_rows: list[dict[str, Any]] = []
    readout_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    top50_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    try:
        loaded = load_probe_model(model)
        layer_count = len(get_layers(loaded.model))
        for batch_index, batch in enumerate(chunks(cases, batch_size), 1):
            batch_readout, batch_top50 = trace_batch(loaded, batch, model, event_rows)
            phrase = phrase_logprobs(loaded, batch)
            rollout = generate_batch(loaded, batch, max_new_tokens)
            for case, readout, phrase_row, rollout_row in zip(batch, batch_readout, phrase, rollout, strict=True):
                readout.update(phrase_row)
                readout_rows.append(readout)
                rollout_rows.append({**output_row_base(case, model), "created_at": now(), **rollout_row})
                behavior_rows.append({
                    **output_row_base(case, model),
                    "created_at": now(),
                    "candidate_winner_is_target": readout["candidate_winner_is_target"],
                    "target_in_top50": readout["target_in_top50"],
                    "target_match": rollout_row["target_match"],
                    "protocol_success": rollout_row["protocol_success"],
                    "behavior_success": rollout_row["target_match"] and rollout_row["protocol_success"],
                })
            top50_rows.extend(batch_top50)
            if batch_index % 8 == 0:
                print(json.dumps({
                    "quality_only": True, "model": model, "family": family,
                    "completed_cases": min(batch_index * batch_size, len(cases)), "total_cases": len(cases),
                }), flush=True)
        signatures = path_signatures(event_rows)
        expected_events = len(cases) * layer_count * len(COMPONENTS) * len(ROLES)
        quality = {
            "schema_version": SCHEMA_VERSION,
            "phase_id": PHASE,
            "created_at": now(),
            "quality_only": True,
            "scientific_analysis_permitted": False,
            "model": model,
            "family_id": family,
            "layer_count": layer_count,
            "counts": {
                "cases": len(cases),
                "behavior_rows": len(behavior_rows),
                "readout_rows": len(readout_rows),
                "rollout_rows": len(rollout_rows),
                "top50_rows": len(top50_rows),
                "component_event_rows": len(event_rows),
                "path_signature_rows": len(signatures),
            },
            "expected": {
                "cases": 576,
                "behavior_rows": 576,
                "readout_rows": 576,
                "rollout_rows": 576,
                "top50_rows": 28800,
                "component_event_rows": expected_events,
                "path_signature_rows": 576 * len(COMPONENTS) * len(ROLES),
            },
        }
        quality["valid"] = quality["counts"] == quality["expected"]
        if not quality["valid"]:
            raise RuntimeError(f"Partition count mismatch: {quality}")
        write_parquet(partition / "behavior.parquet", behavior_rows)
        write_parquet(partition / "readout.parquet", readout_rows)
        write_parquet(partition / "rollout.parquet", rollout_rows)
        write_parquet(partition / "top50.parquet", top50_rows)
        write_parquet(partition / "component_events.parquet", event_rows)
        write_jsonl(partition / "path_signatures.jsonl", signatures)
        write_json(partition / "quality.json", quality)
        write_json(complete, quality)
        return quality
    finally:
        release_loaded(loaded)
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--family", choices=tuple(FAMILY_MECHANISMS), required=True)
    parser.add_argument("--round", default=ROUND_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--refresh-rollout", action="store_true")
    args = parser.parse_args()
    if args.refresh_rollout:
        result = refresh_rollout_partition(
            args.model, args.family, args.round, args.batch_size, args.max_new_tokens
        )
    else:
        result = run_partition(args.model, args.family, args.round, args.batch_size, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
