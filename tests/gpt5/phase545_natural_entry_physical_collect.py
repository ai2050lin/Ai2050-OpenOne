#!/usr/bin/env python3
"""Collect full-layer, multi-position Phase545 natural-entry trajectories.

Only aggregate pair geometry is retained. Full hidden vectors are compared in
memory and discarded; no head, channel, or neuron scan is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
sys.path.insert(0, str(ROOT / "tests/glm5"))

from hf_probe_env import get_layers  # noqa: E402
from model_utils import load_model, release_model  # noqa: E402
from phase358_multiresolution_component_conservation import install_hooks  # noqa: E402


MODELS = ("qwen3", "glm4", "deepseek7b")
SOURCE = ROOT / "tests/gpt5/result/phase544_nine_family_natural_behavior"
OUT_DIR = ROOT / "tests/gpt5/result/phase545_natural_entry_physical_path"
PAIRS_PATH = OUT_DIR / "phase545_registered_physical_pairs.jsonl"
PROTOCOL_PATH = OUT_DIR / "phase545_physical_protocol.json"
AUDIT_PATH = OUT_DIR / "phase545_static_audit.json"
CASES_PATH = SOURCE / "phase544_registered_cases.jsonl"
COMPONENTS = ("layer_input", "attention_output", "mlp_output", "layer_output")
ROLES = ("source", "query", "current")
STAGES = ("prompt_end", "after_first_generated_token", "after_third_generated_token")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite Phase545 scalar: {value}")
    return round(float(value), 9)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float()
    right = right.float()
    denominator = float(torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item())
    if denominator <= 1e-12:
        return 0.0
    return clean(float(torch.dot(left, right).item()) / denominator)


def normalized_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(left.float() - right.float()).item())
    denominator = 0.5 * (
        float(torch.linalg.vector_norm(left.float()).item())
        + float(torch.linalg.vector_norm(right.float()).item())
    )
    return clean(numerator / max(denominator, 1e-8))


def relative_error(actual: torch.Tensor, reconstructed: torch.Tensor) -> float:
    numerator = float(torch.linalg.vector_norm(actual.float() - reconstructed.float()).item())
    denominator = float(torch.linalg.vector_norm(actual.float()).item())
    return clean(numerator / max(denominator, 1e-8))


def find_subsequence(sequence: list[int], fragment: list[int]) -> list[int] | None:
    if not fragment or len(fragment) > len(sequence):
        return None
    for start in range(len(sequence) - len(fragment) + 1):
        if sequence[start:start + len(fragment)] == fragment:
            return list(range(start, start + len(fragment)))
    return None


def fragment_positions(tokenizer: Any, prompt_ids: list[int], fragment: str) -> list[int]:
    candidates = [fragment, " " + fragment]
    for text in candidates:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        variants = [ids]
        if len(ids) > 1:
            variants.extend((ids[1:], ids[:-1]))
        if len(ids) > 2:
            variants.append(ids[1:-1])
        for variant in variants:
            positions = find_subsequence(prompt_ids, [int(value) for value in variant])
            if positions is not None:
                return positions
    raise RuntimeError(f"Cannot locate prompt fragment: {fragment[:80]!r}")


def parse_direct_fragments(raw_prompt: str) -> tuple[str, str]:
    match = re.match(
        r"^Context: (?P<context>.*)\nQuestion: (?P<question>.*)\nInstruction:",
        raw_prompt,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError(f"Unexpected direct prompt: {raw_prompt[:120]!r}")
    return f"Context: {match.group('context')}", f"Question: {match.group('question')}"


def first_divergent_ids(tokenizer: Any, left: str, right: str) -> tuple[int, int, int]:
    left_ids = [int(value) for value in tokenizer(left, add_special_tokens=False)["input_ids"]]
    right_ids = [int(value) for value in tokenizer(right, add_special_tokens=False)["input_ids"]]
    if not left_ids or not right_ids:
        raise RuntimeError((left, right))
    limit = min(len(left_ids), len(right_ids))
    for index in range(limit):
        if left_ids[index] != right_ids[index]:
            return left_ids[index], right_ids[index], index
    eos = int(tokenizer.eos_token_id)
    if len(left_ids) == len(right_ids):
        raise RuntimeError(f"Targets have identical tokenization: {left!r}, {right!r}")
    return (
        left_ids[limit] if len(left_ids) > limit else eos,
        right_ids[limit] if len(right_ids) > limit else eos,
        limit,
    )


def mean_position(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
    return tensor[0].index_select(0, index).float().mean(dim=0).detach().cpu()


def verify() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    audit = read_json(AUDIT_PATH)
    if audit["status"] != "static_pass_no_hidden_state_read" or not audit["valid"]:
        raise RuntimeError("Phase545 static protocol did not pass")
    if protocol["registered_pairs_sha256"] != sha256_file(PAIRS_PATH):
        raise RuntimeError("Phase545 pair registry drift")
    if protocol["claim_boundaries"]["sealed_split_read"]:
        raise RuntimeError("Phase545 must not read a sealed split")
    return protocol


def stage_vectors(
    model: Any,
    device: torch.device,
    layers: list[Any],
    captures: dict[tuple[str, int], torch.Tensor],
    ids: list[int],
    source_positions: list[int],
    query_positions: list[int],
) -> tuple[dict[int, dict[str, dict[str, torch.Tensor]]], float]:
    captures.clear()
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    current = len(ids) - 1
    validation_positions = sorted({current, *source_positions, *query_positions})
    validation_index = torch.tensor(validation_positions, dtype=torch.long, device=device)
    output: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    max_error = 0.0
    for layer_index in range(len(layers)):
        output[layer_index] = {}
        for component in COMPONENTS:
            tensor = captures[(component, layer_index)]
            output[layer_index][component] = {
                "source": mean_position(tensor, source_positions),
                "query": mean_position(tensor, query_positions),
                "current": tensor[0, current].float().detach().cpu(),
            }
        actual = captures[("layer_output", layer_index)][0].index_select(0, validation_index).float()
        reconstructed = (
            captures[("layer_input", layer_index)][0].index_select(0, validation_index)
            + captures[("attention_output", layer_index)][0].index_select(0, validation_index)
            + captures[("mlp_output", layer_index)][0].index_select(0, validation_index)
        ).float()
        errors = torch.linalg.vector_norm(actual - reconstructed, dim=-1) / torch.clamp(
            torch.linalg.vector_norm(actual, dim=-1), min=1e-8
        )
        max_error = max(max_error, float(errors.max().item()))
    del result, input_ids, attention_mask, validation_index
    captures.clear()
    return output, max_error


def collect_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    captures: dict[tuple[str, int], torch.Tensor],
    row: dict[str, Any],
) -> tuple[dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]], dict[str, Any]]:
    encoded = tokenizer(row["prompt"], return_tensors="pt", add_special_tokens=True)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    source_fragment, query_fragment = parse_direct_fragments(row["raw_prompt"])
    source_positions = fragment_positions(tokenizer, prompt_ids, source_fragment)
    query_positions = fragment_positions(tokenizer, prompt_ids, query_fragment)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    captures.clear()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = [int(value) for value in generated[0, len(prompt_ids):].tolist()]
    if not generated_ids:
        generated_ids = [int(tokenizer.eos_token_id)]
    generated_prefix = tokenizer.decode(generated_ids, skip_special_tokens=True)
    stages = {
        "prompt_end": prompt_ids,
        "after_first_generated_token": [*prompt_ids, *generated_ids[:1]],
        "after_third_generated_token": [*prompt_ids, *generated_ids[:3]],
    }
    vectors = {}
    ledger_errors = {}
    for stage, ids in stages.items():
        vectors[stage], ledger_errors[stage] = stage_vectors(
            model, device, layers, captures, ids, source_positions, query_positions
        )
    expected_prefix = " ".join(row["generated_text"].strip().split())
    observed_prefix = " ".join(generated_prefix.strip().split())
    reproducible = (
        not observed_prefix
        or expected_prefix.startswith(observed_prefix)
        or observed_prefix.startswith(expected_prefix)
    )
    del generated, input_ids, attention_mask, encoded
    return vectors, {
        "generated_token_count": len(generated_ids),
        "generated_prefix": generated_prefix,
        "behavior_generation_prefix_reproducible": reproducible,
        "source_position_count": len(source_positions),
        "query_position_count": len(query_positions),
        "stage_ledger_errors": ledger_errors,
    }


def combine_pair(
    loaded_model: Any,
    tokenizer: Any,
    layers: list[Any],
    pair: dict[str, Any],
    row_a: dict[str, Any],
    row_b: dict[str, Any],
    vectors_a: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    vectors_b: dict[str, dict[int, dict[str, dict[str, torch.Tensor]]]],
    meta_a: dict[str, Any],
    meta_b: dict[str, Any],
) -> list[dict[str, Any]]:
    token_a, token_b, divergence_index = first_divergent_ids(
        tokenizer, row_a["strict_expected"], row_b["strict_expected"]
    )
    embedding = loaded_model.get_output_embeddings()
    direction = embedding.weight[token_a].float().detach().cpu() - embedding.weight[token_b].float().detach().cpu()
    rows = []
    for stage in STAGES:
        for layer_index in range(len(layers)):
            features = {}
            for component in COMPONENTS:
                for role in ROLES:
                    left = vectors_a[stage][layer_index][component][role]
                    right = vectors_b[stage][layer_index][component][role]
                    delta = left - right
                    prefix = f"{component}__{role}"
                    features[prefix] = {
                        "normalized_world_delta": normalized_delta(left, right),
                        "world_cosine": cosine(left, right),
                        "pair_direction_alignment": cosine(delta, direction),
                        "world_a_norm": clean(float(torch.linalg.vector_norm(left).item())),
                        "world_b_norm": clean(float(torch.linalg.vector_norm(right).item())),
                    }
            rows.append({
                "schema_version": "phase545_natural_entry_pair_layer.v1",
                "phase_id": "Phase545",
                "created_at": now(),
                "physical_pair_id": pair["physical_pair_id"],
                "model": pair["model"],
                "family_id": pair["family_id"],
                "mechanism_id": pair["mechanism_id"],
                "split": pair["split"],
                "pair_index": pair["pair_index"],
                "stage": stage,
                "layer": layer_index,
                "layer_count": len(layers),
                "relative_depth": clean(layer_index / max(1, len(layers) - 1)),
                "target_divergence_token_index": divergence_index,
                "target_divergence_token_a": token_a,
                "target_divergence_token_b": token_b,
                "features": features,
                "max_component_ledger_relative_error": clean(max(
                    meta_a["stage_ledger_errors"][stage], meta_b["stage_ledger_errors"][stage]
                )),
                "generation_prefix_reproducible": bool(
                    meta_a["behavior_generation_prefix_reproducible"]
                    and meta_b["behavior_generation_prefix_reproducible"]
                ),
                "source_position_count_a": meta_a["source_position_count"],
                "source_position_count_b": meta_b["source_position_count"],
                "query_position_count_a": meta_a["query_position_count"],
                "query_position_count_b": meta_b["query_position_count"],
                "physical": True,
                "observer_only": True,
                "predictive": False,
                "causal": False,
                "compute_edge": False,
                "single_neuron": False,
                "sealed": False,
            })
    return rows


def run_model(model_name: str, restart: bool) -> Path:
    protocol = verify()
    pairs = [row for row in read_jsonl(PAIRS_PATH) if row["model"] == model_name]
    output_path = OUT_DIR / f"phase545_{model_name}_pair_layer_rows.jsonl"
    summary_path = OUT_DIR / f"phase545_{model_name}_collection_summary.json"
    if not pairs:
        payload = {
            "schema_version": "phase545_collection_summary.v1",
            "phase_id": "Phase545",
            "created_at": now(),
            "status": "skipped_by_behavior_gate",
            "model": model_name,
            "registered_pair_count": 0,
            "pair_layer_row_count": 0,
            "cuda_loaded": False,
            "sealed_split_read": False,
        }
        write_json(summary_path, payload)
        return summary_path
    if restart:
        output_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    completed: set[str] = set()
    if output_path.exists():
        for row in read_jsonl(output_path):
            completed.add(row["physical_pair_id"])
    pair_cases = {row["case_id"]: row for row in read_jsonl(CASES_PATH) if row["model"] == model_name}
    behavior_rows = {
        row["case_id"]: row
        for row in read_jsonl(SOURCE / f"phase544_{model_name}_behavior_rows.jsonl")
    }
    model = None
    handles: list[Any] = []
    started = time.monotonic()
    captures: dict[tuple[str, int], torch.Tensor] = {}
    new_pairs = 0
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Phase545 physical collection requires CUDA")
        model, tokenizer, device = load_model(model_name)
        layers = get_layers(model)
        handles = install_hooks(layers, captures)
        for index, pair in enumerate(pairs):
            if pair["physical_pair_id"] in completed:
                continue
            source_a = dict(pair_cases[pair["world_a_case_id"]])
            source_b = dict(pair_cases[pair["world_b_case_id"]])
            source_a["generated_text"] = behavior_rows[source_a["case_id"]]["generated_text"]
            source_b["generated_text"] = behavior_rows[source_b["case_id"]]["generated_text"]
            vectors_a, meta_a = collect_condition(model, tokenizer, device, layers, captures, source_a)
            vectors_b, meta_b = collect_condition(model, tokenizer, device, layers, captures, source_b)
            rows = combine_pair(
                model, tokenizer, layers, pair, source_a, source_b,
                vectors_a, vectors_b, meta_a, meta_b,
            )
            append_jsonl(output_path, rows)
            new_pairs += 1
            del vectors_a, vectors_b, rows
            gc.collect()
            if new_pairs == 1 or new_pairs % 8 == 0 or len(completed) + new_pairs == len(pairs):
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model_name} physical pairs "
                    f"{len(completed) + new_pairs}/{len(pairs)}",
                    flush=True,
                )
        rows = read_jsonl(output_path)
        observed_pairs = {row["physical_pair_id"] for row in rows}
        if len(observed_pairs) != len(pairs):
            raise RuntimeError(f"Incomplete Phase545 {model_name}: {len(observed_pairs)}/{len(pairs)}")
        payload = {
            "schema_version": "phase545_collection_summary.v1",
            "phase_id": "Phase545",
            "created_at": now(),
            "status": "complete",
            "model": model_name,
            "registered_pair_count": len(pairs),
            "completed_pair_count": len(observed_pairs),
            "pair_layer_row_count": len(rows),
            "layer_count": len(layers),
            "new_pairs_this_invocation": new_pairs,
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "generation_prefix_reproducible_rate": sum(row["generation_prefix_reproducible"] for row in rows) / len(rows),
            "max_component_ledger_relative_error": max(row["max_component_ledger_relative_error"] for row in rows),
            "output_path": str(output_path.relative_to(ROOT)),
            "output_sha256": sha256_file(output_path),
            "cuda_loaded": True,
            "full_hidden_vectors_persisted": False,
            "head_channel_neuron_scan_executed": False,
            "sealed_split_read": False,
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
        }
        write_json(summary_path, payload)
        print(summary_path)
        return summary_path
    finally:
        for handle in handles:
            handle.remove()
        captures.clear()
        if model is not None:
            release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run_model(args.model, args.restart)


if __name__ == "__main__":
    main()
