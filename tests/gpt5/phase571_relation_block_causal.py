#!/usr/bin/env python3
"""Run frozen Phase571 coarse contiguous-block delete/restore controls."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/gpt5"))
os.environ.setdefault("PROBE_TORCH_DTYPE", "bfloat16")

from hf_probe_env import get_layers, load_probe_model, release_loaded  # noqa: E402
from phase548_shared_attention_compute_protocol import render_chat  # noqa: E402
from phase569_relation_competition_behavior import classify  # noqa: E402
from phase569_role_position_utils import role_positions  # noqa: E402
import phase571_relation_block_protocol as protocol  # noqa: E402


MODELS = protocol.MODELS
CONDITIONS = tuple(read for read in (
    "baseline",
    "signed_block_remove",
    "full_block_remove",
    "full_block_remove_restore",
    "wrong_depth_full_remove",
    "wrong_role_full_remove",
    "random_matched_replace",
))
OUT_DIR = protocol.OUT_DIR
REGISTRY_PATH = OUT_DIR / "phase571_continuous_block_registry.json"
AMENDMENT_PATH = OUT_DIR / "phase571_causal_reserve_amendment.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_from_output(output: Any) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not torch.is_tensor(value):
        raise TypeError(f"Unexpected hook output: {type(value).__name__}")
    return value


def replace_tensor_output(output: Any, value: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


def matched_summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_causal_reserve_summary.json"


def rows_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_coarse_block_causal_rows.jsonl.gz"


def summary_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_coarse_block_execution_summary.json"


def contract_path(model: str) -> Path:
    return OUT_DIR / f"phase571_{model}_coarse_block_contract.json"


def load_labeled_pairs(model: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    summary = read_json(matched_summary_path(model))
    ids = summary["selected_case_ids_by_phenotype"]
    correct_ids = ids["stable_correct"]
    confusion_ids = ids["stable_relation_confusion"]
    if len(correct_ids) != len(confusion_ids) or len(correct_ids) < 128:
        raise RuntimeError(f"Phase571 causal matched denominator failed for {model}")
    labels = {
        **{case_id: "stable_correct" for case_id in correct_ids},
        **{case_id: "stable_relation_confusion" for case_id in confusion_ids},
    }
    bank = {
        row["case_id"]: {**row, "causal_phenotype": labels[row["case_id"]]}
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["model"] == model and row["case_id"] in labels
    }
    pairs = [(bank[left], bank[right]) for left, right in zip(correct_ids, confusion_ids)]
    if len(pairs) < 128 or any(left["sealed"] or right["sealed"] for left, right in pairs):
        raise RuntimeError(f"Phase571 causal case bank drift for {model}")
    for left, right in pairs:
        left_key = (left["source_factorial_cell"], left["target"], left["other_relation_target"])
        right_key = (right["source_factorial_cell"], right["target"], right["other_relation_target"])
        if left_key != right_key:
            raise RuntimeError("Phase571 matched pair identity drift")
    return pairs


def fixed_pair_batches(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], pair_count: int | None = None
) -> list[list[dict[str, Any]]]:
    selected = pairs[:pair_count] if pair_count is not None else pairs
    if len(selected) % 4:
        raise RuntimeError("Phase571 matched pair bank must be divisible by four")
    batches = []
    for start in range(0, len(selected), 4):
        group = selected[start:start + 4]
        batches.append([left for left, _right in group] + [right for _left, right in group])
    if not batches or any(len(batch) != 8 for batch in batches):
        raise RuntimeError("Phase571 fixed paired batch construction failed")
    return batches


def prepare(
    model: str, block: dict[str, Any], candidate_pair_count: int, restart: bool
) -> None:
    frozen = read_json(protocol.PROTOCOL_PATH)
    payload = {
        "schema_version": "phase571_coarse_block_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": model,
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "causal_reserve_amendment_sha256": sha256_file(AMENDMENT_PATH),
        "matched_behavior_summary_sha256": sha256_file(matched_summary_path(model)),
        "selected_block": block,
        "conditions": list(CONDITIONS),
        "candidate_paired_cases_per_phenotype": candidate_pair_count,
        "final_paired_cases_per_phenotype": 128,
        "fixed_batch_size": frozen["fixed_execution_batch_size"],
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "full_vectors_persisted": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (rows_path(model), summary_path(model), contract_path(model)):
            path.unlink(missing_ok=True)
    if contract_path(model).exists():
        existing = read_json(contract_path(model))
        for key in (
            "model", "protocol_sha256", "registry_sha256", "causal_reserve_amendment_sha256",
            "matched_behavior_summary_sha256", "selected_block", "conditions",
            "candidate_paired_cases_per_phenotype", "final_paired_cases_per_phenotype",
            "fixed_batch_size", "do_sample",
            "torch_dtype_requested", "full_vectors_persisted", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase571 causal contract drift: {model}/{key}")
    else:
        write_json(contract_path(model), payload)


def batch_coordinates(
    tokenizer: Any,
    model: str,
    rows: list[dict[str, Any]],
    target_role: str,
    wrong_role: str,
) -> tuple[list[str], list[list[int]], list[int], list[int]]:
    prompts = [render_chat(tokenizer, model, row["raw_prompt"]) for row in rows]
    individual = [role_positions(tokenizer, prompt, row) for prompt, row in zip(prompts, rows)]
    target_local = [groups[target_role][-1] for _ids, groups in individual]
    wrong_local = [groups[wrong_role][-1] for _ids, groups in individual]
    return prompts, [ids for ids, _groups in individual], target_local, wrong_local


def deterministic_random_unit(
    rows: list[dict[str, Any]], hidden: int, layer: int, component: str, device: torch.device
) -> torch.Tensor:
    vectors = []
    for row in rows:
        payload = f"{row['case_id']}|{layer}|{component}|phase571"
        seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vectors.append(torch.randn(hidden, generator=generator, dtype=torch.float32))
    random = torch.stack(vectors).to(device)
    return random / random.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def run_condition(
    loaded: Any,
    layers: list[Any],
    model: str,
    rows: list[dict[str, Any]],
    condition: str,
    block: dict[str, Any],
    max_new_tokens: int,
    baseline_exit_cache: torch.Tensor | None,
) -> tuple[list[dict[str, Any]], torch.Tensor | None]:
    prompts, individual_ids, target_local, wrong_local = batch_coordinates(
        loaded.tokenizer,
        model,
        rows,
        block["semantic_role"],
        block["wrong_role_control"],
    )
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    target_positions = []
    wrong_positions = []
    for row_index, ids in enumerate(individual_ids):
        mask_ids = encoded["input_ids"][row_index][
            encoded["attention_mask"][row_index].bool()
        ].tolist()
        if [int(value) for value in mask_ids] != ids:
            raise RuntimeError("Phase571 causal individual/batch tokenization drift")
        offset = prompt_width - len(ids)
        target_positions.append(offset + target_local[row_index])
        wrong_positions.append(offset + wrong_local[row_index])
    target_positions_tensor = torch.tensor(target_positions, dtype=torch.long)
    wrong_positions_tensor = torch.tensor(wrong_positions, dtype=torch.long)
    output_embeddings = loaded.model.get_output_embeddings()
    target_ids = torch.tensor([
        row["candidate_token_ids"][row["target"]][0] for row in rows
    ], dtype=torch.long, device=output_embeddings.weight.device)
    other_ids = torch.tensor([
        row["candidate_token_ids"][row["other_relation_target"]][0] for row in rows
    ], dtype=torch.long, device=output_embeddings.weight.device)
    direction = (
        output_embeddings.weight[target_ids] - output_embeddings.weight[other_ids]
    ).detach().float()
    direction_unit = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    handles: list[Any] = []
    captured_exit: list[torch.Tensor] = []

    def select_positions(value: torch.Tensor, wrong: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = wrong_positions_tensor if wrong else target_positions_tensor
        positions = positions.to(value.device)
        batch_indices = torch.arange(len(rows), device=value.device)
        return value[batch_indices, positions, :], batch_indices, positions

    if condition == "baseline":
        def capture_exit(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            value = tensor_from_output(output)
            if value.shape[1] <= max(target_positions):
                return
            selected, _batch_indices, _positions = select_positions(value)
            captured_exit.append(selected.detach().cpu().clone())

        handles.append(layers[int(block["end_layer"])].register_forward_hook(capture_exit))
    else:
        if condition == "wrong_depth_full_remove":
            active_layers = range(int(block["wrong_start_layer"]), int(block["wrong_end_layer"]) + 1)
        else:
            active_layers = range(int(block["start_layer"]), int(block["end_layer"]) + 1)

        def make_component_hook(layer_index: int, component: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                value = tensor_from_output(output)
                use_wrong_role = condition == "wrong_role_full_remove"
                positions_check = wrong_positions if use_wrong_role else target_positions
                if value.shape[1] <= max(positions_check):
                    return output
                modified = value.clone()
                vectors, batch_indices, positions = select_positions(value, use_wrong_role)
                vectors_float = vectors.float()
                if condition == "signed_block_remove":
                    unit = direction_unit.to(value.device)
                    projection = (vectors_float * unit).sum(dim=-1, keepdim=True)
                    replacement = vectors_float - projection * unit
                elif condition == "random_matched_replace":
                    random = deterministic_random_unit(
                        rows, vectors.shape[-1], layer_index, component, value.device
                    )
                    replacement = random * vectors_float.norm(dim=-1, keepdim=True)
                else:
                    replacement = torch.zeros_like(vectors_float)
                modified[batch_indices, positions, :] = replacement.to(modified.dtype)
                return replace_tensor_output(output, modified)
            return hook

        for layer_index in active_layers:
            layer = layers[layer_index]
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_component_hook(layer_index, "attention_output")
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(
                    make_component_hook(layer_index, "mlp_output")
                )
            )
        if condition == "full_block_remove_restore":
            if baseline_exit_cache is None:
                raise RuntimeError("Phase571 restore condition lacks natural exit cache")

            def restore_exit(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                value = tensor_from_output(output)
                if value.shape[1] <= max(target_positions):
                    return output
                modified = value.clone()
                _vectors, batch_indices, positions = select_positions(value)
                modified[batch_indices, positions, :] = baseline_exit_cache.to(
                    modified.device, dtype=modified.dtype
                )
                return replace_tensor_output(output, modified)

            handles.append(layers[int(block["end_layer"])].register_forward_hook(restore_exit))

    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            generated = loaded.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=loaded.tokenizer.pad_token_id,
                eos_token_id=loaded.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    if not generated.scores:
        raise RuntimeError("Phase571 generation returned no first-step scores")
    if condition == "baseline" and len(captured_exit) != 1:
        raise RuntimeError(f"Phase571 natural exit capture count drift: {len(captured_exit)}")
    first_scores = generated.scores[0].float()
    batch_indices = torch.arange(len(rows), device=first_scores.device)
    margins = first_scores[batch_indices, target_ids.to(first_scores.device)] - first_scores[
        batch_indices, other_ids.to(first_scores.device)
    ]
    results = []
    for index, row in enumerate(rows):
        text = loaded.tokenizer.decode(
            generated.sequences[index, prompt_width:], skip_special_tokens=True
        )
        results.append({
            **row,
            **classify(row, text),
            "condition": condition,
            "causal_phenotype": row["causal_phenotype"],
            "first_step_target_minus_other_margin": float(margins[index].item()),
            "selected_block_role": block["semantic_role"],
            "selected_block_start_layer": block["start_layer"],
            "selected_block_end_layer": block["end_layer"],
            "wrong_role_control": block["wrong_role_control"],
            "wrong_block_start_layer": block["wrong_start_layer"],
            "wrong_block_end_layer": block["wrong_end_layer"],
            "full_vectors_persisted": False,
            "causal": condition != "baseline",
            "sealed": False,
        })
    exit_cache = captured_exit[0] if captured_exit else baseline_exit_cache
    del generated, encoded, first_scores, margins
    return results, exit_cache


def run(model: str, max_new_tokens: int, restart: bool) -> Path:
    registry = read_json(REGISTRY_PATH)
    if model not in registry["authorized_models"]:
        raise RuntimeError(f"Phase571 coarse causal is not authorized for {model}")
    block = registry["selected_block_by_model"][model]
    pairs = load_labeled_pairs(model)
    batches = fixed_pair_batches(pairs)
    prepare(model, block, len(pairs), restart)
    loaded = None
    started = time.monotonic()
    output_rows: list[dict[str, Any]] = []
    baseline_valid_ids: set[str] = set()
    try:
        loaded = load_probe_model(model)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase571 causal requires BF16, got {run_dtype}")
        layers = get_layers(loaded.model)
        if len(layers) != int(block["layer_count"]):
            raise RuntimeError("Phase571 causal layer count drift")
        for batch_index, batch in enumerate(batches):
            baseline_rows, exit_cache = run_condition(
                loaded, layers, model, batch, "baseline", block, max_new_tokens, None
            )
            baseline_by_id = {row["case_id"]: row for row in baseline_rows}
            for pair_index in range(4):
                left = baseline_rows[pair_index]
                right = baseline_rows[pair_index + 4]
                left_valid = left["semantic_correct"]
                right_valid = right["relation_confusion"]
                if left_valid and right_valid:
                    baseline_valid_ids.update((left["case_id"], right["case_id"]))
            output_rows.extend(baseline_rows)
            for condition in CONDITIONS[1:]:
                condition_rows, _unused = run_condition(
                    loaded, layers, model, batch, condition, block,
                    max_new_tokens, exit_cache,
                )
                output_rows.extend(condition_rows)
            if batch_index == 0 or batch_index == len(batches) - 1 or batch_index % 8 == 7:
                print(
                    f"[{time.strftime('%H:%M:%S')}] {model} Phase571 coarse causal "
                    f"{batch_index + 1}/{len(batches)} paired batches",
                    flush=True,
                )
        valid_pairs = [
            (left, right) for left, right in pairs
            if left["case_id"] in baseline_valid_ids and right["case_id"] in baseline_valid_ids
        ]
        if len(valid_pairs) < 128:
            raise RuntimeError(
                f"Phase571 fixed causal baseline has only {len(valid_pairs)} valid pairs for {model}"
            )
        final_ids = {
            case["case_id"] for pair in valid_pairs[:128] for case in pair
        }
        output_rows = [row for row in output_rows if row["case_id"] in final_ids]
        write_jsonl(rows_path(model), output_rows)
        summary = {
            "schema_version": "phase571_coarse_block_execution_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": model,
            "torch_dtype": run_dtype,
            "selected_block": block,
            "candidate_paired_cases_per_phenotype": len(pairs),
            "paired_cases_per_phenotype": 128,
            "baseline_valid_pair_count": len(valid_pairs),
            "baseline_drift_pair_count": len(pairs) - len(valid_pairs),
            "condition_count": len(CONDITIONS),
            "causal_row_count": len(output_rows),
            "fixed_batch_count": len(batches),
            "fixed_batch_size": 8,
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(rows_path(model)),
            "full_vectors_persisted": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(summary_path(model), summary)
        print(summary_path(model), flush=True)
        return summary_path(model)
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.model, args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
