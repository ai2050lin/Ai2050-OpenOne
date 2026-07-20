#!/usr/bin/env python3
"""Run Phase571 Qwen3 role-separated relation donor interventions."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
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
import phase571_relation_donor_protocol as donor_protocol  # noqa: E402


MODEL = donor_protocol.MODEL
CONDITIONS = donor_protocol.CONDITIONS
OUT_DIR = protocol.OUT_DIR
ROWS_PATH = OUT_DIR / "phase571_qwen3_relation_donor_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase571_qwen3_relation_donor_execution_summary.json"
CONTRACT_PATH = OUT_DIR / "phase571_qwen3_relation_donor_contract.json"


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


def prepare(registry: dict[str, Any], frozen: dict[str, Any], restart: bool) -> None:
    payload = {
        "schema_version": "phase571_relation_donor_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": MODEL,
        "open_cases_sha256": sha256_file(protocol.OPEN_CASES_PATH),
        "donor_registry_sha256": sha256_file(donor_protocol.DONOR_REGISTRY_PATH),
        "donor_protocol_sha256": sha256_file(donor_protocol.DONOR_PROTOCOL_PATH),
        "candidate_pair_count": registry["candidate_pair_count"],
        "final_pair_count": registry["final_pair_count"],
        "conditions": frozen["conditions"],
        "fixed_batch_size": frozen["fixed_batch_size"],
        "do_sample": False,
        "torch_dtype_requested": "torch.bfloat16",
        "full_vectors_persisted": False,
        "sealed_split_read": False,
    }
    if restart:
        for path in (ROWS_PATH, SUMMARY_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    if CONTRACT_PATH.exists():
        existing = read_json(CONTRACT_PATH)
        for key in (
            "model", "open_cases_sha256", "donor_registry_sha256",
            "donor_protocol_sha256", "candidate_pair_count", "final_pair_count",
            "conditions", "fixed_batch_size", "do_sample", "torch_dtype_requested",
            "full_vectors_persisted", "sealed_split_read",
        ):
            if existing[key] != payload[key]:
                raise RuntimeError(f"Phase571 donor contract drift: {key}")
    else:
        write_json(CONTRACT_PATH, payload)


def load_banks(registry: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[list[dict[str, Any]]]]:
    ids = {
        case_id
        for entry in registry["entries"]
        for case_id in (
            entry["receiver_case_id"],
            entry["matched_correct_donor_case_id"],
            entry["wrong_target_donor_case_id"],
        )
    }
    bank = {
        row["case_id"]: row
        for row in iter_jsonl(protocol.OPEN_CASES_PATH)
        if row["model"] == MODEL and row["case_id"] in ids
    }
    if set(bank) != ids or any(row["sealed"] for row in bank.values()):
        raise RuntimeError("Phase571 donor case bank drift")
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in registry["entries"]:
        by_pair[int(entry["pair_index"])].append(entry)
    pair_entries = []
    for pair_index in sorted(by_pair):
        entries = sorted(
            by_pair[pair_index],
            key=lambda entry: 0 if entry["receiver_phenotype"] == "stable_correct" else 1,
        )
        if len(entries) != 2:
            raise RuntimeError("Phase571 donor pair entry drift")
        pair_entries.append(entries)
    batches = []
    for start in range(0, len(pair_entries), 4):
        group = pair_entries[start:start + 4]
        if len(group) != 4:
            raise RuntimeError("Phase571 donor candidate count must be divisible by four")
        batches.append([pair[0] for pair in group] + [pair[1] for pair in group])
    return bank, batches


def coordinate_data(
    tokenizer: Any, rows: list[dict[str, Any]], roles: tuple[str, ...]
) -> tuple[list[str], list[list[int]], dict[str, list[int]]]:
    prompts = [render_chat(tokenizer, MODEL, row["raw_prompt"]) for row in rows]
    individual = [role_positions(tokenizer, prompt, row) for prompt, row in zip(prompts, rows)]
    local = {
        role: [groups[role][-1] for _ids, groups in individual] for role in roles
    }
    return prompts, [ids for ids, _groups in individual], local


def padded_positions(
    encoded: dict[str, torch.Tensor],
    individual_ids: list[list[int]],
    local: list[int],
) -> torch.Tensor:
    width = int(encoded["input_ids"].shape[1])
    output = []
    for index, ids in enumerate(individual_ids):
        mask_ids = encoded["input_ids"][index][encoded["attention_mask"][index].bool()].tolist()
        if [int(value) for value in mask_ids] != ids:
            raise RuntimeError("Phase571 donor individual/batch tokenization drift")
        output.append(width - len(ids) + local[index])
    return torch.tensor(output, dtype=torch.long)


def capture_states(
    loaded: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    entry_layer: int,
    exit_layer: int,
) -> dict[str, torch.Tensor]:
    roles = ("answer_boundary", "query_relation", "target_fact_value")
    prompts, individual_ids, local = coordinate_data(loaded.tokenizer, rows, roles)
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    positions = {
        role: padded_positions(encoded, individual_ids, local[role]) for role in roles
    }
    captured: dict[str, torch.Tensor] = {}

    def gather(value: torch.Tensor, role: str) -> torch.Tensor:
        pos = positions[role].to(value.device)
        batch_indices = torch.arange(len(rows), device=value.device)
        return value[batch_indices, pos, :].detach().cpu().clone()

    def entry_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        value = inputs[0]
        for role in roles:
            captured[f"entry_{role}"] = gather(value, role)

    def exit_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["exit_answer_boundary"] = gather(
            tensor_from_output(output), "answer_boundary"
        )

    handles = [
        layers[entry_layer].register_forward_pre_hook(entry_hook),
        layers[exit_layer].register_forward_hook(exit_hook),
    ]
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            loaded.model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    expected = {
        "entry_answer_boundary", "entry_query_relation",
        "entry_target_fact_value", "exit_answer_boundary",
    }
    if set(captured) != expected:
        raise RuntimeError(f"Phase571 donor capture incomplete: {set(captured)}")
    del encoded
    return captured


def deterministic_random(states: torch.Tensor, rows: list[dict[str, Any]]) -> torch.Tensor:
    random_rows = []
    for index, row in enumerate(rows):
        seed = int(
            hashlib.sha256(f"{row['case_id']}|phase571_donor".encode("utf-8")).hexdigest()[:16],
            16,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        vector = torch.randn(states.shape[-1], generator=generator, dtype=torch.float32)
        vector = vector / vector.norm().clamp_min(1e-12)
        random_rows.append(vector * states[index].float().norm())
    return torch.stack(random_rows)


def run_generation(
    loaded: Any,
    layers: list[Any],
    receivers: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    condition: str,
    frozen: dict[str, Any],
    self_states: dict[str, torch.Tensor],
    matched_states: dict[str, torch.Tensor],
    wrong_states: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    role_by_condition = {
        "self_entry_restore": "answer_boundary",
        "matched_correct_answer_entry": "answer_boundary",
        "matched_correct_answer_exit": "answer_boundary",
        "matched_correct_query_entry": "query_relation",
        "matched_correct_target_fact_entry": "target_fact_value",
        "wrong_target_answer_entry": "answer_boundary",
        "random_matched_answer_entry": "answer_boundary",
    }
    role = role_by_condition.get(condition, "answer_boundary")
    prompts, individual_ids, local = coordinate_data(loaded.tokenizer, receivers, (role,))
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    positions = padded_positions(encoded, individual_ids, local[role])
    output_embeddings = loaded.model.get_output_embeddings()
    target_ids = torch.tensor([
        row["candidate_token_ids"][row["target"]][0] for row in receivers
    ], dtype=torch.long, device=output_embeddings.weight.device)
    other_ids = torch.tensor([
        row["candidate_token_ids"][row["other_relation_target"]][0] for row in receivers
    ], dtype=torch.long, device=output_embeddings.weight.device)
    handle = None
    if condition != "baseline":
        if condition == "self_entry_restore":
            donor_vectors = self_states["entry_answer_boundary"]
        elif condition == "matched_correct_answer_entry":
            donor_vectors = matched_states["entry_answer_boundary"]
        elif condition == "matched_correct_answer_exit":
            donor_vectors = matched_states["exit_answer_boundary"]
        elif condition == "matched_correct_query_entry":
            donor_vectors = matched_states["entry_query_relation"]
        elif condition == "matched_correct_target_fact_entry":
            donor_vectors = matched_states["entry_target_fact_value"]
        elif condition == "wrong_target_answer_entry":
            donor_vectors = wrong_states["entry_answer_boundary"]
        elif condition == "random_matched_answer_entry":
            donor_vectors = deterministic_random(
                self_states["entry_answer_boundary"], receivers
            )
        else:
            raise ValueError(condition)
        patch_exit = condition == "matched_correct_answer_exit"
        patch_layer = int(frozen["exit_layer"] if patch_exit else frozen["entry_layer"])

        def replace(value: torch.Tensor) -> torch.Tensor:
            if value.shape[1] <= int(positions.max().item()):
                return value
            modified = value.clone()
            pos = positions.to(value.device)
            batch_indices = torch.arange(len(receivers), device=value.device)
            modified[batch_indices, pos, :] = donor_vectors.to(
                value.device, dtype=value.dtype
            )
            return modified

        if patch_exit:
            def output_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
                return replace_tensor_output(output, replace(tensor_from_output(output)))
            handle = layers[patch_layer].register_forward_hook(output_hook)
        else:
            def input_hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                return (replace(inputs[0]), *inputs[1:])
            handle = layers[patch_layer].register_forward_pre_hook(input_hook)
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
        if handle is not None:
            handle.remove()
    first_scores = generated.scores[0].float()
    batch_indices = torch.arange(len(receivers), device=first_scores.device)
    margins = first_scores[batch_indices, target_ids.to(first_scores.device)] - first_scores[
        batch_indices, other_ids.to(first_scores.device)
    ]
    results = []
    for index, (receiver, entry) in enumerate(zip(receivers, entries)):
        text = loaded.tokenizer.decode(
            generated.sequences[index, prompt_width:], skip_special_tokens=True
        )
        results.append({
            **receiver,
            **classify(receiver, text),
            "condition": condition,
            "receiver_phenotype": entry["receiver_phenotype"],
            "pair_index": entry["pair_index"],
            "matched_correct_donor_case_id": entry["matched_correct_donor_case_id"],
            "wrong_target_donor_case_id": entry["wrong_target_donor_case_id"],
            "matched_donor_target": entry["matched_donor_target"],
            "wrong_donor_target": entry["wrong_donor_target"],
            "first_step_target_minus_other_margin": float(margins[index].item()),
            "full_vectors_persisted": False,
            "causal": condition != "baseline",
            "sealed": False,
        })
    del encoded, generated, first_scores, margins
    return results


def run(max_new_tokens: int, restart: bool) -> Path:
    registry = read_json(donor_protocol.DONOR_REGISTRY_PATH)
    frozen = read_json(donor_protocol.DONOR_PROTOCOL_PATH)
    prepare(registry, frozen, restart)
    bank, batches = load_banks(registry)
    loaded = None
    all_rows: list[dict[str, Any]] = []
    baseline_valid_pairs: set[int] = set()
    started = time.monotonic()
    try:
        loaded = load_probe_model(MODEL)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase571 donor requires BF16, got {run_dtype}")
        layers = get_layers(loaded.model)
        for batch_index, entries in enumerate(batches):
            receivers = [bank[entry["receiver_case_id"]] for entry in entries]
            matched_donors = [bank[entry["matched_correct_donor_case_id"]] for entry in entries]
            wrong_donors = [bank[entry["wrong_target_donor_case_id"]] for entry in entries]
            self_states = capture_states(
                loaded, layers, receivers, int(frozen["entry_layer"]), int(frozen["exit_layer"])
            )
            matched_states = capture_states(
                loaded, layers, matched_donors,
                int(frozen["entry_layer"]), int(frozen["exit_layer"]),
            )
            wrong_states = capture_states(
                loaded, layers, wrong_donors,
                int(frozen["entry_layer"]), int(frozen["exit_layer"]),
            )
            baseline = run_generation(
                loaded, layers, receivers, entries, "baseline", frozen,
                self_states, matched_states, wrong_states, max_new_tokens,
            )
            for pair_offset in range(4):
                left = baseline[pair_offset]
                right = baseline[pair_offset + 4]
                if left["semantic_correct"] and right["relation_confusion"]:
                    baseline_valid_pairs.add(int(left["pair_index"]))
            all_rows.extend(baseline)
            for condition in CONDITIONS[1:]:
                all_rows.extend(run_generation(
                    loaded, layers, receivers, entries, condition, frozen,
                    self_states, matched_states, wrong_states, max_new_tokens,
                ))
            if batch_index == 0 or batch_index == len(batches) - 1 or batch_index % 8 == 7:
                print(
                    f"[{time.strftime('%H:%M:%S')}] qwen3 Phase571 donor "
                    f"{batch_index + 1}/{len(batches)} paired batches",
                    flush=True,
                )
        ordered_valid = sorted(baseline_valid_pairs)
        if len(ordered_valid) < int(registry["final_pair_count"]):
            raise RuntimeError(
                f"Phase571 donor has only {len(ordered_valid)} stable pairs"
            )
        final_pairs = set(ordered_valid[: int(registry["final_pair_count"])])
        all_rows = [row for row in all_rows if int(row["pair_index"]) in final_pairs]
        write_jsonl(ROWS_PATH, all_rows)
        summary = {
            "schema_version": "phase571_relation_donor_execution_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": MODEL,
            "torch_dtype": run_dtype,
            "candidate_pair_count": registry["candidate_pair_count"],
            "baseline_valid_pair_count": len(ordered_valid),
            "baseline_drift_pair_count": registry["candidate_pair_count"] - len(ordered_valid),
            "final_pair_count": registry["final_pair_count"],
            "final_receiver_count": registry["final_pair_count"] * 2,
            "condition_count": len(CONDITIONS),
            "causal_row_count": len(all_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "full_vectors_persisted": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, summary)
        print(SUMMARY_PATH, flush=True)
        return SUMMARY_PATH
    finally:
        release_loaded(loaded)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    run(args.max_new_tokens, args.restart)


if __name__ == "__main__":
    main()
