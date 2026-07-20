#!/usr/bin/env python3
"""Run Phase572 Qwen3 joint query/fact/answer entry interventions."""

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
import phase572_relation_joint_protocol as protocol  # noqa: E402
import phase572_relation_joint_behavior as behavior  # noqa: E402


OUT_DIR = protocol.OUT_DIR
ROWS_PATH = OUT_DIR / "phase572_qwen3_joint_causal_rows.jsonl.gz"
SUMMARY_PATH = OUT_DIR / "phase572_qwen3_joint_causal_summary.json"
CONTRACT_PATH = OUT_DIR / "phase572_qwen3_joint_causal_contract.json"
ROLES = ("query_relation", "target_fact_value", "answer_boundary")
ROLE_SETS = {
    "self_qfa_entry_restore": ROLES,
    "matched_answer_entry": ("answer_boundary",),
    "matched_query_entry": ("query_relation",),
    "matched_fact_entry": ("target_fact_value",),
    "matched_query_answer_entry": ("query_relation", "answer_boundary"),
    "matched_fact_answer_entry": ("target_fact_value", "answer_boundary"),
    "matched_query_fact_entry": ("query_relation", "target_fact_value"),
    "matched_query_fact_answer_entry": ROLES,
    "wrong_target_query_fact_answer_entry": ROLES,
    "random_query_fact_answer_entry": ROLES,
}


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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise RuntimeError("Phase572 individual/batch tokenization drift")
        output.append(width - len(ids) + local[index])
    return torch.tensor(output, dtype=torch.long)


def coordinate_data(
    tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[list[str], list[list[int]], dict[str, list[int]]]:
    prompts = [render_chat(tokenizer, protocol.MODEL, row["raw_prompt"]) for row in rows]
    individual = [role_positions(tokenizer, prompt, row) for prompt, row in zip(prompts, rows)]
    local = {role: [groups[role][-1] for _ids, groups in individual] for role in ROLES}
    return prompts, [ids for ids, _groups in individual], local


def capture_states(
    loaded: Any,
    layers: list[Any],
    rows: list[dict[str, Any]],
    entry_layer: int,
) -> dict[str, torch.Tensor]:
    prompts, individual_ids, local = coordinate_data(loaded.tokenizer, rows)
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    positions = {
        role: padded_positions(encoded, individual_ids, local[role]) for role in ROLES
    }
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        value = inputs[0]
        batch_indices = torch.arange(len(rows), device=value.device)
        for role in ROLES:
            pos = positions[role].to(value.device)
            captured[role] = value[batch_indices, pos, :].detach().cpu().clone()

    handle = layers[entry_layer].register_forward_pre_hook(hook)
    encoded = {key: value.to(loaded.input_device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            loaded.model(**encoded, use_cache=False)
    finally:
        handle.remove()
    if set(captured) != set(ROLES):
        raise RuntimeError("Phase572 joint state capture incomplete")
    del encoded
    return captured


def random_states(
    self_states: dict[str, torch.Tensor], rows: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    output = {}
    for role in ROLES:
        vectors = []
        for index, row in enumerate(rows):
            seed = int(
                hashlib.sha256(
                    f"{row['case_id']}|phase572|{role}".encode("utf-8")
                ).hexdigest()[:16],
                16,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            vector = torch.randn(
                self_states[role].shape[-1], generator=generator, dtype=torch.float32
            )
            vector = vector / vector.norm().clamp_min(1e-12)
            vectors.append(vector * self_states[role][index].float().norm())
        output[role] = torch.stack(vectors)
    return output


def load_banks(
    registry: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[list[dict[str, Any]]]]:
    ids = {
        case_id
        for entry in registry["entries"]
        for case_id in (
            entry["receiver_case_id"],
            entry["matched_correct_donor_case_id"],
            entry["wrong_target_donor_case_id"],
        )
    }
    bank = {row["case_id"]: row for row in iter_jsonl(protocol.CASES_PATH) if row["case_id"] in ids}
    if set(bank) != ids or any(row["sealed"] for row in bank.values()):
        raise RuntimeError("Phase572 joint case bank drift")
    by_pair: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in registry["entries"]:
        by_pair[int(entry["pair_index"])].append(entry)
    pair_entries = []
    for pair_index in sorted(by_pair):
        entries = sorted(
            by_pair[pair_index],
            key=lambda row: 0 if row["receiver_phenotype"] == "stable_correct" else 1,
        )
        if len(entries) != 2:
            raise RuntimeError("Phase572 joint pair entry drift")
        pair_entries.append(entries)
    if len(pair_entries) % 4:
        raise RuntimeError("Phase572 candidate pair count must be divisible by four")
    batches = []
    for start in range(0, len(pair_entries), 4):
        group = pair_entries[start:start + 4]
        batches.append([pair[0] for pair in group] + [pair[1] for pair in group])
    return bank, batches


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
    random_vectors: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    prompts, individual_ids, local = coordinate_data(loaded.tokenizer, receivers)
    encoded = loaded.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    prompt_width = int(encoded["input_ids"].shape[1])
    positions = {
        role: padded_positions(encoded, individual_ids, local[role]) for role in ROLES
    }
    output_embeddings = loaded.model.get_output_embeddings()
    target_ids = torch.tensor(
        [row["candidate_token_ids"][row["target"]][0] for row in receivers],
        dtype=torch.long,
        device=output_embeddings.weight.device,
    )
    other_ids = torch.tensor(
        [row["candidate_token_ids"][row["other_relation_target"]][0] for row in receivers],
        dtype=torch.long,
        device=output_embeddings.weight.device,
    )
    handle = None
    if condition != "baseline":
        roles = ROLE_SETS[condition]
        if condition == "self_qfa_entry_restore":
            source = self_states
        elif condition == "wrong_target_query_fact_answer_entry":
            source = wrong_states
        elif condition == "random_query_fact_answer_entry":
            source = random_vectors
        else:
            source = matched_states

        def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            value = inputs[0]
            if value.shape[1] <= max(int(positions[role].max().item()) for role in roles):
                return inputs
            modified = value.clone()
            batch_indices = torch.arange(len(receivers), device=value.device)
            for role in roles:
                pos = positions[role].to(value.device)
                modified[batch_indices, pos, :] = source[role].to(
                    value.device, dtype=value.dtype
                )
            return (modified, *inputs[1:])

        handle = layers[int(frozen["entry_layer"])].register_forward_pre_hook(hook)
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
        results.append(
            {
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
            }
        )
    del encoded, generated, first_scores, margins
    return results


def run(max_new_tokens: int, restart: bool) -> Path:
    frozen = read_json(protocol.PROTOCOL_PATH)
    summary = read_json(behavior.SUMMARY_PATH)
    registry = read_json(behavior.REGISTRY_PATH)
    if not summary["qualified_for_joint_causal"]:
        raise RuntimeError("Phase572 joint causal stage is not behavior-qualified")
    if summary["registry_sha256"] != sha256_file(behavior.REGISTRY_PATH):
        raise RuntimeError("Phase572 donor registry hash drift")
    if restart:
        for path in (ROWS_PATH, SUMMARY_PATH, CONTRACT_PATH):
            path.unlink(missing_ok=True)
    contract = {
        "schema_version": "phase572_joint_causal_contract.v1",
        "phase_id": protocol.PHASE,
        "created_at": now(),
        "model": protocol.MODEL,
        "cases_sha256": sha256_file(protocol.CASES_PATH),
        "protocol_sha256": sha256_file(protocol.PROTOCOL_PATH),
        "registry_sha256": sha256_file(behavior.REGISTRY_PATH),
        "candidate_pair_count": registry["candidate_pair_count"],
        "final_pair_count": registry["final_pair_count"],
        "conditions": frozen["conditions"],
        "fixed_batch_size": frozen["fixed_batch_size"],
        "torch_dtype_requested": "torch.bfloat16",
        "sealed_split_read": False,
    }
    if CONTRACT_PATH.exists():
        existing = read_json(CONTRACT_PATH)
        for key in (
            "model", "cases_sha256", "protocol_sha256", "registry_sha256",
            "candidate_pair_count", "final_pair_count", "conditions",
            "fixed_batch_size", "torch_dtype_requested", "sealed_split_read",
        ):
            if existing[key] != contract[key]:
                raise RuntimeError(f"Phase572 joint causal contract drift: {key}")
    else:
        write_json(CONTRACT_PATH, contract)

    bank, batches = load_banks(registry)
    loaded = None
    all_rows = []
    baseline_valid_pairs: set[int] = set()
    started = time.monotonic()
    try:
        loaded = load_probe_model(protocol.MODEL)
        loaded.tokenizer.padding_side = "left"
        run_dtype = str(next(loaded.model.parameters()).dtype)
        if run_dtype != "torch.bfloat16":
            raise RuntimeError(f"Phase572 requires BF16, got {run_dtype}")
        layers = get_layers(loaded.model)
        for batch_index, entries in enumerate(batches):
            receivers = [bank[row["receiver_case_id"]] for row in entries]
            matched = [bank[row["matched_correct_donor_case_id"]] for row in entries]
            wrong = [bank[row["wrong_target_donor_case_id"]] for row in entries]
            self_states = capture_states(loaded, layers, receivers, int(frozen["entry_layer"]))
            matched_states = capture_states(loaded, layers, matched, int(frozen["entry_layer"]))
            wrong_states = capture_states(loaded, layers, wrong, int(frozen["entry_layer"]))
            random_vectors = random_states(self_states, receivers)
            baseline = run_generation(
                loaded, layers, receivers, entries, "baseline", frozen,
                self_states, matched_states, wrong_states, random_vectors, max_new_tokens,
            )
            for offset in range(4):
                if baseline[offset]["semantic_correct"] and baseline[offset + 4]["relation_confusion"]:
                    baseline_valid_pairs.add(int(baseline[offset]["pair_index"]))
            all_rows.extend(baseline)
            for condition in protocol.CONDITIONS[1:]:
                all_rows.extend(
                    run_generation(
                        loaded, layers, receivers, entries, condition, frozen,
                        self_states, matched_states, wrong_states, random_vectors,
                        max_new_tokens,
                    )
                )
            if batch_index == 0 or batch_index == len(batches) - 1 or batch_index % 8 == 7:
                print(
                    f"[{time.strftime('%H:%M:%S')}] qwen3 Phase572 joint "
                    f"{batch_index + 1}/{len(batches)} paired batches",
                    flush=True,
                )
        ordered_valid = sorted(baseline_valid_pairs)
        if len(ordered_valid) < frozen["final_pair_count"]:
            raise RuntimeError(f"Phase572 has only {len(ordered_valid)} baseline-valid pairs")
        final_pairs = set(ordered_valid[: frozen["final_pair_count"]])
        all_rows = [row for row in all_rows if int(row["pair_index"]) in final_pairs]
        write_jsonl(ROWS_PATH, all_rows)
        output = {
            "schema_version": "phase572_joint_causal_summary.v1",
            "phase_id": protocol.PHASE,
            "created_at": now(),
            "status": "complete",
            "model": protocol.MODEL,
            "torch_dtype": run_dtype,
            "candidate_pair_count": registry["candidate_pair_count"],
            "baseline_valid_pair_count": len(ordered_valid),
            "baseline_drift_pair_count": registry["candidate_pair_count"] - len(ordered_valid),
            "final_pair_count": frozen["final_pair_count"],
            "condition_count": len(protocol.CONDITIONS),
            "causal_row_count": len(all_rows),
            "runtime_seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(ROWS_PATH),
            "full_vectors_persisted": False,
            "head_channel_parameter_neuron_scan_executed": False,
            "sealed_split_read": False,
        }
        write_json(SUMMARY_PATH, output)
        print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)
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
