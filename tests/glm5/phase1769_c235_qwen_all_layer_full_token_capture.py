#!/usr/bin/env python3
"""C235: Qwen3 behavior plus all-layer/all-token/all-coordinate capture."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1768_c234_event_campaign_common as common

core = common.core
OUT = common.OUTS["C235"]
PARENT = common.OUTS["C234"]
BATCH = 2


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(PARENT / "audit/independent_final_audit.json")
    rows = core.rows(PARENT / "compiled/qwen3.jsonl")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C235"),
        "rows": len(rows) == 640,
        "both_orders": {row["order"] for row in rows} == {1, -1},
        "width": max(len(row["prompt_ids"]) for row in rows) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1769,
        "campaign": "C235",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "qwen3_all_layer_full_token_capture_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "rows": 640,
        "batch": BATCH,
        "field_axes": ["case", "checkpoint", "token", "physical_coordinate"],
        "field_shape_expected": [640, 37, common.WIDTH, common.DIM],
        "storage": "float16 loss-limited archive; real token lengths and token ids saved; padded states zeroed",
        "free_generation_rows": 80,
        "numerical_repeat_rows": 2,
        "forbidden": ["attention", "MLP", "weights", "PCA", "Top-K", "row deletion"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_Qwen3_once_then_release_GPU_and_analyze_C235",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


def batch_inputs(rows: list[dict], pad: int, device):
    ids = torch.full((len(rows), common.WIDTH), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(rows):
        values = row["prompt_ids"]
        lengths.append(len(values))
        ids[i, :len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[i, :len(values)] = 1
        positions[i, :len(values)] = torch.arange(len(values), device=device)
    return ids, mask, positions, lengths


@torch.inference_mode()
def numerical_repeat(model, rows: list[dict], pad: int, device) -> dict:
    states = []
    quantization = []
    for _ in range(2):
        ids, mask, positions, lengths = batch_inputs(rows, pad, device)
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
        packed = torch.stack([value[:, :max(lengths)].float().cpu() for value in output.hidden_states], dim=1).numpy()
        states.append(packed)
        quantization.append(np.abs(packed - packed.astype(np.float16).astype(np.float32)).reshape(-1))
        del output, ids, mask, positions, packed
    delta = np.abs(states[1] - states[0])
    quant = np.concatenate(quantization)
    return {
        "repeat_max_abs": float(delta.max()),
        "repeat_q999_abs": float(np.quantile(delta, 0.999)),
        "float16_roundtrip_max_abs": float(quant.max()),
        "float16_roundtrip_q999_abs": float(np.quantile(quant, 0.999)),
    }


@torch.inference_mode()
def run() -> None:
    if (OUT / "raw/full_fields.float16.npy").exists():
        raise RuntimeError("capture already exists")
    rows = core.rows(PARENT / "compiled/qwen3.jsonl")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(640, 37, common.WIDTH, common.DIM))
    token_ids = np.zeros((640, common.WIDTH), np.int32)
    token_mask = np.zeros((640, common.WIDTH), bool)
    logits = np.zeros((640, 2), np.float32)
    hidden_index = []
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        quant_audit = common.previous.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        repeat = numerical_repeat(model, rows[:2], pad, device)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            ids, mask, positions, lengths = batch_inputs(batch, pad, device)
            output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True, output_hidden_states=True)
            if len(output.hidden_states) != 37:
                raise RuntimeError({"hidden_states": len(output.hidden_states)})
            for local, row in enumerate(batch):
                length = lengths[local]
                token_ids[start + local, :length] = np.asarray(row["prompt_ids"], np.int32)
                token_mask[start + local, :length] = True
                for checkpoint, state in enumerate(output.hidden_states):
                    fields[start + local, checkpoint, :length] = state[local, :length].float().cpu().numpy().astype(np.float16)
                logits[start + local] = [float(output.logits[local, length - 1, candidate[0]]) for candidate in row["candidate_ids"]]
                hidden_index.append({
                    "hidden_index": start + local,
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "surface": row["surface"],
                    "partition": row["partition"],
                    "unit": row["unit"],
                    "factor_a": row["factor_a"],
                    "factor_b": row["factor_b"],
                    "order": row["order"],
                    "length": length,
                    "role_positions": row["role_positions"],
                })
            del output, ids, mask, positions
            if start % 40 == 0 or start + len(batch) == len(rows):
                fields.flush()
                print(f"[C235] full fields {start + len(batch)}/{len(rows)}", flush=True)
        fields.flush()
        np.save(OUT / "raw/token_ids.int32.npy", token_ids)
        np.save(OUT / "raw/token_mask.bool.npy", token_mask)
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        behavior = []
        for i, row in enumerate(rows):
            prediction = int(logits[i, 1] > logits[i, 0])
            behavior.append({
                "case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "partition": row["partition"],
                "unit": row["unit"], "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"],
                "gold_position": row["gold_position"], "prediction": prediction, "correct": prediction == row["gold_position"],
                "score0": float(logits[i, 0]), "score1": float(logits[i, 1]),
            })
        core.write_rows(OUT / "raw/behavior_index.jsonl", behavior)
        core.write_rows(OUT / "raw/hidden_index.jsonl", hidden_index)

        free_rows = [row for row in rows if row["order"] == 1 and row["factor_a"] == 0 and row["factor_b"] == 0]
        free_results = []
        for i, row in enumerate(free_rows):
            values = torch.tensor([row["free_prompt_ids"]], dtype=torch.long, device=device)
            generated = model.generate(values, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=pad)
            answer = tokenizer.decode(generated[0, values.shape[1]:], skip_special_tokens=True).strip()
            expected = row["correct_answer"].strip()
            normalized = " ".join(answer.lower().replace(".", " ").replace(",", " ").split())
            correct = normalized == expected.lower() or normalized.startswith(expected.lower() + " ")
            free_results.append({"case_id": row["case_id"], "family": row["family"], "surface": row["surface"], "partition": row["partition"], "expected": expected, "generated": answer, "correct": correct})
            if i % 20 == 0 or i + 1 == len(free_rows):
                print(f"[C235] free generation {i + 1}/{len(free_rows)}", flush=True)
        core.write_rows(OUT / "raw/free_generation.jsonl", free_results)
        metadata = {
            "placement": placement,
            "quantization": quant_audit,
            "numerical_repeat": repeat,
            "elapsed_seconds": time.time() - started,
            "field_bytes": int(fields.nbytes),
        }
        core.save(OUT / "raw/run_metadata.json", metadata)
        checks = {
            "field_shape": list(fields.shape) == [640, 37, 128, 2560],
            "hidden_rows": len(hidden_index) == 640,
            "behavior_rows": len(behavior) == 640,
            "free_rows": len(free_results) == 80,
            "token_mask": bool(np.all(token_mask.sum(axis=1) == np.asarray([row["length"] for row in hidden_index]))),
            "finite_logits": bool(np.isfinite(logits).all()),
            "bf16": quant_audit["has_bf16_parameters"],
            "unquantized": not quant_audit["has_quantized_modules"],
        }
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks, "metadata": metadata}, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


def analyze() -> None:
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    free = core.rows(OUT / "raw/free_generation.jsonl")
    strata = []
    for family in common.FAMILIES:
        for partition in common.PARTITIONS:
            selected = [row for row in behavior if row["family"] == family and row["partition"] == partition]
            accuracy = float(np.mean([row["correct"] for row in selected]))
            strata.append({"family": family, "partition": partition, "support": len(selected), "accuracy": accuracy, "eligible": accuracy >= 0.65})
    report = {
        "phase": 1769,
        "campaign": "C235",
        "status": "qwen3_all_layer_full_token_field_observed",
        "global_accuracy": float(np.mean([row["correct"] for row in behavior])),
        "by_family_accuracy": {family: float(np.mean([row["correct"] for row in behavior if row["family"] == family])) for family in common.FAMILIES},
        "by_partition_accuracy": {partition: float(np.mean([row["correct"] for row in behavior if row["partition"] == partition])) for partition in common.PARTITIONS},
        "free_generation_accuracy": float(np.mean([row["correct"] for row in free])),
        "strata": strata,
        "eligible_strata": sum(row["eligible"] for row in strata),
        "total_strata": len(strata),
        "numerical_repeat": core.load(OUT / "raw/run_metadata.json")["numerical_repeat"],
        "claim_boundary": "This is a behavior-qualified all-layer HiddenState observation. It does not identify a circuit or semantic coordinate.",
        "next_authorization": "C236_full_coordinate_interval_event_extraction",
    }
    core.save(OUT / "analysis/behavior_capture_summary.json", report)
    checks = {"strata": len(strata) == 20, "support": {row["partition"]: row["support"] for row in strata[:4]} == {"discovery": 64, "confirmation": 24, "lockbox": 16, "fresh": 24}, "fields": (OUT / "raw/full_fields.float16.npy").exists(), "finite": bool(np.isfinite([row["accuracy"] for row in strata]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"report": report, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/behavior_capture_summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1769, "campaign": "C235", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()
