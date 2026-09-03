#!/usr/bin/env python3
"""C224: one-time Qwen3 behavior and full-token/full-coordinate field capture."""
from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1757_c223_surface_transport_common as common

core = common.core
OUT = common.OUTS["C224"]
PARENT = common.OUTS["C223"]
BATCH = 8


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(PARENT / "audit/independent_final_audit.json")
    rows = core.rows(PARENT / "compiled/qwen3.jsonl")
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"].startswith("C224"),
        "rows": len(rows) == 2304,
        "hidden_rows": sum(row["order"] == 1 for row in rows) == 1152,
        "width": max(len(row["prompt_ids"]) for row in rows) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": 1758,
        "campaign": "C224",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "qwen3_full_coordinate_capture_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "rows": 2304,
        "hidden_rows": 1152,
        "field_shape": [1152, 4, common.WIDTH, common.DIM],
        "role_shape": [1152, 4, len(common.ROLES), common.DIM],
        "saved": "all prompt tokens and all 2560 physical activation coordinates at embedding/q23/q24/q25; role means are derivative convenience data",
        "behavior_policy": "all strata are reported; strata below 0.65 are missing for mechanism claims but do not stop other routes",
        "forbidden": ["attention", "MLP", "weights", "PCA", "selective row deletion", "simultaneous model loading"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_Qwen3_once_then_close_and_release_GPU",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


@torch.inference_mode()
def run() -> None:
    if (OUT / "raw/full_fields.float16.npy").exists():
        raise RuntimeError("capture already exists")
    rows = core.rows(PARENT / "compiled/qwen3.jsonl")
    hidden_rows = [row for row in rows if row["order"] == 1]
    row_number = {row["case_id"]: i for i, row in enumerate(rows)}
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    fields = np.lib.format.open_memmap(OUT / "raw/full_fields.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, common.WIDTH, common.DIM))
    role_states = np.lib.format.open_memmap(OUT / "raw/role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(hidden_rows), 4, len(common.ROLES), common.DIM))
    logits = np.zeros((len(rows), 2), np.float32)
    hidden_index = []
    model = None
    started = time.time()
    try:
        model, tokenizer, device, placement = common.previous.load_bf16("qwen3")
        quant = common.previous.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(hidden_rows), BATCH):
            batch = hidden_rows[start:start + BATCH]
            batch_fields, scores, lengths = common.previous.baseline_full(model, batch, pad, device, width=common.WIDTH)
            fields[start:start + len(batch)] = batch_fields
            role_states[start:start + len(batch)] = common.previous.role_means(batch_fields.astype(np.float32), batch).astype(np.float16)
            for local, row in enumerate(batch):
                logits[row_number[row["case_id"]]] = scores[local]
                hidden_index.append({
                    "hidden_index": start + local,
                    "case_id": row["case_id"],
                    "family": row["family"],
                    "family_kind": row["family_kind"],
                    "surface": row["surface"],
                    "unit": row["unit"],
                    "partition": row["partition"],
                    "factor_a": row["factor_a"],
                    "factor_b": row["factor_b"],
                    "length": int(lengths[local]),
                })
            fields.flush(); role_states.flush()
            if start % (BATCH * 12) == 0 or start + len(batch) == len(hidden_rows):
                print(f"[C224] hidden {start + len(batch)}/{len(hidden_rows)}", flush=True)
        alternate = [row for row in rows if row["order"] == -1]
        for start in range(0, len(alternate), BATCH):
            batch = alternate[start:start + BATCH]
            _discard, scores, _lengths = common.previous.baseline_full(model, batch, pad, device, width=common.WIDTH)
            for local, row in enumerate(batch):
                logits[row_number[row["case_id"]]] = scores[local]
            if start % (BATCH * 24) == 0 or start + len(batch) == len(alternate):
                print(f"[C224] behavior-only {start + len(batch)}/{len(alternate)}", flush=True)
        behavior = []
        for i, row in enumerate(rows):
            prediction = int(logits[i, 1] > logits[i, 0])
            behavior.append({
                "case_id": row["case_id"], "family": row["family"], "family_kind": row["family_kind"],
                "surface": row["surface"], "unit": row["unit"], "partition": row["partition"],
                "factor_a": row["factor_a"], "factor_b": row["factor_b"], "order": row["order"],
                "gold_position": row["gold_position"], "prediction": prediction,
                "score0": float(logits[i, 0]), "score1": float(logits[i, 1]), "correct": prediction == row["gold_position"],
            })
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        core.write_rows(OUT / "raw/behavior_index.jsonl", behavior)
        core.write_rows(OUT / "raw/hidden_index.jsonl", hidden_index)
        metadata = {"placement": placement, "quantization": quant, "elapsed_seconds": time.time() - started, "field_bytes": int(fields.nbytes), "role_bytes": int(role_states.nbytes)}
        core.save(OUT / "raw/run_metadata.json", metadata)
        checks = {
            "behavior_rows": len(behavior) == 2304,
            "hidden_rows": len(hidden_index) == 1152,
            "field_shape": list(fields.shape) == [1152, 4, 128, 2560],
            "role_shape": list(role_states.shape) == [1152, 4, 6, 2560],
            "finite_logits": bool(np.isfinite(logits).all()),
            "bf16": quant["has_bf16_parameters"],
            "unquantized": not quant["has_quantized_modules"],
        }
        core.save(OUT / "audit/internal_run_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks, "metadata": metadata}, indent=2))
    finally:
        common.previous.release(model)
        gc.collect()


def analyze() -> None:
    behavior = core.rows(OUT / "raw/behavior_index.jsonl")
    floor = core.load(PARENT / "protocol/preregistration.json")["behavior_floor"]
    strata = []
    for family in common.FAMILIES:
        for surface in common.SURFACES:
            for partition in ("discovery", "confirmation", "lockbox"):
                selected = [row for row in behavior if row["family"] == family and row["surface"] == surface and row["partition"] == partition]
                accuracy = float(np.mean([row["correct"] for row in selected]))
                strata.append({"family": family, "surface": surface, "partition": partition, "support": len(selected), "accuracy": accuracy, "behavior_eligible": accuracy >= floor})
    global_accuracy = float(np.mean([row["correct"] for row in behavior]))
    by_family = {family: float(np.mean([row["correct"] for row in behavior if row["family"] == family])) for family in common.FAMILIES}
    report = {
        "phase": 1758, "campaign": "C224", "status": "qwen3_full_field_observed",
        "global_accuracy": global_accuracy, "by_family_accuracy": by_family,
        "strata": strata, "eligible_strata": sum(row["behavior_eligible"] for row in strata), "total_strata": len(strata),
        "claim_boundary": "Behavior qualification licenses HiddenState description only inside eligible strata; it does not identify semantic coordinates or mechanisms.",
        "next_authorization": "C225_coordinate_passport_observation_using_all_rows_with_missingness_labels",
    }
    core.save(OUT / "analysis/behavior_and_capture_summary.json", report)
    checks = {"strata": len(strata) == 96, "support": set(row["support"] for row in strata) == {24}, "finite": bool(np.isfinite([row["accuracy"] for row in strata]).all()), "field_exists": (OUT / "raw/full_fields.float16.npy").exists(), "role_exists": (OUT / "raw/role_states.float16.npy").exists()}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"global_accuracy": global_accuracy, "by_family": by_family, "eligible_strata": report["eligible_strata"], "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    headline = core.load(OUT / "analysis/behavior_and_capture_summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "run": core.load(OUT / "audit/internal_run_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": 1758, "campaign": "C224", "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": headline, "next_authorization": headline["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "run": run, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()

