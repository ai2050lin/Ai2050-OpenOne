#!/usr/bin/env python3
"""C207: separate per-coordinate finite-dose response from joint-coordinate coupling."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import phase1739_c205_response_ecology_common as common

core = common.core
OUT = common.C207
PHASE, CAMPAIGN = 1741, "C207"
BATCH = 8
DOSE_INDEX = list(common.DOSES).index(1.0)


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C206 / "audit/independent_final_audit.json")
    all_rows = core.rows(common.C206 / "compiled/qwen3_anchors.jsonl")
    selected = [row for row in all_rows if row["unit"] in (1, 6)]
    checks = {"authorization": parent["all_checks_passed"], "anchors": len(selected) == 18, "programs": len({row["program"] for row in selected}) == 9, "two_partitions": {row["unit"] for row in selected} == {1, 6}}
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "compiled/qwen3_anchors.jsonl", selected)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "single_joint_same_dose_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "anchors": 18,
        "source_coordinates": 32,
        "dose": 1.0,
        "single_coordinate_delta": "anchor_epsilon/sqrt(32), exactly matching each active coordinate in C206's 32-coordinate joint dose-1 stimulus",
        "comparison": "actual joint odd/even full-token effect versus sum of 32 actual single-coordinate odd/even effects",
        "gates": {"odd_additive_nrmse_max": 0.50, "odd_weighted_sign_min": 0.75, "even_additive_nrmse_max": 0.75},
        "claim_boundary": "failure identifies non-additivity at this finite BF16 dose; it does not identify the source as semantic interaction or a unique circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "using C195 larger single-coordinate derivatives"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C207_cuda_then_C208_complete_orthogonal_block",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "sum_shape": [18, 2, 2, common.WIDTH, common.DIM]}, indent=2))


@torch.inference_mode()
def capture() -> None:
    rows = core.rows(OUT / "compiled/qwen3_anchors.jsonl")
    eps_map = core.load(common.C205 / "protocol/anchor_epsilons.json")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    summed = np.lib.format.open_memmap(OUT / "raw/summed_single_effects.float16.npy", mode="w+", dtype=np.float16, shape=(18, 2, 2, common.WIDTH, common.DIM))
    role_energy = np.zeros((18, 32, 2, 2, len(common.ROLES)), np.float64)
    writes = np.zeros((18, 32, 2), np.float32)
    model = None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3")
        quant = common.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        identity = np.eye(32, dtype=np.float32)
        for case_i, row in enumerate(rows):
            epsilon = float(eps_map[row["case_id"]])
            accum_odd = np.zeros((2, common.WIDTH, common.DIM), np.float32)
            accum_even = np.zeros_like(accum_odd)
            baseline_index = next(item["case_index"] for item in core.rows(common.C206 / "raw/index.jsonl") if item["case_id"] == row["case_id"])
            baseline = np.asarray(np.load(common.C206 / "raw/baseline_full.float16.npy", mmap_mode="r")[baseline_index, 2:4], np.float32)
            for start in range(0, 32, BATCH):
                patterns = identity[start:start + BATCH]
                repeated = [row] * len(patterns)
                doses = np.ones(len(patterns), np.float32)
                epsilons = np.full(len(patterns), epsilon, np.float32)
                plus, plus_write, _ = common.patched_full(model, repeated, patterns, doses, epsilons, 1.0, pad, device)
                minus, minus_write, _ = common.patched_full(model, repeated, patterns, doses, epsilons, -1.0, pad, device)
                odd = 0.5 * (plus.astype(np.float32) - minus.astype(np.float32))
                even = 0.5 * (plus.astype(np.float32) + minus.astype(np.float32)) - baseline[None]
                accum_odd += odd.sum(axis=0)
                accum_even += even.sum(axis=0)
                role_odd = common.role_means(odd, repeated)
                role_even = common.role_means(even, repeated)
                role_energy[case_i, start:start + len(patterns), 0] = np.square(role_odd, dtype=np.float64).sum(axis=-1)
                role_energy[case_i, start:start + len(patterns), 1] = np.square(role_even, dtype=np.float64).sum(axis=-1)
                for local in range(len(patterns)):
                    source_i = start + local
                    writes[case_i, source_i, 0] = plus_write[local, source_i]
                    writes[case_i, source_i, 1] = minus_write[local, source_i]
            summed[case_i, 0] = accum_odd.astype(np.float16)
            summed[case_i, 1] = accum_even.astype(np.float16)
            summed.flush()
            print(f"[C207] {case_i + 1}/18 {row['program']} u{row['unit']}", flush=True)
        np.save(OUT / "raw/single_role_energy.float64.npy", role_energy)
        np.save(OUT / "raw/actual_single_writes.float32.npy", writes)
        checks = {"shape": list(summed.shape) == [18, 2, 2, common.WIDTH, common.DIM], "finite": bool(np.isfinite(summed).all()) and bool(np.isfinite(role_energy).all()) and bool(np.isfinite(writes).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/capture.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        summed.flush()
        del summed
        common.release(model)
        gc.collect()


def analyze() -> None:
    rows = core.rows(OUT / "compiled/qwen3_anchors.jsonl")
    c206_index = core.rows(common.C206 / "raw/index.jsonl")
    c206_by_case = {row["case_id"]: row["case_index"] for row in c206_index}
    c206 = np.load(common.C206 / "raw/joint_effects.float16.npy", mmap_mode="r")
    summed = np.load(OUT / "raw/summed_single_effects.float16.npy", mmap_mode="r")
    metrics = {}
    for label, units in (("discovery", {1}), ("fresh", {6}), ("pooled", {1, 6})):
        truth_chunks = {0: [], 1: []}
        pred_chunks = {0: [], 1: []}
        for local_i, row in enumerate(rows):
            if row["unit"] not in units:
                continue
            source_i = c206_by_case[row["case_id"]]
            length = c206_index[source_i]["length"]
            for component in (0, 1):
                truth_chunks[component].append(np.asarray(c206[source_i, DOSE_INDEX, component, :, :length], np.float32).reshape(-1))
                pred_chunks[component].append(np.asarray(summed[local_i, component, :, :length], np.float32).reshape(-1))
        metrics[label] = {}
        for component, name in ((0, "odd"), (1, "even")):
            truth = np.concatenate(truth_chunks[component])
            prediction = np.concatenate(pred_chunks[component])
            interaction = truth - prediction
            metrics[label][name] = {"additive_nrmse": common.nrmse(prediction, truth), "weighted_sign": common.weighted_sign(prediction, truth), "interaction_to_joint_rms": float(np.sqrt(np.square(interaction, dtype=np.float64).sum() / max(np.square(truth, dtype=np.float64).sum(), 1e-30)))}
    gates = core.load(OUT / "protocol/preregistration.json")["gates"]
    fresh = metrics["fresh"]
    passed = fresh["odd"]["additive_nrmse"] <= gates["odd_additive_nrmse_max"] and fresh["odd"]["weighted_sign"] >= gates["odd_weighted_sign_min"] and fresh["even"]["additive_nrmse"] <= gates["even_additive_nrmse_max"]
    writes = np.load(OUT / "raw/actual_single_writes.float32.npy")
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "same_dose_single_joint_separated", "metrics": metrics, "single_write_zero_fraction": float(np.mean(writes == 0)), "additive_gate_passed": passed, "interpretation": "The comparison holds each active coordinate's requested dose fixed. Failure means that the 32-coordinate joint response is not the sum of separately measured finite-dose responses under this intervention contract; it does not identify which omitted interactions cause the difference.", "next_authorization": "C208_complete_orthogonal_block_and_unseen_direction_prediction"}
    core.save(OUT / "analysis/single_joint_separation.json", report)
    checks = {"three_splits": len(metrics) == 3, "odd_even": all(set(value) == {"odd", "even"} for value in metrics.values()), "finite": bool(np.isfinite([value[key] for split in metrics.values() for value in split.values() for key in ("additive_nrmse", "weighted_sign", "interaction_to_joint_rms")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/single_joint_separation.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "producer_hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "capture", "analyze", "close"))
    args = parser.parse_args()
    {"contract": contract, "capture": capture, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__":
    main()

