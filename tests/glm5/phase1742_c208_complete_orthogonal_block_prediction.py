#!/usr/bin/env python3
"""C208: complete 32-direction calibration and unseen-direction prediction."""
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
OUT = common.C208
PHASE, CAMPAIGN = 1742, "C208"
BATCH = 8


def random_holdout() -> np.ndarray:
    rng = np.random.default_rng(1742)
    calibration = common.hadamard(32)
    rows = []
    while len(rows) < 8:
        value = np.ones(32, np.float32)
        value[rng.choice(32, size=16, replace=False)] = -1.0
        if any(np.array_equal(value, row) or np.array_equal(value, -row) for row in calibration):
            continue
        if any(np.array_equal(value, row) or np.array_equal(value, -row) for row in rows):
            continue
        rows.append(value)
    return np.stack(rows)


def contract() -> None:
    if (OUT / "protocol/preregistration.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C207 / "audit/independent_final_audit.json")
    rows = core.rows(common.C207 / "compiled/qwen3_anchors.jsonl")
    calibration = common.hadamard(32)
    holdout = random_holdout()
    checks = {
        "authorization": parent["all_checks_passed"],
        "anchors": len(rows) == 18,
        "complete_calibration": calibration.shape == (32, 32) and bool(np.allclose(calibration @ calibration.T, 32 * np.eye(32))),
        "holdout": holdout.shape == (8, 32),
        "holdout_unseen": all(not any(np.array_equal(row, known) or np.array_equal(row, -known) for known in calibration) for row in holdout),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    core.write_rows(OUT / "compiled/qwen3_anchors.jsonl", rows)
    np.save(OUT / "protocol/calibration_patterns.float32.npy", calibration)
    np.save(OUT / "protocol/holdout_patterns.float32.npy", holdout)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_orthogonal_block_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "anchors": 18,
        "calibration_patterns": 32,
        "holdout_patterns": 8,
        "dose": 1.0,
        "odd_model": "per-anchor full cross-source-coordinate linear map fitted only on 32 complete orthogonal calibration directions using actual BF16 writes",
        "even_model": "per-anchor calibration-pattern mean; evaluated separately on unseen directions",
        "targets": "q24/q25 x six registered role views x all 2560 coordinates",
        "gates": {"odd_holdout_nrmse_max": 0.50, "odd_holdout_weighted_sign_min": 0.75, "even_holdout_nrmse_max": 0.75, "fresh_anchor_fraction_min": 0.75},
        "claim_boundary": "a pass identifies a locally callable role-projected response map for this source block and dose, not a semantic algebra, closed full-token system or unique circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "fitting holdout directions"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C208_cuda_then_C209_full_token_closure",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "calibration_shape": [18, 32, 2, 2, 6, common.DIM], "holdout_shape": [18, 8, 2, 2, 6, common.DIM]}, indent=2))


@torch.inference_mode()
def capture() -> None:
    rows = core.rows(OUT / "compiled/qwen3_anchors.jsonl")
    eps_map = core.load(common.C205 / "protocol/anchor_epsilons.json")
    calibration_patterns = np.load(OUT / "protocol/calibration_patterns.float32.npy")
    holdout_patterns = np.load(OUT / "protocol/holdout_patterns.float32.npy")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    calibration = np.lib.format.open_memmap(OUT / "raw/calibration_effects.float16.npy", mode="w+", dtype=np.float16, shape=(18, 32, 2, 2, 6, common.DIM))
    holdout = np.lib.format.open_memmap(OUT / "raw/holdout_effects.float16.npy", mode="w+", dtype=np.float16, shape=(18, 8, 2, 2, 6, common.DIM))
    writes = np.zeros((18, 40, 2, 32), np.float32)
    c206_index = {row["case_id"]: row["case_index"] for row in core.rows(common.C206 / "raw/index.jsonl")}
    c206_base = np.load(common.C206 / "raw/baseline_full.float16.npy", mmap_mode="r")
    model = None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3")
        quant = common.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for case_i, row in enumerate(rows):
            epsilon = float(eps_map[row["case_id"]])
            baseline_full = np.asarray(c206_base[c206_index[row["case_id"]], 2:4], np.float32)[None]
            for kind, patterns, target, offset in (("calibration", calibration_patterns, calibration, 0), ("holdout", holdout_patterns, holdout, 32)):
                for start in range(0, len(patterns), BATCH):
                    block = patterns[start:start + BATCH]
                    repeated = [row] * len(block)
                    doses = np.ones(len(block), np.float32)
                    epsilons = np.full(len(block), epsilon, np.float32)
                    plus, plus_write, _ = common.patched_full(model, repeated, block, doses, epsilons, 1.0, pad, device)
                    minus, minus_write, _ = common.patched_full(model, repeated, block, doses, epsilons, -1.0, pad, device)
                    odd = 0.5 * (plus.astype(np.float32) - minus.astype(np.float32))
                    even = 0.5 * (plus.astype(np.float32) + minus.astype(np.float32)) - baseline_full
                    target[case_i, start:start + len(block), 0] = common.role_means(odd, repeated).astype(np.float16)
                    target[case_i, start:start + len(block), 1] = common.role_means(even, repeated).astype(np.float16)
                    writes[case_i, offset + start:offset + start + len(block), 0] = plus_write
                    writes[case_i, offset + start:offset + start + len(block), 1] = minus_write
            calibration.flush()
            holdout.flush()
            print(f"[C208] {case_i + 1}/18 {row['program']} u{row['unit']}", flush=True)
        np.save(OUT / "raw/actual_writes.float32.npy", writes)
        checks = {"calibration_shape": list(calibration.shape) == [18, 32, 2, 2, 6, common.DIM], "holdout_shape": list(holdout.shape) == [18, 8, 2, 2, 6, common.DIM], "finite": bool(np.isfinite(calibration).all()) and bool(np.isfinite(holdout).all()) and bool(np.isfinite(writes).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/capture.json", {"checks": checks, "runtime": placement})
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        calibration.flush()
        holdout.flush()
        del calibration, holdout
        common.release(model)
        gc.collect()


def analyze() -> None:
    calibration = np.load(OUT / "raw/calibration_effects.float16.npy", mmap_mode="r")
    holdout = np.load(OUT / "raw/holdout_effects.float16.npy", mmap_mode="r")
    writes = np.load(OUT / "raw/actual_writes.float32.npy")
    rows = core.rows(OUT / "compiled/qwen3_anchors.jsonl")
    anchor_rows = []
    pooled_truth = []
    pooled_prediction = []
    pooled_even_truth = []
    pooled_even_prediction = []
    for case_i, row in enumerate(rows):
        x_cal = 0.5 * (writes[case_i, :32, 0] - writes[case_i, :32, 1])
        x_hold = 0.5 * (writes[case_i, 32:, 0] - writes[case_i, 32:, 1])
        y_cal = np.asarray(calibration[case_i, :, 0], np.float32).reshape(32, -1)
        y_hold = np.asarray(holdout[case_i, :, 0], np.float32).reshape(8, -1)
        coefficient = np.linalg.lstsq(x_cal.astype(np.float64), y_cal.astype(np.float64), rcond=None)[0]
        prediction = (x_hold.astype(np.float64) @ coefficient).astype(np.float32)
        even_cal = np.asarray(calibration[case_i, :, 1], np.float32).reshape(32, -1)
        even_hold = np.asarray(holdout[case_i, :, 1], np.float32).reshape(8, -1)
        even_prediction = np.broadcast_to(even_cal.mean(axis=0, keepdims=True), even_hold.shape)
        odd_metrics = {"nrmse": common.nrmse(prediction, y_hold), "weighted_sign": common.weighted_sign(prediction, y_hold)}
        even_metrics = {"nrmse": common.nrmse(even_prediction, even_hold), "weighted_sign": common.weighted_sign(even_prediction, even_hold)}
        anchor_rows.append({"case_id": row["case_id"], "program": row["program"], "unit": row["unit"], "odd": odd_metrics, "even": even_metrics, "write_rank": int(np.linalg.matrix_rank(x_cal.astype(np.float64)))})
        pooled_truth.append(y_hold)
        pooled_prediction.append(prediction)
        pooled_even_truth.append(even_hold)
        pooled_even_prediction.append(even_prediction)
    odd_truth = np.concatenate(pooled_truth)
    odd_prediction = np.concatenate(pooled_prediction)
    even_truth = np.concatenate(pooled_even_truth)
    even_prediction = np.concatenate(pooled_even_prediction)
    pooled = {"odd": {"nrmse": common.nrmse(odd_prediction, odd_truth), "weighted_sign": common.weighted_sign(odd_prediction, odd_truth)}, "even": {"nrmse": common.nrmse(even_prediction, even_truth), "weighted_sign": common.weighted_sign(even_prediction, even_truth)}}
    gates = core.load(OUT / "protocol/preregistration.json")["gates"]
    fresh_rows = [row for row in anchor_rows if row["unit"] == 6]
    fresh_fraction = float(np.mean([row["odd"]["nrmse"] <= gates["odd_holdout_nrmse_max"] and row["odd"]["weighted_sign"] >= gates["odd_holdout_weighted_sign_min"] and row["even"]["nrmse"] <= gates["even_holdout_nrmse_max"] for row in fresh_rows]))
    passed = pooled["odd"]["nrmse"] <= gates["odd_holdout_nrmse_max"] and pooled["odd"]["weighted_sign"] >= gates["odd_holdout_weighted_sign_min"] and pooled["even"]["nrmse"] <= gates["even_holdout_nrmse_max"] and fresh_fraction >= gates["fresh_anchor_fraction_min"]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "complete_orthogonal_block_adjudicated", "pooled": pooled, "fresh_anchor_pass_fraction": fresh_fraction, "anchor_rows": anchor_rows, "predictive_gate_passed": passed, "interpretation": "The odd map is calibrated from a complete physical source-coordinate basis and tested on unseen joint directions. Its scope is per-anchor and role-projected. The even component receives a separately judged constant baseline model.", "next_authorization": "C209_omitted_token_closure_comparison"}
    core.save(OUT / "analysis/orthogonal_prediction.json", report)
    checks = {"anchors": len(anchor_rows) == 18, "full_write_rank": all(row["write_rank"] == 32 for row in anchor_rows), "finite": bool(np.isfinite([row[component][key] for row in anchor_rows for component in ("odd", "even") for key in ("nrmse", "weighted_sign")]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"pooled": pooled, "fresh_anchor_pass_fraction": fresh_fraction, "predictive_gate_passed": passed, "checks": checks}, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/orthogonal_prediction.json")
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
