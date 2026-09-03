#!/usr/bin/env python3
"""C206: capture full-token multi-dose odd/even response regimes on Qwen3."""
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
OUT = common.C206
PHASE, CAMPAIGN = 1740, "C206"
BATCH = 6


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(common.C205 / "audit/independent_final_audit.json")
    rows = core.rows(common.C205 / "compiled/qwen3_anchors.jsonl")
    checks = {
        "authorization": parent["all_checks_passed"],
        "rows": len(rows) == 36,
        "programs": len({row["program"] for row in rows}) == 9,
        "width": max(len(row["prompt_ids"]) for row in rows) <= common.WIDTH,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "compiled/qwen3_anchors.jsonl", rows)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "full_sequence_multidose_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized",
        "cases": 36,
        "programs": 9,
        "doses": list(common.DOSES),
        "direction": "all-plus joint direction over the 32 frozen q23 relation-role coordinates",
        "per_coordinate_delta": "dose * anchor_epsilon / sqrt(32)",
        "saved": "all tokens x all 2560 coordinates at embedding/q23/q24/q25 baseline and q24/q25 odd/even response",
        "partitions": {"discovery": [1, 2], "confirmation": [5], "fresh": [6]},
        "registered_descriptors": ["numerical_zero", "sign_stable", "near_proportional", "even_dominant", "transition", "saturation", "amplification"],
        "descriptor_thresholds": {"sign_stable_min": 0.75, "near_proportional_relative_band": [0.8, 1.2], "even_dominant_ratio": 1.0, "transition_sign_below": 0.65, "gain_shift_fraction": 0.5, "saturation_fraction": 0.7, "amplification_fraction": 1.3},
        "regime_consistency_gate": {"confirmation_fresh_sign_min": 0.75, "normalized_gain_relative_spread_max": 0.35},
        "claim_boundary": "finite response of one joint coordinate direction; descriptors are empirical regimes, not semantic modules or a unique circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "dropping tokens after reveal"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_C206_cuda_then_C207_same_dose_separation",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "raw_effect_shape": [36, 6, 2, 2, common.WIDTH, common.DIM]}, indent=2))


@torch.inference_mode()
def capture() -> None:
    rows = core.rows(OUT / "compiled/qwen3_anchors.jsonl")
    eps_map = core.load(common.C205 / "protocol/anchor_epsilons.json")
    epsilons = np.asarray([eps_map[row["case_id"]] for row in rows], np.float32)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    baseline = np.lib.format.open_memmap(OUT / "raw/baseline_full.float16.npy", mode="w+", dtype=np.float16, shape=(36, 4, common.WIDTH, common.DIM))
    effects = np.lib.format.open_memmap(OUT / "raw/joint_effects.float16.npy", mode="w+", dtype=np.float16, shape=(36, len(common.DOSES), 2, 2, common.WIDTH, common.DIM))
    writes = np.zeros((36, len(common.DOSES), 2, 32), np.float32)
    logits = np.zeros((36, 2), np.float32)
    lengths = np.zeros(36, np.int32)
    repeat_hidden = 0.0
    repeat_logits = 0.0
    model = None
    try:
        model, tokenizer, device, placement = common.load_bf16("qwen3")
        quant = common.quantization_audit(model)
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            stop = start + len(batch)
            base, scores, batch_lengths = common.baseline_full(model, batch, pad, device)
            baseline[start:stop] = base
            logits[start:stop] = scores
            lengths[start:stop] = batch_lengths
            if start == 0:
                again, scores_again, _ = common.baseline_full(model, batch, pad, device)
                repeat_hidden = float(np.max(np.abs(again.astype(np.float32) - base.astype(np.float32))))
                repeat_logits = float(np.max(np.abs(scores_again - scores)))
            patterns = np.ones((len(batch), 32), np.float32)
            batch_eps = epsilons[start:stop]
            for dose_i, dose in enumerate(common.DOSES):
                dose_values = np.full(len(batch), dose, np.float32)
                plus, plus_write, _ = common.patched_full(model, batch, patterns, dose_values, batch_eps, 1.0, pad, device)
                minus, minus_write, _ = common.patched_full(model, batch, patterns, dose_values, batch_eps, -1.0, pad, device)
                base_targets = base[:, 2:4].astype(np.float32)
                effects[start:stop, dose_i, 0] = (0.5 * (plus.astype(np.float32) - minus.astype(np.float32))).astype(np.float16)
                effects[start:stop, dose_i, 1] = (0.5 * (plus.astype(np.float32) + minus.astype(np.float32)) - base_targets).astype(np.float16)
                writes[start:stop, dose_i, 0] = plus_write
                writes[start:stop, dose_i, 1] = minus_write
            baseline.flush()
            effects.flush()
            print(f"[C206] {stop}/36 full-token dose fields", flush=True)
        np.save(OUT / "raw/actual_writes.float32.npy", writes)
        np.save(OUT / "raw/behavior_logits.float32.npy", logits)
        core.write_rows(OUT / "raw/index.jsonl", [{"case_index": i, "case_id": row["case_id"], "program": row["program"], "unit": row["unit"], "partition": row["partition"], "length": int(lengths[i]), "epsilon": float(epsilons[i])} for i, row in enumerate(rows)])
        checks = {
            "baseline_shape": list(baseline.shape) == [36, 4, common.WIDTH, common.DIM],
            "effect_shape": list(effects.shape) == [36, 6, 2, 2, common.WIDTH, common.DIM],
            "finite": bool(np.isfinite(baseline).all()) and bool(np.isfinite(effects).all()) and bool(np.isfinite(writes).all()),
            "bf16": quant["has_bf16_parameters"],
            "unquantized": not quant["has_quantized_modules"],
            "repeat_hidden_exact": repeat_hidden == 0.0,
            "repeat_logits_exact": repeat_logits == 0.0,
        }
        core.save(OUT / "analysis/capture.json", {"checks": checks, "repeat_hidden_max_abs": repeat_hidden, "repeat_logit_max_abs": repeat_logits, "runtime": placement})
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks, "repeat_hidden": repeat_hidden, "repeat_logits": repeat_logits}, indent=2))
    finally:
        baseline.flush()
        effects.flush()
        del baseline, effects
        common.release(model)
        gc.collect()


def masked(values: np.ndarray, indices: list[int], index_rows: list[dict]) -> np.ndarray:
    chunks = []
    for case_i in indices:
        chunks.append(np.asarray(values[case_i, :, : index_rows[case_i]["length"]], np.float32).reshape(-1))
    return np.concatenate(chunks) if chunks else np.empty(0, np.float32)


def partition_metrics(effects: np.ndarray, index_rows: list[dict], units: set[int]) -> list[dict]:
    selected = [row["case_index"] for row in index_rows if row["unit"] in units]
    reference = masked(effects[:, 3, 0], selected, index_rows)
    reference_rms = float(np.sqrt(np.mean(np.square(reference, dtype=np.float64))))
    rows = []
    for dose_i, dose in enumerate(common.DOSES):
        odd = masked(effects[:, dose_i, 0], selected, index_rows)
        even = masked(effects[:, dose_i, 1], selected, index_rows)
        odd_rms = float(np.sqrt(np.mean(np.square(odd, dtype=np.float64))))
        even_rms = float(np.sqrt(np.mean(np.square(even, dtype=np.float64))))
        rows.append({
            "dose": dose,
            "odd_rms": odd_rms,
            "even_rms": even_rms,
            "even_to_odd_rms": even_rms / max(odd_rms, 1e-30),
            "weighted_sign_vs_dose1": common.weighted_sign(odd, reference),
            "normalized_gain": odd_rms / max(float(dose), 1e-30),
            "gain_relative_to_dose1": (odd_rms / max(float(dose), 1e-30)) / max(reference_rms, 1e-30),
        })
    return rows


def describe(rows: list[dict], thresholds: dict) -> list[str]:
    labels = []
    if max(row["odd_rms"] for row in rows) <= 1e-7:
        labels.append("numerical_zero")
    if min(row["weighted_sign_vs_dose1"] for row in rows[1:]) >= thresholds["sign_stable_min"]:
        labels.append("sign_stable")
    if all(thresholds["near_proportional_relative_band"][0] <= row["gain_relative_to_dose1"] <= thresholds["near_proportional_relative_band"][1] for row in rows[1:]):
        labels.append("near_proportional")
    if any(row["even_to_odd_rms"] > thresholds["even_dominant_ratio"] for row in rows):
        labels.append("even_dominant")
    weakest_sign = min(row["weighted_sign_vs_dose1"] for row in rows[1:])
    gains = [row["gain_relative_to_dose1"] for row in rows]
    if weakest_sign < thresholds["transition_sign_below"] or max(gains) - min(gains) > thresholds["gain_shift_fraction"]:
        labels.append("transition")
    if rows[-1]["gain_relative_to_dose1"] < thresholds["saturation_fraction"]:
        labels.append("saturation")
    if rows[-1]["gain_relative_to_dose1"] > thresholds["amplification_fraction"]:
        labels.append("amplification")
    return labels


def analyze() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    effects = np.load(OUT / "raw/joint_effects.float16.npy", mmap_mode="r")
    writes = np.load(OUT / "raw/actual_writes.float32.npy")
    index_rows = core.rows(OUT / "raw/index.jsonl")
    discovery = partition_metrics(effects, index_rows, {1, 2})
    confirmation = partition_metrics(effects, index_rows, {5})
    fresh = partition_metrics(effects, index_rows, {6})
    thresholds = protocol["descriptor_thresholds"]
    descriptors = {"discovery": describe(discovery, thresholds), "confirmation": describe(confirmation, thresholds), "fresh": describe(fresh, thresholds)}
    per_coordinate_requested = np.asarray([[dose * row["epsilon"] / np.sqrt(32.0) for dose in common.DOSES] for row in index_rows], np.float64)
    plus_ratio = np.abs(writes[:, :, 0]) / np.maximum(per_coordinate_requested[:, :, None], 1e-30)
    minus_ratio = np.abs(writes[:, :, 1]) / np.maximum(per_coordinate_requested[:, :, None], 1e-30)
    write_summary = {
        "zero_fraction": float(np.mean(writes == 0)),
        "median_abs_actual_to_requested": float(np.median(np.concatenate([plus_ratio.reshape(-1), minus_ratio.reshape(-1)]))),
        "sign_error_fraction": float(np.mean(np.sign(writes[:, :, 0]) <= 0) + np.mean(np.sign(writes[:, :, 1]) >= 0)) / 2.0,
    }
    gate = protocol["regime_consistency_gate"]
    confirmation_fresh_sign = min(min(row["weighted_sign_vs_dose1"] for row in confirmation[1:]), min(row["weighted_sign_vs_dose1"] for row in fresh[1:]))
    spread = max(abs(confirmation[i]["gain_relative_to_dose1"] - fresh[i]["gain_relative_to_dose1"]) for i in range(len(common.DOSES)))
    passed = confirmation_fresh_sign >= gate["confirmation_fresh_sign_min"] and spread <= gate["normalized_gain_relative_spread_max"]
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "full_sequence_response_regimes_observed",
        "write_summary": write_summary,
        "partition_rows": {"discovery": discovery, "confirmation": confirmation, "fresh": fresh},
        "descriptors": descriptors,
        "regime_consistency": {"confirmation_fresh_sign_min": confirmation_fresh_sign, "normalized_gain_relative_spread_max": spread, "passed": passed},
        "interpretation": "The all-plus joint direction has an empirical full-token finite-response regime. Agreement across partitions is a reproducibility result, not a claim that the direction is semantic or that the six roles are closed.",
        "next_authorization": "C207_same_per_coordinate_dose_single_joint_separation",
    }
    core.save(OUT / "analysis/response_regimes.json", report)
    checks = {"six_doses": all(len(rows) == 6 for rows in (discovery, confirmation, fresh)), "finite": bool(np.isfinite([[row[key] for key in ("odd_rms", "even_rms", "weighted_sign_vs_dose1", "gain_relative_to_dose1")] for rows in (discovery, confirmation, fresh) for row in rows]).all()), "writes": 0.8 <= write_summary["median_abs_actual_to_requested"] <= 1.2}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/response_regimes.json")
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
