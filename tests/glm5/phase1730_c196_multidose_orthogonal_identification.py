#!/usr/bin/env python3
"""C196: multi-dose orthogonal validation of the C195 signed local response operator."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1730_c196_multidose_orthogonal_identification"
C195 = RESULT / "phase1729_c195_signed_role_checkpoint_trajectory"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1726_c192_multi_program_response_equivalence as c192

PHASE, CAMPAIGN = 1730, "C196"
DIM, WIDTH = 2560, 224
DOSES = (0.25, 0.5, 1.0)
ROLES = c192.ROLES


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def hadamard(size: int) -> np.ndarray:
    value = np.ones((1, 1), np.float32)
    while value.shape[0] < size:
        value = np.block([[value, value], [value, -value]])
    return value


def contract():
    if (OUT / "protocol/preregistration.json").exists():
        raise RuntimeError(OUT)
    parent = core.load(C195 / "audit/independent_final_audit.json")
    index = core.rows(C195 / "raw/index.jsonl"); anchors = core.rows(C195 / "compiled/anchors.jsonl")
    selected_index = [r["anchor_index"] for r in index if r["unit"] == 1 and r["phrase_variant"] == 0 and r["program"] in ("direct_target", "forward_endpoint")]
    patterns = hadamard(64)[1:17]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C196_multi_dose_orthogonal_system_identification",
        "anchors": len(selected_index) == 14, "families": len({index[i]["family"] for i in selected_index}) == 7,
        "programs": {index[i]["program"] for i in selected_index} == {"direct_target", "forward_endpoint"},
        "patterns": patterns.shape == (16, 64) and bool(np.allclose(patterns @ patterns.T, 64 * np.eye(16))),
    }
    if not all(checks.values()): raise RuntimeError(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    core.write_rows(OUT / "compiled/anchors.jsonl", [anchors[i] for i in selected_index])
    np.save(OUT / "protocol/hadamard_patterns.float32.npy", patterns)
    core.save(OUT / "protocol/source_anchor_indices.json", {"indices": selected_index})
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "multidose_orthogonal_identification_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized", "anchors": 14, "patterns": 16, "doses": list(DOSES),
        "stimulus": "16 mutually orthogonal signed patterns across all frozen 64 q23 relation-role source coordinates",
        "prediction": "linear superposition of C195 one-coordinate signed derivatives predicts complete q24/q25 x six-role x 2560-coordinate response",
        "normalization": "each coordinate delta is dose times anchor epsilon divided by sqrt(64)",
        "gates": {"dose_0.25_nrmse_max": 0.25, "dose_0.5_nrmse_max": 0.35, "dose_1.0_nrmse_max": 0.50, "weighted_sign_min": 0.75},
        "nonlinearity": "symmetric even response is recorded separately and cannot be folded into the linear pass label",
        "claim_boundary": "local multi-coordinate finite-difference predictability; does not identify a semantic algebra or unique circuit",
        "forbidden": ["attention", "MLP", "weights", "PCA", "gate changes", "fitting on the orthogonal reveal"],
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_C196_cuda_then_C197_structure_model_tournament",
    }
    core.save(OUT / "protocol/preregistration.json", protocol); core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "raw_shape": [14, 3, 16, 2, 6, DIM]}, indent=2))


@torch.inference_mode()
def capture():
    rows = core.rows(OUT / "compiled/anchors.jsonl"); source_indices = core.load(OUT / "protocol/source_anchor_indices.json")["indices"]
    patterns = np.load(OUT / "protocol/hadamard_patterns.float32.npy")
    coordinates = core.load(C195 / "protocol/source_coordinates.json")["coordinates"]
    c195_index = core.rows(C195 / "raw/index.jsonl"); c195_raw = np.load(C195 / "raw/signed_q23_q24_q25.float16.npy", mmap_mode="r")
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    actual = np.lib.format.open_memmap(OUT / "raw/orthogonal_actual.float16.npy", mode="w+", dtype=np.float16, shape=(14, 3, 16, 2, 6, DIM))
    predicted = np.lib.format.open_memmap(OUT / "raw/orthogonal_predicted.float16.npy", mode="w+", dtype=np.float16, shape=actual.shape)
    even_energy = np.zeros((14, 3, 16, 2, 6), np.float64)
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3"); quant = quantization_audit(model); base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def run(row, pattern_batch, signed_delta):
            ids, mask, pos, _ = fixed_base.fixed_batch([row] * len(pattern_batch), pad, device, WIDTH); caught = {}
            def patch(_m, _a, value):
                state = tensor(value); changed = state.clone()
                for local, pattern in enumerate(pattern_batch):
                    for source_i, coordinate in enumerate(coordinates):
                        for position in row["role_positions"]["relation"]:
                            changed[local, position, int(coordinate)] += float(pattern[source_i]) * signed_delta
                return (changed,) + value[1:] if isinstance(value, tuple) else changed
            hooks = [base.layers[22].register_forward_hook(patch), base.layers[23].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q24", tensor(v).detach())), base.layers[24].register_forward_hook(lambda _m, _a, v: caught.__setitem__("q25", tensor(v).detach()))]
            try: model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks: hook.remove()
            field = np.empty((len(pattern_batch), 2, 6, DIM), np.float32)
            for local in range(len(pattern_batch)):
                for state_i, name in enumerate(("q24", "q25")):
                    for role_i, role in enumerate(ROLES): field[local, state_i, role_i] = caught[name][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        for local_i, (row, source_i) in enumerate(zip(rows, source_indices)):
            epsilon = float(c195_index[source_i]["epsilon"]); jacobian = np.asarray(c195_raw[source_i], dtype=np.float32)
            ids, mask, pos, _ = fixed_base.fixed_batch([row], pad, device, WIDTH); base_caught = {}
            hooks = [base.layers[23].register_forward_hook(lambda _m, _a, v: base_caught.__setitem__("q24", tensor(v).detach())), base.layers[24].register_forward_hook(lambda _m, _a, v: base_caught.__setitem__("q25", tensor(v).detach()))]
            try: model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks: hook.remove()
            baseline = np.empty((2, 6, DIM), np.float32)
            for state_i, name in enumerate(("q24", "q25")):
                for role_i, role in enumerate(ROLES): baseline[state_i, role_i] = base_caught[name][0, row["role_positions"][role]].mean(0).float().cpu().numpy()
            for dose_i, dose in enumerate(DOSES):
                delta = float(dose * epsilon / np.sqrt(64.0)); plus = run(row, patterns, delta); minus = run(row, patterns, -delta)
                actual_effect = 0.5 * (plus - minus); predicted_effect = np.tensordot(patterns * delta, jacobian, axes=(1, 0))
                actual[local_i, dose_i] = actual_effect.astype(np.float16); predicted[local_i, dose_i] = predicted_effect.astype(np.float16)
                even = 0.5 * (plus + minus) - baseline[None]
                even_energy[local_i, dose_i] = np.square(even, dtype=np.float64).sum(axis=-1)
            actual.flush(); predicted.flush()
            print(f"[C196] {local_i + 1}/14 {c195_index[source_i]['family']} {c195_index[source_i]['program']}", flush=True)
        np.save(OUT / "raw/even_energy.float64.npy", even_energy)
        checks = {"actual_shape": list(actual.shape) == [14, 3, 16, 2, 6, DIM], "predicted_shape": list(predicted.shape) == list(actual.shape), "finite": bool(np.isfinite(actual).all()) and bool(np.isfinite(predicted).all()) and bool(np.isfinite(even_energy).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / "analysis/capture.json", {"checks": checks, "runtime": placement}); core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"checks": checks}, indent=2))
    finally:
        actual.flush(); predicted.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def weighted_sign(left, right):
    weight = np.minimum(np.abs(left), np.abs(right)).astype(np.float64); return float((weight * (np.signbit(left) == np.signbit(right))).sum() / max(weight.sum(), 1e-30))


def analyze():
    actual = np.load(OUT / "raw/orthogonal_actual.float16.npy", mmap_mode="r"); predicted = np.load(OUT / "raw/orthogonal_predicted.float16.npy", mmap_mode="r"); even = np.load(OUT / "raw/even_energy.float64.npy")
    dose_rows = []
    for dose_i, dose in enumerate(DOSES):
        a = np.asarray(actual[:, dose_i], dtype=np.float32); p = np.asarray(predicted[:, dose_i], dtype=np.float32)
        error2 = np.square(a - p, dtype=np.float64).sum(); actual2 = np.square(a, dtype=np.float64).sum(); pred2 = np.square(p, dtype=np.float64).sum()
        dose_rows.append({"dose": dose, "nrmse": float(np.sqrt(error2 / max(actual2, 1e-30))), "weighted_sign_agreement": weighted_sign(a, p), "predicted_to_actual_rms": float(np.sqrt(pred2 / max(actual2, 1e-30))), "even_to_odd_energy_ratio": float(even[:, dose_i].sum() / max(actual2, 1e-30))})
    gates = core.load(OUT / "protocol/preregistration.json")["gates"]
    passed = dose_rows[0]["nrmse"] <= gates["dose_0.25_nrmse_max"] and dose_rows[1]["nrmse"] <= gates["dose_0.5_nrmse_max"] and dose_rows[2]["nrmse"] <= gates["dose_1.0_nrmse_max"] and min(r["weighted_sign_agreement"] for r in dose_rows) >= gates["weighted_sign_min"]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "status": "multidose_orthogonal_identification_analyzed", "dose_rows": dose_rows, "linear_superposition_gate_passed": passed, "interpretation": "Passing would validate a local linear response instrument for unseen joint stimuli at these doses; it would not make the operator semantic or globally linear.", "next_authorization": "C197_structure_model_tournament_and_holdout_prediction"}
    core.save(OUT / "analysis/multidose_identification.json", report)
    checks = {"three_doses": len(dose_rows) == 3, "ordered": [r["dose"] for r in dose_rows] == list(DOSES), "finite": bool(np.isfinite([[r[k] for k in ("nrmse", "weighted_sign_agreement", "predicted_to_actual_rms", "even_to_odd_energy_ratio")] for r in dose_rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())}); print(json.dumps({"dose_rows": dose_rows, "passed": passed, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/multidose_identification.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": report, "next_authorization": report["next_authorization"]}; core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "capture", "analyze", "close")); args = parser.parse_args(); {"contract": contract, "capture": capture, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__": main()
