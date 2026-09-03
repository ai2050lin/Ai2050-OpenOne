#!/usr/bin/env python3
"""C203: measure whether C196 coordinate interventions survive BF16 assignment."""
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
OUT = RESULT / "phase1737_c203_bf16_intervention_representability"
C195 = RESULT / "phase1729_c195_signed_role_checkpoint_trajectory"
C196 = RESULT / "phase1730_c196_multidose_orthogonal_identification"
C202 = RESULT / "phase1736_c202_campaign_theory_adjudication"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base

PHASE, CAMPAIGN, WIDTH = 1737, "C203", 224
DOSES = (0.25, 0.5, 1.0)


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def contract() -> None:
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C202 / "audit/independent_final_audit.json")
    coordinates = core.load(C195 / "protocol/source_coordinates.json")["coordinates"]
    anchors = core.rows(C196 / "compiled/anchors.jsonl")
    source_indices = core.load(C196 / "protocol/source_anchor_indices.json")["indices"]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "C203_precision_calibrated_nonlinear_response_ecology_campaign",
        "anchors": len(anchors) == len(source_indices) == 14,
        "coordinates": len(coordinates) == len(set(coordinates)) == 64,
        "doses": DOSES == (0.25, 0.5, 1.0),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "bf16_representability_frozen",
        "object": "the exact BF16 assignment step used by C196 at layer22 relation-role source coordinates",
        "sample": {"anchors": 14, "source_coordinates": 64, "doses": list(DOSES), "signs": ["plus", "minus"], "all_relation_tokens": True},
        "metrics": ["zero_step_fraction", "applied_to_intended_ratio", "sign_error_fraction", "plus_minus_asymmetry", "intended_to_local_ulp"],
        "diagnostic_rule": "If any dose has zero-step fraction >0.05, median applied/intended outside [0.8,1.2], or sign error >0.01, C196 is numerically confounded at that dose.",
        "claim_boundary": "This calibrates the intervention write operation only; it does not identify downstream linearity, semantics, or a mechanism.",
        "forbidden": ["attention", "MLP", "weights", "PCA", "changing C196 results", "downstream mechanism inference"],
        "producer_sha256": core.sha(Path(__file__)),
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks}, indent=2))


@torch.inference_mode()
def capture() -> None:
    coordinates = core.load(C195 / "protocol/source_coordinates.json")["coordinates"]
    anchors = core.rows(C196 / "compiled/anchors.jsonl")
    source_indices = core.load(C196 / "protocol/source_anchor_indices.json")["indices"]
    c195_index = core.rows(C195 / "raw/index.jsonl")
    model = None
    records = []
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for local_i, (row, source_i) in enumerate(zip(anchors, source_indices)):
            ids, mask, pos, _ = fixed_base.fixed_batch([row], pad, device, WIDTH)
            caught = {}

            def observe(_module, _args, value):
                caught["q23"] = tensor(value).detach()

            hook = base.layers[22].register_forward_hook(observe)
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                hook.remove()
            positions = row["role_positions"]["relation"]
            source = caught["q23"][0, positions][:, coordinates]
            if source.dtype != torch.bfloat16:
                raise RuntimeError(f"expected BF16 source, got {source.dtype}")
            source_f32 = source.float()
            up = torch.nextafter(source, torch.full_like(source, float("inf"))).float() - source_f32
            down = source_f32 - torch.nextafter(source, torch.full_like(source, float("-inf"))).float()
            epsilon = float(c195_index[source_i]["epsilon"])
            for dose in DOSES:
                intended = float(dose * epsilon / np.sqrt(64.0))
                plus = source.clone(); plus.add_(intended)
                minus = source.clone(); minus.add_(-intended)
                plus_step = plus.float() - source_f32
                minus_step = minus.float() - source_f32
                for token_local, position in enumerate(positions):
                    for coordinate_local, coordinate in enumerate(coordinates):
                        records.append({
                            "anchor": local_i,
                            "family": c195_index[source_i]["family"],
                            "program": c195_index[source_i]["program"],
                            "dose": float(dose),
                            "token_local": token_local,
                            "token_position": int(position),
                            "coordinate": int(coordinate),
                            "source_value": float(source_f32[token_local, coordinate_local].item()),
                            "intended": intended,
                            "plus_step": float(plus_step[token_local, coordinate_local].item()),
                            "minus_step": float(minus_step[token_local, coordinate_local].item()),
                            "ulp_up": float(up[token_local, coordinate_local].item()),
                            "ulp_down": float(down[token_local, coordinate_local].item()),
                        })
            print(f"[C203] {local_i + 1}/14 {c195_index[source_i]['family']} {c195_index[source_i]['program']}", flush=True)
        core.write_rows(OUT / "raw/bf16_write_steps.jsonl", records)
        checks = {
            "records": len(records) >= 14 * 3 * 64,
            "all_anchors": len({row["anchor"] for row in records}) == 14,
            "all_coordinates": len({row["coordinate"] for row in records}) == 64,
            "finite": bool(np.isfinite([[row[k] for k in ("source_value", "intended", "plus_step", "minus_step", "ulp_up", "ulp_down")] for row in records]).all()),
            "bf16": quant["has_bf16_parameters"],
            "unquantized": not quant["has_quantized_modules"],
        }
        core.save(OUT / "analysis/capture.json", {"checks": checks, "records": len(records), "runtime": placement, "quantization": quant})
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks, "records": len(records)}, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        gc.collect()
        torch.cuda.empty_cache()


def analyze() -> None:
    records = core.rows(OUT / "raw/bf16_write_steps.jsonl")
    dose_rows = []
    for dose in DOSES:
        subset = [row for row in records if row["dose"] == dose]
        intended = np.asarray([row["intended"] for row in subset], dtype=np.float64)
        plus = np.asarray([row["plus_step"] for row in subset], dtype=np.float64)
        minus = np.asarray([row["minus_step"] for row in subset], dtype=np.float64)
        ulp = 0.5 * np.asarray([row["ulp_up"] + row["ulp_down"] for row in subset], dtype=np.float64)
        signed = np.concatenate((plus, -minus))
        intended_both = np.concatenate((intended, intended))
        zero = np.abs(signed) <= 0
        sign_error = signed < 0
        ratio = signed / intended_both
        asymmetry = np.abs(plus + minus) / np.maximum(np.abs(plus - minus), 1e-30)
        confounded = float(zero.mean()) > 0.05 or not 0.8 <= float(np.median(ratio)) <= 1.2 or float(sign_error.mean()) > 0.01
        dose_rows.append({
            "dose": float(dose),
            "samples_per_sign": len(subset),
            "zero_step_fraction": float(zero.mean()),
            "sign_error_fraction": float(sign_error.mean()),
            "applied_to_intended_ratio_median": float(np.median(ratio)),
            "applied_to_intended_ratio_q05_q95": [float(np.quantile(ratio, 0.05)), float(np.quantile(ratio, 0.95))],
            "plus_minus_asymmetry_median": float(np.median(asymmetry)),
            "intended_to_local_ulp_median": float(np.median(intended / np.maximum(ulp, 1e-30))),
            "numerically_confounded": bool(confounded),
        })
    all_confounded = all(row["numerically_confounded"] for row in dose_rows)
    any_confounded = any(row["numerically_confounded"] for row in dose_rows)
    report = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "status": "bf16_representability_analyzed",
        "dose_rows": dose_rows,
        "any_c196_dose_numerically_confounded": any_confounded,
        "all_c196_doses_numerically_confounded": all_confounded,
        "interpretation": "A confounded dose cannot adjudicate downstream linearity. A representable dose still requires a separate forward-response test.",
        "next_authorization": "C204_ulp_calibrated_symmetric_response" if any_confounded else "C204_nonlinear_basis_response_without_precision_repair",
    }
    core.save(OUT / "analysis/representability.json", report)
    checks = {"three_doses": len(dose_rows) == 3, "ordered": [row["dose"] for row in dose_rows] == list(DOSES), "finite": bool(np.isfinite([[row[k] for k in ("zero_step_fraction", "sign_error_fraction", "applied_to_intended_ratio_median", "plus_minus_asymmetry_median", "intended_to_local_ulp_median")] for row in dose_rows]).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close() -> None:
    protocol = core.load(OUT / "protocol/preregistration.json")
    report = core.load(OUT / "analysis/representability.json")
    checks = {
        "contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"],
        "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"],
        "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"],
        "hash": core.sha(Path(__file__)) == protocol["producer_sha256"],
    }
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
