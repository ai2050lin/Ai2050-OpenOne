#!/usr/bin/env python3
"""C195: capture q23 -> q24 -> q25 signed role/checkpoint trajectories on C192."""
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
OUT = RESULT / "phase1729_c195_signed_role_checkpoint_trajectory"
C167 = RESULT / "phase1701_c167_transport_component_decomposition"
C192 = RESULT / "phase1726_c192_multi_program_response_equivalence"
C194 = RESULT / "phase1728_c194_signed_operator_campaign_contract"
PUBLIC = ROOT / "frontend/public/vis_data/research_kernel/c195_signed_operator_trajectory.json"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16
import phase1572_c099_fixed_width_graph_field_campaign as fixed_base
import phase1726_c192_multi_program_response_equivalence as c192

PHASE, CAMPAIGN = 1729, "C195"
DIM, WIDTH = 2560, 224
ROLES = c192.ROLES
STATES = ("embedding", "q23", "q24", "q25")


def now():
    return datetime.now(timezone.utc).isoformat()


def tensor(value):
    return value[0] if isinstance(value, tuple) else value


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C194 / "audit/independent_final_audit.json")
    c192_final = core.load(C192 / "analysis/final.json")
    compiled = core.rows(C192 / "compiled/qwen3.jsonl")
    anchor_rows = core.load(C192 / "protocol/behavior_eligibility_lock.json")["anchor_rows"]
    coordinates = core.load(C167 / "analysis/top_relation_source_coordinates.json")["coordinates"][:64]
    anchors = [compiled[i] for i in anchor_rows]
    checks = {
        "authorization": parent["all_checks_passed"] and parent["authorization"] == "run_C195_signed_q23_q24_q25_trajectory_capture",
        "c192_complete": c192_final["all_checks_passed"] and len(anchors) == 112,
        "coordinates": len(coordinates) == 64 and len(set(coordinates)) == 64 and min(coordinates) >= 0 and max(coordinates) < DIM,
        "roles": all(set(r["role_positions"]) == set(ROLES) for r in anchors),
        "factorial": len({(r["family"], r["unit"], r["phrase_variant"], r["program"]) for r in anchors}) == 112,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "compiled/anchors.jsonl", anchors)
    protocol = {
        "phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "signed_trajectory_capture_frozen",
        "model": "Qwen3-4B BF16 CUDA nonquantized", "anchors": 112,
        "source": "q23 relation role, 64 physical activation coordinates frozen before this campaign",
        "targets": "q24 and q25, six semantic roles, all 2560 physical activation coordinates",
        "baseline_states": list(STATES),
        "perturbation": "symmetric one-coordinate finite difference; epsilon=0.5 times q23 relation-state RMS for each anchor",
        "raw_shape": [112, 64, 2, 6, DIM],
        "analysis": ["signed mean trajectory", "role energy transport", "weighted sign persistence", "phrase/program/family grouped atlas"],
        "claim_boundary": "local finite-difference response trajectory in one explicit graph micro-language; not a unique causal circuit or semantic equivalence class",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cosine", "target-energy-only classification", "post-reveal changes"],
        "producer_sha256": core.sha(Path(__file__)), "authorization": "run_cuda_capture_then_C196_multi_dose_identification",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "protocol/source_coordinates.json", {"coordinates": coordinates})
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "raw_shape": protocol["raw_shape"]}, indent=2))


@torch.inference_mode()
def capture():
    rows = core.rows(OUT / "compiled/anchors.jsonl")
    coordinates = core.load(OUT / "protocol/source_coordinates.json")["coordinates"]
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    raw = np.lib.format.open_memmap(OUT / "raw/signed_q23_q24_q25.float16.npy", mode="w+", dtype=np.float16, shape=(len(rows), 64, 2, 6, DIM))
    baseline = np.lib.format.open_memmap(OUT / "raw/baseline_role_states.float16.npy", mode="w+", dtype=np.float16, shape=(len(rows), 4, 6, DIM))
    model = None
    try:
        model, tokenizer, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model); base = model.model
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)

        def observe(row, selected, sign, epsilon):
            ids, mask, pos, _ = fixed_base.fixed_batch([row] * len(selected), pad, device, WIDTH)
            caught = {}

            def patch(_module, _args, value):
                state = tensor(value); changed = state.clone()
                for local, coordinate in enumerate(selected):
                    for position in row["role_positions"]["relation"]:
                        changed[local, position, int(coordinate)] += sign * epsilon
                return (changed,) + value[1:] if isinstance(value, tuple) else changed

            hooks = [
                base.layers[22].register_forward_hook(patch),
                base.layers[23].register_forward_hook(lambda _m, _a, value: caught.__setitem__("q24", tensor(value).detach())),
                base.layers[24].register_forward_hook(lambda _m, _a, value: caught.__setitem__("q25", tensor(value).detach())),
            ]
            try:
                model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
            finally:
                for hook in hooks: hook.remove()
            field = np.empty((len(selected), 2, 6, DIM), np.float32)
            for local in range(len(selected)):
                for state_i, name in enumerate(("q24", "q25")):
                    for role_i, role in enumerate(ROLES):
                        field[local, state_i, role_i] = caught[name][local, row["role_positions"][role]].mean(0).float().cpu().numpy()
            return field

        index = []
        for anchor_i, row in enumerate(rows):
            ids, mask, pos, _ = fixed_base.fixed_batch([row], pad, device, WIDTH)
            output = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True, output_hidden_states=True)
            hidden = output.hidden_states
            state_map = (hidden[0], hidden[23], hidden[24], hidden[25])
            for state_i, state in enumerate(state_map):
                for role_i, role in enumerate(ROLES):
                    baseline[anchor_i, state_i, role_i] = state[0, row["role_positions"][role]].mean(0).float().cpu().numpy().astype(np.float16)
            source = np.asarray(baseline[anchor_i, 1, ROLES.index("relation")], dtype=np.float32)
            epsilon = 0.5 * float(np.sqrt(np.mean(np.square(source), dtype=np.float64)))
            for start in range(0, 64, 16):
                selected = coordinates[start:start + 16]
                plus = observe(row, selected, 1.0, epsilon); minus = observe(row, selected, -1.0, epsilon)
                raw[anchor_i, start:start + len(selected)] = ((plus - minus) / (2.0 * epsilon)).astype(np.float16)
            raw.flush(); baseline.flush()
            index.append({"anchor_index": anchor_i, "case_id": row["case_id"], "family": row["family"], "unit": row["unit"], "phrase_variant": row["phrase_variant"], "program": row["program"], "epsilon": epsilon})
            print(f"[C195] {anchor_i + 1}/{len(rows)} {row['family']} {row['program']} u{row['unit']} p{row['phrase_variant']}", flush=True)
        core.write_rows(OUT / "raw/index.jsonl", index)
        checks = {
            "raw_shape": list(raw.shape) == [112, 64, 2, 6, DIM], "baseline_shape": list(baseline.shape) == [112, 4, 6, DIM],
            "finite": bool(np.isfinite(raw).all()) and bool(np.isfinite(baseline).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"],
        }
        core.save(OUT / "analysis/capture.json", {"checks": checks, "runtime": placement, "quantization": quant})
        core.save(OUT / "audit/internal_capture_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps({"checks": checks}, indent=2))
    finally:
        raw.flush(); baseline.flush()
        if model is not None: release_bf16(model)
        gc.collect(); torch.cuda.empty_cache()


def weighted_sign_persistence(left, right):
    weight = np.minimum(np.abs(left), np.abs(right)).astype(np.float64)
    return float((weight * (np.signbit(left) == np.signbit(right))).sum() / max(weight.sum(), 1e-30))


def analyze():
    raw = np.load(OUT / "raw/signed_q23_q24_q25.float16.npy", mmap_mode="r")
    baseline = np.load(OUT / "raw/baseline_role_states.float16.npy", mmap_mode="r")
    index = core.rows(OUT / "raw/index.jsonl")
    grouped = []
    rows_payload = []
    for family in sorted({r["family"] for r in index}):
        for program in c192.PROGRAMS:
            for phrase in (0, 1):
                selected = [r["anchor_index"] for r in index if r["family"] == family and r["program"] == program and r["phrase_variant"] == phrase]
                values = np.asarray(raw[selected], dtype=np.float32).mean(axis=(0, 1))
                q24, q25 = values[0], values[1]
                role_energy24 = np.square(q24, dtype=np.float64).sum(axis=1); role_energy25 = np.square(q25, dtype=np.float64).sum(axis=1)
                grouped.append({
                    "family": family, "program": program, "phrase_variant": phrase, "support": len(selected),
                    "q24_energy": float(role_energy24.sum()), "q25_energy": float(role_energy25.sum()),
                    "gain": float(np.sqrt(role_energy25.sum() / max(role_energy24.sum(), 1e-30))),
                    "weighted_sign_persistence": weighted_sign_persistence(q24, q25),
                    "role_energy_q24": {role: float(role_energy24[i] / max(role_energy24.sum(), 1e-30)) for i, role in enumerate(ROLES)},
                    "role_energy_q25": {role: float(role_energy25[i] / max(role_energy25.sum(), 1e-30)) for i, role in enumerate(ROLES)},
                })
                for state_i, state in enumerate(("q24_response", "q25_response")):
                    for role_i, role in enumerate(ROLES):
                        rows_payload.append({"kind": "signed_response", "family": family, "program": program, "phrase_variant": phrase, "state": state, "role": role, "label": f"{family}/{program}/p{phrase}/{state}/{role}", "values": values[state_i, role_i].astype(np.float32).tolist()})
    baseline_rows = []
    for state_i, state in enumerate(STATES):
        for role_i, role in enumerate(ROLES):
            values = np.asarray(baseline[:, state_i, role_i], dtype=np.float32).mean(axis=0)
            baseline_rows.append({"kind": "baseline_state", "state": state, "role": role, "label": f"mean/{state}/{role}", "values": values.tolist()})
    sign_values = [r["weighted_sign_persistence"] for r in grouped]; gains = [r["gain"] for r in grouped]
    report = {
        "phase": PHASE, "campaign": CAMPAIGN, "status": "signed_trajectory_observed", "groups": len(grouped),
        "weighted_sign_persistence_median": float(np.median(sign_values)), "weighted_sign_persistence_range": [float(min(sign_values)), float(max(sign_values))],
        "q25_over_q24_gain_median": float(np.median(gains)), "q25_over_q24_gain_range": [float(min(gains)), float(max(gains))],
        "group_rows": grouped,
        "interpretation": "The same q23 perturbation has an observable signed two-checkpoint trajectory. Stability or linguistic abstraction requires holdout prediction in C196-C199.",
        "next_authorization": "C196_multi_dose_orthogonal_system_identification",
    }
    core.save(OUT / "analysis/signed_trajectory.json", report)
    payload = {
        "schema": "c195_signed_operator_trajectory.v1", "result_type": "signed_operator_trajectory_heatmap", "phase": PHASE, "campaign": CAMPAIGN, "model": "Qwen3-4B",
        "title": "C195 Signed q23-q24-q25 Role/Coordinate Trajectory", "dimensions": list(range(DIM)),
        "default_coordinates": np.argsort(-np.var(np.asarray(raw[:, :, 1, ROLES.index("boundary")], dtype=np.float32).mean(axis=1), axis=0))[:64].astype(int).tolist(),
        "rows": baseline_rows + rows_payload, "summary": {key: report[key] for key in ("groups", "weighted_sign_persistence_median", "weighted_sign_persistence_range", "q25_over_q24_gain_median", "q25_over_q24_gain_range")},
        "raw_tensor": {"path": "tests/glm5/result/phase1729_c195_signed_role_checkpoint_trajectory/raw/signed_q23_q24_q25.float16.npy", "shape": [112, 64, 2, 6, DIM]},
        "coordinate_semantics": "Baseline rows contain embedding/q23/q24/q25 activations; response rows contain signed derivatives for every q24/q25 physical target activation coordinate after q23 relation-role microstimulation.",
        "claim_boundary": core.load(OUT / "protocol/preregistration.json")["claim_boundary"],
    }
    PUBLIC.parent.mkdir(parents=True, exist_ok=True); PUBLIC.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    asset = {"path": str(PUBLIC.relative_to(ROOT)).replace("\\", "/"), "sha256": core.sha(PUBLIC), "bytes": PUBLIC.stat().st_size, "rows": len(payload["rows"]), "schema": payload["schema"]}
    core.save(OUT / "analysis/public_asset.json", asset)
    checks = {"groups": len(grouped) == 56, "payload_rows": len(payload["rows"]) == 696, "dimensions": len(payload["dimensions"]) == DIM, "finite": bool(np.isfinite(sign_values + gains).all())}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"summary": payload["summary"], "asset": asset, "checks": checks}, indent=2))


def close():
    protocol = core.load(OUT / "protocol/preregistration.json"); report = core.load(OUT / "analysis/signed_trajectory.json"); asset = core.load(OUT / "analysis/public_asset.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "capture": core.load(OUT / "audit/internal_capture_audit.json")["all_checks_passed"], "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"], "hash": core.sha(Path(__file__)) == protocol["producer_sha256"], "asset_hash": core.sha(PUBLIC) == asset["sha256"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {key: report[key] for key in ("groups", "weighted_sign_persistence_median", "weighted_sign_persistence_range", "q25_over_q24_gain_median", "q25_over_q24_gain_range", "interpretation")}, "asset": asset, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final); print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("contract", "capture", "analyze", "close")); args = parser.parse_args()
    {"contract": contract, "capture": capture, "analyze": analyze, "close": close}[args.command]()


if __name__ == "__main__": main()
