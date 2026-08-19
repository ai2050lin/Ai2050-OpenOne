#!/usr/bin/env python3
"""Phase1382: known-truth and exact-shape cameras for C060."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1382, "C060"
CONTRACT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
BEHAVIOR = TESTS / "result/phase1381_c060_qwen_behavior_qualification"
OUT = TESTS / "result/phase1382_c060_response_coalition_camera"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")


def parents() -> dict:
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1382_c060_instrument_calibration" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1381 did not authorize calibration")
    return core.load(CONTRACT / "protocol/preregistration.json")


def select_camera_cases(pairs: list[dict]) -> list[dict]:
    selected = []
    keys = sorted({(r["target_family"], r["partition"], r["surface"]) for r in pairs})
    for key in keys:
        cell = sorted(
            (r for r in pairs if (r["target_family"], r["partition"], r["surface"]) == key),
            key=lambda r: r["pair_id"],
        )
        selected.append(cell[0])
    return selected


def mode_layout(protocol: dict, mode: str) -> list[dict]:
    return [
        {"mode": mode, "direction": direction, "dose": float(dose)}
        for direction in protocol["dose"]["directions"]
        for dose in protocol["dose"]["values"]
    ]


def prepare() -> None:
    protocol = parents()
    if (OUT / "protocol/execution_manifest.json").exists():
        raise RuntimeError("Phase1382 manifest already exists")
    cases = select_camera_cases(core.rows(BEHAVIOR / "material/eligible_pairs.jsonl"))
    if len(cases) != int(protocol["camera"]["qwen_cases"]):
        raise RuntimeError("camera case count mismatch")
    layouts = {mode: mode_layout(protocol, mode) for mode in ("sufficiency", "reverse")}
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "behavior_final_sha256": core.sha(BEHAVIOR / "analysis/final.json"),
        "behavior_audit_sha256": core.sha(BEHAVIOR / "audit/independent_final_audit.json"),
        "model": MODEL,
        "precision": "bfloat16-no-quantization",
        "allowed_observables": protocol["allowed_observables"],
        "forbidden": protocol["forbidden"],
        "paths": protocol["paths"],
        "camera_gate": protocol["camera"],
        "dose_gate": protocol["dose"],
        "fixed_coalition_gate": protocol["fixed_coalitions"],
        "dynamic_coalition_gate": protocol["dynamic_coalitions"],
        "mediation_gate": protocol["mediation"],
        "known_truth_systems": int(protocol["camera"]["known_truth_systems"]),
        "rows_per_mode_path_case": 4 + len(layouts["sufficiency"]),
        "mode_layouts": layouts,
        "camera_case_ids": [r["pair_id"] for r in cases],
        "same_shape_as_phase1383": True,
        "zero_write_identity": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "protocol/execution_manifest.json", manifest)
    core.write_rows(OUT / "material/camera_cases.jsonl", cases)
    print(json.dumps(manifest, indent=2))


def classify_curve(values: list[float], doses: list[float], gate: dict) -> str:
    by = dict(zip(doses, values))
    jumps = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    threshold = (
        abs(by[0.5]) <= gate["threshold_low_dose_abs_median_max"]
        and max(jumps) >= gate["threshold_adjacent_jump_min"]
        and by[1.0] >= gate["threshold_high_dose_gain_min"]
        and abs(by[1.0] - by[0.875]) <= gate["threshold_plateau_abs_difference_max"]
    )
    slopes = [
        (values[i + 1] - values[i]) / (doses[i + 1] - doses[i])
        for i in range(len(values) - 1)
    ]
    linear = max(slopes) - min(slopes) < 1e-10
    if threshold:
        return "threshold"
    if linear:
        return "linear"
    return "nonmonotone"


def known_truth(protocol: dict) -> tuple[list[dict], dict]:
    count = int(protocol["camera"]["known_truth_systems"])
    doses = [float(v) for v in protocol["dose"]["values"]]
    records = []
    curve_exact = True
    coalition_exact = True
    dynamic_exact = True
    topology_exact = True
    for system_id in range(count):
        curve_kind = ("linear", "threshold", "nonmonotone")[system_id % 3]
        if curve_kind == "linear":
            values = [24.0 * dose for dose in doses]
        elif curve_kind == "threshold":
            values = [0.0 if dose < 0.625 else 30.0 for dose in doses]
        else:
            values = [30.0 * math.sin(math.pi * dose) for dose in doses]
        prediction = classify_curve(values, doses, protocol["dose"])
        curve_exact &= prediction == curve_kind

        dim = 16
        a = set(range(0, 6))
        b = set(range(6, dim))
        weights = [2.0 + 0.1 * ((j + system_id) % 3) for j in range(dim)]
        effect_a = sum(weights[j] for j in a)
        effect_b = sum(weights[j] for j in b)
        interaction = -0.25 * effect_a if system_id % 2 == 0 else 0.25 * effect_a
        effect_union = effect_a + effect_b + interaction
        gamma = effect_union - effect_a - effect_b
        coalition_exact &= (a | b == set(range(dim)) and not a & b and abs(gamma - interaction) < 1e-12)

        planted = {1, 4, 7, 12}
        delta = [0.1 * ((j + 2) % 5) for j in range(dim)]
        for j in planted:
            delta[j] = 10.0 + j
        predicted = set(sorted(range(dim), key=lambda j: (-abs(delta[j]), j))[: len(planted)])
        dynamic_exact &= predicted == planted

        topology = "serial" if system_id % 2 == 0 else "parallel"
        query_block = 1.0 if topology == "serial" else 0.0
        topology_prediction = "serial" if query_block >= 0.5 else "parallel"
        topology_exact &= topology_prediction == topology
        records.append({
            "system_id": system_id,
            "curve_kind": curve_kind,
            "curve_prediction": prediction,
            "curve_values": values,
            "coalition_A": sorted(a),
            "coalition_B": sorted(b),
            "effect_A": effect_a,
            "effect_B": effect_b,
            "effect_union": effect_union,
            "gamma": gamma,
            "planted_dynamic": sorted(planted),
            "predicted_dynamic": sorted(predicted),
            "topology": topology,
            "topology_prediction": topology_prediction,
        })
    summary = {
        "system_count": count,
        "curve_classification_exact": curve_exact,
        "coalition_union_complement_exact": coalition_exact,
        "dynamic_mask_exact": dynamic_exact,
        "serial_parallel_exact": topology_exact,
    }
    return records, summary


def make_batch(rows: list[dict], pad: int, device: torch.device):
    width = max(len(r["prompt_ids"]) for r in rows)
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    offsets = []
    for i, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        offset = width - len(value)
        offsets.append(offset)
        ids[i, offset:] = value
        mask[i, offset:] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, offsets


def copy_role(value, original, target_index: int, target: dict, target_offset: int,
              source_index: int, source: dict, source_offset: int, role: str) -> None:
    tp = [target_offset + p for p in target["role_positions"][role]]
    sp = [source_offset + p for p in source["role_positions"][role]]
    if len(tp) != len(sp):
        raise RuntimeError("role span mismatch")
    value[target_index, tp] = original[source_index, sp]


@torch.inference_mode()
def identity_camera(manifest: dict, cases: list[dict]) -> tuple[list[dict], dict, dict]:
    compiled = {r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case_index, case in enumerate(cases):
            donors = {key: compiled[case[key]] for key in DONOR_KEYS}
            donor_rows = [donors[key] for key in DONOR_KEYS]
            for path_name, path in manifest["paths"].items():
                for mode, target_layout in manifest["mode_layouts"].items():
                    rows = list(donor_rows)
                    recipient_indices = []
                    for _spec in target_layout:
                        recipient = 1 if mode == "sufficiency" else 0
                        recipient_indices.append(recipient)
                        rows.append(donor_rows[recipient])
                    ids, mask, positions, offsets = make_batch(rows, pad, device)

                    def hook(_module, args):
                        original = args[0]
                        value = original.clone()
                        for local, recipient in enumerate(recipient_indices):
                            target_index = 4 + local
                            copy_role(
                                value, original, target_index, rows[target_index], offsets[target_index],
                                recipient, rows[recipient], offsets[recipient], path["source"]["role"],
                            )
                        return (value,) + args[1:]

                    handle = model.model.layers[path["source"]["layer"]].register_forward_pre_hook(hook)
                    try:
                        kwargs = {
                            "input_ids": ids,
                            "attention_mask": mask,
                            "position_ids": positions,
                            "use_cache": False,
                            "output_hidden_states": True,
                            "return_dict": True,
                        }
                        if supports:
                            kwargs["logits_to_keep"] = 1
                        output = model(**kwargs)
                    finally:
                        handle.remove()
                    for local, spec in enumerate(target_layout):
                        target_index = 4 + local
                        recipient = recipient_indices[local]
                        candidate_ids = rows[recipient]["candidate_ids"]
                        output_diff = max(
                            abs(float(
                                output.logits[target_index, -1, token_ids[0]].float()
                                - output.logits[recipient, -1, token_ids[0]].float()
                            ))
                            for token_ids in candidate_ids
                        )
                        rels = []
                        for checkpoint in path["checkpoints"]:
                            role, layer = checkpoint["role"], checkpoint["layer"]
                            rp = [offsets[recipient] + p for p in rows[recipient]["role_positions"][role]]
                            tp = [offsets[target_index] + p for p in rows[target_index]["role_positions"][role]]
                            left = output.hidden_states[layer][recipient, rp].float()
                            right = output.hidden_states[layer][target_index, tp].float()
                            rels.append(float((right - left).norm() / (left.norm() + 1e-12)))
                        records.append({
                            "pair_id": case["pair_id"],
                            "path": path_name,
                            **spec,
                            "output_max_abs_diff": output_diff,
                            "checkpoint_relative_l2_max": max(rels),
                        })
                    del output, ids, mask, positions
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"identity_camera": case_index + 1, "total": len(cases)}), flush=True)
        summary = {
            "case_count": len(cases),
            "record_count": len(records),
            "output_max_abs_diff": max(r["output_max_abs_diff"] for r in records),
            "checkpoint_relative_l2_max": max(r["checkpoint_relative_l2_max"] for r in records),
            "all_finite": all(
                math.isfinite(r["output_max_abs_diff"])
                and math.isfinite(r["checkpoint_relative_l2_max"])
                for r in records
            ),
        }
        runtime = {
            "placement": placement,
            "quantization": quant,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return records, summary, runtime
    finally:
        if model is not None:
            release_bf16(model)


def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/calibration_summary.json").exists():
        raise RuntimeError("Phase1382 run already exists")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    known_records, known = known_truth(protocol)
    camera_records, camera, runtime = identity_camera(manifest, core.rows(OUT / "material/camera_cases.jsonl"))
    gate = manifest["camera_gate"]
    checks = {
        "known_count": known["system_count"] == gate["known_truth_systems"],
        "curve_exact": known["curve_classification_exact"] == gate["threshold_linear_nonmonotone_exact"],
        "coalition_exact": known["coalition_union_complement_exact"] == gate["coalition_union_complement_exact"],
        "dynamic_exact": known["dynamic_mask_exact"] == gate["dynamic_mask_exact"],
        "topology_exact": known["serial_parallel_exact"] == gate["serial_parallel_exact"],
        "camera_count": camera["case_count"] == gate["qwen_cases"],
        "same_shape_output": camera["output_max_abs_diff"] <= gate["same_shape_output_max_abs_diff"],
        "same_shape_checkpoint": camera["checkpoint_relative_l2_max"] <= gate["same_shape_checkpoint_relative_l2_max"],
        "finite": camera["all_finite"],
    }
    summary = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "known_truth": known,
        "qwen_identity_camera": camera,
        "checks": checks,
        "camera_qualified": all(checks.values()),
        "runtime": runtime,
        "claim_boundary": "instrument calibration only; no natural mechanism result",
    }
    core.write_rows(OUT / "raw/known_truth_systems.jsonl", known_records)
    core.write_rows(OUT / "raw/qwen_exact_shape_identity.jsonl", camera_records)
    core.save(OUT / "analysis/calibration_summary.json", summary)
    print(json.dumps(summary, indent=2))


def finalize() -> None:
    summary = core.load(OUT / "analysis/calibration_summary.json")
    qualified = bool(summary["camera_qualified"])
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "camera_qualified": qualified,
        "authorization": (
            "run_phase1383_c060_refined_dose_observation"
            if qualified
            else "close_c060_camera_unqualified_before_natural_hidden_reveal"
        ),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "finalize": finalize}[args.command]()


if __name__ == "__main__":
    main()
