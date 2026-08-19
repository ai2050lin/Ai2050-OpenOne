#!/usr/bin/env python3
"""Phase1377: known-truth and exact-shape response-field camera for C059."""
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

PHASE, CAMPAIGN = 1377, "C059"
CONTRACT = TESTS / "result/phase1375_c059_independent_relaunch_contract"
BEHAVIOR = TESTS / "result/phase1376_c059_qwen_behavior_qualification"
OUT = TESTS / "result/phase1377_c059_response_field_camera"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")


def parents() -> dict:
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1377_c059_instrument_calibration" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1376 did not authorize calibration")
    return core.load(CONTRACT / "protocol/preregistration.json")


def select_camera_cases(pairs: list[dict]) -> list[dict]:
    selected = []
    keys = sorted({(r["target_family"], r["partition"], r["surface"]) for r in pairs})
    for key in keys:
        cell = sorted((r for r in pairs if (r["target_family"], r["partition"], r["surface"]) == key),
                      key=lambda r: r["pair_id"])
        selected.append(cell[0])
    return selected


def layout(protocol: dict) -> list[dict]:
    return [
        {"mode": mode, "direction": direction, "dose": float(dose)}
        for mode in ("sufficiency", "reverse")
        for direction in protocol["dose"]["directions"]
        for dose in protocol["dose"]["values"]
    ]


def prepare() -> None:
    protocol = parents()
    if (OUT / "protocol/execution_manifest.json").exists():
        raise RuntimeError("Phase1377 manifest already exists")
    cases = select_camera_cases(core.rows(BEHAVIOR / "material/eligible_pairs.jsonl"))
    if len(cases) != int(protocol["camera"]["qwen_cases"]):
        raise RuntimeError("camera case count mismatch")
    target_layout = layout(protocol)
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "behavior_final_sha256": core.sha(BEHAVIOR / "analysis/final.json"),
        "behavior_audit_sha256": core.sha(BEHAVIOR / "audit/independent_final_audit.json"),
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "allowed_observables": protocol["allowed_observables"], "forbidden": protocol["forbidden"],
        "paths": protocol["paths"], "camera_gate": protocol["camera"],
        "dose_gate": protocol["dose"], "distance_gate": protocol["distance"],
        "coordinate_gate": protocol["coordinate_groups"], "mediation_gate": protocol["mediation"],
        "known_truth_systems": int(protocol["camera"]["known_truth_systems"]),
        "rows_per_path_case": 4 + len(target_layout),
        "target_layout": target_layout,
        "camera_case_ids": [r["pair_id"] for r in cases],
        "same_shape_as_phase1378": True,
        "zero_write_identity": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "protocol/execution_manifest.json", manifest)
    core.write_rows(OUT / "material/camera_cases.jsonl", cases)
    print(json.dumps(manifest, indent=2))


def known_truth(count: int) -> tuple[list[dict], dict]:
    records = []
    doses = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    d = torch.tensor([3.0, 4.0, 0.0], dtype=torch.float64)
    orth = torch.tensor([-4.0, 3.0, 0.0], dtype=torch.float64)
    for system_id in range(count):
        topology = "serial" if system_id % 2 == 0 else "parallel"
        for dose in doses:
            for direction, u in (("correct", dose * d), ("orthogonal", dose * orth)):
                alpha = float(torch.dot(u, d) / torch.dot(d, d))
                residual = u - alpha * d
                omega = float(torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(d))
                eta = float(torch.linalg.vector_norm(d - u) / torch.linalg.vector_norm(d))
                lhs = float(torch.dot(u, u))
                rhs = float(alpha * alpha * torch.dot(d, d) + torch.dot(residual, residual))
                records.append({
                    "system_id": system_id, "topology": topology, "dose": dose,
                    "direction": direction, "alpha": alpha, "omega": omega, "eta": eta,
                    "decomposition_relative_error": abs(lhs - rhs) / (lhs + 1e-12),
                    "query_block_fraction": 1.0 if topology == "serial" else 0.0,
                    "boundary_block_fraction": 1.0,
                    "topology_prediction": "serial" if topology == "serial" else "parallel",
                })

    dim, planted_count = 2560, 64
    rng = random.Random(5901377)
    planted = sorted(rng.sample(range(dim), planted_count))
    planted_set = set(planted)
    magnitude = [0.0] * dim
    signed = [0.0] * dim
    family_abs = [[0.0] * dim for _ in range(4)]
    family_count = [0] * 4
    for system_id in range(count):
        family = system_id % 4
        family_count[family] += 1
        for coordinate in range(dim):
            if coordinate in planted_set:
                value = 2.0 + 0.001 * (coordinate % 7)
            elif coordinate < 192:
                value = (0.15 + 0.001 * (coordinate % 5)) * (1.0 if system_id % 2 == 0 else -1.0)
            else:
                value = 0.0
            magnitude[coordinate] += abs(value)
            signed[coordinate] += value
            family_abs[family][coordinate] += abs(value)
    magnitude_order = sorted(range(dim), key=lambda c: (-magnitude[c] / count, c))
    stable_order = sorted(range(dim), key=lambda c: (-abs(signed[c] / count), c))
    family_order = sorted(range(dim), key=lambda c: (-min(family_abs[f][c] / family_count[f]
                                                               for f in range(4)), c))
    recovered = {
        "magnitude": set(magnitude_order[:planted_count]) == planted_set,
        "stable_sign": set(stable_order[:planted_count]) == planted_set,
        "family_min": set(family_order[:planted_count]) == planted_set,
    }
    summary = {
        "system_count": count,
        "response_record_count": len(records),
        "correct_alpha_exact": all(abs(r["alpha"] - r["dose"]) < 1e-12
                                   for r in records if r["direction"] == "correct"),
        "correct_omega_exact": all(r["omega"] < 1e-12 for r in records if r["direction"] == "correct"),
        "correct_eta_exact": all(abs(r["eta"] - (1.0 - r["dose"])) < 1e-12
                                 for r in records if r["direction"] == "correct"),
        "orthogonal_exact": all(abs(r["alpha"]) < 1e-12 and abs(r["omega"] - r["dose"]) < 1e-12
                                for r in records if r["direction"] == "orthogonal"),
        "decomposition_error_max": max(r["decomposition_relative_error"] for r in records),
        "topology_exact": all(r["topology_prediction"] == r["topology"] for r in records),
        "coordinate_planted_count": planted_count,
        "coordinate_planted": planted,
        "coordinate_routes_recovered": recovered,
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
                rows = list(donor_rows)
                recipient_indices = []
                for spec in manifest["target_layout"]:
                    recipient = 1 if spec["mode"] == "sufficiency" else 0
                    recipient_indices.append(recipient)
                    rows.append(donor_rows[recipient])
                ids, mask, positions, offsets = make_batch(rows, pad, device)

                def hook(_module, args):
                    original = args[0]
                    value = original.clone()
                    for local, recipient in enumerate(recipient_indices):
                        target_index = 4 + local
                        copy_role(value, original, target_index, rows[target_index], offsets[target_index],
                                  recipient, rows[recipient], offsets[recipient], path["source"]["role"])
                    return (value,) + args[1:]

                handle = model.model.layers[path["source"]["layer"]].register_forward_pre_hook(hook)
                try:
                    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                              "use_cache": False, "output_hidden_states": True, "return_dict": True}
                    if supports:
                        kwargs["logits_to_keep"] = 1
                    output = model(**kwargs)
                finally:
                    handle.remove()
                for local, spec in enumerate(manifest["target_layout"]):
                    target_index = 4 + local
                    recipient = recipient_indices[local]
                    candidate_ids = rows[recipient]["candidate_ids"]
                    output_diff = max(abs(float(output.logits[target_index, -1, token_ids[0]].float() -
                                                output.logits[recipient, -1, token_ids[0]].float()))
                                      for token_ids in candidate_ids)
                    rels = []
                    for checkpoint in path["checkpoints"]:
                        role, layer = checkpoint["role"], checkpoint["layer"]
                        rp = [offsets[recipient] + p for p in rows[recipient]["role_positions"][role]]
                        tp = [offsets[target_index] + p for p in rows[target_index]["role_positions"][role]]
                        left = output.hidden_states[layer][recipient, rp].float()
                        right = output.hidden_states[layer][target_index, tp].float()
                        rels.append(float((right - left).norm() / (left.norm() + 1e-12)))
                    records.append({"pair_id": case["pair_id"], "path": path_name, **spec,
                                    "output_max_abs_diff": output_diff,
                                    "checkpoint_relative_l2_max": max(rels)})
                del output, ids, mask, positions
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"identity_camera": case_index + 1, "total": len(cases)}), flush=True)
        summary = {
            "case_count": len(cases), "record_count": len(records),
            "output_max_abs_diff": max(r["output_max_abs_diff"] for r in records),
            "checkpoint_relative_l2_max": max(r["checkpoint_relative_l2_max"] for r in records),
            "all_finite": all(math.isfinite(r["output_max_abs_diff"]) and
                              math.isfinite(r["checkpoint_relative_l2_max"]) for r in records),
        }
        runtime = {"placement": placement, "quantization": quant,
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        return records, summary, runtime
    finally:
        if model is not None:
            release_bf16(model)


def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/calibration_summary.json").exists():
        raise RuntimeError("Phase1377 run already exists")
    known_records, known = known_truth(manifest["known_truth_systems"])
    camera_records, camera, runtime = identity_camera(manifest, core.rows(OUT / "material/camera_cases.jsonl"))
    gate = manifest["camera_gate"]
    checks = {
        "known_count": known["system_count"] == gate["known_truth_systems"],
        "dose_exact": known["correct_alpha_exact"] and known["correct_omega_exact"] and known["correct_eta_exact"],
        "orthogonal_exact": known["orthogonal_exact"],
        "distance_exact": known["decomposition_error_max"] <= manifest["distance_gate"]["direction_decomposition_relative_error_max"],
        "topology_exact": known["topology_exact"],
        "coordinate_routes": all(known["coordinate_routes_recovered"].values()),
        "camera_count": camera["case_count"] == gate["qwen_cases"],
        "same_shape_output": camera["output_max_abs_diff"] <= gate["same_shape_output_max_abs_diff"],
        "same_shape_checkpoint": camera["checkpoint_relative_l2_max"] <= gate["same_shape_checkpoint_relative_l2_max"],
        "finite": camera["all_finite"],
    }
    summary = {"phase": PHASE, "campaign": CAMPAIGN, "known_truth": known,
               "qwen_identity_camera": camera, "checks": checks,
               "camera_qualified": all(checks.values()), "runtime": runtime,
               "claim_boundary": "instrument calibration only; no natural mechanism result"}
    core.write_rows(OUT / "raw/known_truth_response_systems.jsonl", known_records)
    core.write_rows(OUT / "raw/qwen_exact_shape_identity.jsonl", camera_records)
    core.save(OUT / "analysis/calibration_summary.json", summary)
    print(json.dumps(summary, indent=2))


def finalize() -> None:
    summary = core.load(OUT / "analysis/calibration_summary.json")
    qualified = bool(summary["camera_qualified"])
    final = {"phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": qualified,
             "authorization": ("run_phase1378_c059_dose_distance_observation"
                               if qualified else "close_c059_camera_unqualified_before_natural_hidden_reveal"),
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "finalize": finalize}[args.command]()


if __name__ == "__main__":
    main()
