#!/usr/bin/env python3
"""Phase1371: known-truth and exact-shape camera calibration for C057."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1371, "C057"
CONTRACT = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
BEHAVIOR = TESTS / "result/phase1370_c057_qwen_behavior_qualification"
OUT = TESTS / "result/phase1371_c057_bidirectional_mediation_camera"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
ARMS = ("self", "correct", "wrong", "status")


def parents() -> dict:
    final = core.load(BEHAVIOR / "analysis/final.json")
    audit = core.load(BEHAVIOR / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1371_c057_instrument_calibration" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1370 did not authorize calibration")
    return core.load(CONTRACT / "protocol/preregistration.json")


def select_camera_cases(pairs: list[dict]) -> list[dict]:
    selected = []
    keys = sorted({(row["target_family"], row["partition"], row["surface"]) for row in pairs})
    for key in keys:
        cell = sorted((row for row in pairs if (row["target_family"], row["partition"], row["surface"]) == key),
                      key=lambda row: row["pair_id"])
        selected.append(cell[0])
    return selected


def prepare() -> None:
    protocol = parents()
    manifest_path = OUT / "protocol/execution_manifest.json"
    if manifest_path.exists():
        raise RuntimeError("Phase1371 manifest already exists")
    pairs = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    cases = select_camera_cases(pairs)
    if len(cases) != protocol["camera"]["calibration_cases"]:
        raise RuntimeError("camera case count mismatch")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "behavior_final_sha256": core.sha(BEHAVIOR / "analysis/final.json"),
        "behavior_audit_sha256": core.sha(BEHAVIOR / "audit/independent_final_audit.json"),
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "allowed_observables": protocol["allowed_observables"],
        "forbidden": protocol["forbidden"],
        "paths": protocol["paths"], "camera_gate": protocol["camera"],
        "known_truth_systems": protocol["camera"]["known_truth_systems"],
        "rows_per_case": 4 + len(protocol["paths"]) * 8,
        "target_layout_per_path": [
            "suff_self", "suff_correct", "suff_wrong", "suff_status",
            "necessity_self", "necessity_corrupt", "necessity_wrong", "necessity_status",
        ],
        "camera_case_ids": [row["pair_id"] for row in cases],
        "same_shape_as_phase1372": True,
        "serial_parallel_topology_rule": "query corrupt block separates serial from direct-parallel; boundary corrupt block affects both",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(manifest_path, manifest)
    core.write_rows(OUT / "material/camera_cases.jsonl", cases)
    print(json.dumps(manifest, indent=2))


def known_truth_calibration(count: int) -> tuple[list[dict], dict]:
    records = []
    for index in range(count):
        topology = "serial" if index % 2 == 0 else "parallel"
        clean, corrupt = 1.0, -1.0
        upstream_rescue = clean - corrupt
        query_corrupt_after_rescue = corrupt if topology == "serial" else clean
        boundary_corrupt_after_rescue = corrupt
        query_block_fraction = (clean - query_corrupt_after_rescue) / upstream_rescue
        boundary_block_fraction = (clean - boundary_corrupt_after_rescue) / upstream_rescue
        prediction = "serial" if query_block_fraction >= 0.5 else "parallel"
        records.append({
            "system_id": index, "topology": topology, "predicted_topology": prediction,
            "suff_gain": upstream_rescue, "necessity_damage": clean - corrupt,
            "query_block_fraction": query_block_fraction,
            "boundary_block_fraction": boundary_block_fraction,
            "topology_correct": prediction == topology,
        })

    dim, planted_count = 2560, 64
    rng = random.Random(5701369)
    planted = sorted(rng.sample(range(dim), planted_count))
    planted_set = set(planted)
    magnitude, signed_mean, family_abs = [0.0] * dim, [0.0] * dim, [[0.0] * dim for _ in range(4)]
    family_counts = [0] * 4
    for system in records:
        sid, family = system["system_id"], system["system_id"] % 4
        family_counts[family] += 1
        for coordinate in range(dim):
            if coordinate in planted_set:
                value = 2.0 + 0.001 * (coordinate % 7)
            elif coordinate < 192:
                value = (0.15 + 0.001 * (coordinate % 5)) * (1.0 if sid % 2 == 0 else -1.0)
            else:
                value = 0.0
            magnitude[coordinate] += abs(value)
            signed_mean[coordinate] += value
            family_abs[family][coordinate] += abs(value)
    magnitude_order = sorted(range(dim), key=lambda c: (-magnitude[c] / count, c))
    stable_order = sorted(range(dim), key=lambda c: (-abs(signed_mean[c] / count), c))
    family_min_order = sorted(range(dim), key=lambda c: (-min(family_abs[f][c] / family_counts[f] for f in range(4)), c))
    recovered = {
        "magnitude": set(magnitude_order[:planted_count]) == planted_set,
        "stable_sign": set(stable_order[:planted_count]) == planted_set,
        "family_min": set(family_min_order[:planted_count]) == planted_set,
    }
    summary = {
        "system_count": len(records),
        "serial_count": sum(row["topology"] == "serial" for row in records),
        "parallel_count": sum(row["topology"] == "parallel" for row in records),
        "topology_accuracy": sum(row["topology_correct"] for row in records) / len(records),
        "sufficiency_exact": all(row["suff_gain"] == 2.0 for row in records),
        "necessity_exact": all(row["necessity_damage"] == 2.0 for row in records),
        "coordinate_planted_count": planted_count,
        "coordinate_planted": planted,
        "coordinate_routes_recovered": recovered,
    }
    return records, summary


def make_batch(rows: list[dict], pad: int, device: torch.device):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    offsets = []
    for index, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        offset = width - len(value)
        offsets.append(offset)
        ids[index, offset:] = value
        mask[index, offset:] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, offsets


def copy_role(value, original, target_index: int, target: dict, target_offset: int,
              source_index: int, source: dict, source_offset: int, role: str) -> None:
    target_points = [target_offset + p for p in target["role_positions"][role]]
    source_points = [source_offset + p for p in source["role_positions"][role]]
    if len(target_points) != len(source_points):
        raise RuntimeError("role span mismatch")
    value[target_index, target_points] = original[source_index, source_points]


@torch.inference_mode()
def qwen_identity_camera(manifest: dict, cases: list[dict]) -> tuple[list[dict], dict, dict]:
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({row["case_id"]: row for row in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        paths = list(manifest["paths"].items())
        for case_index, case in enumerate(cases):
            donors = {key: compiled[case[key]] for key in DONOR_KEYS}
            rows = [donors[key] for key in DONOR_KEYS]
            for _name, _path in paths:
                rows.extend([donors["corrupt_false"]] * 4)
                rows.extend([donors["clean_true"]] * 4)
            ids, mask, positions, offsets = make_batch(rows, pad, device)
            handles = []
            try:
                for layer in sorted({path["source"]["layer"] for _name, path in paths}):
                    selected = [(i, name, path) for i, (name, path) in enumerate(paths)
                                if path["source"]["layer"] == layer]

                    def hook(_module, args, selected_paths=selected):
                        original = args[0]
                        value = original.clone()
                        for path_index, _name, path in selected_paths:
                            base = 4 + path_index * 8
                            for local in range(8):
                                target_index = base + local
                                source_index = 1 if local < 4 else 0
                                copy_role(value, original, target_index, rows[target_index], offsets[target_index],
                                          source_index, rows[source_index], offsets[source_index], path["source"]["role"])
                        return (value,) + args[1:]

                    handles.append(model.model.layers[layer].register_forward_pre_hook(hook))
                kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                          "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    kwargs["logits_to_keep"] = 1
                output = model(**kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            for path_index, (path_name, path) in enumerate(paths):
                base = 4 + path_index * 8
                for local, label in enumerate(manifest["target_layout_per_path"]):
                    target_index = base + local
                    source_index = 1 if local < 4 else 0
                    source = rows[source_index]
                    logits_diff = max(abs(float(output.logits[target_index, -1, ids_[0]].float() -
                                                output.logits[source_index, -1, ids_[0]].float()))
                                      for ids_ in source["candidate_ids"])
                    checkpoint_rel = []
                    for checkpoint in path["checkpoints"]:
                        layer, role = checkpoint["layer"], checkpoint["role"]
                        source_points = [offsets[source_index] + p for p in source["role_positions"][role]]
                        target_points = [offsets[target_index] + p for p in rows[target_index]["role_positions"][role]]
                        left = output.hidden_states[layer][source_index, source_points].float()
                        right = output.hidden_states[layer][target_index, target_points].float()
                        checkpoint_rel.append(float((right - left).norm() / (left.norm() + 1e-12)))
                    records.append({
                        "pair_id": case["pair_id"], "path": path_name, "arm": label,
                        "output_max_abs_diff": logits_diff,
                        "checkpoint_relative_l2_max": max(checkpoint_rel),
                    })
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"identity_camera": case_index + 1, "total": len(cases)}), flush=True)
            del output, ids, mask, positions
        summary = {
            "case_count": len(cases), "record_count": len(records),
            "output_max_abs_diff": max(row["output_max_abs_diff"] for row in records),
            "checkpoint_relative_l2_max": max(row["checkpoint_relative_l2_max"] for row in records),
            "all_finite": all(math.isfinite(row["output_max_abs_diff"]) and
                              math.isfinite(row["checkpoint_relative_l2_max"]) for row in records),
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
        raise RuntimeError("Phase1371 run already exists")
    known_records, known = known_truth_calibration(manifest["known_truth_systems"])
    camera_records, camera, runtime = qwen_identity_camera(manifest, core.rows(OUT / "material/camera_cases.jsonl"))
    gate = manifest["camera_gate"]
    checks = {
        "known_count": known["system_count"] == gate["known_truth_systems"],
        "topology_exact": known["topology_accuracy"] == 1.0,
        "bidirectional_exact": known["sufficiency_exact"] and known["necessity_exact"],
        "coordinate_routes": all(known["coordinate_routes_recovered"].values()),
        "camera_count": camera["case_count"] == gate["calibration_cases"],
        "same_shape_output": camera["output_max_abs_diff"] <= gate["same_shape_output_max_abs_diff"],
        "same_shape_checkpoint": camera["checkpoint_relative_l2_max"] <= gate["same_shape_checkpoint_relative_l2_max"],
        "finite": camera["all_finite"],
    }
    summary = {
        "phase": PHASE, "campaign": CAMPAIGN, "known_truth": known,
        "qwen_identity_camera": camera, "checks": checks,
        "camera_qualified": all(checks.values()), "runtime": runtime,
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
        "phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": qualified,
        "authorization": "run_phase1372_c057_whole_state_bidirectional" if qualified
                         else "close_c057_camera_unqualified_before_natural_causal_reveal",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    command = parser.parse_args().command
    if command == "prepare":
        prepare()
    elif command == "run":
        run()
    else:
        finalize()


if __name__ == "__main__":
    main()
