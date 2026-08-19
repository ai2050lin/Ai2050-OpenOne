#!/usr/bin/env python3
"""Phase1367: exact-shape self-write identity camera for every C056 path."""
from __future__ import annotations

import inspect
import json
import math
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

PHASE, CAMPAIGN = 1367, "C056"
CONTRACT = TESTS / "result/phase1364_c056_hidden_path_contract"
OBSERVATION = TESTS / "result/phase1366_c056_qwen_hidden_response_paths"
OUT = TESTS / "result/phase1367_c056_qwen_path_identity_camera"
MODEL = "qwen3"
ARMS = ("self", "correct_clean", "wrong_identity_true", "status_true")
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")


def parents() -> dict:
    final = core.load(OBSERVATION / "analysis/final.json")
    if final.get("authorization") != "run_phase1367_c056_qwen_path_identity_camera":
        raise RuntimeError("Phase1366 did not preserve the frozen camera branch")
    # Phase1366's numeric audit failed. The contract explicitly keeps the camera
    # branch alive after observation failure; this phase does not rehabilitate it.
    return core.load(CONTRACT / "protocol/preregistration.json")


def balanced_calibration(cases: list[dict], per_cell: int = 4) -> list[dict]:
    selected = []
    for partition in sorted({row["partition"] for row in cases}):
        for surface in sorted({row["surface"] for row in cases}):
            cell = sorted((row for row in cases
                           if row["partition"] == partition and row["surface"] == surface),
                          key=lambda row: row["pair_id"])
            if len(cell) < per_cell:
                raise RuntimeError(f"insufficient calibration cell: {partition}/{surface}")
            selected.extend(cell[:per_cell])
    return selected


def prepare() -> None:
    protocol = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1367 manifest already exists")
    cases = core.rows(CONTRACT / "material/path_cases.jsonl")
    selected = balanced_calibration(cases)
    if len(selected) != protocol["camera"]["calibration_cases"]:
        raise RuntimeError("calibration count mismatch")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "observation_parent_sha256": core.sha(OBSERVATION / "analysis/final.json"),
        "observation_audit_passed": False,
        "observation_failure_branch": protocol["observation"]["observation_failure_does_not_cancel_camera_or_causal"],
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "paths": protocol["paths"], "arms": list(ARMS), "donor_keys": list(DONOR_KEYS),
        "rows_per_case": len(DONOR_KEYS) + len(protocol["paths"]) * len(ARMS),
        "case_ids": [row["pair_id"] for row in selected],
        "case_count": len(selected), "gate": protocol["camera"],
        "same_execution_shape_as_phase1368": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    core.write_rows(OUT / "material/calibration_cases.jsonl", selected)
    print(json.dumps(manifest, indent=2))


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


def physical_positions(row: dict, role: str) -> list[int]:
    return list(row["role_positions"][role])


def copy_role(value, original, target_index: int, target: dict, target_offset: int,
              source_index: int, source: dict, source_offset: int, role: str) -> None:
    target_positions = [target_offset + p for p in physical_positions(target, role)]
    source_positions = [source_offset + p for p in physical_positions(source, role)]
    if len(target_positions) != len(source_positions):
        raise RuntimeError("source/target role span mismatch")
    value[target_index, target_positions] = original[source_index, source_positions]


def margin(logits: torch.Tensor, candidates: list[list[int]]) -> float:
    return float(logits[candidates[0][0]].float() - logits[candidates[1][0]].float())


@torch.inference_mode()
def run() -> None:
    protocol = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    cases = core.rows(OUT / "material/calibration_cases.jsonl")
    compiled = {row["case_id"]: row for row in core.rows(CONTRACT / "compiled/extended_rows.jsonl")}
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        path_items = list(manifest["paths"].items())
        for case_index, case in enumerate(cases):
            donors = {key: compiled[case[key]] for key in DONOR_KEYS}
            corrupt = donors["corrupt_false"]
            rows = [donors[key] for key in DONOR_KEYS]
            for _name, _path in path_items:
                rows.extend([corrupt] * len(ARMS))
            ids, mask, positions, offsets = make_batch(rows, pad, device)
            handles = []
            try:
                for layer_index in sorted({path["source"]["layer"] for _name, path in path_items}):
                    selected = [(index, name, path) for index, (name, path) in enumerate(path_items)
                                if path["source"]["layer"] == layer_index]

                    def hook(_module, args, selected_paths=selected):
                        original = args[0]
                        value = original.clone()
                        for path_index, _name, path in selected_paths:
                            base = len(DONOR_KEYS) + path_index * len(ARMS)
                            for arm_index in range(len(ARMS)):
                                target_index = base + arm_index
                                copy_role(value, original, target_index, corrupt, offsets[target_index],
                                          1, corrupt, offsets[1], path["source"]["role"])
                        return (value,) + args[1:]

                    handles.append(model.model.layers[layer_index].register_forward_pre_hook(hook))
                kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                          "use_cache": False, "output_hidden_states": True, "return_dict": True}
                if supports:
                    kwargs["logits_to_keep"] = 1
                output = model(**kwargs)
            finally:
                for handle in handles:
                    handle.remove()
            base_logits = output.logits[1, -1]
            base_margin = margin(base_logits, corrupt["candidate_ids"])
            for path_index, (name, path) in enumerate(path_items):
                base = len(DONOR_KEYS) + path_index * len(ARMS)
                for arm_index, arm in enumerate(ARMS):
                    target_index = base + arm_index
                    logits = output.logits[target_index, -1]
                    patched_margin = margin(logits, corrupt["candidate_ids"])
                    candidate_diff = max(abs(float(logits[item[0]].float() - base_logits[item[0]].float()))
                                         for item in corrupt["candidate_ids"])
                    for checkpoint in path["checkpoints"]:
                        layer_index, role = checkpoint["layer"], checkpoint["role"]
                        base_state = output.hidden_states[layer_index][1]
                        patched_state = output.hidden_states[layer_index][target_index]
                        base_points = [offsets[1] + p for p in physical_positions(corrupt, role)]
                        patch_points = [offsets[target_index] + p for p in physical_positions(corrupt, role)]
                        left = base_state[base_points].float()
                        right = patched_state[patch_points].float()
                        relative = float((right - left).norm() / (left.norm() + 1e-12))
                        records.append({
                            "pair_id": case["pair_id"], "partition": case["partition"],
                            "surface": case["surface"], "path": name, "arm": arm,
                            "checkpoint_layer": layer_index, "checkpoint_role": role,
                            "baseline_margin": base_margin, "patched_margin": patched_margin,
                            "margin_diff": patched_margin - base_margin,
                            "candidate_logit_max_abs_diff": candidate_diff,
                            "checkpoint_relative_l2": relative,
                        })
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"camera_cases": case_index + 1, "total": len(cases)}), flush=True)
            del output, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_path_identity_camera.jsonl", records)
        metrics, checks = {}, {}
        gate = manifest["gate"]
        for name, _path in path_items:
            values = [row for row in records if row["path"] == name]
            metrics[name] = {
                "records": len(values),
                "max_abs_margin_diff": max(abs(row["margin_diff"]) for row in values),
                "max_candidate_logit_abs_diff": max(row["candidate_logit_max_abs_diff"] for row in values),
                "max_checkpoint_relative_l2": max(row["checkpoint_relative_l2"] for row in values),
                "median_checkpoint_relative_l2": statistics.median(row["checkpoint_relative_l2"] for row in values),
            }
            checks[name] = {
                "output": metrics[name]["max_abs_margin_diff"] <= gate["output_margin_max_abs_diff"],
                "checkpoint": metrics[name]["max_checkpoint_relative_l2"] <= gate["checkpoint_relative_l2_max"],
            }
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
            "metrics": metrics, "checks": checks,
            "camera_qualified": all(all(value.values()) for value in checks.values()),
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(row["margin_diff"]) and
                                          math.isfinite(row["checkpoint_relative_l2"]) for row in records),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "same-input source-write identity under exact Phase1368 batch shape",
        }
        core.save(OUT / "analysis/qwen3_path_identity_camera.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_path_identity_camera.json")
    qualified = bool(summary["camera_qualified"])
    final = {
        "phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": qualified,
        "authorization": "run_phase1368_c056_all_path_causal_competition" if qualified
                         else "close_c056_camera_unqualified_without_mechanism_claim",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        finalize()


if __name__ == "__main__":
    main()
