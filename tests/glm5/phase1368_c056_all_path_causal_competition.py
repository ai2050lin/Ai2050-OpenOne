#!/usr/bin/env python3
"""Phase1368: all-path single-write Hidden-State cascade competition for C056."""
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

PHASE, CAMPAIGN = 1368, "C056"
CONTRACT = TESTS / "result/phase1364_c056_hidden_path_contract"
CAMERA = TESTS / "result/phase1367_c056_qwen_path_identity_camera"
OUT = TESTS / "result/phase1368_c056_all_path_causal_competition"
MODEL = "qwen3"
ARMS = ("self", "correct_clean", "wrong_identity_true", "status_true")
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
ARM_TO_DONOR_ROW = {"self": 1, "correct_clean": 0, "wrong_identity_true": 2, "status_true": 3}


def parents() -> dict:
    final = core.load(CAMERA / "analysis/final.json")
    audit = core.load(CAMERA / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1368_c056_all_path_causal_competition" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1367 did not authorize causal competition")
    return core.load(CONTRACT / "protocol/preregistration.json")


def prepare() -> None:
    protocol = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1368 manifest already exists")
    cases = core.rows(CONTRACT / "material/path_cases.jsonl")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "camera_parent_sha256": core.sha(CAMERA / "analysis/final.json"),
        "camera_audit_sha256": core.sha(CAMERA / "audit/independent_final_audit.json"),
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "paths": protocol["paths"], "arms": list(ARMS), "donor_keys": list(DONOR_KEYS),
        "arm_to_donor_row": ARM_TO_DONOR_ROW,
        "rows_per_case": len(DONOR_KEYS) + len(protocol["paths"]) * len(ARMS),
        "case_ids": [row["pair_id"] for row in cases], "case_count": len(cases),
        "gate": protocol["causal"], "all_paths_run_even_after_failures": True,
        "single_write_only": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
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


def role_positions(row: dict, role: str) -> list[int]:
    return list(row["role_positions"][role])


def copy_role(value, original, target_index: int, target: dict, target_offset: int,
              source_index: int, source: dict, source_offset: int, role: str) -> None:
    target_points = [target_offset + p for p in role_positions(target, role)]
    source_points = [source_offset + p for p in role_positions(source, role)]
    if len(target_points) != len(source_points):
        raise RuntimeError("source/recipient role span mismatch")
    value[target_index, target_points] = original[source_index, source_points]


def margin(logits: torch.Tensor, candidates: list[list[int]]) -> float:
    return float(logits[candidates[0][0]].float() - logits[candidates[1][0]].float())


def role_state(output, layer: int, row_index: int, row: dict, offset: int, role: str) -> torch.Tensor:
    points = [offset + p for p in role_positions(row, role)]
    return output.hidden_states[layer][row_index, points].float().flatten()


def projection(patched: torch.Tensor, corrupt: torch.Tensor, clean: torch.Tensor) -> float:
    target = clean - corrupt
    return float(torch.dot(patched - corrupt, target) / (torch.dot(target, target) + 1e-12))


def checkpoint_metrics(records: list[dict], path_name: str, checkpoint: dict) -> dict:
    values = [row for row in records if row["path"] == path_name]
    key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
    correct = [row["checkpoint_alpha"][key]["correct_clean"] for row in values]
    wrong = [row["checkpoint_alpha"][key]["wrong_identity_true"] for row in values]
    status = [row["checkpoint_alpha"][key]["status_true"] for row in values]
    advantages = [c - max(w, s) for c, w, s in zip(correct, wrong, status)]
    wins = [c > max(w, s) for c, w, s in zip(correct, wrong, status)]
    self_l2 = [row["self_checkpoint_relative_l2"][key] for row in values]
    return {
        "count": len(values), "correct_projection_median": statistics.median(correct),
        "correct_over_controls_median": statistics.median(advantages),
        "correct_over_controls_win_fraction": sum(wins) / len(wins),
        "self_relative_l2_max": max(self_l2),
    }


def output_metrics(records: list[dict], path_name: str) -> dict:
    values = [row for row in records if row["path"] == path_name]
    correct = [row["output_gain"]["correct_clean"] for row in values]
    wrong = [row["output_gain"]["wrong_identity_true"] for row in values]
    status = [row["output_gain"]["status_true"] for row in values]
    advantages = [c - max(w, s) for c, w, s in zip(correct, wrong, status)]
    wins = [c > max(w, s) for c, w, s in zip(correct, wrong, status)]
    subgroup = {}
    for field in ("partition", "surface", "family_pair"):
        subgroup[field] = {}
        for name in sorted({row[field] for row in values}):
            indexes = [i for i, row in enumerate(values) if row[field] == name]
            subgroup[field][name] = {
                "count": len(indexes),
                "correct_gain_median": statistics.median(correct[i] for i in indexes),
                "win_fraction": sum(wins[i] for i in indexes) / len(indexes),
            }
    return {
        "count": len(values), "correct_gain_median": statistics.median(correct),
        "correct_over_controls_median": statistics.median(advantages),
        "correct_over_controls_win_fraction": sum(wins) / len(wins),
        "self_max_abs_diff": max(abs(row["output_gain"]["self"]) for row in values),
        "subgroups": subgroup,
    }


@torch.inference_mode()
def run() -> None:
    protocol = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    cases = core.rows(CONTRACT / "material/path_cases.jsonl")
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
                            for arm_index, arm in enumerate(ARMS):
                                target_index = base + arm_index
                                source_index = manifest["arm_to_donor_row"][arm]
                                source = rows[source_index]
                                copy_role(value, original, target_index, corrupt, offsets[target_index],
                                          source_index, source, offsets[source_index], path["source"]["role"])
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

            baseline = margin(output.logits[1, -1], corrupt["candidate_ids"])
            for path_index, (name, path) in enumerate(path_items):
                base = len(DONOR_KEYS) + path_index * len(ARMS)
                gains = {}
                alphas = {f'{cp["role"]}@{cp["layer"]}': {} for cp in path["checkpoints"]}
                self_l2 = {}
                for arm_index, arm in enumerate(ARMS):
                    target_index = base + arm_index
                    gains[arm] = margin(output.logits[target_index, -1], corrupt["candidate_ids"]) - baseline
                    for checkpoint in path["checkpoints"]:
                        layer_index, role = checkpoint["layer"], checkpoint["role"]
                        clean_state = role_state(output, layer_index, 0, donors["clean_true"], offsets[0], role)
                        corrupt_state = role_state(output, layer_index, 1, corrupt, offsets[1], role)
                        patched_state = role_state(output, layer_index, target_index, corrupt,
                                                   offsets[target_index], role)
                        key = f"{role}@{layer_index}"
                        alphas[key][arm] = projection(patched_state, corrupt_state, clean_state)
                        if arm == "self":
                            self_l2[key] = float((patched_state - corrupt_state).norm() /
                                                 (corrupt_state.norm() + 1e-12))
                records.append({
                    "pair_id": case["pair_id"], "partition": case["partition"],
                    "surface": case["surface"], "family_pair": case["family_pair"],
                    "direction": case["direction"], "path": name,
                    "baseline_margin": baseline, "output_gain": gains,
                    "checkpoint_alpha": alphas, "self_checkpoint_relative_l2": self_l2,
                })
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"causal_cases": case_index + 1, "total": len(cases)}), flush=True)
            del output, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_all_path_causal.jsonl", records)
        gate = manifest["gate"]
        path_metrics, path_checks, path_qualified = {}, {}, {}
        for name, path in path_items:
            checkpoint = {f'{cp["role"]}@{cp["layer"]}': checkpoint_metrics(records, name, cp)
                          for cp in path["checkpoints"]}
            out = output_metrics(records, name)
            checkpoint_checks = {}
            for key, value in checkpoint.items():
                checkpoint_checks[key] = {
                    "projection": value["correct_projection_median"] >= gate["checkpoint_recovery_projection_median_min"],
                    "advantage": value["correct_over_controls_median"] >= gate["checkpoint_correct_over_controls_median_min"],
                    "win": value["correct_over_controls_win_fraction"] >= gate["checkpoint_correct_over_controls_win_min"],
                    "self": value["self_relative_l2_max"] <= gate["self_checkpoint_relative_l2_max"],
                }
            out_checks = {
                "gain": out["correct_gain_median"] >= gate["output_gain_median_min"],
                "advantage": out["correct_over_controls_median"] >= gate["output_correct_over_controls_median_min"],
                "win": out["correct_over_controls_win_fraction"] >= gate["output_correct_over_controls_win_min"],
                "self": out["self_max_abs_diff"] <= gate["self_output_max_abs_diff"],
            }
            path_metrics[name] = {"checkpoints": checkpoint, "output": out}
            path_checks[name] = {"checkpoints": checkpoint_checks, "output": out_checks}
            path_qualified[name] = all(all(v.values()) for v in checkpoint_checks.values()) and all(out_checks.values())
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
            "path_metrics": path_metrics, "path_checks": path_checks,
            "path_qualified": path_qualified,
            "qualified_paths": [name for name, value in path_qualified.items() if value],
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(value) for row in records
                                          for value in row["output_gain"].values()) and
                                      all(math.isfinite(value) for row in records
                                          for cp in row["checkpoint_alpha"].values() for value in cp.values()),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "calibrated single-source Hidden-State sufficiency/selectivity along five frozen paths",
        }
        core.save(OUT / "analysis/qwen3_all_path_causal.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_all_path_causal.json")
    final = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "qualified_paths": summary["qualified_paths"],
        "authorization": "close_c056_after_frozen_all_path_competition",
        "campaign_closed": True,
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
