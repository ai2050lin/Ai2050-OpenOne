#!/usr/bin/env python3
"""Phase1362: calibrate exact same-batch self transport for every C055 role coalition."""
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

PHASE, CAMPAIGN = 1362, "C055"
CONTRACT = TESTS / "result/phase1360_c055_hidden_state_coalition_contract"
OBSERVATION = TESTS / "result/phase1361_c055_full_width_coalition_observation"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
C054_CONTRACT = TESTS / "result/phase1357_c054_same_batch_causal_contract"
OUT = TESTS / "result/phase1362_c055_coalition_identity_camera"
MODEL = "qwen3"


def parents() -> tuple[dict, dict]:
    final = core.load(OBSERVATION / "analysis/final.json")
    audit = core.load(OBSERVATION / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1362_c055_coalition_camera" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1361 did not authorize the camera")
    return core.load(CONTRACT / "protocol/preregistration.json"), core.load(OBSERVATION / "analysis/coalition_observation.json")


def prepare() -> None:
    protocol, observation = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1362 manifest already exists")
    selected = observation["selected_descriptive_layer"]
    layers = sorted(set([27] + ([int(selected)] if selected is not None else [])))
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "observation_parent_sha256": core.sha(OBSERVATION / "analysis/final.json"), "model": MODEL,
        "precision": "bfloat16-no-quantization", "layers": layers,
        "coalitions": protocol["coalitions"], "calibration_cases": protocol["camera"]["calibration_cases"],
        "batch_sources": 4, "rows_per_source": 2,
        "threshold": protocol["camera"]["same_batch_exact_self_max_abs_margin"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps(manifest, indent=2))


def make_batch(rows, width, pad, device):
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def role_positions(row: dict, role: str) -> list[int]:
    if role == "target":
        return row["target_span"]
    if role == "family":
        return row["tested_family_span"]
    if role == "boundary":
        return [row["boundary_position"]]
    raise RuntimeError(role)


def record(route: str, layer: int, meta: dict, row: dict, output, base: int) -> dict:
    left, right = output.logits[base, -1].float(), output.logits[base + 1, -1].float()
    candidates = row["candidate_ids"]
    lm = float(left[candidates[0][0]] - left[candidates[1][0]])
    rm = float(right[candidates[0][0]] - right[candidates[1][0]])
    return {**meta, "coalition": route, "layer": layer, "baseline_margin": lm,
            "patched_margin": rm, "margin_diff": rm - lm,
            "candidate_logit_max_abs_diff": max(abs(float(left[x[0]] - right[x[0]])) for x in candidates)}


@torch.inference_mode()
def run() -> None:
    protocol, _observation = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    calibration = core.rows(C054_CONTRACT / "material/calibration_cases.jsonl")
    compiled = core.rows(C053 / "compiled/qwen3_B1_binary.jsonl")
    compiled_by_id = {row["case_id"]: row for row in compiled}
    width = max(len(row["prompt_ids"]) for row in compiled)
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for start in range(0, len(calibration), manifest["batch_sources"]):
            meta_group = calibration[start:start + manifest["batch_sources"]]
            source_rows = [compiled_by_id[meta["case_id"]] for meta in meta_group]
            expanded = [row for source in source_rows for row in (source, source)]
            ids, mask, positions = make_batch(expanded, width, pad, device)
            base_kw = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                       "use_cache": False, "return_dict": True}
            if supports:
                base_kw["logits_to_keep"] = 1
            nohook = model(**base_kw)
            for layer_index in manifest["layers"]:
                for local, (meta, row) in enumerate(zip(meta_group, source_rows)):
                    records.append(record("duplicate_no_hook", layer_index, meta, row, nohook, local * 2))
                layer = model.model.layers[layer_index]
                for name, roles in manifest["coalitions"].items():
                    def hook(_module, args, selected_roles=roles):
                        original = args[0]
                        value = original.clone()
                        for local, row in enumerate(source_rows):
                            source_index, target_index = local * 2, local * 2 + 1
                            for role in selected_roles:
                                span = role_positions(row, role)
                                value[target_index, span] = original[source_index, span]
                        return (value,) + args[1:]

                    handle = layer.register_forward_pre_hook(hook)
                    try:
                        output = model(**base_kw)
                    finally:
                        handle.remove()
                    for local, (meta, row) in enumerate(zip(meta_group, source_rows)):
                        records.append(record(name, layer_index, meta, row, output, local * 2))
                    del output
            del nohook, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_coalition_identity.jsonl", records)
        metrics = {}
        routes = ["duplicate_no_hook"] + list(manifest["coalitions"])
        for route in routes:
            metrics[route] = {}
            for layer in manifest["layers"]:
                values = [row for row in records if row["coalition"] == route and row["layer"] == layer]
                metrics[route][str(layer)] = {
                    "count": len(values),
                    "max_abs_margin_diff": max(abs(row["margin_diff"]) for row in values),
                    "median_abs_margin_diff": statistics.median(abs(row["margin_diff"]) for row in values),
                    "max_candidate_logit_abs_diff": max(row["candidate_logit_max_abs_diff"] for row in values),
                }
        checks = {route: all(metrics[route][str(layer)]["max_abs_margin_diff"] <= manifest["threshold"]
                             for layer in manifest["layers"]) for route in routes}
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
            "metrics": metrics, "checks": checks, "camera_qualified": all(checks.values()),
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(row["margin_diff"]) for row in records),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "identity calibration for exact hidden-state role sets only",
        }
        core.save(OUT / "analysis/qwen3_coalition_camera.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_coalition_camera.json")
    final = {"phase": PHASE, "campaign": CAMPAIGN, "camera_qualified": summary["camera_qualified"],
             "authorization": "run_phase1363_c055_coalition_causal" if summary["camera_qualified"]
                              else "close_c055_camera_unqualified_without_mechanism_claim",
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
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
