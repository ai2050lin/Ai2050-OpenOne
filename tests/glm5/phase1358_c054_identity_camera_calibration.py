#!/usr/bin/env python3
"""Phase1358: calibrate C054 same-batch and cached identity intervention routes."""
from __future__ import annotations

import inspect
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1358, "C054"
CONTRACT = TESTS / "result/phase1357_c054_same_batch_causal_contract"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
OUT = TESTS / "result/phase1358_c054_identity_camera_calibration"
MODEL = "qwen3"
ROUTES = ("duplicate_no_hook", "same_batch_exact_token", "cached_fixed_shape_exact_token",
          "same_batch_span_mean_diagnostic", "same_batch_zero_delta")


def parent() -> dict:
    final = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1358_c054_camera_calibration" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1357 did not authorize camera calibration")
    return core.load(CONTRACT / "protocol/preregistration.json")


def prepare() -> None:
    protocol = parent()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1358 manifest already exists")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "contract_parent_sha256": core.sha(CONTRACT / "analysis/final.json"), "model": MODEL,
        "precision": "bfloat16-no-quantization", "layers": protocol["camera_calibration"]["layers"],
        "routes": protocol["camera_calibration"]["routes"],
        "authorized_priority": protocol["camera_calibration"]["authorized_priority"],
        "batch_sources": protocol["camera_calibration"]["batch_sources"],
        "case_count": protocol["material"]["calibration_cases"],
        "gate": protocol["camera_calibration"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps(manifest, indent=2))


def make_batch(rows: list[dict], width: int, pad: int, device):
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


def forward(model, ids, mask, positions, supports: bool, hidden: bool = False):
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
              "use_cache": False, "output_hidden_states": hidden, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs)


def pair_record(route: str, layer: int, meta: dict, compiled: dict, output, base: int) -> dict:
    candidate_ids = compiled["candidate_ids"]
    left = output.logits[base, -1].float()
    right = output.logits[base + 1, -1].float()
    left_margin = float(left[candidate_ids[0][0]] - left[candidate_ids[1][0]])
    right_margin = float(right[candidate_ids[0][0]] - right[candidate_ids[1][0]])
    candidate_diff = max(abs(float(left[value[0]] - right[value[0]])) for value in candidate_ids)
    return {
        **meta, "route": route, "layer": layer,
        "baseline_margin": left_margin, "patched_margin": right_margin,
        "margin_diff": right_margin - left_margin, "candidate_logit_max_abs_diff": candidate_diff,
    }


@torch.inference_mode()
def run() -> None:
    protocol = parent()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    calibration = core.rows(CONTRACT / "material/calibration_cases.jsonl")
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
        group_size = manifest["batch_sources"]
        for start in range(0, len(calibration), group_size):
            meta_group = calibration[start:start + group_size]
            source_rows = [compiled_by_id[row["case_id"]] for row in meta_group]
            expanded = [row for source in source_rows for row in (source, source)]
            ids, mask, positions = make_batch(expanded, width, pad, device)
            base_output = forward(model, ids, mask, positions, supports, hidden=True)
            for layer_index in manifest["layers"]:
                for local, (meta, row) in enumerate(zip(meta_group, source_rows)):
                    records.append(pair_record("duplicate_no_hook", layer_index, meta, row, base_output, local * 2))

                layer = model.model.layers[layer_index]
                cached = base_output.hidden_states[layer_index]

                def run_hook(route: str):
                    def hook(_module, args):
                        original = args[0]
                        value = original.clone()
                        for local, row in enumerate(source_rows):
                            source_index, target_index = local * 2, local * 2 + 1
                            span = row["tested_family_span"]
                            if route == "same_batch_exact_token":
                                value[target_index, span] = original[source_index, span]
                            elif route == "cached_fixed_shape_exact_token":
                                value[target_index, span] = cached[source_index, span]
                            elif route == "same_batch_span_mean_diagnostic":
                                mean = original[source_index, span].mean(0)
                                value[target_index, span] = mean
                            elif route == "same_batch_zero_delta":
                                delta = original[source_index, span] - original[source_index, span]
                                value[target_index, span] = original[target_index, span] + delta
                            else:
                                raise RuntimeError(route)
                        return (value,) + args[1:]

                    handle = layer.register_forward_pre_hook(hook)
                    try:
                        return forward(model, ids, mask, positions, supports, hidden=False)
                    finally:
                        handle.remove()

                for route in ROUTES[1:]:
                    output = run_hook(route)
                    for local, (meta, row) in enumerate(zip(meta_group, source_rows)):
                        records.append(pair_record(route, layer_index, meta, row, output, local * 2))
                    del output
            del base_output, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_identity_camera.jsonl", records)
        route_metrics = {}
        for route in ROUTES:
            route_metrics[route] = {}
            for layer in manifest["layers"]:
                values = [row for row in records if row["route"] == route and row["layer"] == layer]
                route_metrics[route][str(layer)] = {
                    "count": len(values),
                    "max_abs_margin_diff": max(abs(row["margin_diff"]) for row in values),
                    "median_abs_margin_diff": statistics.median(abs(row["margin_diff"]) for row in values),
                    "max_candidate_logit_abs_diff": max(row["candidate_logit_max_abs_diff"] for row in values),
                    "single_token_max_abs_margin_diff": max(abs(row["margin_diff"]) for row in values if row["span_length"] == 1),
                    "double_token_max_abs_margin_diff": max(abs(row["margin_diff"]) for row in values if row["span_length"] == 2),
                }
        gate = manifest["gate"]
        thresholds = {
            "duplicate_no_hook": gate["no_hook_max_abs_margin"],
            "same_batch_exact_token": gate["same_batch_exact_max_abs_margin"],
            "cached_fixed_shape_exact_token": gate["cached_fixed_shape_max_abs_margin"],
            "same_batch_zero_delta": gate["zero_delta_max_abs_margin"],
        }
        route_checks = {
            route: all(route_metrics[route][str(layer)]["max_abs_margin_diff"] <= threshold
                       for layer in manifest["layers"])
            for route, threshold in thresholds.items()
        }
        selected = next((route for route in manifest["authorized_priority"] if route_checks[route]), None)
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
            "route_metrics": route_metrics, "route_checks": route_checks,
            "selected_camera_route": selected, "camera_qualified": selected is not None and route_checks["duplicate_no_hook"]
                                and route_checks["same_batch_zero_delta"],
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(row["margin_diff"]) for row in records),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "known-truth identity calibration; the span-mean route is diagnostic only",
        }
        core.save(OUT / "analysis/qwen3_camera_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    protocol = parent()
    summary = core.load(OUT / "analysis/qwen3_camera_summary.json")
    qualified = bool(summary["camera_qualified"])
    final = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "camera_qualified": qualified, "selected_camera_route": summary["selected_camera_route"],
        "authorization": "run_phase1359_c054_same_batch_causal_replay" if qualified
                         else "close_c054_camera_unqualified_without_mechanism_claim",
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
