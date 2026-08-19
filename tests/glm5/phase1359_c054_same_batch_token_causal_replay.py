#!/usr/bin/env python3
"""Phase1359: replay the frozen C053 relation candidate with a calibrated same-batch camera."""
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

PHASE, CAMPAIGN = 1359, "C054"
CONTRACT = TESTS / "result/phase1357_c054_same_batch_causal_contract"
CAMERA = TESTS / "result/phase1358_c054_identity_camera_calibration"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
OUT = TESTS / "result/phase1359_c054_same_batch_token_causal_replay"
MODEL = "qwen3"
RECIPIENT_ARMS = ("baseline", "self", "state_correct", "state_wrong_true", "state_same_false",
                  "delta_correct", "delta_wrong", "zero_delta")
DONOR_KEYS = ("correct_true", "correct_false", "wrong_true", "wrong_false")


def parents() -> tuple[dict, dict]:
    final = core.load(CAMERA / "analysis/final.json")
    audit = core.load(CAMERA / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1359_c054_same_batch_causal_replay" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1358 did not authorize natural replay")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if final.get("selected_camera_route") != "same_batch_exact_token":
        raise RuntimeError("the frozen natural implementation requires the qualified same-batch route")
    return protocol, final


def prepare() -> None:
    protocol, camera = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1359 manifest already exists")
    entries = core.rows(CONTRACT / "material/causal_replay_manifest.jsonl")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "camera_parent_sha256": core.sha(CAMERA / "analysis/final.json"), "model": MODEL,
        "precision": "bfloat16-no-quantization", "selected_camera_route": camera["selected_camera_route"],
        "layer": protocol["causal_replay"]["layer"], "site_role": protocol["causal_replay"]["site_role"],
        "recipient_arms": list(RECIPIENT_ARMS), "donor_keys": list(DONOR_KEYS),
        "rows_per_entry": len(RECIPIENT_ARMS) + len(DONOR_KEYS),
        "batch_recipients": protocol["causal_replay"]["batch_recipients"],
        "recipient_count": len(entries), "routes": list(protocol["causal_replay"]["routes"]),
        "gate": protocol["causal_replay"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
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


def margin(logits, candidates: list[list[int]]) -> float:
    return float(logits[candidates[0][0]].float() - logits[candidates[1][0]].float())


def route_metrics(records: list[dict], route: str) -> dict:
    if route == "state_transport":
        correct = [row["gains"]["state_correct"] for row in records]
        wrong_sets = [[row["gains"]["state_wrong_true"], row["gains"]["state_same_false"]] for row in records]
    elif route == "paired_delta_transport":
        correct = [row["gains"]["delta_correct"] for row in records]
        wrong_sets = [[row["gains"]["delta_wrong"]] for row in records]
    else:
        raise RuntimeError(route)
    advantages = [value - max(wrongs) for value, wrongs in zip(correct, wrong_sets)]
    wins = [value > max(wrongs) for value, wrongs in zip(correct, wrong_sets)]
    subgroup = {}
    for key in ("partition", "surface", "recipient_tested_family"):
        subgroup[key] = {}
        for value in sorted({row[key] for row in records}):
            indexes = [index for index, row in enumerate(records) if row[key] == value]
            subgroup[key][value] = {
                "count": len(indexes),
                "direction_fraction": sum(correct[index] > 0 for index in indexes) / len(indexes),
                "median_gain": statistics.median(correct[index] for index in indexes),
                "win_fraction": sum(wins[index] for index in indexes) / len(indexes),
            }
    return {
        "count": len(records), "correct_gain_median": statistics.median(correct),
        "correct_direction_fraction": sum(value > 0 for value in correct) / len(correct),
        "correct_over_wrong_median": statistics.median(advantages),
        "correct_over_wrong_win_fraction": sum(wins) / len(wins),
        "subgroups": subgroup,
    }


@torch.inference_mode()
def run() -> None:
    protocol, _camera = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    entries = core.rows(CONTRACT / "material/causal_replay_manifest.jsonl")
    source = core.rows(C053 / "material/b1_binary_cases.jsonl")
    compiled = core.rows(C053 / "compiled/qwen3_B1_binary.jsonl")
    source_by_id = {row["case_id"]: row for row in source}
    compiled_by_id = {row["case_id"]: row for row in compiled}
    width = max(len(row["prompt_ids"]) for row in compiled)
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        layer = model.model.layers[manifest["layer"]]
        records = []
        batch_recipients = manifest["batch_recipients"]
        rows_per_entry = manifest["rows_per_entry"]
        for start in range(0, len(entries), batch_recipients):
            group = entries[start:start + batch_recipients]
            batch_rows = []
            for entry in group:
                recipient = compiled_by_id[entry["recipient"]]
                batch_rows.extend([recipient] * len(RECIPIENT_ARMS))
                batch_rows.extend(compiled_by_id[entry[key]] for key in DONOR_KEYS)
            ids, mask, positions = make_batch(batch_rows, width, pad, device)

            def hook(_module, args):
                original = args[0]
                value = original.clone()
                for local, entry in enumerate(group):
                    base = local * rows_per_entry
                    recipient = compiled_by_id[entry["recipient"]]
                    recipient_pos = recipient["tested_family_span"][0]
                    donor_row = {key: base + len(RECIPIENT_ARMS) + index for index, key in enumerate(DONOR_KEYS)}
                    donor_pos = {key: compiled_by_id[entry[key]]["tested_family_span"][0] for key in DONOR_KEYS}
                    value[base + 1, recipient_pos] = original[base, recipient_pos]
                    value[base + 2, recipient_pos] = original[donor_row["correct_true"], donor_pos["correct_true"]]
                    value[base + 3, recipient_pos] = original[donor_row["wrong_true"], donor_pos["wrong_true"]]
                    value[base + 4, recipient_pos] = original[donor_row["correct_false"], donor_pos["correct_false"]]
                    correct_delta = (original[donor_row["correct_true"], donor_pos["correct_true"]]
                                     - original[donor_row["correct_false"], donor_pos["correct_false"]])
                    wrong_delta = (original[donor_row["wrong_true"], donor_pos["wrong_true"]]
                                   - original[donor_row["wrong_false"], donor_pos["wrong_false"]])
                    value[base + 5, recipient_pos] = original[base + 5, recipient_pos] + correct_delta
                    value[base + 6, recipient_pos] = original[base + 6, recipient_pos] + wrong_delta
                    zero = original[base, recipient_pos] - original[base, recipient_pos]
                    value[base + 7, recipient_pos] = original[base + 7, recipient_pos] + zero
                return (value,) + args[1:]

            handle = layer.register_forward_pre_hook(hook)
            try:
                kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                          "use_cache": False, "return_dict": True}
                if supports:
                    kwargs["logits_to_keep"] = 1
                output = model(**kwargs)
            finally:
                handle.remove()

            for local, entry in enumerate(group):
                base = local * rows_per_entry
                recipient = compiled_by_id[entry["recipient"]]
                margins = {arm: margin(output.logits[base + index, -1], recipient["candidate_ids"])
                           for index, arm in enumerate(RECIPIENT_ARMS)}
                baseline = margins["baseline"]
                gains = {arm: value - baseline for arm, value in margins.items()}
                source_row = source_by_id[entry["recipient"]]
                records.append({**entry, "quartet_key": source_row["quartet_key"],
                                "margins": margins, "gains": gains})
            del output, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_same_batch_causal.jsonl", records)
        metrics = {route: route_metrics(records, route) for route in manifest["routes"]}
        identity = {
            "self_max_abs_diff": max(abs(row["gains"]["self"]) for row in records),
            "zero_delta_max_abs_diff": max(abs(row["gains"]["zero_delta"]) for row in records),
        }
        gate = manifest["gate"]
        identity_checks = {
            "self": identity["self_max_abs_diff"] <= gate["self_max_abs_diff_max"],
            "zero_delta": identity["zero_delta_max_abs_diff"] <= gate["zero_delta_max_abs_diff_max"],
        }
        route_checks = {}
        route_qualified = {}
        for route, values in metrics.items():
            route_checks[route] = {
                "gain": values["correct_gain_median"] >= gate["false_to_true_gain_min"],
                "direction": values["correct_direction_fraction"] >= gate["direction_fraction_min"],
                "selective_median": values["correct_over_wrong_median"] >= gate["correct_over_wrong_median_min"],
                "selective_win": values["correct_over_wrong_win_fraction"] >= gate["correct_over_wrong_win_min"],
                "identity": all(identity_checks.values()),
            }
            route_qualified[route] = all(route_checks[route].values())
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL, "layer": manifest["layer"],
            "identity": identity, "identity_checks": identity_checks,
            "route_metrics": metrics, "route_checks": route_checks, "route_qualified": route_qualified,
            "any_route_qualified": any(route_qualified.values()),
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(value) for row in records
                                          for values in (row["margins"], row["gains"]) for value in values.values()),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "Qwen-specific causal sufficiency/selectivity for two frozen token-level transports at layer 27",
        }
        core.save(OUT / "analysis/qwen3_causal_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    protocol, _camera = parents()
    summary = core.load(OUT / "analysis/qwen3_causal_summary.json")
    final = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "camera_identity_passed": all(summary["identity_checks"].values()),
        "qualified_routes": [route for route, value in summary["route_qualified"].items() if value],
        "authorization": "close_c054_with_calibrated_causal_candidate" if summary["any_route_qualified"]
                         else "close_c054_at_calibrated_causal_selectivity_boundary",
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
