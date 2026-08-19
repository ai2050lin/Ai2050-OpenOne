#!/usr/bin/env python3
"""Phase1363: causal competition for all frozen C055 hidden-state role coalitions."""
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

PHASE, CAMPAIGN = 1363, "C055"
CONTRACT = TESTS / "result/phase1360_c055_hidden_state_coalition_contract"
OBSERVATION = TESTS / "result/phase1361_c055_full_width_coalition_observation"
CAMERA = TESTS / "result/phase1362_c055_coalition_identity_camera"
C053 = TESTS / "result/phase1353_c053_route_portfolio_contract"
OUT = TESTS / "result/phase1363_c055_hidden_state_coalition_causal"
MODEL = "qwen3"
ARMS = ("self", "correct_true", "wrong_family_true", "same_family_false", "status_true")
DONOR_KEYS = ("correct_true", "wrong_true", "correct_false", "status_true")
CONSTITUENTS = {
    "target_family": ("target", "family"),
    "target_boundary": ("target", "boundary"),
    "family_boundary": ("family", "boundary"),
    "all_roles": ("target", "family", "boundary"),
}


def parents() -> tuple[dict, dict]:
    final = core.load(CAMERA / "analysis/final.json")
    audit = core.load(CAMERA / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1363_c055_coalition_causal" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1362 did not authorize causal work")
    return core.load(CONTRACT / "protocol/preregistration.json"), core.load(OBSERVATION / "analysis/coalition_observation.json")


def prepare() -> None:
    protocol, observation = parents()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1363 manifest already exists")
    entries = core.rows(CONTRACT / "material/causal_replay_manifest.jsonl")
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "camera_parent_sha256": core.sha(CAMERA / "analysis/final.json"), "model": MODEL,
        "precision": "bfloat16-no-quantization", "layer": observation["causal_layer"],
        "coalitions": protocol["coalitions"], "arms": list(ARMS), "donor_keys": list(DONOR_KEYS),
        "rows_per_entry": 1 + len(protocol["coalitions"]) * len(ARMS) + len(DONOR_KEYS),
        "recipient_count": len(entries), "gate": protocol["causal"],
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


def copy_roles(value, original, target_row: int, target: dict, source_row: int, source: dict, roles: list[str]):
    for role in roles:
        target_positions = role_positions(target, role)
        source_positions = role_positions(source, role)
        if len(target_positions) != len(source_positions):
            raise RuntimeError("role span mismatch")
        value[target_row, target_positions] = original[source_row, source_positions]


def margin(logits, candidates):
    return float(logits[candidates[0][0]].float() - logits[candidates[1][0]].float())


def metrics(records: list[dict], coalition: str) -> dict:
    correct = [row["gains"][coalition]["correct_true"] for row in records]
    controls = [[row["gains"][coalition][name] for name in
                 ("wrong_family_true", "same_family_false", "status_true")] for row in records]
    advantages = [value - max(wrong) for value, wrong in zip(correct, controls)]
    wins = [value > max(wrong) for value, wrong in zip(correct, controls)]
    subgroup = {}
    for key in ("partition", "surface", "recipient_tested_family"):
        subgroup[key] = {}
        for name in sorted({row[key] for row in records}):
            indexes = [i for i, row in enumerate(records) if row[key] == name]
            subgroup[key][name] = {
                "count": len(indexes), "median_gain": statistics.median(correct[i] for i in indexes),
                "direction_fraction": sum(correct[i] > 0 for i in indexes) / len(indexes),
                "win_fraction": sum(wins[i] for i in indexes) / len(indexes),
            }
    return {
        "count": len(records), "correct_gain_median": statistics.median(correct),
        "correct_direction_fraction": sum(value > 0 for value in correct) / len(correct),
        "correct_over_controls_median": statistics.median(advantages),
        "correct_over_controls_win_fraction": sum(wins) / len(wins),
        "self_max_abs_diff": max(abs(row["gains"][coalition]["self"]) for row in records),
        "subgroups": subgroup,
    }


@torch.inference_mode()
def run() -> None:
    protocol, _observation = parents()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    entries = core.rows(CONTRACT / "material/causal_replay_manifest.jsonl")
    active = core.rows(C053 / "compiled/qwen3_B1_binary.jsonl")
    status = core.rows(C053 / "compiled/qwen3_N_status.jsonl")
    active_by_id = {row["case_id"]: row for row in active}
    status_by_id = {row["case_id"]: row for row in status}
    width = max(max(len(row["prompt_ids"]) for row in active), max(len(row["prompt_ids"]) for row in status))
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        layer = model.model.layers[manifest["layer"]]
        coalition_names = list(manifest["coalitions"])
        donor_start = 1 + len(coalition_names) * len(ARMS)
        records = []
        for entry in entries:
            recipient = active_by_id[entry["recipient"]]
            donors = {
                "correct_true": active_by_id[entry["correct_true"]],
                "wrong_true": active_by_id[entry["wrong_true"]],
                "correct_false": active_by_id[entry["correct_false"]],
                "status_true": status_by_id[entry["status_true"]],
            }
            batch_rows = [recipient]
            for _coalition in coalition_names:
                batch_rows.extend([recipient] * len(ARMS))
            batch_rows.extend(donors[key] for key in DONOR_KEYS)
            ids, mask, positions = make_batch(batch_rows, width, pad, device)
            donor_rows = {key: donor_start + index for index, key in enumerate(DONOR_KEYS)}

            def hook(_module, args):
                original = args[0]
                value = original.clone()
                for coalition_index, (name, roles) in enumerate(manifest["coalitions"].items()):
                    base = 1 + coalition_index * len(ARMS)
                    copy_roles(value, original, base, recipient, 0, recipient, roles)
                    copy_roles(value, original, base + 1, recipient, donor_rows["correct_true"], donors["correct_true"], roles)
                    copy_roles(value, original, base + 2, recipient, donor_rows["wrong_true"], donors["wrong_true"], roles)
                    copy_roles(value, original, base + 3, recipient, donor_rows["correct_false"], donors["correct_false"], roles)
                    copy_roles(value, original, base + 4, recipient, donor_rows["status_true"], donors["status_true"], roles)
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
            baseline = margin(output.logits[0, -1], recipient["candidate_ids"])
            margins, gains = {}, {}
            for coalition_index, name in enumerate(coalition_names):
                base = 1 + coalition_index * len(ARMS)
                margins[name] = {arm: margin(output.logits[base + arm_index, -1], recipient["candidate_ids"])
                                 for arm_index, arm in enumerate(ARMS)}
                gains[name] = {arm: value - baseline for arm, value in margins[name].items()}
            records.append({**entry, "baseline_margin": baseline, "margins": margins, "gains": gains})
            del output, ids, mask, positions

        core.write_rows(OUT / "raw/qwen3_coalition_causal.jsonl", records)
        route_metrics = {name: metrics(records, name) for name in manifest["coalitions"]}
        gate = manifest["gate"]
        route_checks, route_qualified = {}, {}
        for name, values in route_metrics.items():
            route_checks[name] = {
                "gain": values["correct_gain_median"] >= gate["false_to_true_gain_min"],
                "direction": values["correct_direction_fraction"] >= gate["direction_fraction_min"],
                "selective_median": values["correct_over_controls_median"] >= gate["correct_over_all_controls_median_min"],
                "selective_win": values["correct_over_controls_win_fraction"] >= gate["correct_over_all_controls_win_min"],
                "self": values["self_max_abs_diff"] <= gate["self_max_abs_diff_max"],
            }
            route_qualified[name] = all(route_checks[name].values())
        synergy, multi_qualified = {}, {}
        for name, constituents in CONSTITUENTS.items():
            gain_base = max(route_metrics[item]["correct_gain_median"] for item in constituents)
            win_base = max(route_metrics[item]["correct_over_controls_win_fraction"] for item in constituents)
            synergy[name] = {
                "gain_over_best_constituent": route_metrics[name]["correct_gain_median"] - gain_base,
                "win_over_best_constituent": route_metrics[name]["correct_over_controls_win_fraction"] - win_base,
            }
            synergy[name]["checks"] = {
                "gain": synergy[name]["gain_over_best_constituent"] >= gate["multi_synergy_gain_over_best_constituent_min"],
                "win": synergy[name]["win_over_best_constituent"] >= gate["multi_synergy_win_over_best_constituent_min"],
                "causal": route_qualified[name],
            }
            multi_qualified[name] = all(synergy[name]["checks"].values())
        passing = [name for name, value in multi_qualified.items() if value]
        selected = None
        if passing:
            selected = sorted(passing, key=lambda name: (len(manifest["coalitions"][name]),
                              -route_metrics[name]["correct_over_controls_win_fraction"], name))[0]
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL, "layer": manifest["layer"],
            "route_metrics": route_metrics, "route_checks": route_checks, "route_qualified": route_qualified,
            "multi_synergy": synergy, "multi_qualified": multi_qualified,
            "selected_multi_coalition": selected, "any_multi_qualified": selected is not None,
            "runtime": {"placement": placement, "quantization": quant,
                        "all_finite": all(math.isfinite(value) for row in records
                                          for coalition in row["gains"].values() for value in coalition.values()),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "same-batch exact-token coalition sufficiency/selectivity at one frozen layer",
        }
        core.save(OUT / "analysis/qwen3_coalition_causal.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_coalition_causal.json")
    final = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "qualified_routes": [name for name, value in summary["route_qualified"].items() if value],
        "selected_multi_coalition": summary["selected_multi_coalition"],
        "authorization": "run_phase1364_c055_necessity_rescue" if summary["any_multi_qualified"]
                         else "close_c055_at_hidden_state_coalition_boundary",
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
