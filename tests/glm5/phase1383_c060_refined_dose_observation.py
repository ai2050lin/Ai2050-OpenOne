#!/usr/bin/env python3
"""Phase1383: C060 refined whole-state dose response and discovery source."""
from __future__ import annotations

import argparse
import hashlib
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

PHASE, CAMPAIGN = 1383, "C060"
CONTRACT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
BEHAVIOR = TESTS / "result/phase1381_c060_qwen_behavior_qualification"
CAMERA = TESTS / "result/phase1382_c060_response_coalition_camera"
OUT = TESTS / "result/phase1383_c060_refined_dose_observation"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")


def parents() -> dict:
    final = core.load(CAMERA / "analysis/final.json")
    audit = core.load(CAMERA / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1383_c060_refined_dose_observation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1382 did not authorize natural response reveal")
    return core.load(CONTRACT / "protocol/preregistration.json")


def prepare() -> None:
    protocol = parents()
    camera_manifest = core.load(CAMERA / "protocol/execution_manifest.json")
    target = OUT / "protocol/execution_manifest.json"
    if target.exists():
        raise RuntimeError("Phase1383 manifest already exists")
    pairs = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    if len(pairs) != protocol["material"]["eligible_case_target"]:
        raise RuntimeError("eligible pair count changed")
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "camera_final_sha256": core.sha(CAMERA / "analysis/final.json"),
        "camera_audit_sha256": core.sha(CAMERA / "audit/independent_final_audit.json"),
        "model": MODEL,
        "precision": "bfloat16-no-quantization",
        "paths": protocol["paths"],
        "dose_gate": protocol["dose"],
        "mode_layouts": camera_manifest["mode_layouts"],
        "rows_per_mode_path_case": camera_manifest["rows_per_mode_path_case"],
        "case_count": len(pairs),
        "case_ids": [r["pair_id"] for r in pairs],
        "all_paths_arms_and_splits_run_once": True,
        "post_reveal_changes_forbidden": True,
        "allowed_observables": protocol["allowed_observables"],
        "forbidden": protocol["forbidden"],
        "random_rule": "SHA256(pair_id|path|mode|seed) deterministic Gaussian, exact per-example norm match",
        "discovery_source": "response_discovery natural family@3 clean-minus-corrupt only",
        "discovery_not_confirmation_evidence": True,
        "mediation_eligibility": protocol["mediation"]["eligibility"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(target, manifest)
    print(json.dumps(manifest, indent=2))


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


def points(row: dict, offset: int, role: str) -> list[int]:
    return [offset + p for p in row["role_positions"][role]]


def state(output, layer: int, row_index: int, row: dict, offset: int, role: str) -> torch.Tensor:
    return output.hidden_states[layer][row_index, points(row, offset, role)].float().flatten()


def margin(output, row_index: int, row: dict) -> float:
    logits = output.logits[row_index, -1].float()
    return float(logits[row["candidate_ids"][0][0]] - logits[row["candidate_ids"][1][0]])


def scaled(value: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    if float(norm) <= 1e-12:
        raise RuntimeError("zero control direction")
    return value * (target_norm / norm)


def seed_for(pair_id: str, path: str, mode: str, seed: int) -> int:
    raw = f"{pair_id}|{path}|{mode}|{seed}".encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:15], 16) % (2**63 - 1)


def geometry(value: torch.Tensor, origin: torch.Tensor, goal: torch.Tensor) -> dict:
    d = goal - origin
    u = value - origin
    d2 = torch.dot(d, d)
    alpha = torch.dot(u, d) / (d2 + 1e-12)
    orth = u - alpha * d
    dnorm = torch.sqrt(d2 + 1e-12)
    return {
        "alpha": float(alpha),
        "orthogonal_ratio": float(torch.linalg.vector_norm(orth) / dnorm),
        "goal_distance_ratio": float(torch.linalg.vector_norm(value - goal) / dnorm),
    }


def selection_metrics(rows: list[dict], mode: str) -> dict:
    by = {(r["pair_id"], r["direction"], r["dose"]): r for r in rows if r["mode"] == mode}
    pair_ids = sorted({r["pair_id"] for r in rows})
    correct, advantage, wins = [], [], []
    for pair_id in pair_ids:
        c = by[(pair_id, "correct", 1.0)]["output_effect"]
        controls = [by[(pair_id, d, 1.0)]["output_effect"] for d in ("wrong", "status", "random")]
        correct.append(c)
        advantage.append(c - max(controls))
        wins.append(c > max(controls))
    return {
        "count": len(pair_ids),
        "gain_median": statistics.median(correct),
        "advantage_median": statistics.median(advantage),
        "win_fraction": sum(wins) / len(wins),
    }


def reverse_metrics(rows: list[dict]) -> dict:
    by = {(r["pair_id"], r["direction"], r["dose"]): r for r in rows if r["mode"] == "reverse"}
    pair_ids = sorted({r["pair_id"] for r in rows})
    damage, advantage, wins = [], [], []
    for pair_id in pair_ids:
        c = by[(pair_id, "correct", 1.0)]["output_effect"]
        s = by[(pair_id, "status", 1.0)]["output_effect"]
        damage.append(c)
        advantage.append(c - s)
        wins.append(c > s)
    return {
        "count": len(pair_ids),
        "damage_median": statistics.median(damage),
        "over_status_median": statistics.median(advantage),
        "over_status_win_fraction": sum(wins) / len(wins),
    }


def endpoint_pass(metric: dict, gate: dict) -> bool:
    return (
        metric["gain_median"] >= gate["endpoint_gain_median_min"]
        and metric["advantage_median"] >= gate["endpoint_advantage_median_min"]
        and metric["win_fraction"] >= gate["endpoint_win_min"]
    )


def reverse_pass(metric: dict, gate: dict) -> bool:
    return (
        metric["damage_median"] >= gate["mid_reverse_damage_median_min"]
        and metric["over_status_median"] >= gate["mid_reverse_over_status_median_min"]
        and metric["over_status_win_fraction"] >= gate["mid_reverse_over_status_win_min"]
    )


def summarize(records: list[dict], manifest: dict) -> dict:
    gate = manifest["dose_gate"]
    doses = [float(v) for v in gate["values"]]
    partitions = ("pooled", "response_discovery", "confirmation", "lockbox")
    result = {}
    for path_name in manifest["paths"]:
        path_rows = [r for r in records if r["path"] == path_name]
        curve = {}
        for mode in ("sufficiency", "reverse"):
            curve[mode] = {}
            for direction in gate["directions"]:
                curve[mode][direction] = {
                    str(dose): statistics.median(
                        r["output_effect"]
                        for r in path_rows
                        if r["mode"] == mode and r["direction"] == direction and r["dose"] == dose
                    )
                    for dose in doses
                }
        endpoint, reverse = {}, {}
        for partition in partitions:
            subset = path_rows if partition == "pooled" else [r for r in path_rows if r["partition"] == partition]
            endpoint[partition] = selection_metrics(subset, "sufficiency")
            endpoint[partition]["passed"] = endpoint_pass(endpoint[partition], gate)
            reverse[partition] = reverse_metrics(subset)
            reverse[partition]["passed"] = reverse_pass(reverse[partition], gate)
        correct_curve = curve["sufficiency"]["correct"]
        correct_values = [correct_curve[str(d)] for d in doses]
        jumps = [correct_values[i + 1] - correct_values[i] for i in range(len(doses) - 1)]
        threshold = {
            "low_dose_abs": abs(correct_curve[str(0.5)]),
            "max_adjacent_jump": max(jumps),
            "high_dose_gain": correct_curve[str(1.0)],
            "plateau_abs_difference": abs(correct_curve[str(1.0)] - correct_curve[str(0.875)]),
        }
        threshold["passed"] = (
            threshold["low_dose_abs"] <= gate["threshold_low_dose_abs_median_max"]
            and threshold["max_adjacent_jump"] >= gate["threshold_adjacent_jump_min"]
            and threshold["high_dose_gain"] >= gate["threshold_high_dose_gain_min"]
            and threshold["plateau_abs_difference"] <= gate["threshold_plateau_abs_difference_max"]
        )
        result[path_name] = {
            "dose_curve_medians": curve,
            "sufficiency_endpoint": endpoint,
            "reverse_endpoint": reverse,
            "threshold_candidate": threshold,
            "lambda0_output_max_abs_diff": max(abs(r["output_effect"]) for r in path_rows if r["dose"] == 0.0),
            "norm_ratio_abs_error_max": max(r["norm_ratio_abs_error"] for r in path_rows),
            "finite_fraction": sum(r["all_finite"] for r in path_rows) / len(path_rows),
        }
    return result


@torch.inference_mode()
def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/qwen3_refined_dose_summary.json").exists():
        raise RuntimeError("Phase1383 run already exists")
    cases = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    compiled = {r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        discovery_vectors, discovery_meta = [], []
        for case_index, case in enumerate(cases):
            donors = {key: compiled[case[key]] for key in DONOR_KEYS}
            donor_rows = [donors[key] for key in DONOR_KEYS]
            for path_name, path in manifest["paths"].items():
                for mode, layout in manifest["mode_layouts"].items():
                    recipient = 1 if mode == "sufficiency" else 0
                    goal = 0 if mode == "sufficiency" else 1
                    rows = donor_rows + [donor_rows[recipient] for _ in layout]
                    ids, mask, positions, offsets = make_batch(rows, pad, device)
                    norm_errors = [0.0] * len(layout)
                    source_difference = [None]

                    def hook(_module, args):
                        original = args[0]
                        value = original.clone()
                        role = path["source"]["role"]
                        donor_values = [original[i, points(rows[i], offsets[i], role)].float() for i in range(4)]
                        correct = donor_values[goal] - donor_values[recipient]
                        source_difference[0] = (donor_values[0] - donor_values[1]).detach().flatten().cpu()
                        target_norm = torch.linalg.vector_norm(correct)
                        generator = torch.Generator(device=original.device)
                        generator.manual_seed(seed_for(
                            case["pair_id"], path_name, mode, int(manifest["dose_gate"]["random_seed"])
                        ))
                        random_value = torch.randn(
                            correct.shape, generator=generator, device=original.device, dtype=torch.float32
                        )
                        directions = {
                            "correct": correct,
                            "wrong": scaled(donor_values[2] - donor_values[recipient], target_norm),
                            "status": scaled(donor_values[3] - donor_values[recipient], target_norm),
                            "random": scaled(random_value, target_norm),
                        }
                        for local, spec in enumerate(layout):
                            target_index = 4 + local
                            tp = points(rows[target_index], offsets[target_index], role)
                            direction = directions[spec["direction"]]
                            norm_errors[local] = abs(float(
                                torch.linalg.vector_norm(direction) / (target_norm + 1e-12)
                            ) - 1.0)
                            value[target_index, tp] = (
                                original[target_index, tp]
                                + float(spec["dose"]) * direction.to(original.dtype)
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
                    if (
                        case["partition"] == "response_discovery"
                        and path_name == "family_early"
                        and mode == "sufficiency"
                    ):
                        discovery_vectors.append(source_difference[0].to(torch.bfloat16))
                        discovery_meta.append({
                            "pair_id": case["pair_id"],
                            "target_family": case["target_family"],
                            "surface": case["surface"],
                            "partition": case["partition"],
                        })
                    natural_margins = [margin(output, i, rows[i]) for i in range(4)]
                    state_cache = {}
                    for checkpoint in path["checkpoints"]:
                        key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
                        state_cache[key] = [
                            state(output, checkpoint["layer"], i, rows[i], offsets[i], checkpoint["role"])
                            for i in range(4)
                        ]
                    for local, spec in enumerate(layout):
                        target_index = 4 + local
                        target_margin = margin(output, target_index, rows[target_index])
                        effect = (
                            target_margin - natural_margins[1]
                            if mode == "sufficiency"
                            else natural_margins[0] - target_margin
                        )
                        geoms = {}
                        for checkpoint in path["checkpoints"]:
                            key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
                            value = state(
                                output, checkpoint["layer"], target_index, rows[target_index],
                                offsets[target_index], checkpoint["role"],
                            )
                            geoms[key] = geometry(value, state_cache[key][recipient], state_cache[key][goal])
                        values = [effect, norm_errors[local]] + [x for g in geoms.values() for x in g.values()]
                        records.append({
                            "pair_id": case["pair_id"],
                            "partition": case["partition"],
                            "surface": case["surface"],
                            "target_family": case["target_family"],
                            "wrong_family": case["wrong_family"],
                            "path": path_name,
                            **spec,
                            "clean_margin": natural_margins[0],
                            "corrupt_margin": natural_margins[1],
                            "output_effect": effect,
                            "norm_ratio_abs_error": norm_errors[local],
                            "checkpoint_geometry": geoms,
                            "all_finite": all(math.isfinite(v) for v in values),
                        })
                    del output, ids, mask, positions
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"dose_cases": case_index + 1, "total": len(cases)}), flush=True)
        core.write_rows(OUT / "raw/qwen3_refined_dose_response.jsonl", records)
        discovery = {
            "vectors": torch.stack(discovery_vectors),
            "metadata": discovery_meta,
            "source": "response_discovery natural family@3 clean-minus-corrupt only",
        }
        discovery_path = OUT / "raw/response_discovery_family3_differences.pt"
        discovery_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(discovery, discovery_path)
        path_summary = summarize(records, manifest)
        early_endpoint = path_summary["family_early"]["sufficiency_endpoint"]
        mediation_eligible = all(early_endpoint[p]["passed"] for p in ("pooled", "confirmation", "lockbox"))
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "model": MODEL,
            "record_count": len(records),
            "discovery_vector_count": len(discovery_vectors),
            "discovery_vector_shape": list(discovery["vectors"].shape),
            "discovery_sha256": core.sha(discovery_path),
            "path_summary": path_summary,
            "mediation_eligible": mediation_eligible,
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "claim_boundary": "Qwen-specific independent-material whole-state dose response; no coordinate or mediation claim",
        }
        core.save(OUT / "analysis/qwen3_refined_dose_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_refined_dose_summary.json")
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "mediation_eligible": summary["mediation_eligible"],
        "authorization": "run_phase1384_c060_fixed_dynamic_coalitions",
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
