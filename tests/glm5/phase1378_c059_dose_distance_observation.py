#!/usr/bin/env python3
"""Phase1378: C059 whole-state dose response, full geometry, and observation field."""
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

PHASE, CAMPAIGN = 1378, "C059"
CONTRACT = TESTS / "result/phase1375_c059_independent_relaunch_contract"
BEHAVIOR = TESTS / "result/phase1376_c059_qwen_behavior_qualification"
CAMERA = TESTS / "result/phase1377_c059_response_field_camera"
OUT = TESTS / "result/phase1378_c059_dose_distance_observation"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
ROLE_NAMES = ("target", "family", "query", "boundary")


def parents() -> dict:
    final = core.load(CAMERA / "analysis/final.json")
    audit = core.load(CAMERA / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1378_c059_dose_distance_observation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1377 did not authorize natural response reveal")
    return core.load(CONTRACT / "protocol/preregistration.json")


def target_layout(protocol: dict) -> list[dict]:
    return [{"mode": mode, "direction": direction, "dose": float(dose)}
            for mode in ("sufficiency", "reverse")
            for direction in protocol["dose"]["directions"]
            for dose in protocol["dose"]["values"]]


def prepare() -> None:
    protocol = parents()
    target = OUT / "protocol/execution_manifest.json"
    if target.exists():
        raise RuntimeError("Phase1378 manifest already exists")
    pairs = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    if len(pairs) != protocol["material"]["eligible_case_target"]:
        raise RuntimeError("eligible pair count changed")
    layout = target_layout(protocol)
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "camera_final_sha256": core.sha(CAMERA / "analysis/final.json"),
        "camera_audit_sha256": core.sha(CAMERA / "audit/independent_final_audit.json"),
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "paths": protocol["paths"], "dose_gate": protocol["dose"],
        "distance_gate": protocol["distance"], "observation": protocol["observation"],
        "layout": layout, "rows_per_path_case": 4 + len(layout),
        "case_count": len(pairs), "case_ids": [r["pair_id"] for r in pairs],
        "all_paths_and_arms_run_once": True, "post_reveal_changes_forbidden": True,
        "allowed_observables": protocol["allowed_observables"], "forbidden": protocol["forbidden"],
        "random_rule": "SHA256(pair_id|path|mode|seed) deterministic Gaussian, exact per-example norm match",
        "discovery_field_not_confirmation": True,
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
    lhs = torch.dot(u, u)
    rhs = alpha * alpha * d2 + torch.dot(orth, orth)
    return {
        "alpha": float(alpha),
        "orthogonal_ratio": float(torch.linalg.vector_norm(orth) / dnorm),
        "goal_distance_ratio": float(torch.linalg.vector_norm(value - goal) / dnorm),
        "decomposition_relative_error": float(torch.abs(lhs - rhs) / (lhs + 1e-12)),
    }


def save_discovery_field(output, case: dict, rows: list[dict], offsets: list[int], path_name: str) -> dict:
    field_dir = OUT / "raw/response_discovery_field"
    field_dir.mkdir(parents=True, exist_ok=True)
    payload = {"pair_id": case["pair_id"], "path_source": path_name, "roles": {}}
    for role in ROLE_NAMES:
        per_donor = []
        for donor_index in range(4):
            pos = points(rows[donor_index], offsets[donor_index], role)
            per_donor.append(torch.stack([h[donor_index, pos].detach().to("cpu", dtype=torch.bfloat16)
                                          for h in output.hidden_states]))
        # Role spans are semantically aligned but may contain different token
        # counts across donors. Preserve every token as a typed ragged list.
        payload["roles"][role] = per_donor
    name = hashlib.sha256(case["pair_id"].encode("utf-8")).hexdigest()[:20] + ".pt"
    target = field_dir / name
    torch.save(payload, target)
    return {"pair_id": case["pair_id"], "file": target.relative_to(OUT).as_posix(),
            "sha256": core.sha(target), "path_source": path_name,
            "not_confirmation_evidence": True}


def summarize(records: list[dict], manifest: dict) -> tuple[dict, dict, dict]:
    gate = manifest["dose_gate"]
    metrics, checks, eligibility = {}, {}, {}
    doses = [float(v) for v in gate["values"]]
    for path_name in manifest["paths"]:
        path_rows = [r for r in records if r["path"] == path_name]
        by_key = {(r["pair_id"], r["mode"], r["direction"], r["dose"]): r for r in path_rows}
        pair_ids = sorted({r["pair_id"] for r in path_rows})
        dose_curve = {}
        for mode in ("sufficiency", "reverse"):
            dose_curve[mode] = {}
            for direction in gate["directions"]:
                dose_curve[mode][direction] = {}
                for dose in doses:
                    values = [by_key[(pid, mode, direction, dose)]["output_effect"] for pid in pair_ids]
                    dose_curve[mode][direction][str(dose)] = {
                        "median": statistics.median(values),
                        "positive_fraction": sum(v > 0 for v in values) / len(values),
                    }
        suff_correct, suff_adv, suff_win = [], [], []
        reverse_correct, reverse_adv, reverse_win = [], [], []
        for pid in pair_ids:
            c = by_key[(pid, "sufficiency", "correct", 1.0)]["output_effect"]
            controls = [by_key[(pid, "sufficiency", d, 1.0)]["output_effect"]
                        for d in ("wrong", "status", "random")]
            suff_correct.append(c)
            suff_adv.append(c - max(controls))
            suff_win.append(c > max(controls))
            rc = by_key[(pid, "reverse", "correct", 1.0)]["output_effect"]
            rs = by_key[(pid, "reverse", "status", 1.0)]["output_effect"]
            reverse_correct.append(rc)
            reverse_adv.append(rc - rs)
            reverse_win.append(rc > rs)
        correct_medians = [dose_curve["sufficiency"]["correct"][str(d)]["median"] for d in doses]
        metric = {
            "count": len(pair_ids), "dose_curve": dose_curve,
            "lambda1_suff_gain_median": statistics.median(suff_correct),
            "lambda1_suff_advantage_median": statistics.median(suff_adv),
            "lambda1_suff_win_fraction": sum(suff_win) / len(suff_win),
            "population_median_monotone": all(a <= b + 1e-12 for a, b in zip(correct_medians, correct_medians[1:])),
            "reverse_damage_median": statistics.median(reverse_correct),
            "reverse_over_status_median": statistics.median(reverse_adv),
            "reverse_over_status_win_fraction": sum(reverse_win) / len(reverse_win),
            "lambda0_output_max_abs_diff": max(abs(r["output_effect"]) for r in path_rows if r["dose"] == 0.0),
            "norm_ratio_abs_error_max": max(r["norm_ratio_abs_error"] for r in path_rows),
            "decomposition_relative_error_max": max(r["decomposition_relative_error_max"] for r in path_rows),
            "finite_fraction": sum(r["all_finite"] for r in path_rows) / len(path_rows),
        }
        suff_checks = {
            "gain": metric["lambda1_suff_gain_median"] >= gate["lambda1_suff_gain_median_min"],
            "advantage": metric["lambda1_suff_advantage_median"] >= gate["lambda1_suff_advantage_median_min"],
            "win": metric["lambda1_suff_win_fraction"] >= gate["lambda1_suff_win_min"],
            "monotone": metric["population_median_monotone"],
            "self": metric["lambda0_output_max_abs_diff"] <= gate["self_output_max_abs_diff"],
            "norm": metric["norm_ratio_abs_error_max"] <= gate["norm_ratio_abs_error_max"],
            "distance": metric["decomposition_relative_error_max"] <=
                        manifest["distance_gate"]["direction_decomposition_relative_error_max"],
            "finite": metric["finite_fraction"] >= manifest["distance_gate"]["finite_fraction_min"],
        }
        reverse_checks = {
            "damage": metric["reverse_damage_median"] >= gate["reverse_damage_median_min"],
            "over_status": metric["reverse_over_status_median"] >= gate["reverse_over_status_median_min"],
            "win": metric["reverse_over_status_win_fraction"] >= gate["reverse_over_status_win_min"],
            "self": suff_checks["self"], "norm": suff_checks["norm"],
            "distance": suff_checks["distance"], "finite": suff_checks["finite"],
        }
        metrics[path_name] = metric
        checks[path_name] = {"sufficiency": suff_checks, "reverse": reverse_checks}
        eligibility[path_name] = {"sufficiency": all(suff_checks.values()),
                                  "reverse": all(reverse_checks.values())}
    return metrics, checks, eligibility


@torch.inference_mode()
def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/qwen3_dose_distance_summary.json").exists():
        raise RuntimeError("Phase1378 run already exists")
    cases = core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
    compiled = {r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records, distance_records, field_index = [], [], []
        for case_index, case in enumerate(cases):
            donors = {key: compiled[case[key]] for key in DONOR_KEYS}
            donor_rows = [donors[key] for key in DONOR_KEYS]
            for path_index, (path_name, path) in enumerate(manifest["paths"].items()):
                recipient_indices = [1 if spec["mode"] == "sufficiency" else 0 for spec in manifest["layout"]]
                rows = donor_rows + [donor_rows[i] for i in recipient_indices]
                ids, mask, positions, offsets = make_batch(rows, pad, device)
                norm_errors = [0.0] * len(manifest["layout"])

                def hook(_module, args):
                    original = args[0]
                    value = original.clone()
                    role = path["source"]["role"]
                    donor_values = [original[i, points(rows[i], offsets[i], role)].float() for i in range(4)]
                    direction_sets = {}
                    for mode, origin_i, goal_i in (("sufficiency", 1, 0), ("reverse", 0, 1)):
                        correct = donor_values[goal_i] - donor_values[origin_i]
                        target_norm = torch.linalg.vector_norm(correct)
                        generator = torch.Generator(device=original.device)
                        generator.manual_seed(seed_for(case["pair_id"], path_name, mode,
                                                       int(manifest["dose_gate"]["random_seed"])))
                        random_value = torch.randn(correct.shape, generator=generator,
                                                   device=original.device, dtype=torch.float32)
                        direction_sets[mode] = {
                            "correct": correct,
                            "wrong": scaled(donor_values[2] - donor_values[origin_i], target_norm),
                            "status": scaled(donor_values[3] - donor_values[origin_i], target_norm),
                            "random": scaled(random_value, target_norm),
                        }
                    for local, spec in enumerate(manifest["layout"]):
                        target_index = 4 + local
                        tp = points(rows[target_index], offsets[target_index], role)
                        direction = direction_sets[spec["mode"]][spec["direction"]]
                        correct_norm = torch.linalg.vector_norm(direction_sets[spec["mode"]]["correct"])
                        norm_errors[local] = abs(float(torch.linalg.vector_norm(direction) / (correct_norm + 1e-12)) - 1.0)
                        value[target_index, tp] = original[target_index, tp] + float(spec["dose"]) * direction.to(original.dtype)
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
                if case["partition"] == manifest["observation"]["partition"] and path_index == 0:
                    field_index.append(save_discovery_field(output, case, rows, offsets, path_name))
                natural_margins = [margin(output, i, rows[i]) for i in range(4)]
                natural_logits = [output.logits[i, -1].float() for i in range(4)]
                state_cache = {}
                for checkpoint in path["checkpoints"]:
                    key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
                    state_cache[key] = [state(output, checkpoint["layer"], i, rows[i], offsets[i], checkpoint["role"])
                                        for i in range(4)]
                target_states = {}
                for local, spec in enumerate(manifest["layout"]):
                    target_index = 4 + local
                    recipient = recipient_indices[local]
                    goal = 0 if spec["mode"] == "sufficiency" else 1
                    target_margin = margin(output, target_index, rows[target_index])
                    effect = ((target_margin - natural_margins[1]) if spec["mode"] == "sufficiency"
                              else (natural_margins[0] - target_margin))
                    full_l2 = float(torch.linalg.vector_norm(output.logits[target_index, -1].float() -
                                                            natural_logits[recipient]) /
                                    (torch.linalg.vector_norm(natural_logits[goal] - natural_logits[recipient]) + 1e-12))
                    geoms = {}
                    for checkpoint in path["checkpoints"]:
                        key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
                        value = state(output, checkpoint["layer"], target_index, rows[target_index],
                                      offsets[target_index], checkpoint["role"])
                        target_states[(spec["mode"], spec["direction"], spec["dose"], key)] = value
                        geoms[key] = geometry(value, state_cache[key][recipient], state_cache[key][goal])
                    record = {
                        "pair_id": case["pair_id"], "partition": case["partition"],
                        "surface": case["surface"], "target_family": case["target_family"],
                        "wrong_family": case["wrong_family"], "path": path_name, **spec,
                        "clean_margin": natural_margins[0], "corrupt_margin": natural_margins[1],
                        "output_effect": effect, "full_vocab_response_ratio": full_l2,
                        "norm_ratio_abs_error": norm_errors[local], "geometry": geoms,
                        "decomposition_relative_error_max": max(v["decomposition_relative_error"] for v in geoms.values()),
                        "all_finite": all(math.isfinite(v) for v in [effect, full_l2, norm_errors[local]] +
                                          [x for g in geoms.values() for x in g.values()]),
                    }
                    records.append(record)
                for mode in ("sufficiency", "reverse"):
                    for dose in manifest["dose_gate"]["values"]:
                        for checkpoint in path["checkpoints"]:
                            key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
                            vectors = {direction: target_states[(mode, direction, float(dose), key)]
                                       for direction in manifest["dose_gate"]["directions"]}
                            denom = torch.linalg.vector_norm(state_cache[key][0] - state_cache[key][1]) + 1e-12
                            matrix = {}
                            directions = list(manifest["dose_gate"]["directions"])
                            for left in directions:
                                matrix[left] = {right: float(torch.linalg.vector_norm(vectors[left] - vectors[right]) / denom)
                                                for right in directions}
                            distance_records.append({"pair_id": case["pair_id"], "partition": case["partition"],
                                                     "path": path_name, "mode": mode, "dose": float(dose),
                                                     "checkpoint": key, "normalized_pairwise_l2": matrix})
                del output, ids, mask, positions
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"dose_cases": case_index + 1, "total": len(cases)}), flush=True)
        core.write_rows(OUT / "raw/qwen3_dose_response.jsonl", records)
        core.write_rows(OUT / "raw/qwen3_pairwise_distance.jsonl", distance_records)
        core.write_rows(OUT / "raw/response_discovery_field_index.jsonl", field_index)
        metrics, checks, eligibility = summarize(records, manifest)
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
            "record_count": len(records), "distance_record_count": len(distance_records),
            "discovery_field_count": len(field_index), "path_metrics": metrics,
            "path_checks": checks, "path_eligibility": eligibility,
            "mediation_eligible": bool(eligibility.get("family_early", {}).get("sufficiency")),
            "runtime": {"placement": placement, "quantization": quant,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
            "claim_boundary": "Qwen-specific full-state dose and full-dimensional response geometry; discovery field is descriptive only",
        }
        core.save(OUT / "analysis/qwen3_dose_distance_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_dose_distance_summary.json")
    final = {"phase": PHASE, "campaign": CAMPAIGN,
             "path_eligibility": summary["path_eligibility"],
             "mediation_eligible": summary["mediation_eligible"],
             "authorization": "run_phase1379_c059_coordinate_group_evaluation",
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def analyze_geometry() -> None:
    """Summarize frozen full-dimensional records without changing any gate."""
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    records = core.rows(OUT / "raw/qwen3_dose_response.jsonl")
    distances = core.rows(OUT / "raw/qwen3_pairwise_distance.jsonl")
    grouped = {}
    for path_name, path in manifest["paths"].items():
        grouped[path_name] = {}
        for mode in ("sufficiency", "reverse"):
            grouped[path_name][mode] = {}
            for direction in manifest["dose_gate"]["directions"]:
                grouped[path_name][mode][direction] = {}
                for dose in manifest["dose_gate"]["values"]:
                    subset = [r for r in records if r["path"] == path_name and r["mode"] == mode and
                              r["direction"] == direction and r["dose"] == float(dose)]
                    checkpoints = {}
                    for checkpoint in path["checkpoints"]:
                        key = f'{checkpoint["role"]}@{checkpoint["layer"]}'
                        checkpoints[key] = {
                            metric: statistics.median(r["geometry"][key][metric] for r in subset)
                            for metric in ("alpha", "orthogonal_ratio", "goal_distance_ratio")
                        }
                    grouped[path_name][mode][direction][str(float(dose))] = checkpoints
    pairwise = {}
    for path_name in manifest["paths"]:
        pairwise[path_name] = {}
        for mode in ("sufficiency", "reverse"):
            pairwise[path_name][mode] = {}
            for dose in manifest["dose_gate"]["values"]:
                pairwise[path_name][mode][str(float(dose))] = {}
                subset = [r for r in distances if r["path"] == path_name and r["mode"] == mode and
                          r["dose"] == float(dose)]
                for checkpoint in sorted({r["checkpoint"] for r in subset}):
                    cp_rows = [r for r in subset if r["checkpoint"] == checkpoint]
                    directions = list(manifest["dose_gate"]["directions"])
                    pairwise[path_name][mode][str(float(dose))][checkpoint] = {
                        left: {right: statistics.median(r["normalized_pairwise_l2"][left][right]
                                                        for r in cp_rows)
                               for right in directions}
                        for left in directions
                    }
    summary = {"phase": PHASE, "campaign": CAMPAIGN, "postprocessing_only": True,
               "thresholds_or_eligibility_changed": False, "geometry_medians": grouped,
               "pairwise_distance_medians": pairwise,
               "record_count": len(records), "distance_record_count": len(distances),
               "created_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/full_geometry_summary.json", summary)
    print(json.dumps({"phase": PHASE, "geometry_records": len(records),
                      "distance_records": len(distances), "postprocessing_only": True}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize", "analyze_geometry"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "finalize": finalize,
     "analyze_geometry": analyze_geometry}[args.command]()


if __name__ == "__main__":
    main()
