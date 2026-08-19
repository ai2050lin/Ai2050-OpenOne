#!/usr/bin/env python3
"""Phase1379: independent C059 raw-coordinate group evaluation."""
from __future__ import annotations

import argparse
import hashlib
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

PHASE, CAMPAIGN = 1379, "C059"
CONTRACT = TESTS / "result/phase1375_c059_independent_relaunch_contract"
BEHAVIOR = TESTS / "result/phase1376_c059_qwen_behavior_qualification"
DOSE = TESTS / "result/phase1378_c059_dose_distance_observation"
CANDIDATE_SOURCE = TESTS / "result/phase1372_c057_whole_state_bidirectional/raw/family3_source_deltas.pt"
OUT = TESTS / "result/phase1379_c059_coordinate_group_evaluation"
MODEL, CHUNK = "qwen3", 4
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
SPECS = (
    ("sufficiency", "self"), ("sufficiency", "correct"), ("sufficiency", "wrong"),
    ("sufficiency", "status"), ("sufficiency", "random"),
    ("reverse", "self"), ("reverse", "correct"), ("reverse", "status"), ("reverse", "random"),
)


def parents() -> dict:
    final = core.load(DOSE / "analysis/final.json")
    audit = core.load(DOSE / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1379_c059_coordinate_group_evaluation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1378 did not authorize coordinate groups")
    return core.load(CONTRACT / "protocol/preregistration.json")


def build_rankings(protocol: dict) -> dict[str, list[int]]:
    if core.sha(CANDIDATE_SOURCE) != protocol["coordinate_groups"]["candidate_source_sha256"]:
        raise RuntimeError("candidate source hash changed")
    payload = torch.load(CANDIDATE_SOURCE, map_location="cpu", weights_only=False)
    indices = [i for i, meta in enumerate(payload["metadata"]) if meta["partition"] == "prototype_discovery"]
    values = payload["family3_clean_minus_corrupt"][indices].float()
    metadata = [payload["metadata"][i] for i in indices]
    if values.shape != (72, 2560):
        raise RuntimeError("unexpected discovery tensor shape")
    magnitude = values.abs().mean(0)
    stable = values.mean(0).abs()
    families = sorted({m["target_family"] for m in metadata})
    family_score = torch.stack([values[[m["target_family"] == family for m in metadata]].abs().mean(0)
                                for family in families]).amin(0)
    def order(score: torch.Tensor) -> list[int]:
        return sorted(range(score.numel()), key=lambda i: (-float(score[i]), i))
    rng = random.Random(int(protocol["coordinate_groups"]["random_seed"]))
    random_order = list(range(values.shape[1]))
    rng.shuffle(random_order)
    return {"magnitude": order(magnitude), "stable_sign": order(stable),
            "family_min": order(family_score), "deterministic_random": random_order}


def prepare() -> None:
    protocol = parents()
    target = OUT / "protocol/execution_manifest.json"
    if target.exists():
        raise RuntimeError("Phase1379 manifest already exists")
    rankings = build_rankings(protocol)
    groups = {route: {str(size): ranking[:int(size)] for size in protocol["coordinate_groups"]["sizes"]}
              for route, ranking in rankings.items()}
    group_artifact = {"phase": PHASE, "campaign": CAMPAIGN,
                      "candidate_source_sha256": core.sha(CANDIDATE_SOURCE), "groups": groups}
    core.save(OUT / "protocol/candidate_groups.json", group_artifact)
    cases = [r for r in core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
             if r["partition"] in protocol["coordinate_groups"]["evaluation_partitions"]]
    if len(cases) != 144:
        raise RuntimeError("evaluation case count mismatch")
    dose_rows = core.rows(DOSE / "raw/qwen3_dose_response.jsonl")
    whole = {r["pair_id"]: r["output_effect"] for r in dose_rows
             if r["path"] == "family_early" and r["mode"] == "sufficiency" and
             r["direction"] == "correct" and r["dose"] == 1.0 and r["partition"] in
             protocol["coordinate_groups"]["evaluation_partitions"]}
    if set(whole) != {r["pair_id"] for r in cases}:
        raise RuntimeError("whole-effect reference mismatch")
    core.save(OUT / "protocol/whole_effect_reference.json", whole)
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": protocol["contract_sha256"],
        "dose_final_sha256": core.sha(DOSE / "analysis/final.json"),
        "dose_audit_sha256": core.sha(DOSE / "audit/independent_final_audit.json"),
        "candidate_source_sha256": core.sha(CANDIDATE_SOURCE),
        "candidate_groups_sha256": core.sha(OUT / "protocol/candidate_groups.json"),
        "model": MODEL, "precision": "bfloat16-no-quantization",
        "source": {"layer": 3, "role": "family"},
        "routes": protocol["coordinate_groups"]["routes"],
        "sizes": protocol["coordinate_groups"]["sizes"], "specs": [list(v) for v in SPECS],
        "chunk_size": CHUNK, "rows_per_chunk": 4 + CHUNK * len(SPECS),
        "gate": protocol["coordinate_groups"], "case_count": len(cases),
        "case_ids": [r["pair_id"] for r in cases],
        "post_reveal_changes_forbidden": True,
        "allowed_observables": protocol["allowed_observables"], "forbidden": protocol["forbidden"],
        "mediation_was_eligible": core.load(DOSE / "analysis/final.json")["mediation_eligible"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(target, manifest)
    core.write_rows(OUT / "material/evaluation_pairs.jsonl", cases)
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


def margin(output, row_index: int, row: dict) -> float:
    logits = output.logits[row_index, -1].float()
    return float(logits[row["candidate_ids"][0][0]] - logits[row["candidate_ids"][1][0]])


def scaled(value: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value)
    if float(norm) <= 1e-12:
        raise RuntimeError("zero coordinate-group control")
    return value * (target_norm / norm)


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:15], 16) % (2**63 - 1)


def summarize(records: list[dict], manifest: dict) -> tuple[dict, dict]:
    gate = manifest["gate"]
    summary, qualifications = {}, {}
    for route in manifest["routes"]:
        summary[route], qualifications[route] = {}, {"sufficiency": [], "reverse": []}
        for size in manifest["sizes"]:
            rows = [r for r in records if r["route"] == route and r["size"] == size]
            suff_c = [r["suff_gain"]["correct"] for r in rows]
            suff_adv = [r["suff_gain"]["correct"] - max(r["suff_gain"][d]
                                                        for d in ("wrong", "status", "random")) for r in rows]
            suff_win = [r["suff_gain"]["correct"] > max(r["suff_gain"][d]
                                                        for d in ("wrong", "status", "random")) for r in rows]
            fractions = [r["suff_gain"]["correct"] / r["whole_effect"]
                         for r in rows if abs(r["whole_effect"]) > 1e-12]
            rev_c = [r["reverse_damage"]["correct"] for r in rows]
            rev_adv = [r["reverse_damage"]["correct"] - r["reverse_damage"]["status"] for r in rows]
            rev_win = [r["reverse_damage"]["correct"] > r["reverse_damage"]["status"] for r in rows]
            metric = {
                "count": len(rows), "suff_gain_median": statistics.median(suff_c),
                "suff_advantage_median": statistics.median(suff_adv),
                "suff_win_fraction": sum(suff_win) / len(suff_win),
                "whole_effect_fraction_median": statistics.median(fractions),
                "reverse_damage_median": statistics.median(rev_c),
                "reverse_over_status_median": statistics.median(rev_adv),
                "reverse_over_status_win_fraction": sum(rev_win) / len(rev_win),
                "self_max_abs_diff": max(r["self_max_abs_diff"] for r in rows),
                "norm_ratio_abs_error_max": max(r["norm_ratio_abs_error_max"] for r in rows),
            }
            suff_ok = (metric["suff_gain_median"] >= gate["suff_gain_median_min"] and
                       metric["suff_advantage_median"] >= gate["suff_advantage_median_min"] and
                       metric["suff_win_fraction"] >= gate["suff_win_min"] and
                       metric["whole_effect_fraction_median"] >= gate["whole_effect_fraction_median_min"] and
                       metric["self_max_abs_diff"] <= gate["self_max_abs_diff"])
            reverse_ok = (metric["reverse_damage_median"] >= gate["reverse_damage_median_min"] and
                          metric["reverse_over_status_median"] >= gate["reverse_over_status_median_min"] and
                          metric["reverse_over_status_win_fraction"] >= gate["reverse_over_status_win_min"] and
                          metric["self_max_abs_diff"] <= gate["self_max_abs_diff"])
            metric["sufficiency_qualified"] = suff_ok
            metric["reverse_qualified"] = reverse_ok
            summary[route][str(size)] = metric
            if suff_ok:
                qualifications[route]["sufficiency"].append(size)
            if reverse_ok:
                qualifications[route]["reverse"].append(size)
        qualifications[route]["minimal_sufficiency_size"] = min(qualifications[route]["sufficiency"], default=None)
        qualifications[route]["minimal_reverse_size"] = min(qualifications[route]["reverse"], default=None)
    return summary, qualifications


@torch.inference_mode()
def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/qwen3_coordinate_group_summary.json").exists():
        raise RuntimeError("Phase1379 run already exists")
    groups = core.load(OUT / "protocol/candidate_groups.json")["groups"]
    whole = core.load(OUT / "protocol/whole_effect_reference.json")
    cases = core.rows(OUT / "material/evaluation_pairs.jsonl")
    compiled = {r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    group_specs = [(route, int(size), groups[route][str(size)])
                   for route in manifest["routes"] for size in manifest["sizes"]]
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
            per_case = {}
            for chunk_start in range(0, len(group_specs), manifest["chunk_size"]):
                chunk = group_specs[chunk_start:chunk_start + manifest["chunk_size"]]
                rows = list(donor_rows)
                recipient_indices = []
                for _route, _size, _coords in chunk:
                    for mode, _arm in SPECS:
                        recipient_indices.append(1 if mode == "sufficiency" else 0)
                        rows.append(donor_rows[recipient_indices[-1]])
                ids, mask, positions, offsets = make_batch(rows, pad, device)
                norm_errors = [0.0] * (len(chunk) * len(SPECS))

                def hook(_module, args):
                    original = args[0]
                    value = original.clone()
                    role = manifest["source"]["role"]
                    donor_values = [original[i, points(rows[i], offsets[i], role)].float() for i in range(4)]
                    for group_index, (route, size, coords_list) in enumerate(chunk):
                        coords = torch.tensor(coords_list, dtype=torch.long, device=original.device)
                        for local_spec, (mode, arm) in enumerate(SPECS):
                            local = group_index * len(SPECS) + local_spec
                            target_index = 4 + local
                            origin_i, goal_i = ((1, 0) if mode == "sufficiency" else (0, 1))
                            correct_full = donor_values[goal_i] - donor_values[origin_i]
                            correct = torch.zeros_like(correct_full)
                            correct[..., coords] = correct_full[..., coords]
                            target_norm = torch.linalg.vector_norm(correct)
                            if arm == "self":
                                direction = torch.zeros_like(correct)
                            elif arm == "correct":
                                direction = correct
                            elif arm in ("wrong", "status"):
                                donor_i = 2 if arm == "wrong" else 3
                                raw = torch.zeros_like(correct)
                                raw[..., coords] = (donor_values[donor_i] - donor_values[origin_i])[..., coords]
                                direction = scaled(raw, target_norm)
                            else:
                                generator = torch.Generator(device=original.device)
                                generator.manual_seed(stable_seed(case["pair_id"], route, size, mode,
                                                                  manifest["gate"]["random_seed"]))
                                raw = torch.zeros_like(correct)
                                raw[..., coords] = torch.randn((correct.shape[0], len(coords_list)),
                                                               generator=generator, device=original.device)
                                direction = scaled(raw, target_norm)
                            if arm != "self":
                                norm_errors[local] = abs(float(torch.linalg.vector_norm(direction) /
                                                               (target_norm + 1e-12)) - 1.0)
                            tp = points(rows[target_index], offsets[target_index], role)
                            value[target_index, tp] = original[target_index, tp] + direction.to(original.dtype)
                    return (value,) + args[1:]

                handle = model.model.layers[manifest["source"]["layer"]].register_forward_pre_hook(hook)
                try:
                    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
                              "use_cache": False, "return_dict": True}
                    if supports:
                        kwargs["logits_to_keep"] = 1
                    output = model(**kwargs)
                finally:
                    handle.remove()
                clean_margin, corrupt_margin = margin(output, 0, donor_rows[0]), margin(output, 1, donor_rows[1])
                for group_index, (route, size, _coords) in enumerate(chunk):
                    key = (route, size)
                    effects = {"sufficiency": {}, "reverse": {}}
                    self_diffs = []
                    errors = []
                    for local_spec, (mode, arm) in enumerate(SPECS):
                        local = group_index * len(SPECS) + local_spec
                        target_index = 4 + local
                        target_margin = margin(output, target_index, rows[target_index])
                        effect = ((target_margin - corrupt_margin) if mode == "sufficiency"
                                  else (clean_margin - target_margin))
                        effects[mode][arm] = effect
                        errors.append(norm_errors[local])
                        if arm == "self":
                            self_diffs.append(abs(effect))
                    per_case[key] = {"pair_id": case["pair_id"], "partition": case["partition"],
                                     "surface": case["surface"], "target_family": case["target_family"],
                                     "route": route, "size": size, "whole_effect": whole[case["pair_id"]],
                                     "suff_gain": effects["sufficiency"],
                                     "reverse_damage": effects["reverse"],
                                     "self_max_abs_diff": max(self_diffs),
                                     "norm_ratio_abs_error_max": max(errors)}
                del output, ids, mask, positions
            records.extend(per_case[key] for key in sorted(per_case))
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"coordinate_cases": case_index + 1, "total": len(cases)}), flush=True)
        core.write_rows(OUT / "raw/qwen3_coordinate_groups.jsonl", records)
        metrics, qualifications = summarize(records, manifest)
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
                   "record_count": len(records), "metrics": metrics,
                   "qualifications": qualifications,
                   "any_sufficiency_route": any(v["minimal_sufficiency_size"] is not None for v in qualifications.values()),
                   "any_reverse_route": any(v["minimal_reverse_size"] is not None for v in qualifications.values()),
                   "runtime": {"placement": placement, "quantization": quant,
                               "all_finite": all(math.isfinite(v) for r in records
                                                 for v in list(r["suff_gain"].values()) +
                                                 list(r["reverse_damage"].values())),
                               "finished_at_utc": datetime.now(timezone.utc).isoformat()},
                   "claim_boundary": "Qwen-specific C057-discovered raw-coordinate group curves on independent C059 confirmation and lockbox"}
        core.save(OUT / "analysis/qwen3_coordinate_group_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_coordinate_group_summary.json")
    if manifest["mediation_was_eligible"]:
        authorization = "run_phase1380_c059_early_mediation"
    else:
        authorization = "close_c059_after_all_frozen_eligible_branches"
    final = {"phase": PHASE, "campaign": CAMPAIGN,
             "any_sufficiency_route": summary["any_sufficiency_route"],
             "any_reverse_route": summary["any_reverse_route"],
             "mediation_was_eligible": manifest["mediation_was_eligible"],
             "authorization": authorization,
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
