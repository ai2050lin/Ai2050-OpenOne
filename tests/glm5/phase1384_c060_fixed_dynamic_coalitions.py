#!/usr/bin/env python3
"""Phase1384: C060 fixed, complement, union, and dynamic coordinate coalitions."""
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

PHASE, CAMPAIGN = 1384, "C060"
CONTRACT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
BEHAVIOR = TESTS / "result/phase1381_c060_qwen_behavior_qualification"
DOSE = TESTS / "result/phase1383_c060_refined_dose_observation"
OUT = TESTS / "result/phase1384_c060_fixed_dynamic_coalitions"
MODEL, CHUNK = "qwen3", 8
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
SPECS = (
    ("sufficiency", "self"), ("sufficiency", "correct"), ("sufficiency", "wrong"),
    ("sufficiency", "status"), ("sufficiency", "random"),
    ("reverse", "self"), ("reverse", "correct"), ("reverse", "status"), ("reverse", "random"),
)


def parents() -> dict:
    final = core.load(DOSE / "analysis/final.json")
    audit = core.load(DOSE / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1384_c060_fixed_dynamic_coalitions" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1383 did not authorize coalition evaluation")
    return core.load(CONTRACT / "protocol/preregistration.json")


def order(score: torch.Tensor) -> list[int]:
    return sorted(range(score.numel()), key=lambda i: (-float(score[i]), i))


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:15], 16) % (2**63 - 1)


def discovery_rankings(protocol: dict) -> dict:
    source_path = DOSE / "raw/response_discovery_family3_differences.pt"
    summary = core.load(DOSE / "analysis/qwen3_refined_dose_summary.json")
    if core.sha(source_path) != summary["discovery_sha256"]:
        raise RuntimeError("discovery source hash changed")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    values = payload["vectors"].float()
    metadata = payload["metadata"]
    if list(values.shape) != [72, 2560] or any(m["partition"] != "response_discovery" for m in metadata):
        raise RuntimeError("invalid discovery source")
    global_order = order(values.abs().mean(0))
    family_orders = {}
    for family in protocol["material"]["families"]:
        selected = values[[m["target_family"] == family for m in metadata]]
        family_orders[family] = order(selected.abs().mean(0))
    return {
        "source_sha256": core.sha(source_path),
        "global": global_order,
        "families": family_orders,
        "selection_scope": "response_discovery only",
    }


def prepare() -> None:
    protocol = parents()
    target = OUT / "protocol/execution_manifest.json"
    if target.exists():
        raise RuntimeError("Phase1384 manifest already exists")
    fixed_path = CONTRACT / "protocol/fixed_coalitions.json"
    if core.sha(fixed_path) != protocol["fixed_coalitions"]["artifact_sha256"]:
        raise RuntimeError("fixed coalition artifact changed")
    fixed = core.load(fixed_path)["groups"]
    rankings = discovery_rankings(protocol)
    core.save(OUT / "protocol/discovery_rankings.json", rankings)
    groups = []
    for route in protocol["fixed_coalitions"]["routes"]:
        groups.append({"group_id": route, "kind": "fixed", "rule": route, "size": len(fixed[route])})
    for rule in protocol["dynamic_coalitions"]["rules"]:
        for size in protocol["dynamic_coalitions"]["sizes"]:
            groups.append({
                "group_id": f"{rule}@{size}",
                "kind": "dynamic",
                "rule": rule,
                "size": int(size),
            })
    cases = [
        r for r in core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
        if r["partition"] in protocol["dynamic_coalitions"]["evaluation_partitions"]
    ]
    if len(cases) != 144:
        raise RuntimeError("evaluation case count mismatch")
    dose_rows = core.rows(DOSE / "raw/qwen3_refined_dose_response.jsonl")
    whole = {
        r["pair_id"]: r["output_effect"]
        for r in dose_rows
        if r["path"] == "family_early"
        and r["mode"] == "sufficiency"
        and r["direction"] == "correct"
        and r["dose"] == 1.0
        and r["partition"] in protocol["dynamic_coalitions"]["evaluation_partitions"]
    }
    if set(whole) != {r["pair_id"] for r in cases}:
        raise RuntimeError("whole-effect reference mismatch")
    core.save(OUT / "protocol/whole_effect_reference.json", whole)
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "dose_final_sha256": core.sha(DOSE / "analysis/final.json"),
        "dose_audit_sha256": core.sha(DOSE / "audit/independent_final_audit.json"),
        "fixed_coalitions_sha256": core.sha(fixed_path),
        "discovery_rankings_sha256": core.sha(OUT / "protocol/discovery_rankings.json"),
        "model": MODEL,
        "precision": "bfloat16-no-quantization",
        "source": {"layer": 3, "role": "family"},
        "groups": groups,
        "specs": [list(v) for v in SPECS],
        "chunk_size": CHUNK,
        "rows_per_full_chunk": 4 + CHUNK * len(SPECS),
        "fixed_gate": protocol["fixed_coalitions"],
        "dynamic_gate": protocol["dynamic_coalitions"],
        "algebra_gate": protocol["coalition_algebra"],
        "case_count": len(cases),
        "case_ids": [r["pair_id"] for r in cases],
        "dynamic_random_seed": 6001384,
        "post_reveal_changes_forbidden": True,
        "allowed_observables": protocol["allowed_observables"],
        "forbidden": protocol["forbidden"],
        "mediation_was_eligible": core.load(DOSE / "analysis/final.json")["mediation_eligible"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(target, manifest)
    core.write_rows(OUT / "material/evaluation_pairs.jsonl", cases)
    print(json.dumps({
        "phase": PHASE,
        "group_count": len(groups),
        "fixed_count": sum(g["kind"] == "fixed" for g in groups),
        "dynamic_count": sum(g["kind"] == "dynamic" for g in groups),
        "case_count": len(cases),
        "rows_per_full_chunk": manifest["rows_per_full_chunk"],
    }, indent=2))


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


def coords_for(group: dict, case: dict, correct_full: torch.Tensor, fixed: dict,
               rankings: dict, device: torch.device, seed: int) -> torch.Tensor:
    if group["kind"] == "fixed":
        values = fixed[group["rule"]]
    elif group["rule"] == "per_example_top_abs":
        values = torch.argsort(correct_full.flatten().abs(), descending=True, stable=True)[:group["size"]].cpu().tolist()
    elif group["rule"] == "per_example_bottom_abs":
        values = torch.argsort(correct_full.flatten().abs(), descending=False, stable=True)[:group["size"]].cpu().tolist()
    elif group["rule"] == "discovery_global_magnitude":
        values = rankings["global"][:group["size"]]
    elif group["rule"] == "discovery_family_magnitude":
        values = rankings["families"][case["target_family"]][:group["size"]]
    elif group["rule"] == "deterministic_random_prefix":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed(case["pair_id"], seed))
        values = torch.randperm(correct_full.numel(), generator=generator)[:group["size"]].tolist()
    else:
        raise RuntimeError(f"unknown group rule: {group['rule']}")
    return torch.tensor(values, dtype=torch.long, device=device)


def metric(rows: list[dict], gate: dict) -> dict:
    suff_c = [r["suff_gain"]["correct"] for r in rows]
    suff_adv = [
        r["suff_gain"]["correct"] - max(r["suff_gain"][d] for d in ("wrong", "status", "random"))
        for r in rows
    ]
    suff_win = [
        r["suff_gain"]["correct"] > max(r["suff_gain"][d] for d in ("wrong", "status", "random"))
        for r in rows
    ]
    fractions = [r["suff_gain"]["correct"] / r["whole_effect"] for r in rows if abs(r["whole_effect"]) > 1e-12]
    rev_c = [r["reverse_damage"]["correct"] for r in rows]
    rev_adv = [r["reverse_damage"]["correct"] - r["reverse_damage"]["status"] for r in rows]
    rev_win = [r["reverse_damage"]["correct"] > r["reverse_damage"]["status"] for r in rows]
    result = {
        "count": len(rows),
        "suff_gain_median": statistics.median(suff_c),
        "suff_advantage_median": statistics.median(suff_adv),
        "suff_win_fraction": sum(suff_win) / len(suff_win),
        "whole_effect_fraction_median": statistics.median(fractions),
        "reverse_damage_median": statistics.median(rev_c),
        "reverse_over_status_median": statistics.median(rev_adv),
        "reverse_over_status_win_fraction": sum(rev_win) / len(rev_win),
        "self_max_abs_diff": max(r["self_max_abs_diff"] for r in rows),
        "norm_ratio_abs_error_max": max(r["norm_ratio_abs_error_max"] for r in rows),
    }
    result["sufficiency_qualified"] = (
        result["suff_gain_median"] >= gate["suff_gain_median_min"]
        and result["suff_advantage_median"] >= gate["suff_advantage_median_min"]
        and result["suff_win_fraction"] >= gate["suff_win_min"]
        and result["whole_effect_fraction_median"] >= gate["whole_effect_fraction_median_min"]
        and result["self_max_abs_diff"] <= gate["self_max_abs_diff"]
    )
    result["reverse_qualified"] = (
        result["reverse_damage_median"] >= gate["reverse_damage_median_min"]
        and result["reverse_over_status_median"] >= gate["reverse_over_status_median_min"]
        and result["reverse_over_status_win_fraction"] >= gate["reverse_over_status_win_min"]
        and result["self_max_abs_diff"] <= gate["self_max_abs_diff"]
    )
    return result


def summarize(records: list[dict], manifest: dict) -> tuple[dict, dict, dict]:
    groups = manifest["groups"]
    fixed_gate = manifest["fixed_gate"]
    metrics, qualifications = {}, {}
    for group in groups:
        group_rows = [r for r in records if r["group_id"] == group["group_id"]]
        splits = {}
        for partition in ("pooled", "confirmation", "lockbox"):
            rows = group_rows if partition == "pooled" else [r for r in group_rows if r["partition"] == partition]
            splits[partition] = metric(rows, fixed_gate)
        metrics[group["group_id"]] = {"group": group, "splits": splits}
        qualifications[group["group_id"]] = {
            "sufficiency_all_splits": all(splits[p]["sufficiency_qualified"] for p in splits),
            "reverse_all_splits": all(splits[p]["reverse_qualified"] for p in splits),
        }

    dynamic = {}
    random_records = {
        (r["pair_id"], r["size"]): r
        for r in records
        if r["kind"] == "dynamic" and r["rule"] == "deterministic_random_prefix"
    }
    dg = manifest["dynamic_gate"]
    for group in groups:
        if group["kind"] != "dynamic" or group["rule"] == "deterministic_random_prefix":
            continue
        group_rows = [r for r in records if r["group_id"] == group["group_id"]]
        split_result = {}
        for partition in ("pooled", "confirmation", "lockbox"):
            rows = group_rows if partition == "pooled" else [r for r in group_rows if r["partition"] == partition]
            diffs = [
                r["suff_gain"]["correct"] - random_records[(r["pair_id"], r["size"])]["suff_gain"]["correct"]
                for r in rows
            ]
            fractions = [r["suff_gain"]["correct"] / r["whole_effect"] for r in rows if abs(r["whole_effect"]) > 1e-12]
            item = {
                "count": len(rows),
                "advantage_over_random_median": statistics.median(diffs),
                "win_over_random_fraction": sum(v > 0 for v in diffs) / len(diffs),
                "whole_effect_fraction_median": statistics.median(fractions),
            }
            item["qualified"] = (
                item["advantage_over_random_median"] >= dg["dynamic_advantage_over_random_median_min"]
                and item["win_over_random_fraction"] >= dg["dynamic_win_over_random_min"]
                and (
                    group["size"] > dg["small_group_max_size"]
                    or item["whole_effect_fraction_median"] >= dg["small_group_whole_effect_fraction_min"]
                )
            )
            split_result[partition] = item
        dynamic[group["group_id"]] = {
            "group": group,
            "splits": split_result,
            "qualified_all_splits": all(v["qualified"] for v in split_result.values()),
        }

    by_case = {(r["pair_id"], r["group_id"]): r for r in records}
    algebra = {}
    for partition in ("pooled", "confirmation", "lockbox"):
        pair_ids = sorted({
            r["pair_id"] for r in records
            if partition == "pooled" or r["partition"] == partition
        })
        rows = []
        for pair_id in pair_ids:
            s = by_case[(pair_id, "inherited_S1024")]
            c = by_case[(pair_id, "inherited_C1536")]
            f = by_case[(pair_id, "inherited_full2560")]
            a_s = s["suff_gain"]["correct"] - s["suff_gain"]["status"]
            a_c = c["suff_gain"]["correct"] - c["suff_gain"]["status"]
            a_f = f["suff_gain"]["correct"] - f["suff_gain"]["status"]
            gamma = a_f - a_s - a_c
            rows.append({
                "pair_id": pair_id,
                "A_S": a_s,
                "A_C": a_c,
                "A_full": a_f,
                "Gamma": gamma,
                "cancellation_candidate": a_s >= 0.25 and a_f <= 0.0 and gamma < 0.0,
            })
        algebra[partition] = {
            "count": len(rows),
            "A_S_median": statistics.median(r["A_S"] for r in rows),
            "A_C_median": statistics.median(r["A_C"] for r in rows),
            "A_full_median": statistics.median(r["A_full"] for r in rows),
            "Gamma_median": statistics.median(r["Gamma"] for r in rows),
            "cancellation_candidate_fraction": sum(r["cancellation_candidate"] for r in rows) / len(rows),
        }
    return metrics, qualifications, {"dynamic_comparison": dynamic, "inherited_algebra": algebra}


@torch.inference_mode()
def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/qwen3_coalition_summary.json").exists():
        raise RuntimeError("Phase1384 run already exists")
    fixed = core.load(CONTRACT / "protocol/fixed_coalitions.json")["groups"]
    rankings = core.load(OUT / "protocol/discovery_rankings.json")
    whole = core.load(OUT / "protocol/whole_effect_reference.json")
    cases = core.rows(OUT / "material/evaluation_pairs.jsonl")
    compiled = {r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_active.jsonl")}
    compiled.update({r["case_id"]: r for r in core.rows(CONTRACT / "compiled/qwen3_status.jsonl")})
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        records = []
        for case_index, case in enumerate(cases):
            donor_rows = [compiled[case[key]] for key in DONOR_KEYS]
            per_case = {}
            for chunk_start in range(0, len(manifest["groups"]), manifest["chunk_size"]):
                chunk = manifest["groups"][chunk_start:chunk_start + manifest["chunk_size"]]
                rows = list(donor_rows)
                for _group in chunk:
                    for mode, _arm in SPECS:
                        rows.append(donor_rows[1 if mode == "sufficiency" else 0])
                ids, mask, positions, offsets = make_batch(rows, pad, device)
                norm_errors = [0.0] * (len(chunk) * len(SPECS))

                def hook(_module, args):
                    original = args[0]
                    value = original.clone()
                    role = manifest["source"]["role"]
                    donor_values = [original[i, points(rows[i], offsets[i], role)].float() for i in range(4)]
                    full_for_mask = donor_values[0] - donor_values[1]
                    coord_cache = {
                        group["group_id"]: coords_for(
                            group, case, full_for_mask, fixed, rankings, original.device,
                            manifest["dynamic_random_seed"],
                        )
                        for group in chunk
                    }
                    for group_index, group in enumerate(chunk):
                        coords = coord_cache[group["group_id"]]
                        for local_spec, (mode, arm) in enumerate(SPECS):
                            local = group_index * len(SPECS) + local_spec
                            target_index = 4 + local
                            origin_i, goal_i = ((1, 0) if mode == "sufficiency" else (0, 1))
                            full = donor_values[goal_i] - donor_values[origin_i]
                            correct = torch.zeros_like(full)
                            correct[..., coords] = full[..., coords]
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
                                generator.manual_seed(stable_seed(
                                    case["pair_id"], group["group_id"], mode, manifest["dynamic_random_seed"]
                                ))
                                raw = torch.zeros_like(correct)
                                raw[..., coords] = torch.randn(
                                    (correct.shape[0], coords.numel()), generator=generator,
                                    device=original.device, dtype=torch.float32,
                                )
                                direction = scaled(raw, target_norm)
                            if arm != "self":
                                norm_errors[local] = abs(float(
                                    torch.linalg.vector_norm(direction) / (target_norm + 1e-12)
                                ) - 1.0)
                            tp = points(rows[target_index], offsets[target_index], role)
                            value[target_index, tp] = original[target_index, tp] + direction.to(original.dtype)
                    return (value,) + args[1:]

                handle = model.model.layers[manifest["source"]["layer"]].register_forward_pre_hook(hook)
                try:
                    kwargs = {
                        "input_ids": ids,
                        "attention_mask": mask,
                        "position_ids": positions,
                        "use_cache": False,
                        "return_dict": True,
                    }
                    if supports:
                        kwargs["logits_to_keep"] = 1
                    output = model(**kwargs)
                finally:
                    handle.remove()
                clean_margin = margin(output, 0, donor_rows[0])
                corrupt_margin = margin(output, 1, donor_rows[1])
                for group_index, group in enumerate(chunk):
                    effects = {"sufficiency": {}, "reverse": {}}
                    self_diffs, errors = [], []
                    for local_spec, (mode, arm) in enumerate(SPECS):
                        local = group_index * len(SPECS) + local_spec
                        target_index = 4 + local
                        target_margin = margin(output, target_index, rows[target_index])
                        effect = (
                            target_margin - corrupt_margin
                            if mode == "sufficiency"
                            else clean_margin - target_margin
                        )
                        effects[mode][arm] = effect
                        errors.append(norm_errors[local])
                        if arm == "self":
                            self_diffs.append(abs(effect))
                    per_case[group["group_id"]] = {
                        "pair_id": case["pair_id"],
                        "partition": case["partition"],
                        "surface": case["surface"],
                        "target_family": case["target_family"],
                        **group,
                        "whole_effect": whole[case["pair_id"]],
                        "suff_gain": effects["sufficiency"],
                        "reverse_damage": effects["reverse"],
                        "self_max_abs_diff": max(self_diffs),
                        "norm_ratio_abs_error_max": max(errors),
                    }
                del output, ids, mask, positions
            records.extend(per_case[g["group_id"]] for g in manifest["groups"])
            if (case_index + 1) % 12 == 0:
                print(json.dumps({"coalition_cases": case_index + 1, "total": len(cases)}), flush=True)
        core.write_rows(OUT / "raw/qwen3_coalitions.jsonl", records)
        metrics, qualifications, comparisons = summarize(records, manifest)
        inherited = qualifications["inherited_S1024"]
        new_random_hits = sum(
            qualifications[f"new_random_{i}_S1024"]["reverse_all_splits"]
            for i in range(1, 5)
        )
        dynamic_hits = [k for k, v in comparisons["dynamic_comparison"].items() if v["qualified_all_splits"]]
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "model": MODEL,
            "record_count": len(records),
            "metrics": metrics,
            "qualifications": qualifications,
            "inherited_S1024_reverse_replicated": inherited["reverse_all_splits"],
            "new_random_S1024_reverse_hit_count": new_random_hits,
            "dynamic_qualified_all_splits": dynamic_hits,
            **comparisons,
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "all_finite": all(
                    math.isfinite(v)
                    for r in records
                    for v in list(r["suff_gain"].values()) + list(r["reverse_damage"].values())
                ),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "claim_boundary": "Qwen-specific fixed-candidate replication and discovery-only dynamic coalition comparison",
        }
        core.save(OUT / "analysis/qwen3_coalition_summary.json", summary)
        print(json.dumps({
            "phase": PHASE,
            "record_count": len(records),
            "inherited_S1024_reverse_replicated": summary["inherited_S1024_reverse_replicated"],
            "new_random_S1024_reverse_hit_count": new_random_hits,
            "dynamic_qualified_all_splits": dynamic_hits,
            "inherited_algebra": comparisons["inherited_algebra"],
        }, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    summary = core.load(OUT / "analysis/qwen3_coalition_summary.json")
    authorization = (
        "run_phase1385_c060_early_mediation"
        if manifest["mediation_was_eligible"]
        else "run_phase1386_c060_campaign_closure"
    )
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "inherited_S1024_reverse_replicated": summary["inherited_S1024_reverse_replicated"],
        "new_random_S1024_reverse_hit_count": summary["new_random_S1024_reverse_hit_count"],
        "dynamic_qualified_count": len(summary["dynamic_qualified_all_splits"]),
        "mediation_was_eligible": manifest["mediation_was_eligible"],
        "authorization": authorization,
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
