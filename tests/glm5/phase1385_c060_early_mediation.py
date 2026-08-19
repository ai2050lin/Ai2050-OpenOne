#!/usr/bin/env python3
"""Phase1385: C060 early whole-state mediation with typed checkpoint resets."""
from __future__ import annotations

import argparse
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

PHASE, CAMPAIGN = 1385, "C060"
CONTRACT = TESTS / "result/phase1380_c060_conditional_coalition_campaign_contract"
BEHAVIOR = TESTS / "result/phase1381_c060_qwen_behavior_qualification"
DOSE = TESTS / "result/phase1383_c060_refined_dose_observation"
COALITION = TESTS / "result/phase1384_c060_fixed_dynamic_coalitions"
OUT = TESTS / "result/phase1385_c060_early_mediation"
MODEL = "qwen3"
DONOR_KEYS = ("clean_true", "corrupt_false", "wrong_identity_true", "status_true")
ARMS = (
    "upstream_self",
    "upstream_rescue",
    "upstream_wrong",
    "upstream_status",
    "rescue_query_corrupt",
    "rescue_query_clean",
    "rescue_query_wrong",
    "rescue_boundary_corrupt",
    "rescue_boundary_clean",
    "rescue_boundary_wrong",
)


def parents() -> dict:
    final = core.load(COALITION / "analysis/final.json")
    audit = core.load(COALITION / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1385_c060_early_mediation" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1384 did not authorize mediation")
    dose_final = core.load(DOSE / "analysis/final.json")
    if not dose_final.get("mediation_eligible"):
        raise RuntimeError("early mediation was not independently eligible")
    return core.load(CONTRACT / "protocol/preregistration.json")


def prepare() -> None:
    protocol = parents()
    target = OUT / "protocol/execution_manifest.json"
    if target.exists():
        raise RuntimeError("Phase1385 manifest already exists")
    cases = [
        r for r in core.rows(BEHAVIOR / "material/eligible_pairs.jsonl")
        if r["partition"] in ("confirmation", "lockbox")
    ]
    if len(cases) != 144:
        raise RuntimeError("mediation case count mismatch")
    path = protocol["paths"]["family_early"]
    query, boundary = path["checkpoints"]
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "coalition_final_sha256": core.sha(COALITION / "analysis/final.json"),
        "coalition_audit_sha256": core.sha(COALITION / "audit/independent_final_audit.json"),
        "model": MODEL,
        "precision": "bfloat16-no-quantization",
        "source": path["source"],
        "query_checkpoint": query,
        "boundary_checkpoint": boundary,
        "arms": list(ARMS),
        "rows_per_case": 4 + len(ARMS),
        "case_count": len(cases),
        "case_ids": [r["pair_id"] for r in cases],
        "gate": protocol["mediation"],
        "partitions": ["confirmation", "lockbox"],
        "same_forward_natural_donors": True,
        "post_reveal_changes_forbidden": True,
        "allowed_observables": protocol["allowed_observables"],
        "forbidden": protocol["forbidden"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(target, manifest)
    core.write_rows(OUT / "material/mediation_pairs.jsonl", cases)
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


def copy_role(value, original, target_index: int, target: dict, target_offset: int,
              source_index: int, source: dict, source_offset: int, role: str) -> None:
    tp = points(target, target_offset, role)
    sp = points(source, source_offset, role)
    if len(tp) != len(sp):
        raise RuntimeError(f"role span mismatch for {role}")
    value[target_index, tp] = original[source_index, sp]


def margin(output, row_index: int, row: dict) -> float:
    logits = output.logits[row_index, -1].float()
    return float(logits[row["candidate_ids"][0][0]] - logits[row["candidate_ids"][1][0]])


def split_metric(rows: list[dict], gate: dict) -> dict:
    rescue = [r["effects"]["upstream_rescue"] for r in rows]
    query_block = [
        (r["effects"]["upstream_rescue"] - r["effects"]["rescue_query_corrupt"])
        / (abs(r["effects"]["upstream_rescue"]) + 1e-12)
        for r in rows
    ]
    boundary_block = [
        (r["effects"]["upstream_rescue"] - r["effects"]["rescue_boundary_corrupt"])
        / (abs(r["effects"]["upstream_rescue"]) + 1e-12)
        for r in rows
    ]
    query_clean_loss = [
        abs(r["effects"]["upstream_rescue"] - r["effects"]["rescue_query_clean"])
        / (abs(r["effects"]["upstream_rescue"]) + 1e-12)
        for r in rows
    ]
    boundary_clean_loss = [
        abs(r["effects"]["upstream_rescue"] - r["effects"]["rescue_boundary_clean"])
        / (abs(r["effects"]["upstream_rescue"]) + 1e-12)
        for r in rows
    ]
    metric = {
        "count": len(rows),
        "upstream_rescue_median": statistics.median(rescue),
        "query_block_fraction_median": statistics.median(query_block),
        "query_block_positive_fraction": sum(v > 0 for v in query_block) / len(query_block),
        "boundary_block_fraction_median": statistics.median(boundary_block),
        "boundary_block_positive_fraction": sum(v > 0 for v in boundary_block) / len(boundary_block),
        "query_clean_control_loss_fraction_median": statistics.median(query_clean_loss),
        "boundary_clean_control_loss_fraction_median": statistics.median(boundary_clean_loss),
        "self_max_abs_diff": max(abs(r["effects"]["upstream_self"]) for r in rows),
    }
    checks = {
        "upstream_rescue": metric["upstream_rescue_median"] >= gate["upstream_rescue_median_min"],
        "query_block": metric["query_block_fraction_median"] >= gate["query_block_fraction_median_min"],
        "query_positive": metric["query_block_positive_fraction"] >= gate["query_block_positive_fraction_min"],
        "boundary_block": metric["boundary_block_fraction_median"] >= gate["boundary_block_fraction_median_min"],
        "boundary_positive": metric["boundary_block_positive_fraction"] >= gate["boundary_block_positive_fraction_min"],
        "query_clean_control": metric["query_clean_control_loss_fraction_median"] <= gate["clean_checkpoint_control_loss_fraction_max"],
        "boundary_clean_control": metric["boundary_clean_control_loss_fraction_median"] <= gate["clean_checkpoint_control_loss_fraction_max"],
        "self": metric["self_max_abs_diff"] <= 1e-4,
    }
    metric["checks"] = checks
    metric["qualified"] = all(checks.values())
    return metric


@torch.inference_mode()
def run() -> None:
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    if (OUT / "analysis/qwen3_early_mediation_summary.json").exists():
        raise RuntimeError("Phase1385 run already exists")
    cases = core.rows(OUT / "material/mediation_pairs.jsonl")
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
            rows = donor_rows + [donor_rows[1] for _ in ARMS]
            ids, mask, positions, offsets = make_batch(rows, pad, device)
            arm_index = {arm: 4 + i for i, arm in enumerate(ARMS)}
            handles = []

            def source_hook(_module, args):
                original = args[0]
                value = original.clone()
                role = manifest["source"]["role"]
                source_for = {
                    "upstream_self": 1,
                    "upstream_rescue": 0,
                    "upstream_wrong": 2,
                    "upstream_status": 3,
                    "rescue_query_corrupt": 0,
                    "rescue_query_clean": 0,
                    "rescue_query_wrong": 0,
                    "rescue_boundary_corrupt": 0,
                    "rescue_boundary_clean": 0,
                    "rescue_boundary_wrong": 0,
                }
                for arm, source_index in source_for.items():
                    target_index = arm_index[arm]
                    copy_role(
                        value, original, target_index, rows[target_index], offsets[target_index],
                        source_index, rows[source_index], offsets[source_index], role,
                    )
                return (value,) + args[1:]

            def checkpoint_hook(role: str, mappings: dict[str, int]):
                def hook(_module, args):
                    original = args[0]
                    value = original.clone()
                    for arm, source_index in mappings.items():
                        target_index = arm_index[arm]
                        copy_role(
                            value, original, target_index, rows[target_index], offsets[target_index],
                            source_index, rows[source_index], offsets[source_index], role,
                        )
                    return (value,) + args[1:]
                return hook

            handles.append(model.model.layers[manifest["source"]["layer"]].register_forward_pre_hook(source_hook))
            q = manifest["query_checkpoint"]
            handles.append(model.model.layers[q["layer"]].register_forward_pre_hook(checkpoint_hook(q["role"], {
                "rescue_query_corrupt": 1,
                "rescue_query_clean": 0,
                "rescue_query_wrong": 2,
            })))
            b = manifest["boundary_checkpoint"]
            handles.append(model.model.layers[b["layer"]].register_forward_pre_hook(checkpoint_hook(b["role"], {
                "rescue_boundary_corrupt": 1,
                "rescue_boundary_clean": 0,
                "rescue_boundary_wrong": 2,
            })))
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
                for handle in handles:
                    handle.remove()
            corrupt_margin = margin(output, 1, donor_rows[1])
            effects = {
                arm: margin(output, arm_index[arm], rows[arm_index[arm]]) - corrupt_margin
                for arm in ARMS
            }
            records.append({
                "pair_id": case["pair_id"],
                "partition": case["partition"],
                "surface": case["surface"],
                "target_family": case["target_family"],
                "clean_margin": margin(output, 0, donor_rows[0]),
                "corrupt_margin": corrupt_margin,
                "effects": effects,
                "all_finite": all(math.isfinite(v) for v in effects.values()),
            })
            del output, ids, mask, positions
            if (case_index + 1) % 24 == 0:
                print(json.dumps({"mediation_cases": case_index + 1, "total": len(cases)}), flush=True)
        core.write_rows(OUT / "raw/qwen3_early_mediation.jsonl", records)
        splits = {}
        for partition in ("pooled", "confirmation", "lockbox"):
            rows = records if partition == "pooled" else [r for r in records if r["partition"] == partition]
            splits[partition] = split_metric(rows, manifest["gate"])
        qualified = all(v["qualified"] for v in splits.values())
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "model": MODEL,
            "record_count": len(records),
            "splits": splits,
            "mediation_qualified": qualified,
            "runtime": {
                "placement": placement,
                "quantization": quant,
                "all_finite": all(r["all_finite"] for r in records),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "claim_boundary": "Qwen-specific typed whole-state checkpoint mediation in C060 confirmation and lockbox",
        }
        core.save(OUT / "analysis/qwen3_early_mediation_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize() -> None:
    summary = core.load(OUT / "analysis/qwen3_early_mediation_summary.json")
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "mediation_qualified": summary["mediation_qualified"],
        "authorization": "run_phase1386_c060_campaign_closure",
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
