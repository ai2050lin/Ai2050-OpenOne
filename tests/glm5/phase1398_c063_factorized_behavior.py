#!/usr/bin/env python3
"""Phase1398: Qwen3 four-answer behavior qualification for C063."""
from __future__ import annotations

import inspect
import json
import math
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

PHASE, CAMPAIGN = 1398, "C063"
CONTRACT = TESTS / "result/phase1397_c063_identity_polarity_campaign_contract"
OUT = TESTS / "result/phase1398_c063_factorized_behavior"
BATCH = 12


def make_batch(rows, pad, device):
    width = max(len(r["prompt_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, width - len(values):] = values
        mask[i, width - len(values):] = 1
    pos = mask.cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 0)
    return ids, mask, pos


@torch.inference_mode()
def forward(model, rows, pad, device, supports):
    ids, mask, pos = make_batch(rows, pad, device)
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": pos, "use_cache": False, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    result = []
    for i, row in enumerate(rows):
        logits = output.logits[i, -1].float()
        scores = [float(logits[value[0]]) for value in row["candidate_ids"]]
        prediction = max(range(len(scores)), key=scores.__getitem__)
        result.append({"scores": scores, "prediction": prediction})
    return result


def accuracy(rows):
    return sum(r["correct"] for r in rows) / len(rows)


def main() -> None:
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1398 already exists")
    contract_final = core.load(CONTRACT / "analysis/final.json")
    contract_audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    protocol = core.load(CONTRACT / "protocol/preregistration.json")
    if contract_final["authorization"] != "run_phase1398_c063_factorized_behavior" or not contract_audit["all_checks_passed"]:
        raise RuntimeError("Phase1397 did not authorize behavior")

    active_source = core.rows(CONTRACT / "material/active_cases.jsonl")
    status_source = core.rows(CONTRACT / "material/status_cases.jsonl")
    active_compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    status_compiled = core.rows(CONTRACT / "compiled/qwen3_status.jsonl")
    factor_sets = core.rows(CONTRACT / "material/factor_sets.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16("qwen3")
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        active_predictions, status_predictions, first = [], [], None
        for start in range(0, len(active_compiled), BATCH):
            values = forward(model, active_compiled[start:start + BATCH], pad, device, supports)
            if start == 0:
                first = values
            active_predictions.extend(values)
        repeat = forward(model, active_compiled[:BATCH], pad, device, supports)
        numeric = max(abs(a["scores"][j] - b["scores"][j]) for a, b in zip(first, repeat) for j in range(4))
        for start in range(0, len(status_compiled), BATCH):
            status_predictions.extend(forward(model, status_compiled[start:start + BATCH], pad, device, supports))

        active = [{**source, **pred, "correct": pred["prediction"] == source["gold_position"]}
                  for source, pred in zip(active_source, active_predictions)]
        status = [{**source, **pred, "correct": pred["prediction"] == source["gold_position"]}
                  for source, pred in zip(status_source, status_predictions)]
        active_by = {r["case_id"]: r for r in active}
        status_by = {r["case_id"]: r for r in status}
        factor_case_keys = ("recipient", "surface_same", "member_same", "family_only", "answer_only", "family_and_answer", "polarity_only", "family_and_polarity")
        eligible = [r for r in factor_sets if all(active_by[r[k]]["correct"] for k in factor_case_keys) and status_by[r["status_null"]]["correct"]]

        gate = protocol["behavior"]
        per_cell = protocol["material"]["eligible_per_family_partition_surface_key"]
        family_results, selected = {}, []
        for family in protocol["material"]["families"]:
            rows = [r for r in active if r["record_family"] == family]
            status_rows = [r for r in status if r["record_family"] == family]
            paired = defaultdict(list)
            for row in rows:
                paired[(row["pair"], row["index"], row["surface"], row["key"], row["record_family"])].append(row["correct"])
            family_eligible = [r for r in eligible if r["family"] == family]
            cells = defaultdict(list)
            for row in family_eligible:
                cells[(row["partition"], row["surface"], row["key"])].append(row)
            chosen = []
            if len(cells) == 18 and min(len(v) for v in cells.values()) >= per_cell:
                for key in sorted(cells):
                    chosen.extend(sorted(cells[key], key=lambda r: r["set_id"])[:per_cell])
            metrics = {
                "active_count": len(rows),
                "active_accuracy": accuracy(rows),
                "partition": {p: accuracy([r for r in rows if r["partition"] == p]) for p in protocol["material"]["partitions"]},
                "surface": {s: accuracy([r for r in rows if r["surface"] == s]) for s in protocol["material"]["surfaces"]},
                "key": {k: accuracy([r for r in rows if r["key"] == k]) for k in ("alpha", "beta")},
                "truth": {str(v).lower(): accuracy([r for r in rows if r["truth"] == v]) for v in (True, False)},
                "pair_all_fraction": sum(all(v) for v in paired.values()) / len(paired),
                "status_accuracy": accuracy(status_rows),
                "eligible_count": len(family_eligible),
                "eligible_cell_min": min((len(v) for v in cells.values()), default=0),
                "selected_count": len(chosen),
            }
            checks = {
                "active": metrics["active_accuracy"] >= gate["family_active_accuracy_min"],
                "partition": min(metrics["partition"].values()) >= gate["family_partition_min"],
                "surface": min(metrics["surface"].values()) >= gate["family_surface_min"],
                "key": min(metrics["key"].values()) >= gate["family_key_min"],
                "truth": min(metrics["truth"].values()) >= gate["family_truth_min"],
                "pair_all": metrics["pair_all_fraction"] >= gate["family_quartet_all_min"],
                "status": metrics["status_accuracy"] >= gate["status_accuracy_min"],
                "eligible_cells": len(cells) == 18 and metrics["eligible_cell_min"] >= per_cell,
                "selected": len(chosen) == protocol["material"]["selected_per_family"],
            }
            family_results[family] = {"metrics": metrics, "checks": checks, "qualified": all(checks.values())}
            if all(checks.values()):
                selected.extend(chosen)

        qualified = [f for f, result in family_results.items() if result["qualified"]]
        breadth = {
            "family_count": len(qualified) >= protocol["material"]["minimum_qualified_families"],
            "status_global": accuracy(status) >= gate["status_accuracy_min"],
            "numeric": numeric <= gate["same_shape_repeat_max_abs_diff"],
            "finite": all(math.isfinite(score) for r in active + status for score in r["scores"]),
        }
        behavior_qualified = all(breadth.values())
        core.write_rows(OUT / "raw/active_behavior.jsonl", active)
        core.write_rows(OUT / "raw/status_behavior.jsonl", status)
        core.write_rows(OUT / "material/eligible_factor_sets.jsonl", selected)
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "family_results": family_results,
            "qualified_families": qualified, "breadth_checks": breadth,
            "behavior_qualified": behavior_qualified, "selected_count": len(selected),
            "selected_partition_counts": {p: sum(r["partition"] == p for r in selected) for p in protocol["material"]["partitions"]},
            "global": {"active_accuracy": accuracy(active), "status_accuracy": accuracy(status), "numeric_same_shape_max_abs_diff": numeric},
            "runtime": {"placement": placement, "quantization": quant, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/qwen3_behavior_summary.json", summary)
        authorization = "run_phase1399_c063_state_swap_camera" if behavior_qualified else "close_c063_at_behavior_gate"
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN, "behavior_qualified": behavior_qualified,
                  "qualified_families": qualified, "authorization": authorization})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
