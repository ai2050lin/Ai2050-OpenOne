#!/usr/bin/env python3
"""Phase1370: qualify the independent C057 behavior object before hidden access."""
from __future__ import annotations

import inspect
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1370, "C057"
CONTRACT = TESTS / "result/phase1369_c057_independent_relation_campaign_contract"
OUT = TESTS / "result/phase1370_c057_qwen_behavior_qualification"
MODEL = "qwen3"
BATCH = 8


def parent() -> dict:
    final = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1370_c057_behavior_qualification" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1369 did not authorize behavior")
    return core.load(CONTRACT / "protocol/preregistration.json")


def make_batch(rows: list[dict], pad: int, device: torch.device):
    width = max(len(row["prompt_ids"]) for row in rows)
    ids = torch.full((len(rows), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, width - len(value):] = value
        mask[index, width - len(value):] = 1
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions


@torch.inference_mode()
def forward(model, rows: list[dict], pad: int, device: torch.device, supports: bool) -> list[dict]:
    ids, mask, positions = make_batch(rows, pad, device)
    kwargs = {"input_ids": ids, "attention_mask": mask, "position_ids": positions,
              "use_cache": False, "return_dict": True}
    if supports:
        kwargs["logits_to_keep"] = 1
    output = model(**kwargs)
    result = []
    for index, row in enumerate(rows):
        logits = output.logits[index, -1].float()
        scores = [float(logits[ids_[0]]) for ids_ in row["candidate_ids"]]
        result.append({"scores": scores, "margin": scores[0] - scores[1],
                       "prediction": int(scores[1] > scores[0])})
    del output, ids, mask, positions
    return result


def summarize_active(source: list[dict], records: list[dict]) -> dict:
    values = [{**meta, **record} for meta, record in zip(source, records)]
    for row in values:
        row["correct"] = row["prediction"] == row["gold_position"]
    quartets = defaultdict(list)
    for row in values:
        quartets[row["quartet_key"]].append(row["correct"])
    summary = {
        "count": len(values), "accuracy": sum(r["correct"] for r in values) / len(values),
        "partition": {name: sum(r["correct"] for r in values if r["partition"] == name) /
                      sum(r["partition"] == name for r in values) for name in sorted({r["partition"] for r in values})},
        "surface": {name: sum(r["correct"] for r in values if r["surface"] == name) /
                    sum(r["surface"] == name for r in values) for name in sorted({r["surface"] for r in values})},
        "family": {name: sum(r["correct"] for r in values if r["target_family"] == name) /
                   sum(r["target_family"] == name for r in values) for name in sorted({r["target_family"] for r in values})},
        "truth": {str(name).lower(): sum(r["correct"] for r in values if r["truth"] == name) /
                  sum(r["truth"] == name for r in values) for name in (False, True)},
        "quartet_all_fraction": sum(all(group) for group in quartets.values()) / len(quartets),
    }
    return summary, values


def summarize_status(source: list[dict], records: list[dict]) -> dict:
    values = [{**meta, **record} for meta, record in zip(source, records)]
    for row in values:
        row["correct"] = row["prediction"] == row["gold_position"]
    return {
        "count": len(values), "accuracy": sum(r["correct"] for r in values) / len(values),
        "partition": {name: sum(r["correct"] for r in values if r["partition"] == name) /
                      sum(r["partition"] == name for r in values) for name in sorted({r["partition"] for r in values})},
    }, values


def main() -> None:
    protocol = parent()
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1370 already exists")
    active_source = core.rows(CONTRACT / "material/active_membership_cases.jsonl")
    status_source = core.rows(CONTRACT / "material/status_cases.jsonl")
    active_compiled = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    status_compiled = core.rows(CONTRACT / "compiled/qwen3_status.jsonl")
    pairs = core.rows(CONTRACT / "material/candidate_pairs.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        active_records, status_records = [], []
        numeric_reference = None
        for start in range(0, len(active_compiled), BATCH):
            group = active_compiled[start:start + BATCH]
            result = forward(model, group, pad, device, supports)
            if start == 0:
                numeric_reference = result
            active_records.extend(result)
        repeat = forward(model, active_compiled[:BATCH], pad, device, supports)
        numeric_max = max(abs(a["scores"][j] - b["scores"][j])
                          for a, b in zip(numeric_reference, repeat) for j in range(2))
        for start in range(0, len(status_compiled), BATCH):
            status_records.extend(forward(model, status_compiled[start:start + BATCH], pad, device, supports))
        active_summary, active_values = summarize_active(active_source, active_records)
        status_summary, status_values = summarize_status(status_source, status_records)
        active_by = {row["case_id"]: row for row in active_values}
        status_by = {row["case_id"]: row for row in status_values}
        eligible = []
        for pair in pairs:
            if all(active_by[pair[key]]["correct"] for key in ("clean_true", "corrupt_false", "wrong_identity_true")) \
                    and status_by[pair["status_true"]]["correct"]:
                eligible.append(pair)
        cells = defaultdict(list)
        for row in eligible:
            cells[(row["target_family"], row["partition"], row["surface"])].append(row)
        selected = []
        per_cell = protocol["material"]["eligible_cases_per_cell"]
        if len(cells) == 48 and min(len(v) for v in cells.values()) >= per_cell:
            for key in sorted(cells):
                selected.extend(sorted(cells[key], key=lambda row: row["pair_id"])[:per_cell])
        gate = protocol["behavior"]
        checks = {
            "active_accuracy": active_summary["accuracy"] >= gate["active_accuracy_min"],
            "partition": min(active_summary["partition"].values()) >= gate["partition_min"],
            "surface": min(active_summary["surface"].values()) >= gate["surface_min"],
            "family": min(active_summary["family"].values()) >= gate["family_min"],
            "truth": min(active_summary["truth"].values()) >= gate["truth_min"],
            "quartet": active_summary["quartet_all_fraction"] >= gate["quartet_all_min"],
            "status": status_summary["accuracy"] >= gate["status_accuracy_min"],
            "status_partition": min(status_summary["partition"].values()) >= gate["status_partition_min"],
            "finite": all(math.isfinite(r["margin"]) for r in active_records + status_records),
            "numeric": numeric_max <= gate["same_shape_repeat_max_abs_diff"],
            "eligible_cells": len(cells) == 48 and min((len(v) for v in cells.values()), default=0) >= per_cell,
            "selected_count": len(selected) == protocol["material"]["eligible_case_target"],
        }
        core.write_rows(OUT / "raw/active_behavior.jsonl", active_values)
        core.write_rows(OUT / "raw/status_behavior.jsonl", status_values)
        core.write_rows(OUT / "material/eligible_pairs.jsonl", selected)
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "active": active_summary, "status": status_summary,
            "numeric_same_shape_max_abs_diff": numeric_max,
            "eligible_pair_count_before_balance": len(eligible),
            "eligible_cell_min": min((len(v) for v in cells.values()), default=0),
            "selected_pair_count": len(selected), "checks": checks,
            "behavior_qualified": all(checks.values()),
            "runtime": {"placement": placement, "quantization": quant,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/qwen3_behavior_summary.json", summary)
        core.save(OUT / "analysis/final.json", {
            "phase": PHASE, "campaign": CAMPAIGN,
            "behavior_qualified": summary["behavior_qualified"],
            "authorization": "run_phase1371_c057_instrument_calibration" if summary["behavior_qualified"]
                             else "close_c057_behavior_unqualified_before_hidden_access",
        })
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
