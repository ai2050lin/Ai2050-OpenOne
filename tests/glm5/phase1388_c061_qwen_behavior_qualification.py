#!/usr/bin/env python3
"""Phase1388: Qwen3 behavior qualification for C061."""
from __future__ import annotations

import inspect, json, math, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1388, "C061"
CONTRACT = TESTS / "result/phase1387_c061_full_field_transfer_campaign_contract"
OUT = TESTS / "result/phase1388_c061_qwen_behavior_qualification"
MODEL, BATCH = "qwen3", 8


def parent() -> dict:
    final = core.load(CONTRACT / "analysis/final.json")
    audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1388_c061_behavior_qualification" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1387 did not authorize behavior")
    return core.load(CONTRACT / "protocol/preregistration.json")


def make_batch(rows: list[dict], pad: int, device: torch.device):
    width = max(len(r["prompt_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, width - len(value):] = value
        mask[i, width - len(value):] = 1
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
    values = []
    for i, row in enumerate(rows):
        logits = output.logits[i, -1].float()
        scores = [float(logits[x[0]]) for x in row["candidate_ids"]]
        values.append({"scores": scores, "margin": scores[0] - scores[1],
                       "prediction": int(scores[1] > scores[0])})
    return values


def summarize(source: list[dict], predicted: list[dict], fields: list[str]):
    rows = [{**a, **b} for a, b in zip(source, predicted)]
    for row in rows:
        row["correct"] = row["prediction"] == row["gold_position"]
    result = {"count": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows)}
    for field in fields:
        result[field] = {name: sum(r["correct"] for r in rows if r[field] == name) /
                               sum(r[field] == name for r in rows)
                         for name in sorted({r[field] for r in rows})}
    return result, rows


def main() -> None:
    protocol = parent()
    if (OUT / "analysis/final.json").exists():
        raise RuntimeError("Phase1388 already exists")
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
        active_pred, status_pred, first = [], [], None
        for start in range(0, len(active_compiled), BATCH):
            values = forward(model, active_compiled[start:start + BATCH], pad, device, supports)
            if start == 0:
                first = values
            active_pred.extend(values)
        repeat = forward(model, active_compiled[:BATCH], pad, device, supports)
        numeric_max = max(abs(a["scores"][j] - b["scores"][j])
                          for a, b in zip(first, repeat) for j in range(2))
        for start in range(0, len(status_compiled), BATCH):
            status_pred.extend(forward(model, status_compiled[start:start + BATCH], pad, device, supports))
        active, active_rows = summarize(active_source, active_pred,
                                        ["partition", "surface", "target_family", "panel", "truth"])
        status, status_rows = summarize(status_source, status_pred, ["partition", "panel"])
        quartets = defaultdict(list)
        for row in active_rows:
            quartets[row["quartet_key"]].append(row["correct"])
        active["quartet_all_fraction"] = sum(all(v) for v in quartets.values()) / len(quartets)
        active_by = {r["case_id"]: r for r in active_rows}
        status_by = {r["case_id"]: r for r in status_rows}
        eligible = [p for p in pairs
                    if all(active_by[p[k]]["correct"] for k in ("clean_true", "corrupt_false", "wrong_identity_true"))
                    and status_by[p["status_true"]]["correct"]]
        cells: dict[tuple, list[dict]] = defaultdict(list)
        for row in eligible:
            cells[(row["target_family"], row["partition"], row["surface"])].append(row)
        per_cell = int(protocol["material"]["eligible_cases_per_cell"])
        selected = []
        if len(cells) == 72 and min(len(v) for v in cells.values()) >= per_cell:
            for key in sorted(cells):
                selected.extend(sorted(cells[key], key=lambda r: r["pair_id"])[:per_cell])
        split_target = int(protocol["material"]["discovery_target"])
        gate = protocol["behavior"]
        checks = {
            "active": active["accuracy"] >= gate["active_accuracy_min"],
            "partition": min(active["partition"].values()) >= gate["partition_min"],
            "surface": min(active["surface"].values()) >= gate["surface_min"],
            "family": min(active["target_family"].values()) >= gate["family_min"],
            "panel": min(active["panel"].values()) >= gate["panel_min"],
            "truth": min(active["truth"].values()) >= gate["truth_min"],
            "quartet": active["quartet_all_fraction"] >= gate["quartet_all_min"],
            "status": status["accuracy"] >= gate["status_accuracy_min"],
            "status_partition": min(status["partition"].values()) >= gate["status_partition_min"],
            "finite": all(math.isfinite(r["margin"]) for r in active_rows + status_rows),
            "numeric": numeric_max <= gate["same_shape_repeat_max_abs_diff"],
            "eligible_cells": len(cells) == 72 and min((len(v) for v in cells.values()), default=0) >= per_cell,
            "selected": len(selected) == protocol["material"]["eligible_case_target"],
            "partition_selected": all(sum(r["partition"] == p for r in selected) == split_target
                                      for p in protocol["material"]["partitions"]),
            "panel_selected": all(sum(r["panel"] == panel for r in selected) == 144
                                  for panel in ("transfer", "novel")),
        }
        core.write_rows(OUT / "raw/active_behavior.jsonl", active_rows)
        core.write_rows(OUT / "raw/status_behavior.jsonl", status_rows)
        core.write_rows(OUT / "material/eligible_pairs.jsonl", selected)
        summary = {
            "phase": PHASE, "campaign": CAMPAIGN, "active": active, "status": status,
            "numeric_same_shape_max_abs_diff": numeric_max,
            "eligible_pair_count_before_balance": len(eligible),
            "eligible_cell_min": min((len(v) for v in cells.values()), default=0),
            "selected_pair_count": len(selected), "selected_panel_counts": dict(
                (p, sum(r["panel"] == p for r in selected)) for p in ("transfer", "novel")),
            "checks": checks, "behavior_qualified": all(checks.values()),
            "runtime": {"placement": placement, "quantization": quant,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat()},
        }
        core.save(OUT / "analysis/qwen3_behavior_summary.json", summary)
        authorization = ("run_phase1389_c061_full_field_camera"
                         if summary["behavior_qualified"] else "close_c061_behavior_unqualified_before_hidden_access")
        core.save(OUT / "analysis/final.json", {"phase": PHASE, "campaign": CAMPAIGN,
                                                 "behavior_qualified": summary["behavior_qualified"],
                                                 "authorization": authorization})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


if __name__ == "__main__":
    main()
