#!/usr/bin/env python3
"""Phase1391: family-factorized Qwen3 behavior qualification for C062."""
from __future__ import annotations

import inspect, json, math, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"; sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE, CAMPAIGN = 1391, "C062"
CONTRACT = TESTS / "result/phase1390_c062_route_factorized_field_campaign_contract"
OUT = TESTS / "result/phase1391_c062_family_factorized_behavior"
MODEL, BATCH = "qwen3", 8


def parent() -> dict:
    final = core.load(CONTRACT / "analysis/final.json"); audit = core.load(CONTRACT / "audit/independent_final_audit.json")
    if final["authorization"] != "run_phase1391_c062_family_factorized_behavior" or not audit["all_checks_passed"]:
        raise RuntimeError("Phase1390 did not authorize behavior")
    return core.load(CONTRACT / "protocol/preregistration.json")


def make_batch(rows, pad, device):
    width = max(len(r["prompt_ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        v = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, width-len(v):] = v; mask[i, width-len(v):] = 1
    pos = mask.cumsum(-1)-1; pos.masked_fill_(mask == 0, 0)
    return ids, mask, pos


@torch.inference_mode()
def forward(model, rows, pad, device, supports):
    ids, mask, pos = make_batch(rows, pad, device)
    kw = {"input_ids": ids, "attention_mask": mask, "position_ids": pos,
          "use_cache": False, "return_dict": True}
    if supports: kw["logits_to_keep"] = 1
    out = model(**kw); values = []
    for i, row in enumerate(rows):
        logits = out.logits[i, -1].float(); scores = [float(logits[x[0]]) for x in row["candidate_ids"]]
        values.append({"scores": scores, "margin": scores[0]-scores[1], "prediction": int(scores[1] > scores[0])})
    return values


def attach(source, predictions):
    rows = [{**a, **b} for a, b in zip(source, predictions)]
    for r in rows: r["correct"] = r["prediction"] == r["gold_position"]
    return rows


def accuracy(rows): return sum(r["correct"] for r in rows) / len(rows)


def main() -> None:
    protocol = parent()
    if (OUT / "analysis/final.json").exists(): raise RuntimeError("Phase1391 already exists")
    active_src = core.rows(CONTRACT / "material/active_membership_cases.jsonl")
    status_src = core.rows(CONTRACT / "material/status_cases.jsonl")
    active_cmp = core.rows(CONTRACT / "compiled/qwen3_active.jsonl")
    status_cmp = core.rows(CONTRACT / "compiled/qwen3_status.jsonl")
    pairs = core.rows(CONTRACT / "material/candidate_pairs.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL); quant = quantization_audit(model)
        pad = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        supports = "logits_to_keep" in inspect.signature(model.forward).parameters
        ap, sp, first = [], [], None
        for start in range(0, len(active_cmp), BATCH):
            v = forward(model, active_cmp[start:start+BATCH], pad, device, supports)
            if start == 0: first = v
            ap.extend(v)
        repeat = forward(model, active_cmp[:BATCH], pad, device, supports)
        numeric = max(abs(a["scores"][j]-b["scores"][j]) for a,b in zip(first,repeat) for j in range(2))
        for start in range(0, len(status_cmp), BATCH):
            sp.extend(forward(model, status_cmp[start:start+BATCH], pad, device, supports))
        active, status = attach(active_src, ap), attach(status_src, sp)
        active_by = {r["case_id"]: r for r in active}; status_by = {r["case_id"]: r for r in status}
        eligible = [p for p in pairs if all(active_by[p[k]]["correct"] for k in
                    ("clean_true", "corrupt_false", "wrong_identity_true")) and status_by[p["status_true"]]["correct"]]
        gate = protocol["behavior"]; per_cell = protocol["material"]["eligible_cases_per_family_cell"]
        family_results, selected = {}, []
        for family in protocol["material"]["families"]:
            rows = [r for r in active if r["target_family"] == family]
            sr = [r for r in status if r["target_family"] == family]
            pair_cells = defaultdict(list)
            for r in rows: pair_cells[(r["quartet_key"], r["target_family"])].append(r["correct"])
            family_eligible = [r for r in eligible if r["target_family"] == family]
            cells = defaultdict(list)
            for r in family_eligible: cells[(r["partition"], r["surface"])].append(r)
            chosen = []
            if len(cells) == 9 and min(len(v) for v in cells.values()) >= per_cell:
                for key in sorted(cells): chosen.extend(sorted(cells[key], key=lambda r:r["pair_id"])[:per_cell])
            metrics = {
                "active_count": len(rows), "active_accuracy": accuracy(rows),
                "partition": {p: accuracy([r for r in rows if r["partition"] == p]) for p in protocol["material"]["partitions"]},
                "surface": {s: accuracy([r for r in rows if r["surface"] == s]) for s in protocol["material"]["surfaces"]},
                "truth": {str(v).lower(): accuracy([r for r in rows if r["truth"] == v]) for v in (True, False)},
                "pair_all_fraction": sum(all(v) for v in pair_cells.values()) / len(pair_cells),
                "status_accuracy": accuracy(sr), "eligible_count": len(family_eligible),
                "eligible_cell_min": min((len(v) for v in cells.values()), default=0), "selected_count": len(chosen),
            }
            checks = {
                "active": metrics["active_accuracy"] >= gate["family_active_accuracy_min"],
                "partition": min(metrics["partition"].values()) >= gate["family_partition_min"],
                "surface": min(metrics["surface"].values()) >= gate["family_surface_min"],
                "true": metrics["truth"]["true"] >= gate["family_true_min"],
                "false": metrics["truth"]["false"] >= gate["family_false_min"],
                "pair_all": metrics["pair_all_fraction"] >= gate["family_quartet_all_min"],
                "status": metrics["status_accuracy"] >= gate["status_accuracy_min"],
                "eligible_cells": len(cells) == 9 and metrics["eligible_cell_min"] >= per_cell,
                "selected": len(chosen) == protocol["material"]["selected_per_family"],
            }
            family_results[family] = {"metrics": metrics, "checks": checks, "qualified": all(checks.values())}
            if all(checks.values()): selected.extend(chosen)
        qualified = [f for f,v in family_results.items() if v["qualified"]]
        transfer = [f for f in qualified if f in protocol["material"]["transfer_families"]]
        novel = [f for f in qualified if f in protocol["material"]["novel_families"]]
        breadth = {
            "family_count": len(qualified) >= protocol["material"]["minimum_qualified_families"],
            "transfer_count": len(transfer) >= protocol["material"]["minimum_qualified_transfer_families"],
            "novel_count": len(novel) >= protocol["material"]["minimum_qualified_novel_families"],
            "status_global": accuracy(status) >= gate["status_accuracy_min"],
            "numeric": numeric <= gate["same_shape_repeat_max_abs_diff"],
            "finite": all(math.isfinite(r["margin"]) for r in active+status),
        }
        behavior_qualified = all(breadth.values())
        core.write_rows(OUT / "raw/active_behavior.jsonl", active); core.write_rows(OUT / "raw/status_behavior.jsonl", status)
        core.write_rows(OUT / "material/eligible_pairs.jsonl", selected)
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "family_results": family_results,
                   "qualified_families": qualified, "qualified_transfer_families": transfer,
                   "qualified_novel_families": novel, "breadth_checks": breadth,
                   "behavior_qualified": behavior_qualified, "selected_count": len(selected),
                   "selected_partition_counts": {p: sum(r["partition"]==p for r in selected)
                                                 for p in protocol["material"]["partitions"]},
                   "global": {"active_accuracy": accuracy(active), "status_accuracy": accuracy(status),
                              "numeric_same_shape_max_abs_diff": numeric},
                   "runtime": {"placement": placement, "quantization": quant,
                               "finished_at_utc": datetime.now(timezone.utc).isoformat()}}
        core.save(OUT / "analysis/qwen3_family_behavior_summary.json", summary)
        authorization = "run_phase1392_c062_full_field_camera" if behavior_qualified else "close_c062_at_factorized_behavior_breadth_gate"
        core.save(OUT / "analysis/final.json", {"phase":PHASE,"campaign":CAMPAIGN,
                  "behavior_qualified":behavior_qualified,"qualified_families":qualified,"authorization":authorization})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        if model is not None: release_bf16(model)


if __name__ == "__main__": main()
