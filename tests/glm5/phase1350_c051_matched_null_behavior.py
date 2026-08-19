#!/usr/bin/env python3
"""Phase1350: behavior qualification for the frozen C051 null library."""
from __future__ import annotations

import argparse
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

PHASE, CAMPAIGN = 1350, "C051"
PARENT = TESTS / "result/phase1349_c051_matched_null_library_contract"
OUT = TESTS / "result/phase1350_c051_matched_null_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def protocol():
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1350_c051_null_behavior" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1349 is not authorized")
    return core.load(PARENT / "protocol/preregistration.json")


def tensors(batch, width, pad, device):
    ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for i, row in enumerate(batch):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[i, : len(value)] = value
        mask[i, : len(value)] = 1
        lengths.append(len(value))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def score(model, device, batch, width, pad):
    ids, mask, positions, lengths = tensors(batch, width, pad, device)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                   use_cache=False, return_dict=True)
    values = []
    for i, row in enumerate(batch):
        lp = torch.log_softmax(output.logits[i, lengths[i] - 1].float(), -1)
        values.append([float(lp[c[0]]) for c in row["candidate_ids"]])
    del ids, mask, positions, output
    return values


def prepare():
    p = protocol()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1350 manifest already exists")
    widths, sentinels = {}, {}
    material = core.rows(PARENT / "material/frozen_cases.jsonl")
    sentinel_ids = [
        row["case_id"] for row in material
        if row["partition"] == "qualification" and row["pair_index"] == 0 and row["pair_offset"] == 0
    ][:48]
    for model in MODELS:
        rows = core.rows(PARENT / f"compiled/{model}_cases.jsonl")
        widths[model] = max(len(row["prompt_ids"]) for row in rows)
        rotated = sentinel_ids[1:] + sentinel_ids[:1]
        sentinels[model] = {
            "canonical": [sentinel_ids[i:i + 4] for i in range(0, len(sentinel_ids), 4)],
            "permuted": [rotated[i:i + 4] for i in range(0, len(rotated), 4)],
        }
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": p["contract_sha256"],
        "model_order": list(MODELS), "precision": "bfloat16-no-quantization", "batch_size": 4,
        "widths": widths, "sentinels": sentinels, "gate": p["behavior_gate"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps(manifest, indent=2))


def grouped_accuracy(rows, key):
    values = sorted({str(row[key]) for row in rows})
    return {v: sum(row["correct"] for row in rows if str(row[key]) == v)
            / sum(str(row[key]) == v for row in rows) for v in values}


def panel_summary(rows, panel, gate):
    selected = [row for row in rows if row["panel"] == panel]
    quartets = defaultdict(list)
    for row in selected:
        quartets[row["quartet_key"]].append(row)
    summary = {
        "count": len(selected),
        "accuracy": sum(row["correct"] for row in selected) / len(selected),
        "partition_accuracy": grouped_accuracy(selected, "partition"),
        "surface_accuracy": grouped_accuracy(selected, "surface"),
        "truth_accuracy": grouped_accuracy(selected, "truth"),
        "mismatch_accuracy": grouped_accuracy(selected, "mismatch_type"),
        "quartet_all_correct_fraction": sum(all(r["correct"] for r in q) for q in quartets.values()) / len(quartets),
    }
    checks = {
        "overall": summary["accuracy"] >= gate["panel_overall_accuracy_min"],
        "partitions": min(summary["partition_accuracy"].values()) >= gate["panel_partition_accuracy_min"],
        "surfaces": min(summary["surface_accuracy"].values()) >= gate["panel_surface_accuracy_min"],
        "truth": min(summary["truth_accuracy"].values()) >= gate["panel_truth_accuracy_min"],
        "quartets": summary["quartet_all_correct_fraction"] >= gate["quartet_all_correct_fraction_min"],
    }
    if panel == "role_bound_lexical":
        checks["mismatch"] = min(summary["mismatch_accuracy"].values()) >= gate["role_mismatch_accuracy_min"]
    summary["checks"] = checks
    summary["qualified"] = all(checks.values())
    return summary


def run_model(model_name):
    p, manifest = protocol(), core.load(OUT / "protocol/execution_manifest.json")
    material = core.rows(PARENT / "material/frozen_cases.jsonl")
    compiled = core.rows(PARENT / f"compiled/{model_name}_cases.jsonl")
    by_id = {row["case_id"]: row for row in compiled}
    model = None
    try:
        model, tok, device, placement = load_bf16(model_name)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        width = manifest["widths"][model_name]

        def execute(groups):
            output = {}
            for group in groups:
                output.update(zip(group, score(model, device, [by_id[x] for x in group], width, pad)))
            return output

        canonical = execute(manifest["sentinels"][model_name]["canonical"])
        permuted = execute(manifest["sentinels"][model_name]["permuted"])
        repeated = execute(manifest["sentinels"][model_name]["canonical"])
        ids = list(canonical)
        finite = all(math.isfinite(v) for cid in ids for vector in (canonical[cid], permuted[cid], repeated[cid]) for v in vector)
        rank = sum((canonical[c][0] > canonical[c][1]) == (permuted[c][0] > permuted[c][1]) for c in ids) / len(ids)
        max_diff = max(abs(a - b) for c in ids for vector in (permuted[c], repeated[c])
                       for a, b in zip(canonical[c], vector))
        executor_ok = finite and rank >= p["behavior_gate"]["executor_rank_agreement_min"] \
            and max_diff <= p["behavior_gate"]["executor_max_abs_diff_max"]

        records = []
        if executor_ok:
            for start in range(0, len(compiled), 4):
                values = score(model, device, compiled[start:start + 4], width, pad)
                for source, candidate_scores in zip(material[start:start + 4], values):
                    pred = 0 if candidate_scores[0] > candidate_scores[1] else 1
                    records.append({**source, "scores": candidate_scores,
                                    "margin": candidate_scores[0] - candidate_scores[1],
                                    "prediction": pred, "correct": pred == source["gold_position"]})
        core.write_rows(OUT / f"raw/{model_name}_behavior.jsonl", records)
        executor = {"finite": finite, "rank_agreement": rank, "max_abs_diff": max_diff,
                    "qualified": executor_ok, "placement": placement, "quantization": quant}
        core.save(OUT / f"raw/{model_name}_executor.json", executor)
        panels = {panel: panel_summary(records, panel, p["behavior_gate"])
                  for panel in p["panels"]} if records else {}
        qualified = executor_ok and bool(panels) and all(x["qualified"] for x in panels.values())
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name,
                   "executor": executor, "panels": panels, "qualified": qualified}
        core.save(OUT / f"analysis/{model_name}_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    p = protocol()
    summaries = {m: core.load(OUT / f"analysis/{m}_summary.json") for m in MODELS}
    qualified = [m for m, s in summaries.items() if s["qualified"]]
    common = all(summaries[m]["qualified"] for m in p["required_common_models"])
    authorization = "freeze_phase1351_c052_formation_contract" if common else "close_c051_null_library"
    final = {"phase": PHASE, "campaign": CAMPAIGN, "qualified_models": qualified,
             "required_common_models_passed": common, "all_gates_passed": common,
             "authorization": authorization}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        if not args.model:
            raise SystemExit("--model is required")
        run_model(args.model)
    else:
        finalize()


if __name__ == "__main__":
    main()
