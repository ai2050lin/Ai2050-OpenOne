#!/usr/bin/env python3
"""Phase1352: Qwen-only behavior qualification for the frozen C052 contract."""
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

PHASE, CAMPAIGN = 1352, "C052"
PARENT = TESTS / "result/phase1351_c052_qwen_pair_probe_contract"
OUT = TESTS / "result/phase1352_c052_qwen_pair_probe_behavior"
MODEL = "qwen3"


def protocol():
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    expected = "run_phase1352_c052_qwen_behavior"
    if final.get("authorization") != expected or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1351 is not authorized")
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
        raise RuntimeError("Phase1352 manifest already exists")
    compiled = core.rows(PARENT / "compiled/qwen3_cases.jsonl")
    sentinel_ids = [row["case_id"] for row in compiled[:48]]
    rotated = sentinel_ids[1:] + sentinel_ids[:1]
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": p["contract_sha256"],
        "model": MODEL,
        "precision": "bfloat16-no-quantization",
        "batch_size": 4,
        "width": max(len(row["prompt_ids"]) for row in compiled),
        "sentinels": {
            "canonical": [sentinel_ids[i:i + 4] for i in range(0, len(sentinel_ids), 4)],
            "permuted": [rotated[i:i + 4] for i in range(0, len(rotated), 4)],
        },
        "gate": p["behavior_gate"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps(manifest, indent=2))


def grouped_accuracy(rows, key):
    values = sorted({str(row[key]) for row in rows})
    return {
        value: sum(row["correct"] for row in rows if str(row[key]) == value)
        / sum(str(row[key]) == value for row in rows)
        for value in values
    }


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
        "quartet_all_correct_fraction": (
            sum(len(q) == 4 and all(r["correct"] for r in q) for q in quartets.values())
            / len(quartets)
        ),
    }
    if panel == "core_membership":
        summary["family_accuracy"] = grouped_accuracy(selected, "target_family")
        checks = {
            "overall": summary["accuracy"] >= gate["core_accuracy_min"],
            "partitions": min(summary["partition_accuracy"].values()) >= gate["core_partition_min"],
            "surfaces": min(summary["surface_accuracy"].values()) >= gate["core_surface_min"],
            "families": min(summary["family_accuracy"].values()) >= gate["core_family_min"],
            "truth": min(summary["truth_accuracy"].values()) >= gate["core_truth_min"],
            "quartets": summary["quartet_all_correct_fraction"] >= gate["core_quartet_all_min"],
        }
    else:
        checks = {
            "overall": summary["accuracy"] >= gate["control_accuracy_min"],
            "partitions": min(summary["partition_accuracy"].values()) >= gate["control_partition_min"],
            "surfaces": min(summary["surface_accuracy"].values()) >= gate["control_surface_min"],
            "truth": min(summary["truth_accuracy"].values()) >= gate["control_truth_min"],
            "quartets": summary["quartet_all_correct_fraction"] >= gate["control_quartet_all_min"],
        }
    summary["checks"] = checks
    summary["qualified"] = all(checks.values())
    return summary


def run():
    p = protocol()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    material = core.rows(PARENT / "material/frozen_cases.jsonl")
    compiled = core.rows(PARENT / "compiled/qwen3_cases.jsonl")
    if [x["case_id"] for x in material] != [x["case_id"] for x in compiled]:
        raise RuntimeError("material/compiled ordering mismatch")
    by_id = {row["case_id"]: row for row in compiled}
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def execute(groups):
            output = {}
            for group in groups:
                output.update(zip(group, score(model, device, [by_id[x] for x in group],
                                               manifest["width"], pad)))
            return output

        canonical = execute(manifest["sentinels"]["canonical"])
        permuted = execute(manifest["sentinels"]["permuted"])
        repeated = execute(manifest["sentinels"]["canonical"])
        ids = list(canonical)
        finite = all(
            math.isfinite(v)
            for cid in ids
            for vector in (canonical[cid], permuted[cid], repeated[cid])
            for v in vector
        )
        rank = sum(
            (canonical[c][0] > canonical[c][1]) == (permuted[c][0] > permuted[c][1])
            for c in ids
        ) / len(ids)
        max_diff = max(
            abs(a - b)
            for c in ids
            for vector in (permuted[c], repeated[c])
            for a, b in zip(canonical[c], vector)
        )
        executor_ok = finite and rank == 1.0 and max_diff <= p["behavior_gate"]["executor_max_abs_diff_max"]

        records = []
        if executor_ok:
            for start in range(0, len(compiled), manifest["batch_size"]):
                batch = compiled[start:start + manifest["batch_size"]]
                values = score(model, device, batch, manifest["width"], pad)
                for source, candidate_scores in zip(material[start:start + len(batch)], values):
                    pred = 0 if candidate_scores[0] > candidate_scores[1] else 1
                    records.append({
                        **source,
                        "scores": candidate_scores,
                        "margin": candidate_scores[0] - candidate_scores[1],
                        "prediction": pred,
                        "correct": pred == source["gold_position"],
                    })
        core.write_rows(OUT / "raw/qwen3_behavior.jsonl", records)
        executor = {
            "finite": finite,
            "rank_agreement": rank,
            "max_abs_diff": max_diff,
            "qualified": executor_ok,
            "placement": placement,
            "quantization": quant,
        }
        core.save(OUT / "raw/qwen3_executor.json", executor)
        panels = {
            panel: panel_summary(records, panel, p["behavior_gate"])
            for panel in p["material"]["panels"]
        } if records else {}
        qualified = executor_ok and bool(panels) and all(x["qualified"] for x in panels.values())
        summary = {
            "phase": PHASE,
            "campaign": CAMPAIGN,
            "model": MODEL,
            "executor": executor,
            "panels": panels,
            "qualified": qualified,
        }
        core.save(OUT / "analysis/qwen3_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    summary = core.load(OUT / "analysis/qwen3_summary.json")
    passed = bool(summary["qualified"])
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "model": MODEL,
        "behavior_gate_passed": passed,
        "all_gates_passed": passed,
        "authorization": "run_phase1353_c052_qwen_full_probe" if passed else "close_c052_behavior",
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "finalize"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        finalize()


if __name__ == "__main__":
    main()
