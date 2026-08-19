#!/usr/bin/env python3
"""Phase1354: run all frozen C053 behavior routes in one Qwen load."""
from __future__ import annotations

import argparse
import json
import math
import statistics
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

PHASE, CAMPAIGN = 1354, "C053"
PARENT = TESTS / "result/phase1353_c053_route_portfolio_contract"
OUT = TESTS / "result/phase1354_c053_behavior_route_competition"
MODEL = "qwen3"
SOURCES = {
    "B1_binary": "b1_binary_cases.jsonl",
    "B3_choice": "b3_choice_cases.jsonl",
    "N_status": "status_null_cases.jsonl",
}


def protocol():
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1354_c053_behavior_routes" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1353 is not authorized")
    return core.load(PARENT / "protocol/preregistration.json")


def tensors(batch, width, pad, device):
    ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(batch):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, :len(value)] = value
        mask[index, :len(value)] = 1
        lengths.append(len(value))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def score(model, device, batch, width, pad):
    ids, mask, positions, lengths = tensors(batch, width, pad, device)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                   use_cache=False, return_dict=True)
    result = []
    for index, row in enumerate(batch):
        lp = torch.log_softmax(output.logits[index, lengths[index] - 1].float(), -1)
        result.append([float(lp[c[0]]) for c in row["candidate_ids"]])
    del ids, mask, positions, output
    return result


def prepare():
    p = protocol()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists():
        raise RuntimeError("Phase1354 manifest already exists")
    widths, sentinels = {}, {}
    for route in SOURCES:
        rows = core.rows(PARENT / f"compiled/qwen3_{route}.jsonl")
        widths[route] = max(len(row["prompt_ids"]) for row in rows)
        ids = [row["case_id"] for row in rows[:48]]
        rotated = ids[1:] + ids[:1]
        sentinels[route] = {
            "canonical": [ids[i:i + 4] for i in range(0, len(ids), 4)],
            "permuted": [rotated[i:i + 4] for i in range(0, len(rotated), 4)],
        }
    manifest = {
        "phase": PHASE, "campaign": CAMPAIGN, "contract_sha256": p["contract_sha256"],
        "model": MODEL, "precision": "bfloat16-no-quantization", "batch_size": 4,
        "route_order": list(SOURCES), "widths": widths, "sentinels": sentinels,
        "routes": p["routes"], "status_gate": p["status_gate"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(path, manifest)
    print(json.dumps(manifest, indent=2))


def grouped_accuracy(rows, key):
    return {value: sum(r["correct"] for r in rows if str(r[key]) == value)
            / sum(str(r[key]) == value for r in rows)
            for value in sorted({str(r[key]) for r in rows})}


def absolute_summary(rows, gate):
    quartets = defaultdict(list)
    for row in rows:
        quartets[row["quartet_key"]].append(row)
    result = {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "partition_accuracy": grouped_accuracy(rows, "partition"),
        "surface_accuracy": grouped_accuracy(rows, "surface"),
        "family_accuracy": grouped_accuracy(rows, "target_family"),
        "truth_accuracy": grouped_accuracy(rows, "truth"),
        "quartet_all_correct_fraction": sum(len(q) == 4 and all(x["correct"] for x in q)
                                             for q in quartets.values()) / len(quartets),
    }
    result["checks"] = {
        "accuracy": result["accuracy"] >= gate["accuracy_min"],
        "partitions": min(result["partition_accuracy"].values()) >= gate["partition_min"],
        "surfaces": min(result["surface_accuracy"].values()) >= gate["surface_min"],
        "families": min(result["family_accuracy"].values()) >= gate["family_min"],
        "truth": min(result["truth_accuracy"].values()) >= gate["truth_min"],
        "quartets": result["quartet_all_correct_fraction"] >= gate["quartet_all_min"],
    }
    result["qualified"] = all(result["checks"].values())
    return result


def relative_summary(rows, gate):
    quartets = defaultdict(dict)
    for row in rows:
        quartets[row["quartet_key"]][row["cell"]] = row
    values = []
    for key, q in quartets.items():
        margins = {cell: q[cell]["margin"] for cell in ("aa", "ab", "ba", "bb")}
        values.append({
            "quartet_key": key, "partition": q["aa"]["partition"], "surface": q["aa"]["surface"],
            "win_a": margins["aa"] > margins["ab"],
            "win_b": margins["bb"] > margins["ba"],
            "interaction": margins["aa"] - margins["ab"] - margins["ba"] + margins["bb"],
        })
    pairwise = [x for row in values for x in (row["win_a"], row["win_b"])]
    by_partition = {}
    for part in sorted({x["partition"] for x in values}):
        subset = [x for x in values if x["partition"] == part]
        by_partition[part] = sum(v for x in subset for v in (x["win_a"], x["win_b"])) / (2 * len(subset))
    by_surface = {}
    for surface in sorted({x["surface"] for x in values}):
        subset = [x for x in values if x["surface"] == surface]
        by_surface[surface] = sum(v for x in subset for v in (x["win_a"], x["win_b"])) / (2 * len(subset))
    result = {
        "quartet_count": len(values),
        "pairwise_win_fraction": sum(pairwise) / len(pairwise),
        "partition_pairwise_win": by_partition,
        "surface_pairwise_win": by_surface,
        "positive_interaction_fraction": sum(x["interaction"] > 0 for x in values) / len(values),
        "median_interaction": statistics.median(x["interaction"] for x in values),
    }
    result["checks"] = {
        "pairwise": result["pairwise_win_fraction"] >= gate["pairwise_win_min"],
        "partitions": min(by_partition.values()) >= gate["partition_pairwise_min"],
        "surfaces": min(by_surface.values()) >= gate["surface_pairwise_min"],
        "positive_interaction": result["positive_interaction_fraction"] >= gate["positive_interaction_min"],
        "median_interaction": result["median_interaction"] >= gate["median_interaction_min"],
    }
    result["qualified"] = all(result["checks"].values())
    return result


def choice_summary(rows, gate):
    groups = defaultdict(list)
    for row in rows:
        groups[row["choice_group"]].append(row)
    result = {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "partition_accuracy": grouped_accuracy(rows, "partition"),
        "surface_accuracy": grouped_accuracy(rows, "surface"),
        "family_accuracy": grouped_accuracy(rows, "target_family"),
        "position_accuracy": grouped_accuracy(rows, "gold_position"),
        "choice_group_all_correct_fraction": sum(len(g) == 2 and all(x["correct"] for x in g)
                                                  for g in groups.values()) / len(groups),
    }
    result["checks"] = {
        "accuracy": result["accuracy"] >= gate["accuracy_min"],
        "partitions": min(result["partition_accuracy"].values()) >= gate["partition_min"],
        "surfaces": min(result["surface_accuracy"].values()) >= gate["surface_min"],
        "families": min(result["family_accuracy"].values()) >= gate["family_min"],
        "positions": min(result["position_accuracy"].values()) >= gate["position_min"],
        "groups": result["choice_group_all_correct_fraction"] >= gate["choice_group_all_min"],
    }
    result["qualified"] = all(result["checks"].values())
    return result


def status_summary(rows, gate):
    quartets = defaultdict(list)
    for row in rows:
        quartets[row["quartet_key"]].append(row)
    result = {
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "partition_accuracy": grouped_accuracy(rows, "partition"),
        "truth_accuracy": grouped_accuracy(rows, "truth"),
        "quartet_all_correct_fraction": sum(len(q) == 4 and all(x["correct"] for x in q)
                                             for q in quartets.values()) / len(quartets),
    }
    result["checks"] = {
        "accuracy": result["accuracy"] >= gate["accuracy_min"],
        "partitions": min(result["partition_accuracy"].values()) >= gate["partition_min"],
        "truth": min(result["truth_accuracy"].values()) >= gate["truth_min"],
        "quartets": result["quartet_all_correct_fraction"] >= gate["quartet_all_min"],
    }
    result["qualified"] = all(result["checks"].values())
    return result


def run():
    p = protocol()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    route_data = {}
    compiled_data = {}
    for route, source_name in SOURCES.items():
        route_data[route] = core.rows(PARENT / f"material/{source_name}")
        compiled_data[route] = core.rows(PARENT / f"compiled/qwen3_{route}.jsonl")
    model = None
    try:
        model, tok, device, placement = load_bf16(MODEL)
        quant = quantization_audit(model)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        executor_routes = {}
        for route in SOURCES:
            compiled = compiled_data[route]
            by_id = {x["case_id"]: x for x in compiled}
            def execute(groups):
                output = {}
                for group in groups:
                    output.update(zip(group, score(model, device, [by_id[x] for x in group],
                                                   manifest["widths"][route], pad)))
                return output
            canonical = execute(manifest["sentinels"][route]["canonical"])
            permuted = execute(manifest["sentinels"][route]["permuted"])
            repeated = execute(manifest["sentinels"][route]["canonical"])
            ids = list(canonical)
            finite = all(math.isfinite(v) for cid in ids for vector in
                         (canonical[cid], permuted[cid], repeated[cid]) for v in vector)
            rank = sum((canonical[c][0] > canonical[c][1]) == (permuted[c][0] > permuted[c][1])
                       for c in ids) / len(ids)
            diff = max(abs(a - b) for c in ids for vector in (permuted[c], repeated[c])
                       for a, b in zip(canonical[c], vector))
            ok = finite and rank == 1.0 and diff <= p["executor_gate"]["max_abs_diff_max"]
            executor_routes[route] = {"finite": finite, "rank_agreement": rank,
                                      "max_abs_diff": diff, "qualified": ok}
            records = []
            if ok:
                for start in range(0, len(compiled), manifest["batch_size"]):
                    batch = compiled[start:start + manifest["batch_size"]]
                    values = score(model, device, batch, manifest["widths"][route], pad)
                    for source, scores in zip(route_data[route][start:start + len(batch)], values):
                        pred = 0 if scores[0] > scores[1] else 1
                        records.append({**source, "scores": scores, "margin": scores[0] - scores[1],
                                        "prediction": pred, "correct": pred == source["gold_position"]})
            core.write_rows(OUT / f"raw/{route}_behavior.jsonl", records)
        core.save(OUT / "raw/qwen3_executor.json", {
            "routes": executor_routes, "qualified": all(x["qualified"] for x in executor_routes.values()),
            "placement": placement, "quantization": quant,
        })
        b1_rows = core.rows(OUT / "raw/B1_binary_behavior.jsonl")
        b3_rows = core.rows(OUT / "raw/B3_choice_behavior.jsonl")
        status_rows = core.rows(OUT / "raw/N_status_behavior.jsonl")
        status = status_summary(status_rows, p["status_gate"])
        summaries = {
            "B1_absolute": absolute_summary(b1_rows, p["routes"]["B1_absolute"]),
            "B2_relative": relative_summary(b1_rows, p["routes"]["B2_relative"]),
            "B3_choice": choice_summary(b3_rows, p["routes"]["B3_choice"]),
            "N_status": status,
        }
        executor_ok = all(x["qualified"] for x in executor_routes.values())
        route_qualified = {
            "B1_absolute": executor_ok and status["qualified"] and summaries["B1_absolute"]["qualified"],
            "B2_relative": executor_ok and status["qualified"] and summaries["B2_relative"]["qualified"],
            "B3_choice": executor_ok and summaries["B3_choice"]["qualified"],
        }
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "model": MODEL,
                   "executor": core.load(OUT / "raw/qwen3_executor.json"), "summaries": summaries,
                   "route_qualified": route_qualified}
        core.save(OUT / "analysis/qwen3_summary.json", summary)
        compact = {"route_qualified": route_qualified, "summaries": summaries,
                   "executor_qualified": executor_ok}
        print(json.dumps(compact, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    summary = core.load(OUT / "analysis/qwen3_summary.json")
    qualified = [route for route, passed in summary["route_qualified"].items() if passed]
    fields = []
    if "B2_relative" in qualified:
        fields.append("quartet_interaction_field")
    if "B3_choice" in qualified:
        fields.append("choice_order_invariance_field")
    authorization = "run_phase1355_c053_fields" if fields else "close_c053_after_behavior_routes"
    final = {"phase": PHASE, "campaign": CAMPAIGN, "qualified_routes": qualified,
             "authorized_fields": fields, "any_route_passed": bool(qualified),
             "all_behavior_routes_failed": not qualified, "authorization": authorization,
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
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
