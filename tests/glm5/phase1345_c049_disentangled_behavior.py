#!/usr/bin/env python3
"""Phase1345: run frozen C049 behavior with disentangled evidence ledgers."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
sys.path.insert(0, str(TESTS))
import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16

PHASE = 1345
CAMPAIGN = "C049"
PARENT = TESTS / "result/phase1344_c049_disentangled_relation_contract"
OUT = TESTS / "result/phase1345_c049_disentangled_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def parent():
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1345_c049_disentangled_behavior" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1344 parent not authorized")
    return core.load(PARENT / "protocol/preregistration.json")


def tensors(batch, width, pad, device):
    ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(batch):
        value = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, : len(value)] = value
        mask[index, : len(value)] = 1
        lengths.append(len(value))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def score(model, device, batch, width, pad):
    ids, mask, positions, lengths = tensors(batch, width, pad, device)
    output = model(
        input_ids=ids,
        attention_mask=mask,
        position_ids=positions,
        use_cache=False,
        return_dict=True,
    )
    result = []
    for index, row in enumerate(batch):
        log_prob = torch.log_softmax(output.logits[index, lengths[index] - 1].float(), -1)
        result.append([float(log_prob[candidate[0]]) for candidate in row["candidate_ids"]])
    del ids, mask, positions, output
    return result


def prepare():
    protocol = parent()
    if (OUT / "protocol/execution_manifest.json").exists():
        raise RuntimeError("Phase1345 manifest already exists")
    sentinel = protocol["executor_gate"]["sentinel_case_ids"]
    widths, groups = {}, {}
    for model in MODELS:
        rows = core.rows(PARENT / f"compiled/{model}_factorial.jsonl")
        widths[model] = max(len(row["prompt_ids"]) for row in rows)
        permuted = sentinel[::2] + sentinel[1::2]
        groups[model] = {
            "canonical": [sentinel[i : i + 4] for i in range(0, len(sentinel), 4)],
            "permuted": [permuted[i : i + 4] for i in range(0, len(permuted), 4)],
        }
    manifest = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "contract_sha256": protocol["contract_sha256"],
        "model_order": list(MODELS),
        "precision": "bfloat16-no-quantization",
        "batch_size": 4,
        "widths": widths,
        "executor_groups": groups,
        "relation_gate": protocol["behavior_ledgers"]["relation_interaction_authorization"],
        "joint_ledger": protocol["behavior_ledgers"]["quartet_joint_reliability_report"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "protocol/execution_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def grouped_accuracy(rows, key, values):
    return {
        str(value): sum(row["correct"] for row in rows if row[key] == value)
        / sum(row[key] == value for row in rows)
        for value in values
    }


def wilson_lower(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1 + z * z / total
    center = proportion + z * z / (2 * total)
    radius = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total)
    return (center - radius) / denominator


def run_model(model_name):
    protocol = parent()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    source = core.rows(PARENT / "material/frozen_factorial_cases.jsonl")
    compiled = core.rows(PARENT / f"compiled/{model_name}_factorial.jsonl")
    compiled_by_id = {row["case_id"]: row for row in compiled}
    model = None
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        quant = quantization_audit(model)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = manifest["widths"][model_name]

        def run_groups(groups):
            values = {}
            for group in groups:
                values.update(
                    zip(group, score(model, device, [compiled_by_id[case_id] for case_id in group], width, pad))
                )
            return values

        canonical = run_groups(manifest["executor_groups"][model_name]["canonical"])
        permuted = run_groups(manifest["executor_groups"][model_name]["permuted"])
        repeated = run_groups(manifest["executor_groups"][model_name]["canonical"])
        executor_rows = [
            {
                "case_id": case_id,
                "canonical": canonical[case_id],
                "permuted": permuted[case_id],
                "repeated": repeated[case_id],
            }
            for case_id in protocol["executor_gate"]["sentinel_case_ids"]
        ]
        finite = all(
            math.isfinite(value)
            for row in executor_rows
            for key in ("canonical", "permuted", "repeated")
            for value in row[key]
        )
        rank_agreement = sum(
            (row["canonical"][0] > row["canonical"][1])
            == (row["permuted"][0] > row["permuted"][1])
            for row in executor_rows
        ) / len(executor_rows)
        max_abs_diff = max(
            abs(left - right)
            for row in executor_rows
            for key in ("permuted", "repeated")
            for left, right in zip(row["canonical"], row[key])
        )
        executor_qualified = finite and rank_agreement >= 1.0 and max_abs_diff <= 1e-6

        records = []
        if executor_qualified:
            for start in range(0, len(compiled), 4):
                values = score(model, device, compiled[start : start + 4], width, pad)
                for source_row, candidate_scores in zip(source[start : start + 4], values):
                    margin = candidate_scores[0] - candidate_scores[1]
                    records.append(
                        {
                            **{
                                key: source_row[key]
                                for key in (
                                    "case_id",
                                    "partition",
                                    "family_pair",
                                    "pair_index",
                                    "pair_offset",
                                    "surface",
                                    "quartet_key",
                                    "cell",
                                    "interaction_sign",
                                    "target",
                                    "target_family",
                                    "tested_family",
                                    "truth",
                                )
                            },
                            "scores": candidate_scores,
                            "semantic_margin": margin,
                            "correct": (margin > 0) == source_row["truth"],
                        }
                    )

        quartets = defaultdict(list)
        for row in records:
            quartets[row["quartet_key"]].append(row)
        interactions, pairwise, quartet_correct = [], [], []
        for rows in quartets.values():
            cells = {row["cell"]: row for row in rows}
            interactions.append(
                cells["aa"]["semantic_margin"]
                - cells["ab"]["semantic_margin"]
                - cells["ba"]["semantic_margin"]
                + cells["bb"]["semantic_margin"]
            )
            pairwise.extend(
                [
                    cells["aa"]["semantic_margin"] > cells["ab"]["semantic_margin"],
                    cells["bb"]["semantic_margin"] > cells["ba"]["semantic_margin"],
                ]
            )
            quartet_correct.append(all(row["correct"] for row in rows))

        families = tuple(protocol["material"]["families"])
        metrics = {
            "accuracy": sum(row["correct"] for row in records) / len(records),
            "partition": grouped_accuracy(records, "partition", ("discovery", "confirmation", "holdout")),
            "surface": grouped_accuracy(records, "surface", ("ordinary", "dictionary", "claim")),
            "family": grouped_accuracy(records, "target_family", families),
            "truth": grouped_accuracy(records, "truth", (True, False)),
            "pairwise_true_over_false": sum(pairwise) / len(pairwise),
            "positive_interaction_fraction": sum(value > 0 for value in interactions) / len(interactions),
            "median_interaction": median(interactions),
            "case_count": len(records),
            "quartet_count": len(quartets),
        }
        successes = sum(quartet_correct)
        joint = {
            "quartet_all_correct": successes / len(quartet_correct),
            "quartet_successes": successes,
            "quartet_total": len(quartet_correct),
            "wilson_95_lower_bound": wilson_lower(successes, len(quartet_correct)),
        }
        gate = manifest["relation_gate"]
        relation_gates = {
            "accuracy": metrics["accuracy"] >= gate["accuracy_min"],
            "partition": min(metrics["partition"].values()) >= gate["partition_min"],
            "surface": min(metrics["surface"].values()) >= gate["surface_min"],
            "family": min(metrics["family"].values()) >= gate["family_min"],
            "truth": min(metrics["truth"].values()) >= gate["truth_min"],
            "pairwise": metrics["pairwise_true_over_false"] >= gate["pairwise_true_over_false_min"],
            "interaction_direction": metrics["positive_interaction_fraction"]
            >= gate["positive_interaction_fraction_min"],
            "interaction_magnitude": metrics["median_interaction"] >= gate["median_interaction_min"],
        }
        joint_gate = manifest["joint_ledger"]
        joint_gates = {
            "point_rate": joint["quartet_all_correct"] >= joint_gate["point_rate_target"],
            "wilson_lower": joint["wilson_95_lower_bound"] >= joint_gate["wilson_95_lower_bound_target"],
        }
        relation_qualified = executor_qualified and all(relation_gates.values())
        joint_qualified = executor_qualified and all(joint_gates.values())
        core.write_rows(OUT / f"raw/{model_name}_executor.jsonl", executor_rows)
        core.write_rows(OUT / f"raw/{model_name}_behavior.jsonl", records)
        core.save(
            OUT / f"runtime/{model_name}.json",
            {
                "placement": placement,
                "quantization_audit": quant,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        summary = {
            "model": model_name,
            "executor": {
                "finite": finite,
                "rank_agreement": rank_agreement,
                "max_abs_diff": max_abs_diff,
                "qualified": executor_qualified,
            },
            "relation_interaction_metrics": metrics,
            "relation_interaction_gates": relation_gates,
            "relation_interaction_qualified": relation_qualified,
            "quartet_joint_reliability": joint,
            "quartet_joint_gates": joint_gates,
            "quartet_joint_qualified": joint_qualified,
        }
        core.save(OUT / f"analysis/{model_name}_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    parent()
    summaries = {model: core.load(OUT / f"analysis/{model}_summary.json") for model in MODELS}
    relation_models = [model for model in MODELS if summaries[model]["relation_interaction_qualified"]]
    joint_models = [model for model in MODELS if summaries[model]["quartet_joint_qualified"]]
    authorization = "run_phase1346_c049_full_interaction_field" if relation_models else "close_c049_behavior"
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "relation_interaction_qualified_models": relation_models,
        "quartet_joint_qualified_models": joint_models,
        "cross_model_behavior_repetition": len(relation_models) >= 2,
        "all_gates_passed": bool(relation_models),
        "authorization": authorization,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run_model(args.model) if args.model else finalize()
