#!/usr/bin/env python3
"""Phase1348: C050 core-membership and null-panel behavior qualification."""
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

PHASE, CAMPAIGN = 1348, "C050"
PARENT = TESTS / "result/phase1347_c050_formation_clock_contract"
OUT = TESTS / "result/phase1348_c050_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def parent():
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1348_c050_behavior" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1347 parent not authorized")
    return core.load(PARENT / "protocol/preregistration.json")


def tensors(batch, width, pad, device):
    ids = torch.full((len(batch), width), int(pad), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(batch):
        values = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, : len(values)] = values
        mask[index, : len(values)] = 1
        lengths.append(len(values))
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
        raise RuntimeError("Phase1348 manifest exists")
    sentinel = protocol["executor_gate"]["sentinel_case_ids"]
    widths, groups = {}, {}
    for model_name in MODELS:
        rows = core.rows(PARENT / f"compiled/{model_name}_cases.jsonl")
        widths[model_name] = max(len(row["prompt_ids"]) for row in rows)
        permutation = sentinel[::2] + sentinel[1::2]
        groups[model_name] = {
            "canonical": [sentinel[i : i + 4] for i in range(0, len(sentinel), 4)],
            "permuted": [permutation[i : i + 4] for i in range(0, len(permutation), 4)],
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
        "gate": protocol["behavior_gate"],
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


def run_model(model_name):
    protocol = parent()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    source = core.rows(PARENT / "material/frozen_cases.jsonl")
    compiled = core.rows(PARENT / f"compiled/{model_name}_cases.jsonl")
    compiled_by_id = {row["case_id"]: row for row in compiled}
    model = None
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        quant = quantization_audit(model)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = manifest["widths"][model_name]

        def execute(groups):
            output = {}
            for group in groups:
                output.update(
                    zip(group, score(model, device, [compiled_by_id[case_id] for case_id in group], width, pad))
                )
            return output

        canonical = execute(manifest["executor_groups"][model_name]["canonical"])
        permuted = execute(manifest["executor_groups"][model_name]["permuted"])
        repeated = execute(manifest["executor_groups"][model_name]["canonical"])
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
        rank = sum(
            (row["canonical"][0] > row["canonical"][1])
            == (row["permuted"][0] > row["permuted"][1])
            for row in executor_rows
        ) / len(executor_rows)
        max_diff = max(
            abs(left - right)
            for row in executor_rows
            for key in ("permuted", "repeated")
            for left, right in zip(row["canonical"], row[key])
        )
        executor_qualified = finite and rank >= 1.0 and max_diff <= 1e-6

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
                                    "panel",
                                    "partition",
                                    "family_pair",
                                    "pair_index",
                                    "pair_offset",
                                    "surface",
                                    "quartet_key",
                                    "cell",
                                    "target",
                                    "target_family",
                                    "tested_family",
                                    "reference",
                                    "truth",
                                )
                            },
                            "scores": candidate_scores,
                            "semantic_margin": margin,
                            "correct": (margin > 0) == source_row["truth"],
                        }
                    )

        core_rows = [row for row in records if row["panel"] == "core_membership"]
        label_rows = [row for row in records if row["panel"] == "label_only"]
        equality_rows = [row for row in records if row["panel"] == "generic_equality"]
        quartets = defaultdict(list)
        for row in core_rows:
            quartets[row["quartet_key"]].append(row)
        interactions, pairwise = [], []
        for quartet in quartets.values():
            cells = {row["cell"]: row for row in quartet}
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
        core_metrics = {
            "accuracy": sum(row["correct"] for row in core_rows) / len(core_rows),
            "partition": grouped_accuracy(core_rows, "partition", protocol["material"]["partitions"]),
            "surface": grouped_accuracy(core_rows, "surface", protocol["material"]["surfaces"]),
            "family": grouped_accuracy(core_rows, "target_family", protocol["material"]["families"]),
            "truth": grouped_accuracy(core_rows, "truth", (True, False)),
            "pairwise_true_over_false": sum(pairwise) / len(pairwise),
            "positive_interaction_fraction": sum(value > 0 for value in interactions) / len(interactions),
            "median_interaction": median(interactions),
            "case_count": len(core_rows),
            "quartet_count": len(quartets),
        }
        null_metrics = {
            "label_only_accuracy": sum(row["correct"] for row in label_rows) / len(label_rows),
            "generic_equality_accuracy": sum(row["correct"] for row in equality_rows) / len(equality_rows),
            "generic_equality_truth": grouped_accuracy(equality_rows, "truth", (True, False)),
            "label_only_case_count": len(label_rows),
            "generic_equality_case_count": len(equality_rows),
        }
        gate = manifest["gate"]
        gates = {
            "core_accuracy": core_metrics["accuracy"] >= gate["core_accuracy_min"],
            "core_partition": min(core_metrics["partition"].values()) >= gate["core_partition_min"],
            "core_surface": min(core_metrics["surface"].values()) >= gate["core_surface_min"],
            "core_family": min(core_metrics["family"].values()) >= gate["core_family_min"],
            "core_truth": min(core_metrics["truth"].values()) >= gate["core_truth_min"],
            "core_pairwise": core_metrics["pairwise_true_over_false"] >= gate["core_pairwise_min"],
            "core_interaction_direction": core_metrics["positive_interaction_fraction"]
            >= gate["core_positive_interaction_min"],
            "core_interaction_magnitude": core_metrics["median_interaction"]
            >= gate["core_median_interaction_min"],
            "label_only": null_metrics["label_only_accuracy"] >= gate["label_only_accuracy_min"],
            "generic_equality": null_metrics["generic_equality_accuracy"]
            >= gate["generic_equality_accuracy_min"],
            "generic_equality_truth": min(null_metrics["generic_equality_truth"].values())
            >= gate["generic_equality_truth_min"],
        }
        qualified = executor_qualified and all(gates.values())
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
                "rank_agreement": rank,
                "max_abs_diff": max_diff,
                "qualified": executor_qualified,
            },
            "core_metrics": core_metrics,
            "null_metrics": null_metrics,
            "gates": gates,
            "qualified": qualified,
        }
        core.save(OUT / f"analysis/{model_name}_summary.json", summary)
        print(json.dumps(summary, indent=2))
    finally:
        if model is not None:
            release_bf16(model)


def finalize():
    parent()
    summaries = {model: core.load(OUT / f"analysis/{model}_summary.json") for model in MODELS}
    qualified = [model for model in MODELS if summaries[model]["qualified"]]
    authorization = "run_phase1349_c050_formation_field" if qualified else "close_c050_behavior"
    final = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "qualified_models": qualified,
        "cross_model_behavior_repetition": len(qualified) >= 2,
        "all_gates_passed": bool(qualified),
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
