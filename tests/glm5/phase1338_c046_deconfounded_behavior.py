#!/usr/bin/env python3
"""Phase1338: C046 standard-executor and polarity-deconfounded behavior."""
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
T = ROOT / "tests/glm5"
sys.path.insert(0, str(T))
import phase1331_relational_measurement_core as core  # noqa: E402
from phase1332_bf16_utils import load_bf16, quantization_audit, release_bf16  # noqa: E402

PHASE, CAMPAIGN = 1338, "C046"
SCRIPT = Path(__file__).resolve()
AUDITOR = T / "phase1338_c046_deconfounded_behavior_audit.py"
UTIL = T / "phase1332_bf16_utils.py"
PARENT = T / "result/phase1337_c046_polarity_deconfounded_relation_contract"
OUT = T / "result/phase1338_c046_deconfounded_behavior"
MODELS = ("qwen3", "glm4", "deepseek7b")


def parent_ok():
    final = core.load(PARENT / "analysis/final.json")
    audit = core.load(PARENT / "audit/independent_final_audit.json")
    if final.get("authorization") != "run_phase1338_c046_deconfounded_behavior" or not audit.get("all_checks_passed"):
        raise RuntimeError("Phase1337 did not authorize Phase1338")
    return core.load(PARENT / "protocol/preregistration.json")


def prepare(force: bool) -> None:
    protocol = parent_ok()
    path = OUT / "protocol/execution_manifest.json"
    if path.exists() and not force:
        raise RuntimeError(f"{path} exists")
    if any((OUT / f"analysis/{model}_summary.json").exists() for model in MODELS):
        raise RuntimeError("formal results already exist")
    groups, widths = {}, {}
    sentinel = set(protocol["executor_gate"]["case_ids"])
    for model in MODELS:
        compiled = core.rows(PARENT / f"compiled/{model}_behavior.jsonl")
        widths[model] = max(len(row["prompt_ids"]) for row in compiled)
        order = [row["case_id"] for row in compiled if row["case_id"] in sentinel]
        permuted = order[::2] + order[1::2]
        groups[model] = {
            "cohort_a": [order[i:i + 8] for i in range(0, len(order), 8)],
            "cohort_permuted": [permuted[i:i + 8] for i in range(0, len(permuted), 8)],
        }
    frozen = {
        "phase": PHASE, "campaign": CAMPAIGN, "schema": "phase1338.c046.behavior.v1",
        "parent_contract_sha256": protocol["contract_sha256"],
        "parent_protocol_sha256": core.sha(PARENT / "protocol/preregistration.json"),
        "model_order": list(MODELS), "precision": "bfloat16-no-quantization",
        "batch_size": 8, "padding_side": "right", "explicit_position_ids": True,
        "width_by_model": widths, "executor_groups": groups,
        "executor_gate": protocol["executor_gate"], "behavior_gate": protocol["behavior_gate"],
        "script_sha256": core.sha(SCRIPT), "auditor_sha256": core.sha(AUDITOR), "util_sha256": core.sha(UTIL),
    }
    frozen["manifest_sha256"] = core.digest(frozen)
    frozen["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.save(path, frozen)
    print(json.dumps(frozen, ensure_ascii=False, indent=2))


def tensors(batch, width, pad_id, device):
    ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    lengths = []
    for index, row in enumerate(batch):
        seq = torch.tensor(row["prompt_ids"], dtype=torch.long, device=device)
        ids[index, :len(seq)] = seq
        mask[index, :len(seq)] = 1
        lengths.append(len(seq))
    positions = mask.cumsum(-1) - 1
    positions.masked_fill_(mask == 0, 0)
    return ids, mask, positions, lengths


@torch.inference_mode()
def score_batch(model, device, batch, width, pad_id):
    ids, mask, positions, lengths = tensors(batch, width, pad_id, device)
    output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True)
    scores = []
    for index, row in enumerate(batch):
        log_probs = torch.log_softmax(output.logits[index, lengths[index] - 1].float(), dim=-1)
        scores.append([float(log_probs[candidate[0]].item()) for candidate in row["candidate_ids"]])
    del ids, mask, positions, output
    return scores


def metric(records, key, values):
    return {str(value): sum(row["correct"] for row in records if row[key] == value)
            / sum(row[key] == value for row in records) for value in values}


def run_model(model_name: str) -> None:
    protocol = parent_ok()
    manifest = core.load(OUT / "protocol/execution_manifest.json")
    frozen = {key: value for key, value in manifest.items() if key not in {"manifest_sha256", "created_at_utc"}}
    if core.digest(frozen) != manifest["manifest_sha256"]:
        raise RuntimeError("execution manifest hash mismatch")
    summary_path = OUT / f"analysis/{model_name}_summary.json"
    if summary_path.exists():
        raise RuntimeError("formal model result already exists")
    source = core.rows(PARENT / "material/frozen_behavior_cases.jsonl")
    compiled = core.rows(PARENT / f"compiled/{model_name}_behavior.jsonl")
    if len(source) != len(compiled) or any(a["case_id"] != b["case_id"] for a, b in zip(source, compiled)):
        raise RuntimeError("compiled material mismatch")
    by_id = {row["case_id"]: row for row in compiled}
    model = None
    print(f"[Phase1338] loading {model_name}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        qa = quantization_audit(model)
        if qa.get("has_quantized_modules") or not qa.get("has_bf16_parameters"):
            raise RuntimeError(qa)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        width = int(manifest["width_by_model"][model_name])
        groups = manifest["executor_groups"][model_name]

        def grouped(group_rows):
            values = {}
            for group in group_rows:
                batch = [by_id[case_id] for case_id in group]
                values.update(zip(group, score_batch(model, device, batch, width, int(pad_id))))
            return values

        cohort = grouped(groups["cohort_a"])
        permuted = grouped(groups["cohort_permuted"])
        repeated = grouped(groups["cohort_a"])
        executor_rows = [{"case_id": case_id, "cohort_a": cohort[case_id],
                          "cohort_permuted": permuted[case_id], "cohort_a_repeat": repeated[case_id]}
                         for case_id in protocol["executor_gate"]["case_ids"]]
        all_scores = [value for row in executor_rows for key in ("cohort_a", "cohort_permuted", "cohort_a_repeat")
                      for value in row[key]]
        perm_diff = max(abs(a - b) for row in executor_rows for a, b in zip(row["cohort_a"], row["cohort_permuted"]))
        repeat_diff = max(abs(a - b) for row in executor_rows for a, b in zip(row["cohort_a"], row["cohort_a_repeat"]))
        rank = sum((row["cohort_a"][0] > row["cohort_a"][1]) ==
                   (row["cohort_permuted"][0] > row["cohort_permuted"][1]) for row in executor_rows) / len(executor_rows)
        executor_metrics = {"finite_fraction": sum(math.isfinite(value) for value in all_scores) / len(all_scores),
                            "permuted_rank_agreement": rank, "permuted_max_abs_score_diff": perm_diff,
                            "repeat_max_abs_score_diff": repeat_diff, "case_count": len(executor_rows)}
        eg = protocol["executor_gate"]
        executor_gates = {
            "finite": executor_metrics["finite_fraction"] >= eg["finite_fraction_min"],
            "rank": rank >= eg["permuted_rank_agreement_min"],
            "permuted": perm_diff <= eg["permuted_max_abs_score_diff_max"],
            "repeat": repeat_diff <= eg["repeat_max_abs_score_diff_max"],
            "count": len(executor_rows) == 48,
        }
        executor_qualified = all(executor_gates.values())
        records = []
        if executor_qualified:
            for start in range(0, len(compiled), 8):
                batch_compiled = compiled[start:start + 8]
                batch_source = source[start:start + 8]
                values = score_batch(model, device, batch_compiled, width, int(pad_id))
                for src, scores in zip(batch_source, values):
                    gold = int(src["gold_position"])
                    predicted = int(scores[1] > scores[0])
                    records.append({
                        "case_id": src["case_id"], "partition": src["partition"], "surface": src["surface"],
                        "target": src["target"], "target_family": src["target_family"],
                        "tested_family": src["tested_family"], "truth": src["truth"],
                        "codebook": src["codebook"], "semantic_key": src["semantic_key"],
                        "scores": scores, "gold_position": gold, "predicted_position": predicted,
                        "margin": scores[gold] - scores[1 - gold], "correct": predicted == gold,
                    })
        behavior_metrics, behavior_gates, behavior_qualified = {}, {}, False
        if records:
            pair_groups = defaultdict(list)
            for row in records:
                pair_groups[row["semantic_key"]].append(row["correct"])
            behavior_metrics = {
                "accuracy": sum(row["correct"] for row in records) / len(records),
                "partition": metric(records, "partition", ("discovery", "confirmation", "holdout")),
                "surface": metric(records, "surface", ("noun_class", "dictionary_relation", "category_claim")),
                "family": metric(records, "target_family", ("mammal", "gemstone", "vehicle", "vegetable")),
                "codebook": metric(records, "codebook", ("standard", "reversed")),
                "truth": metric(records, "truth", (True, False)),
                "truth_codebook": {f"{truth}:{codebook}": sum(row["correct"] for row in records
                    if row["truth"] == truth and row["codebook"] == codebook) /
                    sum(row["truth"] == truth and row["codebook"] == codebook for row in records)
                    for truth in (True, False) for codebook in ("standard", "reversed")},
                "semantic_pair_success": sum(len(values) == 2 and all(values) for values in pair_groups.values()) / len(pair_groups),
                "median_margin": median(row["margin"] for row in records),
                "case_count": len(records),
            }
            bg = protocol["behavior_gate"]
            behavior_gates = {
                "accuracy": behavior_metrics["accuracy"] >= bg["accuracy_min"],
                "partition": min(behavior_metrics["partition"].values()) >= bg["partition_min"],
                "surface": min(behavior_metrics["surface"].values()) >= bg["surface_min"],
                "family": min(behavior_metrics["family"].values()) >= bg["family_min"],
                "codebook": min(behavior_metrics["codebook"].values()) >= bg["codebook_min"],
                "truth": min(behavior_metrics["truth"].values()) >= bg["truth_min"],
                "truth_codebook": min(behavior_metrics["truth_codebook"].values()) >= bg["truth_codebook_cell_min"],
                "semantic_pairs": behavior_metrics["semantic_pair_success"] >= bg["semantic_pair_success_min"],
                "margin": behavior_metrics["median_margin"] >= bg["median_margin_min"],
            }
            behavior_qualified = all(behavior_gates.values())
        qualified = executor_qualified and behavior_qualified
        core.write_rows(OUT / f"raw/{model_name}_executor.jsonl", executor_rows)
        core.write_rows(OUT / f"raw/{model_name}_behavior.jsonl", records)
        runtime = {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "placement": placement,
                   "quantization_audit": qa, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated()
                   if torch.cuda.is_available() else 0, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        core.save(OUT / f"runtime/{model_name}.json", runtime)
        summary = {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name,
                   "executor_metrics": executor_metrics, "executor_gates": executor_gates,
                   "executor_qualified": executor_qualified, "behavior_metrics": behavior_metrics,
                   "behavior_gates": behavior_gates, "behavior_qualified": behavior_qualified,
                   "qualified": qualified,
                   "executor_raw_sha256": core.sha(OUT / f"raw/{model_name}_executor.jsonl"),
                   "behavior_raw_sha256": core.sha(OUT / f"raw/{model_name}_behavior.jsonl"),
                   "runtime_sha256": core.sha(OUT / f"runtime/{model_name}.json"),
                   "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        core.save(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        if model is not None:
            release_bf16(model)
        print(f"[Phase1338] released {model_name}", flush=True)


def finalize() -> None:
    protocol = parent_ok()
    summaries = {model: core.load(OUT / f"analysis/{model}_summary.json") for model in MODELS}
    qualified = [model for model, summary in summaries.items() if summary["qualified"]]
    passed = len(qualified) >= protocol["behavior_gate"]["minimum_authorized_models"]
    final = {"phase": PHASE, "campaign": CAMPAIGN, "qualified_models": qualified,
             "qualified_model_count": len(qualified), "all_gates_passed": passed,
             "authorization": "run_phase1339_c046_full_relation_field" if passed else "close_c046_behavior",
             "model_summary_sha256": {model: core.sha(OUT / f"analysis/{model}_summary.json") for model in MODELS},
             "manifest_sha256": core.sha(OUT / "protocol/execution_manifest.json"),
             "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--model", choices=MODELS)
    group.add_argument("--finalize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.force)
    elif args.model:
        run_model(args.model)
    else:
        finalize()
