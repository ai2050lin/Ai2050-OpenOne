#!/usr/bin/env python3
"""C181: sequential three-model sequence-logprob qualification on C180 material."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
OUT = RESULT / "phase1715_c181_cross_model_functional_eligibility"
C180 = RESULT / "phase1714_c180_reachable_target_choice_ecology"
sys.path.insert(0, str(TESTS))

import phase1331_relational_measurement_core as core
from phase1332_bf16_utils import MODELS, load_bf16, quantization_audit, release_bf16
from model_utils import get_model_info

PHASE, CAMPAIGN = 1715, "C181"
BATCH = {"qwen3": 8, "glm4": 2, "deepseek7b": 2}


def now():
    return datetime.now(timezone.utc).isoformat()


def render_ids(tokenizer, prompt):
    messages = [{"role": "system", "content": "Use only the supplied directed links. Answer exactly A or B."}, {"role": "user", "content": prompt}]
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except (TypeError, ValueError):
        try:
            ids = tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
            ids = tokenizer.encode(text, add_special_tokens=True)
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def contract():
    if OUT.exists():
        raise RuntimeError(OUT)
    parent = core.load(C180 / "audit/independent_final_audit.json")
    cases = core.rows(C180 / "material/cases.jsonl")
    checks = {
        "authorization": parent["all_checks_passed"] and "C181" in parent["authorization"],
        "cases": len(cases) == 192,
        "models": tuple(MODELS) == ("qwen3", "glm4", "deepseek7b"),
        "candidate_balance": float(np.mean([r["gold_position"] == 0 for r in cases])) == 0.5,
        "families": len({r["family"] for r in cases}) == 8,
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    OUT.mkdir(parents=True)
    core.write_rows(OUT / "material/cases.jsonl", cases)
    protocol = {
        "phase": PHASE,
        "campaign": CAMPAIGN,
        "created_at_utc": now(),
        "status": "cross_model_sequence_logprob_contract_frozen",
        "models": list(MODELS),
        "sequential_loading": True,
        "precision": "BF16 nonquantized; one model at a time; CPU-GPU offload permitted",
        "score": "teacher-forced sum log probability of visible continuation ' A' versus ' B'",
        "cases": 192,
        "qualification": {"global_min": 0.80, "family_partition_min": 0.75},
        "common_rule": "same family qualifies in all three partitions on at least two models",
        "claim_boundary": "behavioral functional interface only; no coordinate equivalence",
        "forbidden": ["attention", "MLP", "weights", "PCA", "cross-model token-id equality", "simultaneous model loading"],
        "producer_sha256": core.sha(Path(__file__)),
        "authorization": "run_qwen3_then_glm4_then_deepseek7b",
    }
    core.save(OUT / "protocol/preregistration.json", protocol)
    core.save(OUT / "audit/internal_contract_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps({"checks": checks, "models": list(MODELS)}, indent=2))


@torch.inference_mode()
def run_model(model_name):
    if (OUT / f"analysis/{model_name}.json").exists():
        raise RuntimeError(f"already run {model_name}")
    cases = core.rows(OUT / "material/cases.jsonl")
    model = tokenizer = None
    started = time.time()
    try:
        model, tokenizer, device, placement = load_bf16(model_name)
        quant = quantization_audit(model)
        if quant["has_quantized_modules"] or not quant["has_bf16_parameters"]:
            raise RuntimeError(quant)
        info = get_model_info(model, model_name)
        candidates = [tokenizer.encode(" A", add_special_tokens=False), tokenizer.encode(" B", add_special_tokens=False)]
        if any(not x for x in candidates):
            raise RuntimeError(candidates)
        compiled = [{**row, "prompt_ids": render_ids(tokenizer, row["prompt"])} for row in cases]
        core.write_rows(OUT / f"compiled/{model_name}.jsonl", [{"case_id": r["case_id"], "prompt_length": len(r["prompt_ids"]), "candidate_lengths": [len(x) for x in candidates]} for r in compiled])
        scores = np.zeros((len(compiled), 2), np.float32)
        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in range(0, len(compiled), BATCH[model_name]):
            batch = compiled[start:start + BATCH[model_name]]
            expanded = []
            for row in batch:
                for candidate_i, candidate in enumerate(candidates):
                    expanded.append((row, candidate_i, row["prompt_ids"] + candidate, len(row["prompt_ids"]), candidate))
            width = max(len(item[2]) for item in expanded)
            ids = torch.full((len(expanded), width), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, (_row, _ci, values, _prompt_length, _candidate) in enumerate(expanded):
                tensor = torch.tensor(values, dtype=torch.long, device=device)
                ids[i, width - len(values):] = tensor
                mask[i, width - len(values):] = 1
            output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
            logp = torch.log_softmax(output.logits.float(), dim=-1)
            for i, (_row, candidate_i, values, prompt_length, candidate) in enumerate(expanded):
                left = width - len(values)
                value = 0.0
                for k, token_id in enumerate(candidate):
                    token_position = left + prompt_length + k
                    value += float(logp[i, token_position - 1, token_id])
                scores[start + i // 2, candidate_i] = value
            del ids, mask, output, logp
            if start and start % 64 == 0:
                print(f"[C181] {model_name} {start}/{len(compiled)} elapsed={time.time()-started:.1f}s", flush=True)
        predictions = np.argmax(scores, axis=1)
        results = [{"case_id": row["case_id"], "family": row["family"], "partition": row["partition"], "surface": row["surface"], "codebook": row["codebook"], "gold_position": row["gold_position"], "prediction": int(predictions[i]), "correct": bool(predictions[i] == row["gold_position"]), "score_A": float(scores[i, 0]), "score_B": float(scores[i, 1])} for i, row in enumerate(compiled)]
        core.write_rows(OUT / f"raw/{model_name}.jsonl", results)
        np.save(OUT / f"raw/{model_name}_scores.float32.npy", scores)
        q = core.load(OUT / "protocol/preregistration.json")["qualification"]
        accuracy = lambda xs: float(np.mean([r["correct"] for r in xs]))
        by_family_partition = {family: {part: accuracy([r for r in results if r["family"] == family and r["partition"] == part]) for part in ("discovery", "confirmation", "fresh")} for family in sorted({r["family"] for r in results})}
        global_accuracy = accuracy(results)
        eligible = [family for family, parts in by_family_partition.items() if min(parts.values()) >= q["family_partition_min"]]
        if global_accuracy < q["global_min"]:
            eligible = []
        report = {"phase": PHASE, "campaign": CAMPAIGN, "model": model_name, "status": "model_functional_eligibility_adjudicated", "global_accuracy": global_accuracy, "by_family_partition": by_family_partition, "eligible_families": eligible, "placement": placement, "quantization_audit": quant, "model_info": {"layers": info.n_layers, "d_model": info.d_model, "class": info.model_class}, "candidate_token_lengths": [len(x) for x in candidates], "elapsed_seconds": time.time() - started}
        core.save(OUT / f"analysis/{model_name}.json", report)
        checks = {"rows": len(results) == 192, "finite": bool(np.isfinite(scores).all()), "bf16": quant["has_bf16_parameters"], "unquantized": not quant["has_quantized_modules"]}
        core.save(OUT / f"audit/internal_{model_name}_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
        print(json.dumps(report, indent=2))
    finally:
        if model is not None:
            release_bf16(model)
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()


def analyze():
    reports = {model: core.load(OUT / f"analysis/{model}.json") for model in MODELS}
    families = sorted({family for report in reports.values() for family in report["by_family_partition"]})
    common = {family: [model for model in MODELS if family in reports[model]["eligible_families"]] for family in families}
    common_two = [family for family, models in common.items() if len(models) >= 2]
    common_three = [family for family, models in common.items() if len(models) == 3]
    report = {"phase": PHASE, "campaign": CAMPAIGN, "created_at_utc": now(), "status": "cross_model_functional_eligibility_adjudicated", "models": reports, "common_family_models": common, "common_two_model_families": common_two, "common_three_model_families": common_three, "cross_model_hidden_eligible": len(common_two) >= 4, "claim_boundary": "behavioral functional alignment only", "next_authorization": "run_C182_cross_model_relative_hidden_topology" if len(common_two) >= 4 else "C182_typed_not_tested_then_C183_synthesis_heatmap"}
    core.save(OUT / "analysis/summary.json", report)
    checks = {"models": set(reports) == set(MODELS), "families": len(families) == 8, "typed": isinstance(report["cross_model_hidden_eligible"], bool)}
    core.save(OUT / "audit/internal_analysis_audit.json", {"checks": checks, "all_checks_passed": all(checks.values())})
    print(json.dumps(report, indent=2))


def close():
    report = core.load(OUT / "analysis/summary.json")
    checks = {"contract": core.load(OUT / "audit/internal_contract_audit.json")["all_checks_passed"], "models": all(core.load(OUT / f"audit/internal_{m}_audit.json")["all_checks_passed"] for m in MODELS), "analysis": core.load(OUT / "audit/internal_analysis_audit.json")["all_checks_passed"]}
    final = {"phase": PHASE, "campaign": CAMPAIGN, "status": "closed", "checks": checks, "all_checks_passed": all(checks.values()), "headline": {"model_accuracy": {m: report["models"][m]["global_accuracy"] for m in MODELS}, "common_two_model_families": report["common_two_model_families"], "common_three_model_families": report["common_three_model_families"], "hidden_eligible": report["cross_model_hidden_eligible"]}, "next_authorization": report["next_authorization"]}
    core.save(OUT / "analysis/final.json", final)
    print(json.dumps(final, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("contract", "run", "analyze", "close"))
    parser.add_argument("--model", choices=MODELS)
    args = parser.parse_args()
    if args.command == "contract": contract()
    elif args.command == "run":
        if not args.model: raise SystemExit("--model required")
        run_model(args.model)
    elif args.command == "analyze": analyze()
    else: close()


if __name__ == "__main__":
    main()
