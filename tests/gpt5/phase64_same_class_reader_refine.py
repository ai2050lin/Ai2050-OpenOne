from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from model_registry import get_model_spec  # noqa: E402


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def safe_mean(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def safe_std(xs: list[float]) -> float:
    return float(pstdev(xs)) if len(xs) > 1 else 0.0


def load_model(model_name: str, attn_impls: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model_spec(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.local_dir, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    errors: list[str] = []
    for impl in [x.strip() for x in attn_impls.split(",") if x.strip()]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=impl,
            )
            log(f"Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as exc:
            errors.append(f"{impl}: {exc}")
            log(f"Failed with {impl}: {exc}")
    if model is None:
        raise RuntimeError("failed to load model: " + " | ".join(errors))
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def sequence_logprob(model, tokenizer, device, prompt: str, completion: str, max_length: int) -> dict[str, Any]:
    prompt_ids_cpu = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    comp_ids_cpu = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids
    comp_token_ids = [int(x) for x in comp_ids_cpu[0].tolist()]
    input_ids_cpu = torch.cat([prompt_ids_cpu, comp_ids_cpu], dim=1)
    if input_ids_cpu.shape[1] > max_length:
        overflow = input_ids_cpu.shape[1] - max_length
        input_ids_cpu = input_ids_cpu[:, overflow:]
        prompt_len = max(1, prompt_ids_cpu.shape[1] - overflow)
    else:
        prompt_len = prompt_ids_cpu.shape[1]
    input_ids = input_ids_cpu.to(device)
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    start = prompt_len - 1
    vals: list[float] = []
    pieces: list[str] = []
    for i, tok in enumerate(comp_token_ids):
        pos = start + i
        if pos < 0 or pos >= log_probs.shape[1]:
            continue
        val = float(log_probs[0, pos, tok].detach().cpu())
        vals.append(val)
        pieces.append(tokenizer.decode([tok]))
    return {
        "logprob": float(sum(vals)),
        "mean_logprob": float(sum(vals) / max(1, len(vals))),
        "num_tokens": len(vals),
        "pieces": pieces,
        "finite": all(torch.isfinite(torch.tensor(vals)).tolist()) if vals else False,
    }


CATEGORIES = [
    ("K0", "K1"),
    ("G7", "G9"),
    ("MIP", "TEV"),
    ("ALPHA", "BETA"),
    ("TYPE17", "TYPE42"),
    ("NORTH", "SOUTH"),
    ("RED", "BLUE"),
    ("P3", "Q8"),
]

CONTEXTS = [
    (
        "key_value",
        "A_CAT={cat_a}\nB_CAT={cat_b}\nC_CAT={cat_c}\nRule: same means equal CAT value.",
    ),
    (
        "csv",
        "object,cat\nA,{cat_a}\nB,{cat_b}\nC,{cat_c}\nRule: choose by exact cat equality.",
    ),
    (
        "json",
        '{{"A_CAT":"{cat_a}","B_CAT":"{cat_b}","C_CAT":"{cat_c}"}}\nRule: compare CAT strings exactly.',
    ),
    (
        "equation",
        "cat(A)={cat_a}; cat(B)={cat_b}; cat(C)={cat_c}.\nRule: equal cat values are same class.",
    ),
]


def build_cases(max_cases: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for context_id, template in CONTEXTS:
        for cat_same, cat_diff in CATEGORIES:
            for variant, answer in [("B_same", "B"), ("C_same", "C")]:
                if answer == "B":
                    cat_a, cat_b, cat_c = cat_same, cat_same, cat_diff
                else:
                    cat_a, cat_b, cat_c = cat_same, cat_diff, cat_same
                context = template.format(cat_a=cat_a, cat_b=cat_b, cat_c=cat_c)
                cases.append(
                    {
                        "case_id": f"{context_id}:{cat_same}:{cat_diff}:{variant}",
                        "context_id": context_id,
                        "cat_same": cat_same,
                        "cat_diff": cat_diff,
                        "variant": variant,
                        "answer": answer,
                        "wrong": "C" if answer == "B" else "B",
                        "context": context,
                    }
                )
                if len(cases) >= max_cases:
                    return cases
    base = list(cases)
    cycle = 0
    while len(cases) < max_cases:
        for item in base:
            clone = dict(item)
            clone["case_id"] = f"{item['case_id']}:rep{cycle}"
            clone["context"] = item["context"] + f"\nID={cycle}."
            cases.append(clone)
            if len(cases) >= max_cases:
                return cases
        cycle += 1
    return cases


def reader_items(case: dict[str, Any]) -> list[dict[str, Any]]:
    ans = case["answer"]
    wrong = case["wrong"]
    context = case["context"]
    same_b = ans == "B"
    same_c = ans == "C"
    return [
        {
            "reader_id": "same_key_letter",
            "target_type": "same_choice",
            "prompt": context + "\nSAME_AS_A is B or C.\nSAME_AS_A=",
            "correct": ans,
            "wrong": wrong,
        },
        {
            "reader_id": "same_key_space",
            "target_type": "same_choice",
            "prompt": context + "\nSAME_AS_A is B or C.\nSAME_AS_A =",
            "correct": f" {ans}",
            "wrong": f" {wrong}",
        },
        {
            "reader_id": "same_json_min",
            "target_type": "same_choice",
            "prompt": context + '\nReturn only JSON.\n{"same":"',
            "correct": ans,
            "wrong": wrong,
        },
        {
            "reader_id": "same_option_line",
            "target_type": "same_choice",
            "prompt": context + "\nOptions: B, C.\nQuestion: equal CAT to A?\nAnswer:",
            "correct": f" {ans}",
            "wrong": f" {wrong}",
        },
        {
            "reader_id": "different_key_letter",
            "target_type": "different_choice",
            "prompt": context + "\nDIFFERENT_FROM_A is B or C.\nDIFFERENT_FROM_A=",
            "correct": wrong,
            "wrong": ans,
        },
        {
            "reader_id": "different_key_space",
            "target_type": "different_choice",
            "prompt": context + "\nDIFFERENT_FROM_A is B or C.\nDIFFERENT_FROM_A =",
            "correct": f" {wrong}",
            "wrong": f" {ans}",
        },
        {
            "reader_id": "different_json_min",
            "target_type": "different_choice",
            "prompt": context + '\nReturn only JSON.\n{"different":"',
            "correct": wrong,
            "wrong": ans,
        },
        {
            "reader_id": "different_option_line",
            "target_type": "different_choice",
            "prompt": context + "\nOptions: B, C.\nQuestion: unequal CAT to A?\nAnswer:",
            "correct": f" {wrong}",
            "wrong": f" {ans}",
        },
        {
            "reader_id": "same_natural_control",
            "target_type": "same_choice",
            "prompt": context + "\nWhich object has the same category as Object A? Answer B or C.\nANSWER =",
            "correct": f" {ans}",
            "wrong": f" {wrong}",
        },
        {
            "reader_id": "different_natural_control",
            "target_type": "different_choice",
            "prompt": context + "\nWhich object has a different category from Object A? Answer B or C.\nANSWER =",
            "correct": f" {wrong}",
            "wrong": f" {ans}",
        },
        {
            "reader_id": "same_compare_values",
            "target_type": "same_choice",
            "prompt": context + "\nCompare the category values. The object with the same value as A is",
            "correct": f" {ans}",
            "wrong": f" {wrong}",
        },
        {
            "reader_id": "different_compare_values",
            "target_type": "different_choice",
            "prompt": context + "\nCompare the category values. The object with a different value from A is",
            "correct": f" {wrong}",
            "wrong": f" {ans}",
        },
        {
            "reader_id": "b_eq_a_binary",
            "target_type": "truth_query",
            "prompt": context + "\nB_CAT == A_CAT ?\nAnswer:",
            "correct": " YES" if same_b else " NO",
            "wrong": " NO" if same_b else " YES",
        },
        {
            "reader_id": "c_eq_a_binary",
            "target_type": "truth_query",
            "prompt": context + "\nC_CAT == A_CAT ?\nAnswer:",
            "correct": " YES" if same_c else " NO",
            "wrong": " NO" if same_c else " YES",
        },
    ]


def summarize(rows: list[dict[str, Any]], min_accuracy: float, min_group_accuracy: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    context_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    variant_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["reader_id"]].append(row)
        context_groups[(row["reader_id"], row["context_id"])].append(row)
        variant_groups[(row["reader_id"], row["variant"])].append(row)

    def pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [float(x["margin"]) for x in items]
        return {
            "accuracy": safe_mean([1.0 if x["correct_choice"] else 0.0 for x in items]),
            "mean_margin": safe_mean(margins),
            "std_margin": safe_std(margins),
            "mean_abs_margin": safe_mean([abs(x) for x in margins]),
            "n": len(items),
            "nonfinite": sum(1 for x in items if not x["finite"]),
        }

    by_reader: list[dict[str, Any]] = []
    for reader_id, items in sorted(groups.items()):
        item = pack(items)
        context_acc = {
            context_id: pack(context_groups[(reader_id, context_id)])["accuracy"]
            for context_id, _ in CONTEXTS
            if (reader_id, context_id) in context_groups
        }
        variant_acc = {
            variant: pack(variant_groups[(reader_id, variant)])["accuracy"]
            for variant in ["B_same", "C_same"]
            if (reader_id, variant) in variant_groups
        }
        item.update(
            {
                "reader_id": reader_id,
                "context_accuracy": context_acc,
                "variant_accuracy": variant_acc,
                "min_context_accuracy": min(context_acc.values()) if context_acc else 0.0,
                "min_variant_accuracy": min(variant_acc.values()) if variant_acc else 0.0,
            }
        )
        item["passes_gate"] = (
            item["accuracy"] >= min_accuracy
            and item["min_context_accuracy"] >= min_group_accuracy
            and item["min_variant_accuracy"] >= min_group_accuracy
        )
        by_reader.append(item)
    by_reader.sort(
        key=lambda x: (
            x["passes_gate"],
            x["accuracy"],
            x["min_context_accuracy"],
            x["min_variant_accuracy"],
            x["mean_margin"],
        ),
        reverse=True,
    )
    return {"by_reader": by_reader}


def run(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE64_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    cases = build_cases(args.max_cases)
    cases = cases[args.case_offset : args.case_offset + args.case_count]
    rows: list[dict[str, Any]] = []
    log(f"Phase64 model={args.model} cases={len(cases)} offset={args.case_offset}")
    for case_idx, case in enumerate(cases):
        if case_idx % args.progress_every == 0:
            log(f"{args.model}: case {case_idx}/{len(cases)} rows={len(rows)}")
        for reader in reader_items(case):
            correct = sequence_logprob(
                model, tokenizer, device, reader["prompt"], reader["correct"], args.max_length
            )
            wrong = sequence_logprob(
                model, tokenizer, device, reader["prompt"], reader["wrong"], args.max_length
            )
            margin = correct["logprob"] - wrong["logprob"]
            finite = bool(correct["finite"] and wrong["finite"])
            rows.append(
                {
                    **{k: case[k] for k in ["case_id", "context_id", "cat_same", "cat_diff", "variant", "answer", "wrong"]},
                    "reader_id": reader["reader_id"],
                    "target_type": reader["target_type"],
                    "correct_completion": reader["correct"],
                    "wrong_completion": reader["wrong"],
                    "correct_logprob": correct["logprob"],
                    "wrong_logprob": wrong["logprob"],
                    "margin": margin,
                    "correct_choice": bool(margin > 0),
                    "finite": finite,
                }
            )
    summary = summarize(rows, args.min_accuracy, args.min_group_accuracy)
    result = {
        "phase": 64,
        "model": args.model,
        "max_cases": args.max_cases,
        "case_offset": args.case_offset,
        "case_count": args.case_count,
        "num_cases": len(cases),
        "num_rows": len(rows),
        "min_accuracy": args.min_accuracy,
        "min_group_accuracy": args.min_group_accuracy,
        "summary": summary,
        "rows": rows,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out = output_dir / f"{args.model}_phase64_same_class_reader_refine{suffix}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out}")
    for item in summary["by_reader"][:5]:
        log(
            f"top {item['reader_id']}: acc={item['accuracy']:.4f} "
            f"min_ctx={item['min_context_accuracy']:.4f} "
            f"min_variant={item['min_variant_accuracy']:.4f} pass={item['passes_gate']}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=384)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=384)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-group-accuracy", type=float, default=0.85)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
