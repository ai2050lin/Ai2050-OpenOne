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
            msg = f"{impl}: {exc}"
            errors.append(msg)
            log(f"Failed with {msg}")
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


EVENT_PAIRS = [
    ("dax smiled", "wug left"),
    ("mip slept", "tev arrived"),
    ("zog jumped", "nif waited"),
    ("pav cooked", "lom ate"),
    ("sarn opened", "fep closed"),
    ("kiv started", "ral finished"),
    ("norb called", "tess answered"),
    ("vok entered", "pem exited"),
    ("laz won", "gup lost"),
    ("dorn broke", "mav repaired"),
    ("siv wrote", "tul read"),
    ("bex rose", "karn fell"),
    ("hesh packed", "rul unpacked"),
    ("marn hid", "tov searched"),
    ("fesk rang", "nup listened"),
    ("gorn signed", "vel copied"),
    ("pesh climbed", "dak rested"),
    ("lorn paid", "miv counted"),
    ("tarn mixed", "zef poured"),
    ("wesh woke", "kob yawned"),
    ("relm landed", "sok waved"),
    ("pim knocked", "narl opened"),
    ("gath blinked", "tevon stared"),
    ("sula waited", "kesh returned"),
]


RELATIONS = [
    ("A_before_B", "{a} happened before {b}.", "A"),
    ("A_after_B", "{a} happened after {b}.", "B"),
    ("B_before_A", "{b} happened before {a}.", "B"),
    ("B_after_A", "{b} happened after {a}.", "A"),
]


CONTEXTS = [
    ("plain", "A = {a}. B = {b}. Relation: {relation}"),
    ("record", "Record:\nA: {a}\nB: {b}\nRelation: {relation}"),
    ("timeline", "Timeline note. Event A is {a}. Event B is {b}. The relation says: {relation}"),
    ("compact", "A:{a}\nB:{b}\nREL:{relation}"),
]


def build_prompts(max_cases: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for context_id, context_template in CONTEXTS:
        for a, b in EVENT_PAIRS:
            for relation_id, relation_template, first in RELATIONS:
                relation = relation_template.format(a=a, b=b)
                context = context_template.format(a=a, b=b, relation=relation)
                second = "B" if first == "A" else "A"
                prompts.append(
                    {
                        "case_id": f"{context_id}:{a}:{b}:{relation_id}",
                        "context_id": context_id,
                        "event_a": a,
                        "event_b": b,
                        "relation_id": relation_id,
                        "operator": "before" if "before" in relation_template else "after",
                        "first": first,
                        "second": second,
                        "context": context,
                    }
                )
                if len(prompts) >= max_cases:
                    return prompts
    return prompts


def reader_items(case: dict[str, Any]) -> list[dict[str, Any]]:
    first = case["first"]
    second = case["second"]
    first_event = "EVENT_A" if first == "A" else "EVENT_B"
    second_event = "EVENT_A" if second == "A" else "EVENT_B"
    first_event_spaced = f" {first_event}"
    second_event_spaced = f" {second_event}"
    first_letter = f" {first}"
    second_letter = f" {second}"
    first_order = " A before B" if first == "A" else " B before A"
    second_order = " B before A" if first == "A" else " A before B"

    context = case["context"]
    a_first_question = (
        context
        + "\nQuestion: Did EVENT_A happen first? Answer yes or no.\nANSWER ="
    )
    b_first_question = (
        context
        + "\nQuestion: Did EVENT_B happen first? Answer yes or no.\nANSWER ="
    )
    return [
        {
            "reader_id": "first_event_letter",
            "prompt": context + "\nAnswer with A or B.\nFIRST_EVENT =",
            "correct": first_letter,
            "wrong": second_letter,
            "target_type": "first_event",
        },
        {
            "reader_id": "first_event_event_label",
            "prompt": context + "\nAnswer with EVENT_A or EVENT_B.\nFIRST_EVENT =",
            "correct": first_event_spaced,
            "wrong": second_event_spaced,
            "target_type": "first_event",
        },
        {
            "reader_id": "json_first_event",
            "prompt": context + '\nReturn JSON with the first event.\n{"FIRST_EVENT":"',
            "correct": first,
            "wrong": second,
            "target_type": "first_event",
        },
        {
            "reader_id": "order_pair",
            "prompt": context + "\nChoose the chronological order.\nORDER =",
            "correct": first_order,
            "wrong": second_order,
            "target_type": "order_pair",
        },
        {
            "reader_id": "a_first_yesno",
            "prompt": a_first_question,
            "correct": " yes" if first == "A" else " no",
            "wrong": " no" if first == "A" else " yes",
            "target_type": "truth_query",
        },
        {
            "reader_id": "b_first_yesno",
            "prompt": b_first_question,
            "correct": " yes" if first == "B" else " no",
            "wrong": " no" if first == "B" else " yes",
            "target_type": "truth_query",
        },
        {
            "reader_id": "before_statement_yesno",
            "prompt": context + "\nStatement: EVENT_A happened before EVENT_B.\nIs the statement true? Answer yes or no.\nANSWER =",
            "correct": " yes" if case["relation_id"] == "A_before_B" or case["relation_id"] == "B_after_A" else " no",
            "wrong": " no" if case["relation_id"] == "A_before_B" or case["relation_id"] == "B_after_A" else " yes",
            "target_type": "truth_query",
        },
        {
            "reader_id": "after_statement_yesno",
            "prompt": context + "\nStatement: EVENT_A happened after EVENT_B.\nIs the statement true? Answer yes or no.\nANSWER =",
            "correct": " yes" if case["relation_id"] == "A_after_B" or case["relation_id"] == "B_before_A" else " no",
            "wrong": " no" if case["relation_id"] == "A_after_B" or case["relation_id"] == "B_before_A" else " yes",
            "target_type": "truth_query",
        },
    ]


def summarize(rows: list[dict[str, Any]], min_accuracy: float, min_group_accuracy: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    context_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    relation_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    target_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["reader_id"]].append(row)
        context_groups[(row["reader_id"], row["context_id"])].append(row)
        relation_groups[(row["reader_id"], row["relation_id"])].append(row)
        target_groups[(row["reader_id"], row["target_type"])].append(row)

    def pack(items: list[dict[str, Any]]) -> dict[str, Any]:
        margins = [float(x["margin"]) for x in items]
        return {
            "accuracy": safe_mean([1.0 if x["correct_choice"] else 0.0 for x in items]),
            "mean_margin": safe_mean(margins),
            "std_margin": safe_std(margins),
            "mean_abs_margin": safe_mean([abs(x) for x in margins]),
            "n": len(items),
            "nonfinite": sum(0 if x["finite"] else 1 for x in items),
        }

    by_reader = []
    for reader_id, items in sorted(groups.items()):
        ctx = [pack(v)["accuracy"] for (rid, _), v in context_groups.items() if rid == reader_id]
        rel = [pack(v)["accuracy"] for (rid, _), v in relation_groups.items() if rid == reader_id]
        base = pack(items)
        base.update(
            {
                "reader_id": reader_id,
                "min_context_accuracy": min(ctx) if ctx else 0.0,
                "min_relation_accuracy": min(rel) if rel else 0.0,
            }
        )
        base["passes_gate"] = (
            base["accuracy"] >= min_accuracy
            and base["min_context_accuracy"] >= min_group_accuracy
            and base["min_relation_accuracy"] >= min_group_accuracy
            and base["nonfinite"] == 0
        )
        by_reader.append(base)
    by_reader.sort(key=lambda x: (x["passes_gate"], x["accuracy"], x["min_relation_accuracy"]), reverse=True)

    return {
        "by_reader": by_reader,
        "by_context": [
            {"reader_id": rid, "context_id": cid, **pack(items)}
            for (rid, cid), items in sorted(context_groups.items())
        ],
        "by_relation": [
            {"reader_id": rid, "relation_id": rel, **pack(items)}
            for (rid, rel), items in sorted(relation_groups.items())
        ],
        "by_target_type": [
            {"reader_id": rid, "target_type": target, **pack(items)}
            for (rid, target), items in sorted(target_groups.items())
        ],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model, tokenizer, device = load_model(args.model, args.attn_implementations)
    all_cases = build_prompts(args.max_cases)
    if args.case_count is not None:
        cases = all_cases[args.case_offset : args.case_offset + args.case_count]
    else:
        cases = all_cases[args.case_offset :]
    log(
        f"Phase60 Temporal Order Reader Calibration — {args.model}, "
        f"base_cases={len(cases)}, offset={args.case_offset}, total_pool={len(all_cases)}"
    )
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, 1):
        for item in reader_items(case):
            correct_lp = sequence_logprob(model, tokenizer, device, item["prompt"], item["correct"], args.max_length)
            wrong_lp = sequence_logprob(model, tokenizer, device, item["prompt"], item["wrong"], args.max_length)
            margin = correct_lp["logprob"] - wrong_lp["logprob"]
            rows.append(
                {
                    "model": args.model,
                    "case_id": case["case_id"],
                    "context_id": case["context_id"],
                    "relation_id": case["relation_id"],
                    "operator": case["operator"],
                    "first": case["first"],
                    "reader_id": item["reader_id"],
                    "target_type": item["target_type"],
                    "correct_completion": item["correct"],
                    "wrong_completion": item["wrong"],
                    "correct_logprob": correct_lp["logprob"],
                    "wrong_logprob": wrong_lp["logprob"],
                    "margin": margin,
                    "correct_choice": margin > 0,
                    "finite": correct_lp["finite"] and wrong_lp["finite"],
                    "correct_num_tokens": correct_lp["num_tokens"],
                    "wrong_num_tokens": wrong_lp["num_tokens"],
                }
            )
        if i % args.progress_every == 0 or i == len(cases):
            current = summarize(rows, args.min_accuracy, args.min_group_accuracy)["by_reader"]
            top = current[0] if current else {}
            log(
                f"  {i}/{len(cases)} rows={len(rows)} "
                f"top={top.get('reader_id')} acc={top.get('accuracy', 0.0):.3f} "
                f"elapsed={time.time() - t0:.0f}s"
            )

    summary = summarize(rows, args.min_accuracy, args.min_group_accuracy)
    result = {
        "phase": 60,
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    out_path = out_dir / f"{args.model}_phase60_temporal_order_reader_calibration{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved {out_path}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--output-dir", default="results/gpt5_phase60_temporal_order_reader_calibration_full")
    parser.add_argument("--max-cases", type=int, default=384)
    parser.add_argument("--case-offset", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=24)
    parser.add_argument("--attn-implementations", default=os.environ.get("PHASE60_ATTN_IMPLEMENTATIONS", "flash_attention_2,sdpa,eager"))
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-group-accuracy", type=float, default=0.85)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
