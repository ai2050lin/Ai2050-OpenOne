from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
from phase72_object_relation_value_fullseq_closure import stats_from_scores  # noqa: E402
from phase76_object_frame_joint_closure import fullseq_logprob_multi, uniq  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items  # noqa: E402
from phase86_answer_only_reader_calibration import first_answer_span, hit_metrics  # noqa: E402


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def avg(xs: list[float]) -> float:
    return float(mean(xs)) if xs else 0.0


def generate_text(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] > max_length:
        return ""
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, input_ids.shape[1]:].detach().cpu().tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def option_letters(n: int) -> list[str]:
    return [chr(ord("A") + i) for i in range(n)]


def option_orders(item: dict[str, Any], item_idx: int, max_distractors: int) -> dict[str, list[str]]:
    base = uniq([item["target"], *item["distractors"][:max_distractors]])
    if item["target"] not in base:
        base = [item["target"], *base]
    distractors = [x for x in base if x != item["target"]]
    target_first = [item["target"], *distractors]
    target_last = [*distractors, item["target"]]
    mixed = target_first[:]
    if mixed:
        shift = item_idx % len(mixed)
        mixed = mixed[shift:] + mixed[:shift]
    return {
        "target_first": target_first,
        "target_last": target_last,
        "rotating": mixed,
    }


def render_options(candidates: list[str]) -> str:
    return "\n".join(f"{letter}. {value}" for letter, value in zip(option_letters(len(candidates)), candidates))


def choice_templates() -> dict[str, str]:
    return {
        "choice_plain": "Context: {clean_prompt}\nWhich answer value is best?\n{options}\nAnswer with only the letter:",
        "choice_blank": "Complete the statement with the best option.\nStatement: {clean_prompt} ___\n{options}\nAnswer letter only:",
        "choice_no_explain": "Choose the correct value. Do not explain.\n{clean_prompt}\n{options}\nLetter:",
        "choice_json_letter": "Context: {clean_prompt}\nOptions:\n{options}\nReturn JSON: {{\"letter\":\"",
    }


def open_templates() -> dict[str, str]:
    return {
        "open_fill_blank": "Fill the blank with only the answer.\n{clean_prompt} ___\nAnswer:",
        "open_short_phrase": "Return exactly one short noun phrase. Do not explain.\nContext: {clean_prompt}\nAnswer:",
    }


def parse_choice(generated: str, candidates: list[str]) -> dict[str, Any]:
    text = generated.strip()
    m = re.search(r"\b([A-H])\b", text, flags=re.I)
    if not m:
        m = re.search(r'["\']?([A-H])["\']?', text, flags=re.I)
    selected_letter = m.group(1).upper() if m else ""
    letters = option_letters(len(candidates))
    if selected_letter in letters:
        idx = letters.index(selected_letter)
        return {
            "selected_letter": selected_letter,
            "selected_value": candidates[idx],
            "choice_valid": True,
        }
    span = first_answer_span(text)
    for value in candidates:
        if span and (span == value.lower() or span in value.lower() or value.lower() in span):
            return {
                "selected_letter": "",
                "selected_value": value,
                "choice_valid": True,
            }
    return {
        "selected_letter": selected_letter,
        "selected_value": "",
        "choice_valid": False,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
        choice_vals = [v for v in vals if v["reader_type"] == "choice"]
        open_vals = [v for v in vals if v["reader_type"] == "open"]
        closed_vals = [v for v in vals if v["reader_type"] == "closed"]
        letters = Counter(v.get("selected_letter", "") for v in choice_vals)
        return {
            "n": len(vals),
            "closed_n": len(closed_vals),
            "choice_n": len(choice_vals),
            "open_n": len(open_vals),
            "closed_top1": avg([float(v.get("closed_rank") == 1) for v in closed_vals]),
            "closed_margin": avg([float(v.get("closed_margin", 0.0)) for v in closed_vals]),
            "choice_top1": avg([float(v.get("choice_correct", False)) for v in choice_vals]),
            "choice_valid": avg([float(v.get("choice_valid", False)) for v in choice_vals]),
            "choice_target_letter_rate": avg([float(v.get("selected_letter") == v.get("target_letter")) for v in choice_vals]),
            "open_exact_hit": avg([float(v.get("exact_hit", False)) for v in open_vals]),
            "open_word_subset_hit": avg([float(v.get("word_subset_hit", False)) for v in open_vals]),
            "open_family_overlap_hit": avg([float(v.get("family_overlap_hit", False)) for v in open_vals]),
            "open_format_violation": avg([float(v.get("format_violation", False)) for v in open_vals]),
            "selected_letters": dict(letters),
        }

    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_reader": defaultdict(list),
        "by_relation": defaultdict(list),
        "by_reader_template": defaultdict(list),
        "by_choice_order": defaultdict(list),
        "by_template_relation": defaultdict(list),
    }
    for row in rows:
        groups["by_reader"][row["reader_type"]].append(row)
        groups["by_relation"][row["relation"]].append(row)
        groups["by_reader_template"][(row["reader_type"], row.get("template_key", ""))].append(row)
        if row["reader_type"] == "choice":
            groups["by_choice_order"][(row["template_key"], row["order_key"])].append(row)
        groups["by_template_relation"][(row["reader_type"], row.get("template_key", ""), row["relation"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE87_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    c_templates = choice_templates()
    o_templates = open_templates()
    if args.choice_templates:
        keep = set(parse_csv(args.choice_templates))
        c_templates = {k: v for k, v in c_templates.items() if k in keep}
    if args.open_templates:
        keep = set(parse_csv(args.open_templates))
        o_templates = {k: v for k, v in o_templates.items() if k in keep}
    log(f"Phase87 model={args.model} items={len(items)} choice_templates={list(c_templates)} open_templates={list(o_templates)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase87_reader_stack_calibration.json"
    partial_path = out_dir / f"{args.model}_phase87_reader_stack_calibration.partial.json"
    results: dict[str, Any] = {
        "phase": 87,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "reader_stack_calibration",
        "num_items": len(items),
        "relations": sorted({x["relation"] for x in items}),
        "choice_templates": list(c_templates.keys()),
        "open_templates": list(o_templates.keys()),
        "max_distractors": args.max_distractors,
        "rows": [],
        "summary": {},
        "samples": [],
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 87 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results.setdefault("samples", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {(int(r["item_idx"]), r["reader_type"], r.get("template_key", ""), r.get("order_key", "")) for r in results["rows"]}
    t0 = time.time()
    for idx, item in enumerate(items):
        orders = option_orders(item, idx, args.max_distractors)
        closed_key = (idx, "closed", "fullseq_candidate", "target_plus_distractors")
        if closed_key not in completed:
            closed_candidates = orders["target_first"]
            scores = {
                v: fullseq_logprob_multi(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length, args.module)
                for v in closed_candidates
            }
            stats = stats_from_scores(scores, item["target"], [v for v in closed_candidates if v != item["target"]])
            row = {
                "item_idx": idx,
                "reader_type": "closed",
                "template_key": "fullseq_candidate",
                "order_key": "target_plus_distractors",
                "relation": item["relation"],
                "frame_key": item["frame_key"],
                "object": item["object"],
                "target": item["target"],
                "candidates": closed_candidates,
                "closed_rank": stats["rank"],
                "closed_margin": stats["margin"],
                "closed_top": stats["top"],
                "scores": stats["scores"] if args.save_scores else {},
            }
            results["rows"].append(row)
            if len(results["samples"]) < args.max_samples:
                results["samples"].append(row)

        for order_key, candidates in orders.items():
            target_letter = option_letters(len(candidates))[candidates.index(item["target"])]
            options = render_options(candidates)
            for template_key, template in c_templates.items():
                comp_key = (idx, "choice", template_key, order_key)
                if comp_key in completed:
                    continue
                prompt = template.format(clean_prompt=item["clean_prompt"], options=options)
                generated = generate_text(model, tokenizer, device, prompt, args.max_length, args.choice_max_new_tokens)
                parsed = parse_choice(generated, candidates)
                row = {
                    "item_idx": idx,
                    "reader_type": "choice",
                    "template_key": template_key,
                    "order_key": order_key,
                    "relation": item["relation"],
                    "frame_key": item["frame_key"],
                    "object": item["object"],
                    "target": item["target"],
                    "candidates": candidates,
                    "target_letter": target_letter,
                    "generated": generated,
                    **parsed,
                    "choice_correct": parsed["selected_value"] == item["target"],
                }
                results["rows"].append(row)
                if len(results["samples"]) < args.max_samples and (row["choice_correct"] or not row["choice_valid"]):
                    results["samples"].append(row)

        for template_key, template in o_templates.items():
            comp_key = (idx, "open", template_key, "")
            if comp_key in completed:
                continue
            prompt = template.format(clean_prompt=item["clean_prompt"])
            generated = generate_text(model, tokenizer, device, prompt, args.max_length, args.open_max_new_tokens)
            metrics = hit_metrics(generated, item["target"])
            row = {
                "item_idx": idx,
                "reader_type": "open",
                "template_key": template_key,
                "order_key": "",
                "relation": item["relation"],
                "frame_key": item["frame_key"],
                "object": item["object"],
                "target": item["target"],
                "generated": generated,
                **metrics,
            }
            results["rows"].append(row)
            if len(results["samples"]) < args.max_samples and (metrics["word_subset_hit"] or idx < 3):
                results["samples"].append(row)

        if (idx + 1) % args.progress_every == 0:
            log(f"item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
            partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            cleanup_cuda()

    results["summary"] = summarize(results["rows"])
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--relations", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-items", type=int, default=672)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--choice-max-new-tokens", type=int, default=4)
    parser.add_argument("--open-max-new-tokens", type=int, default=8)
    parser.add_argument("--choice-templates", default="")
    parser.add_argument("--open-templates", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=84)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        release_loaded(None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
