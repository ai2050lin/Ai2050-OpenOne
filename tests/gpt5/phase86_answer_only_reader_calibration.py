from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
from phase77_balanced_cross_relation_joint_closure import build_expanded_items  # noqa: E402


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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[\s\"'`:{\[\(=,_\-]+", "", text)
    text = re.sub(r"[\s\"'`\}\]\),;:.!?]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_answer_span(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^\s*(answer|value)\s*[:=]\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*[\{\"'`]+", "", text)
    span = re.split(r"[\n\r\.;]", text, maxsplit=1)[0]
    if "," in span and len(span.split("," )[0].split()) <= 5:
        span = span.split(",", 1)[0]
    return normalize_text(span)


def content_words(text: str) -> set[str]:
    stop = {"a", "an", "the", "of", "kind", "type", "sort"}
    return {x for x in re.findall(r"[a-z0-9]+", normalize_text(text)) if x not in stop}


def hit_metrics(generated: str, target: str) -> dict[str, Any]:
    first = first_answer_span(generated)
    g_norm = normalize_text(generated)
    t_norm = normalize_text(target)
    target_words = content_words(target)
    first_words = content_words(first)
    overlap = len(target_words & first_words)
    target_coverage = (overlap / len(target_words)) if target_words else 0.0
    first_precision = (overlap / len(first_words)) if first_words else 0.0
    return {
        "first_span": first,
        "exact_hit": first == t_norm,
        "prefix_hit": first.startswith(t_norm),
        "contains_hit": t_norm in first or t_norm in g_norm,
        "word_subset_hit": bool(target_words) and target_words.issubset(first_words),
        "family_overlap_hit": target_coverage >= 0.5 and first_precision >= 0.5,
        "target_word_coverage": target_coverage,
        "first_word_precision": first_precision,
        "short_output": len(first_words) <= 5,
        "format_violation": bool(re.search(r"\b(because|therefore|this means|for example|is a|are a)\b", g_norm)) or len(first_words) > 8,
    }


def build_reader_templates() -> dict[str, str]:
    return {
        "answer_only_plain": "Complete with only the answer value.\n{clean_prompt}\nAnswer:",
        "answer_only_short_phrase": "Return exactly one short noun phrase. Do not explain.\nContext: {clean_prompt}\nAnswer:",
        "question_value": "Question: What is the answer value?\nContext: {clean_prompt}\nAnswer:",
        "fill_blank_answer": "Fill the blank with only the answer.\n{clean_prompt} ___\nAnswer:",
        "json_value": "Context: {clean_prompt}\nReturn only the JSON value for answer.\n{{\"answer\":\"",
        "value_equals": "Context: {clean_prompt}\nVALUE =",
        "bare_answer": "{clean_prompt}\nAnswer:",
        "one_phrase": "One short phrase only:\n{clean_prompt}\n",
    }


def render_template(template: str, item: dict[str, Any]) -> str:
    return template.format(clean_prompt=item["clean_prompt"], object=item["object"], relation=item["relation"])


def generate_answer(
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(vals),
            "exact_hit": avg([float(v["exact_hit"]) for v in vals]),
            "prefix_hit": avg([float(v["prefix_hit"]) for v in vals]),
            "contains_hit": avg([float(v["contains_hit"]) for v in vals]),
            "word_subset_hit": avg([float(v["word_subset_hit"]) for v in vals]),
            "family_overlap_hit": avg([float(v["family_overlap_hit"]) for v in vals]),
            "target_word_coverage": avg([float(v["target_word_coverage"]) for v in vals]),
            "first_word_precision": avg([float(v["first_word_precision"]) for v in vals]),
            "format_violation": avg([float(v["format_violation"]) for v in vals]),
            "short_output": avg([float(v["short_output"]) for v in vals]),
        }

    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_template_relation: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[row["template_key"]].append(row)
        by_template_relation[(row["template_key"], row["relation"])].append(row)
        by_relation[row["relation"]].append(row)

    return {
        "by_template": {k: group_summary(v) for k, v in by_template.items()},
        "by_relation": {k: group_summary(v) for k, v in by_relation.items()},
        "by_template_relation": {f"{k[0]}:{k[1]}": group_summary(v) for k, v in by_template_relation.items()},
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE86_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    items = build_expanded_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    template_map = build_reader_templates()
    if args.templates:
        keep = set(parse_csv(args.templates))
        template_map = {k: v for k, v in template_map.items() if k in keep}
    log(f"Phase86 model={args.model} items={len(items)} templates={list(template_map)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase86_answer_only_reader_calibration.json"
    partial_path = out_dir / f"{args.model}_phase86_answer_only_reader_calibration.partial.json"

    results: dict[str, Any] = {
        "phase": 86,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "answer_only_reader_calibration",
        "num_items": len(items),
        "max_new_tokens": args.max_new_tokens,
        "relations": sorted({x["relation"] for x in items}),
        "templates": list(template_map.keys()),
        "rows": [],
        "summary": {},
        "samples": [],
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 86 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results.setdefault("samples", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {(int(r["item_idx"]), str(r["template_key"])) for r in results["rows"]}
    t0 = time.time()
    for idx, item in enumerate(items):
        for key, template in template_map.items():
            if (idx, key) in completed:
                continue
            prompt = render_template(template, item)
            generated = generate_answer(model, tokenizer, device, prompt, args.max_length, args.max_new_tokens)
            metrics = hit_metrics(generated, item["target"])
            row = {
                "item_idx": idx,
                "template_key": key,
                "relation": item["relation"],
                "frame_key": item["frame_key"],
                "object": item["object"],
                "target": item["target"],
                "prompt": prompt if args.save_prompts else "",
                "generated": generated,
                **metrics,
            }
            results["rows"].append(row)
            if len(results["samples"]) < args.max_samples and (metrics["prefix_hit"] or metrics["word_subset_hit"] or idx < 3):
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
    parser.add_argument("--templates", default="")
    parser.add_argument("--max-items", type=int, default=672)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=84)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=80)
    parser.add_argument("--save-prompts", action="store_true")
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
