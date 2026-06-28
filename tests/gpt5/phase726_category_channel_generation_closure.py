#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import load_model, release_model  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase724_fruit_route_channel_group_drilldown import install_head_channel_ablation  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import write_json, write_jsonl  # noqa: E402


OUT_ROOT = Path("results/glm5_phase726_category_channel_generation_closure")
PHASE725_ROOT = Path("results/glm5_phase725_fine_channel_category_route_scan")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_probe_channel(model_name: str) -> dict[str, Any]:
    path = PHASE725_ROOT / f"phase725_{model_name}_fine_channel_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["top_category_selective_channels"][0]
    return {
        "layer": int(row["layer"]),
        "head": int(row["head"]),
        "head_key": row["head_key"],
        "channel": int(row["channel"]),
        "parent_channel_group": row["parent_channel_group"],
        "phase725_category_selectivity": row["category_selectivity"],
        "phase725_mean_logprob_delta": row["mean_mean_logprob_delta"],
    }


def category_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    rows = [c for c in build_cases(None) if c["relation"] == "category"]
    return rows[:max_cases] if max_cases else rows


def install_ablation(model, spec: dict[str, Any]):
    handles, _head_dim = install_head_channel_ablation(
        model,
        int(spec["layer"]),
        int(spec["head"]),
        int(spec["channel"]),
        int(spec["channel"]) + 1,
    )
    return handles


def next_logits(model, device, ids: list[int], ablation: dict[str, Any] | None = None) -> torch.Tensor:
    handles = install_ablation(model, ablation) if ablation else []
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def greedy_generate(
    model,
    tokenizer,
    device,
    prompt: str,
    max_new_tokens: int,
    ablation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = []
    token_records = []
    for _ in range(max_new_tokens):
        logits = next_logits(model, device, ids, ablation)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        token_records.append({"id": tok, "text": tokenizer.decode([tok])})
        ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {
        "new_token_ids": new_ids,
        "new_token_texts": token_records,
        "text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
    }


def norm_text(text: str) -> str:
    return " ".join(text.strip().lower().replace(".", " ").replace(",", " ").split())


def hit_answer(generated: str, answer: str) -> bool:
    g = norm_text(generated)
    a = norm_text(answer)
    return bool(a) and (g == a or g.startswith(a + " ") or (" " + a + " ") in (" " + g + " "))


def summarize(rows: list[dict[str, Any]], model_name: str, channel: dict[str, Any]) -> dict[str, Any]:
    n = len(rows)
    changed = [r for r in rows if norm_text(r["baseline_text"]) != norm_text(r["ablated_text"])]
    base_hits = [r for r in rows if r["baseline_hit"]]
    ablated_hits = [r for r in rows if r["ablated_hit"]]
    hit_drop = [r for r in rows if r["baseline_hit"] and not r["ablated_hit"]]
    hit_gain = [r for r in rows if (not r["baseline_hit"]) and r["ablated_hit"]]
    by_prompt = {}
    for kind in sorted({r["prompt_type"] for r in rows}):
        vals = [r for r in rows if r["prompt_type"] == kind]
        by_prompt[kind] = {
            "n": len(vals),
            "changed_rate": sum(1 for r in vals if norm_text(r["baseline_text"]) != norm_text(r["ablated_text"])) / len(vals),
            "baseline_hit_rate": sum(1 for r in vals if r["baseline_hit"]) / len(vals),
            "ablated_hit_rate": sum(1 for r in vals if r["ablated_hit"]) / len(vals),
            "hit_drop_rate": sum(1 for r in vals if r["baseline_hit"] and not r["ablated_hit"]) / len(vals),
        }
    return {
        "phase": 726,
        "title": "Category Channel Natural Generation Closure",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "probe_channel": channel,
        "n_cases": n,
        "changed_rate": len(changed) / n if n else None,
        "baseline_hit_rate": len(base_hits) / n if n else None,
        "ablated_hit_rate": len(ablated_hits) / n if n else None,
        "hit_drop_rate": len(hit_drop) / n if n else None,
        "hit_gain_rate": len(hit_gain) / n if n else None,
        "by_prompt_type": by_prompt,
        "baseline_outputs": Counter(norm_text(r["baseline_text"]) for r in rows).most_common(20),
        "ablated_outputs": Counter(norm_text(r["ablated_text"]) for r in rows).most_common(20),
        "changed_examples": changed[:24],
    }


def run_model(args) -> dict[str, Any]:
    channel = load_probe_channel(args.model)
    cases = category_cases(args.max_cases)
    log(f"{args.model}: cases={len(cases)}, channel={channel}")
    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            baseline = greedy_generate(model, tokenizer, device, prompt, args.max_new_tokens)
            ablated = greedy_generate(model, tokenizer, device, prompt, args.max_new_tokens, channel)
            rows.append(
                {
                    "model": args.model,
                    "case_id": case["case_id"],
                    "prompt_type": case["prompt_type"],
                    "object": case["object"],
                    "object_group": case["object_group"],
                    "answer": case["answer"],
                    "probe_head_key": channel["head_key"],
                    "probe_channel": channel["channel"],
                    "baseline_text": baseline["text"],
                    "ablated_text": ablated["text"],
                    "baseline_hit": hit_answer(baseline["text"], case["answer"]),
                    "ablated_hit": hit_answer(ablated["text"], case["answer"]),
                    "changed": norm_text(baseline["text"]) != norm_text(ablated["text"]),
                    "baseline_tokens": baseline["new_token_texts"],
                    "ablated_tokens": ablated["new_token_texts"],
                }
            )
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: {idx}/{len(cases)} cases")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = summarize(rows, args.model, channel)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase726_{args.model}_generation_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase726_{args.model}_generation_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ["model", "n_cases", "changed_rate", "baseline_hit_rate", "ablated_hit_rate", "hit_drop_rate"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase726_{model}_generation_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 726,
        "title": "Category Channel Natural Generation Closure",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "greedy natural generation under single-channel ablation",
        "small_model_caution": "greedy short generation is a narrow behavioral closure test; weak changes do not falsify likelihood-level causality",
        "by_model": {
            s["model"]: {
                "probe_channel": s["probe_channel"],
                "n_cases": s["n_cases"],
                "changed_rate": s["changed_rate"],
                "baseline_hit_rate": s["baseline_hit_rate"],
                "ablated_hit_rate": s["ablated_hit_rate"],
                "hit_drop_rate": s["hit_drop_rate"],
                "by_prompt_type": s["by_prompt_type"],
                "changed_examples": s["changed_examples"][:8],
            }
            for s in summaries
        },
    }
    write_json(OUT_ROOT / "phase726_cross_model_summary.json", payload)
    lines = [
        "# Phase 726 Category Channel Natural Generation Closure",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: greedy natural generation under single-channel ablation.",
        "",
        "| model | channel | n | changed_rate | baseline_hit | ablated_hit | hit_drop |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, item in payload["by_model"].items():
        ch = item["probe_channel"]
        lines.append(
            f"| {model} | {ch['head_key']}:{ch['channel']} | {item['n_cases']} | "
            f"{item['changed_rate']:.3f} | {item['baseline_hit_rate']:.3f} | "
            f"{item['ablated_hit_rate']:.3f} | {item['hit_drop_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- This tests natural greedy output, not stochastic decoding.",
            "- A low hit-drop rate means the single channel affects likelihood more than final greedy category choice.",
            "- Strong generation closure would require output category changes under ablation.",
            "",
        ]
    )
    (OUT_ROOT / "phase726_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"]}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=8)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        write_cross_summary()
        return
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    run_model(args)
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
