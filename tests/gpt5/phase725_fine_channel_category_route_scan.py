#!/usr/bin/env python3
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
from typing import Any

import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))

from model_utils import load_model, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase724_fruit_route_channel_group_drilldown import install_head_channel_ablation  # noqa: E402


OUT_ROOT = Path("results/glm5_phase725_fine_channel_category_route_scan")
PHASE724_ROOT = Path("results/glm5_phase724_fruit_route_channel_group_drilldown")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return out


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def load_channel_groups(model_name: str, n_groups: int) -> list[dict[str, Any]]:
    path = PHASE724_ROOT / f"phase724_{model_name}_channel_group_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    pool = data["top_harmful_channel_groups"] + data["top_fruit_shared_channel_groups"]
    out = []
    seen = set()
    for row in pool:
        key = (int(row["layer"]), int(row["head"]), int(row["channel_start"]), int(row["channel_end"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "layer": key[0],
                "head": key[1],
                "head_key": row["head_key"],
                "channel_start": key[2],
                "channel_end": key[3],
                "phase724_mean_logprob_delta": float(row["mean_mean_logprob_delta"]),
                "phase724_reuse_difference": row["reuse_difference"],
            }
        )
        if len(out) >= n_groups:
            break
    return out


def run_logits_ids(model, device, ids: list[int], ablation: dict[str, int] | None = None) -> torch.Tensor:
    handles = []
    if ablation:
        handles, _head_dim = install_head_channel_ablation(
            model,
            int(ablation["layer"]),
            int(ablation["head"]),
            int(ablation["channel"]),
            int(ablation["channel"]) + 1,
        )
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().cpu()
    finally:
        for h in handles:
            h.remove()


def phrase_diag(model, tokenizer, device, prompt: str, answer: str, ablation: dict[str, int] | None = None) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(prompt_ids)
    token_diags = []
    for target_id in ans_ids:
        diag = logit_diag(run_logits_ids(model, device, cur, ablation), int(target_id))
        token_diags.append(diag)
        cur.append(int(target_id))
    return {
        "sum_logprob": sum(d["target_logprob"] for d in token_diags),
        "mean_logprob": sum(d["target_logprob"] for d in token_diags) / len(token_diags),
        "first_logprob": token_diags[0]["target_logprob"],
        "first_rank": token_diags[0]["target_rank"],
        "first_top1": token_diags[0]["target_top1"],
        "n_answer_tokens": len(ans_ids),
    }


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["head_key"], int(row["channel"]))].append(row)
    channel_summaries = []
    for (head_key, channel), vals in groups.items():
        by_relation = group_by(vals, "relation")
        relation_need = {rel: mean([-v["mean_logprob_delta"] for v in rvals]) for rel, rvals in by_relation.items()}
        category_need = relation_need.get("category")
        other_relation_needs = [v for k, v in relation_need.items() if k != "category" and v is not None]
        other_relation_mean = mean(other_relation_needs)

        explicit = [v for v in vals if v["prompt_type"] == "explicit_profile"]
        explicit_groups = group_by(explicit, "object_group")
        apple_need = mean([-v["mean_logprob_delta"] for v in explicit_groups.get("apple", [])])
        fruit_need = mean([-v["mean_logprob_delta"] for v in explicit_groups.get("other_fruit", [])])
        nonfruit_need = mean([-v["mean_logprob_delta"] for v in explicit_groups.get("nonfruit", [])])

        channel_summaries.append(
            {
                "model": model_name,
                "head_key": head_key,
                "layer": vals[0]["layer"],
                "head": vals[0]["head"],
                "channel": channel,
                "parent_channel_group": f"{vals[0]['group_start']}-{vals[0]['group_end']}",
                "n": len(vals),
                "mean_mean_logprob_delta": mean([v["mean_logprob_delta"] for v in vals]),
                "mean_first_rank_delta": mean([v["first_rank_delta"] for v in vals]),
                "first_top1_drop_rate": sum(1 for v in vals if v["baseline_first_top1"] and not v["patched_first_top1"]) / len(vals),
                "logprob_worse_rate": sum(1 for v in vals if v["mean_logprob_delta"] < 0) / len(vals),
                "by_relation_necessity": relation_need,
                "category_selectivity": None
                if category_need is None or other_relation_mean is None
                else category_need - other_relation_mean,
                "reuse_difference": {
                    "apple_explicit_necessity": apple_need,
                    "other_fruit_explicit_necessity": fruit_need,
                    "nonfruit_explicit_necessity": nonfruit_need,
                    "apple_minus_other_fruit": None if apple_need is None or fruit_need is None else apple_need - fruit_need,
                    "other_fruit_minus_nonfruit": None if fruit_need is None or nonfruit_need is None else fruit_need - nonfruit_need,
                },
            }
        )
    return {
        "phase": 725,
        "title": "Fine Channel Category Route Scan",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "evidence_type": "single-channel zero ablation inside selected high-effect attention head channel groups",
        "top_harmful_channels": sorted(channel_summaries, key=lambda r: r["mean_mean_logprob_delta"] or 0)[:24],
        "top_category_selective_channels": sorted(
            channel_summaries,
            key=lambda r: r["category_selectivity"] if r["category_selectivity"] is not None else -999,
            reverse=True,
        )[:24],
        "top_fruit_shared_channels": sorted(
            channel_summaries,
            key=lambda r: r["reuse_difference"]["other_fruit_minus_nonfruit"]
            if r["reuse_difference"]["other_fruit_minus_nonfruit"] is not None
            else -999,
            reverse=True,
        )[:24],
        "all_channel_summaries": channel_summaries,
    }


def run_model(args) -> dict[str, Any]:
    cases = build_cases(args.max_cases)
    groups = load_channel_groups(args.model, args.channel_groups)
    log(
        f"{args.model}: cases={len(cases)}, groups="
        f"{[(g['head_key'], g['channel_start'], g['channel_end']) for g in groups]}"
    )

    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        single_channels = []
        for g in groups:
            for c in range(g["channel_start"], g["channel_end"]):
                rec = dict(g)
                rec["channel"] = c
                single_channels.append(rec)
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            baseline = phrase_diag(model, tokenizer, device, prompt, case["answer"])
            for g in single_channels:
                patched = phrase_diag(
                    model,
                    tokenizer,
                    device,
                    prompt,
                    case["answer"],
                    {"layer": g["layer"], "head": g["head"], "channel": g["channel"]},
                )
                rows.append(
                    {
                        "model": args.model,
                        "case_id": case["case_id"],
                        "prompt_type": case["prompt_type"],
                        "object": case["object"],
                        "object_group": case["object_group"],
                        "relation": case["relation"],
                        "answer": case["answer"],
                        "layer": int(g["layer"]),
                        "head": int(g["head"]),
                        "head_key": g["head_key"],
                        "group_start": int(g["channel_start"]),
                        "group_end": int(g["channel_end"]),
                        "channel": int(g["channel"]),
                        "baseline_mean_logprob": baseline["mean_logprob"],
                        "patched_mean_logprob": patched["mean_logprob"],
                        "mean_logprob_delta": patched["mean_logprob"] - baseline["mean_logprob"],
                        "baseline_first_rank": baseline["first_rank"],
                        "patched_first_rank": patched["first_rank"],
                        "first_rank_delta": patched["first_rank"] - baseline["first_rank"],
                        "baseline_first_top1": baseline["first_top1"],
                        "patched_first_top1": patched["first_top1"],
                    }
                )
            if idx % args.log_every == 0 or idx == len(cases):
                log(f"{args.model}: {idx}/{len(cases)} cases; rows={len(rows)}")
    finally:
        release_model(model)
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(rows, args.model)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase725_{args.model}_fine_channel_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase725_{args.model}_fine_channel_summary.json", summary)
    print(
        json.dumps(
            {
                "model": args.model,
                "n_cases": summary["n_cases"],
                "n_rows": summary["n_rows"],
                "top_harmful_channels": summary["top_harmful_channels"][:5],
                "top_category_selective_channels": summary["top_category_selective_channels"][:5],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase725_{model}_fine_channel_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 725,
        "title": "Fine Channel Category Route Scan",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "single-channel zero ablation in high-effect channel groups",
        "small_model_caution": "single-channel effects are very local and off-manifold; interpret as localization hints only",
        "by_model": {
            s["model"]: {
                "n_cases": s["n_cases"],
                "n_rows": s["n_rows"],
                "top_harmful_channels": s["top_harmful_channels"],
                "top_category_selective_channels": s["top_category_selective_channels"],
                "top_fruit_shared_channels": s["top_fruit_shared_channels"],
            }
            for s in summaries
        },
    }
    write_json(OUT_ROOT / "phase725_cross_model_summary.json", payload)
    lines = [
        "# Phase 725 Fine Channel Category Route Scan",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: single-channel zero ablation inside selected high-effect channel groups.",
        "",
        "## Top Harmful Channels",
        "",
    ]
    for model, item in payload["by_model"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| head | channel | parent | mean_delta | rank_delta | top1_drop | category_selectivity | fruit-nonfruit | apple-fruit |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in item["top_harmful_channels"][:12]:
            rd = r["reuse_difference"]
            lines.append(
                f"| {r['head_key']} | {r['channel']} | {r['parent_channel_group']} | "
                f"{r['mean_mean_logprob_delta']:.4f} | {r['mean_first_rank_delta']:.2f} | "
                f"{r['first_top1_drop_rate']:.3f} | {fmt(r['category_selectivity'])} | "
                f"{fmt(rd['other_fruit_minus_nonfruit'])} | {fmt(rd['apple_minus_other_fruit'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Strict Interpretation",
            "",
            "- This is finer than channel-group scanning but still not neuron-level coding.",
            "- A category-selective single channel is a route candidate, not a semantic atom.",
            "- The next required test is residual/MLP propagation and natural generation closure.",
            "",
        ]
    )
    (OUT_ROOT / "phase725_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"]}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--channel-groups", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
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
