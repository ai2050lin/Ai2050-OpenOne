#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import random
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
from phase693_boundary_attention_head_candidate_audit import head_meta  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402


OUT_ROOT = Path("results/glm5_phase724_fruit_route_channel_group_drilldown")
PHASE723_ROOT = Path("results/glm5_phase723_apple_fruit_attribute_micro_atlas")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_heads(model_name: str, top_heads: int) -> list[dict[str, Any]]:
    path = PHASE723_ROOT / f"phase723_{model_name}_micro_atlas_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    heads = []
    for row in data["most_harmful_candidate_heads"][:top_heads]:
        heads.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "head_key": row["head_key"],
                "phase723_mean_logprob_delta": float(row["mean_mean_logprob_delta"]),
                "phase723_reuse_difference": row["reuse_difference"],
            }
        )
    return heads


def install_head_channel_ablation(model, layer: int, head: int, start: int, end: int):
    o_proj, n_heads, head_dim = head_meta(model, layer)
    if not 0 <= head < n_heads:
        raise ValueError(f"invalid head {head} for layer {layer}, n_heads={n_heads}")
    start = max(0, min(start, head_dim))
    end = max(start, min(end, head_dim))

    def pre_hook(_module, inputs):
        x = inputs[0]
        y = x.clone()
        yv = y.view(y.shape[0], y.shape[1], n_heads, head_dim)
        yv[0, -1, head, start:end] = 0
        return (y,) + tuple(inputs[1:])

    return [o_proj.register_forward_pre_hook(pre_hook)], head_dim


def run_logits_ids(
    model,
    device,
    ids: list[int],
    ablation: dict[str, int] | None = None,
) -> torch.Tensor:
    handles = []
    if ablation:
        handles, _head_dim = install_head_channel_ablation(
            model,
            int(ablation["layer"]),
            int(ablation["head"]),
            int(ablation["start"]),
            int(ablation["end"]),
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


def channel_groups(model, layer: int, n_groups: int) -> list[tuple[int, int]]:
    _o_proj, _n_heads, head_dim = head_meta(model, layer)
    step = max(1, (head_dim + n_groups - 1) // n_groups)
    return [(i, min(i + step, head_dim)) for i in range(0, head_dim, step)]


def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return out


def summarize(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["head_key"], row["channel_start"], row["channel_end"])].append(row)
    channel_summaries = []
    for (head_key, start, end), vals in groups.items():
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
                "channel_start": start,
                "channel_end": end,
                "n": len(vals),
                "mean_mean_logprob_delta": mean([v["mean_logprob_delta"] for v in vals]),
                "mean_first_rank_delta": mean([v["first_rank_delta"] for v in vals]),
                "first_top1_drop_rate": sum(1 for v in vals if v["baseline_first_top1"] and not v["patched_first_top1"]) / len(vals),
                "logprob_worse_rate": sum(1 for v in vals if v["mean_logprob_delta"] < 0) / len(vals),
                "by_relation_necessity": {
                    rel: mean([-v["mean_logprob_delta"] for v in rvals])
                    for rel, rvals in group_by(vals, "relation").items()
                },
                "reuse_difference": {
                    "apple_explicit_necessity": apple_need,
                    "other_fruit_explicit_necessity": fruit_need,
                    "nonfruit_explicit_necessity": nonfruit_need,
                    "apple_minus_other_fruit": None if apple_need is None or fruit_need is None else apple_need - fruit_need,
                    "other_fruit_minus_nonfruit": None if fruit_need is None or nonfruit_need is None else fruit_need - nonfruit_need,
                },
            }
        )
    by_head = {}
    for head_key, vals in group_by(channel_summaries, "head_key").items():
        by_head[head_key] = {
            "top_harmful_channel_groups": sorted(vals, key=lambda r: r["mean_mean_logprob_delta"] or 0)[:8],
            "top_fruit_shared_channel_groups": sorted(
                vals,
                key=lambda r: r["reuse_difference"]["other_fruit_minus_nonfruit"]
                if r["reuse_difference"]["other_fruit_minus_nonfruit"] is not None
                else -999,
                reverse=True,
            )[:8],
            "top_apple_specific_channel_groups": sorted(
                vals,
                key=lambda r: r["reuse_difference"]["apple_minus_other_fruit"]
                if r["reuse_difference"]["apple_minus_other_fruit"] is not None
                else -999,
                reverse=True,
            )[:8],
        }
    return {
        "phase": 724,
        "title": "Fruit Route Channel Group Drilldown",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": len(rows),
        "n_cases": len({r["case_id"] for r in rows}),
        "evidence_type": "teacher-forced phrase likelihood under contiguous channel-group zero ablation inside selected attention head output",
        "by_head": by_head,
        "top_harmful_channel_groups": sorted(channel_summaries, key=lambda r: r["mean_mean_logprob_delta"] or 0)[:16],
        "top_fruit_shared_channel_groups": sorted(
            channel_summaries,
            key=lambda r: r["reuse_difference"]["other_fruit_minus_nonfruit"]
            if r["reuse_difference"]["other_fruit_minus_nonfruit"] is not None
            else -999,
            reverse=True,
        )[:16],
        "top_apple_specific_channel_groups": sorted(
            channel_summaries,
            key=lambda r: r["reuse_difference"]["apple_minus_other_fruit"]
            if r["reuse_difference"]["apple_minus_other_fruit"] is not None
            else -999,
            reverse=True,
        )[:16],
    }


def run_model(args) -> dict[str, Any]:
    random.seed(args.seed)
    cases = build_cases(args.max_cases)
    heads = load_heads(args.model, args.top_heads)
    log(f"{args.model}: cases={len(cases)}, heads={[h['head_key'] for h in heads]}, channel_groups={args.channel_groups}")

    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        groups_by_head = {h["head_key"]: channel_groups(model, h["layer"], args.channel_groups) for h in heads}
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            baseline = phrase_diag(model, tokenizer, device, prompt, case["answer"])
            for h in heads:
                for start, end in groups_by_head[h["head_key"]]:
                    patched = phrase_diag(
                        model,
                        tokenizer,
                        device,
                        prompt,
                        case["answer"],
                        {"layer": h["layer"], "head": h["head"], "start": start, "end": end},
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
                            "layer": int(h["layer"]),
                            "head": int(h["head"]),
                            "head_key": h["head_key"],
                            "channel_start": start,
                            "channel_end": end,
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
    write_jsonl(OUT_ROOT / f"phase724_{args.model}_channel_group_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase724_{args.model}_channel_group_summary.json", summary)
    compact = {
        "model": args.model,
        "n_cases": summary["n_cases"],
        "n_rows": summary["n_rows"],
        "top_harmful_channel_groups": summary["top_harmful_channel_groups"][:5],
        "top_fruit_shared_channel_groups": summary["top_fruit_shared_channel_groups"][:5],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return summary


def fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.4f}"


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase724_{model}_channel_group_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 724,
        "title": "Fruit Route Channel Group Drilldown",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "contiguous channel-group zero ablation inside selected attention head output",
        "small_model_caution": "channel groups are coarse contiguous slices, not discovered neurons; effects are localization hints only",
        "by_model": {
            s["model"]: {
                "n_cases": s["n_cases"],
                "n_rows": s["n_rows"],
                "top_harmful_channel_groups": s["top_harmful_channel_groups"],
                "top_fruit_shared_channel_groups": s["top_fruit_shared_channel_groups"],
                "top_apple_specific_channel_groups": s["top_apple_specific_channel_groups"],
            }
            for s in summaries
        },
    }
    write_json(OUT_ROOT / "phase724_cross_model_summary.json", payload)
    lines = [
        "# Phase 724 Fruit Route Channel Group Drilldown",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: contiguous channel-group zero ablation inside selected attention head output.",
        "",
        "## Top Harmful Channel Groups",
        "",
    ]
    for model, item in payload["by_model"].items():
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| head | channels | mean_logprob_delta | rank_delta | top1_drop | fruit-nonfruit | apple-fruit |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in item["top_harmful_channel_groups"][:10]:
            rd = r["reuse_difference"]
            lines.append(
                f"| {r['head_key']} | {r['channel_start']}-{r['channel_end']} | {r['mean_mean_logprob_delta']:.4f} | "
                f"{r['mean_first_rank_delta']:.2f} | {r['first_top1_drop_rate']:.3f} | "
                f"{fmt(rd['other_fruit_minus_nonfruit'])} | {fmt(rd['apple_minus_other_fruit'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Strict Interpretation",
            "",
            "- This is not neuron-level decoding.",
            "- Contiguous channel groups are coarse probes of head output subspace.",
            "- A strong channel group suggests where to run finer channel or neuron scans.",
            "",
        ]
    )
    (OUT_ROOT / "phase724_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "models": payload["models"]}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--top-heads", type=int, default=2)
    parser.add_argument("--channel-groups", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=12)
    parser.add_argument("--seed", type=int, default=724)
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
