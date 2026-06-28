#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from collections import Counter, defaultdict
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
from phase724_fruit_route_channel_group_drilldown import install_head_channel_ablation  # noqa: E402


OUT_ROOT = Path("results/glm5_phase727_category_fruit_cluster_intervention")
PHASE724_ROOT = Path("results/glm5_phase724_fruit_route_channel_group_drilldown")
PHASE725_ROOT = Path("results/glm5_phase725_fine_channel_category_route_scan")
MODELS = ["qwen3", "glm4", "deepseek7b"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def norm_text(text: str) -> str:
    return " ".join(text.strip().lower().replace(".", " ").replace(",", " ").split())


def hit_answer(generated: str, answer: str) -> bool:
    g = norm_text(generated)
    a = norm_text(answer)
    return bool(a) and (g == a or g.startswith(a + " ") or (" " + a + " ") in (" " + g + " "))


def category_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    rows = [c for c in build_cases(None) if c["relation"] == "category"]
    return rows[:max_cases] if max_cases else rows


def load_category_channels(model_name: str, n: int) -> list[dict[str, int]]:
    path = PHASE725_ROOT / f"phase725_{model_name}_fine_channel_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if model_name == "deepseek7b":
        # Keep the theoretically important category cluster found in Phase 725.
        return [{"layer": 20, "head": 17, "head_key": "L20H17", "channel": c} for c in [25, 30, 24, 23][:n]]
    out = []
    seen = set()
    for row in data["top_category_selective_channels"] + data["top_harmful_channels"]:
        key = (int(row["layer"]), int(row["head"]), int(row["channel"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"layer": key[0], "head": key[1], "head_key": row["head_key"], "channel": key[2]})
        if len(out) >= n:
            break
    return out


def load_fruit_groups(model_name: str, avoid_head_key: str, n: int) -> list[dict[str, int]]:
    path = PHASE724_ROOT / f"phase724_{model_name}_channel_group_summary.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    seen = set()
    for row in data["top_fruit_shared_channel_groups"] + data["top_harmful_channel_groups"]:
        if row["head_key"] == avoid_head_key:
            continue
        key = (int(row["layer"]), int(row["head"]), int(row["channel_start"]), int(row["channel_end"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "layer": key[0],
                "head": key[1],
                "head_key": row["head_key"],
                "start": key[2],
                "end": key[3],
            }
        )
        if len(out) >= n:
            break
    return out


def install_multi_ablation(model, ranges: list[dict[str, int]]):
    handles = []
    for r in ranges:
        hs, _head_dim = install_head_channel_ablation(
            model,
            int(r["layer"]),
            int(r["head"]),
            int(r["start"]),
            int(r["end"]),
        )
        handles.extend(hs)
    return handles


def next_logits(model, device, ids: list[int], ranges: list[dict[str, int]] | None = None) -> torch.Tensor:
    handles = install_multi_ablation(model, ranges or []) if ranges else []
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def phrase_diag(model, tokenizer, device, prompt: str, answer: str, ranges: list[dict[str, int]] | None = None) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(prompt_ids)
    diags = []
    for tid in ans_ids:
        diag = logit_diag(next_logits(model, device, cur, ranges), int(tid))
        diags.append(diag)
        cur.append(int(tid))
    return {
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "sum_logprob": sum(d["target_logprob"] for d in diags),
        "first_rank": diags[0]["target_rank"],
        "first_top1": diags[0]["target_top1"],
    }


def greedy_generate(model, tokenizer, device, prompt: str, ranges: list[dict[str, int]], max_new_tokens: int) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = []
    token_records = []
    for _ in range(max_new_tokens):
        logits = next_logits(model, device, ids, ranges)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        token_records.append({"id": tok, "text": tokenizer.decode([tok])})
        ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "tokens": token_records}


def build_interventions(model, model_name: str) -> dict[str, list[dict[str, int]]]:
    cat = load_category_channels(model_name, 4)
    cat_head = cat[0]
    _o_proj, _n_heads, head_dim = head_meta(model, cat_head["layer"])
    fruit = load_fruit_groups(model_name, cat_head["head_key"], 2)
    category_single = [{"layer": cat_head["layer"], "head": cat_head["head"], "start": cat_head["channel"], "end": cat_head["channel"] + 1}]
    category_cluster = [{"layer": c["layer"], "head": c["head"], "start": c["channel"], "end": c["channel"] + 1} for c in cat]
    category_full_head = [{"layer": cat_head["layer"], "head": cat_head["head"], "start": 0, "end": head_dim}]
    fruit_cluster = [{"layer": g["layer"], "head": g["head"], "start": g["start"], "end": g["end"]} for g in fruit]
    return {
        "baseline": [],
        "category_single": category_single,
        "category_cluster": category_cluster,
        "category_full_head": category_full_head,
        "fruit_cluster": fruit_cluster,
        "category_plus_fruit_cluster": category_cluster + fruit_cluster,
    }


def summarize(rows: list[dict[str, Any]], model_name: str, interventions: dict[str, list[dict[str, int]]]) -> dict[str, Any]:
    by_intervention = {}
    base_rows = [r for r in rows if r["intervention"] == "baseline"]
    base_by_case = {r["case_id"]: r for r in base_rows}
    for name in sorted({r["intervention"] for r in rows}):
        vals = [r for r in rows if r["intervention"] == name]
        if name == "baseline":
            changed_rate = 0.0
            hit_drop_rate = 0.0
            mean_logprob_delta = 0.0
        else:
            changed_rate = sum(1 for r in vals if norm_text(r["generated_text"]) != norm_text(base_by_case[r["case_id"]]["generated_text"])) / len(vals)
            hit_drop_rate = sum(1 for r in vals if base_by_case[r["case_id"]]["hit"] and not r["hit"]) / len(vals)
            mean_logprob_delta = sum(r["mean_logprob_delta"] for r in vals) / len(vals)
        by_prompt = {}
        for p in sorted({r["prompt_type"] for r in vals}):
            pvals = [r for r in vals if r["prompt_type"] == p]
            by_prompt[p] = {
                "n": len(pvals),
                "hit_rate": sum(1 for r in pvals if r["hit"]) / len(pvals),
                "mean_logprob_delta": sum(r["mean_logprob_delta"] for r in pvals) / len(pvals),
            }
        by_intervention[name] = {
            "n": len(vals),
            "hit_rate": sum(1 for r in vals if r["hit"]) / len(vals),
            "changed_rate_vs_baseline": changed_rate,
            "hit_drop_rate_vs_baseline": hit_drop_rate,
            "mean_logprob_delta": mean_logprob_delta,
            "first_rank_delta": sum(r["first_rank_delta"] for r in vals) / len(vals),
            "outputs": Counter(norm_text(r["generated_text"]) for r in vals).most_common(12),
            "by_prompt_type": by_prompt,
        }
    return {
        "phase": 727,
        "title": "Category/Fruit Route Cluster Intervention",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len({r["case_id"] for r in rows}),
        "n_rows": len(rows),
        "interventions": interventions,
        "by_intervention": by_intervention,
    }


def run_model(args) -> dict[str, Any]:
    cases = category_cases(args.max_cases)
    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        interventions = build_interventions(model, args.model)
        compact_interventions = {k: v for k, v in interventions.items()}
        log(f"{args.model}: cases={len(cases)}, interventions={json.dumps(compact_interventions, ensure_ascii=False)}")
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            baseline_diag = phrase_diag(model, tokenizer, device, prompt, case["answer"], [])
            for name, ranges in interventions.items():
                diag = baseline_diag if name == "baseline" else phrase_diag(model, tokenizer, device, prompt, case["answer"], ranges)
                generated = greedy_generate(model, tokenizer, device, prompt, ranges, args.max_new_tokens)
                rows.append(
                    {
                        "model": args.model,
                        "case_id": case["case_id"],
                        "prompt_type": case["prompt_type"],
                        "object": case["object"],
                        "object_group": case["object_group"],
                        "answer": case["answer"],
                        "intervention": name,
                        "generated_text": generated["text"],
                        "hit": hit_answer(generated["text"], case["answer"]),
                        "mean_logprob": diag["mean_logprob"],
                        "mean_logprob_delta": diag["mean_logprob"] - baseline_diag["mean_logprob"],
                        "first_rank": diag["first_rank"],
                        "first_rank_delta": diag["first_rank"] - baseline_diag["first_rank"],
                        "first_top1": diag["first_top1"],
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
    summary = summarize(rows, args.model, interventions)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase727_{args.model}_cluster_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase727_{args.model}_cluster_summary.json", summary)
    print(json.dumps({"model": args.model, "n_cases": summary["n_cases"], "by_intervention": summary["by_intervention"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase727_{model}_cluster_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 727,
        "title": "Category/Fruit Route Cluster Intervention",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "cluster-level likelihood and greedy generation intervention",
        "by_model": {s["model"]: {"n_cases": s["n_cases"], "interventions": s["interventions"], "by_intervention": s["by_intervention"]} for s in summaries},
    }
    write_json(OUT_ROOT / "phase727_cross_model_summary.json", payload)
    lines = [
        "# Phase 727 Category/Fruit Route Cluster Intervention",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: cluster-level likelihood and greedy generation intervention.",
        "",
        "| model | intervention | mean_delta | hit_rate | changed | hit_drop | rank_delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, item in payload["by_model"].items():
        for name, rec in item["by_intervention"].items():
            lines.append(
                f"| {model} | {name} | {rec['mean_logprob_delta']:.4f} | {rec['hit_rate']:.3f} | "
                f"{rec['changed_rate_vs_baseline']:.3f} | {rec['hit_drop_rate_vs_baseline']:.3f} | {rec['first_rank_delta']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Cluster likelihood drops without generation hit drops indicate downstream compensation or a generation gate.",
            "- Full-head effects are stronger but less localized.",
            "- This phase still uses greedy category answers only.",
            "",
        ]
    )
    (OUT_ROOT / "phase727_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
