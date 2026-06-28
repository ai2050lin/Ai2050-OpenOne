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

from model_utils import get_layers, load_model, release_model  # noqa: E402
from phase722_functional_head_atlas_causal_ablation import logit_diag, target_token_ids, write_json, write_jsonl  # noqa: E402
from phase723_apple_fruit_attribute_micro_atlas import build_cases, prompt_for  # noqa: E402
from phase724_fruit_route_channel_group_drilldown import install_head_channel_ablation  # noqa: E402
from phase727_category_fruit_cluster_intervention import build_interventions, hit_answer, norm_text  # noqa: E402


OUT_ROOT = Path("results/glm5_phase730_downstream_node_cancellation")
PHASE729_ROOT = Path("results/glm5_phase729_full_head_vs_cluster_residual_propagation")
MODELS = ["qwen3", "glm4", "deepseek7b"]
UPSTREAMS = ["category_cluster", "category_full_head"]
CONDITIONS = ["upstream_only", "cancel_top_layer_out", "cancel_top_mlp_out"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def category_cases(max_cases: int | None = None) -> list[dict[str, Any]]:
    rows = [c for c in build_cases(None) if c["relation"] == "category"]
    return rows[:max_cases] if max_cases else rows


def get_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_tensor(output: Any, vec: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        first = output[0].clone()
        first[0, -1] = vec.to(device=first.device, dtype=first.dtype)
        return (first,) + output[1:]
    y = output.clone()
    y[0, -1] = vec.to(device=y.device, dtype=y.dtype)
    return y


def site_to_kind_layer(site: str) -> tuple[str, int]:
    if site.startswith("hidden_"):
        hidden_idx = int(site.split("_", 1)[1])
        return "layer_out", hidden_idx - 1
    if site.startswith("L") and site.endswith("_mlp_out"):
        prefix = site.split("_", 1)[0]
        return "mlp_out", int(prefix[1:])
    if site.startswith("L") and site.endswith("_attn_out"):
        prefix = site.split("_", 1)[0]
        return "attn_out", int(prefix[1:])
    raise ValueError(f"unsupported site: {site}")


def module_for_site(model, site: str):
    kind, layer = site_to_kind_layer(site)
    layers = get_layers(model)
    if kind == "layer_out":
        return layers[layer]
    if kind == "mlp_out":
        return layers[layer].mlp
    if kind == "attn_out":
        return layers[layer].self_attn
    raise ValueError(kind)


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


def capture_site_vec(model, device, ids: list[int], site: str) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}
    module = module_for_site(model, site)

    def hook(_module, _inputs, output):
        captured["vec"] = get_tensor(output)[0, -1].detach().float().cpu()

    handle = module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
    finally:
        handle.remove()
    return captured["vec"]


def next_logits(
    model,
    device,
    ids: list[int],
    upstream_ranges: list[dict[str, int]] | None = None,
    cancel_site: str | None = None,
    cancel_vec: torch.Tensor | None = None,
) -> torch.Tensor:
    handles = install_multi_ablation(model, upstream_ranges or []) if upstream_ranges else []
    if cancel_site and cancel_vec is not None:
        module = module_for_site(model, cancel_site)

        def cancel_hook(_module, _inputs, output, vec=cancel_vec):
            return replace_tensor(output, vec)

        handles.append(module.register_forward_hook(cancel_hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=torch.tensor([ids], device=device), return_dict=True, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def condition_logits(model, device, ids: list[int], ranges: list[dict[str, int]], condition: str, site_map: dict[str, str]) -> torch.Tensor:
    if condition == "upstream_only":
        return next_logits(model, device, ids, ranges)
    if condition == "cancel_top_layer_out":
        site = site_map["top_layer_out"]
    elif condition == "cancel_top_mlp_out":
        site = site_map["top_mlp_out"]
    else:
        raise ValueError(condition)
    baseline_vec = capture_site_vec(model, device, ids, site)
    return next_logits(model, device, ids, ranges, cancel_site=site, cancel_vec=baseline_vec)


def phrase_diag(
    model,
    tokenizer,
    device,
    prompt: str,
    answer: str,
    ranges: list[dict[str, int]],
    condition: str,
    site_map: dict[str, str],
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = target_token_ids(tokenizer, answer)
    cur = list(prompt_ids)
    diags = []
    for tid in ans_ids:
        logits = condition_logits(model, device, cur, ranges, condition, site_map)
        diag = logit_diag(logits, int(tid))
        diags.append(diag)
        cur.append(int(tid))
    return {
        "mean_logprob": sum(d["target_logprob"] for d in diags) / len(diags),
        "sum_logprob": sum(d["target_logprob"] for d in diags),
        "first_rank": diags[0]["target_rank"],
        "first_top1": diags[0]["target_top1"],
    }


def greedy_generate(
    model,
    tokenizer,
    device,
    prompt: str,
    ranges: list[dict[str, int]],
    condition: str,
    site_map: dict[str, str],
    max_new_tokens: int,
) -> dict[str, Any]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    new_ids = []
    records = []
    for _ in range(max_new_tokens):
        logits = condition_logits(model, device, ids, ranges, condition, site_map)
        tok = int(torch.argmax(logits).item())
        new_ids.append(tok)
        records.append({"id": tok, "text": tokenizer.decode([tok])})
        ids.append(tok)
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if "\n" in text or "." in text or ";" in text:
            break
    return {"text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(), "tokens": records}


def load_site_map(model_name: str, upstream: str) -> dict[str, str]:
    data = json.loads((PHASE729_ROOT / f"phase729_{model_name}_propagation_summary.json").read_text(encoding="utf-8"))
    sites = data["site_summary"]
    vals = [s for s in sites if s["intervention"] == upstream]
    layer_sites = [s for s in vals if s["site_kind"] == "layer_out"]
    mlp_sites = [s for s in vals if s["site_kind"] == "mlp_out"]
    if not layer_sites or not mlp_sites:
        raise ValueError(f"missing layer/mlp site for {model_name} {upstream}")
    top_layer = max(layer_sites, key=lambda s: float(s["mean_delta_norm"]))
    top_mlp = max(mlp_sites, key=lambda s: float(s["mean_delta_norm"]))
    return {
        "top_layer_out": top_layer["site"],
        "top_mlp_out": top_mlp["site"],
        "top_layer_delta": top_layer["mean_delta_norm"],
        "top_mlp_delta": top_mlp["mean_delta_norm"],
    }


def summarize(rows: list[dict[str, Any]], model_name: str, site_maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    base_rows = [r for r in rows if r["upstream"] == "baseline" and r["condition"] == "baseline"]
    base_by_case = {r["case_id"]: r for r in base_rows}
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["upstream"], row["condition"])].append(row)

    by_condition = {}
    upstream_delta: dict[str, float] = {}
    for (upstream, condition), vals in sorted(by_key.items()):
        if upstream == "baseline":
            mean_delta = 0.0
            changed = 0.0
            hit_drop = 0.0
        else:
            mean_delta = sum(v["mean_logprob_delta"] for v in vals) / len(vals)
            changed = sum(1 for v in vals if norm_text(v["generated_text"]) != norm_text(base_by_case[v["case_id"]]["generated_text"])) / len(vals)
            hit_drop = sum(1 for v in vals if base_by_case[v["case_id"]]["hit"] and not v["hit"]) / len(vals)
        if condition == "upstream_only":
            upstream_delta[upstream] = mean_delta
        by_condition[f"{upstream}|{condition}"] = {
            "n": len(vals),
            "mean_logprob_delta": mean_delta,
            "hit_rate": sum(1 for v in vals if v["hit"]) / len(vals),
            "changed_rate_vs_baseline": changed,
            "hit_drop_rate_vs_baseline": hit_drop,
            "first_rank_delta": sum(v["first_rank_delta"] for v in vals) / len(vals),
            "outputs": Counter(norm_text(v["generated_text"]) for v in vals).most_common(10),
        }

    for key, rec in by_condition.items():
        upstream, condition = key.split("|", 1)
        if upstream == "baseline" or condition == "upstream_only":
            rec["recovery_fraction_vs_upstream"] = 0.0
            continue
        base_delta = upstream_delta.get(upstream, 0.0)
        if abs(base_delta) < 1e-9:
            rec["recovery_fraction_vs_upstream"] = 0.0
        else:
            rec["recovery_fraction_vs_upstream"] = (rec["mean_logprob_delta"] - base_delta) / (0.0 - base_delta)

    return {
        "phase": 730,
        "title": "Downstream Propagation Node Cancellation",
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(base_by_case),
        "n_rows": len(rows),
        "site_maps": site_maps,
        "by_condition": by_condition,
    }


def run_model(args) -> dict[str, Any]:
    cases = category_cases(args.max_cases)
    model, tokenizer, device = load_model(args.model)
    rows: list[dict[str, Any]] = []
    try:
        interventions = build_interventions(model, args.model)
        site_maps = {up: load_site_map(args.model, up) for up in UPSTREAMS}
        log(f"{args.model}: cases={len(cases)}, site_maps={json.dumps(site_maps, ensure_ascii=False)}")
        for idx, case in enumerate(cases, 1):
            prompt = prompt_for(case)
            baseline_diag = phrase_diag(model, tokenizer, device, prompt, case["answer"], [], "upstream_only", {})
            baseline_gen = greedy_generate(model, tokenizer, device, prompt, [], "upstream_only", {}, args.max_new_tokens)
            rows.append(
                {
                    "model": args.model,
                    "case_id": case["case_id"],
                    "prompt_type": case["prompt_type"],
                    "object": case["object"],
                    "object_group": case["object_group"],
                    "answer": case["answer"],
                    "upstream": "baseline",
                    "condition": "baseline",
                    "cancel_site": None,
                    "generated_text": baseline_gen["text"],
                    "hit": hit_answer(baseline_gen["text"], case["answer"]),
                    "mean_logprob": baseline_diag["mean_logprob"],
                    "mean_logprob_delta": 0.0,
                    "first_rank": baseline_diag["first_rank"],
                    "first_rank_delta": 0,
                    "first_top1": baseline_diag["first_top1"],
                }
            )
            for upstream in UPSTREAMS:
                ranges = interventions[upstream]
                site_map = site_maps[upstream]
                for condition in CONDITIONS:
                    diag = phrase_diag(model, tokenizer, device, prompt, case["answer"], ranges, condition, site_map)
                    generated = greedy_generate(model, tokenizer, device, prompt, ranges, condition, site_map, args.max_new_tokens)
                    cancel_site = None
                    if condition == "cancel_top_layer_out":
                        cancel_site = site_map["top_layer_out"]
                    elif condition == "cancel_top_mlp_out":
                        cancel_site = site_map["top_mlp_out"]
                    rows.append(
                        {
                            "model": args.model,
                            "case_id": case["case_id"],
                            "prompt_type": case["prompt_type"],
                            "object": case["object"],
                            "object_group": case["object_group"],
                            "answer": case["answer"],
                            "upstream": upstream,
                            "condition": condition,
                            "cancel_site": cancel_site,
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

    summary = summarize(rows, args.model, site_maps)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT_ROOT / f"phase730_{args.model}_cancellation_rows.jsonl", rows)
    write_json(OUT_ROOT / f"phase730_{args.model}_cancellation_summary.json", summary)
    print(json.dumps({"model": args.model, "n_cases": summary["n_cases"], "by_condition": summary["by_condition"]}, ensure_ascii=False, indent=2), flush=True)
    return summary


def write_cross_summary() -> dict[str, Any]:
    summaries = []
    for model in MODELS:
        path = OUT_ROOT / f"phase730_{model}_cancellation_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "phase": 730,
        "title": "Downstream Propagation Node Cancellation",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": [s["model"] for s in summaries],
        "status": "complete" if len(summaries) == len(MODELS) else "partial",
        "evidence_type": "downstream baseline-state cancellation of propagation sites",
        "by_model": {s["model"]: {"n_cases": s["n_cases"], "site_maps": s["site_maps"], "by_condition": s["by_condition"]} for s in summaries},
    }
    write_json(OUT_ROOT / "phase730_cross_model_summary.json", payload)
    lines = [
        "# Phase 730 Downstream Propagation Node Cancellation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Models: `{payload['models']}`",
        "- Evidence type: downstream propagation node cancellation.",
        "",
        "| model | condition | mean_delta | recovery | hit_rate | changed | hit_drop | rank_delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, item in payload["by_model"].items():
        for key, rec in item["by_condition"].items():
            if key == "baseline|baseline":
                continue
            lines.append(
                f"| {model} | {key} | {rec['mean_logprob_delta']:.4f} | {rec['recovery_fraction_vs_upstream']:.3f} | "
                f"{rec['hit_rate']:.3f} | {rec['changed_rate_vs_baseline']:.3f} | "
                f"{rec['hit_drop_rate_vs_baseline']:.3f} | {rec['first_rank_delta']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Strict Interpretation",
            "",
            "- Cancellation toward baseline is a mediation test, not a complete circuit proof.",
            "- Positive recovery means the downstream site carries part of the upstream perturbation effect.",
            "- No recovery means the site is visible in propagation but not sufficient as a bottleneck.",
            "",
        ]
    )
    (OUT_ROOT / "phase730_cross_model_summary.md").write_text("\n".join(lines), encoding="utf-8")
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
