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
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from phase68_object_attribute_natural_exchange import (  # noqa: E402
    encode,
    get_module,
    get_positions,
    load_model,
    parse_csv,
)
from phase70_object_relation_value_closure import (  # noqa: E402
    build_items,
    parse_layer_pairs,
    pick_control,
)


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def candidate_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
    return list(ids)


def capture_state(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer: Any,
    module_name: str,
    prompt: str,
    max_length: int,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}
    module = get_module(layer, module_name)

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["h"] = tensor.detach().float().cpu()

    handle = module.register_forward_hook(hook_fn)
    try:
        inputs = encode(tokenizer, device, prompt, max_length)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["h"][0]


def fullseq_logprob(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    value: str,
    max_length: int,
    module_name: str | None = None,
    destroy_layer: int | None = None,
    restore_layer: int | None = None,
    token_pos: int | None = None,
    destroy_state: torch.Tensor | None = None,
    restore_state: torch.Tensor | None = None,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    value_ids = candidate_ids(tokenizer, value)
    if not value_ids:
        return float("-inf")
    full_ids = prompt_ids + value_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []

    def make_replace_hook(replacement_cpu: torch.Tensor):
        def hook_fn(_module: Any, _inputs: Any, output: Any):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            pos = int(token_pos) if token_pos is not None and token_pos >= 0 else hs.shape[1] + int(token_pos)
            if 0 <= pos < hs.shape[1]:
                hs[0, pos, :] = replacement_cpu.to(device=hs.device, dtype=hs.dtype)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        return hook_fn

    try:
        if module_name is not None and destroy_layer is not None and token_pos is not None and destroy_state is not None:
            handles.append(get_module(layers[destroy_layer], module_name).register_forward_hook(make_replace_hook(destroy_state)))
        if module_name is not None and restore_layer is not None and token_pos is not None and restore_state is not None:
            handles.append(get_module(layers[restore_layer], module_name).register_forward_hook(make_replace_hook(restore_state)))
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
    finally:
        for h in handles:
            h.remove()

    start = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(value_ids):
        logit_pos = start + i - 1
        if logit_pos < 0 or logit_pos >= log_probs.shape[0]:
            return float("-inf")
        total += float(log_probs[logit_pos, tok].detach().cpu())
    return total


def stats_from_scores(scores: dict[str, float], target: str, distractors: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    comp = max((scores[d] for d in distractors if d in scores), default=-1e9)
    return {
        "margin": float(scores.get(target, -1e9) - comp),
        "rank": {name: i + 1 for i, (name, _v) in enumerate(ordered)}.get(target),
        "top": ordered[0][0] if ordered else None,
        "scores": scores,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(xs: list[float]) -> float:
        return float(mean(xs)) if xs else 0.0

    def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
        eligible = [v for v in vals if v["clean_target_rank"] == 1]
        return {
            "n": len(vals),
            "eligible_n": len(eligible),
            "destroy_drop": avg([float(v["destroy_drop"]) for v in vals]),
            "restore_gain": avg([float(v["restore_gain"]) for v in vals]),
            "restore_to_clean_gap": avg([float(v["restore_to_clean_gap"]) for v in vals]),
            "eligible_destroy_drop": avg([float(v["destroy_drop"]) for v in eligible]),
            "eligible_restore_gain": avg([float(v["restore_gain"]) for v in eligible]),
            "eligible_restore_to_clean_gap": avg([float(v["restore_to_clean_gap"]) for v in eligible]),
            "clean_top1": avg([1.0 if v["clean_target_rank"] == 1 else 0.0 for v in vals]),
            "destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in vals]),
            "restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in vals]),
            "eligible_destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in eligible]),
            "eligible_restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in eligible]),
        }

    by_path: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_relation_path: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dl, rl, pos, rel = int(row["destroy_layer"]), int(row["restore_layer"]), str(row["position"]), str(row["relation"])
        by_path[(dl, rl, pos)].append(row)
        by_relation[rel].append(row)
        by_relation_path[(rel, dl, rl, pos)].append(row)
    return {
        "by_path": {f"L{dl}->L{rl}:{pos}": group_summary(vals) for (dl, rl, pos), vals in by_path.items()},
        "by_relation": {rel: group_summary(vals) for rel, vals in by_relation.items()},
        "by_relation_path": {f"{rel}:L{dl}->L{rl}:{pos}": group_summary(vals) for (rel, dl, rl, pos), vals in by_relation_path.items()},
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE72_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    positions = parse_csv(args.positions)
    items = build_items(args.max_items, parse_csv(args.relations), parse_csv(args.frames))
    log(f"Phase72 model={args.model} items={len(items)} layer_pairs={layer_pairs} positions={positions}")

    results: dict[str, Any] = {
        "phase": 72,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "object_relation_value_fullseq_closure",
        "layer_pairs": layer_pairs,
        "module": args.module,
        "positions": positions,
        "relations": sorted({x["relation"] for x in items}),
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    t0 = time.time()

    for destroy_layer, restore_layer in layer_pairs:
        for idx, item in enumerate(items):
            control = pick_control(items, idx)
            clean_pos = get_positions(tokenizer, item["clean_prompt"], item["object"])
            control_pos = get_positions(tokenizer, control["clean_prompt"], control["object"])
            h_control_d = capture_state(model, tokenizer, device, layers[destroy_layer], args.module, control["clean_prompt"], args.max_length)
            h_clean_r = capture_state(model, tokenizer, device, layers[restore_layer], args.module, item["clean_prompt"], args.max_length)
            values = [item["target"]] + item["distractors"]
            clean_scores = {v: fullseq_logprob(model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length) for v in values}
            clean_stats = stats_from_scores(clean_scores, item["target"], item["distractors"])

            for pos_name in positions:
                sp = clean_pos.get(pos_name)
                cp = control_pos.get(pos_name)
                if sp is None or cp is None:
                    continue
                destroy_scores = {
                    v: fullseq_logprob(
                        model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length,
                        args.module, destroy_layer, None, int(sp), h_control_d[int(cp)], None
                    )
                    for v in values
                }
                restore_scores = {
                    v: fullseq_logprob(
                        model, tokenizer, device, layers, item["clean_prompt"], v, args.max_length,
                        args.module, destroy_layer, restore_layer, int(sp), h_control_d[int(cp)], h_clean_r[int(sp)]
                    )
                    for v in values
                }
                destroy_stats = stats_from_scores(destroy_scores, item["target"], item["distractors"])
                restore_stats = stats_from_scores(restore_scores, item["target"], item["distractors"])
                results["rows"].append(
                    {
                        "destroy_layer": destroy_layer,
                        "restore_layer": restore_layer,
                        "module": args.module,
                        "position": pos_name,
                        "relation": item["relation"],
                        "frame_key": item["frame_key"],
                        "object": item["object"],
                        "target": item["target"],
                        "control_object": control["object"],
                        "control_target": control["target"],
                        "clean_margin": clean_stats["margin"],
                        "destroy_margin": destroy_stats["margin"],
                        "restore_margin": restore_stats["margin"],
                        "destroy_drop": clean_stats["margin"] - destroy_stats["margin"],
                        "restore_gain": restore_stats["margin"] - destroy_stats["margin"],
                        "restore_to_clean_gap": clean_stats["margin"] - restore_stats["margin"],
                        "clean_target_rank": clean_stats["rank"],
                        "destroy_target_rank": destroy_stats["rank"],
                        "restore_target_rank": restore_stats["rank"],
                        "clean_top": clean_stats["top"],
                        "destroy_top": destroy_stats["top"],
                        "restore_top": restore_stats["top"],
                    }
                )
            if (idx + 1) % args.progress_every == 0:
                log(f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase72_object_relation_value_fullseq_closure.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase72_object_relation_value_fullseq_closure.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--positions", default="object_first,object_last")
    parser.add_argument("--relations", default="")
    parser.add_argument("--frames", default="")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
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
