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


sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests" / "gpt5"))

from hf_probe_env import get_layers, release_loaded  # noqa: E402
from model_registry import get_model_spec  # noqa: E402
from phase68_object_attribute_natural_exchange import (  # noqa: E402
    FRAMES,
    RELATION_DATA,
    build_items,
    encode,
    first_token_id,
    get_candidate_stats,
    get_module,
    get_positions,
    load_model,
    margin_from_stats,
    parse_csv,
    pick_control,
)


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_layer_pairs(text: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"layer pair must be D-R, got {item!r}")
        a, b = item.split("-", 1)
        da, rb = int(a), int(b)
        if rb <= da:
            raise ValueError(f"restore layer must be > destroy layer, got {item!r}")
        pairs.append((da, rb))
    return pairs


def capture_activation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer: Any,
    module_name: str,
    prompt: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    module = get_module(layer, module_name)

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["h"] = tensor.detach().float().cpu()

    handle = module.register_forward_hook(hook_fn)
    try:
        inputs = encode(tokenizer, device, prompt, max_length)
        with torch.no_grad():
            out = model(**inputs)
    finally:
        handle.remove()
    return captured["h"][0], out.logits[0, -1].detach().float().cpu()


def forward_with_destroy_restore(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    module_name: str,
    prompt: str,
    destroy_layer: int,
    restore_layer: int | None,
    token_pos: int,
    destroy_state: torch.Tensor,
    restore_state: torch.Tensor | None,
    max_length: int,
) -> torch.Tensor:
    handles = []

    def make_replace_hook(replacement_cpu: torch.Tensor):
        def hook_fn(_module: Any, _inputs: Any, output: Any):
            hs = output[0].clone() if isinstance(output, tuple) else output.clone()
            pos = token_pos if token_pos >= 0 else hs.shape[1] + token_pos
            if 0 <= pos < hs.shape[1]:
                hs[0, pos, :] = replacement_cpu.to(device=hs.device, dtype=hs.dtype)
            return (hs,) + output[1:] if isinstance(output, tuple) else hs

        return hook_fn

    try:
        handles.append(get_module(layers[destroy_layer], module_name).register_forward_hook(make_replace_hook(destroy_state)))
        if restore_layer is not None and restore_state is not None:
            handles.append(get_module(layers[restore_layer], module_name).register_forward_hook(make_replace_hook(restore_state)))
        inputs = encode(tokenizer, device, prompt, max_length)
        with torch.no_grad():
            out = model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, -1].detach().float().cpu()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["destroy_layer"]), int(r["restore_layer"]), str(r["position"]))].append(r)
    out: dict[str, Any] = {}
    for key, vals in groups.items():
        dl, rl, pos = key

        def avg(xs: list[float]) -> float:
            return float(mean(xs)) if xs else 0.0

        eligible = [v for v in vals if v["clean_target_rank"] == 1]
        out[f"L{dl}->L{rl}:{pos}"] = {
            "n": len(vals),
            "eligible_n": len(eligible),
            "destroy_drop": avg([float(v["clean_margin"] - v["destroy_margin"]) for v in vals]),
            "restore_gain": avg([float(v["restore_margin"] - v["destroy_margin"]) for v in vals]),
            "restore_to_clean_gap": avg([float(v["clean_margin"] - v["restore_margin"]) for v in vals]),
            "eligible_destroy_drop": avg([float(v["clean_margin"] - v["destroy_margin"]) for v in eligible]),
            "eligible_restore_gain": avg([float(v["restore_margin"] - v["destroy_margin"]) for v in eligible]),
            "eligible_restore_to_clean_gap": avg([float(v["clean_margin"] - v["restore_margin"]) for v in eligible]),
            "clean_top1": avg([1.0 if v["clean_target_rank"] == 1 else 0.0 for v in vals]),
            "destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in vals]),
            "restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in vals]),
            "eligible_destroy_top1": avg([1.0 if v["destroy_target_rank"] == 1 else 0.0 for v in eligible]),
            "eligible_restore_top1": avg([1.0 if v["restore_target_rank"] == 1 else 0.0 for v in eligible]),
        }
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE69_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_pairs = parse_layer_pairs(args.layer_pairs)
    positions = parse_csv(args.positions)
    module_name = args.module
    items = build_items(args.max_items, parse_csv(args.frames))
    log(f"Phase69 model={args.model} items={len(items)} layer_pairs={layer_pairs} module={module_name} positions={positions}")

    all_values = sorted({row["target"] for row in items} | {d for row in items for d in row["distractors"]})
    value_ids = {v: first_token_id(tokenizer, v) for v in all_values}
    missing_values = sorted(v for v, tid in value_ids.items() if tid is None)
    if missing_values:
        log(f"Skipping multi-token candidates: {missing_values}")

    results: dict[str, Any] = {
        "phase": 69,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "layer_pairs": layer_pairs,
        "module": module_name,
        "positions": positions,
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }

    t0 = time.time()
    for destroy_layer, restore_layer in layer_pairs:
        for idx, item in enumerate(items):
            control = pick_control(items, idx)
            candidates = [item["target"]] + item["distractors"]
            candidate_ids = {v: value_ids[v] for v in candidates if value_ids.get(v) is not None}
            if item["target"] not in candidate_ids or len(candidate_ids) < 2:
                continue

            clean_pos = get_positions(tokenizer, item["clean_prompt"], item["object"])
            control_pos = get_positions(tokenizer, control["clean_prompt"], control["object"])

            h_clean_d, clean_logits = capture_activation(
                model, tokenizer, device, layers[destroy_layer], module_name, item["clean_prompt"], args.max_length
            )
            h_control_d, _ = capture_activation(
                model, tokenizer, device, layers[destroy_layer], module_name, control["clean_prompt"], args.max_length
            )
            h_clean_r, _ = capture_activation(
                model, tokenizer, device, layers[restore_layer], module_name, item["clean_prompt"], args.max_length
            )

            clean_stats = get_candidate_stats(clean_logits, candidate_ids)
            clean_margin = margin_from_stats(clean_stats, item["target"], item["distractors"])

            for pos_name in positions:
                sp = clean_pos.get(pos_name)
                cp = control_pos.get(pos_name)
                if sp is None or cp is None:
                    continue
                destroy_logits = forward_with_destroy_restore(
                    model,
                    tokenizer,
                    device,
                    layers,
                    module_name,
                    item["clean_prompt"],
                    destroy_layer,
                    None,
                    int(sp),
                    h_control_d[int(cp)],
                    None,
                    args.max_length,
                )
                restore_logits = forward_with_destroy_restore(
                    model,
                    tokenizer,
                    device,
                    layers,
                    module_name,
                    item["clean_prompt"],
                    destroy_layer,
                    restore_layer,
                    int(sp),
                    h_control_d[int(cp)],
                    h_clean_r[int(sp)],
                    args.max_length,
                )
                destroy_stats = get_candidate_stats(destroy_logits, candidate_ids)
                restore_stats = get_candidate_stats(restore_logits, candidate_ids)
                destroy_margin = margin_from_stats(destroy_stats, item["target"], item["distractors"])
                restore_margin = margin_from_stats(restore_stats, item["target"], item["distractors"])
                results["rows"].append(
                    {
                        "destroy_layer": destroy_layer,
                        "restore_layer": restore_layer,
                        "module": module_name,
                        "position": pos_name,
                        "relation": item["relation"],
                        "object": item["object"],
                        "target": item["target"],
                        "control_object": control["object"],
                        "control_target": control["target"],
                        "frame_key": item["frame_key"],
                        "clean_margin": clean_margin,
                        "destroy_margin": destroy_margin,
                        "restore_margin": restore_margin,
                        "destroy_drop": clean_margin - destroy_margin,
                        "restore_gain": restore_margin - destroy_margin,
                        "restore_to_clean_gap": clean_margin - restore_margin,
                        "clean_target_rank": clean_stats["rank"].get(item["target"]),
                        "destroy_target_rank": destroy_stats["rank"].get(item["target"]),
                        "restore_target_rank": restore_stats["rank"].get(item["target"]),
                        "clean_top": clean_stats["top"],
                        "destroy_top": destroy_stats["top"],
                        "restore_top": restore_stats["top"],
                    }
                )

            if (idx + 1) % args.progress_every == 0:
                log(
                    f"pair={destroy_layer}->{restore_layer} item={idx + 1}/{len(items)} "
                    f"rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s"
                )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase69_object_attribute_destroy_restore.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase69_object_attribute_destroy_restore.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layer-pairs", required=True)
    parser.add_argument("--module", default="resid_out")
    parser.add_argument("--positions", default="object_first,object_last,last")
    parser.add_argument("--frames", default="the,this,that,a")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
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
