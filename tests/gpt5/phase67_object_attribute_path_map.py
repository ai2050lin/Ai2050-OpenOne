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


FRAMES = [
    ("the", "The {obj} is"),
    ("this", "This {obj} is"),
    ("that", "That {obj} is"),
    ("a", "A {obj} is"),
]

CORRUPT_OBJECT = {
    "the": "item",
    "this": "item",
    "that": "item",
    "a": "thing",
}

RELATION_DATA = {
    "color": [
        ("apple", "red", ["blue", "white", "black", "small"]),
        ("cherry", "red", ["blue", "white", "black", "small"]),
        ("sky", "blue", ["red", "green", "black", "tiny"]),
        ("grass", "green", ["red", "blue", "white", "tiny"]),
        ("snow", "white", ["black", "red", "green", "large"]),
        ("coal", "black", ["white", "blue", "green", "small"]),
    ],
    "moisture": [
        ("ocean", "wet", ["dry", "arid", "small", "red"]),
        ("rain", "wet", ["dry", "arid", "large", "black"]),
        ("river", "wet", ["dry", "arid", "tiny", "green"]),
        ("desert", "dry", ["wet", "moist", "blue", "small"]),
        ("sand", "dry", ["wet", "moist", "red", "large"]),
        ("dust", "dry", ["wet", "moist", "white", "big"]),
    ],
    "size": [
        ("elephant", "large", ["tiny", "small", "red", "dry"]),
        ("mountain", "large", ["tiny", "small", "blue", "wet"]),
        ("whale", "large", ["tiny", "small", "green", "dry"]),
        ("ant", "tiny", ["large", "big", "white", "wet"]),
        ("grain", "tiny", ["large", "big", "black", "moist"]),
        ("pin", "tiny", ["large", "big", "red", "dry"]),
    ],
    "temperature": [
        ("fire", "hot", ["cold", "frozen", "blue", "small"]),
        ("flame", "hot", ["cold", "frozen", "green", "wet"]),
        ("ice", "cold", ["hot", "warm", "red", "large"]),
        ("snow", "cold", ["hot", "warm", "black", "big"]),
        ("tea", "warm", ["frozen", "cold", "tiny", "blue"]),
        ("soup", "warm", ["frozen", "cold", "small", "green"]),
    ],
    "texture": [
        ("silk", "smooth", ["rough", "sharp", "wet", "large"]),
        ("glass", "smooth", ["rough", "soft", "tiny", "dry"]),
        ("sandpaper", "rough", ["smooth", "soft", "blue", "wet"]),
        ("rock", "rough", ["smooth", "soft", "red", "tiny"]),
        ("pillow", "soft", ["sharp", "rough", "black", "hot"]),
        ("knife", "sharp", ["soft", "smooth", "white", "cold"]),
    ],
    "material": [
        ("spoon", "metal", ["wooden", "glass", "wet", "tiny"]),
        ("coin", "metal", ["wooden", "cloth", "blue", "soft"]),
        ("table", "wooden", ["metal", "cloth", "cold", "tiny"]),
        ("shirt", "cloth", ["metal", "wooden", "hot", "sharp"]),
        ("window", "glass", ["cloth", "wooden", "soft", "dry"]),
        ("bottle", "glass", ["cloth", "wooden", "rough", "small"]),
    ],
}


def log(msg: str = "") -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_model(model_name: str, attn_impls: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model_spec(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        spec.local_dir, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    errors: list[str] = []
    model = None
    for impl in [x.strip() for x in attn_impls.split(",") if x.strip()]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: os.environ.get("PHASE67_MAX_GPU_MEMORY", "22GiB"), "cpu": os.environ.get("PHASE67_MAX_CPU_MEMORY", "96GiB")},
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation=impl,
            )
            log(f"Loaded {model_name} with attn_impl={impl}")
            break
        except Exception as exc:
            errors.append(f"{impl}: {exc}")
            log(f"Failed loading {model_name} with {impl}: {exc}")
    if model is None:
        raise RuntimeError("failed to load model: " + " | ".join(errors))
    model.eval()
    return model, tokenizer, next(model.parameters()).device


def get_module(layer: Any, module_name: str) -> Any:
    if module_name == "resid_out":
        return layer
    if module_name == "attn_out":
        return getattr(layer, "self_attn", None) or getattr(layer, "attention")
    if module_name == "mlp_out":
        return getattr(layer, "mlp")
    raise ValueError(f"unknown module_name={module_name}")


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def first_token_id(tokenizer: Any, text: str) -> int | None:
    ids = token_ids(tokenizer, " " + text)
    if len(ids) == 1:
        return int(ids[0])
    ids = token_ids(tokenizer, text)
    return int(ids[0]) if len(ids) == 1 else None


def find_subseq(seq: list[int], subseq: list[int]) -> tuple[int, int] | None:
    if not subseq:
        return None
    for i in range(0, len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            return i, i + len(subseq) - 1
    return None


def build_items(max_items: int | None, frames: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relation, examples in RELATION_DATA.items():
        for obj, target, distractors in examples:
            for frame_key, frame in FRAMES:
                if frames and frame_key not in frames:
                    continue
                corrupt_obj = CORRUPT_OBJECT[frame_key]
                rows.append(
                    {
                        "relation": relation,
                        "object": obj,
                        "target": target,
                        "distractors": distractors,
                        "frame_key": frame_key,
                        "clean_prompt": frame.format(obj=obj),
                        "corrupt_prompt": frame.format(obj=corrupt_obj),
                        "corrupt_object": corrupt_obj,
                    }
                )
    return rows[:max_items] if max_items else rows


def encode(tokenizer: Any, device: torch.device, prompt: str, max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in batch.items()}


def get_candidate_stats(logits: torch.Tensor, candidate_ids: dict[str, int]) -> dict[str, Any]:
    vals = {name: float(logits[cid].float().detach().cpu()) for name, cid in candidate_ids.items()}
    ordered = sorted(vals.items(), key=lambda x: x[1], reverse=True)
    return {
        "logits": vals,
        "rank": {name: i + 1 for i, (name, _v) in enumerate(ordered)},
        "top": ordered[0][0] if ordered else None,
    }


def margin_from_stats(stats: dict[str, Any], target: str, distractors: list[str]) -> float:
    vals = stats["logits"]
    if target not in vals:
        return 0.0
    comp = max((vals.get(d, -1e9) for d in distractors if d in vals), default=-1e9)
    return float(vals[target] - comp)


def capture_activation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer: Any,
    module_name: str,
    prompt: str,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
    return captured["h"][0], out.logits[0, -1].detach().float().cpu(), inputs


def forward_with_token_delta(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer: Any,
    module_name: str,
    prompt: str,
    target_pos: int,
    delta_cpu: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    module = get_module(layer, module_name)

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
        pos = target_pos if target_pos >= 0 else hs.shape[1] + target_pos
        if 0 <= pos < hs.shape[1]:
            hs[0, pos, :] += delta_cpu.to(device=hs.device, dtype=hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    handle = module.register_forward_hook(hook_fn)
    try:
        inputs = encode(tokenizer, device, prompt, max_length)
        with torch.no_grad():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


def get_positions(tokenizer: Any, prompt: str, object_text: str) -> dict[str, int | None]:
    ids = token_ids(tokenizer, prompt)
    match = None
    for variant in (object_text, " " + object_text):
        obj_ids = token_ids(tokenizer, variant)
        match = find_subseq(ids, obj_ids)
        if match is not None:
            break
    return {
        "object_first": match[0] if match else None,
        "object_last": match[1] if match else None,
        "last": len(ids) - 1 if ids else None,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE67_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    items = build_items(args.max_items, parse_csv(args.frames))
    modules = parse_csv(args.modules)
    positions = parse_csv(args.positions)
    log(f"Phase67 model={args.model} items={len(items)} layers={args.layers} modules={modules} positions={positions}")

    all_values = sorted({row["target"] for row in items} | {d for row in items for d in row["distractors"]})
    value_ids = {v: first_token_id(tokenizer, v) for v in all_values}
    missing_values = sorted(v for v, tid in value_ids.items() if tid is None)
    if missing_values:
        log(f"Skipping multi-token candidates: {missing_values}")

    results: dict[str, Any] = {
        "phase": 67,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "layers": args.layers,
        "modules": modules,
        "positions": positions,
        "num_items": len(items),
        "items": items,
        "rows": [],
        "summary": {},
    }

    t0 = time.time()
    for li in args.layers:
        layer = layers[li]
        for module_name in modules:
            for idx, item in enumerate(items):
                candidates = [item["target"]] + item["distractors"]
                candidate_ids = {v: value_ids[v] for v in candidates if value_ids.get(v) is not None}
                if item["target"] not in candidate_ids or len(candidate_ids) < 2:
                    continue

                clean_prompt = item["clean_prompt"]
                corrupt_prompt = item["corrupt_prompt"]
                clean_pos = get_positions(tokenizer, clean_prompt, item["object"])
                corrupt_pos = get_positions(tokenizer, corrupt_prompt, item["corrupt_object"])

                h_clean, clean_logits, _ = capture_activation(model, tokenizer, device, layer, module_name, clean_prompt, args.max_length)
                h_corrupt, corrupt_logits, _ = capture_activation(model, tokenizer, device, layer, module_name, corrupt_prompt, args.max_length)
                clean_stats = get_candidate_stats(clean_logits, candidate_ids)
                corrupt_stats = get_candidate_stats(corrupt_logits, candidate_ids)
                clean_margin = margin_from_stats(clean_stats, item["target"], item["distractors"])
                corrupt_margin = margin_from_stats(corrupt_stats, item["target"], item["distractors"])

                for pos_name in positions:
                    sp = clean_pos.get(pos_name)
                    tp = corrupt_pos.get(pos_name)
                    if sp is None or tp is None:
                        continue
                    delta = h_clean[sp] - h_corrupt[tp]
                    patch_logits = forward_with_token_delta(
                        model,
                        tokenizer,
                        device,
                        layer,
                        module_name,
                        corrupt_prompt,
                        int(tp),
                        delta,
                        args.max_length,
                    )
                    patch_stats = get_candidate_stats(patch_logits, candidate_ids)
                    patch_margin = margin_from_stats(patch_stats, item["target"], item["distractors"])
                    denom = abs(clean_margin - corrupt_margin) + 1e-6
                    progress = (patch_margin - corrupt_margin) / denom
                    row = {
                        "layer": li,
                        "module": module_name,
                        "position": pos_name,
                        "relation": item["relation"],
                        "object": item["object"],
                        "target": item["target"],
                        "frame_key": item["frame_key"],
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "patch_margin": patch_margin,
                        "margin_progress": float(progress),
                        "clean_top": clean_stats["top"],
                        "corrupt_top": corrupt_stats["top"],
                        "patch_top": patch_stats["top"],
                        "clean_target_rank": clean_stats["rank"].get(item["target"]),
                        "corrupt_target_rank": corrupt_stats["rank"].get(item["target"]),
                        "patch_target_rank": patch_stats["rank"].get(item["target"]),
                        "delta_norm": float(delta.norm().item()),
                    }
                    results["rows"].append(row)

                if (idx + 1) % args.progress_every == 0:
                    log(
                        f"layer={li} module={module_name} item={idx + 1}/{len(items)} "
                        f"rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s"
                    )

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase67_object_attribute_path_map.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase67_object_attribute_path_map.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["layer"]), str(r["module"]), str(r["position"]))].append(r)
    out: dict[str, Any] = {}
    for key, vals in groups.items():
        layer, module, pos = key
        progresses = [float(v["margin_progress"]) for v in vals]
        flips = [1 if v["patch_target_rank"] == 1 and v["corrupt_target_rank"] != 1 else 0 for v in vals]
        improves = [1 if float(v["patch_margin"]) > float(v["corrupt_margin"]) else 0 for v in vals]
        clean_good = [1 if v["clean_target_rank"] == 1 else 0 for v in vals]
        corrupt_bad = [1 if v["corrupt_target_rank"] != 1 else 0 for v in vals]
        out[f"L{layer}:{module}:{pos}"] = {
            "n": len(vals),
            "mean_progress": float(mean(progresses)) if progresses else 0.0,
            "max_progress": float(max(progresses)) if progresses else 0.0,
            "improve_rate": float(mean(improves)) if improves else 0.0,
            "rank_flip_rate": float(mean(flips)) if flips else 0.0,
            "clean_top1_rate": float(mean(clean_good)) if clean_good else 0.0,
            "corrupt_not_top1_rate": float(mean(corrupt_bad)) if corrupt_bad else 0.0,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layers", required=True)
    parser.add_argument("--modules", default="resid_out,attn_out,mlp_out")
    parser.add_argument("--positions", default="object_first,object_last,last")
    parser.add_argument("--frames", default="the,this,that,a")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=24)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    args.layers = parse_layers(args.layers)
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
