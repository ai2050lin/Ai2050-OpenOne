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


RELATION_DATA: dict[str, list[tuple[str, str, list[str]]]] = {
    "color": [
        ("apple", "red", ["blue", "white", "black", "small"]),
        ("cherry", "red", ["blue", "green", "black", "tiny"]),
        ("rose", "red", ["blue", "white", "black", "large"]),
        ("sky", "blue", ["red", "green", "black", "tiny"]),
        ("ocean", "blue", ["red", "white", "black", "small"]),
        ("grass", "green", ["red", "blue", "white", "tiny"]),
        ("leaf", "green", ["red", "blue", "black", "large"]),
        ("snow", "white", ["black", "red", "green", "large"]),
        ("cloud", "white", ["black", "red", "green", "tiny"]),
        ("coal", "black", ["white", "blue", "green", "small"]),
    ],
    "moisture": [
        ("ocean", "wet", ["dry", "small", "red", "black"]),
        ("rain", "wet", ["dry", "large", "white", "green"]),
        ("river", "wet", ["dry", "tiny", "red", "black"]),
        ("lake", "wet", ["dry", "small", "white", "green"]),
        ("desert", "dry", ["wet", "blue", "small", "red"]),
        ("sand", "dry", ["wet", "green", "large", "black"]),
        ("dust", "dry", ["wet", "white", "big", "blue"]),
        ("powder", "dry", ["wet", "red", "large", "green"]),
    ],
    "size": [
        ("elephant", "large", ["tiny", "small", "red", "dry"]),
        ("mountain", "large", ["tiny", "small", "blue", "wet"]),
        ("whale", "large", ["tiny", "small", "green", "dry"]),
        ("planet", "large", ["tiny", "small", "red", "cold"]),
        ("ant", "tiny", ["large", "big", "white", "wet"]),
        ("grain", "tiny", ["large", "big", "black", "wet"]),
        ("pin", "tiny", ["large", "big", "red", "dry"]),
        ("seed", "tiny", ["large", "big", "blue", "hot"]),
    ],
    "temperature": [
        ("fire", "hot", ["cold", "blue", "small", "wet"]),
        ("flame", "hot", ["cold", "green", "tiny", "dry"]),
        ("lava", "hot", ["cold", "white", "small", "wet"]),
        ("ice", "cold", ["hot", "red", "large", "dry"]),
        ("snow", "cold", ["hot", "black", "big", "wet"]),
        ("freezer", "cold", ["hot", "green", "tiny", "dry"]),
        ("tea", "warm", ["cold", "tiny", "blue", "dry"]),
        ("soup", "warm", ["cold", "small", "green", "dry"]),
    ],
    "texture": [
        ("silk", "smooth", ["rough", "sharp", "wet", "large"]),
        ("glass", "smooth", ["rough", "soft", "tiny", "dry"]),
        ("marble", "smooth", ["rough", "soft", "red", "wet"]),
        ("sandpaper", "rough", ["smooth", "soft", "blue", "wet"]),
        ("rock", "rough", ["smooth", "soft", "red", "tiny"]),
        ("bark", "rough", ["smooth", "soft", "white", "wet"]),
        ("pillow", "soft", ["sharp", "rough", "black", "hot"]),
        ("knife", "sharp", ["soft", "smooth", "white", "cold"]),
    ],
    "material": [
        ("spoon", "metal", ["wooden", "glass", "wet", "tiny"]),
        ("coin", "metal", ["wooden", "cloth", "blue", "soft"]),
        ("key", "metal", ["wooden", "cloth", "red", "soft"]),
        ("table", "wooden", ["metal", "cloth", "cold", "tiny"]),
        ("chair", "wooden", ["metal", "cloth", "wet", "sharp"]),
        ("shirt", "cloth", ["metal", "wooden", "hot", "sharp"]),
        ("window", "glass", ["cloth", "wooden", "soft", "dry"]),
        ("bottle", "glass", ["cloth", "wooden", "rough", "small"]),
    ],
    "taste": [
        ("sugar", "sweet", ["sour", "bitter", "metal", "large"]),
        ("honey", "sweet", ["sour", "bitter", "wooden", "cold"]),
        ("lemon", "sour", ["sweet", "bitter", "metal", "large"]),
        ("vinegar", "sour", ["sweet", "bitter", "wooden", "hot"]),
        ("coffee", "bitter", ["sweet", "sour", "glass", "tiny"]),
        ("cocoa", "bitter", ["sweet", "sour", "metal", "cold"]),
    ],
    "weight": [
        ("feather", "light", ["heavy", "large", "metal", "wet"]),
        ("balloon", "light", ["heavy", "rough", "wooden", "cold"]),
        ("brick", "heavy", ["light", "soft", "blue", "wet"]),
        ("anvil", "heavy", ["light", "tiny", "cloth", "cold"]),
        ("stone", "heavy", ["light", "smooth", "red", "hot"]),
        ("paper", "light", ["heavy", "sharp", "metal", "wet"]),
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
    model = None
    errors: list[str] = []
    for impl in [x.strip() for x in attn_impls.split(",") if x.strip()]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.local_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory={0: os.environ.get("PHASE68_MAX_GPU_MEMORY", "22GiB"), "cpu": os.environ.get("PHASE68_MAX_CPU_MEMORY", "96GiB")},
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


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def get_module(layer: Any, module_name: str) -> Any:
    if module_name == "resid_out":
        return layer
    if module_name == "attn_out":
        return getattr(layer, "self_attn", None) or getattr(layer, "attention")
    if module_name == "mlp_out":
        return getattr(layer, "mlp")
    raise ValueError(f"unknown module_name={module_name}")


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


def get_positions(tokenizer: Any, prompt: str, object_text: str) -> dict[str, int | None]:
    ids = token_ids(tokenizer, prompt)
    match = None
    for variant in (object_text, " " + object_text):
        match = find_subseq(ids, token_ids(tokenizer, variant))
        if match is not None:
            break
    return {
        "object_first": match[0] if match else None,
        "object_last": match[1] if match else None,
        "last": len(ids) - 1 if ids else None,
    }


def encode(tokenizer: Any, device: torch.device, prompt: str, max_length: int) -> dict[str, torch.Tensor]:
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    return {k: v.to(device) for k, v in batch.items()}


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


def pick_control(items: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    item = items[idx]
    same_relation = [
        x for x in items
        if x["relation"] == item["relation"]
        and x["target"] != item["target"]
        and x["frame_key"] == item["frame_key"]
    ]
    if same_relation:
        return same_relation[idx % len(same_relation)]
    any_other = [x for x in items if x["target"] != item["target"] and x["frame_key"] == item["frame_key"]]
    return any_other[idx % len(any_other)]


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
    comp = max((vals.get(d, -1e9) for d in distractors if d in vals), default=-1e9)
    return float(vals.get(target, 0.0) - comp)


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


def forward_with_replacement(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer: Any,
    module_name: str,
    prompt: str,
    target_pos: int,
    replacement_cpu: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    module = get_module(layer, module_name)

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0].clone() if isinstance(output, tuple) else output.clone()
        pos = target_pos if target_pos >= 0 else hs.shape[1] + target_pos
        if 0 <= pos < hs.shape[1]:
            hs[0, pos, :] = replacement_cpu.to(device=hs.device, dtype=hs.dtype)
        return (hs,) + output[1:] if isinstance(output, tuple) else hs

    handle = module.register_forward_hook(hook_fn)
    try:
        inputs = encode(tokenizer, device, prompt, max_length)
        with torch.no_grad():
            out = model(**inputs)
    finally:
        handle.remove()
    return out.logits[0, -1].detach().float().cpu()


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(int(r["layer"]), str(r["module"]), str(r["position"]))].append(r)
    out: dict[str, Any] = {}
    for key, vals in groups.items():
        layer, module, pos = key

        def avg(xs: list[float]) -> float:
            return float(mean(xs)) if xs else 0.0

        eligible = [v for v in vals if v["clean_target_rank"] == 1 and v["corrupt_target_rank"] != 1]
        out[f"L{layer}:{module}:{pos}"] = {
            "n": len(vals),
            "eligible_n": len(eligible),
            "correct_delta": avg([float(v["correct_margin"] - v["corrupt_margin"]) for v in vals]),
            "control_delta": avg([float(v["control_margin"] - v["corrupt_margin"]) for v in vals]),
            "net_delta": avg([float(v["correct_margin"] - v["control_margin"]) for v in vals]),
            "correct_flip_rate": avg([1.0 if v["correct_target_rank"] == 1 and v["corrupt_target_rank"] != 1 else 0.0 for v in vals]),
            "control_flip_rate": avg([1.0 if v["control_target_rank"] == 1 and v["corrupt_target_rank"] != 1 else 0.0 for v in vals]),
            "eligible_correct_top1": avg([1.0 if v["correct_target_rank"] == 1 else 0.0 for v in eligible]),
            "eligible_control_top1": avg([1.0 if v["control_target_rank"] == 1 else 0.0 for v in eligible]),
            "eligible_net_delta": avg([float(v["correct_margin"] - v["control_margin"]) for v in eligible]),
        }
    return out


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE68_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    items = build_items(args.max_items, parse_csv(args.frames))
    modules = parse_csv(args.modules)
    positions = parse_csv(args.positions)
    log(f"Phase68 model={args.model} items={len(items)} layers={args.layers} modules={modules} positions={positions}")

    all_values = sorted({row["target"] for row in items} | {d for row in items for d in row["distractors"]})
    value_ids = {v: first_token_id(tokenizer, v) for v in all_values}
    missing_values = sorted(v for v, tid in value_ids.items() if tid is None)
    if missing_values:
        log(f"Skipping multi-token candidates: {missing_values}")

    results: dict[str, Any] = {
        "phase": 68,
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
                control = pick_control(items, idx)
                candidates = [item["target"]] + item["distractors"]
                candidate_ids = {v: value_ids[v] for v in candidates if value_ids.get(v) is not None}
                if item["target"] not in candidate_ids or len(candidate_ids) < 2:
                    continue

                clean_pos = get_positions(tokenizer, item["clean_prompt"], item["object"])
                corrupt_pos = get_positions(tokenizer, item["corrupt_prompt"], item["corrupt_object"])
                control_pos = get_positions(tokenizer, control["clean_prompt"], control["object"])

                h_clean, clean_logits = capture_activation(model, tokenizer, device, layer, module_name, item["clean_prompt"], args.max_length)
                h_corrupt, corrupt_logits = capture_activation(model, tokenizer, device, layer, module_name, item["corrupt_prompt"], args.max_length)
                h_control, _control_logits = capture_activation(model, tokenizer, device, layer, module_name, control["clean_prompt"], args.max_length)

                clean_stats = get_candidate_stats(clean_logits, candidate_ids)
                corrupt_stats = get_candidate_stats(corrupt_logits, candidate_ids)
                clean_margin = margin_from_stats(clean_stats, item["target"], item["distractors"])
                corrupt_margin = margin_from_stats(corrupt_stats, item["target"], item["distractors"])

                for pos_name in positions:
                    sp = clean_pos.get(pos_name)
                    tp = corrupt_pos.get(pos_name)
                    cp = control_pos.get(pos_name)
                    if sp is None or tp is None or cp is None:
                        continue
                    correct_logits = forward_with_replacement(
                        model, tokenizer, device, layer, module_name, item["corrupt_prompt"], int(tp), h_clean[int(sp)], args.max_length
                    )
                    control_logits = forward_with_replacement(
                        model, tokenizer, device, layer, module_name, item["corrupt_prompt"], int(tp), h_control[int(cp)], args.max_length
                    )
                    correct_stats = get_candidate_stats(correct_logits, candidate_ids)
                    control_stats = get_candidate_stats(control_logits, candidate_ids)
                    correct_margin = margin_from_stats(correct_stats, item["target"], item["distractors"])
                    control_margin = margin_from_stats(control_stats, item["target"], item["distractors"])
                    row = {
                        "layer": li,
                        "module": module_name,
                        "position": pos_name,
                        "relation": item["relation"],
                        "object": item["object"],
                        "target": item["target"],
                        "control_object": control["object"],
                        "control_target": control["target"],
                        "frame_key": item["frame_key"],
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "correct_margin": correct_margin,
                        "control_margin": control_margin,
                        "correct_net_over_control": correct_margin - control_margin,
                        "clean_top": clean_stats["top"],
                        "corrupt_top": corrupt_stats["top"],
                        "correct_top": correct_stats["top"],
                        "control_top": control_stats["top"],
                        "clean_target_rank": clean_stats["rank"].get(item["target"]),
                        "corrupt_target_rank": corrupt_stats["rank"].get(item["target"]),
                        "correct_target_rank": correct_stats["rank"].get(item["target"]),
                        "control_target_rank": control_stats["rank"].get(item["target"]),
                    }
                    results["rows"].append(row)

                if (idx + 1) % args.progress_every == 0:
                    log(
                        f"layer={li} module={module_name} item={idx + 1}/{len(items)} "
                        f"rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s"
                    )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        partial = out_dir / f"{args.model}_phase68_object_attribute_natural_exchange.partial.json"
        partial.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    results["summary"] = summarize_rows(results["rows"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_phase68_object_attribute_natural_exchange.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--layers", required=True)
    parser.add_argument("--modules", default="resid_out,mlp_out")
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
