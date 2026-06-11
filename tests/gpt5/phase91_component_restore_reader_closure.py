from __future__ import annotations

import argparse
import ctypes
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
from phase68_object_attribute_natural_exchange import get_module, load_model, parse_csv  # noqa: E402
from phase87_reader_stack_calibration import choice_templates, option_letters, parse_choice, render_options  # noqa: E402
from phase90_component_margin_reader_alignment import build_items, uniq  # noqa: E402


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


def parse_nodes(text: str) -> list[tuple[int, str]]:
    out = []
    for raw in parse_csv(text):
        layer, comp = raw.split(":", 1)
        if comp not in {"attn", "mlp"}:
            raise ValueError(f"unknown component in node: {raw}")
        out.append((int(layer), comp))
    return out


def module_name(component: str) -> str:
    return "attn_out" if component == "attn" else "mlp_out"


def capture_component_output(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    text_ids: list[int],
    layer_idx: int,
    component: str,
) -> torch.Tensor:
    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module: Any, _inputs: Any, output: Any):
        tensor = output[0] if isinstance(output, tuple) else output
        captured["h"] = tensor.detach().float().cpu()

    handle = get_module(layers[layer_idx], module_name(component)).register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


def make_zero_hook():
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    return hook_fn


def make_restore_hook(clean_output: torch.Tensor):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        restored = clean_output.to(device=hs.device, dtype=hs.dtype)
        if restored.shape == hs.shape:
            out = restored
        elif restored.ndim == 3 and hs.ndim == 3 and restored.shape[1] >= hs.shape[1] and restored.shape[2] == hs.shape[2]:
            out = hs.clone()
            out[:, : hs.shape[1], :] = restored[:, : hs.shape[1], :]
        else:
            out = hs
        return (out,) + output[1:] if isinstance(output, tuple) else out

    return hook_fn


def continuation_logprob_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    continuation: str,
    max_length: int,
    layer_idx: int,
    component: str,
    condition: str,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        module = get_module(layers[layer_idx], module_name(component))
        if condition == "zero":
            handles.append(module.register_forward_hook(make_zero_hook()))
        elif condition == "restore":
            clean_output = capture_component_output(model, tokenizer, device, layers, full_ids, layer_idx, component)
            handles.append(module.register_forward_hook(make_restore_hook(clean_output)))
        elif condition != "clean":
            raise ValueError(f"unknown condition={condition}")
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
    finally:
        for handle in handles:
            handle.remove()
    start = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        logit_pos = start + i - 1
        if logit_pos < 0 or logit_pos >= log_probs.shape[0]:
            return float("-inf")
        total += float(log_probs[logit_pos, tok].detach().cpu())
    return total


def generate_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    layer_idx: int,
    component: str,
    condition: str,
) -> str:
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] > max_length:
        return ""
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        module = get_module(layers[layer_idx], module_name(component))
        if condition == "zero":
            handles.append(module.register_forward_hook(make_zero_hook()))
        elif condition == "restore":
            clean_output = capture_component_output(model, tokenizer, device, layers, input_ids[0].detach().cpu().tolist(), layer_idx, component)
            handles.append(module.register_forward_hook(make_restore_hook(clean_output)))
        elif condition != "clean":
            raise ValueError(f"unknown condition={condition}")
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
    finally:
        for handle in handles:
            handle.remove()
    return tokenizer.decode(out[0, input_ids.shape[1]:].detach().cpu().tolist(), skip_special_tokens=True)


def score_stats(scores: dict[str, float], target: str, candidates: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    competitors = [x for x in candidates if x != target]
    max_comp = max((scores[x] for x in competitors), default=-1e9)
    mean_comp = avg([scores[x] for x in competitors])
    return {
        "top": ordered[0][0] if ordered else "",
        "rank": {name: i + 1 for i, (name, _score) in enumerate(ordered)}.get(target, 999),
        "top1": bool(ordered and ordered[0][0] == target),
        "top1_margin": float(scores.get(target, -1e9) - max_comp),
        "mean_margin": float(scores.get(target, -1e9) - mean_comp),
    }


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def run_item_node(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    item_idx: int,
    layer_idx: int,
    component: str,
    choice_template: str,
    max_distractors: int,
    max_length: int,
    choice_max_new_tokens: int,
    generate: bool,
) -> dict[str, Any]:
    candidates = uniq([item["target"], *item["distractors"][:max_distractors]])
    letters = option_letters(len(candidates))
    target_letter = letters[candidates.index(item["target"])]
    choice_prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
    value_stats: dict[str, dict[str, Any]] = {}
    letter_stats: dict[str, dict[str, Any]] = {}
    gen: dict[str, dict[str, Any]] = {}
    for condition in ["clean", "zero", "restore"]:
        value_scores = {
            value: continuation_logprob_condition(
                model, tokenizer, device, layers, item["prompt"], " " + value,
                max_length, layer_idx, component, condition
            )
            for value in candidates
        }
        value_stats[condition] = score_stats(value_scores, item["target"], candidates)
        letter_scores = {
            letter: continuation_logprob_condition(
                model, tokenizer, device, layers, choice_prompt, letter,
                max_length, layer_idx, component, condition
            )
            for letter in letters
        }
        letter_stats[condition] = score_stats(letter_scores, target_letter, letters)
        generated = ""
        parsed = {"selected_value": "", "selected_letter": "", "choice_valid": False}
        if generate:
            generated = generate_condition(
                model, tokenizer, device, layers, choice_prompt,
                max_length, choice_max_new_tokens, layer_idx, component, condition
            )
            parsed = parse_choice(generated, candidates)
        gen[condition] = {
            "generated": generated,
            "selected_value": parsed["selected_value"],
            "selected_letter": parsed["selected_letter"],
            "valid": bool(parsed["choice_valid"]),
            "correct": parsed["selected_value"] == item["target"],
        }
    return {
        "item_idx": item_idx,
        "layer": layer_idx,
        "component": component,
        "slot": item["slot"],
        "template_key": item["template_key"],
        "choice_template_key": choice_template,
        "object": item["object"],
        "target": item["target"],
        "target_letter": target_letter,
        "candidates": candidates,
        "clean_value_margin": value_stats["clean"]["top1_margin"],
        "zero_value_margin": value_stats["zero"]["top1_margin"],
        "restore_value_margin": value_stats["restore"]["top1_margin"],
        "clean_letter_margin": letter_stats["clean"]["top1_margin"],
        "zero_letter_margin": letter_stats["zero"]["top1_margin"],
        "restore_letter_margin": letter_stats["restore"]["top1_margin"],
        "clean_value_top1": value_stats["clean"]["top1"],
        "zero_value_top1": value_stats["zero"]["top1"],
        "restore_value_top1": value_stats["restore"]["top1"],
        "clean_letter_top1": letter_stats["clean"]["top1"],
        "zero_letter_top1": letter_stats["zero"]["top1"],
        "restore_letter_top1": letter_stats["restore"]["top1"],
        "clean_choice_correct": gen["clean"]["correct"],
        "zero_choice_correct": gen["zero"]["correct"],
        "restore_choice_correct": gen["restore"]["correct"],
        "clean_choice_valid": gen["clean"]["valid"],
        "zero_choice_valid": gen["zero"]["valid"],
        "restore_choice_valid": gen["restore"]["valid"],
        "clean_generated": gen["clean"]["generated"],
        "zero_generated": gen["zero"]["generated"],
        "restore_generated": gen["restore"]["generated"],
        "value_drop": value_stats["clean"]["top1_margin"] - value_stats["zero"]["top1_margin"],
        "value_restore_gain": value_stats["restore"]["top1_margin"] - value_stats["zero"]["top1_margin"],
        "value_restore_gap": value_stats["clean"]["top1_margin"] - value_stats["restore"]["top1_margin"],
        "letter_drop": letter_stats["clean"]["top1_margin"] - letter_stats["zero"]["top1_margin"],
        "letter_restore_gain": letter_stats["restore"]["top1_margin"] - letter_stats["zero"]["top1_margin"],
        "letter_restore_gap": letter_stats["clean"]["top1_margin"] - letter_stats["restore"]["top1_margin"],
        "choice_drop": float(gen["clean"]["correct"]) - float(gen["zero"]["correct"]),
        "choice_restore_gain": float(gen["restore"]["correct"]) - float(gen["zero"]["correct"]),
        "choice_restore_gap": float(gen["clean"]["correct"]) - float(gen["restore"]["correct"]),
    }


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "clean_value_top1": avg([float(v["clean_value_top1"]) for v in vals]),
        "zero_value_top1": avg([float(v["zero_value_top1"]) for v in vals]),
        "restore_value_top1": avg([float(v["restore_value_top1"]) for v in vals]),
        "clean_letter_top1": avg([float(v["clean_letter_top1"]) for v in vals]),
        "zero_letter_top1": avg([float(v["zero_letter_top1"]) for v in vals]),
        "restore_letter_top1": avg([float(v["restore_letter_top1"]) for v in vals]),
        "clean_choice_top1": avg([float(v["clean_choice_correct"]) for v in vals]),
        "zero_choice_top1": avg([float(v["zero_choice_correct"]) for v in vals]),
        "restore_choice_top1": avg([float(v["restore_choice_correct"]) for v in vals]),
        "value_drop": avg([float(v["value_drop"]) for v in vals]),
        "value_restore_gain": avg([float(v["value_restore_gain"]) for v in vals]),
        "value_restore_gap": avg([float(v["value_restore_gap"]) for v in vals]),
        "letter_drop": avg([float(v["letter_drop"]) for v in vals]),
        "letter_restore_gain": avg([float(v["letter_restore_gain"]) for v in vals]),
        "letter_restore_gap": avg([float(v["letter_restore_gap"]) for v in vals]),
        "choice_drop": avg([float(v["choice_drop"]) for v in vals]),
        "choice_restore_gain": avg([float(v["choice_restore_gain"]) for v in vals]),
        "choice_restore_gap": avg([float(v["choice_restore_gap"]) for v in vals]),
        "clean_choice_valid": avg([float(v["clean_choice_valid"]) for v in vals]),
        "zero_choice_valid": avg([float(v["zero_choice_valid"]) for v in vals]),
        "restore_choice_valid": avg([float(v["restore_choice_valid"]) for v in vals]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_node": defaultdict(list),
        "by_slot": defaultdict(list),
        "by_node_slot": defaultdict(list),
    }
    for row in rows:
        node = f"L{row['layer']}:{row['component']}"
        groups["by_node"][node].append(row)
        groups["by_slot"][row["slot"]].append(row)
        groups["by_node_slot"][(node, row["slot"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE91_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    nodes = parse_nodes(args.nodes)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    log(f"Phase91 model={args.model} items={len(items)} nodes={nodes} generate={args.generate}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase91_component_restore_reader_closure.json"
    partial_path = out_dir / f"{args.model}_phase91_component_restore_reader_closure.partial.json"
    results: dict[str, Any] = {
        "phase": 91,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "component_restore_reader_closure",
        "nodes": [f"{l}:{c}" for l, c in nodes],
        "num_items": len(items),
        "slots": sorted({x["slot"] for x in items}),
        "choice_template": args.choice_template,
        "generate": args.generate,
        "rows": [],
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 91 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed = {(int(r["layer"]), r["component"], int(r["item_idx"])) for r in results["rows"]}
    t0 = time.time()
    for layer_idx, component in nodes:
        for idx, item in enumerate(items):
            if (layer_idx, component, idx) in completed:
                continue
            row = run_item_node(
                model, tokenizer, device, layers, item, idx, layer_idx, component,
                args.choice_template, args.max_distractors, args.max_length,
                args.choice_max_new_tokens, args.generate,
            )
            results["rows"].append(row)
            completed.add((layer_idx, component, idx))
            if (idx + 1) % args.progress_every == 0:
                log(f"node=L{layer_idx}:{component} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
                partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                cleanup_cuda()
        partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    results["summary"] = summarize(results["rows"])
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--slots", default="category,color,function,material,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=420)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--choice-max-new-tokens", type=int, default=4)
    parser.add_argument("--generate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=70)
    parser.add_argument("--attn-implementations", default="flash_attention_2,sdpa,eager")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()
    try:
        run_model(args)
    finally:
        release_loaded(None)
        cleanup_cuda()
    if args.hard_exit_after_model:
        log("Hard exit after model requested.")
        os._exit(0)


if __name__ == "__main__":
    main()
