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
from phase87_reader_stack_calibration import (  # noqa: E402
    choice_templates,
    option_letters,
    parse_choice,
    render_options,
)


OBJECTS: list[dict[str, str]] = [
    {"object": "apple", "category": "fruit", "color": "red", "function": "eating", "material": "flesh", "location": "store"},
    {"object": "banana", "category": "fruit", "color": "yellow", "function": "eating", "material": "flesh", "location": "store"},
    {"object": "cherry", "category": "fruit", "color": "red", "function": "eating", "material": "flesh", "location": "tree"},
    {"object": "orange", "category": "fruit", "color": "orange", "function": "eating", "material": "flesh", "location": "store"},
    {"object": "grape", "category": "fruit", "color": "purple", "function": "eating", "material": "flesh", "location": "vine"},
    {"object": "lemon", "category": "fruit", "color": "yellow", "function": "flavoring", "material": "flesh", "location": "tree"},
    {"object": "dog", "category": "animal", "color": "brown", "function": "guarding", "material": "fur", "location": "home"},
    {"object": "cat", "category": "animal", "color": "black", "function": "hunting", "material": "fur", "location": "home"},
    {"object": "horse", "category": "animal", "color": "brown", "function": "riding", "material": "fur", "location": "farm"},
    {"object": "bird", "category": "animal", "color": "white", "function": "flying", "material": "feathers", "location": "sky"},
    {"object": "fish", "category": "animal", "color": "silver", "function": "swimming", "material": "scales", "location": "water"},
    {"object": "sheep", "category": "animal", "color": "white", "function": "grazing", "material": "wool", "location": "farm"},
    {"object": "knife", "category": "tool", "color": "silver", "function": "cutting", "material": "metal", "location": "kitchen"},
    {"object": "hammer", "category": "tool", "color": "black", "function": "hitting", "material": "metal", "location": "toolbox"},
    {"object": "saw", "category": "tool", "color": "silver", "function": "cutting", "material": "metal", "location": "workshop"},
    {"object": "pen", "category": "tool", "color": "blue", "function": "writing", "material": "plastic", "location": "desk"},
    {"object": "broom", "category": "tool", "color": "brown", "function": "cleaning", "material": "wood", "location": "closet"},
    {"object": "spoon", "category": "tool", "color": "silver", "function": "eating", "material": "metal", "location": "kitchen"},
    {"object": "car", "category": "vehicle", "color": "red", "function": "driving", "material": "metal", "location": "road"},
    {"object": "bus", "category": "vehicle", "color": "yellow", "function": "transporting", "material": "metal", "location": "road"},
    {"object": "train", "category": "vehicle", "color": "silver", "function": "transporting", "material": "metal", "location": "railway"},
    {"object": "boat", "category": "vehicle", "color": "white", "function": "sailing", "material": "wood", "location": "water"},
    {"object": "plane", "category": "vehicle", "color": "white", "function": "flying", "material": "metal", "location": "sky"},
    {"object": "bicycle", "category": "vehicle", "color": "black", "function": "riding", "material": "metal", "location": "road"},
    {"object": "school", "category": "place", "color": "white", "function": "teaching", "material": "brick", "location": "city"},
    {"object": "hospital", "category": "place", "color": "white", "function": "healing", "material": "brick", "location": "city"},
    {"object": "forest", "category": "place", "color": "green", "function": "growing", "material": "wood", "location": "nature"},
    {"object": "kitchen", "category": "place", "color": "white", "function": "cooking", "material": "tile", "location": "home"},
    {"object": "library", "category": "place", "color": "brown", "function": "reading", "material": "brick", "location": "city"},
    {"object": "garden", "category": "place", "color": "green", "function": "growing", "material": "soil", "location": "home"},
    {"object": "hand", "category": "body part", "color": "pink", "function": "holding", "material": "skin", "location": "body"},
    {"object": "eye", "category": "body part", "color": "blue", "function": "seeing", "material": "tissue", "location": "face"},
    {"object": "ear", "category": "body part", "color": "pink", "function": "hearing", "material": "skin", "location": "head"},
    {"object": "foot", "category": "body part", "color": "pink", "function": "walking", "material": "skin", "location": "body"},
    {"object": "heart", "category": "body part", "color": "red", "function": "pumping", "material": "muscle", "location": "chest"},
    {"object": "tongue", "category": "body part", "color": "pink", "function": "tasting", "material": "muscle", "location": "mouth"},
    {"object": "doctor", "category": "profession", "color": "white", "function": "healing", "material": "person", "location": "hospital"},
    {"object": "teacher", "category": "profession", "color": "blue", "function": "teaching", "material": "person", "location": "school"},
    {"object": "farmer", "category": "profession", "color": "brown", "function": "growing", "material": "person", "location": "farm"},
    {"object": "driver", "category": "profession", "color": "black", "function": "driving", "material": "person", "location": "road"},
    {"object": "chef", "category": "profession", "color": "white", "function": "cooking", "material": "person", "location": "kitchen"},
    {"object": "writer", "category": "profession", "color": "black", "function": "writing", "material": "person", "location": "desk"},
]


SLOT_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "category": [
        ("kind", "A {object} is a kind of"),
        ("category", "The category of a {object} is"),
    ],
    "color": [
        ("color", "The usual color of a {object} is"),
        ("looks", "A {object} often looks"),
    ],
    "function": [
        ("used_for", "A {object} is used for"),
        ("main_action", "The main action of a {object} is"),
    ],
    "material": [
        ("made_of", "A {object} is made of"),
        ("material", "The material of a {object} is"),
    ],
    "location": [
        ("found_in", "A {object} is usually found in"),
        ("place", "The common place for a {object} is"),
    ],
}


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


def uniq(xs: list[str]) -> list[str]:
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def build_items(max_items: int | None, slots: list[str], templates: list[str]) -> list[dict[str, Any]]:
    wanted_slots = set(slots)
    wanted_templates = set(templates)
    rows: list[dict[str, Any]] = []
    for obj in OBJECTS:
        for slot, slot_templates in SLOT_TEMPLATES.items():
            if wanted_slots and slot not in wanted_slots:
                continue
            values = uniq([x[slot] for x in OBJECTS if x[slot] != obj[slot]])
            distractors = values[:4]
            for template_key, template in slot_templates:
                if wanted_templates and template_key not in wanted_templates:
                    continue
                rows.append({
                    "object": obj["object"],
                    "slot": slot,
                    "template_key": template_key,
                    "prompt": template.format(**obj),
                    "target": obj[slot],
                    "distractors": distractors,
                })
    if not max_items or max_items >= len(rows):
        return rows
    idxs = sorted({round(i * (len(rows) - 1) / max(max_items - 1, 1)) for i in range(max_items)})
    return [rows[i] for i in idxs]


def make_zero_hook():
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    return hook_fn


def prompt_with_options(template: str, prompt: str, candidates: list[str]) -> str:
    return template.format(clean_prompt=prompt, options=render_options(candidates))


def continuation_logprob(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    continuation: str,
    max_length: int,
    component: str = "clean",
    layer_idx: int | None = None,
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
        if component != "clean":
            if layer_idx is None:
                raise ValueError("layer_idx is required for component ablation")
            module_name = "attn_out" if component == "zero_attn" else "mlp_out"
            handles.append(get_module(layers[layer_idx], module_name).register_forward_hook(make_zero_hook()))
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


def generate_choice(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    component: str = "clean",
    layer_idx: int | None = None,
) -> str:
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] > max_length:
        return ""
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if component != "clean":
            if layer_idx is None:
                raise ValueError("layer_idx is required for component ablation")
            module_name = "attn_out" if component == "zero_attn" else "mlp_out"
            handles.append(get_module(layers[layer_idx], module_name).register_forward_hook(make_zero_hook()))
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
        "scores": scores,
    }


def component_rows_for_item(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    item_idx: int,
    layer_idx: int,
    choice_template_key: str,
    max_distractors: int,
    max_length: int,
    max_new_tokens: int,
    generate: bool,
) -> list[dict[str, Any]]:
    candidates = uniq([item["target"], *item["distractors"][:max_distractors]])
    letters = option_letters(len(candidates))
    target_letter = letters[candidates.index(item["target"])]
    c_template = choice_templates()[choice_template_key]
    c_prompt = prompt_with_options(c_template, item["prompt"], candidates)
    letter_candidates = {letter: value for letter, value in zip(letters, candidates)}
    rows = []
    base_value_stats = None
    base_letter_stats = None
    base_generated = ""
    base_choice_correct = False
    for component in ["clean", "zero_attn", "zero_mlp"]:
        value_scores = {
            value: continuation_logprob(
                model,
                tokenizer,
                device,
                layers,
                item["prompt"],
                " " + value,
                max_length,
                component,
                layer_idx if component != "clean" else None,
            )
            for value in candidates
        }
        value_stats = score_stats(value_scores, item["target"], candidates)
        letter_scores = {
            letter: continuation_logprob(
                model,
                tokenizer,
                device,
                layers,
                c_prompt,
                letter,
                max_length,
                component,
                layer_idx if component != "clean" else None,
            )
            for letter in letters
        }
        letter_stats = score_stats(letter_scores, target_letter, letters)
        generated = ""
        selected_value = ""
        choice_valid = False
        choice_correct = False
        if generate:
            generated = generate_choice(
                model,
                tokenizer,
                device,
                layers,
                c_prompt,
                max_length,
                max_new_tokens,
                component,
                layer_idx if component != "clean" else None,
            )
            parsed = parse_choice(generated, candidates)
            selected_value = parsed["selected_value"]
            choice_valid = bool(parsed["choice_valid"])
            choice_correct = selected_value == item["target"]
        if component == "clean":
            base_value_stats = value_stats
            base_letter_stats = letter_stats
            base_generated = generated
            base_choice_correct = choice_correct
        row = {
            "item_idx": item_idx,
            "layer": layer_idx,
            "component": component,
            "slot": item["slot"],
            "template_key": item["template_key"],
            "choice_template_key": choice_template_key,
            "object": item["object"],
            "target": item["target"],
            "target_letter": target_letter,
            "candidates": candidates,
            "letter_candidates": letter_candidates,
            "value_top": value_stats["top"],
            "value_rank": value_stats["rank"],
            "value_top1": value_stats["top1"],
            "value_top1_margin": value_stats["top1_margin"],
            "value_mean_margin": value_stats["mean_margin"],
            "letter_top": letter_stats["top"],
            "letter_rank": letter_stats["rank"],
            "letter_top1": letter_stats["top1"],
            "letter_top1_margin": letter_stats["top1_margin"],
            "letter_mean_margin": letter_stats["mean_margin"],
            "generated": generated,
            "selected_value": selected_value,
            "choice_valid": choice_valid,
            "choice_correct": choice_correct,
            "base_generated": base_generated,
            "base_choice_correct": base_choice_correct,
        }
        if base_value_stats is not None and base_letter_stats is not None:
            row.update({
                "component_value_effect_top1": float(base_value_stats["top1_margin"]) - float(value_stats["top1_margin"]),
                "component_value_effect_mean": float(base_value_stats["mean_margin"]) - float(value_stats["mean_margin"]),
                "component_letter_effect_top1": float(base_letter_stats["top1_margin"]) - float(letter_stats["top1_margin"]),
                "component_letter_effect_mean": float(base_letter_stats["mean_margin"]) - float(letter_stats["mean_margin"]),
                "choice_drop": float(base_choice_correct) - float(choice_correct),
            })
        rows.append(row)
    return rows


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    ablated = [v for v in vals if v["component"] != "clean"]
    return {
        "n": len(vals),
        "clean_n": len([v for v in vals if v["component"] == "clean"]),
        "value_top1": avg([float(v["value_top1"]) for v in vals]),
        "letter_top1": avg([float(v["letter_top1"]) for v in vals]),
        "choice_top1": avg([float(v["choice_correct"]) for v in vals if v.get("generated", "") != ""]),
        "value_top1_margin": avg([float(v["value_top1_margin"]) for v in vals]),
        "letter_top1_margin": avg([float(v["letter_top1_margin"]) for v in vals]),
        "component_value_effect_top1": avg([float(v.get("component_value_effect_top1", 0.0)) for v in ablated]),
        "component_value_effect_mean": avg([float(v.get("component_value_effect_mean", 0.0)) for v in ablated]),
        "component_letter_effect_top1": avg([float(v.get("component_letter_effect_top1", 0.0)) for v in ablated]),
        "component_letter_effect_mean": avg([float(v.get("component_letter_effect_mean", 0.0)) for v in ablated]),
        "choice_drop": avg([float(v.get("choice_drop", 0.0)) for v in ablated if v.get("generated", "") != ""]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_component": defaultdict(list),
        "by_layer_component": defaultdict(list),
        "by_slot_component": defaultdict(list),
        "by_layer_slot_component": defaultdict(list),
    }
    for row in rows:
        groups["by_component"][row["component"]].append(row)
        groups["by_layer_component"][(row["layer"], row["component"])].append(row)
        groups["by_slot_component"][(row["slot"], row["component"])].append(row)
        groups["by_layer_slot_component"][(row["layer"], row["slot"], row["component"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE90_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_idxs = [int(x) for x in parse_csv(args.layers)]
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    log(f"Phase90 model={args.model} items={len(items)} layers={layer_idxs} choice_template={args.choice_template} generate={args.generate}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase90_component_margin_reader_alignment.json"
    partial_path = out_dir / f"{args.model}_phase90_component_margin_reader_alignment.partial.json"
    results: dict[str, Any] = {
        "phase": 90,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "component_margin_reader_alignment",
        "layers": layer_idxs,
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
            if loaded.get("phase") == 90 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed = {(int(r["layer"]), int(r["item_idx"]), r["component"]) for r in results["rows"]}
    t0 = time.time()
    for layer_idx in layer_idxs:
        for idx, item in enumerate(items):
            if all((layer_idx, idx, component) in completed for component in ["clean", "zero_attn", "zero_mlp"]):
                continue
            rows = component_rows_for_item(
                model,
                tokenizer,
                device,
                layers,
                item,
                idx,
                layer_idx,
                args.choice_template,
                args.max_distractors,
                args.max_length,
                args.choice_max_new_tokens,
                args.generate,
            )
            for row in rows:
                key = (int(row["layer"]), int(row["item_idx"]), row["component"])
                if key not in completed:
                    results["rows"].append(row)
                    completed.add(key)
            if (idx + 1) % args.progress_every == 0:
                log(f"layer={layer_idx} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
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
    parser.add_argument("--layers", required=True)
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
