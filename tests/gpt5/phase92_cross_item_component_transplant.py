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
from phase87_reader_stack_calibration import choice_templates, option_letters, render_options  # noqa: E402
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


def make_zero_hook():
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    return hook_fn


def transplant_tensor(target: torch.Tensor, donor_cpu: torch.Tensor, copy_mode: str) -> torch.Tensor:
    donor = donor_cpu.to(device=target.device, dtype=target.dtype)
    if donor.shape == target.shape:
        return donor
    if donor.ndim != 3 or target.ndim != 3 or donor.shape[0] != target.shape[0] or donor.shape[2] != target.shape[2]:
        return target
    out = target.clone()
    n = min(int(donor.shape[1]), int(target.shape[1]))
    if n <= 0:
        return out
    if copy_mode == "prefix":
        out[:, :n, :] = donor[:, :n, :]
    elif copy_mode == "tail":
        out[:, -n:, :] = donor[:, -n:, :]
    elif copy_mode == "both":
        k = max(1, n // 2)
        out[:, :k, :] = donor[:, :k, :]
        out[:, -k:, :] = donor[:, -k:, :]
    else:
        raise ValueError(f"unknown copy_mode={copy_mode}")
    return out


def make_transplant_hook(donor_output: torch.Tensor, copy_mode: str):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        patched = transplant_tensor(hs, donor_output, copy_mode)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    return hook_fn


def capture_component_output(
    model: Any,
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


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def fullseq_logprob_with_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    target_prompt: str,
    donor_prompt: str | None,
    continuation: str,
    max_length: int,
    layer_idx: int,
    component: str,
    condition: str,
    copy_mode: str,
) -> float:
    target_prompt_ids = token_ids(tokenizer, target_prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    target_full_ids = target_prompt_ids + cont_ids
    if len(target_full_ids) > max_length:
        return float("-inf")
    input_ids = torch.tensor([target_full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        module = get_module(layers[layer_idx], module_name(component))
        if condition == "zero":
            handles.append(module.register_forward_hook(make_zero_hook()))
        elif condition == "self_restore":
            donor_output = capture_component_output(model, device, layers, target_full_ids, layer_idx, component)
            handles.append(module.register_forward_hook(make_transplant_hook(donor_output, copy_mode)))
        elif condition.startswith("transplant:"):
            if donor_prompt is None:
                raise ValueError("donor_prompt is required for transplant")
            donor_full_ids = token_ids(tokenizer, donor_prompt) + cont_ids
            if len(donor_full_ids) > max_length:
                return float("-inf")
            donor_output = capture_component_output(model, device, layers, donor_full_ids, layer_idx, component)
            handles.append(module.register_forward_hook(make_transplant_hook(donor_output, copy_mode)))
        elif condition != "clean":
            raise ValueError(f"unknown condition={condition}")
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
    finally:
        for handle in handles:
            handle.remove()
    start = len(target_prompt_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        logit_pos = start + i - 1
        if logit_pos < 0 or logit_pos >= log_probs.shape[0]:
            return float("-inf")
        total += float(log_probs[logit_pos, tok].detach().cpu())
    return total


def score_stats(scores: dict[str, float], target: str, candidates: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    competitors = [x for x in candidates if x != target]
    max_comp = max((scores[x] for x in competitors), default=-1e9)
    mean_comp = avg([scores[x] for x in competitors])
    return {
        "top": ordered[0][0] if ordered else "",
        "top1": bool(ordered and ordered[0][0] == target),
        "top1_margin": float(scores.get(target, -1e9) - max_comp),
        "mean_margin": float(scores.get(target, -1e9) - mean_comp),
    }


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def donor_kinds() -> list[str]:
    return [
        "self_restore",
        "same_slot_same_target",
        "same_slot_diff_target",
        "diff_slot_same_object",
        "diff_slot_diff_object",
    ]


def select_donors(items: list[dict[str, Any]]) -> dict[int, dict[str, int | None]]:
    by_slot_target: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_slot: dict[str, list[int]] = defaultdict(list)
    by_object: dict[str, list[int]] = defaultdict(list)
    all_idxs = list(range(len(items)))
    for idx, item in enumerate(items):
        by_slot_target[(item["slot"], item["target"])].append(idx)
        by_slot[item["slot"]].append(idx)
        by_object[item["object"]].append(idx)

    donors: dict[int, dict[str, int | None]] = {}
    for idx, item in enumerate(items):
        same_slot_same_target = next((j for j in by_slot_target[(item["slot"], item["target"])] if j != idx), None)
        same_slot_diff_target = next((j for j in by_slot[item["slot"]] if j != idx and items[j]["target"] != item["target"]), None)
        diff_slot_same_object = next((j for j in by_object[item["object"]] if j != idx and items[j]["slot"] != item["slot"]), None)
        diff_slot_diff_object = next((j for j in all_idxs if items[j]["slot"] != item["slot"] and items[j]["object"] != item["object"]), None)
        donors[idx] = {
            "self_restore": idx,
            "same_slot_same_target": same_slot_same_target,
            "same_slot_diff_target": same_slot_diff_target,
            "diff_slot_same_object": diff_slot_same_object,
            "diff_slot_diff_object": diff_slot_diff_object,
        }
    return donors


def condition_prompt(item: dict[str, Any], candidates: list[str], choice_template: str, score_type: str) -> str:
    if score_type == "value":
        return item["prompt"]
    if score_type == "letter":
        return build_choice_prompt(choice_template, item["prompt"], candidates)
    raise ValueError(score_type)


def score_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    donor_item: dict[str, Any] | None,
    candidates: list[str],
    letters: list[str],
    target_letter: str,
    choice_template: str,
    max_length: int,
    layer_idx: int,
    component: str,
    condition: str,
    copy_mode: str,
) -> dict[str, Any]:
    value_scores = {}
    for value in candidates:
        donor_prompt = donor_item["prompt"] if donor_item else None
        value_scores[value] = fullseq_logprob_with_condition(
            model, tokenizer, device, layers,
            item["prompt"], donor_prompt, " " + value,
            max_length, layer_idx, component, condition, copy_mode,
        )
    choice_prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
    donor_choice_prompt = build_choice_prompt(choice_template, donor_item["prompt"], candidates) if donor_item else None
    letter_scores = {}
    for letter in letters:
        letter_scores[letter] = fullseq_logprob_with_condition(
            model, tokenizer, device, layers,
            choice_prompt, donor_choice_prompt, letter,
            max_length, layer_idx, component, condition, copy_mode,
        )
    value_stats = score_stats(value_scores, item["target"], candidates)
    letter_stats = score_stats(letter_scores, target_letter, letters)
    return {
        "value_margin": value_stats["top1_margin"],
        "letter_margin": letter_stats["top1_margin"],
        "value_top1": value_stats["top1"],
        "letter_top1": letter_stats["top1"],
        "value_top": value_stats["top"],
        "letter_top": letter_stats["top"],
    }


def run_item_node(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    donors_by_idx: dict[int, dict[str, int | None]],
    item_idx: int,
    layer_idx: int,
    component: str,
    choice_template: str,
    max_distractors: int,
    max_length: int,
    copy_mode: str,
) -> list[dict[str, Any]]:
    item = items[item_idx]
    candidates = uniq([item["target"], *item["distractors"][:max_distractors]])
    letters = option_letters(len(candidates))
    target_letter = letters[candidates.index(item["target"])]
    clean = score_condition(
        model, tokenizer, device, layers, item, None, candidates, letters, target_letter,
        choice_template, max_length, layer_idx, component, "clean", copy_mode,
    )
    zero = score_condition(
        model, tokenizer, device, layers, item, None, candidates, letters, target_letter,
        choice_template, max_length, layer_idx, component, "zero", copy_mode,
    )
    rows: list[dict[str, Any]] = []
    for donor_kind in donor_kinds():
        donor_idx = donors_by_idx[item_idx].get(donor_kind)
        if donor_idx is None:
            continue
        donor_item = items[donor_idx] if donor_kind != "self_restore" else None
        condition = "self_restore" if donor_kind == "self_restore" else f"transplant:{donor_kind}"
        patched = score_condition(
            model, tokenizer, device, layers, item, donor_item, candidates, letters, target_letter,
            choice_template, max_length, layer_idx, component, condition, copy_mode,
        )
        rows.append({
            "item_idx": item_idx,
            "donor_idx": donor_idx,
            "donor_kind": donor_kind,
            "layer": layer_idx,
            "component": component,
            "slot": item["slot"],
            "template_key": item["template_key"],
            "object": item["object"],
            "target": item["target"],
            "donor_slot": items[donor_idx]["slot"],
            "donor_object": items[donor_idx]["object"],
            "donor_target": items[donor_idx]["target"],
            "target_letter": target_letter,
            "num_candidates": len(candidates),
            "copy_mode": copy_mode,
            "clean_value_margin": clean["value_margin"],
            "zero_value_margin": zero["value_margin"],
            "patched_value_margin": patched["value_margin"],
            "clean_letter_margin": clean["letter_margin"],
            "zero_letter_margin": zero["letter_margin"],
            "patched_letter_margin": patched["letter_margin"],
            "clean_value_top1": clean["value_top1"],
            "zero_value_top1": zero["value_top1"],
            "patched_value_top1": patched["value_top1"],
            "clean_letter_top1": clean["letter_top1"],
            "zero_letter_top1": zero["letter_top1"],
            "patched_letter_top1": patched["letter_top1"],
            "value_drop": clean["value_margin"] - zero["value_margin"],
            "value_patch_gain": patched["value_margin"] - zero["value_margin"],
            "value_patch_gap": clean["value_margin"] - patched["value_margin"],
            "letter_drop": clean["letter_margin"] - zero["letter_margin"],
            "letter_patch_gain": patched["letter_margin"] - zero["letter_margin"],
            "letter_patch_gap": clean["letter_margin"] - patched["letter_margin"],
            "value_top1_drop": float(clean["value_top1"]) - float(zero["value_top1"]),
            "value_top1_patch_gain": float(patched["value_top1"]) - float(zero["value_top1"]),
            "letter_top1_drop": float(clean["letter_top1"]) - float(zero["letter_top1"]),
            "letter_top1_patch_gain": float(patched["letter_top1"]) - float(zero["letter_top1"]),
        })
    return rows


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "clean_value_top1": avg([float(v["clean_value_top1"]) for v in vals]),
        "zero_value_top1": avg([float(v["zero_value_top1"]) for v in vals]),
        "patched_value_top1": avg([float(v["patched_value_top1"]) for v in vals]),
        "clean_letter_top1": avg([float(v["clean_letter_top1"]) for v in vals]),
        "zero_letter_top1": avg([float(v["zero_letter_top1"]) for v in vals]),
        "patched_letter_top1": avg([float(v["patched_letter_top1"]) for v in vals]),
        "value_drop": avg([float(v["value_drop"]) for v in vals]),
        "value_patch_gain": avg([float(v["value_patch_gain"]) for v in vals]),
        "value_patch_gap": avg([float(v["value_patch_gap"]) for v in vals]),
        "letter_drop": avg([float(v["letter_drop"]) for v in vals]),
        "letter_patch_gain": avg([float(v["letter_patch_gain"]) for v in vals]),
        "letter_patch_gap": avg([float(v["letter_patch_gap"]) for v in vals]),
        "value_top1_drop": avg([float(v["value_top1_drop"]) for v in vals]),
        "value_top1_patch_gain": avg([float(v["value_top1_patch_gain"]) for v in vals]),
        "letter_top1_drop": avg([float(v["letter_top1_drop"]) for v in vals]),
        "letter_top1_patch_gain": avg([float(v["letter_top1_patch_gain"]) for v in vals]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_node": defaultdict(list),
        "by_donor_kind": defaultdict(list),
        "by_node_donor_kind": defaultdict(list),
        "by_node_slot_donor_kind": defaultdict(list),
    }
    for row in rows:
        node = f"L{row['layer']}:{row['component']}"
        groups["by_node"][node].append(row)
        groups["by_donor_kind"][row["donor_kind"]].append(row)
        groups["by_node_donor_kind"][(node, row["donor_kind"])].append(row)
        groups["by_node_slot_donor_kind"][(node, row["slot"], row["donor_kind"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE92_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    nodes = parse_nodes(args.nodes)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors_by_idx = select_donors(items)
    log(f"Phase92 model={args.model} items={len(items)} nodes={nodes} copy_mode={args.copy_mode}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase92_cross_item_component_transplant.json"
    partial_path = out_dir / f"{args.model}_phase92_cross_item_component_transplant.partial.json"
    results: dict[str, Any] = {
        "phase": 92,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "cross_item_component_transplant",
        "nodes": [f"{l}:{c}" for l, c in nodes],
        "num_items": len(items),
        "slots": sorted({x["slot"] for x in items}),
        "choice_template": args.choice_template,
        "copy_mode": args.copy_mode,
        "rows": [],
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 92 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed = {(int(r["layer"]), r["component"], int(r["item_idx"]), r["donor_kind"]) for r in results["rows"]}
    t0 = time.time()
    for layer_idx, component in nodes:
        for idx, _item in enumerate(items):
            pending = [k for k in donor_kinds() if (layer_idx, component, idx, k) not in completed]
            if not pending:
                continue
            item_rows = run_item_node(
                model, tokenizer, device, layers, items, donors_by_idx, idx,
                layer_idx, component, args.choice_template, args.max_distractors,
                args.max_length, args.copy_mode,
            )
            for row in item_rows:
                key = (int(row["layer"]), row["component"], int(row["item_idx"]), row["donor_kind"])
                if key not in completed:
                    results["rows"].append(row)
                    completed.add(key)
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
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--copy-mode", choices=["tail", "prefix", "both"], default="tail")
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
