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
from phase92_cross_item_component_transplant import select_donors  # noqa: E402


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
            raise ValueError(f"unknown node={raw}")
        out.append((int(layer), comp))
    return out


def module_name(component: str) -> str:
    return "attn_out" if component == "attn" else "mlp_out"


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    except Exception:
        return token_ids(tokenizer, text), []


def span_token_indices(
    tokenizer: Any,
    text: str,
    start: int,
    end: int,
    fallback: list[int],
) -> list[int]:
    ids, offs = offsets(tokenizer, text)
    if not ids:
        return []
    if offs:
        hits = [i for i, (a, b) in enumerate(offs) if b > start and a < end]
        return hits or fallback
    return fallback


def local_positions(tokenizer: Any, prompt: str, clean_prompt: str, obj: str, position_kind: str) -> list[int]:
    prompt_ids = token_ids(tokenizer, prompt)
    if not prompt_ids:
        return []
    if position_kind == "prefix8":
        return list(range(min(8, len(prompt_ids))))
    if position_kind == "prompt_tail":
        return [len(prompt_ids) - 1]
    if position_kind == "last4":
        return list(range(max(0, len(prompt_ids) - 4), len(prompt_ids)))

    base = prompt.find(clean_prompt)
    if base < 0:
        base = 0
        clean_prompt = prompt
    obj_start_in_clean = clean_prompt.find(obj)
    if obj_start_in_clean < 0:
        return [len(prompt_ids) - 1]
    obj_start = base + obj_start_in_clean
    obj_end = obj_start + len(obj)
    if position_kind == "object_span":
        return span_token_indices(tokenizer, prompt, obj_start, obj_end, [len(prompt_ids) - 1])
    if position_kind == "relation_span":
        rel_start = obj_end
        rel_end = base + len(clean_prompt)
        return span_token_indices(tokenizer, prompt, rel_start, rel_end, [len(prompt_ids) - 1])
    raise ValueError(f"unknown position_kind={position_kind}")


def capture_output(
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
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs.detach().float().cpu()

    handle = get_module(layers[layer_idx], module_name(component)).register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


def make_local_hook(op: str, target_positions: list[int], donor_output: torch.Tensor | None, donor_positions: list[int]):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        patched = hs.clone()
        tpos = [p for p in target_positions if 0 <= p < patched.shape[1]]
        if not tpos:
            return output
        if op == "zero":
            patched[:, tpos, :] = 0
        elif op == "transplant":
            if donor_output is None:
                return output
            donor = donor_output.to(device=patched.device, dtype=patched.dtype)
            dpos = [p for p in donor_positions if 0 <= p < donor.shape[1]]
            if not dpos:
                return output
            if len(dpos) == len(tpos):
                patched[:, tpos, :] = donor[:, dpos, :]
            else:
                donor_mean = donor[:, dpos, :].mean(dim=1, keepdim=True)
                patched[:, tpos, :] = donor_mean.expand(-1, len(tpos), -1)
        else:
            raise ValueError(op)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    return hook_fn


def fullseq_logprob_local(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    target_prompt: str,
    target_clean_prompt: str,
    target_object: str,
    donor_prompt: str | None,
    donor_clean_prompt: str | None,
    donor_object: str | None,
    continuation: str,
    max_length: int,
    layer_idx: int,
    component: str,
    condition: str,
    position_kind: str,
) -> float:
    prompt_ids = token_ids(tokenizer, target_prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    target_positions = local_positions(tokenizer, target_prompt, target_clean_prompt, target_object, position_kind)
    donor_output = None
    donor_positions: list[int] = []
    op = "clean"
    if condition == "zero":
        op = "zero"
    elif condition.startswith("transplant:"):
        if donor_prompt is None or donor_clean_prompt is None or donor_object is None:
            return float("-inf")
        donor_ids = token_ids(tokenizer, donor_prompt) + cont_ids
        if len(donor_ids) > max_length:
            return float("-inf")
        donor_output = capture_output(model, device, layers, donor_ids, layer_idx, component)
        donor_positions = local_positions(tokenizer, donor_prompt, donor_clean_prompt, donor_object, position_kind)
        op = "transplant"
    elif condition != "clean":
        raise ValueError(condition)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if op != "clean":
            handles.append(
                get_module(layers[layer_idx], module_name(component)).register_forward_hook(
                    make_local_hook(op, target_positions, donor_output, donor_positions)
                )
            )
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            log_probs = F.log_softmax(logits.float(), dim=-1)
    finally:
        for handle in handles:
            handle.remove()
    start = len(prompt_ids)
    total = 0.0
    for i, tok in enumerate(cont_ids):
        pos = start + i - 1
        if pos < 0 or pos >= log_probs.shape[0]:
            return float("-inf")
        total += float(log_probs[pos, tok].detach().cpu())
    return total


def score_stats(scores: dict[str, float], target: str, candidates: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    competitors = [x for x in candidates if x != target]
    max_comp = max((scores[x] for x in competitors), default=-1e9)
    return {
        "top": ordered[0][0] if ordered else "",
        "top1": bool(ordered and ordered[0][0] == target),
        "margin": float(scores.get(target, -1e9) - max_comp),
    }


def condition_score(
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
    position_kind: str,
) -> dict[str, Any]:
    value_scores = {}
    donor_prompt = donor_item["prompt"] if donor_item else None
    donor_object = donor_item["object"] if donor_item else None
    for value in candidates:
        value_scores[value] = fullseq_logprob_local(
            model,
            tokenizer,
            device,
            layers,
            item["prompt"],
            item["prompt"],
            item["object"],
            donor_prompt,
            donor_prompt,
            donor_object,
            " " + value,
            max_length,
            layer_idx,
            component,
            condition,
            position_kind,
        )
    choice_prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
    donor_choice_prompt = build_choice_prompt(choice_template, donor_item["prompt"], candidates) if donor_item else None
    letter_scores = {}
    for letter in letters:
        letter_scores[letter] = fullseq_logprob_local(
            model,
            tokenizer,
            device,
            layers,
            choice_prompt,
            item["prompt"],
            item["object"],
            donor_choice_prompt,
            donor_item["prompt"] if donor_item else None,
            donor_item["object"] if donor_item else None,
            letter,
            max_length,
            layer_idx,
            component,
            condition,
            position_kind,
        )
    vs = score_stats(value_scores, item["target"], candidates)
    ls = score_stats(letter_scores, target_letter, letters)
    return {
        "value_margin": vs["margin"],
        "letter_margin": ls["margin"],
        "value_top1": vs["top1"],
        "letter_top1": ls["top1"],
    }


def specs(donor_kinds: list[str]) -> list[tuple[str, str]]:
    out = [("zero", "")]
    out.extend((f"transplant:{kind}", kind) for kind in donor_kinds)
    return out


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "value_delta": avg([float(v["value_delta"]) for v in vals]),
        "letter_delta": avg([float(v["letter_delta"]) for v in vals]),
        "value_top1_delta": avg([float(v["value_top1_delta"]) for v in vals]),
        "letter_top1_delta": avg([float(v["letter_top1_delta"]) for v in vals]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_node": defaultdict(list),
        "by_position": defaultdict(list),
        "by_condition": defaultdict(list),
        "by_node_position": defaultdict(list),
        "by_node_condition": defaultdict(list),
        "by_node_position_condition": defaultdict(list),
    }
    for row in rows:
        node = f"L{row['layer']}:{row['component']}"
        groups["by_node"][node].append(row)
        groups["by_position"][row["position_kind"]].append(row)
        groups["by_condition"][row["condition"]].append(row)
        groups["by_node_position"][(node, row["position_kind"])].append(row)
        groups["by_node_condition"][(node, row["condition"])].append(row)
        groups["by_node_position_condition"][(node, row["position_kind"], row["condition"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE97_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    parsed_nodes = parse_nodes(args.nodes)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    positions = parse_csv(args.positions)
    donor_kinds = parse_csv(args.donor_kinds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase97_token_route_local_patch.json"
    partial_path = out_dir / f"{args.model}_phase97_token_route_local_patch.partial.json"
    results: dict[str, Any] = {
        "phase": 97,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "token_route_local_patch",
        "nodes": [f"{l}:{c}" for l, c in parsed_nodes],
        "num_items": len(items),
        "positions": positions,
        "donor_kinds": donor_kinds,
        "rows": [],
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 97 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed = {
        (
            int(r["layer"]),
            r["component"],
            int(r["item_idx"]),
            r["position_kind"],
            r["condition"],
        )
        for r in results["rows"]
    }
    t0 = time.time()
    log(f"Phase97 model={args.model} items={len(items)} nodes={parsed_nodes} positions={positions} donors={donor_kinds}")
    for layer_idx, component in parsed_nodes:
        clean_cache: dict[int, dict[str, Any]] = {}
        for idx, item in enumerate(items):
            candidates = uniq([item["target"], *item["distractors"][: args.max_distractors]])
            letters = option_letters(len(candidates))
            target_letter = letters[candidates.index(item["target"])]
            clean_cache[idx] = {
                "candidates": candidates,
                "letters": letters,
                "target_letter": target_letter,
                "score": condition_score(
                    model,
                    tokenizer,
                    device,
                    layers,
                    item,
                    None,
                    candidates,
                    letters,
                    target_letter,
                    args.choice_template,
                    args.max_length,
                    layer_idx,
                    component,
                    "clean",
                    "prompt_tail",
                ),
            }
        for position_kind in positions:
            for idx, item in enumerate(items):
                cache = clean_cache[idx]
                for condition, donor_kind in specs(donor_kinds):
                    key = (layer_idx, component, idx, position_kind, condition)
                    if key in completed:
                        continue
                    donor_item = None
                    donor_idx = None
                    if donor_kind:
                        donor_idx = donors[idx].get(donor_kind)
                        if donor_idx is None:
                            continue
                        donor_item = items[donor_idx]
                    patched = condition_score(
                        model,
                        tokenizer,
                        device,
                        layers,
                        item,
                        donor_item,
                        cache["candidates"],
                        cache["letters"],
                        cache["target_letter"],
                        args.choice_template,
                        args.max_length,
                        layer_idx,
                        component,
                        condition,
                        position_kind,
                    )
                    clean = cache["score"]
                    row = {
                        "item_idx": idx,
                        "donor_idx": donor_idx,
                        "condition": condition,
                        "donor_kind": donor_kind,
                        "position_kind": position_kind,
                        "layer": layer_idx,
                        "component": component,
                        "slot": item["slot"],
                        "template_key": item["template_key"],
                        "object": item["object"],
                        "target": item["target"],
                        "clean_value_margin": clean["value_margin"],
                        "patched_value_margin": patched["value_margin"],
                        "clean_letter_margin": clean["letter_margin"],
                        "patched_letter_margin": patched["letter_margin"],
                        "clean_value_top1": clean["value_top1"],
                        "patched_value_top1": patched["value_top1"],
                        "clean_letter_top1": clean["letter_top1"],
                        "patched_letter_top1": patched["letter_top1"],
                        "value_delta": patched["value_margin"] - clean["value_margin"],
                        "letter_delta": patched["letter_margin"] - clean["letter_margin"],
                        "value_top1_delta": float(patched["value_top1"]) - float(clean["value_top1"]),
                        "letter_top1_delta": float(patched["letter_top1"]) - float(clean["letter_top1"]),
                    }
                    results["rows"].append(row)
                    completed.add(key)
                if (idx + 1) % args.progress_every == 0:
                    log(
                        f"model={args.model} node=L{layer_idx}:{component} pos={position_kind} "
                        f"item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s"
                    )
                    partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
                    cleanup_cuda()
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
    parser.add_argument("--max-items", type=int, default=210)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--positions", default="object_span,relation_span,prompt_tail,last4,prefix8")
    parser.add_argument("--donor-kinds", default="same_slot_same_target,same_slot_diff_target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=35)
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
