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
from phase68_object_attribute_natural_exchange import load_model, parse_csv  # noqa: E402
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


def parse_nodes(text: str) -> list[int]:
    return [int(x) for x in parse_csv(text)]


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def local_positions(tokenizer: Any, prompt: str, position_kind: str) -> list[int]:
    ids = token_ids(tokenizer, prompt)
    if not ids:
        return []
    if position_kind == "prompt_tail":
        return [len(ids) - 1]
    if position_kind == "last4":
        return list(range(max(0, len(ids) - 4), len(ids)))
    if position_kind == "prefix8":
        return list(range(min(8, len(ids))))
    raise ValueError(position_kind)


def attention_meta(layer: Any, model: Any) -> tuple[Any, int, int]:
    sa = layer.self_attn
    n_heads = int(getattr(sa, "num_heads", getattr(model.config, "num_attention_heads")))
    in_features = int(sa.o_proj.in_features)
    if in_features % n_heads != 0:
        raise ValueError(f"o_proj.in_features={in_features} not divisible by heads={n_heads}")
    return sa, n_heads, in_features // n_heads


def capture_oproj_input(
    model: Any,
    device: torch.device,
    layers: list[Any],
    text_ids: list[int],
    layer_idx: int,
) -> torch.Tensor:
    input_ids = torch.tensor([text_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}
    sa = layers[layer_idx].self_attn

    def pre_hook(_module: Any, inputs: Any):
        captured["h"] = inputs[0].detach().float().cpu()
        return None

    handle = sa.o_proj.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


def make_head_pre_hook(
    op: str,
    head_idx: int,
    head_dim: int,
    target_positions: list[int],
    donor_input: torch.Tensor | None,
    donor_positions: list[int],
):
    start = head_idx * head_dim
    end = start + head_dim

    def pre_hook(_module: Any, inputs: Any):
        x = inputs[0]
        patched = x.clone()
        tpos = [p for p in target_positions if 0 <= p < patched.shape[1]]
        if not tpos:
            return inputs
        if op == "zero":
            patched[:, tpos, start:end] = 0
        elif op == "transplant":
            if donor_input is None:
                return inputs
            donor = donor_input.to(device=patched.device, dtype=patched.dtype)
            dpos = [p for p in donor_positions if 0 <= p < donor.shape[1]]
            if not dpos:
                return inputs
            if len(dpos) == len(tpos):
                patched[:, tpos, start:end] = donor[:, dpos, start:end]
            else:
                donor_mean = donor[:, dpos, start:end].mean(dim=1, keepdim=True)
                patched[:, tpos, start:end] = donor_mean.expand(-1, len(tpos), -1)
        else:
            raise ValueError(op)
        return (patched,) + tuple(inputs[1:])

    return pre_hook


def fullseq_logprob_head(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    donor_prompt: str | None,
    continuation: str,
    max_length: int,
    layer_idx: int,
    head_idx: int,
    head_dim: int,
    condition: str,
    position_kind: str,
) -> float:
    prompt_ids = token_ids(tokenizer, prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    target_positions = local_positions(tokenizer, prompt, position_kind)
    donor_input = None
    donor_positions: list[int] = []
    op = "clean"
    if condition == "zero":
        op = "zero"
    elif condition.startswith("transplant:"):
        if donor_prompt is None:
            return float("-inf")
        donor_ids = token_ids(tokenizer, donor_prompt) + cont_ids
        if len(donor_ids) > max_length:
            return float("-inf")
        donor_input = capture_oproj_input(model, device, layers, donor_ids, layer_idx)
        donor_positions = local_positions(tokenizer, donor_prompt, position_kind)
        op = "transplant"
    elif condition != "clean":
        raise ValueError(condition)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if op != "clean":
            handles.append(
                layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(
                    make_head_pre_hook(op, head_idx, head_dim, target_positions, donor_input, donor_positions)
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


def letter_score(
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
    head_idx: int,
    head_dim: int,
    condition: str,
    position_kind: str,
) -> dict[str, Any]:
    prompt = build_choice_prompt(choice_template, item["prompt"], candidates)
    donor_prompt = build_choice_prompt(choice_template, donor_item["prompt"], candidates) if donor_item else None
    scores = {}
    for letter in letters:
        scores[letter] = fullseq_logprob_head(
            model,
            tokenizer,
            device,
            layers,
            prompt,
            donor_prompt,
            letter,
            max_length,
            layer_idx,
            head_idx,
            head_dim,
            condition,
            position_kind,
        )
    stats = score_stats(scores, target_letter, letters)
    return {
        "letter_margin": stats["margin"],
        "letter_top1": stats["top1"],
    }


def specs(donor_kinds: list[str]) -> list[tuple[str, str]]:
    out = [("zero", "")]
    out.extend((f"transplant:{kind}", kind) for kind in donor_kinds)
    return out


def group_summary(vals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(vals),
        "letter_delta": avg([float(v["letter_delta"]) for v in vals]),
        "letter_top1_delta": avg([float(v["letter_top1_delta"]) for v in vals]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[Any, list[dict[str, Any]]]] = {
        "by_layer": defaultdict(list),
        "by_position": defaultdict(list),
        "by_condition": defaultdict(list),
        "by_head": defaultdict(list),
        "by_layer_head": defaultdict(list),
        "by_layer_head_position": defaultdict(list),
        "by_layer_head_position_condition": defaultdict(list),
    }
    for row in rows:
        layer = f"L{row['layer']}"
        groups["by_layer"][layer].append(row)
        groups["by_position"][row["position_kind"]].append(row)
        groups["by_condition"][row["condition"]].append(row)
        groups["by_head"][row["head"]].append(row)
        groups["by_layer_head"][(layer, row["head"])].append(row)
        groups["by_layer_head_position"][(layer, row["head"], row["position_kind"])].append(row)
        groups["by_layer_head_position_condition"][(layer, row["head"], row["position_kind"], row["condition"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def parse_heads(head_text: str, n_heads: int) -> list[int]:
    if not head_text or head_text == "all":
        return list(range(n_heads))
    heads = [int(x) for x in parse_csv(head_text)]
    return [h for h in heads if 0 <= h < n_heads]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE98_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    layer_ids = parse_nodes(args.layers)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    positions = parse_csv(args.positions)
    donor_kinds = parse_csv(args.donor_kinds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase98_head_readout_route_mapping.json"
    partial_path = out_dir / f"{args.model}_phase98_head_readout_route_mapping.partial.json"
    results: dict[str, Any] = {
        "phase": 98,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "head_readout_route_mapping",
        "layers": layer_ids,
        "num_items": len(items),
        "positions": positions,
        "donor_kinds": donor_kinds,
        "rows": [],
        "head_meta": {},
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 98 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results.setdefault("head_meta", {})
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")

    completed = {
        (
            int(r["layer"]),
            int(r["head"]),
            int(r["item_idx"]),
            r["position_kind"],
            r["condition"],
        )
        for r in results["rows"]
    }
    t0 = time.time()
    log(f"Phase98 model={args.model} items={len(items)} layers={layer_ids} positions={positions} donors={donor_kinds}")
    for layer_idx in layer_ids:
        _sa, n_heads, head_dim = attention_meta(layers[layer_idx], model)
        heads = parse_heads(args.heads, n_heads)
        results["head_meta"][f"L{layer_idx}"] = {"n_heads": n_heads, "head_dim": head_dim, "tested_heads": heads}
        clean_cache: dict[int, dict[str, Any]] = {}
        for idx, item in enumerate(items):
            candidates = uniq([item["target"], *item["distractors"][: args.max_distractors]])
            letters = option_letters(len(candidates))
            target_letter = letters[candidates.index(item["target"])]
            clean_cache[idx] = {
                "candidates": candidates,
                "letters": letters,
                "target_letter": target_letter,
                "score": letter_score(
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
                    heads[0],
                    head_dim,
                    "clean",
                    "prompt_tail",
                ),
            }
        for head_idx in heads:
            for position_kind in positions:
                for idx, item in enumerate(items):
                    cache = clean_cache[idx]
                    for condition, donor_kind in specs(donor_kinds):
                        key = (layer_idx, head_idx, idx, position_kind, condition)
                        if key in completed:
                            continue
                        donor_item = None
                        donor_idx = None
                        if donor_kind:
                            donor_idx = donors[idx].get(donor_kind)
                            if donor_idx is None:
                                continue
                            donor_item = items[donor_idx]
                        patched = letter_score(
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
                            head_idx,
                            head_dim,
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
                            "head": head_idx,
                            "slot": item["slot"],
                            "template_key": item["template_key"],
                            "object": item["object"],
                            "target": item["target"],
                            "clean_letter_margin": clean["letter_margin"],
                            "patched_letter_margin": patched["letter_margin"],
                            "clean_letter_top1": clean["letter_top1"],
                            "patched_letter_top1": patched["letter_top1"],
                            "letter_delta": patched["letter_margin"] - clean["letter_margin"],
                            "letter_top1_delta": float(patched["letter_top1"]) - float(clean["letter_top1"]),
                        }
                        results["rows"].append(row)
                        completed.add(key)
                    if (idx + 1) % args.progress_every == 0:
                        log(
                            f"model={args.model} layer=L{layer_idx} head={head_idx}/{heads[-1]} pos={position_kind} "
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
    parser.add_argument("--layers", required=True)
    parser.add_argument("--heads", default="all")
    parser.add_argument("--slots", default="category,color,function,material,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=105)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--positions", default="prompt_tail,last4")
    parser.add_argument("--donor-kinds", default="same_slot_diff_target")
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
