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
from phase94_factor_subspace_closure import orthonormal_basis, pool_tensor, project  # noqa: E402
from phase101_value_choice_bridge_mapping import attention_meta, capture_oproj_input, make_choice_pre_hook, parse_heads  # noqa: E402
from phase102_value_factor_bridge_decomposition import capture_component_output, local_positions, make_value_hook, module_name, token_ids  # noqa: E402


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
        layer_s, comp = raw.split(":", 1)
        if comp not in {"attn", "mlp", "choice_heads"}:
            raise ValueError(f"bad node={raw}")
        out.append((int(layer_s), comp))
    return out


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def score_stats(scores: dict[str, float], target: str, candidates: list[str]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    competitors = [x for x in candidates if x != target]
    max_comp = max((scores[x] for x in competitors), default=-1e9)
    return {"top1": bool(ordered and ordered[0][0] == target), "margin": float(scores.get(target, -1e9) - max_comp)}


def orth_basis_from_rows(rows: list[torch.Tensor], dim: int, rank: int) -> torch.Tensor:
    return orthonormal_basis(torch.stack(rows) if rows else torch.empty((0, dim), dtype=torch.float32), rank)


def build_value_bases(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    items: list[dict[str, Any]],
    donors: dict[int, dict[str, int | None]],
    layer_idx: int,
    component: str,
    rank: int,
    pool_mode: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    vecs: list[torch.Tensor | None] = []
    for item in items:
        ids = token_ids(tokenizer, item["prompt"]) + token_ids(tokenizer, " " + item["target"])
        if len(ids) > max_length:
            vecs.append(None)
            continue
        vecs.append(pool_tensor(capture_component_output(model, device, layers, ids, layer_idx, component), pool_mode))
    dim = next((int(v.shape[0]) for v in vecs if v is not None), 0)
    rows_all = []
    rows_by_slot: dict[str, list[torch.Tensor]] = defaultdict(list)
    for idx, item in enumerate(items):
        j = donors[idx].get("same_slot_diff_target")
        if j is None:
            continue
        a = vecs[idx]
        b = vecs[int(j)]
        if a is None or b is None:
            continue
        row = a - b
        rows_all.append(row)
        rows_by_slot[item["slot"]].append(row)
    bases = {"value_all": orth_basis_from_rows(rows_all, dim, rank)}
    for slot, rows in rows_by_slot.items():
        bases[f"value_{slot}"] = orth_basis_from_rows(rows, dim, rank)
    return bases


def capture_restore_output(
    model: Any,
    device: torch.device,
    layers: list[Any],
    full_ids: list[int],
    restore_layer: int,
    restore_component: str,
) -> torch.Tensor:
    return capture_component_output(model, device, layers, full_ids, restore_layer, restore_component)


def make_restore_hook(clean_output: torch.Tensor):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        clean = clean_output.to(device=hs.device, dtype=hs.dtype)
        patched = clean if clean.shape == hs.shape else hs
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    return hook_fn


def fullseq_logprob(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    continuation: str,
    max_length: int,
    value_layer: int,
    value_component: str,
    value_position: str,
    basis: torch.Tensor | None,
    restore_node: tuple[int, str] | None,
    choice_heads: list[int],
    choice_position: str,
    condition: str,
) -> float:
    prompt_ids = token_ids(tokenizer, prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    value_positions = local_positions(tokenizer, prompt, value_position)
    restore_output = None
    clean_choice_input = None
    head_dim = None
    if condition.startswith("destroy_restore:") and restore_node is not None:
        r_layer, r_comp = restore_node
        if r_comp == "choice_heads":
            _n_heads, head_dim = attention_meta(layers[r_layer], model)
            clean_choice_input = capture_oproj_input(model, device, layers, full_ids, r_layer)
        else:
            restore_output = capture_restore_output(model, device, layers, full_ids, r_layer, r_comp)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if condition != "clean":
            handles.append(
                get_module(layers[value_layer], module_name(value_component)).register_forward_hook(
                    make_value_hook("factor_destroy", value_positions, None, [], basis, "both")
                )
            )
        if condition.startswith("destroy_restore:") and restore_node is not None:
            r_layer, r_comp = restore_node
            if r_comp == "choice_heads":
                handles.append(
                    layers[r_layer].self_attn.o_proj.register_forward_pre_hook(
                        make_choice_pre_hook(
                            "choice_restore_clean_heads",
                            choice_heads,
                            int(head_dim),
                            local_positions(tokenizer, prompt, choice_position),
                            clean_choice_input,
                            None,
                            [],
                        )
                    )
                )
            else:
                handles.append(get_module(layers[r_layer], module_name(r_comp)).register_forward_hook(make_restore_hook(restore_output)))
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


def joint_score(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    candidates: list[str],
    letters: list[str],
    target_letter: str,
    args: argparse.Namespace,
    basis: torch.Tensor | None,
    condition: str,
    restore_node: tuple[int, str] | None,
) -> dict[str, Any]:
    heads = parse_heads(args.choice_heads)
    value_scores = {}
    for value in candidates:
        value_scores[value] = fullseq_logprob(
            model,
            tokenizer,
            device,
            layers,
            item["prompt"],
            value,
            args.max_length,
            args.value_layer,
            args.value_component,
            args.value_position,
            basis,
            restore_node,
            heads,
            args.choice_position,
            condition,
        )
    letter_prompt = build_choice_prompt(args.choice_template, item["prompt"], candidates)
    letter_scores = {}
    for letter in letters:
        letter_scores[letter] = fullseq_logprob(
            model,
            tokenizer,
            device,
            layers,
            letter_prompt,
            letter,
            args.max_length,
            args.value_layer,
            args.value_component,
            args.value_position,
            basis,
            restore_node,
            heads,
            args.choice_position,
            condition,
        )
    vs = score_stats(value_scores, item["target"], candidates)
    ls = score_stats(letter_scores, target_letter, letters)
    return {"value_margin": vs["margin"], "value_top1": vs["top1"], "letter_margin": ls["margin"], "letter_top1": ls["top1"]}


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
        "by_condition": defaultdict(list),
        "by_restore_node": defaultdict(list),
        "by_slot_condition": defaultdict(list),
        "by_factor_condition": defaultdict(list),
    }
    for row in rows:
        groups["by_condition"][row["condition"]].append(row)
        groups["by_restore_node"][row["restore_node"]].append(row)
        groups["by_slot_condition"][(row["slot"], row["condition"])].append(row)
        groups["by_factor_condition"][(row["factor"], row["condition"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE103_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    restore_nodes = parse_nodes(args.restore_nodes)
    factor_keys = parse_csv(args.factors)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase103_bridge_localization_restore_sweep.json"
    partial_path = out_dir / f"{args.model}_phase103_bridge_localization_restore_sweep.partial.json"
    log(f"building bases model={args.model} items={len(items)} value=L{args.value_layer}:{args.value_component} factors={factor_keys}")
    bases = build_value_bases(model, tokenizer, device, layers, items, donors, args.value_layer, args.value_component, args.rank, args.pool_mode, args.max_length)
    results: dict[str, Any] = {
        "phase": 103,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "bridge_localization_restore_sweep",
        "value_layer": args.value_layer,
        "value_component": args.value_component,
        "value_position": args.value_position,
        "choice_heads": parse_heads(args.choice_heads),
        "choice_position": args.choice_position,
        "restore_nodes": [f"L{l}:{c}" for l, c in restore_nodes],
        "factors": factor_keys,
        "rank": args.rank,
        "num_items": len(items),
        "basis_dims": {k: int(v.shape[1]) for k, v in bases.items()},
        "rows": [],
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 103 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed = {(int(r["item_idx"]), r["factor"], r["condition"]) for r in results["rows"]}
    clean_cache: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    for idx, item in enumerate(items):
        candidates = uniq([item["target"], *item["distractors"][: args.max_distractors]])
        letters = option_letters(len(candidates))
        target_letter = letters[candidates.index(item["target"])]
        clean_cache[idx] = {
            "candidates": candidates,
            "letters": letters,
            "target_letter": target_letter,
            "score": joint_score(model, tokenizer, device, layers, item, candidates, letters, target_letter, args, None, "clean", None),
        }
    log(f"Phase103 model={args.model} items={len(items)} restore_nodes={len(restore_nodes)}")
    for idx, item in enumerate(items):
        cache = clean_cache[idx]
        for factor in factor_keys:
            basis_key = f"value_{item['slot']}" if factor == "own" else factor
            basis = bases.get(basis_key)
            if basis is None or basis.numel() == 0:
                continue
            specs: list[tuple[str, tuple[int, str] | None]] = [("destroy_only", None)]
            specs += [(f"destroy_restore:L{l}:{c}", (l, c)) for l, c in restore_nodes]
            for condition, restore_node in specs:
                key = (idx, factor, condition)
                if key in completed:
                    continue
                patched = joint_score(model, tokenizer, device, layers, item, cache["candidates"], cache["letters"], cache["target_letter"], args, basis, condition, restore_node)
                clean = cache["score"]
                row = {
                    "item_idx": idx,
                    "condition": condition,
                    "restore_node": "none" if restore_node is None else f"L{restore_node[0]}:{restore_node[1]}",
                    "factor": factor,
                    "basis_key": basis_key,
                    "slot": item["slot"],
                    "template_key": item["template_key"],
                    "object": item["object"],
                    "target": item["target"],
                    "clean_value_margin": clean["value_margin"],
                    "patched_value_margin": patched["value_margin"],
                    "value_delta": patched["value_margin"] - clean["value_margin"],
                    "clean_letter_margin": clean["letter_margin"],
                    "patched_letter_margin": patched["letter_margin"],
                    "letter_delta": patched["letter_margin"] - clean["letter_margin"],
                    "clean_value_top1": clean["value_top1"],
                    "patched_value_top1": patched["value_top1"],
                    "value_top1_delta": float(patched["value_top1"]) - float(clean["value_top1"]),
                    "clean_letter_top1": clean["letter_top1"],
                    "patched_letter_top1": patched["letter_top1"],
                    "letter_top1_delta": float(patched["letter_top1"]) - float(clean["letter_top1"]),
                }
                results["rows"].append(row)
                completed.add(key)
        if (idx + 1) % args.progress_every == 0:
            log(f"model={args.model} item={idx + 1}/{len(items)} rows={len(results['rows'])} elapsed={time.time() - t0:.0f}s")
            partial_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            cleanup_cuda()
    results["summary"] = summarize(results["rows"])
    final_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {final_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--value-layer", type=int, required=True)
    parser.add_argument("--value-component", choices=["attn", "mlp"], required=True)
    parser.add_argument("--value-position", default="prefix8")
    parser.add_argument("--restore-nodes", required=True)
    parser.add_argument("--choice-heads", required=True)
    parser.add_argument("--choice-position", default="prompt_tail")
    parser.add_argument("--factors", default="value_all,own")
    parser.add_argument("--slots", default="category,function,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=180)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pool-mode", default="prefix")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=10)
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
