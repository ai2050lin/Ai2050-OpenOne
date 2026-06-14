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
from phase94_factor_subspace_closure import align_like, orthonormal_basis, pool_tensor, project  # noqa: E402


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


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_choice_prompt(template_key: str, prompt: str, candidates: list[str]) -> str:
    return choice_templates()[template_key].format(clean_prompt=prompt, options=render_options(candidates))


def local_positions(tokenizer: Any, prompt: str, position_kind: str) -> list[int]:
    ids = token_ids(tokenizer, prompt)
    if not ids:
        return []
    if position_kind == "prefix8":
        return list(range(min(8, len(ids))))
    if position_kind == "prompt_tail":
        return [len(ids) - 1]
    if position_kind == "last4":
        return list(range(max(0, len(ids) - 4), len(ids)))
    raise ValueError(position_kind)


def module_name(component: str) -> str:
    if component == "attn":
        return "attn_out"
    if component == "mlp":
        return "mlp_out"
    raise ValueError(component)


def attention_meta(layer: Any, model: Any) -> tuple[int, int]:
    sa = layer.self_attn
    n_heads = int(getattr(sa, "num_heads", getattr(model.config, "num_attention_heads")))
    in_features = int(sa.o_proj.in_features)
    if in_features % n_heads != 0:
        raise ValueError(f"o_proj.in_features={in_features} not divisible by heads={n_heads}")
    return n_heads, in_features // n_heads


def parse_heads(text: str) -> list[int]:
    return [int(x) for x in parse_csv(text)]


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
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs.detach().float().cpu()

    handle = get_module(layers[layer_idx], module_name(component)).register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


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

    def pre_hook(_module: Any, inputs: Any):
        captured["h"] = inputs[0].detach().float().cpu()
        return None

    handle = layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()
    return captured["h"]


def make_value_hook(
    op: str,
    positions: list[int],
    donor_output: torch.Tensor | None,
    donor_positions: list[int],
    basis: torch.Tensor | None = None,
    copy_mode: str = "both",
):
    def hook_fn(_module: Any, _inputs: Any, output: Any):
        hs = output[0] if isinstance(output, tuple) else output
        patched = hs.clone()
        tpos = [p for p in positions if 0 <= p < patched.shape[1]]
        if not tpos:
            return output
        if op == "value_zero":
            patched[:, tpos, :] = 0
        elif op == "value_transplant":
            if donor_output is None:
                return output
            donor = donor_output.to(device=patched.device, dtype=patched.dtype)
            dpos = [p for p in donor_positions if 0 <= p < donor.shape[1]]
            if not dpos:
                return output
            block = donor[:, dpos, :]
            patched[:, tpos, :] = block if len(dpos) == len(tpos) else block.mean(dim=1, keepdim=True).expand(-1, len(tpos), -1)
        elif op == "factor_destroy":
            if basis is None:
                return output
            patched = hs - project(hs, basis)
        elif op == "factor_transplant":
            if basis is None or donor_output is None:
                return output
            donor = align_like(hs, donor_output, copy_mode)
            patched = hs - project(hs, basis) + project(donor, basis)
        else:
            raise ValueError(op)
        return (patched,) + output[1:] if isinstance(output, tuple) else patched

    return hook_fn


def factor_spec(value_condition: str, item_slot: str) -> tuple[str, str | None, str | None]:
    if value_condition == "value_clean":
        return "value_clean", None, None
    if value_condition in {"value_zero", "value_transplant"}:
        return value_condition, None, "same_slot_diff_target" if value_condition == "value_transplant" else None
    table = {
        "destroy_own_value": ("factor_destroy", f"value_{item_slot}", None),
        "transplant_own_value": ("factor_transplant", f"value_{item_slot}", "same_slot_diff_target"),
        "destroy_all_value": ("factor_destroy", "value_all", None),
        "transplant_all_value": ("factor_transplant", "value_all", "same_slot_diff_target"),
        "destroy_relation": ("factor_destroy", "relation", None),
        "transplant_relation": ("factor_transplant", "relation", "diff_slot_same_object"),
        "destroy_object": ("factor_destroy", "object", None),
        "transplant_object": ("factor_transplant", "object", "same_slot_same_target"),
    }
    if value_condition not in table:
        raise ValueError(f"unknown value condition={value_condition}")
    return table[value_condition]


def make_choice_pre_hook(
    op: str,
    head_indices: list[int],
    head_dim: int,
    positions: list[int],
    clean_input: torch.Tensor | None,
    donor_input: torch.Tensor | None,
    donor_positions: list[int],
):
    spans = [(h * head_dim, (h + 1) * head_dim) for h in head_indices]

    def pre_hook(_module: Any, inputs: Any):
        x = inputs[0]
        patched = x.clone()
        tpos = [p for p in positions if 0 <= p < patched.shape[1]]
        if not tpos:
            return inputs
        clean = clean_input.to(device=patched.device, dtype=patched.dtype) if clean_input is not None else None
        donor = donor_input.to(device=patched.device, dtype=patched.dtype) if donor_input is not None else None
        dpos = [p for p in donor_positions if donor is not None and 0 <= p < donor.shape[1]]

        if op == "choice_transplant_heads":
            if donor is None or not dpos:
                return inputs
            for start, end in spans:
                block = donor[:, dpos, start:end]
                patched[:, tpos, start:end] = block if len(dpos) == len(tpos) else block.mean(dim=1, keepdim=True).expand(-1, len(tpos), -1)
        elif op == "choice_restore_clean_heads":
            if clean is None:
                return inputs
            for start, end in spans:
                patched[:, tpos, start:end] = clean[:, tpos, start:end]
        elif op == "choice_transplant_all_restore_clean_heads":
            if donor is None or not dpos:
                return inputs
            block = donor[:, dpos, :]
            patched[:, tpos, :] = block if len(dpos) == len(tpos) else block.mean(dim=1, keepdim=True).expand(-1, len(tpos), -1)
            if clean is not None:
                for start, end in spans:
                    patched[:, tpos, start:end] = clean[:, tpos, start:end]
        else:
            raise ValueError(op)
        return (patched,) + tuple(inputs[1:])

    return pre_hook


def parse_condition(raw: str) -> tuple[str, str]:
    if raw == "clean":
        return "value_clean", "choice_clean"
    if "+" in raw:
        a, b = raw.split("+", 1)
        return a, b
    if raw.startswith("value_") or raw.startswith("destroy_") or raw.startswith("transplant_"):
        return raw, "choice_clean"
    return "value_clean", raw


def fullseq_logprob_bridge(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    donor_prompt: str | None,
    continuation: str,
    max_length: int,
    value_layer: int,
    value_component: str,
    value_position: str,
    choice_layer: int,
    choice_heads: list[int],
    choice_head_dim: int,
    choice_position: str,
    condition: str,
    item_slot: str,
    bases: dict[str, torch.Tensor],
    copy_mode: str,
) -> float:
    prompt_ids = token_ids(tokenizer, prompt)
    cont_ids = token_ids(tokenizer, continuation)
    if not cont_ids:
        return float("-inf")
    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        return float("-inf")
    value_condition, choice_op = parse_condition(condition)
    value_op, basis_key, donor_kind = factor_spec(value_condition, item_slot)
    value_basis = bases.get(basis_key or "", None)
    value_positions = local_positions(tokenizer, prompt, value_position)
    choice_positions = local_positions(tokenizer, prompt, choice_position)

    donor_full_ids = None
    donor_value_output = None
    donor_value_positions: list[int] = []
    donor_choice_input = None
    donor_choice_positions: list[int] = []
    clean_choice_input = None

    needs_donor = "transplant" in condition
    if needs_donor:
        if donor_prompt is None:
            return float("-inf")
        donor_full_ids = token_ids(tokenizer, donor_prompt) + cont_ids
        if len(donor_full_ids) > max_length:
            return float("-inf")

    if value_op in {"value_transplant", "factor_transplant"} and donor_full_ids is not None:
        donor_value_output = capture_component_output(model, device, layers, donor_full_ids, value_layer, value_component)
        donor_value_positions = local_positions(tokenizer, donor_prompt or "", value_position)
    if choice_op != "choice_clean":
        clean_choice_input = capture_oproj_input(model, device, layers, full_ids, choice_layer)
    if "transplant" in choice_op and donor_full_ids is not None:
        donor_choice_input = capture_oproj_input(model, device, layers, donor_full_ids, choice_layer)
        donor_choice_positions = local_positions(tokenizer, donor_prompt or "", choice_position)

    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    handles = []
    try:
        if value_op in {"value_zero", "value_transplant", "factor_destroy", "factor_transplant"}:
            handles.append(
                get_module(layers[value_layer], module_name(value_component)).register_forward_hook(
                    make_value_hook(value_op, value_positions, donor_value_output, donor_value_positions, value_basis, copy_mode)
                )
            )
        if choice_op != "choice_clean":
            handles.append(
                layers[choice_layer].self_attn.o_proj.register_forward_pre_hook(
                    make_choice_pre_hook(
                        choice_op,
                        choice_heads,
                        choice_head_dim,
                        choice_positions,
                        clean_choice_input,
                        donor_choice_input,
                        donor_choice_positions,
                    )
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
    return {"top1": bool(ordered and ordered[0][0] == target), "margin": float(scores.get(target, -1e9) - max_comp)}


def joint_score(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    item: dict[str, Any],
    donor_item: dict[str, Any] | None,
    candidates: list[str],
    letters: list[str],
    target_letter: str,
    args: argparse.Namespace,
    condition: str,
    bases: dict[str, torch.Tensor],
) -> dict[str, Any]:
    n_heads, head_dim = attention_meta(layers[args.choice_layer], model)
    heads = parse_heads(args.choice_heads)
    bad = [h for h in heads if h < 0 or h >= n_heads]
    if bad:
        raise ValueError(f"bad heads {bad}; n_heads={n_heads}")
    value_scores = {}
    for value in candidates:
        value_scores[value] = fullseq_logprob_bridge(
            model,
            tokenizer,
            device,
            layers,
            item["prompt"],
            donor_item["prompt"] if donor_item else None,
            value,
            args.max_length,
            args.value_layer,
            args.value_component,
            args.value_position,
            args.choice_layer,
            heads,
            head_dim,
            args.choice_position,
            condition,
            item["slot"],
            bases,
            args.copy_mode,
        )
    letter_prompt = build_choice_prompt(args.choice_template, item["prompt"], candidates)
    donor_letter_prompt = build_choice_prompt(args.choice_template, donor_item["prompt"], candidates) if donor_item else None
    letter_scores = {}
    for letter in letters:
        letter_scores[letter] = fullseq_logprob_bridge(
            model,
            tokenizer,
            device,
            layers,
            letter_prompt,
            donor_letter_prompt,
            letter,
            args.max_length,
            args.value_layer,
            args.value_component,
            args.value_position,
            args.choice_layer,
            heads,
            head_dim,
            args.choice_position,
            condition,
            item["slot"],
            bases,
            args.copy_mode,
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
        "by_slot": defaultdict(list),
        "by_condition_slot": defaultdict(list),
    }
    for row in rows:
        groups["by_condition"][row["condition"]].append(row)
        groups["by_slot"][row["slot"]].append(row)
        groups["by_condition_slot"][(row["condition"], row["slot"])].append(row)
    return {
        key: {":".join(map(str, k if isinstance(k, tuple) else (k,))): group_summary(v) for k, v in group.items()}
        for key, group in groups.items()
    }


def build_value_factor_bases(
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

    def diff_rows(kind: str, slot: str | None = None) -> torch.Tensor:
        rows = []
        for idx, item in enumerate(items):
            if slot is not None and item["slot"] != slot:
                continue
            j = donors[idx].get(kind)
            if j is None:
                continue
            a = vecs[idx]
            b = vecs[int(j)]
            if a is None or b is None:
                continue
            rows.append(a - b)
        return torch.stack(rows) if rows else torch.empty((0, dim), dtype=torch.float32)

    bases: dict[str, torch.Tensor] = {
        "value_all": orthonormal_basis(diff_rows("same_slot_diff_target"), rank),
        "relation": orthonormal_basis(diff_rows("diff_slot_same_object"), rank),
        "object": orthonormal_basis(diff_rows("same_slot_same_target"), rank),
    }
    for slot in sorted({item["slot"] for item in items}):
        bases[f"value_{slot}"] = orthonormal_basis(diff_rows("same_slot_diff_target", slot), rank)
    return bases


def donor_kind_for_condition(condition: str, item_slot: str) -> str | None:
    value_condition, choice_condition = parse_condition(condition)
    _op, _basis, value_donor_kind = factor_spec(value_condition, item_slot)
    if value_donor_kind:
        return value_donor_kind
    if "transplant" in choice_condition:
        return "same_slot_diff_target"
    return None


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_model(args.model, os.environ.get("PHASE102_ATTN_IMPLEMENTATIONS", args.attn_implementations))
    layers = get_layers(model)
    items = build_items(args.max_items, parse_csv(args.slots), parse_csv(args.slot_templates))
    donors = select_donors(items)
    conditions = parse_csv(args.conditions)
    log(f"building value factor bases model={args.model} value=L{args.value_layer}:{args.value_component} rank={args.rank}")
    bases = build_value_factor_bases(
        model, tokenizer, device, layers, items, donors, args.value_layer, args.value_component, args.rank, args.pool_mode, args.max_length
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_phase102_value_factor_bridge_decomposition.json"
    partial_path = out_dir / f"{args.model}_phase102_value_factor_bridge_decomposition.partial.json"
    results: dict[str, Any] = {
        "phase": 102,
        "model": args.model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "task": "value_factor_bridge_decomposition",
        "value_layer": args.value_layer,
        "value_component": args.value_component,
        "value_position": args.value_position,
        "choice_layer": args.choice_layer,
        "choice_heads": parse_heads(args.choice_heads),
        "choice_position": args.choice_position,
        "rank": args.rank,
        "pool_mode": args.pool_mode,
        "copy_mode": args.copy_mode,
        "basis_dims": {k: int(v.shape[1]) for k, v in bases.items()},
        "conditions": conditions,
        "num_items": len(items),
        "rows": [],
        "summary": {},
    }
    if args.resume:
        resume_path = final_path if final_path.exists() else partial_path
        if resume_path.exists():
            loaded = json.loads(resume_path.read_text(encoding="utf-8"))
            if loaded.get("phase") == 102 and loaded.get("model") == args.model:
                results = loaded
                results.setdefault("rows", [])
                results["summary"] = {}
                log(f"resume loaded {resume_path} rows={len(results['rows'])}")
    completed = {(int(r["item_idx"]), r["condition"]) for r in results["rows"]}
    clean_cache: dict[int, dict[str, Any]] = {}
    t0 = time.time()
    log(
        f"Phase102 model={args.model} items={len(items)} value=L{args.value_layer}:{args.value_component}:{args.value_position} "
        f"choice=L{args.choice_layer}:heads={args.choice_heads}:{args.choice_position}"
    )
    for idx, item in enumerate(items):
        candidates = uniq([item["target"], *item["distractors"][: args.max_distractors]])
        letters = option_letters(len(candidates))
        target_letter = letters[candidates.index(item["target"])]
        clean_cache[idx] = {
            "candidates": candidates,
            "letters": letters,
            "target_letter": target_letter,
            "score": joint_score(model, tokenizer, device, layers, item, None, candidates, letters, target_letter, args, "clean", bases),
        }
    for idx, item in enumerate(items):
        cache = clean_cache[idx]
        for condition in conditions:
            if condition == "clean" or (idx, condition) in completed:
                continue
            donor_kind = donor_kind_for_condition(condition, item["slot"])
            donor_idx = donors[idx].get(donor_kind) if donor_kind else None
            donor_item = items[donor_idx] if donor_idx is not None else None
            if donor_kind and donor_item is None:
                continue
            patched = joint_score(model, tokenizer, device, layers, item, donor_item, cache["candidates"], cache["letters"], cache["target_letter"], args, condition, bases)
            clean = cache["score"]
            row = {
                "item_idx": idx,
                "donor_idx": donor_idx,
                "donor_kind": donor_kind or "",
                "condition": condition,
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
            completed.add((idx, condition))
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
    parser.add_argument("--choice-layer", type=int, required=True)
    parser.add_argument("--choice-heads", required=True)
    parser.add_argument("--choice-position", default="prompt_tail")
    parser.add_argument(
        "--conditions",
        default=(
            "destroy_own_value,transplant_own_value,destroy_all_value,transplant_all_value,"
            "destroy_relation,transplant_relation,destroy_object,transplant_object,"
            "destroy_own_value+choice_restore_clean_heads,transplant_own_value+choice_restore_clean_heads,"
            "destroy_all_value+choice_restore_clean_heads,transplant_all_value+choice_restore_clean_heads"
        ),
    )
    parser.add_argument("--slots", default="category,color,function,material,location")
    parser.add_argument("--slot-templates", default="")
    parser.add_argument("--max-items", type=int, default=120)
    parser.add_argument("--max-distractors", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--choice-template", default="choice_json_letter")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--pool-mode", default="prefix")
    parser.add_argument("--copy-mode", choices=["tail", "prefix", "both"], default="both")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=30)
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
