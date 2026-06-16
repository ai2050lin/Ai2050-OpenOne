#!/usr/bin/env python3
"""
Phase 510: surface-format axis and stepwise token probe.

Phase509 showed that some Phi_perp causal subspaces are rotation-stable, but
the surface/token path was still only a one-step logit probe. This script
connects support/release axes to a 3-step generation trace.

It uses category-completion prompts that do not contain the answer category:
  "The apple belongs to the category of"

For each focused category it:
  1. Builds Phi_perp SVD basis from train rich-neutral prompts.
  2. Finds support/release axes by one-step category D on heldout prompts.
  3. Runs clean/remove/add interventions for support and release axes.
  4. Records step1/step2/step3 category, punctuation, generic, object-copy
     margins and greedy tokens.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase507_orthogonal_field import CATEGORIES, get_norm_g, get_token_ids, load_bf16_auto  # noqa: E402
from phase508_orthogonal_field_basis_decomposition import (  # noqa: E402
    NEUTRAL_TEMPLATES,
    RICH_TEMPLATES,
    batched_hidden,
    build_cat_meta,
    build_examples,
    svd_basis,
)
from phase509_rotation_stable_orthogonal_field import FOCUS_CATEGORIES, make_candidate_axes  # noqa: E402


OUT_ROOT = Path("results/glm5_phase510_surface_format_stepwise_probe")
GEN_TEMPLATES = [
    ("category_of", "The {obj} belongs to the category of"),
    ("taxonomy_as", "In taxonomy, {obj} is classified as"),
    ("classify_colon", "Classify {obj}:"),
]
PUNCTUATION = [".", ",", ":", ";", "\n"]
GENERIC = [" thing", " item", " type", " kind", " object", " entity"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    ids: list[int] = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if toks:
            ids.append(int(toks[0]))
    return sorted(set(ids))


def build_generation_examples(cat: str, train_n: int, test_n: int) -> list[dict[str, Any]]:
    objs = CATEGORIES[cat]["objects"][train_n: train_n + test_n]
    rows = []
    for obj in objs:
        for tid, (name, tpl) in enumerate(GEN_TEMPLATES):
            rows.append({
                "cat": cat,
                "obj": obj,
                "template_id": tid,
                "template_name": name,
                "prompt": tpl.format(obj=obj),
            })
    return rows


def max_for_ids(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    if not ids:
        return torch.full((logits.shape[0],), -1e9, device=logits.device, dtype=logits.dtype)
    valid = [i for i in ids if 0 <= i < logits.shape[-1]]
    if not valid:
        return torch.full((logits.shape[0],), -1e9, device=logits.device, dtype=logits.dtype)
    return logits[:, valid].max(dim=1).values


def mean_for_ids_np(logits: np.ndarray, ids: list[int]) -> np.ndarray:
    valid = [i for i in ids if 0 <= i < logits.shape[1]]
    if not valid:
        return np.zeros(logits.shape[0], dtype=np.float32)
    return logits[:, valid].mean(axis=1)


def build_token_groups(tokenizer: Any, cat: str, objects: list[str], competitor_cats: list[str]) -> dict[str, list[int]]:
    return {
        "category": token_ids(tokenizer, [cat, " " + cat]),
        "competitor_category": token_ids(tokenizer, competitor_cats + [" " + c for c in competitor_cats]),
        "punctuation": token_ids(tokenizer, PUNCTUATION),
        "generic": token_ids(tokenizer, GENERIC),
        "object_copy": token_ids(tokenizer, objects + [" " + o for o in objects]),
    }


def logits_with_axis_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    layer_id: int,
    axis: np.ndarray | None,
    mode: str,
    scale: float,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outs = []
    module_index = max(0, min(layer_id - 1, len(layers) - 1))
    axis_t = None if axis is None else torch.tensor(axis, device=device, dtype=torch.float32)
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start:start + batch_size]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handle = None
        if axis_t is not None and mode != "clean":
            pos_t = pos.to(device)

            def hook(_module, _inp, output):
                sign = -1.0 if mode.startswith("remove") else 1.0
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    cur = hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)].float()
                    a = axis_t.to(hs.device)
                    proj = (cur @ a)[:, None] * a[None, :]
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                    return (hs,) + output[1:]
                hs = output.clone()
                cur = hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)].float()
                a = axis_t.to(hs.device)
                proj = (cur @ a)[:, None] * a[None, :]
                hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                return hs

            handle = layers[module_index].register_forward_hook(hook)
        with torch.no_grad():
            out = model(**batch, return_dict=True, use_cache=False)
        if handle is not None:
            handle.remove()
        logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos.to(out.logits.device)]
        outs.append(logits.float().cpu().numpy().astype(np.float32))
        del out, batch
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def choose_axes(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    gen_examples: list[dict[str, Any]],
    hidden_layers: list[int],
    bases: dict[int, np.ndarray],
    token_groups: dict[str, list[int]],
    batch_size: int,
    max_length: int,
    scale: float,
    candidate_random_axes: int,
) -> dict[str, Any]:
    prompts = [x["prompt"] for x in gen_examples]
    clean_logits_cache: dict[int, np.ndarray] = {}
    candidates = []
    for layer_id in hidden_layers:
        clean_logits = logits_with_axis_condition(
            model, tokenizer, device, layers, prompts, layer_id, None, "clean", scale, batch_size, max_length
        )
        clean_logits_cache[layer_id] = clean_logits
        base_d = mean_for_ids_np(clean_logits, token_groups["category"]) - mean_for_ids_np(clean_logits, token_groups["competitor_category"])
        axes = make_candidate_axes(bases[layer_id], 51000 + layer_id, candidate_random_axes)
        for ax in axes:
            logits = logits_with_axis_condition(
                model, tokenizer, device, layers, prompts, layer_id, ax["vec"], "remove", scale, batch_size, max_length
            )
            d = mean_for_ids_np(logits, token_groups["category"]) - mean_for_ids_np(logits, token_groups["competitor_category"])
            candidates.append({
                "layer": layer_id,
                "name": ax["name"],
                "axis": ax["vec"].astype(np.float32),
                "delta_D": float(np.mean(d - base_d)),
            })
    support = min(candidates, key=lambda x: x["delta_D"])
    release = max(candidates, key=lambda x: x["delta_D"])
    return {
        "support": support,
        "release": release,
        "candidate_count": len(candidates),
        "clean_logits_cache": clean_logits_cache,
    }


def stepwise_trace(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    examples: list[dict[str, Any]],
    layer_id: int,
    axis: np.ndarray | None,
    mode: str,
    scale: float,
    token_groups: dict[str, list[int]],
    steps: int,
    batch_size: int,
    max_length: int,
) -> dict[str, Any]:
    prompts = [x["prompt"] for x in examples]
    cur = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    generated: list[list[int]] = [[] for _ in prompts]
    step_rows = []
    module_index = max(0, min(layer_id - 1, len(layers) - 1))
    axis_t = None if axis is None else torch.tensor(axis, device=device, dtype=torch.float32)

    for step in range(steps):
        all_logits = []
        all_next = []
        input_ids = cur["input_ids"]
        attn = cur["attention_mask"]
        for start in range(0, input_ids.shape[0], batch_size):
            sub_ids = input_ids[start:start + batch_size]
            sub_attn = attn[start:start + batch_size]
            pos = sub_attn.sum(dim=1) - 1
            handle = None
            if axis_t is not None and mode != "clean":
                pos_t = pos.to(device)

                def hook(_module, _inp, output):
                    sign = -1.0 if mode.startswith("remove") else 1.0
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)].float()
                        a = axis_t.to(hs.device)
                        proj = (cur_h @ a)[:, None] * a[None, :]
                        hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                        return (hs,) + output[1:]
                    hs = output.clone()
                    cur_h = hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)].float()
                    a = axis_t.to(hs.device)
                    proj = (cur_h @ a)[:, None] * a[None, :]
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_t.to(hs.device)] += (sign * scale * proj).to(hs.dtype)
                    return hs

                handle = layers[module_index].register_forward_hook(hook)
            with torch.no_grad():
                out = model(input_ids=sub_ids, attention_mask=sub_attn, return_dict=True, use_cache=False)
            if handle is not None:
                handle.remove()
            logits = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), pos.to(out.logits.device)]
            next_ids = logits.argmax(dim=1)
            all_logits.append(logits.detach())
            all_next.append(next_ids.detach())
            del out
        logits_step = torch.cat(all_logits, dim=0)
        next_step = torch.cat(all_next, dim=0)
        for i, tid in enumerate(next_step.detach().cpu().tolist()):
            generated[i].append(int(tid))
        cat = max_for_ids(logits_step, token_groups["category"])
        comp = max_for_ids(logits_step, token_groups["competitor_category"])
        punct = max_for_ids(logits_step, token_groups["punctuation"])
        generic = max_for_ids(logits_step, token_groups["generic"])
        obj = max_for_ids(logits_step, token_groups["object_copy"])
        step_rows.append({
            "step": step + 1,
            "category_mean": float(cat.mean().item()),
            "competitor_mean": float(comp.mean().item()),
            "punctuation_mean": float(punct.mean().item()),
            "generic_mean": float(generic.mean().item()),
            "object_copy_mean": float(obj.mean().item()),
            "category_vs_competitor": float((cat - comp).mean().item()),
            "category_vs_punctuation": float((cat - punct).mean().item()),
            "category_vs_generic": float((cat - generic).mean().item()),
            "category_vs_object_copy": float((cat - obj).mean().item()),
            "category_top1_rate": float(torch.isin(next_step, torch.tensor(token_groups["category"], device=next_step.device)).float().mean().item()) if token_groups["category"] else 0.0,
            "punctuation_top1_rate": float(torch.isin(next_step, torch.tensor(token_groups["punctuation"], device=next_step.device)).float().mean().item()) if token_groups["punctuation"] else 0.0,
            "generic_top1_rate": float(torch.isin(next_step, torch.tensor(token_groups["generic"], device=next_step.device)).float().mean().item()) if token_groups["generic"] else 0.0,
            "object_copy_top1_rate": float(torch.isin(next_step, torch.tensor(token_groups["object_copy"], device=next_step.device)).float().mean().item()) if token_groups["object_copy"] else 0.0,
            "top_tokens": top_token_counts(tokenizer, next_step.detach().cpu().tolist(), 8),
        })

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.cat([input_ids, next_step[:, None].to(input_ids.device)], dim=1)
        attn = torch.cat([attn, torch.ones((attn.shape[0], 1), device=attn.device, dtype=attn.dtype)], dim=1)
        cur = {"input_ids": input_ids, "attention_mask": attn}
        if pad_id is None:
            pad_id = 0
        del logits_step
        torch.cuda.empty_cache()

    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    return {
        "mode": mode,
        "layer": layer_id,
        "step_metrics": step_rows,
        "generated_samples": decoded[:10],
        "category_hit_rate": float(np.mean([contains_category(x, examples[i]["cat"]) for i, x in enumerate(decoded)])),
    }


def contains_category(text: str, cat: str) -> float:
    return 1.0 if cat.lower() in text.lower() else 0.0


def top_token_counts(tokenizer: Any, ids: list[int], k: int) -> list[list[Any]]:
    counts: dict[str, int] = {}
    for tid in ids:
        tok = tokenizer.decode([int(tid)])
        counts[tok] = counts.get(tok, 0) + 1
    return [[a, b] for a, b in sorted(counts.items(), key=lambda x: -x[1])[:k]]


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device = load_bf16_auto(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        L, d = info.n_layers, info.d_model
        hidden_layers = sorted(set([max(1, min(L, int(x))) for x in [L // 2, 3 * L // 4, L - 3]]))
        W_U = get_W_U(model, args.model).astype(np.float32)
        g = get_norm_g(model, args.model)
        if g is None:
            raise RuntimeError("cannot read final norm gain")
        cat_meta = build_cat_meta(tokenizer, W_U, g.astype(np.float32), d)
        categories = args.categories.split(",") if args.categories else FOCUS_CATEGORIES[args.model]
        log(f"{args.model}: L={L}, d={d}, categories={categories}, layers={hidden_layers}")

        result = {
            "phase": 510,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "L": L,
            "d_model": d,
            "categories": categories,
            "train_objects": args.train_objects,
            "test_objects": args.test_objects,
            "templates": [x[0] for x in GEN_TEMPLATES],
            "basis_templates": [x[0] for x in RICH_TEMPLATES],
            "layers": hidden_layers,
            "rank": args.rank,
            "steps": args.steps,
            "scale": args.scale,
            "category_results": {},
        }

        for ci, cat in enumerate(categories, 1):
            log(f"{args.model}: category {ci}/{len(categories)} {cat}")
            train_ex, _ = build_examples(cat, args.train_objects, args.test_objects)
            rich_train = [x["rich"] for x in train_ex]
            neutral_train = [x["neutral"] for x in train_ex]
            gen_examples = build_generation_examples(cat, args.train_objects, args.test_objects)
            q_hat = cat_meta[cat]["q_hat"]
            train_r = batched_hidden(model, tokenizer, device, rich_train, hidden_layers, args.batch_size, args.max_length)
            train_n = batched_hidden(model, tokenizer, device, neutral_train, hidden_layers, args.batch_size, args.max_length)
            bases = {}
            for layer_id in hidden_layers:
                phi_train = train_r[layer_id] - train_n[layer_id]
                para_train = (phi_train @ q_hat)[:, None] * q_hat[None, :]
                perp_train = (phi_train - para_train).astype(np.float32)
                basis, _singular_values, _var_ratio = svd_basis(perp_train, args.rank)
                bases[layer_id] = basis

            objects = [x["obj"] for x in gen_examples]
            competitor_cats = [c for c in CATEGORIES if c != cat]
            token_groups = build_token_groups(tokenizer, cat, objects, competitor_cats)
            chosen = choose_axes(
                model, tokenizer, device, layers, gen_examples, hidden_layers, bases,
                token_groups, args.batch_size, args.max_length, args.scale, args.candidate_random_axes
            )
            support = chosen["support"]
            release = chosen["release"]
            log(
                f"  axes: support {support['name']} L{support['layer']} ΔD={support['delta_D']:+.3f}; "
                f"release {release['name']} L{release['layer']} ΔD={release['delta_D']:+.3f}"
            )

            traces = {}
            trace_specs = [
                ("clean", None, None, "clean"),
                ("remove_support", support["axis"], support["layer"], "remove_support"),
                ("add_support", support["axis"], support["layer"], "add_support"),
                ("remove_release", release["axis"], release["layer"], "remove_release"),
                ("add_release", release["axis"], release["layer"], "add_release"),
            ]
            for name, axis, layer_id, mode in trace_specs:
                traces[name] = stepwise_trace(
                    model, tokenizer, device, layers, gen_examples,
                    layer_id if layer_id is not None else hidden_layers[-1],
                    axis, mode, args.scale, token_groups, args.steps,
                    args.batch_size, args.max_length,
                )
            result["category_results"][cat] = {
                "n_generation_prompts": len(gen_examples),
                "chosen_axes": {
                    "support": {k: v for k, v in support.items() if k != "axis"},
                    "release": {k: v for k, v in release.items() if k != "axis"},
                    "candidate_count": chosen["candidate_count"],
                },
                "token_group_sizes": {k: len(v) for k, v in token_groups.items()},
                "traces": traces,
            }
            del train_r, train_n
            gc.collect()
            torch.cuda.empty_cache()

        return result
    finally:
        release_model(model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--train-objects", type=int, default=20)
    parser.add_argument("--test-objects", type=int, default=10)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--candidate-random-axes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--categories", default="")
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_model(args)
    path = out_dir / f"phase510_{args.model}_surface_format_stepwise_probe.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
