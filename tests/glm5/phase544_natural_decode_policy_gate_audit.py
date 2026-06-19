#!/usr/bin/env python3
"""
Phase 544: natural answer and decode-mode policy gate audit.

Phase543 showed that generation closure is scaffold-conditioned. This phase
tests whether that effect survives outside label-like scaffolds and whether
non-greedy decoding exposes hidden paths that greedy decoding misses.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
from phase530_state_pair_decomposition import hidden_at_layer, load_model_bf16_flash, mean_dir  # noqa: E402
from phase532_multi_seed_controls import normalize  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK, encode_batch  # noqa: E402
from phase539_interface_cluster_mechanism import (  # noqa: E402
    CORE_SOURCES,
    PAIR_SPECS,
    build_candidates,
    build_components,
    layer_windows,
)
from phase542_generation_closure_audit import CONDITIONS, token_groups  # noqa: E402


OUT_ROOT = Path("results/glm5_phase544_natural_decode_policy_gate_audit")
SCAFFOLDS = ["direct", "one_word", "natural_qa", "definition", "sentence_completion"]
DECODE_MODES = ["greedy", "temperature", "top_p", "beam"]
CHECKPOINTS = [1, 3, 5, 10, 12]
FAMILY_TERMS = {
    "vehicle": ["vehicle", "vehicles", "transport", "transportation", "automobile", "automobiles", "car", "truck", "bus", "train", "boat", "airplane"],
    "furniture": ["furniture", "chair", "table", "sofa", "bed", "desk", "cabinet", "shelf", "couch", "wardrobe"],
    "tool": ["tool", "tools", "instrument", "instruments", "device", "devices", "equipment", "hammer", "wrench", "drill"],
    "clothing": ["clothing", "clothes", "garment", "garments", "apparel", "shirt", "pants", "coat", "jacket", "dress"],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stable_offset(*parts: str) -> int:
    text = "|".join(parts)
    acc = 0
    for ch in text:
        acc = (acc * 131 + ord(ch)) % 1000003
    return acc


def scaffold_prompt(scaffold: str, obj: str, pos_label: str, neg_label: str) -> str:
    cap = obj.capitalize()
    if scaffold == "direct":
        return f"The category of {obj} is"
    if scaffold == "one_word":
        return f"Answer with one category word. {cap} is a"
    if scaffold == "natural_qa":
        return f"What kind of thing is a {obj}? A {obj} is"
    if scaffold == "definition":
        return f"A {obj} is best defined as"
    if scaffold == "sentence_completion":
        return f"A {obj} is commonly used as"
    raise ValueError(f"unknown scaffold: {scaffold}")


def build_prompts(test_n: int, scaffolds: list[str]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    for pair in CORE_SOURCES:
        pos_label, neg_label = PAIR_SPECS[pair]
        objects = CATEGORY_BANK[pos_label][-test_n:]
        out[pair] = {}
        for scaffold in scaffolds:
            out[pair][scaffold] = [scaffold_prompt(scaffold, x, pos_label, neg_label) for x in objects]
    return out


def interventions_for(
    components_by_layer: dict[str, dict[str, dict[str, np.ndarray]]],
    source_pair: str,
    window: list[int],
    condition: str,
    alpha: float,
) -> dict[int, tuple[np.ndarray, float]] | None:
    if condition == "baseline":
        return None
    return {layer_id: (components_by_layer[str(layer_id)][source_pair][condition], alpha) for layer_id in window}


def prepare_interventions(interventions: dict[int, tuple[np.ndarray, float]] | None) -> dict[int, torch.Tensor]:
    prepared = {}
    if interventions:
        for layer_id, (direction, alpha) in interventions.items():
            prepared[layer_id] = torch.tensor(normalize(direction) * float(alpha), dtype=torch.bfloat16)
    return prepared


def batched_next_logits(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    texts: list[str],
    prepared: dict[int, torch.Tensor],
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outs = []
    for start in range(0, len(texts), batch_size):
        batch = encode_batch(tokenizer, texts[start:start + batch_size], device, max_length)
        pos = batch["attention_mask"].sum(dim=1) - 1
        handles = []
        for layer_id, d_tensor in prepared.items():
            layer = layers[layer_id]
            layer_device = next(layer.parameters()).device
            d_local = d_tensor.to(layer_device)
            pos_local = pos.to(layer_device)

            def make_hook(d_vec: torch.Tensor, pos_vec: torch.Tensor):
                def hook(_module, _inp, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[torch.arange(hs.shape[0], device=hs.device), pos_vec.to(hs.device)] += d_vec.to(hs.dtype)
                        return (hs,) + output[1:]
                    hs = output.clone()
                    hs[torch.arange(hs.shape[0], device=hs.device), pos_vec.to(hs.device)] += d_vec.to(hs.dtype)
                    return hs
                return hook

            handles.append(layer.register_forward_hook(make_hook(d_local, pos_local)))
        with torch.inference_mode():
            out = model(**batch, return_dict=True, use_cache=False)
            idx = pos.to(out.logits.device)
            rows = out.logits[torch.arange(out.logits.shape[0], device=out.logits.device), idx]
            outs.append(rows.float().cpu().numpy().astype(np.float32))
        for handle in handles:
            handle.remove()
        del out, batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)


def token_type(tok: int, groups: dict[str, list[int]]) -> str:
    for name, ids in groups.items():
        if tok in ids:
            return name
    return "other"


def best_rank(row: np.ndarray, ids: list[int]) -> float:
    valid = [i for i in ids if 0 <= i < row.shape[0]]
    if not valid:
        return float(row.shape[0])
    val = float(np.max(row[valid]))
    return float(1 + np.sum(row > val))


def top_candidates(logits: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_k, logits.shape[0])
    idx = np.argpartition(logits, -k)[-k:]
    vals = logits[idx]
    order = np.argsort(vals)[::-1]
    return idx[order], vals[order]


def softmax(vals: np.ndarray) -> np.ndarray:
    vals = vals.astype(np.float64)
    vals = vals - np.max(vals)
    probs = np.exp(vals)
    denom = float(np.sum(probs))
    if denom <= 0 or not math.isfinite(denom):
        return np.ones_like(vals, dtype=np.float64) / max(1, vals.shape[0])
    return probs / denom


def choose_token(logits: np.ndarray, mode: str, rng: np.random.Generator, temperature: float, top_p: float) -> int:
    if mode == "greedy":
        return int(np.argmax(logits))
    if mode == "temperature":
        idx, vals = top_candidates(logits, 256)
        probs = softmax(vals / max(1e-4, temperature))
        return int(rng.choice(idx, p=probs))
    if mode == "top_p":
        idx, vals = top_candidates(logits, 512)
        probs = softmax(vals / max(1e-4, temperature))
        csum = np.cumsum(probs)
        keep_n = int(np.searchsorted(csum, top_p, side="left") + 1)
        keep_n = max(1, min(keep_n, probs.shape[0]))
        probs = probs[:keep_n] / np.sum(probs[:keep_n])
        return int(rng.choice(idx[:keep_n], p=probs))
    raise ValueError(f"unknown decode mode: {mode}")


def run_linear_decode(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    prepared: dict[int, torch.Tensor],
    groups: dict[str, list[int]],
    mode: str,
    max_new_tokens: int,
    batch_size: int,
    max_length: int,
    rng: np.random.Generator,
    temperature: float,
    top_p: float,
) -> tuple[list[list[int]], list[str], list[str], list[float], list[float]]:
    texts = list(prompts)
    generated: list[list[int]] = [[] for _ in prompts]
    first_types, target_ranks, competitor_ranks = [], [], []
    for step in range(max_new_tokens):
        logits = batched_next_logits(model, tokenizer, device, layers, texts, prepared, batch_size, max_length)
        toks = [choose_token(row, mode, rng, temperature, top_p) for row in logits]
        if step == 0:
            for row, tok in zip(logits, toks):
                first_types.append(token_type(tok, groups))
                target_ranks.append(best_rank(row, groups["target"]))
                competitor_ranks.append(best_rank(row, groups["competitor"]))
        for i, tok in enumerate(toks):
            generated[i].append(tok)
            texts[i] += tokenizer.decode([tok], skip_special_tokens=False)
    suffixes = [texts[i][len(prompts[i]):] for i in range(len(prompts))]
    return generated, suffixes, first_types, target_ranks, competitor_ranks


def beam_one(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt: str,
    prepared: dict[int, torch.Tensor],
    groups: dict[str, list[int]],
    max_new_tokens: int,
    max_length: int,
    beam_width: int,
) -> tuple[list[int], str, str, float, float]:
    beams = [(prompt, [], 0.0)]
    first_type, target_rank, competitor_rank = "other", 0.0, 0.0
    for step in range(max_new_tokens):
        texts = [b[0] for b in beams]
        logits = batched_next_logits(model, tokenizer, device, layers, texts, prepared, beam_width, max_length)
        if step == 0:
            top_tok = int(np.argmax(logits[0]))
            first_type = token_type(top_tok, groups)
            target_rank = best_rank(logits[0], groups["target"])
            competitor_rank = best_rank(logits[0], groups["competitor"])
        new_beams = []
        for (text, ids, score), row in zip(beams, logits):
            idx, vals = top_candidates(row, beam_width)
            probs = softmax(vals)
            for tok, prob in zip(idx, probs):
                tok_i = int(tok)
                new_beams.append((text + tokenizer.decode([tok_i], skip_special_tokens=False), ids + [tok_i], score + float(np.log(max(prob, 1e-12)))))
        new_beams.sort(key=lambda x: x[2], reverse=True)
        beams = new_beams[:beam_width]
    best_text, best_ids, _score = beams[0]
    return best_ids, best_text[len(prompt):], first_type, target_rank, competitor_rank


def run_beam_decode(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    prepared: dict[int, torch.Tensor],
    groups: dict[str, list[int]],
    max_new_tokens: int,
    max_length: int,
    beam_width: int,
) -> tuple[list[list[int]], list[str], list[str], list[float], list[float]]:
    generated, suffixes, first_types, target_ranks, competitor_ranks = [], [], [], [], []
    for prompt in prompts:
        ids, suffix, first_type, target_rank, competitor_rank = beam_one(
            model, tokenizer, device, layers, prompt, prepared, groups, max_new_tokens, max_length, beam_width
        )
        generated.append(ids)
        suffixes.append(suffix)
        first_types.append(first_type)
        target_ranks.append(target_rank)
        competitor_ranks.append(competitor_rank)
    return generated, suffixes, first_types, target_ranks, competitor_ranks


def text_has_any(text: str, terms: list[str]) -> bool:
    low = text.lower()
    return any(term in low for term in terms)


def score_outputs(
    tokenizer: Any,
    generated: list[list[int]],
    suffixes: list[str],
    first_types: list[str],
    target_ranks: list[float],
    competitor_ranks: list[float],
    groups: dict[str, list[int]],
    source_pair: str,
    checkpoints: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    pos_label, neg_label = PAIR_SPECS[source_pair]
    group_sets = {k: set(v) for k, v in groups.items()}
    checkpoint_metrics = {}
    for cp in checkpoints:
        cp = min(cp, max_new_tokens)
        exact = {"target": 0, "competitor": 0, "cluster_other": 0, "off_cluster": 0}
        family = {"target": 0, "competitor": 0}
        other_only = 0
        for ids, suffix in zip(generated, suffixes):
            prefix_ids = ids[:cp]
            prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
            exact_any = False
            for name in exact:
                if any(tok in group_sets[name] for tok in prefix_ids):
                    exact[name] += 1
                    exact_any = True
            if text_has_any(prefix_text, FAMILY_TERMS.get(pos_label, [pos_label])):
                family["target"] += 1
            if text_has_any(prefix_text, FAMILY_TERMS.get(neg_label, [neg_label])):
                family["competitor"] += 1
            if not exact_any and not family["target"] and not family["competitor"]:
                other_only += 1
        n = max(1, len(generated))
        checkpoint_metrics[str(cp)] = {
            "exact_target": float(exact["target"] / n),
            "exact_competitor": float(exact["competitor"] / n),
            "exact_cluster_other": float(exact["cluster_other"] / n),
            "exact_off_cluster": float(exact["off_cluster"] / n),
            "family_target": float(family["target"] / n),
            "family_competitor": float(family["competitor"] / n),
            "other_only": float(other_only / n),
        }
    n = max(1, len(generated))
    first_counts = {k: first_types.count(k) for k in ["target", "competitor", "cluster_other", "off_cluster", "other"]}
    return {
        "n": len(generated),
        "checkpoints": sorted(set(min(x, max_new_tokens) for x in checkpoints)),
        "hit_at_k": checkpoint_metrics,
        "hit_rates": checkpoint_metrics[str(max_new_tokens)],
        "first_type_rates": {k: float(v / n) for k, v in first_counts.items()},
        "mean_first_target_rank": float(np.mean(target_ranks)) if target_ranks else 0.0,
        "mean_first_competitor_rank": float(np.mean(competitor_ranks)) if competitor_ranks else 0.0,
        "sample_outputs": [
            {"generated_suffix": suffixes[i], "generated_ids": generated[i]}
            for i in range(min(4, len(suffixes)))
        ],
    }


def decode_probe(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    interventions: dict[int, tuple[np.ndarray, float]] | None,
    groups: dict[str, list[int]],
    source_pair: str,
    mode: str,
    max_new_tokens: int,
    checkpoints: list[int],
    batch_size: int,
    max_length: int,
    seed: int,
    temperature: float,
    top_p: float,
    beam_width: int,
) -> dict[str, Any]:
    prepared = prepare_interventions(interventions)
    rng = np.random.default_rng(seed)
    if mode == "beam":
        generated, suffixes, first_types, target_ranks, competitor_ranks = run_beam_decode(
            model, tokenizer, device, layers, prompts, prepared, groups, max_new_tokens, max_length, beam_width
        )
    else:
        generated, suffixes, first_types, target_ranks, competitor_ranks = run_linear_decode(
            model, tokenizer, device, layers, prompts, prepared, groups, mode,
            max_new_tokens, batch_size, max_length, rng, temperature, top_p
        )
    scored = score_outputs(
        tokenizer, generated, suffixes, first_types, target_ranks, competitor_ranks,
        groups, source_pair, checkpoints, max_new_tokens
    )
    for i, sample in enumerate(scored["sample_outputs"]):
        sample["prompt"] = prompts[i]
    return scored


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        all_layers = sorted(set(x for vals in windows.values() for x in vals))
        alpha = max(float(x) for x in args.alphas.split(",") if x.strip())
        seeds = [int(x) for x in args.random_seeds.split(",") if x.strip()]
        seed0 = seeds[0] if seeds else 11
        scaffolds = [x.strip() for x in args.scaffolds.split(",") if x.strip()]
        decode_modes = [x.strip() for x in args.decode_modes.split(",") if x.strip()]
        checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
        checkpoints = sorted(set(x for x in checkpoints if 1 <= x <= args.max_new_tokens))
        if args.max_new_tokens not in checkpoints:
            checkpoints.append(args.max_new_tokens)
        W_U = get_W_U(model, args.model).astype(np.float32)
        log(f"{args.model}: natural decode policy audit, windows={windows}, scaffolds={scaffolds}, modes={decode_modes}")

        candidates = build_candidates(args.train_n)
        source_prompts = build_prompts(args.test_n, scaffolds)

        components_by_layer = {}
        for layer_id in all_layers:
            log(f"  collect L{layer_id}")
            dirs = {}
            for name, meta in candidates.items():
                pos_h = hidden_at_layer(model, tokenizer, device, meta["pos"], layer_id, args.batch_size, args.max_length)
                neg_h = hidden_at_layer(model, tokenizer, device, meta["neg"], layer_id, args.batch_size, args.max_length)
                dirs[name] = mean_dir(pos_h, neg_h)
            components_by_layer[str(layer_id)] = build_components(dirs, W_U, tokenizer, seeds)

        audit = {}
        for win_name, window in windows.items():
            audit[win_name] = {"window": window, "sources": {}}
            for source_pair, by_scaffold in source_prompts.items():
                groups = token_groups(tokenizer, source_pair)
                audit[win_name]["sources"][source_pair] = {}
                for scaffold, prompts in by_scaffold.items():
                    audit[win_name]["sources"][source_pair][scaffold] = {}
                    for mode in decode_modes:
                        row = {}
                        for condition in CONDITIONS:
                            row[condition] = decode_probe(
                                model, tokenizer, device, layers, prompts,
                                interventions_for(components_by_layer, source_pair, window, condition, alpha),
                                groups, source_pair, mode, args.max_new_tokens, checkpoints, args.batch_size,
                                args.max_length, seed0 + stable_offset(source_pair, scaffold, mode, condition),
                                args.temperature, args.top_p, args.beam_width,
                            )
                        audit[win_name]["sources"][source_pair][scaffold][mode] = row
                        base = row["baseline"]["hit_rates"]["family_target"]
                        rp = row["residual_parallel"]["hit_rates"]["family_target"]
                        log(f"    {win_name} {source_pair} {scaffold} {mode}: baseFam={base:.2f} parallelFam={rp:.2f}")

        return {
            "phase": 544,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "core_sources": CORE_SOURCES,
            "conditions": CONDITIONS,
            "scaffolds": scaffolds,
            "decode_modes": decode_modes,
            "windows": windows,
            "all_layers": all_layers,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "max_new_tokens": args.max_new_tokens,
            "checkpoints": checkpoints,
            "alpha": alpha,
            "random_seeds": seeds,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "beam_width": args.beam_width,
            "family_terms": FAMILY_TERMS,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit,
        }
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=10)
    parser.add_argument("--alphas", default="6")
    parser.add_argument("--random-seeds", default="11,23")
    parser.add_argument("--scaffolds", default="direct,one_word,natural_qa,definition,sentence_completion")
    parser.add_argument("--decode-modes", default="greedy,temperature,top_p,beam")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--checkpoints", default="1,3,5,10,12")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase544_{args.model}_natural_decode_policy_gate_audit.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
