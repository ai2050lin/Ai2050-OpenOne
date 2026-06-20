#!/usr/bin/env python3
"""
Phase 559: Prototype Generation Closure and Object Exemplar Audit
原型生成闭合与对象样例审计

Phase 558 confirmed (next-token margin level):
  - mean_cache ≈ or > repeat donor → category prototype candidate
  - random_cache fails → structured donor state
  - same - shuffle ≈ 0 → weak object binding
  - repeat2/repeat4 strongest → possible exemplar/lexical artifact

Phase 559 brings the key conditions back to GENERATION CLOSURE:
  - Actual multi-token autoregressive generation (not just next-token margin)
  - Compute S_clean-net (clean non-object paraphrase rate) from generated text
  - Audit repeat2=helicopter, repeat4=rocket object names + tokenization
  - Verify: does one-shot prototype restore at answer position cause generation closure?

Design: ONE-SHOT surgery + KV cache
  - Step 0: apply remove_perp + restore donor cache at answer position (with hooks)
  - Steps 1+: FREE generation with KV cache (no surgery hooks)
  - This tests: is restoring the answer-position state SUFFICIENT for clean generation?

This is faster than Phase 558's continuous surgery (re-encode each step) because:
  - Donor forward: once (not every step)
  - Recipient forward: 1 full + (max_new_tokens-1) single-token forwards with KV cache
"""
from __future__ import annotations

import argparse
import gc
import itertools
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
from phase530_state_pair_decomposition import load_model_bf16_flash, mean_dir, hidden_at_layer  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK  # noqa: E402


OUT_ROOT = Path("results/glm5_phase559_prototype_generation_closure")

DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]

DEFAULT_CONDITIONS = [
    "baseline",
    "add_perp",
    "resid_remove_perp",
    "resid_donor_vehicle_same_add",
    "resid_donor_vehicle_shuffle_add",
    "resid_donor_vehicle_repeat2_add",
    "resid_donor_vehicle_repeat4_add",
    "resid_donor_vehicle_mean_cache_add",
    "resid_donor_vehicle_pca1_cache_add",
    "resid_donor_vehicle_random_cache_add",
    "resid_donor_tool_same_add",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Object name audit: map repeat index → object name + tokenization
# ============================================================================

def object_name_audit(pair: str, test_n: int, tokenizer: Any) -> list[dict[str, Any]]:
    """Map repeat index to object name, tokenization length, etc."""
    pos_label, _ = PAIR_SPECS[pair]
    objects = CATEGORY_BANK[pos_label][-test_n:]
    audit = []
    for idx, obj in enumerate(objects):
        toks = tokenizer.encode(obj, add_special_tokens=False)
        audit.append({
            "repeat_index": idx,
            "object": obj,
            "token_ids": toks,
            "token_length": len(toks),
            "char_length": len(obj),
        })
    return audit


# ============================================================================
# One-shot surgery generation with KV cache
# ============================================================================

def tensor_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def replace_output(output: Any, new_tensor: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (new_tensor,) + output[1:]
    return new_tensor


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = vec.astype(np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return arr
    return arr / norm


def project_remove(x: torch.Tensor, pos: torch.Tensor, direction: torch.Tensor, scale: float) -> torch.Tensor:
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    vecs = out[bidx, pos, :].float()
    d = direction.to(out.device).float()
    d = d / (d.norm() + 1e-8)
    coeff = (vecs * d).sum(dim=-1, keepdim=True)
    proj = coeff * d.unsqueeze(0)
    out[bidx, pos, :] = out[bidx, pos, :] - float(scale) * proj.to(out.dtype)
    return out


def add_direction(x: torch.Tensor, pos: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    d = direction.to(out.device).float()
    d = d / (d.norm() + 1e-8)
    out[bidx, pos, :] = out[bidx, pos, :] + (float(alpha) * d).to(out.dtype)
    return out


def module_for_site(layer: Any, site: str) -> Any:
    if site == "resid":
        return layer
    if site == "attn":
        return layer.self_attn
    if site == "mlp":
        return layer.mlp
    raise ValueError(f"unknown site: {site}")


def collect_donor_cache(
    model: Any,
    layers: list[Any],
    donor_batch: dict[str, torch.Tensor],
    donor_pos: torch.Tensor,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    site_name: str,
    donor_condition: str,
    add_alpha: float,
) -> dict[int, torch.Tensor]:
    """Collect donor activations at the target site and answer position.
    Uses right-padded donor_batch. donor_pos is per-sequence last-non-pad position.
    Tensors created on CPU; moved to out.device inside hooks to avoid meta-tensor issues."""
    cache: dict[int, torch.Tensor] = {}
    handles = []
    donor_add = donor_condition == "add_perp"

    for layer_id in layer_ids:
        layer = layers[layer_id]
        site = module_for_site(layer, site_name)
        pos_cpu = donor_pos.cpu()  # keep on CPU, move inside hook

        if donor_add:
            direction_np = components_by_layer[str(layer_id)]["residual_perp"]
            direction_cpu = torch.tensor(normalize_vec(direction_np), dtype=torch.float32)
        else:
            direction_cpu = None

        if site_name == "resid":
            def make_resid_cache_hook(lid: int, d_vec_cpu: torch.Tensor | None, p_vec_cpu: torch.Tensor):
                def hook(_module: Any, _inp: Any, output: Any):
                    hs = tensor_from_output(output)
                    out = hs
                    p_dev = p_vec_cpu.to(out.device)
                    if donor_add and d_vec_cpu is not None:
                        out = add_direction(out, p_dev, d_vec_cpu, add_alpha)
                    bidx = torch.arange(out.shape[0], device=out.device)
                    cache[lid] = out[bidx, p_dev, :].detach()
                    return replace_output(output, out)
                return hook
            handles.append(layer.register_forward_hook(make_resid_cache_hook(layer_id, direction_cpu, pos_cpu)))
        else:
            def make_site_cache_hook(lid: int, p_vec_cpu: torch.Tensor):
                def hook(_module: Any, _inp: Any, output: Any):
                    hs = tensor_from_output(output)
                    p_dev = p_vec_cpu.to(hs.device)
                    bidx = torch.arange(hs.shape[0], device=hs.device)
                    cache[lid] = hs[bidx, p_dev, :].detach()
                    return output
                return hook
            handles.append(site.register_forward_hook(make_site_cache_hook(layer_id, pos_cpu)))
            if donor_add:
                def make_add_hook(d_vec_cpu: torch.Tensor, p_vec_cpu: torch.Tensor):
                    def hook(_module: Any, _inp: Any, output: Any):
                        hs = tensor_from_output(output)
                        p_dev = p_vec_cpu.to(hs.device)
                        out = add_direction(hs, p_dev, d_vec_cpu, add_alpha)
                        return replace_output(output, out)
                    return hook
                handles.append(layer.register_forward_hook(make_add_hook(direction_cpu, pos_cpu)))

    with torch.inference_mode():
        model(**donor_batch, return_dict=True, use_cache=False)
    for handle in handles:
        handle.remove()
    return cache


def transform_restore_cache(
    cache: dict[int, torch.Tensor],
    variant: str | None,
    start: int,
) -> dict[int, torch.Tensor]:
    if variant is None:
        return cache
    if variant == "random_cache":
        out = {}
        for lid, cached in cache.items():
            gen = torch.Generator(device=cached.device)
            gen.manual_seed(918000 + lid * 1000 + start)
            rand = torch.randn(cached.shape, generator=gen, device=cached.device, dtype=torch.float32)
            rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-8)
            norms = cached.float().norm(dim=-1, keepdim=True)
            out[lid] = (rand * norms).to(cached.dtype)
        return out
    if variant == "mean_cache":
        out = {}
        for lid, cached in cache.items():
            mean = cached.float().mean(dim=0, keepdim=True)
            out[lid] = mean.repeat(cached.shape[0], 1).to(cached.dtype)
        return out
    if variant in {"pca1_cache", "pca3_cache"}:
        rank = 1 if variant == "pca1_cache" else 3
        out = {}
        for lid, cached in cache.items():
            x = cached.float()
            mean = x.mean(dim=0, keepdim=True)
            centered = x - mean
            max_rank = max(1, min(rank, centered.shape[0] - 1, centered.shape[1]))
            try:
                _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
                basis = vh[:max_rank]
                recon = mean + (centered @ basis.T) @ basis
            except RuntimeError:
                recon = mean.repeat(cached.shape[0], 1)
            out[lid] = recon.to(cached.dtype)
        return out
    return cache


def generate_batch_oneshot(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    donor_prompts: list[str] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    mode: str,
    max_new_tokens: int,
    seed: int,
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
    max_length: int,
) -> tuple[list[list[int]], list[str], list[str], list[float], list[float]]:
    """One-shot surgery at answer position + KV cache free generation.

    Surgery (remove + restore) is applied ONLY at step 0 (answer position).
    Steps 1+ are free generation using KV cache (no hooks).
    """
    plan = p558.condition_plan(condition)
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)

    # --- Left-pad prompts for batch KV cache generation ---
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    # With left-padding, last position is seq_len-1 for ALL sequences
    answer_pos = seq_len - 1
    tokenizer.padding_side = original_padding_side

    # --- Collect donor cache (right-padded, per-sequence pos) ---
    restore_cache: dict[int, torch.Tensor] = {}
    if plan["restore"] is not None and donor_prompts is not None:
        donor_enc = tokenizer(donor_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
        donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
        raw_cache = collect_donor_cache(
            model, layers, donor_batch, donor_pos, components_by_layer, layer_ids,
            plan["site"], plan["restore"], add_alpha,
        )
        restore_cache = transform_restore_cache(raw_cache, plan.get("donor_variant"), 0)

    # --- Set up surgery hooks at answer position ---
    # NOTE: Create all tensors on CPU; move to out.device inside hooks.
    # This avoids meta-tensor issues with device_map="auto" in transformers 5.x.
    handles: list[Any] = []
    if plan["site"] != "none":
        for layer_id in layer_ids:
            layer = layers[layer_id]
            site = module_for_site(layer, plan["site"])
            # answer_pos is the same for all sequences (left-padding)
            pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)
            direction_np = components_by_layer[str(layer_id)][plan["component"]]
            direction_cpu = torch.tensor(normalize_vec(direction_np), dtype=torch.float32)
            cached = restore_cache.get(layer_id)  # may be on GPU/CPU depending on donor collection
            should_remove = plan["remove"]
            should_add = plan["add"] and plan["site"] == "resid"
            should_restore = plan["restore"] is not None and cached is not None

            def make_hook(d_vec_cpu, p_vec_cpu, cached_vec, rm, ad, rs, site_name):
                def hook(_module, _inp, output):
                    hs = tensor_from_output(output)
                    out = hs
                    p_dev = p_vec_cpu.to(out.device)
                    if rm:
                        out = project_remove(out, p_dev, d_vec_cpu, remove_scale)
                    if ad and site_name == "resid":
                        out = add_direction(out, p_dev, d_vec_cpu, add_alpha)
                    if rs and cached_vec is not None:
                        bidx = torch.arange(out.shape[0], device=out.device)
                        out = out.clone()
                        out[bidx, p_dev, :] = cached_vec.to(out.device, dtype=out.dtype)
                    return replace_output(output, out)
                return hook

            handles.append(site.register_forward_hook(
                make_hook(direction_cpu, pos_cpu, cached, should_remove, should_add, should_restore, plan["site"])
            ))

    # --- Step 0: full forward with surgery + KV cache ---
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits_step0 = out.logits[:, answer_pos, :].float().cpu().numpy()

    # Remove surgery hooks after step 0
    for h in handles:
        h.remove()

    # --- Sample first token ---
    toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits_step0]
    first_types = [p544.token_type(int(t), groups) for t in toks]
    first_target_ranks = [float(p544.best_rank(row, groups["target"])) for row in logits_step0]
    first_competitor_ranks = [float(p544.best_rank(row, groups["competitor"])) for row in logits_step0]
    generated: list[list[int]] = [[int(t)] for t in toks]

    # --- Steps 1+: free generation with KV cache (NO surgery) ---
    # Extend attention mask for each new token
    full_attn_mask = attention_mask
    for step in range(1, max_new_tokens):
        new_ids = torch.tensor([[t] for t in toks], dtype=torch.long, device=device)
        new_mask_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
        full_attn_mask = torch.cat([full_attn_mask, new_mask_col], dim=1)
        with torch.inference_mode():
            out = model(input_ids=new_ids, attention_mask=full_attn_mask,
                        past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu().numpy()
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, t in enumerate(toks):
            generated[i].append(int(t))

    # --- Decode suffixes ---
    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]

    # --- Free GPU memory ---
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return generated, suffixes, first_types, first_target_ranks, first_competitor_ranks


# ============================================================================
# Main audit loop
# ============================================================================

def run_condition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompt_rows: list[dict[str, str]],
    donor_rows: list[dict[str, str]] | None,
    components_by_layer: dict[str, dict[str, np.ndarray]],
    layer_ids: list[int],
    condition: str,
    groups: dict[str, list[int]],
    pair: str,
    mode: str,
    max_new_tokens: int,
    sample_seeds: list[int],
    temperature: float,
    top_p: float,
    remove_scale: float,
    add_alpha: float,
    max_length: int,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one condition across all seeds and prompts. Returns aggregated + per-record."""
    pos_label, neg_label = PAIR_SPECS[pair]
    all_records: list[dict[str, Any]] = []
    seed_aggs: list[dict[str, Any]] = []

    for seed in sample_seeds:
        # Process prompts in batches
        batch_generated: list[list[int]] = []
        batch_suffixes: list[str] = []
        batch_first_types: list[str] = []
        batch_first_target_ranks: list[float] = []
        batch_first_competitor_ranks: list[float] = []

        for start in range(0, len(prompt_rows), batch_size):
            b_prompts = [r["prompt"] for r in prompt_rows[start:start + batch_size]]
            b_donor = None
            if donor_rows is not None:
                b_donor = [r["prompt"] for r in donor_rows[start:start + batch_size]]

            gen, suf, ft, ftr, fcr = generate_batch_oneshot(
                model, tokenizer, device, layers, b_prompts, b_donor,
                components_by_layer, layer_ids, condition, groups, mode,
                max_new_tokens, seed, temperature, top_p,
                remove_scale, add_alpha, max_length,
            )
            batch_generated.extend(gen)
            batch_suffixes.extend(suf)
            batch_first_types.extend(ft)
            batch_first_target_ranks.extend(ftr)
            batch_first_competitor_ranks.extend(fcr)

        # Classify each generated suffix
        records = []
        for i, (row, suffix, ids, ft, ftr, fcr) in enumerate(zip(
            prompt_rows, batch_suffixes, batch_generated,
            batch_first_types, batch_first_target_ranks, batch_first_competitor_ranks
        )):
            cls = p548.classify_suffix(suffix, row["object"], pos_label, neg_label)
            rec = {
                "prompt_index": i,
                "object": row["object"],
                "prompt": row["prompt"],
                "donor_object": donor_rows[i]["object"] if donor_rows is not None else "",
                "donor_prompt": donor_rows[i]["prompt"] if donor_rows is not None else "",
                "condition": condition,
                "seed": seed,
                "generated_suffix": suffix,
                "generated_ids": ids,
                "first_type": ft,
                "first_target_rank": float(ftr),
                "first_competitor_rank": float(fcr),
                **cls,
            }
            records.append(rec)
            all_records.append(rec)

        seed_agg = p548.aggregate(records)
        seed_agg["seed"] = seed
        seed_aggs.append(seed_agg)

    agg = p548.aggregate(all_records)
    agg["seed_aggregates"] = seed_aggs
    return agg, all_records


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    conditions = p558.parse_csv(args.conditions)
    sample_seeds = [int(x.strip()) for x in args.sample_seeds.split(",") if x.strip()]

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        if len(windows) != 1:
            raise ValueError(f"Phase559 expects one window, got {windows}")
        _, window = next(iter(windows.items()))
        combos = p558.combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)

        # Object name audit
        obj_audit = object_name_audit(pair, args.test_n, tokenizer)
        log(f"{args.model}: object audit (repeat index → object):")
        for a in obj_audit:
            log(f"    repeat{a['repeat_index']} = {a['object']} (tokens={a['token_length']}, chars={a['char_length']})")

        # Build components
        components_by_layer = p558.build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        log(f"{args.model}: phase559 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")
        log(f"  conditions={len(conditions)}, seeds={len(sample_seeds)}, max_new_tokens={args.max_new_tokens}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        saved_samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase559_{args.model}_checkpoint.json"

        total_combos = len(combos)
        total_routes = len(routes)
        total_conditions = len(conditions)
        total_units = total_combos * total_routes * total_conditions
        done_units = 0
        t_start = time.time()

        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                recipient_scaffold = route["recipient_scaffold"]
                donor_scaffold = route["donor_scaffold"]
                mode = route["mode"]
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[recipient_scaffold]

                for condition in conditions:
                    plan = p558.condition_plan(condition)
                    donor_rows = p558.donor_rows_for(
                        pair, donor_scaffold, plan.get("donor_category"),
                        plan.get("donor_variant"), args.test_n,
                    )

                    t_cond = time.time()
                    agg, records = run_condition(
                        model, tokenizer, device, layers, prompt_rows, donor_rows,
                        components_by_layer, layer_ids, condition, groups, pair, mode,
                        args.max_new_tokens, sample_seeds, args.temperature, args.top_p,
                        args.remove_scale, args.add_alpha, args.max_length, args.batch_size,
                    )
                    audit[combo_name]["rows"][route_name][condition] = agg
                    saved_samples.extend(records[: args.samples_per_row])

                    done_units += 1
                    elapsed = time.time() - t_start
                    eta = (elapsed / done_units) * (total_units - done_units) if done_units > 0 else 0
                    log(
                        f"  [{done_units}/{total_units}] {combo_name} {route_name} {condition}: "
                        f"clean_no={agg['clean_non_object_rate']:.2f}, "
                        f"label={agg['any_label_violation_rate']:.2f}, "
                        f"echo={agg['object_echo_rate']:.2f}, "
                        f"score={agg['clean_non_object_score']:.2f} "
                        f"({time.time()-t_cond:.1f}s, ETA {eta/60:.1f}min)"
                    )

                    # Write checkpoint
                    checkpoint = {
                        "phase": 559,
                        "model": args.model,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "pair": pair,
                        "window": window,
                        "combos": combos,
                        "conditions": conditions,
                        "routes": routes,
                        "train_n": args.train_n,
                        "test_n": args.test_n,
                        "sample_seeds": sample_seeds,
                        "max_new_tokens": args.max_new_tokens,
                        "done_units": done_units,
                        "total_units": total_units,
                        "last_combo": combo_name,
                        "last_route": route_name,
                        "last_condition": condition,
                        "audit": audit,
                    }
                    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

                # Compute compact metrics for this route
                rows = audit[combo_name]["rows"][route_name]
                base = rows.get("baseline", {"clean_non_object_rate": 0.0})
                remove_ref = rows.get("resid_remove_perp", base)
                random_ref = rows.get("resid_donor_vehicle_random_cache_add", base)
                add_ref = rows.get("add_perp", base)

                for condition, row in rows.items():
                    if condition == "baseline":
                        continue
                    plan = p558.condition_plan(condition)
                    clean_delta = row["clean_non_object_rate"] - base["clean_non_object_rate"]
                    score_delta = row["clean_non_object_score"] - base["clean_non_object_score"]
                    label_delta = row["any_label_violation_rate"] - base["any_label_violation_rate"]
                    remove_delta = remove_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
                    restore_gain = row["clean_non_object_rate"] - remove_ref["clean_non_object_rate"]
                    random_delta = random_ref["clean_non_object_rate"] - base["clean_non_object_rate"]
                    is_restore = "_donor_" in condition

                    if is_restore:
                        if remove_delta <= -0.06 and restore_gain >= 0.08 and label_delta <= 0.05:
                            cls = "restore_success"
                        elif remove_delta <= -0.06 and restore_gain >= 0.04 and label_delta <= 0.08:
                            cls = "weak_restore"
                        elif restore_gain >= 0.08:
                            cls = "restore_without_drop_or_leaky"
                        else:
                            cls = "restore_fail"
                    elif clean_delta <= -0.10 and score_delta <= -0.08 and label_delta <= 0.05:
                        cls = "necessity_drop"
                    elif clean_delta <= -0.06 and score_delta <= -0.04:
                        cls = "weak_drop"
                    elif label_delta >= 0.12:
                        cls = "label_leak_or_noise"
                    elif clean_delta >= 0.08:
                        cls = "positive_add_or_release"
                    else:
                        cls = "flat"

                    compact.append({
                        "combo": combo_name,
                        "layers": layer_ids,
                        "route": route_name,
                        "recipient_scaffold": recipient_scaffold,
                        "donor_scaffold": donor_scaffold,
                        "mode": mode,
                        "condition": condition,
                        "donor_category": plan.get("donor_category") or "",
                        "donor_variant": plan.get("donor_variant") or "",
                        "base_clean_non_object_rate": base["clean_non_object_rate"],
                        "clean_non_object_rate": row["clean_non_object_rate"],
                        "base_label_violation_rate": base["any_label_violation_rate"],
                        "label_violation_rate": row["any_label_violation_rate"],
                        "object_echo_rate": row["object_echo_rate"],
                        "prompt_echo_rate": row["prompt_echo_rate"],
                        "clean_non_object_score": row["clean_non_object_score"],
                        "clean_delta": float(clean_delta),
                        "score_delta": float(score_delta),
                        "label_delta": float(label_delta),
                        "remove_delta": float(remove_delta),
                        "restore_gain": float(restore_gain),
                        "random_delta": float(random_delta),
                        "class": cls,
                    })

                # Log route summary
                rp = rows.get("resid_remove_perp", base)
                ap = rows.get("add_perp", base)
                mc = rows.get("resid_donor_vehicle_mean_cache_add", base)
                rc = rows.get("resid_donor_vehicle_random_cache_add", base)
                sm = rows.get("resid_donor_vehicle_same_add", base)
                sh = rows.get("resid_donor_vehicle_shuffle_add", base)
                r2 = rows.get("resid_donor_vehicle_repeat2_add", base)
                r4 = rows.get("resid_donor_vehicle_repeat4_add", base)
                pc = rows.get("resid_donor_vehicle_pca1_cache_add", base)
                log(
                    f"  SUMMARY {combo_name} {route_name}: "
                    f"base={base['clean_non_object_rate']:.2f}; "
                    f"add={ap['clean_non_object_rate']:.2f}; "
                    f"rm={rp['clean_non_object_rate']:.2f}; "
                    f"same={sm['clean_non_object_rate']:.2f}; "
                    f"shuf={sh['clean_non_object_rate']:.2f}; "
                    f"r2={r2['clean_non_object_rate']:.2f}; "
                    f"r4={r4['clean_non_object_rate']:.2f}; "
                    f"mean={mc['clean_non_object_rate']:.2f}; "
                    f"pca1={pc['clean_non_object_rate']:.2f}; "
                    f"rand={rc['clean_non_object_rate']:.2f}"
                )

        return {
            "phase": 559,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "pair": pair,
            "window": window,
            "combos": combos,
            "conditions": conditions,
            "routes": routes,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "sample_seeds": sample_seeds,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "surgery_mode": "one_shot_step0_then_free_kv_cache",
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "object_audit": obj_audit,
            "audit": audit,
            "compact_rows": compact,
            "sample_records": saved_samples[: args.max_saved_samples],
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
    parser.add_argument("--pair", default="vehicle_tool")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=2)
    parser.add_argument("--max-saved-samples", type=int, default=1200)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase559_{args.model}_prototype_generation_closure.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
