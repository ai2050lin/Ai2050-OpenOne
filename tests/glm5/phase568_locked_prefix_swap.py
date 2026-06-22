#!/usr/bin/env python3
"""
Phase 568: Locked Prefix Replication and Residual-State Swap Audit
锁定前缀复验与残差状态交换审计

Resolves two conflicts from Phase 566-567:
1. Phase 566 prefix2 was strong (T2=1.14); Phase 567 prefix2 was weak (T2=0.25).
   → Replicate prefix2 with larger sample (test_n=24, 8 seeds).
2. Phase 567 showed only full layer restore flips step0 margin, not module deltas.
   → Decompose: is it h_in (layer input), h_attn (post-attention), or h_out (layer output)?
   → Also test layer-wise: L24 only, L26 only, L28 only, L24+L28.

Three-state swap:
  h_in swap:  replace layer INPUT with donor's (before attention computation)
  h_attn swap: replace post-ATTENTION state with donor's (before MLP)
  h_out swap:  replace layer OUTPUT with donor's (after both = Phase 567 "all")
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import get_W_U, get_layers, get_model_info, release_model  # noqa: E402
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase562_trajectory_response_audit as p562  # noqa: E402
import phase565_early_gate_token_fork as p565  # noqa: E402
import phase567_logit_field_source as p567  # noqa: E402


OUT_ROOT = Path("results/glm5_phase568_locked_prefix_swap")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Donor three-state cache collection: h_in, h_attn, h_out
# ============================================================================

def collect_three_state_cache(
    model: Any,
    layers: list[Any],
    donor_batch: dict[str, torch.Tensor],
    donor_pos: torch.Tensor,
    layer_ids: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Collect donor h_in, h_attn, h_out at answer position.

    h_in:  input to the layer (before any computation)
    h_attn: state after attention, before MLP (= h_in + attn_out)
    h_out:  layer output (= h_attn + mlp_out)
    """
    caches: dict[str, dict[int, torch.Tensor]] = {"h_in": {}, "h_attn": {}, "h_out": {}}
    handles = []

    for lid in layer_ids:
        layer = layers[lid]
        pos_cpu = donor_pos.cpu()

        # h_in: capture via forward_pre_hook on the layer
        def make_pre_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(module, args):
                # args[0] is hidden_states
                hs = args[0] if isinstance(args, tuple) else args
                if hs is None:
                    return
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["h_in"][layer_id] = hs[bidx, p_dev, :].detach()
            return hook

        # h_attn: hook on self_attn output (this is attn_out, not h_in+attn_out)
        # Actually, we need the post-attention residual = h_in + attn_out
        # The layer output hook gives h_out. We can compute h_attn = h_out - mlp_out.
        # But easier: hook on mlp input to get h_attn (the input to MLP is h_attn)
        def make_mlp_pre_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(module, args):
                hs = args[0] if isinstance(args, tuple) else args
                if hs is None:
                    return
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["h_attn"][layer_id] = hs[bidx, p_dev, :].detach()
            return hook

        # h_out: hook on layer output
        def make_out_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["h_out"][layer_id] = hs[bidx, p_dev, :].detach()
                return output
            return hook

        handles.append(layer.register_forward_pre_hook(make_pre_hook(lid, pos_cpu)))
        handles.append(layer.mlp.register_forward_pre_hook(make_mlp_pre_hook(lid, pos_cpu)))
        handles.append(layer.register_forward_hook(make_out_hook(lid, pos_cpu)))

    with torch.inference_mode():
        model(**donor_batch, return_dict=True, use_cache=False)
    for h in handles:
        h.remove()
    return caches


def transform_three_state_cache(
    caches: dict[str, dict[int, torch.Tensor]],
    variant: str,
    start: int,
) -> dict[str, dict[int, torch.Tensor]]:
    if variant == "repeat4":
        return caches
    if variant == "random":
        out = {"h_in": {}, "h_attn": {}, "h_out": {}}
        for site in ["h_in", "h_attn", "h_out"]:
            for lid, cached in caches[site].items():
                gen = torch.Generator(device=cached.device)
                gen.manual_seed(918000 + lid * 1000 + start)
                rand = torch.randn(cached.shape, generator=gen, device=cached.device, dtype=torch.float32)
                rand = rand / (rand.norm(dim=-1, keepdim=True) + 1e-8)
                norms = cached.float().norm(dim=-1, keepdim=True)
                out[site][lid] = (rand * norms).to(cached.dtype)
        return out
    return caches


# ============================================================================
# State swap injection hooks
# ============================================================================

def install_state_swap_hooks(
    layers: list[Any],
    layer_ids: list[int],
    batch_size: int,
    answer_pos: int,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    swap_spec: str,  # "h_in", "h_attn", "h_out", "baseline"
) -> list[Any]:
    """Install hooks to swap donor state at specified site for specified layers.

    swap_spec:
      "baseline": no swap
      "h_in": replace layer input with donor's h_in
      "h_attn": replace post-attention state (MLP input) with donor's h_attn
      "h_out": replace layer output with donor's h_out
    """
    handles = []
    if donor_caches is None or swap_spec == "baseline":
        return handles

    pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)

    for lid in layer_ids:
        layer = layers[lid]

        if swap_spec == "h_in":
            donor_h_in = donor_caches["h_in"].get(lid)
            if donor_h_in is None:
                continue

            def make_h_in_swap_hook(donor_vec: torch.Tensor, p_cpu: torch.Tensor):
                def hook(module, args):
                    hs = args[0] if isinstance(args, tuple) else args
                    if hs is None:
                        return args
                    p_dev = p_cpu.to(hs.device)
                    bidx = torch.arange(hs.shape[0], device=hs.device)
                    new_hs = hs.clone()
                    new_hs[bidx, p_dev, :] = donor_vec.to(hs.device, dtype=hs.dtype)
                    if isinstance(args, tuple):
                        return (new_hs,) + args[1:]
                    return (new_hs,)
                return hook

            handles.append(layer.register_forward_pre_hook(make_h_in_swap_hook(donor_h_in, pos_cpu)))

        elif swap_spec == "h_attn":
            donor_h_attn = donor_caches["h_attn"].get(lid)
            if donor_h_attn is None:
                continue

            def make_h_attn_swap_hook(donor_vec: torch.Tensor, p_cpu: torch.Tensor):
                def hook(module, args):
                    hs = args[0] if isinstance(args, tuple) else args
                    if hs is None:
                        return args
                    p_dev = p_cpu.to(hs.device)
                    bidx = torch.arange(hs.shape[0], device=hs.device)
                    new_hs = hs.clone()
                    new_hs[bidx, p_dev, :] = donor_vec.to(hs.device, dtype=hs.dtype)
                    if isinstance(args, tuple):
                        return (new_hs,) + args[1:]
                    return (new_hs,)
                return hook

            handles.append(layer.mlp.register_forward_pre_hook(make_h_attn_swap_hook(donor_h_attn, pos_cpu)))

        elif swap_spec == "h_out":
            donor_h_out = donor_caches["h_out"].get(lid)
            if donor_h_out is None:
                continue

            def make_h_out_swap_hook(donor_vec: torch.Tensor, p_cpu: torch.Tensor):
                def hook(_module, _inp, output):
                    hs = p559.tensor_from_output(output)
                    out = hs.clone()
                    p_dev = p_cpu.to(out.device)
                    bidx = torch.arange(out.shape[0], device=out.device)
                    out[bidx, p_dev, :] = donor_vec.to(out.device, dtype=out.dtype)
                    return p559.replace_output(output, out)
                return hook

            handles.append(layer.register_forward_hook(make_h_out_swap_hook(donor_h_out, pos_cpu)))

    return handles


# ============================================================================
# Generation with state swap + forced prefix + logit recording
# ============================================================================

def generate_with_swap(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    layer_ids: list[int],
    prompts: list[str],
    swap_spec: str,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    groups: dict[str, list[int]],
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    forced_prefix_ids: list[list[int]] | None = None,
    logit_steps: int = 6,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)
    forced_len = 0 if forced_prefix_ids is None else min(len(x) for x in forced_prefix_ids)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    handles = install_state_swap_hooks(
        layers, layer_ids, batch_size, answer_pos, donor_caches, swap_spec
    )

    generated: list[list[int]] = [[] for _ in prompts]
    step_summaries: list[list[dict[str, Any]]] = [[] for _ in prompts]
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()
    for h in handles:
        h.remove()

    if forced_prefix_ids is not None and forced_len >= 1:
        toks = [int(ids[0]) for ids in forced_prefix_ids]
    else:
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
    for i, tok in enumerate(toks):
        generated[i].append(int(tok))
        if logit_steps > 0:
            step_summaries[i].append(p565.logits_summary(logits0[i], groups, int(tok)))

    full_mask = attention_mask
    for step in range(1, max_new_tokens):
        new_ids = torch.tensor([[int(t)] for t in toks], dtype=torch.long, device=device)
        full_mask = torch.cat(
            [full_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)], dim=1
        )
        with torch.inference_mode():
            out = model(
                input_ids=new_ids, attention_mask=full_mask,
                past_key_values=past_kv, use_cache=True, return_dict=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu().numpy()
        if forced_prefix_ids is not None and step < forced_len:
            toks = [int(ids[step]) for ids in forced_prefix_ids]
        else:
            toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))
            if step < logit_steps:
                step_summaries[i].append(p565.logits_summary(logits[i], groups, int(tok)))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generated_ids": generated, "generated_suffix": suffixes, "step_summaries": step_summaries}


def prefix_rows(source_ids: list[list[int]], prefix_len: int) -> list[list[int]]:
    return [ids[:prefix_len] for ids in source_ids]


# ============================================================================
# Main
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    seeds = parse_int_csv(args.sample_seeds)
    prefix_lengths = parse_int_csv(args.prefix_lengths)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        _, window = next(iter(windows.items()))
        combos = p558.combo_layers(window, args.layer_sets)
        all_layers = sorted(set(itertools.chain.from_iterable(combos.values())))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        components_by_layer = p558.build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        pos_label, neg_label = PAIR_SPECS[pair]
        objects = CATEGORY_BANK[pos_label][-args.test_n:]
        log(f"{args.model}: phase568 pair={pair}, combos={combos}")

        # Build layer subsets for layer-wise decomposition
        layer_subsets = {
            "L24": [window[0]] if len(window) >= 1 else [],
            "L26": [window[1]] if len(window) >= 2 else [],
            "L28": [window[2]] if len(window) >= 3 else [],
            "L24_L28": [window[0], window[2]] if len(window) >= 3 else [],
            "all": list(window),
        }
        # Filter to only valid subsets
        layer_subsets = {k: v for k, v in layer_subsets.items() if v}

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase568_{args.model}_checkpoint.json"

        # Count total units
        total_units = len(combos) * len(routes) * len(seeds) * (
            2 +  # baseline_free, repeat4_free
            len(prefix_lengths) * 2 +  # bfi_p{1,2,3}, ifb_p{1,2,3}
            3 +  # h_in_swap, h_attn_swap, random_h_out_swap
            len(layer_subsets)  # layer-wise h_out swap
        )
        done_units = 0
        t_start = time.time()

        for combo_name, combo_layer_ids in combos.items():
            audit[combo_name] = {"layers": combo_layer_ids, "rows": {}}
            for route in routes:
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[route["recipient_scaffold"]]
                prompts = [r["prompt"] for r in prompt_rows]

                # Collect donor three-state caches for repeat4 and random
                donor_scaffold = route["donor_scaffold"]
                donor_objs = CATEGORY_BANK[pos_label][-args.test_n:]
                repeat_idx = min(4, len(donor_objs) - 1)
                donor_repeat4_prompts = [
                    p548.forbidden_prompt(donor_scaffold, donor_objs[repeat_idx], pos_label, neg_label)
                ] * args.test_n
                donor_enc = tokenizer(donor_repeat4_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
                donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
                donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
                repeat4_caches = collect_three_state_cache(model, layers, donor_batch, donor_pos, combo_layer_ids)
                random_caches = transform_three_state_cache(repeat4_caches, "random", 0)
                log(f"  Collected repeat4 + random three-state caches (h_in/h_attn/h_out)")

                all_records: dict[str, list[dict[str, Any]]] = {}

                for seed in seeds:
                    # === Part A: Prefix replication ===
                    # baseline free
                    baseline = generate_with_swap(
                        model, tokenizer, device, layers, combo_layer_ids, prompts,
                        "baseline", None, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                        None, args.logit_steps,
                    )
                    done_units += 1
                    all_records.setdefault("baseline_free", []).extend([
                        {"prompt_index": i, "object": row["object"], "seed": seed,
                         "condition": "baseline_free", **{k: baseline[k][i] for k in ["generated_suffix", "generated_ids", "step_summaries"]}}
                        for i, row in enumerate(prompt_rows)
                    ])

                    # repeat4 free (h_out swap on all layers)
                    repeat4 = generate_with_swap(
                        model, tokenizer, device, layers, combo_layer_ids, prompts,
                        "h_out", repeat4_caches, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                        None, args.logit_steps,
                    )
                    done_units += 1
                    all_records.setdefault("repeat4_free", []).extend([
                        {"prompt_index": i, "object": row["object"], "seed": seed,
                         "condition": "repeat4_free", **{k: repeat4[k][i] for k in ["generated_suffix", "generated_ids", "step_summaries"]}}
                        for i, row in enumerate(prompt_rows)
                    ])

                    # bfi prefix 1,2,3
                    for plen in prefix_lengths:
                        bfi = generate_with_swap(
                            model, tokenizer, device, layers, combo_layer_ids, prompts,
                            "baseline", None, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                            prefix_rows(repeat4["generated_ids"], plen), args.logit_steps,
                        )
                        done_units += 1
                        all_records.setdefault(f"bfi_p{plen}", []).extend([
                            {"prompt_index": i, "object": row["object"], "seed": seed,
                             "condition": f"bfi_p{plen}", **{k: bfi[k][i] for k in ["generated_suffix", "generated_ids", "step_summaries"]}}
                            for i, row in enumerate(prompt_rows)
                        ])

                    # ifb prefix 2
                    ifb = generate_with_swap(
                        model, tokenizer, device, layers, combo_layer_ids, prompts,
                        "h_out", repeat4_caches, groups, route["mode"], seed,
                        args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                        prefix_rows(baseline["generated_ids"], 2), args.logit_steps,
                    )
                    done_units += 1
                    all_records.setdefault("ifb_p2", []).extend([
                        {"prompt_index": i, "object": row["object"], "seed": seed,
                         "condition": "ifb_p2", **{k: ifb[k][i] for k in ["generated_suffix", "generated_ids", "step_summaries"]}}
                        for i, row in enumerate(prompt_rows)
                    ])

                    # === Part B: State swap decomposition ===
                    for swap_spec, cache, label in [
                        ("h_in", repeat4_caches, "repeat4_h_in_swap"),
                        ("h_attn", repeat4_caches, "repeat4_h_attn_swap"),
                        ("h_out", random_caches, "random_h_out_swap"),
                    ]:
                        result = generate_with_swap(
                            model, tokenizer, device, layers, combo_layer_ids, prompts,
                            swap_spec, cache, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                            None, args.logit_steps,
                        )
                        done_units += 1
                        all_records.setdefault(label, []).extend([
                            {"prompt_index": i, "object": row["object"], "seed": seed,
                             "condition": label, **{k: result[k][i] for k in ["generated_suffix", "generated_ids", "step_summaries"]}}
                            for i, row in enumerate(prompt_rows)
                        ])

                    # === Part C: Layer-wise h_out swap ===
                    for subset_name, subset_layers in layer_subsets.items():
                        if subset_name == "all":
                            continue  # already done as repeat4_free
                        result = generate_with_swap(
                            model, tokenizer, device, layers, subset_layers, prompts,
                            "h_out", repeat4_caches, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                            None, args.logit_steps,
                        )
                        done_units += 1
                        all_records.setdefault(f"repeat4_{subset_name}", []).extend([
                            {"prompt_index": i, "object": row["object"], "seed": seed,
                             "condition": f"repeat4_{subset_name}", **{k: result[k][i] for k in ["generated_suffix", "generated_ids", "step_summaries"]}}
                            for i, row in enumerate(prompt_rows)
                        ])

                    # Progress log
                    elapsed = time.time() - t_start
                    eta = (elapsed / done_units) * (total_units - done_units) if done_units > 0 else 0
                    log(f"  [{done_units}/{total_units}] seed={seed} done (ETA {eta/60:.1f}min)")

                # Aggregate all conditions
                cond_order = ["baseline_free", "repeat4_free"] + \
                             [f"bfi_p{p}" for p in prefix_lengths] + \
                             ["ifb_p2", "repeat4_h_in_swap", "repeat4_h_attn_swap", "random_h_out_swap"] + \
                             [f"repeat4_{s}" for s in layer_subsets if s != "all"]

                for cond in cond_order:
                    recs = all_records.get(cond, [])
                    if not recs:
                        continue
                    classified = [
                        {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
                        for r in recs
                    ]
                    agg = p548.aggregate(classified)
                    # Step metrics
                    for step in range(args.logit_steps):
                        vals = [r["step_summaries"][step] for r in recs if len(r.get("step_summaries", [])) > step]
                        if vals:
                            agg[f"step{step}_margin"] = float(np.mean([v["target_minus_competitor"] for v in vals]))
                            agg[f"step{step}_target_rank"] = float(np.mean([v["target_rank"] for v in vals]))
                    audit[combo_name]["rows"][route_name][cond] = agg

                    compact.append({
                        "combo": combo_name, "route": route_name, "condition": cond,
                        "free_clean": agg["clean_non_object_rate"],
                        "free_echo": agg["object_echo_rate"],
                        "step0_margin": agg.get("step0_margin", 0),
                        "step0_rank": agg.get("step0_target_rank", 0),
                    })
                    log(f"  {cond}: clean={agg['clean_non_object_rate']:.2f}, s0_margin={agg.get('step0_margin',0):.2f}, s0_rank={agg.get('step0_target_rank',0):.0f}")

                    for rec in classified[:3]:
                        samples.append({**rec, "combo": combo_name, "route": route_name})

                    checkpoint = {
                        "phase": 568, "model": args.model,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "done_units": done_units, "total_units": total_units,
                        "audit": audit,
                    }
                    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "phase": 568, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl, "pair": pair, "window": window,
            "combos": combos, "routes": routes,
            "train_n": args.train_n, "test_n": args.test_n,
            "sample_seeds": seeds, "max_new_tokens": args.max_new_tokens,
            "prefix_lengths": prefix_lengths,
            "layer_subsets": layer_subsets,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "audit": audit, "compact_rows": compact,
            "sample_records": samples[:args.max_saved_samples],
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
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--prefix-lengths", default="1,2,3")
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--logit-steps", type=int, default=6)
    parser.add_argument("--max-saved-samples", type=int, default=400)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase568_{args.model}_locked_prefix_swap.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
