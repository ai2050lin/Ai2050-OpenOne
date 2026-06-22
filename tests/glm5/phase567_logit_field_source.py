#!/usr/bin/env python3
"""
Phase 567: Step0 Logit-Field Source Decomposition
第0步词表场来源分解

Phase566 established that GLM4 repeat4's step0 logit-field flip → prefix2 lock →
recursive expansion is the causal chain. Phase567 asks: which module (residual
restore, attention contribution, MLP contribution, or combinations) is the
source of the step0 target-competitor margin flip?

Method: Collect donor activations at THREE sites per layer:
  - layer output (residual): h_out = h_in + attn_out + mlp_out
  - attention output: attn_out
  - MLP output: mlp_out

Then inject ONLY the specified module's contribution from donor into recipient
at step 0, measuring:
  1. step0 target-competitor margin
  2. step0 target rank
  3. free clean_non_object_rate (full 12-token generation)
  4. prefix2 transfer clean (baseline forced to use intervention's first 2 tokens)

Conditions:
  baseline
  repeat4_all          (full layer restore = Phase566 style)
  repeat4_attn_only    (restore only donor's attention output)
  repeat4_mlp_only     (restore only donor's MLP output)
  repeat4_attn_mlp     (restore attention + MLP, skip residual baseline)
  random_all           (full layer restore with random donor)
  random_attn_only
  random_mlp_only
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


OUT_ROOT = Path("results/glm5_phase567_logit_field_source")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]
DEFAULT_CONDITIONS = [
    "baseline",
    "repeat4_all",
    "repeat4_attn_only",
    "repeat4_mlp_only",
    "repeat4_attn_mlp",
    "random_all",
    "random_attn_only",
    "random_mlp_only",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Donor module-level cache collection
# ============================================================================

def collect_module_donor_cache(
    model: Any,
    layers: list[Any],
    donor_batch: dict[str, torch.Tensor],
    donor_pos: torch.Tensor,
    layer_ids: list[int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Collect donor activations at layer, attention, and MLP output sites.

    Returns dict with keys 'layer', 'attn', 'mlp', each mapping layer_id -> tensor.
    The 'layer' cache is the full residual output (h_in + attn + mlp).
    The 'attn' cache is just the attention output.
    The 'mlp' cache is just the MLP output.
    """
    caches: dict[str, dict[int, torch.Tensor]] = {"layer": {}, "attn": {}, "mlp": {}}
    handles = []

    for lid in layer_ids:
        layer = layers[lid]
        pos_cpu = donor_pos.cpu()

        # Hook on the full layer (residual output)
        def make_layer_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["layer"][layer_id] = hs[bidx, p_dev, :].detach()
                return output
            return hook

        # Hook on attention output
        def make_attn_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["attn"][layer_id] = hs[bidx, p_dev, :].detach()
                return output
            return hook

        # Hook on MLP output
        def make_mlp_hook(layer_id: int, p_cpu: torch.Tensor):
            def hook(_module, _inp, output):
                hs = p559.tensor_from_output(output)
                p_dev = p_cpu.to(hs.device)
                bidx = torch.arange(hs.shape[0], device=hs.device)
                caches["mlp"][layer_id] = hs[bidx, p_dev, :].detach()
                return output
            return hook

        handles.append(layer.register_forward_hook(make_layer_hook(lid, pos_cpu)))
        handles.append(layer.self_attn.register_forward_hook(make_attn_hook(lid, pos_cpu)))
        handles.append(layer.mlp.register_forward_hook(make_mlp_hook(lid, pos_cpu)))

    with torch.inference_mode():
        model(**donor_batch, return_dict=True, use_cache=False)
    for h in handles:
        h.remove()
    return caches


def transform_module_cache(
    caches: dict[str, dict[int, torch.Tensor]],
    variant: str | None,
    start: int,
) -> dict[str, dict[int, torch.Tensor]]:
    """Apply donor variant transforms to each module cache."""
    if variant is None or variant == "repeat4":
        return caches
    if variant == "random":
        out = {"layer": {}, "attn": {}, "mlp": {}}
        for site in ["layer", "attn", "mlp"]:
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
# Module-level injection hooks
# ============================================================================

def install_module_hooks(
    layers: list[Any],
    layer_ids: list[int],
    batch_size: int,
    answer_pos: int,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    module_spec: str,  # "all", "attn_only", "mlp_only", "attn_mlp"
) -> list[Any]:
    """Install hooks to inject donor module contributions at step 0.

    module_spec:
      "all": replace residual with donor's full layer output
      "attn_only": add (donor_attn - recipient_attn) to residual, leaving MLP as-is
      "mlp_only": add (donor_mlp - recipient_mlp) to residual, leaving attn as-is
      "attn_mlp": add both (donor_attn - recipient_attn) + (donor_mlp - recipient_mlp)
    """
    handles = []
    if donor_caches is None or module_spec == "baseline":
        return handles

    pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)

    for lid in layer_ids:
        layer = layers[lid]
        layer_device = next(layer.parameters()).device
        pos_dev = pos_cpu.to(layer_device)

        if module_spec == "all":
            # Full layer restore: replace residual output with donor's
            donor_layer = donor_caches["layer"].get(lid)
            if donor_layer is None:
                continue

            def make_all_hook(donor_vec: torch.Tensor, p_cpu: torch.Tensor):
                def hook(_module, _inp, output):
                    hs = p559.tensor_from_output(output)
                    out = hs.clone()
                    p_dev = p_cpu.to(out.device)
                    bidx = torch.arange(out.shape[0], device=out.device)
                    out[bidx, p_dev, :] = donor_vec.to(out.device, dtype=out.dtype)
                    return p559.replace_output(output, out)
                return hook

            handles.append(layer.register_forward_hook(make_all_hook(donor_layer, pos_cpu)))

        else:
            # Module-specific injection: capture recipient's attn/mlp output, then
            # add the delta at the layer level
            # We need to capture recipient attn and mlp to compute deltas
            recipient_attn: dict[int, torch.Tensor] = {}
            recipient_mlp: dict[int, torch.Tensor] = {}

            use_attn = module_spec in ("attn_only", "attn_mlp")
            use_mlp = module_spec in ("mlp_only", "attn_mlp")

            def make_capture_hook(site: str, lid_local: int, p_cpu: torch.Tensor):
                def hook(_module, _inp, output):
                    hs = p559.tensor_from_output(output)
                    p_dev = p_cpu.to(hs.device)
                    bidx = torch.arange(hs.shape[0], device=hs.device)
                    if site == "attn":
                        recipient_attn[lid_local] = hs[bidx, p_dev, :].detach()
                    else:
                        recipient_mlp[lid_local] = hs[bidx, p_dev, :].detach()
                    return output
                return hook

            if use_attn:
                handles.append(layer.self_attn.register_forward_hook(
                    make_capture_hook("attn", lid, pos_cpu)))
            if use_mlp:
                handles.append(layer.mlp.register_forward_hook(
                    make_capture_hook("mlp", lid, pos_cpu)))

            # Layer-level hook: add delta after both modules have run
            donor_attn = donor_caches["attn"].get(lid) if use_attn else None
            donor_mlp = donor_caches["mlp"].get(lid) if use_mlp else None

            def make_delta_hook(d_a: torch.Tensor | None, d_m: torch.Tensor | None,
                                r_attn: dict, r_mlp: dict, lid_local: int, p_cpu: torch.Tensor):
                def hook(_module, _inp, output):
                    hs = p559.tensor_from_output(output)
                    out = hs.clone()
                    p_dev = p_cpu.to(out.device)
                    bidx = torch.arange(out.shape[0], device=out.device)
                    delta = torch.zeros(out[bidx, p_dev, :].shape, device=out.device, dtype=torch.float32)
                    if d_a is not None and lid_local in r_attn:
                        delta = delta + (d_a.to(out.device).float() - r_attn[lid_local].to(out.device).float())
                    if d_m is not None and lid_local in r_mlp:
                        delta = delta + (d_m.to(out.device).float() - r_mlp[lid_local].to(out.device).float())
                    out[bidx, p_dev, :] = out[bidx, p_dev, :] + delta.to(out.dtype)
                    return p559.replace_output(output, out)
                return hook

            handles.append(layer.register_forward_hook(
                make_delta_hook(donor_attn, donor_mlp, recipient_attn, recipient_mlp, lid, pos_cpu)))

    return handles


# ============================================================================
# Generation with module injection + forced prefix + logit recording
# ============================================================================

def generate_with_module_injection(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    layer_ids: list[int],
    prompts: list[str],
    module_spec: str,
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
    """Generate with module-level donor injection at step 0, then free generation."""
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

    handles = install_module_hooks(
        layers, layer_ids, batch_size, answer_pos, donor_caches, module_spec
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
    return {"generated_ids": generated, "suffixes": suffixes, "step_summaries": step_summaries}


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
    conditions = parse_csv(args.conditions)
    seeds = parse_int_csv(args.sample_seeds)

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
        log(f"{args.model}: phase567 pair={pair}, combos={combos}")
        log(f"  conditions={conditions}, seeds={seeds}, test_n={args.test_n}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase567_{args.model}_checkpoint.json"

        total_units = len(combos) * len(routes) * len(seeds) * (1 + len(conditions) * 2)
        done_units = 0
        t_start = time.time()

        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            for route in routes:
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[route["recipient_scaffold"]]
                prompts = [r["prompt"] for r in prompt_rows]

                # Collect donor module caches for repeat4 and random
                donor_scaffold = route["donor_scaffold"]
                donor_objs = CATEGORY_BANK[pos_label][-args.test_n:]
                # repeat4 donor prompts (repeat_index=4 = rocket, if available)
                # Use last object as fallback for small test_n
                repeat_idx = min(4, len(donor_objs) - 1)
                donor_repeat4_rows = [
                    {"object": donor_objs[repeat_idx], "prompt": p548.forbidden_prompt(donor_scaffold, donor_objs[repeat_idx], pos_label, neg_label)}
                ] * args.test_n
                donor_random_rows = donor_repeat4_rows  # same prompts, will be randomized in transform

                # Collect repeat4 module caches
                donor_repeat4_prompts = [r["prompt"] for r in donor_repeat4_rows]
                donor_enc = tokenizer(donor_repeat4_prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
                donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
                donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
                repeat4_caches = collect_module_donor_cache(model, layers, donor_batch, donor_pos, layer_ids)
                random_caches = transform_module_cache(repeat4_caches, "random", 0)
                log(f"  Collected repeat4 + random module caches (layer/attn/mlp)")

                # Map condition -> (donor_caches, module_spec)
                cond_map = {
                    "baseline": (None, "baseline"),
                    "repeat4_all": (repeat4_caches, "all"),
                    "repeat4_attn_only": (repeat4_caches, "attn_only"),
                    "repeat4_mlp_only": (repeat4_caches, "mlp_only"),
                    "repeat4_attn_mlp": (repeat4_caches, "attn_mlp"),
                    "random_all": (random_caches, "all"),
                    "random_attn_only": (random_caches, "attn_only"),
                    "random_mlp_only": (random_caches, "mlp_only"),
                }

                for condition in conditions:
                    t_cond = time.time()
                    donor_caches, module_spec = cond_map.get(condition, (None, "baseline"))
                    all_records: list[dict[str, Any]] = []

                    for seed in seeds:
                        # Free generation
                        result = generate_with_module_injection(
                            model, tokenizer, device, layers, layer_ids, prompts,
                            module_spec, donor_caches, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                            None, args.logit_steps,
                        )
                        done_units += 1
                        for i, row in enumerate(prompt_rows):
                            all_records.append({
                                "prompt_index": i, "object": row["object"], "seed": seed,
                                "condition": condition, "variant": "free",
                                "generated_suffix": result["suffixes"][i],
                                "generated_ids": result["generated_ids"][i],
                                "step_summaries": result["step_summaries"][i],
                            })

                        # Prefix2 transfer: baseline forced to use intervention's first 2 tokens
                        if condition not in ("baseline",) and donor_caches is not None:
                            bfi_result = generate_with_module_injection(
                                model, tokenizer, device, layers, layer_ids, prompts,
                                "baseline", None, groups, route["mode"], seed,
                                args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                                prefix_rows(result["generated_ids"], 2), args.logit_steps,
                            )
                            done_units += 1
                            for i, row in enumerate(prompt_rows):
                                all_records.append({
                                    "prompt_index": i, "object": row["object"], "seed": seed,
                                    "condition": condition, "variant": "bfi_prefix2",
                                    "generated_suffix": bfi_result["suffixes"][i],
                                    "generated_ids": bfi_result["generated_ids"][i],
                                    "step_summaries": bfi_result["step_summaries"][i],
                                })

                    # Aggregate free and bfi_prefix2 separately
                    free_records = [r for r in all_records if r["variant"] == "free"]
                    bfi_records = [r for r in all_records if r["variant"] == "bfi_prefix2"]

                    free_agg = p548.aggregate([
                        {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
                        for r in free_records
                    ])
                    # Step metrics
                    for step in range(args.logit_steps):
                        vals = [r["step_summaries"][step] for r in free_records if len(r.get("step_summaries", [])) > step]
                        if vals:
                            free_agg[f"step{step}_margin"] = float(np.mean([v["target_minus_competitor"] for v in vals]))
                            free_agg[f"step{step}_target_rank"] = float(np.mean([v["target_rank"] for v in vals]))

                    bfi_agg = None
                    if bfi_records:
                        bfi_agg = p548.aggregate([
                            {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
                            for r in bfi_records
                        ])

                    audit[combo_name]["rows"][route_name][condition] = {
                        "free": free_agg,
                        "bfi_prefix2": bfi_agg,
                    }

                    elapsed = time.time() - t_start
                    eta = (elapsed / done_units) * (total_units - done_units) if done_units > 0 else 0
                    bfi_p2_str = f"{bfi_agg['clean_non_object_rate']:.2f}" if bfi_agg else "N/A"
                    log(
                        f"  [{done_units}/{total_units}] {combo_name} {route_name[:25]} {condition}: "
                        f"free_clean={free_agg['clean_non_object_rate']:.2f}, "
                        f"s0_margin={free_agg.get('step0_margin', 0):.2f}, "
                        f"s0_rank={free_agg.get('step0_target_rank', 0):.0f}, "
                        f"bfi_p2={bfi_p2_str} "
                        f"({time.time()-t_cond:.1f}s, ETA {eta/60:.1f}min)"
                    )

                    for rec in free_records[:3]:
                        samples.append({**rec, "combo": combo_name, "route": route_name})

                    checkpoint = {
                        "phase": 567, "model": args.model,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "done_units": done_units, "total_units": total_units,
                        "audit": audit,
                    }
                    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

                    compact.append({
                        "combo": combo_name, "route": route_name, "condition": condition,
                        "free_clean": free_agg["clean_non_object_rate"],
                        "free_echo": free_agg["object_echo_rate"],
                        "step0_margin": free_agg.get("step0_margin", 0),
                        "step0_rank": free_agg.get("step0_target_rank", 0),
                        "bfi_prefix2_clean": bfi_agg["clean_non_object_rate"] if bfi_agg else None,
                    })

        return {
            "phase": 567, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl, "pair": pair, "window": window,
            "combos": combos, "conditions": conditions, "routes": routes,
            "train_n": args.train_n, "test_n": args.test_n,
            "sample_seeds": seeds, "max_new_tokens": args.max_new_tokens,
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
    parser.add_argument("--test-n", type=int, default=12)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--add-alpha", type=float, default=6.0)
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
    out_path = out_dir / f"phase567_{args.model}_logit_field_source.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
