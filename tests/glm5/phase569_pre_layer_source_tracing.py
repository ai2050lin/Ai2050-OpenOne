#!/usr/bin/env python3
"""
Phase 569: Pre-Layer Source Tracing and Readout Geometry Audit
层前来源追踪与读出几何审计

Resolves the open questions from Phase 568:
1. h_in swap ≈ h_out swap, but WHERE does the h_in state come from?
   → Trace h_out swap and h_in swap at L{16,18,20,22,24,26,28} individually
   → Find the earliest layer whose h_in swap reproduces margin flip
2. Why does margin flip but target rank stays poor (e.g., rank=6537 with margin=+0.40)?
   → Audit target_group_mass / competitor_group_mass / entropy / best_non_target_rank
3. Is L28 a semantic origin or just a readout converter?
   → Compare pre-norm logits (W_U @ h_raw) vs post-norm logits (W_U @ Norm(h))
   → If post-norm margin >> pre-norm margin, Norm/Readout amplifies the flip

Three-state cache (h_in / h_attn / h_out) reused from Phase 568.
Extended layer list per model:
  GLM4      (peak=26, n_layers=40): [16,18,20,22,24,26,28]
  Qwen3     (peak=12, n_layers=36): [6,8,10,12,14]
  DS7B      (peak=18, n_layers=30): [12,14,16,18,20]

Run:
  python tests/glm5/phase569_pre_layer_source_tracing.py qwen3 --smoke
  python tests/glm5/phase569_pre_layer_source_tracing.py qwen3
  python tests/glm5/phase569_pre_layer_source_tracing.py glm4
  python tests/glm5/phase569_pre_layer_source_tracing.py deepseek7b
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
import phase559_prototype_generation_closure as p559  # noqa: E402
import phase565_early_gate_token_fork as p565  # noqa: E402
import phase568_locked_prefix_swap as p568  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402


OUT_ROOT = Path("results/glm5_phase569_pre_layer_source")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]

# Per-model extended trace layer list (peak ± offsets, no deeper than n_layers-2)
TRACE_LAYERS = {
    "qwen3": [6, 8, 10, 12, 14],
    "glm4": [16, 18, 20, 22, 24, 26, 28],
    "deepseek7b": [12, 14, 16, 18, 20],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Detailed logit statistics: margin, rank, group mass, entropy
# ============================================================================

def detailed_logit_stats(
    row: np.ndarray,
    groups: dict[str, list[int]],
) -> dict[str, float]:
    """Compute detailed logit-field statistics at one decoding step.

    Metrics:
      target_minus_competitor:    max(target logits) - max(competitor logits)
      target_best_rank:           1 + #tokens with logit > max(target logit)
      competitor_best_rank:       same for competitor group
      best_non_target_rank:       1 + #tokens with logit > max(non-target logit)
                                  where non-target = max of (competitor, cluster_other, off_cluster)
      target_group_mass:          softmax(row)[target_ids].sum()
      competitor_group_mass:      softmax(row)[competitor_ids].sum()
      max_logit:                  max over full vocab
      entropy:                    -sum p log p over full vocab (in nats)
      target_minus_best_non_target: max(target) - max(non_target)
    """
    row = row.astype(np.float64, copy=False)
    tgt_ids = [i for i in groups["target"] if 0 <= i < row.shape[0]]
    cmp_ids = [i for i in groups["competitor"] if 0 <= i < row.shape[0]]
    co_ids = [i for i in groups.get("cluster_other", []) if 0 <= i < row.shape[0]]
    off_ids = [i for i in groups.get("off_cluster", []) if 0 <= i < row.shape[0]]

    def gmax(ids):
        return float(np.max(row[ids])) if ids else float("-inf")

    target = gmax(tgt_ids)
    competitor = gmax(cmp_ids)
    cluster_other = gmax(co_ids)
    off_cluster = gmax(off_ids)
    non_target = max(competitor, cluster_other, off_cluster)

    def best_rank(val):
        if not np.isfinite(val):
            return float(row.shape[0])
        return float(1 + int(np.sum(row > val)))

    # Softmax with numerical stability
    m = float(np.max(row))
    exp = np.exp(row - m)
    p = exp / exp.sum()
    tgt_mass = float(p[tgt_ids].sum()) if tgt_ids else 0.0
    cmp_mass = float(p[cmp_ids].sum()) if cmp_ids else 0.0
    # Entropy in nats, ignoring numerical zeros
    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)))

    return {
        "target_minus_competitor": float(target - competitor),
        "target_minus_best_non_target": float(target - non_target),
        "target_best_rank": best_rank(target),
        "competitor_best_rank": best_rank(competitor),
        "best_non_target_rank": best_rank(non_target),
        "target_group_mass": tgt_mass,
        "competitor_group_mass": cmp_mass,
        "max_logit": m,
        "entropy": entropy,
    }


# ============================================================================
# Generation with state swap + detailed logit recording
# ============================================================================

def generate_with_swap_and_stats(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    swap_layer_ids: list[int],
    prompts: list[str],
    swap_spec: str,            # "baseline", "h_in", "h_attn", "h_out"
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    groups: dict[str, list[int]],
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    logit_steps: int = 4,
) -> dict[str, Any]:
    """Generate with optional state swap; record detailed logit stats at early steps."""
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)

    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    handles = p568.install_state_swap_hooks(
        layers, swap_layer_ids, batch_size, answer_pos, donor_caches, swap_spec
    )

    generated: list[list[int]] = [[] for _ in prompts]
    step_stats: list[list[dict[str, float]]] = [[] for _ in prompts]
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()
    for h in handles:
        h.remove()

    toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
    for i, tok in enumerate(toks):
        generated[i].append(int(tok))
        if logit_steps > 0:
            step_stats[i].append(detailed_logit_stats(logits0[i], groups))

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
        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))
            if step < logit_steps:
                step_stats[i].append(detailed_logit_stats(logits[i], groups))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generated_ids": generated, "generated_suffix": suffixes, "step_stats": step_stats}


# ============================================================================
# Norm/Readout audit: pre-norm vs post-norm logits
# ============================================================================

def audit_norm_vs_readout(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    swap_layer_ids: list[int],
    prompts: list[str],
    swap_spec: str,
    donor_caches: dict[str, dict[int, torch.Tensor]] | None,
    groups: dict[str, list[int]],
    W_U_np: np.ndarray,
    max_length: int,
) -> dict[str, Any]:
    """Compare pre-norm logits (W_U @ h_raw) vs post-norm logits (W_U @ Norm(h)).

    We hook the FINAL transformer layer's output (h_raw) and also capture the
    model output logits (which already include the final norm + unembed).

    For pre-norm, we manually compute W_U @ h_raw at the answer position.
    For post-norm, we use the model's output logits directly.
    """
    batch_size = len(prompts)
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    # Capture the final layer's output (h_raw before final norm)
    final_layer = layers[-1]
    captured_h_raw: dict[str, torch.Tensor] = {}

    def make_final_hook():
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            bidx = torch.arange(hs.shape[0], device=hs.device)
            p_dev = torch.full((hs.shape[0],), answer_pos, dtype=torch.long, device=hs.device)
            captured_h_raw["h_raw"] = hs[bidx, p_dev, :].detach().float().cpu()
            return output
        return hook

    # Install swap hooks (same as generation, but we don't need KV cache beyond step 0)
    swap_handles = p568.install_state_swap_hooks(
        layers, swap_layer_ids, batch_size, answer_pos, donor_caches, swap_spec
    )
    final_handle = final_layer.register_forward_hook(make_final_hook())

    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=False, return_dict=True)
            post_logits = out.logits[:, answer_pos, :].float().cpu().numpy()
    finally:
        for h in swap_handles:
            h.remove()
        final_handle.remove()

    # h_raw shape: [batch, d_model]
    h_raw = captured_h_raw["h_raw"].numpy().astype(np.float32)  # [B, d]
    # W_U_np shape: [vocab, d] → logits_pre = h_raw @ W_U_np.T  → [B, vocab]
    pre_logits = h_raw @ W_U_np.T  # raw hidden state projected to vocab

    # Compute stats per example, then average
    pre_stats = [detailed_logit_stats(pre_logits[i], groups) for i in range(batch_size)]
    post_stats = [detailed_logit_stats(post_logits[i], groups) for i in range(batch_size)]

    def avg(key):
        return float(np.mean([s[key] for s in pre_stats])), float(np.mean([s[key] for s in post_stats]))

    keys = ["target_minus_competitor", "target_best_rank", "competitor_best_rank",
            "best_non_target_rank", "target_group_mass", "competitor_group_mass",
            "entropy", "max_logit"]
    result = {}
    for k in keys:
        pre_v, post_v = avg(k)
        result[f"pre_{k}"] = pre_v
        result[f"post_{k}"] = post_v
        result[f"delta_{k}"] = float(post_v - pre_v)
    result["n_examples"] = batch_size
    return result


# ============================================================================
# Main run loop per model
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        _, window = next(iter(windows.items()))
        W_U = get_W_U(model, args.model).astype(np.float32)
        groups = p544.token_groups(tokenizer, pair)
        prompt_sets = p548.build_prompts(pair, args.test_n, scaffolds)
        pos_label, neg_label = PAIR_SPECS[pair]
        log(f"{args.model}: phase569 pair={pair}, window={window}, trace_layers={TRACE_LAYERS[args.model]}")

        # The "all layers" combo is the original Phase 568 window (peak-2, peak, peak+2)
        all_layers_window = list(window)
        # Extended trace layers (model-specific), filtered to valid range
        trace_layers = [L for L in TRACE_LAYERS[args.model] if 0 <= L < info.n_layers]

        # Need caches at ALL trace layers + the window layers for various swaps
        all_cache_layers = sorted(set(trace_layers + all_layers_window))

        audit: dict[str, Any] = {"trace_layers": trace_layers, "window": all_layers_window, "rows": {}}
        compact: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase569_{args.model}_checkpoint.json"

        # Count total units
        n_per_seed = 2 + 2 * len(trace_layers) + 1  # baseline + repeat4 + (h_in + h_out per trace) + random
        total_units = len(routes) * len(seeds) * n_per_seed + 2 * len(routes)  # +2 norm audits per route
        done_units = 0
        t_start = time.time()

        for route in routes:
            route_name = route["name"]
            audit["rows"][route_name] = {}
            prompt_rows = prompt_sets[route["recipient_scaffold"]]
            prompts = [r["prompt"] for r in prompt_rows]

            # Collect donor three-state caches at all relevant layers
            donor_scaffold = route["donor_scaffold"]
            donor_objs = CATEGORY_BANK[pos_label][-args.test_n:]
            repeat_idx = min(4, len(donor_objs) - 1)
            donor_repeat4_prompts = [
                p548.forbidden_prompt(donor_scaffold, donor_objs[repeat_idx], pos_label, neg_label)
            ] * args.test_n
            donor_enc = tokenizer(donor_repeat4_prompts, return_tensors="pt",
                                  padding=True, truncation=True, max_length=args.max_length)
            donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
            donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
            repeat4_caches = p568.collect_three_state_cache(
                model, layers, donor_batch, donor_pos, all_cache_layers
            )
            random_caches = p568.transform_three_state_cache(repeat4_caches, "random", 0)
            log(f"  Collected repeat4 + random caches at layers {all_cache_layers}")

            all_records: dict[str, list[dict[str, Any]]] = {}

            for seed in seeds:
                # === baseline_free ===
                baseline = generate_with_swap_and_stats(
                    model, tokenizer, device, layers, all_layers_window, prompts,
                    "baseline", None, groups, route["mode"], seed,
                    args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                    args.logit_steps,
                )
                done_units += 1
                all_records.setdefault("baseline_free", []).extend([
                    {"prompt_index": i, "object": row["object"], "seed": seed,
                     "condition": "baseline_free",
                     **{k: baseline[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, row in enumerate(prompt_rows)
                ])

                # === repeat4_free (h_out swap on all_layers_window) ===
                repeat4 = generate_with_swap_and_stats(
                    model, tokenizer, device, layers, all_layers_window, prompts,
                    "h_out", repeat4_caches, groups, route["mode"], seed,
                    args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                    args.logit_steps,
                )
                done_units += 1
                all_records.setdefault("repeat4_free", []).extend([
                    {"prompt_index": i, "object": row["object"], "seed": seed,
                     "condition": "repeat4_free",
                     **{k: repeat4[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, row in enumerate(prompt_rows)
                ])

                # === Layer-wise h_out and h_in swap at each trace layer ===
                for L in trace_layers:
                    for swap_spec, suffix in [("h_out", "h_out_swap"), ("h_in", "h_in_swap")]:
                        cond_name = f"L{L}_{suffix}"
                        result = generate_with_swap_and_stats(
                            model, tokenizer, device, layers, [L], prompts,
                            swap_spec, repeat4_caches, groups, route["mode"], seed,
                            args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                            args.logit_steps,
                        )
                        done_units += 1
                        all_records.setdefault(cond_name, []).extend([
                            {"prompt_index": i, "object": row["object"], "seed": seed,
                             "condition": cond_name,
                             **{k: result[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                            for i, row in enumerate(prompt_rows)
                        ])

                # === random_h_out_swap control (on all_layers_window) ===
                random_res = generate_with_swap_and_stats(
                    model, tokenizer, device, layers, all_layers_window, prompts,
                    "h_out", random_caches, groups, route["mode"], seed,
                    args.max_new_tokens, args.temperature, args.top_p, args.max_length,
                    args.logit_steps,
                )
                done_units += 1
                all_records.setdefault("random_h_out_swap", []).extend([
                    {"prompt_index": i, "object": row["object"], "seed": seed,
                     "condition": "random_h_out_swap",
                     **{k: random_res[k][i] for k in ["generated_suffix", "generated_ids", "step_stats"]}}
                    for i, row in enumerate(prompt_rows)
                ])

                elapsed = time.time() - t_start
                eta = (elapsed / done_units) * (total_units - done_units) if done_units > 0 else 0
                log(f"  [{done_units}/{total_units}] seed={seed} done (ETA {eta/60:.1f}min)")

            # === Norm/Readout audit (no generation, just forward pass) ===
            log(f"  Running Norm/Readout audit (baseline vs repeat4)...")
            norm_audit_baseline = audit_norm_vs_readout(
                model, tokenizer, device, layers, all_layers_window, prompts,
                "baseline", None, groups, W_U, args.max_length,
            )
            done_units += 1
            norm_audit_repeat4 = audit_norm_vs_readout(
                model, tokenizer, device, layers, all_layers_window, prompts,
                "h_out", repeat4_caches, groups, W_U, args.max_length,
            )
            done_units += 1
            audit["rows"][route_name]["norm_readout_audit"] = {
                "baseline": norm_audit_baseline,
                "repeat4_h_out_swap": norm_audit_repeat4,
            }
            log(f"  Norm/Readout audit done: "
                f"baseline post_margin={norm_audit_baseline['post_target_minus_competitor']:.2f} "
                f"pre_margin={norm_audit_baseline['pre_target_minus_competitor']:.2f} | "
                f"repeat4 post_margin={norm_audit_repeat4['post_target_minus_competitor']:.2f} "
                f"pre_margin={norm_audit_repeat4['pre_target_minus_competitor']:.2f}")

            # === Aggregate all conditions ===
            cond_order = (
                ["baseline_free", "repeat4_free"]
                + [f"L{L}_h_out_swap" for L in trace_layers]
                + [f"L{L}_h_in_swap" for L in trace_layers]
                + ["random_h_out_swap"]
            )
            for cond in cond_order:
                recs = all_records.get(cond, [])
                if not recs:
                    continue
                classified = [
                    {**r, **p548.classify_suffix(r["generated_suffix"], r["object"], pos_label, neg_label)}
                    for r in recs
                ]
                agg = p548.aggregate(classified)
                # Detailed step statistics
                for step in range(args.logit_steps):
                    vals = [r["step_stats"][step] for r in recs if len(r.get("step_stats", [])) > step]
                    if vals:
                        for key in ["target_minus_competitor", "target_best_rank",
                                    "competitor_best_rank", "best_non_target_rank",
                                    "target_group_mass", "competitor_group_mass",
                                    "entropy", "target_minus_best_non_target"]:
                            agg[f"step{step}_{key}"] = float(np.mean([v[key] for v in vals]))
                audit["rows"][route_name][cond] = agg

                compact.append({
                    "route": route_name, "condition": cond,
                    "free_clean": agg["clean_non_object_rate"],
                    "step0_margin": agg.get("step0_target_minus_competitor", 0),
                    "step0_target_rank": agg.get("step0_target_best_rank", 0),
                    "step0_best_non_target_rank": agg.get("step0_best_non_target_rank", 0),
                    "step0_target_mass": agg.get("step0_target_group_mass", 0),
                    "step0_competitor_mass": agg.get("step0_competitor_group_mass", 0),
                    "step0_entropy": agg.get("step0_entropy", 0),
                })
                log(f"  {cond}: clean={agg['clean_non_object_rate']:.2f}, "
                    f"s0_margin={agg.get('step0_target_minus_competitor',0):.2f}, "
                    f"s0_tgt_rank={agg.get('step0_target_best_rank',0):.0f}, "
                    f"s0_tgt_mass={agg.get('step0_target_group_mass',0):.4f}, "
                    f"s0_entropy={agg.get('step0_entropy',0):.2f}")

                for rec in classified[:2]:
                    samples.append({**rec, "route": route_name})

                # Checkpoint after each condition
                checkpoint = {
                    "phase": 569, "model": args.model,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "done_units": done_units, "total_units": total_units,
                    "audit": audit,
                }
                checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2),
                                           encoding="utf-8")

        return {
            "phase": 569, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl, "pair": pair, "window": all_layers_window,
            "trace_layers": trace_layers, "routes": routes,
            "train_n": args.train_n, "test_n": args.test_n,
            "sample_seeds": seeds, "max_new_tokens": args.max_new_tokens,
            "logit_steps": args.logit_steps,
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
    parser.add_argument("--windows", default=None,
                        help="Layer window spec (default: peak-2,peak,peak+2)")
    parser.add_argument("--pair", default="vehicle_tool")
    parser.add_argument("--train-n", type=int, default=12)
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127,131,137")
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--logit-steps", type=int, default=4)
    parser.add_argument("--max-saved-samples", type=int, default=300)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: small sample, few seeds, short generation")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.test_n = 4
        args.sample_seeds = "101,103"
        args.max_new_tokens = 8
        args.logit_steps = 3
        log("SMOKE TEST MODE: test_n=4, seeds=2, max_tokens=8")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase569_{args.model}_pre_layer_source{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
