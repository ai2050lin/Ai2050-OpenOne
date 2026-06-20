#!/usr/bin/env python3
"""
Phase 562: Trajectory Response and Covariant Donor Audit
轨迹响应与协变供体审计

From Phase 561 beta sweep → Phase 562 dynamical response measurement.

Experiments integrated in one script:
  Exp1: Relaxation length — one-shot inject then free-generate, record per-step token type
  Exp2: Tangent vs Normal — decompose donor direction vs baseline trajectory tangent
  Exp3: Trajectory distance — record hidden state norms + distance to baseline
  Exp4: Degeneration taxonomy — classify output quality (repetition, collapse, etc.)

Conditions:
  1. baseline (record hidden states for tangent computation)
  2. one_shot_repeat2 (helicopter) — relaxation + trajectory
  3. one_shot_repeat4 (rocket) — relaxation + trajectory
  4. one_shot_mean — relaxation + trajectory
  5. one_shot_random — relaxation + trajectory
  6. add_tangent_repeat2 — tangent component injection
  7. add_normal_repeat2 — normal component injection

All conditions: one-shot surgery at step 0, then free KV-cache generation for 16 tokens.
Per-step recording: token_type, target_rank, hidden_state_norm.
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
from phase530_state_pair_decomposition import load_model_bf16_flash  # noqa: E402
from phase539_interface_cluster_mechanism import PAIR_SPECS, layer_windows  # noqa: E402
import phase544_natural_decode_policy_gate_audit as p544  # noqa: E402
import phase545_sampling_stability_cross_category as p545  # noqa: E402
import phase548_paraphrase_candidate_robustness as p548  # noqa: E402
import phase558_prototype_object_binding_audit as p558  # noqa: E402
import phase559_prototype_generation_closure as p559  # noqa: E402
from phase536_pair_quality_selectivity import CATEGORY_BANK  # noqa: E402


OUT_ROOT = Path("results/glm5_phase562_trajectory_response")

DEFAULT_ROUTES = [
    "forbidden_sentence_completion:temperature<-forbidden_definition",
    "forbidden_definition:top_p<-forbidden_definition",
]

CONDITIONS_562 = [
    "baseline",
    "one_shot_repeat2",
    "one_shot_repeat4",
    "one_shot_mean",
    "one_shot_random",
    "add_tangent_repeat2",
    "add_normal_repeat2",
]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Condition plan for Phase 562
# ============================================================================

def condition_plan_562(condition: str) -> dict[str, Any]:
    """Map Phase 562 condition names to surgery plans."""
    if condition == "baseline":
        return {"type": "baseline", "site": "none", "remove": False, "add": False,
                "restore": None, "donor_category": None, "donor_variant": None,
                "component": None}

    if condition.startswith("one_shot_"):
        tail = condition[len("one_shot_"):]
        # Map short names to Phase 558 donor variant names
        variant_map = {"mean": "mean_cache", "random": "random_cache"}
        p558_variant = variant_map.get(tail, tail)
        p558_cond = f"resid_donor_vehicle_{p558_variant}_add"
        plan = p558.condition_plan(p558_cond)
        plan["type"] = "one_shot"
        # Store the short variant name for donor cache lookup
        plan["short_variant"] = tail
        return plan

    if condition == "add_tangent_repeat2":
        return {"type": "add_tangent", "site": "resid", "remove": False, "add": True,
                "restore": None, "donor_category": "vehicle", "donor_variant": "repeat2",
                "component": "residual_perp"}

    if condition == "add_normal_repeat2":
        return {"type": "add_normal", "site": "resid", "remove": False, "add": True,
                "restore": None, "donor_category": "vehicle", "donor_variant": "repeat2",
                "component": "residual_perp"}

    raise ValueError(f"unknown condition: {condition}")


# ============================================================================
# Per-sample direction addition (for tangent/normal injection)
# ============================================================================

def add_direction_batch(x: torch.Tensor, pos: torch.Tensor,
                        directions: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add per-sample directions at given positions.
    x: (batch, seq, d_model)
    pos: (batch,) long tensor on x.device
    directions: (batch, d_model) on CPU (float32)
    """
    out = x.clone()
    bidx = torch.arange(out.shape[0], device=out.device)
    d = directions.to(out.device).float()
    norms = d.norm(dim=-1, keepdim=True) + 1e-8
    d = d / norms
    out[bidx, pos, :] = out[bidx, pos, :] + (float(alpha) * d).to(out.dtype)
    return out


# ============================================================================
# Hidden state recorder (for baseline trajectory tangent)
# ============================================================================

class HiddenStateRecorder:
    """Records hidden states at step 0 and step 1 for tangent computation."""

    def __init__(self, layer_ids: list[int]):
        self.layer_ids = layer_ids
        self.step = 0
        self.h_step0: dict[int, torch.Tensor] = {}
        self.h_step1: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.step = 0
        self.h_step0 = {}
        self.h_step1 = {}

    def make_hook(self, layer_id: int):
        def hook(_module, _inp, output):
            hs = p559.tensor_from_output(output)
            bidx = torch.arange(hs.shape[0], device=hs.device)
            last_pos = hs.shape[1] - 1
            if self.step == 0:
                self.h_step0[layer_id] = hs[bidx, last_pos, :].detach().float().cpu()
            elif self.step == 1:
                self.h_step1[layer_id] = hs[bidx, last_pos, :].detach().float().cpu()
        return hook


# ============================================================================
# Tangent / Normal decomposition
# ============================================================================

def compute_tangent_normal(
    donor_cache: dict[int, torch.Tensor],
    h_step0: dict[int, torch.Tensor],
    h_step1: dict[int, torch.Tensor],
    layer_ids: list[int],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Compute tangent and normal components of donor direction per layer.

    u = a_donor - h_step0  (donor direction)
    v = h_step1 - h_step0  (baseline trajectory tangent)
    u_parallel = proj_v(u)
    u_perp = u - u_parallel
    """
    tangent_dirs: dict[int, torch.Tensor] = {}
    normal_dirs: dict[int, torch.Tensor] = {}

    for lid in layer_ids:
        a = donor_cache[lid].float().cpu()  # (batch, d) — ensure CPU
        h0 = h_step0[lid].float().cpu()     # (batch, d) — ensure CPU
        h1 = h_step1[lid].float().cpu()     # (batch, d) — ensure CPU

        u = a - h0  # (batch, d)
        v = h1 - h0  # (batch, d)

        # Per-sample projection: u_parallel = (u·v / |v|^2) * v
        v_sq = (v * v).sum(dim=-1, keepdim=True) + 1e-8  # (batch, 1)
        u_dot_v = (u * v).sum(dim=-1, keepdim=True)  # (batch, 1)
        u_parallel = (u_dot_v / v_sq) * v  # (batch, d)
        u_perp = u - u_parallel  # (batch, d)

        tangent_dirs[lid] = u_parallel
        normal_dirs[lid] = u_perp

    return tangent_dirs, normal_dirs


# ============================================================================
# Core generation function with per-step recording
# ============================================================================

def generate_with_recording(
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
    donor_cache: dict[int, torch.Tensor] | None = None,
    tangent_dirs: dict[int, torch.Tensor] | None = None,
    normal_dirs: dict[int, torch.Tensor] | None = None,
    recorder: HiddenStateRecorder | None = None,
) -> dict[str, Any]:
    """Generate with one-shot surgery at step 0, record per-step metrics.

    Returns dict with:
      generated_ids, suffixes, per_step_types, per_step_target_ranks,
      first_types, hidden_norms (step0 and stepT)
    """
    plan = condition_plan_562(condition)
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)

    # Left-pad for KV cache
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    answer_pos = seq_len - 1
    tokenizer.padding_side = original_padding_side

    # Collect donor cache if needed (for one_shot conditions)
    local_donor_cache: dict[int, torch.Tensor] = {}
    if plan["type"] == "one_shot" and plan["restore"] is not None and donor_prompts is not None:
        if donor_cache is not None:
            local_donor_cache = donor_cache
        else:
            donor_enc = tokenizer(donor_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
            donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
            raw_cache = p559.collect_donor_cache(
                model, layers, donor_batch, donor_pos, components_by_layer, layer_ids,
                plan["site"], plan["restore"], add_alpha,
            )
            local_donor_cache = p559.transform_restore_cache(raw_cache, plan.get("donor_variant"), 0)

    # Also collect donor cache for add_tangent/add_normal (to get donor direction)
    if plan["type"] in ("add_tangent", "add_normal") and donor_cache is not None:
        local_donor_cache = donor_cache

    # --- Install hooks for step 0 ---
    handles: list[Any] = []
    pos_cpu = torch.full((batch_size,), answer_pos, dtype=torch.long)

    if plan["site"] != "none":
        for layer_id in layer_ids:
            layer = layers[layer_id]
            site = p559.module_for_site(layer, plan["site"])
            direction_np = components_by_layer[str(layer_id)][plan["component"]]
            direction_cpu = torch.tensor(p559.normalize_vec(direction_np), dtype=torch.float32)
            cached = local_donor_cache.get(layer_id)
            should_remove = bool(plan["remove"])
            should_restore = bool(plan["restore"] is not None and cached is not None)

            # Determine injection direction for add conditions
            tan_dir = tangent_dirs.get(layer_id) if tangent_dirs else None
            nor_dir = normal_dirs.get(layer_id) if normal_dirs else None

            def make_hook(d_vec_cpu, p_vec_cpu, cached_vec, rm, rs, site_name,
                          cond_type, tan_d, nor_d):
                def hook(_module, _inp, output):
                    hs = p559.tensor_from_output(output)
                    out = hs
                    p_dev = p_vec_cpu.to(out.device)
                    if rm:
                        out = p559.project_remove(out, p_dev, d_vec_cpu, remove_scale)
                    if rs and cached_vec is not None:
                        bidx = torch.arange(out.shape[0], device=out.device)
                        out = out.clone()
                        out[bidx, p_dev, :] = cached_vec.to(out.device, dtype=out.dtype)
                    if cond_type == "add_tangent" and tan_d is not None:
                        out = add_direction_batch(out, p_dev, tan_d, add_alpha)
                    if cond_type == "add_normal" and nor_d is not None:
                        out = add_direction_batch(out, p_dev, nor_d, add_alpha)
                    return p559.replace_output(output, out)
                return hook

            handles.append(site.register_forward_hook(
                make_hook(direction_cpu, pos_cpu, cached, should_remove, should_restore,
                          plan["site"], plan["type"], tan_dir, nor_dir)
            ))

    # Install recorder hooks for baseline
    rec_handles: list[Any] = []
    if recorder is not None:
        recorder.reset()
        for layer_id in layer_ids:
            layer = layers[layer_id]
            rec_handles.append(layer.register_forward_hook(recorder.make_hook(layer_id)))

    # --- Step 0: full forward with surgery ---
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits_step0 = out.logits[:, answer_pos, :].float().cpu().numpy()

    # Record step 0 hidden state norms
    step0_norms: dict[int, list[float]] = {}
    if recorder is not None:
        for lid in layer_ids:
            if lid in recorder.h_step0:
                step0_norms[lid] = recorder.h_step0[lid].norm(dim=-1).tolist()

    # Remove surgery hooks
    for h in handles:
        h.remove()

    # Update recorder step
    if recorder is not None:
        recorder.step = 1

    # --- Sample first token ---
    toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits_step0]
    first_types = [p544.token_type(int(t), groups) for t in toks]
    first_target_ranks = [float(p544.best_rank(row, groups["target"])) for row in logits_step0]
    first_competitor_ranks = [float(p544.best_rank(row, groups["competitor"])) for row in logits_step0]
    generated: list[list[int]] = [[int(t)] for t in toks]
    per_step_types: list[list[str]] = [list(first_types)]
    per_step_target_ranks: list[list[float]] = [list(first_target_ranks)]

    # --- Steps 1+: free generation with KV cache ---
    full_attn_mask = attention_mask
    for step in range(1, max_new_tokens):
        if recorder is not None:
            recorder.step = step if step <= 1 else 2  # only record step 0 and 1

        new_ids = torch.tensor([[t] for t in toks], dtype=torch.long, device=device)
        new_mask_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
        full_attn_mask = torch.cat([full_attn_mask, new_mask_col], dim=1)

        with torch.inference_mode():
            out = model(input_ids=new_ids, attention_mask=full_attn_mask,
                        past_key_values=past_kv, use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float().cpu().numpy()

        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        step_types = [p544.token_type(int(t), groups) for t in toks]
        step_ranks = [float(p544.best_rank(row, groups["target"])) for row in logits]
        per_step_types.append(step_types)
        per_step_target_ranks.append(step_ranks)
        for i, t in enumerate(toks):
            generated[i].append(int(t))

    # Remove recorder hooks
    for h in rec_handles:
        h.remove()

    # Record step 1 hidden state norms (if available)
    step1_norms: dict[int, list[float]] = {}
    if recorder is not None:
        for lid in layer_ids:
            if lid in recorder.h_step1:
                step1_norms[lid] = recorder.h_step1[lid].norm(dim=-1).tolist()

    # Decode suffixes
    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]

    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "generated_ids": generated,
        "suffixes": suffixes,
        "first_types": first_types,
        "first_target_ranks": first_target_ranks,
        "first_competitor_ranks": first_competitor_ranks,
        "per_step_types": per_step_types,
        "per_step_target_ranks": per_step_target_ranks,
        "step0_norms": step0_norms,
        "step1_norms": step1_norms,
    }


# ============================================================================
# Relaxation length computation
# ============================================================================

def compute_relaxation_length(per_step_types: list[list[str]]) -> dict[str, Any]:
    """Compute relaxation metrics from per-step token types.

    per_step_types: (n_steps, batch_size) — token type per step per sample.
    Returns aggregated metrics.
    """
    n_steps = len(per_step_types)
    n_samples = len(per_step_types[0]) if n_steps > 0 else 0

    # Per-step target rate
    target_rates = []
    competitor_rates = []
    other_rates = []
    for step_types in per_step_types:
        n = max(1, len(step_types))
        target_rates.append(sum(1 for t in step_types if t == "target") / n)
        competitor_rates.append(sum(1 for t in step_types if t == "competitor") / n)
        other_rates.append(sum(1 for t in step_types if t == "other") / n)

    # Per-sample relaxation length: steps until first "other" after target
    sample_relax: list[int] = []
    for i in range(n_samples):
        relax = 0
        for s in range(n_steps):
            if per_step_types[s][i] == "target":
                relax = s + 1
            elif per_step_types[s][i] == "other" and relax > 0:
                break
        sample_relax.append(relax)

    # Target persistence: how many of first K steps have target
    k = min(4, n_steps)
    target_persistence_4 = sum(1 for i in range(n_samples)
                                if any(per_step_types[s][i] == "target" for s in range(k))) / max(1, n_samples)

    return {
        "target_rates": target_rates,
        "competitor_rates": competitor_rates,
        "other_rates": other_rates,
        "mean_relaxation_length": float(np.mean(sample_relax)) if sample_relax else 0.0,
        "median_relaxation_length": float(np.median(sample_relax)) if sample_relax else 0.0,
        "max_relaxation_length": int(max(sample_relax)) if sample_relax else 0,
        "target_persistence_first4": float(target_persistence_4),
    }


# ============================================================================
# Degeneration taxonomy
# ============================================================================

def classify_degeneration(suffix: str) -> str:
    """Classify degeneration type of generated suffix."""
    low = suffix.lower().strip()
    if len(low) < 2:
        return "short"
    words = low.split()
    # Repetition: same word repeated 3+ times or half==half
    if len(words) >= 6:
        half = len(words) // 2
        if words[:half] == words[half:half * 2]:
            return "repetition"
    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    if word_counts and max(word_counts.values()) >= 4 and len(words) >= 6:
        return "repetition"
    # Garbage: low alpha ratio
    alpha_count = sum(1 for c in low if c.isalpha() or c.isspace())
    if alpha_count / max(1, len(low)) < 0.5:
        return "garbage"
    # Syntax collapse: high punctuation ratio
    punct_count = sum(1 for c in low if not c.isalnum() and not c.isspace())
    if punct_count / max(1, len(low)) > 0.35:
        return "syntax_collapse"
    return "normal"


# ============================================================================
# Main run
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
    p544.FAMILY_TERMS.update(p545.EXTRA_FAMILY_TERMS)
    pair = args.pair
    routes = p558.parse_routes(args.routes)
    scaffolds = sorted(set([r["recipient_scaffold"] for r in routes] + [r["donor_scaffold"] for r in routes]))
    conditions = CONDITIONS_562
    sample_seeds = [int(x.strip()) for x in args.sample_seeds.split(",") if x.strip()]

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

        # Object audit
        pos_label = PAIR_SPECS[pair][0]
        objects = CATEGORY_BANK[pos_label][-args.test_n:]
        obj_audit = [{"repeat_index": i, "object": o,
                      "token_length": len(tokenizer.encode(o, add_special_tokens=False))}
                     for i, o in enumerate(objects)]
        log(f"{args.model}: object audit: repeat2={objects[2] if len(objects)>2 else '?'}, "
            f"repeat4={objects[4] if len(objects)>4 else '?'}")

        # Build components
        components_by_layer = p558.build_components_by_layer(
            model, tokenizer, device, pair, all_layers, args.train_n, args.batch_size, args.max_length, W_U
        )
        log(f"{args.model}: phase562 pair={pair}, combos={combos}, routes={[r['name'] for r in routes]}")
        log(f"  conditions={len(conditions)}, seeds={len(sample_seeds)}, max_tokens={args.max_new_tokens}")

        audit: dict[str, Any] = {}
        compact: list[dict[str, Any]] = []
        saved_samples: list[dict[str, Any]] = []
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase562_{args.model}_checkpoint.json"

        total_units = len(combos) * len(routes) * len(conditions)
        done_units = 0
        t_start = time.time()

        for combo_name, layer_ids in combos.items():
            audit[combo_name] = {"layers": layer_ids, "rows": {}}
            recorder = HiddenStateRecorder(layer_ids)

            for route in routes:
                recipient_scaffold = route["recipient_scaffold"]
                donor_scaffold = route["donor_scaffold"]
                mode = route["mode"]
                route_name = route["name"]
                audit[combo_name]["rows"][route_name] = {}
                prompt_rows = prompt_sets[recipient_scaffold]

                # Reset baseline trajectory data for this route
                baseline_h_step0 = {}
                baseline_h_step1 = {}
                tangent_dirs = None
                normal_dirs = None

                # Pre-collect donor caches for repeat2, repeat4, mean, random
                # (to avoid re-collecting for each seed)
                donor_caches: dict[str, dict[int, torch.Tensor]] = {}
                for variant in ["repeat2", "repeat4", "mean", "random"]:
                    # Map to Phase 558 condition name
                    variant_map = {"mean": "mean_cache", "random": "random_cache"}
                    p558_variant = variant_map.get(variant, variant)
                    cond_name = f"resid_donor_vehicle_{p558_variant}_add"
                    plan = p558.condition_plan(cond_name)
                    donor_rows = p558.donor_rows_for(
                        pair, donor_scaffold, plan.get("donor_category"),
                        plan.get("donor_variant"), args.test_n,
                    )
                    donor_prompts = [r["prompt"] for r in donor_rows] if donor_rows else None
                    if donor_prompts:
                        donor_enc = tokenizer(donor_prompts, return_tensors="pt", padding=True,
                                              truncation=True, max_length=args.max_length)
                        donor_batch = {k: v.to(device) for k, v in donor_enc.items()}
                        donor_pos = donor_batch["attention_mask"].sum(dim=1) - 1
                        raw_cache = p559.collect_donor_cache(
                            model, layers, donor_batch, donor_pos, components_by_layer, layer_ids,
                            "resid", "add_perp", args.add_alpha,
                        )
                        donor_caches[variant] = p559.transform_restore_cache(raw_cache, plan.get("donor_variant"), 0)

                log(f"  Collected donor caches for {len(donor_caches)} variants")

                # Baseline trajectory data persists across conditions within a route
                baseline_h_step0: dict[int, torch.Tensor] = {}
                baseline_h_step1: dict[int, torch.Tensor] = {}
                tangent_dirs: dict[int, torch.Tensor] | None = None
                normal_dirs: dict[int, torch.Tensor] | None = None

                for condition in conditions:
                    t_cond = time.time()
                    all_records: list[dict[str, Any]] = []
                    all_relaxation: list[dict[str, Any]] = []
                    all_degeneration: list[str] = []

                    for seed in sample_seeds:
                        prompts = [r["prompt"] for r in prompt_rows]

                        # Determine donor prompts and cache for this condition
                        donor_prompts = None
                        donor_cache = None
                        if condition.startswith("one_shot_"):
                            variant = condition[len("one_shot_"):]
                            # Map to Phase 558 condition for donor_rows
                            variant_map = {"mean": "mean_cache", "random": "random_cache"}
                            p558_variant = variant_map.get(variant, variant)
                            cond_name = f"resid_donor_vehicle_{p558_variant}_add"
                            plan = p558.condition_plan(cond_name)
                            donor_rows = p558.donor_rows_for(
                                pair, donor_scaffold, plan.get("donor_category"),
                                plan.get("donor_variant"), args.test_n,
                            )
                            donor_prompts = [r["prompt"] for r in donor_rows] if donor_rows else None
                            donor_cache = donor_caches.get(variant)
                        elif condition in ("add_tangent_repeat2", "add_normal_repeat2"):
                            donor_cache = donor_caches.get("repeat2")

                        # For baseline, use recorder
                        use_recorder = (condition == "baseline")

                        # For add_tangent/add_normal, need baseline h_step0/h_step1
                        if condition in ("add_tangent_repeat2", "add_normal_repeat2"):
                            if not baseline_h_step0:
                                log(f"    Need baseline trajectory for {condition}, but no baseline recorded yet. Skipping tangent computation.")
                                # Use donor_cache directly as direction (fallback)
                                tangent_dirs = None
                                normal_dirs = None

                        gen_result = generate_with_recording(
                            model, tokenizer, device, layers, prompts, donor_prompts,
                            components_by_layer, layer_ids, condition, groups, mode,
                            args.max_new_tokens, seed, args.temperature, args.top_p,
                            args.remove_scale, args.add_alpha, args.max_length,
                            donor_cache=donor_cache,
                            tangent_dirs=tangent_dirs,
                            normal_dirs=normal_dirs,
                            recorder=recorder if use_recorder else None,
                        )

                        # Save baseline hidden states for tangent computation
                        if use_recorder:
                            baseline_h_step0 = dict(recorder.h_step0)
                            baseline_h_step1 = dict(recorder.h_step1)
                            # Compute tangent/normal for repeat2
                            if "repeat2" in donor_caches:
                                tangent_dirs, normal_dirs = compute_tangent_normal(
                                    donor_caches["repeat2"], baseline_h_step0, baseline_h_step1, layer_ids
                                )
                                # Log tangent/normal stats
                                for lid in layer_ids:
                                    if lid in tangent_dirs:
                                        tan_norm = tangent_dirs[lid].norm(dim=-1).mean().item()
                                        nor_norm = normal_dirs[lid].norm(dim=-1).mean().item()
                                        u_norm = donor_caches["repeat2"][lid].float().norm(dim=-1).mean().item()
                                        log(f"    L{lid} dir decomp: |u|={u_norm:.2f}, |tangent|={tan_norm:.2f}, |normal|={nor_norm:.2f}")

                        # Classify suffixes
                        pos_label_route, neg_label_route = PAIR_SPECS[pair]
                        records = []
                        for i, (row, suffix, ids, ft, ftr, fcr) in enumerate(zip(
                            prompt_rows, gen_result["suffixes"], gen_result["generated_ids"],
                            gen_result["first_types"], gen_result["first_target_ranks"],
                            gen_result["first_competitor_ranks"]
                        )):
                            cls = p548.classify_suffix(suffix, row["object"], pos_label_route, neg_label_route)
                            deg = classify_degeneration(suffix)
                            all_degeneration.append(deg)
                            rec = {
                                "prompt_index": i,
                                "object": row["object"],
                                "condition": condition,
                                "seed": seed,
                                "generated_suffix": suffix,
                                "first_type": ft,
                                "first_target_rank": float(ftr),
                                "first_competitor_rank": float(fcr),
                                "per_step_types": [s[i] for s in gen_result["per_step_types"]],
                                "degeneration": deg,
                                **cls,
                            }
                            records.append(rec)
                            all_records.append(rec)

                        # Relaxation metrics
                        relax = compute_relaxation_length(gen_result["per_step_types"])
                        all_relaxation.append({"seed": seed, **relax})

                    # Aggregate
                    agg = p548.aggregate(all_records)
                    # Average relaxation
                    avg_relax = np.mean([r["mean_relaxation_length"] for r in all_relaxation]) if all_relaxation else 0
                    avg_target_persist = np.mean([r["target_persistence_first4"] for r in all_relaxation]) if all_relaxation else 0
                    # Average per-step target rate
                    n_steps = len(all_relaxation[0]["target_rates"]) if all_relaxation else 0
                    avg_target_rates = []
                    for s in range(n_steps):
                        avg_target_rates.append(float(np.mean([r["target_rates"][s] for r in all_relaxation])))
                    # Degeneration distribution
                    from collections import Counter
                    deg_counts = Counter(all_degeneration)
                    deg_dist = {k: v / max(1, len(all_degeneration)) for k, v in deg_counts.items()}

                    agg["mean_relaxation_length"] = float(avg_relax)
                    agg["target_persistence_first4"] = float(avg_target_persist)
                    agg["avg_target_rates"] = avg_target_rates
                    agg["degeneration_distribution"] = deg_dist
                    agg["seed_relaxation"] = all_relaxation
                    audit[combo_name]["rows"][route_name][condition] = agg

                    done_units += 1
                    elapsed = time.time() - t_start
                    eta = (elapsed / done_units) * (total_units - done_units) if done_units > 0 else 0
                    log(
                        f"  [{done_units}/{total_units}] {combo_name} {route_name[:30]} {condition}: "
                        f"clean_no={agg['clean_non_object_rate']:.2f}, "
                        f"relax={avg_relax:.1f}, "
                        f"persist4={avg_target_persist:.2f}, "
                        f"deg={deg_dist.get('normal',0):.2f}n/{deg_dist.get('repetition',0):.2f}r/{deg_dist.get('garbage',0):.2f}g "
                        f"({time.time()-t_cond:.1f}s, ETA {eta/60:.1f}min)"
                    )

                    # Save sample suffixes
                    for rec in all_records[:args.samples_per_row]:
                        saved_samples.append({**rec, "combo": combo_name, "route": route_name})

                    # Compact row
                    base = audit[combo_name]["rows"][route_name].get("baseline", agg)
                    rm_ref = audit[combo_name]["rows"][route_name].get("one_shot_random", base)
                    compact.append({
                        "combo": combo_name,
                        "route": route_name,
                        "condition": condition,
                        "clean_non_object_rate": agg["clean_non_object_rate"],
                        "object_echo_rate": agg["object_echo_rate"],
                        "any_label_violation_rate": agg["any_label_violation_rate"],
                        "clean_non_object_score": agg["clean_non_object_score"],
                        "mean_relaxation_length": float(avg_relax),
                        "target_persistence_first4": float(avg_target_persist),
                        "degeneration_normal": deg_dist.get("normal", 0),
                        "degeneration_repetition": deg_dist.get("repetition", 0),
                        "degeneration_garbage": deg_dist.get("garbage", 0),
                        "degeneration_syntax_collapse": deg_dist.get("syntax_collapse", 0),
                        "degeneration_short": deg_dist.get("short", 0),
                        "delta_vs_baseline": agg["clean_non_object_rate"] - base["clean_non_object_rate"],
                        "semantic_specificity": agg["clean_non_object_rate"] - rm_ref["clean_non_object_rate"],
                    })

                    # Checkpoint
                    checkpoint = {
                        "phase": 562,
                        "model": args.model,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "done_units": done_units,
                        "total_units": total_units,
                        "audit": audit,
                    }
                    checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "phase": 562,
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
            "remove_scale": args.remove_scale,
            "add_alpha": args.add_alpha,
            "surgery_mode": "one_shot_step0_then_free_kv_cache",
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
            "object_audit": obj_audit,
            "audit": audit,
            "compact_rows": compact,
            "sample_records": saved_samples[:args.max_saved_samples],
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
    parser.add_argument("--layer-sets", default="all")
    parser.add_argument("--remove-scale", type=float, default=1.0)
    parser.add_argument("--add-alpha", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--samples-per-row", type=int, default=3)
    parser.add_argument("--max-saved-samples", type=int, default=800)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"phase562_{args.model}_trajectory_response.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
