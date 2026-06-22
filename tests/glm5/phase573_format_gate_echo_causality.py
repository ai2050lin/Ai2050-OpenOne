#!/usr/bin/env python3
"""
Phase 573: Format-Gate Dimension and Echo-Path Causality Audit
格式门控维度与回声路径因果审计

Exp1: dim=2319 cross-condition audit — is it vehicle_tool-specific or global?
Exp2: dim=2319 causal manipulation — w=0, w=1, w=10, h=0, h=donor
Exp3: object_echo token competition — object vs target rank/mass/margin
Exp4: echo suppression — suppress object tokens, does clean improve?
Exp5: clean output taxonomy — what does forbidden_definition actually generate?

Run:
  python tests/glm5/phase573_format_gate_echo_causality.py qwen3 --smoke
  python tests/glm5/phase573_format_gate_echo_causality.py qwen3
  python tests/glm5/phase573_format_gate_echo_causality.py glm4
  python tests/glm5/phase573_format_gate_echo_causality.py deepseek7b
"""
from __future__ import annotations

import argparse
import gc
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
import phase568_locked_prefix_swap as p568  # noqa: E402
import phase569_pre_layer_source_tracing as p569  # noqa: E402
import phase571_norm_weight_dimension_gate as p571  # noqa: E402


OUT_ROOT = Path("results/glm5_phase573_format_gate_echo")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]

# Top negative gate dims from Phase 572 (GLM4 vehicle_tool)
# These will be re-computed per model at runtime
GATE_DIMS_TO_AUDIT_COUNT = 5  # top-5 negative gate dims

# Echo suppression lambdas
ECHO_SUPPRESS_LAMBDAS = [1.0, 2.0, 4.0, 8.0]

# Scaffolds for echo and taxonomy analysis
ECHO_SCAFFOLDS = ["forbidden_sentence_completion", "forbidden_definition", "sentence_completion", "definition"]

# Pairs for testing
TEST_PAIRS = ["vehicle_tool", "clothing_tool", "furniture_tool", "animal_tool"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Exp1: Cross-condition audit of top gate dims
# ============================================================================

def cross_condition_gate_audit(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    W_U: np.ndarray,
    eps: float,
    pairs: list[str],
    test_n: int,
    max_length: int,
) -> dict[str, Any]:
    """For each pair, compute top-5 negative gate dims and their properties.
    Then check overlap and cross-condition stability.
    """
    results: dict[str, Any] = {"per_pair": {}, "cross_pair_overlap": {}}
    all_neg_dims: dict[str, list[int]] = {}

    for pair in pairs:
        try:
            pos_label, neg_label = PAIR_SPECS[pair]
            groups = p544.token_groups(tokenizer, pair)
            prompt_rows = p548.build_prompts(pair, test_n, ["forbidden_sentence_completion"])
            prompts = [r["prompt"] for r in prompt_rows["forbidden_sentence_completion"]]

            # Capture baseline hidden states
            cap = p571.capture_hidden_states(
                model, tokenizer, device, layers, prompts, window, "baseline", None, [], max_length
            )
            h_raw = cap["captured"]["h_raw"].numpy().astype(np.float64)
            h_norm = cap["captured"]["h_norm"].numpy().astype(np.float64)

            # Compute w_norm, d_TC, delta
            variance = np.mean(h_raw ** 2, axis=-1, keepdims=True)
            h_rms = h_raw / np.sqrt(variance + eps)
            h_rms_safe = np.where(np.abs(h_rms) < 1e-10, 1.0, h_rms)
            w_norm = np.nan_to_num((h_norm / h_rms_safe).mean(0), nan=1.0, posinf=1.0, neginf=1.0)

            tgt_ids = [i for i in groups["target"] if 0 <= i < W_U.shape[0]]
            cmp_ids = [i for i in groups["competitor"] if 0 <= i < W_U.shape[0]]
            d_TC = W_U[tgt_ids].mean(0).astype(np.float64) - W_U[cmp_ids].mean(0).astype(np.float64)

            rms_contrib = h_rms.mean(0) * d_TC
            norm_contrib = (w_norm * h_rms).mean(0) * d_TC
            delta = norm_contrib - rms_contrib

            neg_dims = np.argsort(delta)[:GATE_DIMS_TO_AUDIT_COUNT].tolist()
            pos_dims = np.argsort(delta)[::-1][:GATE_DIMS_TO_AUDIT_COUNT].tolist()

            # For each neg dim, get identity
            dim_info = []
            for d in neg_dims:
                col = W_U[:, d]
                top_idx = np.argsort(col)[::-1][:5]
                top_tok = [tokenizer.decode([int(i)]) for i in top_idx]
                dim_info.append({
                    "dim": int(d),
                    "w_norm": float(w_norm[d]),
                    "d_TC": float(d_TC[d]),
                    "delta": float(delta[d]),
                    "top_tokens": top_tok,
                })

            results["per_pair"][pair] = {
                "neg_dims": neg_dims,
                "pos_dims": pos_dims,
                "dim_info": dim_info,
            }
            all_neg_dims[pair] = neg_dims
            log(f"  {pair} neg gate dims: {neg_dims}")
            for di in dim_info:
                log(f"    dim={di['dim']} w={di['w_norm']:.3f} delta={di['delta']:.4f} "
                    f"tokens={di['top_tokens'][:3]}")
        except Exception as e:
            log(f"  {pair} failed: {e}")
            import traceback; traceback.print_exc()

    # Cross-pair overlap
    import itertools
    for a, b in itertools.combinations(all_neg_dims.keys(), 2):
        sa, sb = set(all_neg_dims[a]), set(all_neg_dims[b])
        union = sa | sb
        jac = len(sa & sb) / len(union) if union else 0.0
        results["cross_pair_overlap"][f"{a}_vs_{b}"] = {
            "jaccard": round(jac, 4),
            "intersection": sorted(sa & sb),
            "union_size": len(union),
        }

    return results


# ============================================================================
# Exp2: dim causal manipulation
# ============================================================================

def dim_causal_manipulation(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    W_U: np.ndarray,
    eps: float,
    pair: str,
    test_n: int,
    target_dims: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logit_steps: int,
    seeds: list[int],
) -> dict[str, Any]:
    """Manipulate specific gate dims during generation and measure effect.

    We implement this by hooking the final norm to modify w_norm for target dims.
    """
    pos_label, neg_label = PAIR_SPECS[pair]
    groups = p544.token_groups(tokenizer, pair)
    prompt_rows = p548.build_prompts(pair, test_n, ["forbidden_sentence_completion"])
    prompts = [r["prompt"] for r in prompt_rows["forbidden_sentence_completion"]]

    # First capture baseline to get original w_norm
    cap = p571.capture_hidden_states(
        model, tokenizer, device, layers, prompts, window, "baseline", None, [], max_length
    )
    h_raw_base = cap["captured"]["h_raw"].numpy().astype(np.float64)
    h_norm_base = cap["captured"]["h_norm"].numpy().astype(np.float64)
    variance = np.mean(h_raw_base ** 2, axis=-1, keepdims=True)
    h_rms_base = h_raw_base / np.sqrt(variance + eps)
    h_rms_safe = np.where(np.abs(h_rms_base) < 1e-10, 1.0, h_rms_base)
    w_norm_orig = np.nan_to_num((h_norm_base / h_rms_safe).mean(0), nan=1.0, posinf=1.0, neginf=1.0)

    # Define manipulation conditions
    manipulations = {
        "baseline": {},  # no change
    }
    for d in target_dims:
        manipulations[f"dim{d}_w0"] = {d: 0.0}
        manipulations[f"dim{d}_w1"] = {d: 1.0}
        manipulations[f"dim{d}_w10"] = {d: 10.0}

    results: dict[str, Any] = {}

    for cond_name, dim_overrides in manipulations.items():
        # Create modified w_norm
        w_mod = w_norm_orig.copy()
        for d, val in dim_overrides.items():
            w_mod[d] = val

        # Compute step0 margin with modified w_norm
        h_rms_f64 = h_rms_base
        h_mod = w_mod * h_rms_f64
        z_mod = h_mod @ W_U.T.astype(np.float64)
        batch_size = z_mod.shape[0]
        stats = [p569.detailed_logit_stats(z_mod[i], groups) for i in range(batch_size)]
        m_margin = float(np.mean([s["target_minus_competitor"] for s in stats]))
        m_rank = float(np.mean([s["target_best_rank"] for s in stats]))

        # Generation with modified norm (hook the final norm)
        # We need to override the norm weight during forward pass
        all_clean = []
        all_echo = []
        all_suffixes = []

        for seed in seeds:
            generated = generate_with_norm_override(
                model, tokenizer, device, layers, prompts, groups,
                w_norm_orig, dim_overrides, eps,
                "temperature", seed, max_new_tokens, temperature, top_p, max_length,
                logit_steps,
            )
            for i, row in enumerate(prompt_rows["forbidden_sentence_completion"]):
                suffix = generated["generated_suffix"][i]
                classified = p548.classify_suffix(suffix, row["object"], pos_label, neg_label)
                all_clean.append(classified["clean_non_object"])
                all_echo.append(classified["object_echo"])
                all_suffixes.append(suffix)

        clean_rate = float(np.mean(all_clean))
        echo_rate = float(np.mean(all_echo))
        results[cond_name] = {
            "step0_margin": m_margin,
            "step0_rank": m_rank,
            "clean_rate": clean_rate,
            "echo_rate": echo_rate,
            "sample_suffixes": all_suffixes[:5],
        }
        log(f"  {cond_name}: margin={m_margin:+.2f} rank={m_rank:.0f} "
            f"clean={clean_rate:.2f} echo={echo_rate:.2f}")

    return results


def generate_with_norm_override(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    prompts: list[str],
    groups: dict[str, list[int]],
    w_norm_orig: np.ndarray,
    dim_overrides: dict[int, float],
    eps: float,
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    logit_steps: int,
) -> dict[str, Any]:
    """Generate with final norm weight overridden for specific dims.

    We temporarily replace the norm weight, generate, then restore.
    """
    final_norm = model.model.norm
    # Store original weight
    orig_weight = final_norm.weight.data.clone()

    # Apply overrides
    if dim_overrides:
        new_weight = orig_weight.clone()
        for d, val in dim_overrides.items():
            if 0 <= d < new_weight.shape[0]:
                new_weight[d] = val
        final_norm.weight.data.copy_(new_weight)

    try:
        # Generate normally
        rng = np.random.default_rng(seed)
        batch_size = len(prompts)
        old_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        answer_pos = input_ids.shape[1] - 1
        tokenizer.padding_side = old_padding

        generated: list[list[int]] = [[] for _ in prompts]
        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()

        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))

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

        suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
        del past_kv, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"generated_ids": generated, "generated_suffix": suffixes}
    finally:
        # Restore original weight
        final_norm.weight.data.copy_(orig_weight)


# ============================================================================
# Exp3: object_echo token competition audit
# ============================================================================

def object_echo_competition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    W_U: np.ndarray,
    eps: float,
    pairs: list[str],
    scaffolds: list[str],
    test_n: int,
    max_length: int,
) -> dict[str, Any]:
    """For each pair+scaffold, compute object vs target competition at step0."""
    results: dict[str, Any] = {}

    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        objects = CATEGORY_BANK[pos_label][-test_n:]

        # Get object token ids
        object_token_ids = []
        for obj in objects:
            ids = tokenizer.encode(" " + obj, add_special_tokens=False)
            object_token_ids.extend(ids)
            ids2 = tokenizer.encode(obj, add_special_tokens=False)
            object_token_ids.extend(ids2)
        object_token_ids = list(set(object_token_ids))

        pair_results: dict[str, Any] = {}

        for scaffold in scaffolds:
            if scaffold.startswith("forbidden_"):
                prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
            else:
                prompts = [scaffold_prompt_simple(scaffold, obj) for obj in objects]

            # Capture step0 logits
            old_padding = tokenizer.padding_side
            tokenizer.padding_side = "left"
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            answer_pos = input_ids.shape[1] - 1
            tokenizer.padding_side = old_padding

            with torch.inference_mode():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True)
                logits = out.logits[:, answer_pos, :].float().cpu().numpy()

            batch_size = logits.shape[0]

            # Compute object vs target metrics
            tgt_ids = [i for i in groups["target"] if 0 <= i < logits.shape[1]]
            obj_ids = [i for i in object_token_ids if 0 <= i < logits.shape[1]]

            def best_logit(ids):
                if not ids:
                    return float("-inf")
                return float(np.max(logits[:, ids], axis=1).mean())

            def best_rank(ids):
                if not ids:
                    return float(logits.shape[1])
                vals = np.max(logits[:, ids], axis=1)
                return float(np.mean([1 + np.sum(logits[j] > vals[j]) for j in range(batch_size)]))

            def mass(ids):
                if not ids:
                    return 0.0
                m = np.max(logits, axis=1, keepdims=True)
                exp = np.exp(logits - m)
                p = exp / exp.sum(axis=1, keepdims=True)
                return float(np.mean(np.sum(p[:, ids], axis=1)))

            tgt_logit = best_logit(tgt_ids)
            obj_logit = best_logit(obj_ids)
            tgt_rank = best_rank(tgt_ids)
            obj_rank = best_rank(obj_ids)
            tgt_mass = mass(tgt_ids)
            obj_mass = mass(obj_ids)

            # Object-target margin
            obj_tgt_margin = obj_logit - tgt_logit

            pair_results[scaffold] = {
                "target_best_logit": tgt_logit,
                "object_best_logit": obj_logit,
                "object_target_margin": obj_tgt_margin,
                "target_rank": tgt_rank,
                "object_rank": obj_rank,
                "target_mass": tgt_mass,
                "object_mass": obj_mass,
            }
            log(f"  {pair}/{scaffold}: obj_tgt_margin={obj_tgt_margin:+.2f} "
                f"tgt_rank={tgt_rank:.0f} obj_rank={obj_rank:.0f} "
                f"tgt_mass={tgt_mass:.2e} obj_mass={obj_mass:.2e}")

        results[pair] = pair_results

    return results


def scaffold_prompt_simple(scaffold: str, obj: str) -> str:
    if scaffold == "sentence_completion":
        return f"A {obj} is commonly used as"
    if scaffold == "definition":
        return f"A {obj} is best defined as"
    if scaffold == "natural_qa":
        return f"What kind of thing is a {obj}? A {obj} is"
    raise ValueError(f"unknown scaffold: {scaffold}")


# ============================================================================
# Exp4: Echo suppression during generation
# ============================================================================

def echo_suppression_experiment(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    pairs: list[str],
    scaffolds: list[str],
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    logit_steps: int,
) -> dict[str, Any]:
    """Suppress object tokens during generation and measure clean rate change."""
    results: dict[str, Any] = {}

    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        objects = CATEGORY_BANK[pos_label][-test_n:]

        # Get object token ids
        object_token_ids = set()
        for obj in objects:
            ids = tokenizer.encode(" " + obj, add_special_tokens=False)
            object_token_ids.update(ids)
            ids2 = tokenizer.encode(obj, add_special_tokens=False)
            object_token_ids.update(ids2)
        object_token_ids = sorted(object_token_ids)

        pair_results: dict[str, Any] = {}

        for scaffold in scaffolds:
            if scaffold.startswith("forbidden_"):
                prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
            else:
                prompts = [scaffold_prompt_simple(scaffold, obj) for obj in objects]

            scaffold_results: dict[str, Any] = {}

            # Test: no suppression (lambda=0) and with suppression
            for lam in [0.0] + ECHO_SUPPRESS_LAMBDAS:
                all_clean = []
                all_echo = []
                all_label_violation = []
                all_suffixes = []

                for seed in seeds:
                    generated = generate_with_echo_suppression(
                        model, tokenizer, device, prompts, groups, object_token_ids,
                        lam, "temperature", seed, max_new_tokens, temperature, top_p, max_length,
                    )
                    for i, obj in enumerate(objects):
                        suffix = generated["generated_suffix"][i]
                        classified = p548.classify_suffix(suffix, obj, pos_label, neg_label)
                        all_clean.append(classified["clean_non_object"])
                        all_echo.append(classified["object_echo"])
                        all_label_violation.append(classified["any_label_violation"])
                        all_suffixes.append(suffix)

                clean_rate = float(np.mean(all_clean))
                echo_rate = float(np.mean(all_echo))
                violation_rate = float(np.mean(all_label_violation))
                scaffold_results[f"lambda_{lam}"] = {
                    "clean_rate": clean_rate,
                    "echo_rate": echo_rate,
                    "label_violation_rate": violation_rate,
                    "sample_suffixes": all_suffixes[:5],
                }
                log(f"  {pair}/{scaffold} λ={lam}: clean={clean_rate:.2f} "
                    f"echo={echo_rate:.2f} violation={violation_rate:.2f}")

            pair_results[scaffold] = scaffold_results

        results[pair] = pair_results

    return results


def generate_with_echo_suppression(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    groups: dict[str, list[int]],
    suppress_token_ids: list[int],
    suppress_lambda: float,
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
) -> dict[str, Any]:
    """Generate with object token logits suppressed by lambda."""
    rng = np.random.default_rng(seed)
    batch_size = len(prompts)
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    answer_pos = input_ids.shape[1] - 1
    tokenizer.padding_side = old_padding

    generated: list[list[int]] = [[] for _ in prompts]
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits0 = out.logits[:, answer_pos, :].float().cpu().numpy()

    # Suppress object tokens
    if suppress_lambda > 0:
        for tid in suppress_token_ids:
            if 0 <= tid < logits0.shape[1]:
                logits0[:, tid] -= suppress_lambda

    toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits0]
    for i, tok in enumerate(toks):
        generated[i].append(int(tok))

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

        if suppress_lambda > 0:
            for tid in suppress_token_ids:
                if 0 <= tid < logits.shape[1]:
                    logits[:, tid] -= suppress_lambda

        toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generated_ids": generated, "generated_suffix": suffixes}


# ============================================================================
# Exp5: Clean output taxonomy
# ============================================================================

def clean_output_taxonomy(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    pairs: list[str],
    scaffolds: list[str],
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Classify clean outputs into categories."""
    results: dict[str, Any] = {}

    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        objects = CATEGORY_BANK[pos_label][-test_n:]
        target_terms = p548.all_family_terms(pos_label)
        competitor_terms = p548.all_family_terms(neg_label)

        pair_results: dict[str, Any] = {}

        for scaffold in scaffolds:
            if scaffold.startswith("forbidden_"):
                prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
            else:
                prompts = [scaffold_prompt_simple(scaffold, obj) for obj in objects]

            all_clean_suffixes = []
            all_clean_objects = []
            clean_types = Counter()

            for seed in seeds:
                generated = generate_with_echo_suppression(
                    model, tokenizer, device, prompts, groups, [],
                    0.0, "temperature", seed, max_new_tokens, temperature, top_p, max_length,
                )
                for i, obj in enumerate(objects):
                    suffix = generated["generated_suffix"][i]
                    classified = p548.classify_suffix(suffix, obj, pos_label, neg_label)
                    if classified["clean_non_object"]:
                        all_clean_suffixes.append(suffix)
                        all_clean_objects.append(obj)
                        # Classify clean output type
                        low = suffix.lower()
                        has_target_syn = any(t in low for t in target_terms if t != obj.lower())
                        has_function = any(w in low for w in ["used", "for", "tool", "function", "purpose", "designed"])
                        has_attribute = any(w in low for w in ["has", "with", "made", "material", "shape", "size"])
                        has_generic = any(w in low for w in ["thing", "item", "object", "kind", "type", "sort"])

                        if has_target_syn and not has_function and not has_attribute:
                            clean_types["label_synonym"] += 1
                        elif has_function and not has_target_syn:
                            clean_types["functional_description"] += 1
                        elif has_attribute and not has_target_syn:
                            clean_types["attribute_description"] += 1
                        elif has_generic and not has_target_syn:
                            clean_types["generic_definition"] += 1
                        elif has_target_syn and (has_function or has_attribute):
                            clean_types["mixed_synonym_descriptive"] += 1
                        else:
                            clean_types["other_clean"] += 1

            total_clean = len(all_clean_suffixes)
            pair_results[scaffold] = {
                "total_clean": total_clean,
                "clean_type_counts": dict(clean_types),
                "clean_type_rates": {k: v / max(1, total_clean) for k, v in clean_types.items()},
                "sample_clean_suffixes": all_clean_suffixes[:10],
                "sample_clean_objects": all_clean_objects[:10],
            }
            log(f"  {pair}/{scaffold}: total_clean={total_clean} types={dict(clean_types)}")

        results[pair] = pair_results

    return results


# ============================================================================
# Main
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
    routes = p558.parse_routes(args.routes)
    seeds = parse_int_csv(args.sample_seeds)

    model, tokenizer, device, attn_impl = load_model_bf16_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        windows = layer_windows(args.model, info.n_layers, args.windows)
        _, window = next(iter(windows.items()))
        W_U = get_W_U(model, args.model).astype(np.float32)

        final_norm = model.model.norm
        eps = getattr(final_norm, 'variance_epsilon', None)
        if eps is None:
            eps = getattr(final_norm, 'eps', None)
        if eps is None:
            eps = getattr(model.config, 'rms_norm_eps', 1e-6)

        log(f"{args.model}: phase573 window={window}, eps={eps}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase573_{args.model}_checkpoint.json"

        result: dict[str, Any] = {
            "phase": 573, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "window": window, "eps": eps,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
        }

        # === Exp1: Cross-condition gate audit ===
        log("=== Exp1: Cross-condition gate dim audit ===")
        exp1 = cross_condition_gate_audit(
            model, tokenizer, device, layers, window, W_U, eps,
            TEST_PAIRS, args.test_n, args.max_length,
        )
        result["exp1_gate_audit"] = exp1
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp2: Dim causal manipulation (vehicle_tool only) ===
        log("=== Exp2: Dim causal manipulation (vehicle_tool) ===")
        # Get top-5 negative gate dims for vehicle_tool
        vt_neg_dims = exp1["per_pair"].get("vehicle_tool", {}).get("neg_dims", [])
        if vt_neg_dims:
            exp2 = dim_causal_manipulation(
                model, tokenizer, device, layers, window, W_U, eps,
                "vehicle_tool", args.test_n, vt_neg_dims[:3],  # top 3 only for speed
                args.max_length, args.max_new_tokens, args.temperature, args.top_p,
                args.logit_steps, seeds,
            )
            result["exp2_dim_causal"] = exp2
            checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")

        # === Exp3: Object_echo competition ===
        log("=== Exp3: Object_echo token competition ===")
        exp3 = object_echo_competition(
            model, tokenizer, device, layers, window, W_U, eps,
            TEST_PAIRS, ECHO_SCAFFOLDS, args.test_n, args.max_length,
        )
        result["exp3_echo_competition"] = exp3
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp4: Echo suppression ===
        log("=== Exp4: Echo suppression ===")
        # Only test forbidden_sentence_completion (highest echo) and forbidden_definition (best clean)
        echo_scaffolds = ["forbidden_sentence_completion", "forbidden_definition"]
        exp4 = echo_suppression_experiment(
            model, tokenizer, device, layers, window,
            TEST_PAIRS, echo_scaffolds, args.test_n, seeds,
            args.max_length, args.max_new_tokens, args.temperature, args.top_p, args.logit_steps,
        )
        result["exp4_echo_suppression"] = exp4
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp5: Clean output taxonomy ===
        log("=== Exp5: Clean output taxonomy ===")
        exp5 = clean_output_taxonomy(
            model, tokenizer, device, layers, window,
            TEST_PAIRS, ECHO_SCAFFOLDS, args.test_n, seeds,
            args.max_length, args.max_new_tokens, args.temperature, args.top_p,
        )
        result["exp5_clean_taxonomy"] = exp5
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        return result
    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--windows", default=None)
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--logit-steps", type=int, default=4)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
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
    out_path = out_dir / f"phase573_{args.model}_format_gate_echo{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
