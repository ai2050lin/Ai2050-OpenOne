#!/usr/bin/env python3
"""
Phase 574: Suppressed-Path Taxonomy and Alternative Semantic Path Induction
抑制后路径分类与替代语义路径诱导

Exp1: echo suppression output taxonomy — where does probability mass go?
Exp2: object-target-generic three-way competition at step0
Exp3: clean prefix induction — can forced clean prefix bypass echo path?
Exp4: combined intervention — dim2319_w1 + echo_suppress + clean_prefix
Exp5: forbidden_definition semantic authenticity audit

Run:
  python tests/glm5/phase574_suppressed_path_induction.py qwen3 --smoke
  python tests/glm5/phase574_suppressed_path_induction.py qwen3
  python tests/glm5/phase574_suppressed_path_induction.py glm4
  python tests/glm5/phase574_suppressed_path_induction.py deepseek7b
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
import phase573_format_gate_echo_causality as p573  # noqa: E402


OUT_ROOT = Path("results/glm5_phase574_suppressed_path")
DEFAULT_ROUTES = ["forbidden_sentence_completion:temperature<-forbidden_definition"]

ECHO_SUPPRESS_LAMBDAS = [0.0, 4.0, 8.0]

# Generic and format token words for competition analysis
GENERIC_WORDS = ["thing", "item", "object", "something", "entity", "concept",
                 "kind", "type", "sort", "category", "class", "group", "part", "piece"]
FORMAT_WORDS = ["a", "an", "the", "is", "are", "was", "were", " ", ".", ",", "(", ")",
                "used", "for", "to", "of", "in", "on", "with"]

# Pairs for testing
TEST_PAIRS = ["vehicle_tool", "clothing_tool", "furniture_tool", "animal_tool"]

# Clean prefix lengths to test
PREFIX_LENS = [1, 2, 3]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


# ============================================================================
# Exp1: Suppressed-path taxonomy — classify outputs at each lambda
# ============================================================================

def classify_output_detailed(suffix: str, obj: str, pos_label: str, neg_label: str) -> dict[str, Any]:
    """Classify output into detailed failure/success categories."""
    low = suffix.lower().strip()
    obj_low = obj.lower()
    target_terms = p548.all_family_terms(pos_label)
    competitor_terms = p548.all_family_terms(neg_label)
    exact_target = p548.exact_terms(pos_label)
    exact_competitor = p548.exact_terms(neg_label)

    has_obj = obj_low in low
    has_target_syn = any(t in low for t in target_terms if t != obj_low)
    has_competitor = any(t in low for t in competitor_terms)
    has_target_label = any(t in low for t in exact_target)
    has_competitor_label = any(t in low for t in exact_competitor)
    has_generic = any(w in low for w in GENERIC_WORDS)
    has_format = any(w in low for w in ["  ", "..", "((", "))"])
    is_short = len(low.strip()) < 3
    has_function = any(w in low for w in ["used", "for", "tool", "function", "purpose", "designed", "made"])

    # Classify
    if has_target_label or has_competitor_label:
        category = "label_violation"
    elif has_obj and not has_target_syn:
        category = "object_echo"
    elif has_target_syn and not has_obj:
        if has_function:
            category = "clean_synonym_descriptive"
        else:
            category = "clean_synonym"
    elif has_target_syn and has_obj:
        category = "mixed_clean"
    elif has_generic and not has_obj:
        category = "generic_output"
    elif is_short:
        category = "short_fragment"
    elif has_format:
        category = "format_failure"
    elif has_competitor and not has_target_syn:
        category = "wrong_category"
    else:
        category = "other"

    return {
        "category": category,
        "has_object_echo": has_obj and not has_target_syn,
        "has_target_synonym": has_target_syn,
        "has_competitor": has_competitor,
        "has_generic": has_generic,
        "is_short": is_short,
        "suffix": suffix,
    }


def suppressed_path_taxonomy(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pairs: list[str],
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """For each pair+scaffold+lambda, classify all outputs."""
    results: dict[str, Any] = {}
    scaffolds = ["forbidden_sentence_completion", "forbidden_definition"]

    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        objects = CATEGORY_BANK[pos_label][-test_n:]

        # Object token ids
        object_token_ids = set()
        for obj in objects:
            object_token_ids.update(tokenizer.encode(" " + obj, add_special_tokens=False))
            object_token_ids.update(tokenizer.encode(obj, add_special_tokens=False))
        object_token_ids = sorted(object_token_ids)

        pair_results: dict[str, Any] = {}

        for scaffold in scaffolds:
            if scaffold.startswith("forbidden_"):
                prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
            else:
                prompts = [p573.scaffold_prompt_simple(scaffold, obj) for obj in objects]

            scaffold_results: dict[str, Any] = {}

            for lam in ECHO_SUPPRESS_LAMBDAS:
                all_categories = Counter()
                all_suffixes = []

                for seed in seeds:
                    generated = p573.generate_with_echo_suppression(
                        model, tokenizer, device, prompts, groups, object_token_ids,
                        lam, "temperature", seed, max_new_tokens, temperature, top_p, max_length,
                    )
                    for i, obj in enumerate(objects):
                        suffix = generated["generated_suffix"][i]
                        cls = classify_output_detailed(suffix, obj, pos_label, neg_label)
                        all_categories[cls["category"]] += 1
                        all_suffixes.append(suffix)

                total = sum(all_categories.values())
                rates = {k: v / total for k, v in all_categories.items()}
                scaffold_results[f"lambda_{lam}"] = {
                    "total": total,
                    "category_counts": dict(all_categories),
                    "category_rates": rates,
                    "sample_suffixes": all_suffixes[:8],
                }
                log(f"  {pair}/{scaffold} λ={lam}: {dict(all_categories)}")

            pair_results[scaffold] = scaffold_results

        results[pair] = pair_results

    return results


# ============================================================================
# Exp2: Three-way competition at step0
# ============================================================================

def three_way_competition(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pairs: list[str],
    scaffolds: list[str],
    test_n: int,
    max_length: int,
) -> dict[str, Any]:
    """Compute object vs target vs generic vs format competition at step0."""
    results: dict[str, Any] = {}

    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        objects = CATEGORY_BANK[pos_label][-test_n:]

        # Token sets
        def get_token_ids(words):
            ids = set()
            for w in words:
                ids.update(tokenizer.encode(" " + w, add_special_tokens=False))
                ids.update(tokenizer.encode(w, add_special_tokens=False))
            return sorted(ids)

        object_ids = get_token_ids(objects)
        target_ids = [i for i in groups["target"] if 0 <= i < 151552]
        generic_ids = get_token_ids(GENERIC_WORDS)
        format_ids = get_token_ids(FORMAT_WORDS)

        pair_results: dict[str, Any] = {}

        for scaffold in scaffolds:
            if scaffold.startswith("forbidden_"):
                prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
            else:
                prompts = [p573.scaffold_prompt_simple(scaffold, obj) for obj in objects]

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

            def best_logit(ids):
                valid = [i for i in ids if 0 <= i < logits.shape[1]]
                if not valid:
                    return float("-inf")
                return float(np.max(logits[:, valid], axis=1).mean())

            def mass(ids):
                valid = [i for i in ids if 0 <= i < logits.shape[1]]
                if not valid:
                    return 0.0
                m = np.max(logits, axis=1, keepdims=True)
                exp = np.exp(logits - m)
                p = exp / exp.sum(axis=1, keepdims=True)
                return float(np.mean(np.sum(p[:, valid], axis=1)))

            obj_logit = best_logit(object_ids)
            tgt_logit = best_logit(target_ids)
            gen_logit = best_logit(generic_ids)
            fmt_logit = best_logit(format_ids)

            pair_results[scaffold] = {
                "object_best_logit": obj_logit,
                "target_best_logit": tgt_logit,
                "generic_best_logit": gen_logit,
                "format_best_logit": fmt_logit,
                "object_target_margin": obj_logit - tgt_logit,
                "generic_target_margin": gen_logit - tgt_logit,
                "format_target_margin": fmt_logit - tgt_logit,
                "object_mass": mass(object_ids),
                "target_mass": mass(target_ids),
                "generic_mass": mass(generic_ids),
                "format_mass": mass(format_ids),
            }
            log(f"  {pair}/{scaffold}: obj-tgt={obj_logit-tgt_logit:+.2f} "
                f"gen-tgt={gen_logit-tgt_logit:+.2f} fmt-tgt={fmt_logit-tgt_logit:+.2f} "
                f"obj_mass={mass(object_ids):.2e} tgt_mass={mass(target_ids):.2e} "
                f"gen_mass={mass(generic_ids):.2e}")

        results[pair] = pair_results

    return results


# ============================================================================
# Exp3: Clean prefix induction
# ============================================================================

def find_clean_prefixes(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pair: str,
    scaffold: str,
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, list[list[int]]]:
    """Generate baseline outputs and extract clean prefixes of various lengths."""
    pos_label, neg_label = PAIR_SPECS[pair]
    groups = p544.token_groups(tokenizer, pair)
    objects = CATEGORY_BANK[pos_label][-test_n:]

    if scaffold.startswith("forbidden_"):
        prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]
    else:
        prompts = [p573.scaffold_prompt_simple(scaffold, obj) for obj in objects]

    # Generate without suppression
    clean_prefixes: dict[str, list[list[int]]] = {f"prefix{i+1}": [] for i in range(max(PREFIX_LENS))}

    for seed in seeds:
        generated = p573.generate_with_echo_suppression(
            model, tokenizer, device, prompts, groups, [],
            0.0, "temperature", seed, max_new_tokens, temperature, top_p, max_length,
        )
        for i, obj in enumerate(objects):
            suffix = generated["generated_suffix"][i]
            cls = classify_output_detailed(suffix, obj, pos_label, neg_label)
            if "clean" in cls["category"]:
                gen_ids = generated["generated_ids"][i]
                for plen in PREFIX_LENS:
                    if len(gen_ids) >= plen:
                        clean_prefixes[f"prefix{plen}"].append(gen_ids[:plen])

    return clean_prefixes


def generate_with_forced_prefix(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    groups: dict[str, list[int]],
    forced_prefix_ids: list[int] | None,
    mode: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_length: int,
    suppress_token_ids: list[int] = None,
    suppress_lambda: float = 0.0,
) -> dict[str, Any]:
    """Generate with optional forced prefix and echo suppression."""
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

    if suppress_lambda > 0 and suppress_token_ids:
        for tid in suppress_token_ids:
            if 0 <= tid < logits0.shape[1]:
                logits0[:, tid] -= suppress_lambda

    if forced_prefix_ids and len(forced_prefix_ids) >= 1:
        toks = [int(forced_prefix_ids[0])] * batch_size
    else:
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

        if suppress_lambda > 0 and suppress_token_ids:
            for tid in suppress_token_ids:
                if 0 <= tid < logits.shape[1]:
                    logits[:, tid] -= suppress_lambda

        if forced_prefix_ids and step < len(forced_prefix_ids):
            toks = [int(forced_prefix_ids[step])] * batch_size
        else:
            toks = [p544.choose_token(row, mode, rng, temperature, top_p) for row in logits]
        for i, tok in enumerate(toks):
            generated[i].append(int(tok))

    suffixes = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated]
    del past_kv, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"generated_ids": generated, "generated_suffix": suffixes}


def clean_prefix_induction(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    pairs: list[str],
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Test if forced clean prefixes improve clean rate."""
    results: dict[str, Any] = {}

    for pair in pairs:
        pos_label, neg_label = PAIR_SPECS[pair]
        groups = p544.token_groups(tokenizer, pair)
        objects = CATEGORY_BANK[pos_label][-test_n:]

        # Object token ids for suppression
        object_token_ids = set()
        for obj in objects:
            object_token_ids.update(tokenizer.encode(" " + obj, add_special_tokens=False))
            object_token_ids.update(tokenizer.encode(obj, add_special_tokens=False))
        object_token_ids = sorted(object_token_ids)

        scaffold = "forbidden_sentence_completion"
        prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]

        # Find clean prefixes from baseline generation
        clean_prefixes = find_clean_prefixes(
            model, tokenizer, device, pair, scaffold, test_n, seeds,
            max_length, max_new_tokens, temperature, top_p,
        )
        log(f"  {pair} clean prefixes found: "
            + ", ".join(f"{k}={len(v)}" for k, v in clean_prefixes.items()))

        pair_results: dict[str, Any] = {}

        # Test conditions
        conditions = [
            ("baseline", None, 0.0),
            ("echo_suppress8", None, 8.0),
        ]

        # Add prefix conditions if we have clean prefixes
        for plen in PREFIX_LENS:
            key = f"prefix{plen}"
            if clean_prefixes[key]:
                # Use the first available clean prefix
                conditions.append((f"clean_prefix{plen}", clean_prefixes[key][0], 0.0))
                conditions.append((f"prefix{plen}+suppress8", clean_prefixes[key][0], 8.0))

        for cond_name, forced_prefix, suppress_lam in conditions:
            all_clean = []
            all_echo = []
            all_categories = Counter()

            for seed in seeds:
                generated = generate_with_forced_prefix(
                    model, tokenizer, device, prompts, groups,
                    forced_prefix, "temperature", seed,
                    max_new_tokens, temperature, top_p, max_length,
                    object_token_ids, suppress_lam,
                )
                for i, obj in enumerate(objects):
                    suffix = generated["generated_suffix"][i]
                    cls = classify_output_detailed(suffix, obj, pos_label, neg_label)
                    all_categories[cls["category"]] += 1
                    all_clean.append("clean" in cls["category"])
                    all_echo.append(cls["has_object_echo"])

            clean_rate = float(np.mean(all_clean))
            echo_rate = float(np.mean(all_echo))
            pair_results[cond_name] = {
                "clean_rate": clean_rate,
                "echo_rate": echo_rate,
                "category_counts": dict(all_categories),
                "forced_prefix": forced_prefix,
            }
            log(f"  {pair}/{cond_name}: clean={clean_rate:.2f} echo={echo_rate:.2f} "
                f"cats={dict(all_categories)}")

        results[pair] = pair_results

    return results


# ============================================================================
# Exp4: Combined intervention (dim manipulation + echo suppress + clean prefix)
# ============================================================================

def combined_intervention(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layers: list[Any],
    window: list[int],
    W_U: np.ndarray,
    eps: float,
    pair: str,
    test_n: int,
    seeds: list[int],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Test combined: dim2319_w1 + echo_suppress + clean_prefix."""
    pos_label, neg_label = PAIR_SPECS[pair]
    groups = p544.token_groups(tokenizer, pair)
    objects = CATEGORY_BANK[pos_label][-test_n:]

    object_token_ids = set()
    for obj in objects:
        object_token_ids.update(tokenizer.encode(" " + obj, add_special_tokens=False))
        object_token_ids.update(tokenizer.encode(obj, add_special_tokens=False))
    object_token_ids = sorted(object_token_ids)

    scaffold = "forbidden_sentence_completion"
    prompts = [p548.forbidden_prompt(scaffold, obj, pos_label, neg_label) for obj in objects]

    # Find clean prefix
    clean_prefixes = find_clean_prefixes(
        model, tokenizer, device, pair, scaffold, test_n, seeds,
        max_length, max_new_tokens, temperature, top_p,
    )
    prefix2 = clean_prefixes.get("prefix2", [None])[0] if clean_prefixes.get("prefix2") else None

    # Get top neg gate dim for this model/pair
    cap = p571.capture_hidden_states(
        model, tokenizer, device, layers, prompts, window, "baseline", None, [], max_length
    )
    h_raw = cap["captured"]["h_raw"].numpy().astype(np.float64)
    h_norm = cap["captured"]["h_norm"].numpy().astype(np.float64)
    variance = np.mean(h_raw ** 2, axis=-1, keepdims=True)
    h_rms = h_raw / np.sqrt(variance + eps)
    h_rms_safe = np.where(np.abs(h_rms) < 1e-10, 1.0, h_rms)
    w_norm_orig = np.nan_to_num((h_norm / h_rms_safe).mean(0), nan=1.0, posinf=1.0, neginf=1.0)

    tgt_ids = [i for i in groups["target"] if 0 <= i < W_U.shape[0]]
    cmp_ids = [i for i in groups["competitor"] if 0 <= i < W_U.shape[0]]
    d_TC = W_U[tgt_ids].mean(0).astype(np.float64) - W_U[cmp_ids].mean(0).astype(np.float64)
    rms_contrib = h_rms.mean(0) * d_TC
    norm_contrib = (w_norm_orig * h_rms).mean(0) * d_TC
    delta = norm_contrib - rms_contrib
    top_neg_dim = int(np.argmin(delta))

    log(f"  {pair} top neg gate dim: {top_neg_dim} (delta={delta[top_neg_dim]:.4f})")

    # Conditions: (name, dim_override, forced_prefix, suppress_lambda)
    conditions = [
        ("baseline", {}, None, 0.0),
        ("dim_w1", {top_neg_dim: 1.0}, None, 0.0),
        ("suppress8", {}, None, 8.0),
        ("dim_w1+suppress8", {top_neg_dim: 1.0}, None, 8.0),
    ]
    if prefix2:
        conditions.append(("prefix2", {}, prefix2, 0.0))
        conditions.append(("prefix2+suppress8", {}, prefix2, 8.0))
        conditions.append(("dim_w1+prefix2+suppress8", {top_neg_dim: 1.0}, prefix2, 8.0))

    results: dict[str, Any] = {}
    final_norm = model.model.norm

    for cond_name, dim_overrides, forced_prefix, suppress_lam in conditions:
        # Apply dim override to norm weight
        orig_weight = final_norm.weight.data.clone()
        if dim_overrides:
            new_weight = orig_weight.clone()
            for d, val in dim_overrides.items():
                if 0 <= d < new_weight.shape[0]:
                    new_weight[d] = val
            final_norm.weight.data.copy_(new_weight)

        try:
            all_clean = []
            all_echo = []
            all_categories = Counter()

            for seed in seeds:
                generated = generate_with_forced_prefix(
                    model, tokenizer, device, prompts, groups,
                    forced_prefix, "temperature", seed,
                    max_new_tokens, temperature, top_p, max_length,
                    object_token_ids, suppress_lam,
                )
                for i, obj in enumerate(objects):
                    suffix = generated["generated_suffix"][i]
                    cls = classify_output_detailed(suffix, obj, pos_label, neg_label)
                    all_categories[cls["category"]] += 1
                    all_clean.append("clean" in cls["category"])
                    all_echo.append(cls["has_object_echo"])

            clean_rate = float(np.mean(all_clean))
            echo_rate = float(np.mean(all_echo))
            results[cond_name] = {
                "clean_rate": clean_rate,
                "echo_rate": echo_rate,
                "category_counts": dict(all_categories),
            }
            log(f"  {pair}/{cond_name}: clean={clean_rate:.2f} echo={echo_rate:.2f} "
                f"cats={dict(all_categories)}")
        finally:
            final_norm.weight.data.copy_(orig_weight)

    return results


# ============================================================================
# Main
# ============================================================================

def run_model(args: argparse.Namespace) -> dict[str, Any]:
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

        log(f"{args.model}: phase574 window={window}, eps={eps}")

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"phase574_{args.model}_checkpoint.json"

        result: dict[str, Any] = {
            "phase": 574, "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "attn_implementation": attn_impl,
            "window": window, "eps": eps,
            "model_info": {"n_layers": info.n_layers, "d_model": info.d_model, "class": info.model_class},
        }

        # === Exp1: Suppressed-path taxonomy ===
        log("=== Exp1: Suppressed-path taxonomy ===")
        exp1 = suppressed_path_taxonomy(
            model, tokenizer, device,
            TEST_PAIRS, args.test_n, seeds,
            args.max_length, args.max_new_tokens, args.temperature, args.top_p,
        )
        result["exp1_taxonomy"] = exp1
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp2: Three-way competition ===
        log("=== Exp2: Three-way competition ===")
        exp2 = three_way_competition(
            model, tokenizer, device,
            TEST_PAIRS, ["forbidden_sentence_completion", "forbidden_definition"],
            args.test_n, args.max_length,
        )
        result["exp2_competition"] = exp2
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp3: Clean prefix induction ===
        log("=== Exp3: Clean prefix induction ===")
        exp3 = clean_prefix_induction(
            model, tokenizer, device,
            TEST_PAIRS, args.test_n, seeds,
            args.max_length, args.max_new_tokens, args.temperature, args.top_p,
        )
        result["exp3_prefix_induction"] = exp3
        checkpoint_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                                   encoding="utf-8")

        # === Exp4: Combined intervention ===
        log("=== Exp4: Combined intervention ===")
        exp4 = combined_intervention(
            model, tokenizer, device, layers, window, W_U, eps,
            "vehicle_tool", args.test_n, seeds,
            args.max_length, args.max_new_tokens, args.temperature, args.top_p,
        )
        result["exp4_combined"] = exp4
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
    parser.add_argument("--test-n", type=int, default=24)
    parser.add_argument("--sample-seeds", default="101,103,107,109,113,127")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.test_n = 4
        args.sample_seeds = "101,103"
        args.max_new_tokens = 8
        log("SMOKE TEST MODE: test_n=4, seeds=2, max_tokens=8")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase574_{args.model}_suppressed_path{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
