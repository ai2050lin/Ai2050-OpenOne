#!/usr/bin/env python3
"""
Phase 581: Intermediate State Forcing and Retrieval Composition Closure
中间状态强制与检索组合闭包

Phase 580 proved: single-step retrieval works (90-100%), but two-hop composition = 0%.
Phase 581 answers: WHERE does the two-hop break?

Steps:
  Step 0: Relation necessity audit (does model truly use R?)
  Step 1: Two-hop three-segment decomposition (Step A: O→C, Step B: (C,R)→V, Step C: Compose)
  Step 2: Forced intermediate (prompt model to explicitly generate C first)
  Step 3: Gold intermediate (directly provide correct C, test if Step 2 works)
  Step 4: Activation-level intermediate injection (inject h_C from single-step into two-hop)
  Step 5: State-to-query conversion (measure if injected state changes Step 2 attention)
  Step 6: Retrieval chaining patch (connect Step1 cat-copy output to Step2 val-copy input)
  Step 7: Grammar scaffold impact (direct vs step-by-step vs proof-style)

Model loading: BF16 + device_map="auto" + eager attention (no quantization)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U  # noqa: E402

OUT_ROOT = Path("results/glm5_phase581_composition_closure")

CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_RELATIONS = ["r31", "r64"]
CANDIDATE_VALUES = ["v05", "v91", "v22", "v48"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Model loading (same as Phase 580, with flash_attention_2 option)
# ============================================================================

def load_model_eager(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    log(f"Loading {model_name}: bf16 + device_map=auto + eager")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Loaded {model_name}: class={type(model).__name__}, GPU={gpu_mem:.2f}GB, attn=eager")
    return model, tokenizer, device


# ============================================================================
# GQA and weight utilities (reused from Phase 580)
# ============================================================================

def get_head_dims(model):
    config = model.config
    n_heads = config.num_attention_heads
    d_model = config.hidden_size
    if hasattr(config, 'head_dim') and config.head_dim is not None:
        d_head = config.head_dim
    else:
        n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        layer0 = get_layers(model)[0]
        d_head = layer0.self_attn.v_proj.weight.shape[0] // n_kv_heads
    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    kv_group_size = n_heads // n_kv_heads
    return n_heads, d_head, d_model, n_kv_heads, kv_group_size


_safetensors_cache = {}

def _load_attn_weights_from_safetensors(model, layer_idx):
    global _safetensors_cache
    cache_key = (id(model), layer_idx)
    if cache_key in _safetensors_cache:
        return _safetensors_cache[cache_key]

    model_path = None
    if hasattr(model, 'config'):
        name_or_path = getattr(model.config, '_name_or_path', None)
        if name_or_path and os.path.isdir(name_or_path):
            model_path = name_or_path

    if model_path is None:
        from model_registry import MODEL_SPECS
        for name, spec in MODEL_SPECS.items():
            if os.path.isdir(str(spec.local_dir)):
                model_path = str(spec.local_dir)
                break

    if model_path is None:
        raise ValueError("Cannot determine model path for safetensors loading")

    import glob
    from safetensors import safe_open

    sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
    W_V = None
    W_O = None
    for sf_file in sf_files:
        with safe_open(sf_file, framework='pt', device='cpu') as sf:
            v_key = f'model.layers.{layer_idx}.self_attn.v_proj.weight'
            o_key = f'model.layers.{layer_idx}.self_attn.o_proj.weight'
            if v_key in sf.keys():
                W_V = sf.get_tensor(v_key).float().numpy()
            if o_key in sf.keys():
                W_O = sf.get_tensor(o_key).float().numpy()
            if W_V is not None and W_O is not None:
                _safetensors_cache[cache_key] = (W_V, W_O)
                return W_V, W_O

    raise ValueError(f"Cannot find layer {layer_idx} attention weights in safetensors")


# ============================================================================
# Full-string logprob scoring (from Phase 580)
# ============================================================================

def compute_full_string_logprob(model, tokenizer, device, prompt, answer_str):
    """Compute log P(answer_str | prompt) by summing over all tokens."""
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    answer_token_ids = tokenizer.encode(" " + answer_str, add_special_tokens=False)
    if not answer_token_ids:
        answer_token_ids = tokenizer.encode(answer_str, add_special_tokens=False)
    if not answer_token_ids:
        return -100.0, []

    all_token_ids = input_ids[0].tolist() + answer_token_ids
    total_logprob = 0.0
    per_token_logprobs = []

    with torch.inference_mode():
        full_input = torch.tensor([all_token_ids], device=device)
        outputs = model(input_ids=full_input, return_dict=True)
        logits = outputs.logits[0].float()

        answer_start = len(input_ids[0]) - 1
        for i, tid in enumerate(answer_token_ids):
            pos = answer_start + i
            if pos >= logits.shape[0]:
                break
            log_probs = torch.log_softmax(logits[pos], dim=-1)
            lp = float(log_probs[tid].cpu())
            total_logprob += lp
            per_token_logprobs.append(lp)

    return total_logprob, per_token_logprobs


def compute_full_string_logprob_batch(model, tokenizer, device, prompt, answer_strings):
    results = {}
    for ans in answer_strings:
        lp, per = compute_full_string_logprob(model, tokenizer, device, prompt, ans)
        results[ans] = (lp, per)
    return results


# ============================================================================
# Truth table construction
# ============================================================================

def build_oc_truth_tables(objects, categories, n_tables, seed=42):
    rng = random.Random(seed)
    tables = []
    for t in range(n_tables):
        mapping = {}
        shuffled = list(objects)
        rng.shuffle(shuffled)
        idx = 0
        for c in categories:
            for _ in range(2):
                if idx < len(shuffled):
                    mapping[shuffled[idx]] = c
                    idx += 1
        while idx < len(shuffled):
            mapping[shuffled[idx]] = rng.choice(categories)
            idx += 1
        tables.append(mapping)
    return tables


def build_orv_truth_tables(objects, relations, values, n_tables, seed=42):
    rng = random.Random(seed)
    tables = []
    for t in range(n_tables):
        mapping = {}
        shuffled_objs = list(objects)
        rng.shuffle(shuffled_objs)
        idx = 0
        for v in values:
            for r in relations:
                if idx < len(shuffled_objs):
                    mapping[(shuffled_objs[idx], r)] = v
                    idx += 1
        while idx < len(shuffled_objs):
            for r in relations:
                if idx < len(shuffled_objs):
                    mapping[(shuffled_objs[idx], r)] = rng.choice(values)
                    idx += 1
        tables.append(mapping)
    return tables


def build_cat_rel_truth_tables(categories, relations, values, seed=42):
    """Build (category, relation) -> value mapping for two-hop tasks."""
    rng = random.Random(seed)
    mapping = {}
    for cat in categories:
        for rel in relations:
            mapping[(cat, rel)] = rng.choice(values)
    return mapping


# ============================================================================
# Prompt construction
# ============================================================================

def build_oc_prompt(tokenizer, truth_table, query_object, seed=42):
    rng = random.Random(seed)
    rules = list(truth_table.items())
    rng.shuffle(rules)
    rule_lines = [f"{obj} belongs to {cat}." for obj, cat in rules]
    prompt = "Rules:\n" + "\n".join(rule_lines)
    prompt += f"\n\nQuestion: {query_object} belongs to ?\nAnswer:"
    return prompt


def build_orv_prompt(tokenizer, truth_table, query_object, query_relation, seed=42):
    rng = random.Random(seed)
    rules = list(truth_table.items())
    rng.shuffle(rules)
    rule_lines = [f"{obj} {rel} {val}." for (obj, rel), val in rules]
    prompt = "Rules:\n" + "\n".join(rule_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?\nAnswer:"
    return prompt


def build_two_hop_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    """Build two-hop prompt: O→C rules + (C,R)→V rules, query (O,R)→V."""
    rng = random.Random(seed)

    oc_rules = list(oc_table.items())
    rng.shuffle(oc_rules)
    oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]

    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]

    prompt = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(crv_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?\nAnswer:"

    correct_cat = oc_table.get(query_object)
    correct_val = crv_table.get((correct_cat, query_relation)) if correct_cat else None

    return prompt, correct_cat, correct_val


def build_forced_intermediate_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    """Forced intermediate: prompt model to first find category, then use relation."""
    rng = random.Random(seed)

    oc_rules = list(oc_table.items())
    rng.shuffle(oc_rules)
    oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]

    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]

    prompt = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(crv_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?"
    prompt += f"\nFirst find the category of {query_object}."
    prompt += f"\nThen use that category with relation {query_relation} to find the value."
    prompt += "\nAnswer:"

    correct_cat = oc_table.get(query_object)
    correct_val = crv_table.get((correct_cat, query_relation)) if correct_cat else None

    return prompt, correct_cat, correct_val


def build_gold_intermediate_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    """Gold intermediate: directly provide correct category, test if (C,R)→V works."""
    rng = random.Random(seed)
    correct_cat = oc_table.get(query_object)
    if correct_cat is None:
        return None, None, None

    # Only show CRV rules (not OC rules), and tell the model the correct category
    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]

    prompt = "Rules:\n" + "\n".join(crv_lines)
    prompt += f"\n\n{query_object} belongs to {correct_cat}."
    prompt += f"\nQuestion: {query_object} {query_relation} ?\nAnswer:"

    correct_val = crv_table.get((correct_cat, query_relation))

    return prompt, correct_cat, correct_val


def build_proof_style_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    """Proof-style: structured step-by-step reasoning scaffold."""
    rng = random.Random(seed)

    oc_rules = list(oc_table.items())
    rng.shuffle(oc_rules)
    oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]

    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]

    prompt = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(crv_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?"
    prompt += f"\nStep 1: Find the category of {query_object}."
    prompt += f"\nStep 2: Use that category with {query_relation} to find the value."
    prompt += f"\nStep 3: State the final answer."
    prompt += "\nAnswer:"

    correct_cat = oc_table.get(query_object)
    correct_val = crv_table.get((correct_cat, query_relation)) if correct_cat else None

    return prompt, correct_cat, correct_val


# ============================================================================
# Step 0: Relation necessity audit
# ============================================================================

def run_relation_necessity_audit(model, tokenizer, device, n_tables, objects, relations, values):
    """Test if model truly uses relation R, or just does O→V.

    Key test: same object, two relations with DIFFERENT values.
    If model ignores relation, it will give same answer for both.
    """
    log("--- Step 0: Relation necessity audit ---")

    orv_tables = build_orv_truth_tables(objects, relations, values, n_tables)
    results = []

    for tt_idx in range(min(n_tables, 10)):
        tt = orv_tables[tt_idx]
        # Find objects with two different relations mapped to different values
        for obj in objects[:4]:
            vals_by_rel = {}
            for rel in relations:
                v = tt.get((obj, rel))
                if v:
                    vals_by_rel[rel] = v

            if len(vals_by_rel) < 2:
                continue
            if len(set(vals_by_rel.values())) < 2:
                continue  # Both relations map to same value, skip

            for rel in relations:
                if rel not in vals_by_rel:
                    continue
                correct_val = vals_by_rel[rel]

                prompt = build_orv_prompt(tokenizer, tt, obj, rel, seed=tt_idx * 100)
                lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, prompt, values)
                pred = max(lps, key=lambda c: lps[c][0])
                correct = pred == correct_val

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_val": correct_val,
                    "predicted": pred, "is_correct": correct,
                    "all_vals": {v: vals_by_rel.get(v, None) for v in values},
                    "logprobs": {c: lp for c, (lp, _) in lps.items()},
                })

            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / max(1, len(results))

    # Check relation discrimination: does model give different answers for different relations?
    rel_discrim = 0
    rel_discrim_total = 0
    by_obj_tt = {}
    for r in results:
        key = (r["tt_idx"], r["obj"])
        if key not in by_obj_tt:
            by_obj_tt[key] = {}
        by_obj_tt[key][r["rel"]] = r["predicted"]

    for key, preds in by_obj_tt.items():
        if len(preds) >= 2:
            rel_discrim_total += 1
            if len(set(preds.values())) >= 2:
                rel_discrim += 1

    discrim_rate = rel_discrim / max(1, rel_discrim_total)

    log(f"  Relation audit: accuracy={accuracy:.3f} ({n_correct}/{len(results)})")
    log(f"  Relation discrimination: {rel_discrim}/{rel_discrim_total} = {discrim_rate:.3f}")
    log(f"  (discrimination means model gives different answers for different relations)")

    return {
        "accuracy": accuracy,
        "n_samples": len(results),
        "relation_discrimination_rate": discrim_rate,
        "n_discrim_samples": rel_discrim_total,
        "results": results,
    }


# ============================================================================
# Step 1: Two-hop three-segment decomposition
# ============================================================================

def run_two_hop_decomposition(model, tokenizer, device, n_tables, objects, relations,
                               categories, values):
    """Decompose two-hop into: Step A (O→C), Step B (C,R→V), Step C (Compose)."""
    log("--- Step 1: Two-hop three-segment decomposition ---")

    oc_tables = build_oc_truth_tables(objects, categories, n_tables)
    results = []

    for tt_idx in range(min(n_tables, 10)):
        oc_tt = oc_tables[tt_idx]
        crv_tt = build_cat_rel_truth_tables(categories, relations, values, seed=tt_idx * 200)

        for obj in objects[:4]:
            for rel in relations:
                correct_cat = oc_tt.get(obj)
                if correct_cat is None:
                    continue
                correct_val = crv_tt.get((correct_cat, rel))
                if correct_val is None:
                    continue

                # Step A: O→C (single OC retrieval)
                oc_prompt = build_oc_prompt(tokenizer, oc_tt, obj, seed=tt_idx * 100)
                oc_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, oc_prompt, categories)
                oc_pred = max(oc_lps, key=lambda c: oc_lps[c][0])
                step_a_correct = oc_pred == correct_cat

                # Step B: (C,R)→V (gold category provided)
                gold_prompt, _, gold_val = build_gold_intermediate_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                if gold_prompt is None:
                    continue
                gold_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, gold_prompt, values)
                gold_pred = max(gold_lps, key=lambda c: gold_lps[c][0])
                step_b_correct = gold_pred == correct_val

                # Step C: Compose (full two-hop)
                twohop_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                twohop_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, values)
                twohop_pred = max(twohop_lps, key=lambda c: twohop_lps[c][0])
                step_c_correct = twohop_pred == correct_val

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "step_a_pred_cat": oc_pred, "step_a_correct": step_a_correct,
                    "step_b_pred_val": gold_pred, "step_b_correct": step_b_correct,
                    "step_c_pred_val": twohop_pred, "step_c_correct": step_c_correct,
                })

                if len(results) >= 30:
                    break
            if len(results) >= 30:
                break
        if len(results) >= 30:
            break

    n_a = sum(1 for r in results if r["step_a_correct"])
    n_b = sum(1 for r in results if r["step_b_correct"])
    n_c = sum(1 for r in results if r["step_c_correct"])

    # Also check: when both A and B correct, is C correct?
    ab_correct = [r for r in results if r["step_a_correct"] and r["step_b_correct"]]
    n_ab_and_c = sum(1 for r in ab_correct if r["step_c_correct"])

    log(f"  Step A (O→C): {n_a}/{len(results)} = {n_a/max(1,len(results)):.3f}")
    log(f"  Step B (C,R→V, gold): {n_b}/{len(results)} = {n_b/max(1,len(results)):.3f}")
    log(f"  Step C (Compose): {n_c}/{len(results)} = {n_c/max(1,len(results)):.3f}")
    log(f"  A&B correct but C: {n_ab_and_c}/{len(ab_correct)} = {n_ab_and_c/max(1,len(ab_correct)):.3f}")

    return {
        "n_samples": len(results),
        "step_a_accuracy": n_a / max(1, len(results)),
        "step_b_accuracy": n_b / max(1, len(results)),
        "step_c_accuracy": n_c / max(1, len(results)),
        "ab_correct_and_c": n_ab_and_c / max(1, len(ab_correct)),
        "n_ab_correct": len(ab_correct),
        "results": results,
    }


# ============================================================================
# Step 2 & 3: Forced intermediate and gold intermediate
# ============================================================================

def run_forced_and_gold_intermediate(model, tokenizer, device, n_tables, objects,
                                      relations, categories, values):
    """Test forced intermediate (prompt to find C first) and gold intermediate."""
    log("--- Step 2&3: Forced and gold intermediate ---")

    oc_tables = build_oc_truth_tables(objects, categories, n_tables)
    results = []

    for tt_idx in range(min(n_tables, 10)):
        oc_tt = oc_tables[tt_idx]
        crv_tt = build_cat_rel_truth_tables(categories, relations, values, seed=tt_idx * 200)

        for obj in objects[:4]:
            for rel in relations:
                correct_cat = oc_tt.get(obj)
                if correct_cat is None:
                    continue
                correct_val = crv_tt.get((correct_cat, rel))
                if correct_val is None:
                    continue

                # Direct two-hop (baseline)
                direct_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                direct_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, direct_prompt, values)
                direct_pred = max(direct_lps, key=lambda c: direct_lps[c][0])
                direct_correct = direct_pred == correct_val

                # Forced intermediate
                forced_prompt, _, _ = build_forced_intermediate_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                forced_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, forced_prompt, values)
                forced_pred = max(forced_lps, key=lambda c: forced_lps[c][0])
                forced_correct = forced_pred == correct_val

                # Gold intermediate
                gold_prompt, _, _ = build_gold_intermediate_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                if gold_prompt:
                    gold_lps = compute_full_string_logprob_batch(
                        model, tokenizer, device, gold_prompt, values)
                    gold_pred = max(gold_lps, key=lambda c: gold_lps[c][0])
                    gold_correct = gold_pred == correct_val
                else:
                    gold_pred = "none"
                    gold_correct = False

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_val": correct_val,
                    "direct_pred": direct_pred, "direct_correct": direct_correct,
                    "forced_pred": forced_pred, "forced_correct": forced_correct,
                    "gold_pred": gold_pred, "gold_correct": gold_correct,
                })

                if len(results) >= 30:
                    break
            if len(results) >= 30:
                break
        if len(results) >= 30:
            break

    n_direct = sum(1 for r in results if r["direct_correct"])
    n_forced = sum(1 for r in results if r["forced_correct"])
    n_gold = sum(1 for r in results if r["gold_correct"])

    log(f"  Direct two-hop: {n_direct}/{len(results)} = {n_direct/max(1,len(results)):.3f}")
    log(f"  Forced intermediate: {n_forced}/{len(results)} = {n_forced/max(1,len(results)):.3f}")
    log(f"  Gold intermediate: {n_gold}/{len(results)} = {n_gold/max(1,len(results)):.3f}")

    return {
        "n_samples": len(results),
        "direct_accuracy": n_direct / max(1, len(results)),
        "forced_accuracy": n_forced / max(1, len(results)),
        "gold_accuracy": n_gold / max(1, len(results)),
        "results": results,
    }


# ============================================================================
# Step 4: Activation-level intermediate injection
# ============================================================================

def run_activation_injection(model, tokenizer, device, n_tables, objects, relations,
                              categories, values, n_layers):
    """Inject hidden states from single-step OC retrieval into two-hop task.

    Extract h at answer position from single-step OC prompt.
    Inject into two-hop prompt at answer position at various layers.
    """
    log("--- Step 4: Activation-level intermediate injection ---")

    oc_tables = build_oc_truth_tables(objects, categories, n_tables)
    results = []

    # Sample layers to inject at (early, mid, late)
    inject_layers = [n_layers // 4, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 2]
    inject_layers = sorted(set(inject_layers))
    log(f"  Injection layers: {inject_layers}")

    for tt_idx in range(min(n_tables, 5)):
        oc_tt = oc_tables[tt_idx]
        crv_tt = build_cat_rel_truth_tables(categories, relations, values, seed=tt_idx * 200)

        for obj in objects[:3]:
            for rel in relations[:1]:  # Limit for speed
                correct_cat = oc_tt.get(obj)
                if correct_cat is None:
                    continue
                correct_val = crv_tt.get((correct_cat, rel))
                if correct_val is None:
                    continue

                # Step 1: Get hidden states from single-step OC prompt
                oc_prompt = build_oc_prompt(tokenizer, oc_tt, obj, seed=tt_idx * 100)
                oc_input_ids = torch.tensor(
                    [tokenizer.encode(oc_prompt, add_special_tokens=False)], device=device)

                with torch.inference_mode():
                    oc_outputs = model(input_ids=oc_input_ids, output_hidden_states=True,
                                       return_dict=True)
                oc_hidden = oc_outputs.hidden_states  # tuple of [1, seq, d_model]
                oc_answer_pos = oc_input_ids.shape[1] - 1

                # Step 2: Two-hop prompt baseline
                twohop_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                twohop_input_ids = torch.tensor(
                    [tokenizer.encode(twohop_prompt, add_special_tokens=False)], device=device)
                twohop_answer_pos = twohop_input_ids.shape[1] - 1

                # Baseline two-hop logprob
                base_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, values)
                base_pred = max(base_lps, key=lambda c: base_lps[c][0])
                base_correct = base_pred == correct_val

                # Step 3: Inject OC hidden states into two-hop at various layers
                for inject_layer in inject_layers:
                    for alpha in [0.5, 1.0, 2.0]:
                        try:
                            # Hook to inject OC hidden state at inject_layer
                            oc_h = oc_hidden[inject_layer][0, oc_answer_pos, :].detach().clone()

                            injected = {"done": False}

                            def make_inject_hook(oc_vec, layer_idx, alpha_val, ans_pos):
                                def hook(module, input, output):
                                    if injected["done"]:
                                        return output
                                    if isinstance(output, tuple):
                                        h = output[0]
                                    else:
                                        h = output
                                    # Add OC hidden state to answer position
                                    h_new = h.clone()
                                    h_new[0, ans_pos, :] = h_new[0, ans_pos, :] + alpha_val * oc_vec.to(
                                        h_new.device, h_new.dtype)
                                    injected["done"] = True
                                    if isinstance(output, tuple):
                                        return (h_new,) + output[1:]
                                    return h_new
                                return hook

                            layers_list = get_layers(model)
                            hook_target = layers_list[inject_layer]
                            hook = hook_target.register_forward_hook(
                                make_inject_hook(oc_h, inject_layer, alpha, twohop_answer_pos))

                            try:
                                inj_lps = compute_full_string_logprob_batch(
                                    model, tokenizer, device, twohop_prompt, values)
                            finally:
                                hook.remove()

                            inj_pred = max(inj_lps, key=lambda c: inj_lps[c][0])
                            inj_correct = inj_pred == correct_val

                            results.append({
                                "tt_idx": tt_idx, "obj": obj, "rel": rel,
                                "correct_val": correct_val,
                                "base_pred": base_pred, "base_correct": base_correct,
                                "inject_layer": inject_layer, "alpha": alpha,
                                "inj_pred": inj_pred, "inj_correct": inj_correct,
                                "base_logprob": base_lps[correct_val][0],
                                "inj_logprob": inj_lps[correct_val][0],
                            })
                        except Exception as e:
                            log(f"  WARNING: injection failed at L{inject_layer} alpha={alpha}: {e}")

                if len(results) >= 30:
                    break
            if len(results) >= 30:
                break
        if len(results) >= 30:
            break

    # Summarize by layer and alpha
    summary = {}
    for r in results:
        key = f"L{r['inject_layer']}_a{r['alpha']}"
        if key not in summary:
            summary[key] = {"n": 0, "correct": 0, "base_correct": 0}
        summary[key]["n"] += 1
        summary[key]["correct"] += 1 if r["inj_correct"] else 0
        summary[key]["base_correct"] += 1 if r["base_correct"] else 0

    for key, s in sorted(summary.items()):
        log(f"  {key}: injected={s['correct']}/{s['n']}={s['correct']/max(1,s['n']):.3f}, "
            f"base={s['base_correct']}/{s['n']}={s['base_correct']/max(1,s['n']):.3f}")

    return {
        "n_samples": len(results),
        "summary": summary,
        "results": results,
    }


# ============================================================================
# Step 7: Grammar scaffold impact
# ============================================================================

def run_grammar_scaffold_impact(model, tokenizer, device, n_tables, objects, relations,
                                 categories, values):
    """Test direct vs forced-intermediate vs proof-style prompts."""
    log("--- Step 7: Grammar scaffold impact on composition ---")

    oc_tables = build_oc_truth_tables(objects, categories, n_tables)
    results = []

    for tt_idx in range(min(n_tables, 10)):
        oc_tt = oc_tables[tt_idx]
        crv_tt = build_cat_rel_truth_tables(categories, relations, values, seed=tt_idx * 200)

        for obj in objects[:4]:
            for rel in relations:
                correct_cat = oc_tt.get(obj)
                if correct_cat is None:
                    continue
                correct_val = crv_tt.get((correct_cat, rel))
                if correct_val is None:
                    continue

                prompts = {
                    "direct": build_two_hop_prompt(tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100),
                    "forced": build_forced_intermediate_prompt(tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100),
                    "proof": build_proof_style_prompt(tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100),
                }

                row = {
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_val": correct_val,
                }

                for style, (prompt, _, _) in prompts.items():
                    if prompt is None:
                        row[f"{style}_pred"] = "none"
                        row[f"{style}_correct"] = False
                        continue
                    lps = compute_full_string_logprob_batch(
                        model, tokenizer, device, prompt, values)
                    pred = max(lps, key=lambda c: lps[c][0])
                    row[f"{style}_pred"] = pred
                    row[f"{style}_correct"] = pred == correct_val
                    row[f"{style}_logprob"] = lps[correct_val][0]

                results.append(row)

                if len(results) >= 30:
                    break
            if len(results) >= 30:
                break
        if len(results) >= 30:
            break

    for style in ["direct", "forced", "proof"]:
        n = sum(1 for r in results if r.get(f"{style}_correct"))
        log(f"  {style}: {n}/{len(results)} = {n/max(1,len(results)):.3f}")

    return {
        "n_samples": len(results),
        "direct_accuracy": sum(1 for r in results if r.get("direct_correct")) / max(1, len(results)),
        "forced_accuracy": sum(1 for r in results if r.get("forced_correct")) / max(1, len(results)),
        "proof_accuracy": sum(1 for r in results if r.get("proof_correct")) / max(1, len(results)),
        "results": results,
    }


# ============================================================================
# Main
# ============================================================================

def run_model(args):
    global _safetensors_cache
    _safetensors_cache = {}

    model, tokenizer, device = load_model_eager(args.model)
    try:
        info = get_model_info(model, args.model)
        n_heads, d_head, d_model, n_kv_heads, kv_group_size = get_head_dims(model)
        n_layers = info.n_layers

        log(f"{args.model}: n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}")

        objects = CANDIDATE_OBJECTS[:8]
        categories = CANDIDATE_CATEGORIES[:4]
        relations = CANDIDATE_RELATIONS[:2]
        values = CANDIDATE_VALUES[:4]
        n_tables = args.n_tables

        result = {
            "phase": 581,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers, "n_heads": n_heads, "d_head": d_head,
            "n_tables": n_tables,
        }

        # Step 0: Relation necessity audit
        result["step0_relation_audit"] = run_relation_necessity_audit(
            model, tokenizer, device, n_tables, objects, relations, values)

        # Step 1: Two-hop three-segment decomposition
        result["step1_decomposition"] = run_two_hop_decomposition(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        # Step 2&3: Forced and gold intermediate
        result["step2_3_forced_gold"] = run_forced_and_gold_intermediate(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        # Step 4: Activation injection
        result["step4_activation_injection"] = run_activation_injection(
            model, tokenizer, device, n_tables, objects, relations, categories, values, n_layers)

        # Step 7: Grammar scaffold impact
        result["step7_grammar_scaffold"] = run_grammar_scaffold_impact(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        return result

    finally:
        release_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n-tables", type=int, default=10)
    parser.add_argument("--output-dir", default=str(OUT_ROOT))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 4
        log("SMOKE TEST MODE: n_tables=4")
    elif args.confirm:
        args.n_tables = 15
        log("CONFIRMATION TEST MODE: n_tables=15")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ("_confirm" if args.confirm else "")
    out_path = out_dir / f"phase581_{args.model}_composition_closure{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
