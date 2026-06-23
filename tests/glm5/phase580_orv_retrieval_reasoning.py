#!/usr/bin/env python3
"""
Phase 580: Micro-World ORV Retrieval-to-Reasoning Closure
微世界对象—关系—值检索到推理闭包

Key improvements over Phase 579:
1. FIX: Full-string logprob scoring (not single token)
2. FIX: True edge ablation with BOTH renormalized AND no-renorm versions
3. NEW: Object-Relation-Value (ORV) task — (O,R)→V
4. NEW: Retrieval circuit localization for ORV (obj/rel/val heads)
5. NEW: Value-copy attribution for ORV
6. NEW: Compositional reasoning test (two-hop: O→C then (C,R)→V)
7. NEW: Syntax template variation (same truth, different syntax)
8. NEW: Balanced table-conditioned binding for ORV

Three-round testing:
  Round 1 (smoke): --smoke, 4 tables, quick validation
  Round 2 (main):  default, 10 tables, full analysis
  Round 3 (confirm): --confirm, more samples for key results

Model loading: BF16 + device_map="auto" + eager attention
GQA handling: Proper W_V/W_O slicing using config.head_dim + kv_group_size
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

OUT_ROOT = Path("results/glm5_phase580_orv_closure")

# ============================================================================
# Constants
# ============================================================================

CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_RELATIONS = ["r31", "r64"]
CANDIDATE_VALUES = ["v05", "v91", "v22", "v48"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]

SYNTAX_TEMPLATES = {
    "minimal": "{obj} {rel} {val}.",
    "natural": "The {rel} of {obj} is {val}.",
    "prepositional": "For {obj}, relation {rel} gives {val}.",
    "tabular": "Object {obj} under {rel} maps to {val}.",
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Model loading
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
# GQA head dimensions
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


# ============================================================================
# Safetensors weight loading (handles meta device)
# ============================================================================

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
        from model_registry import get_model_spec, MODEL_SPECS
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


def get_W_V_W_O(model, layer_idx, d_head, kv_group_size, h):
    """Get W_V_h and W_O_h for a specific head, handling meta device and GQA."""
    layers_list = get_layers(model)
    layer_obj = layers_list[layer_idx]
    kv_h = h // kv_group_size

    W_V_weight = layer_obj.self_attn.v_proj.weight
    W_O_weight = layer_obj.self_attn.o_proj.weight

    if W_V_weight.is_meta or W_O_weight.is_meta:
        W_V_np, W_O_np = _load_attn_weights_from_safetensors(model, layer_idx)
        W_V_h = W_V_np[kv_h * d_head:(kv_h + 1) * d_head, :]
        W_O_h = W_O_np[:, h * d_head:(h + 1) * d_head]
    else:
        W_V_h = W_V_weight[kv_h * d_head:(kv_h + 1) * d_head, :].detach().float().cpu().numpy()
        W_O_h = W_O_weight[:, h * d_head:(h + 1) * d_head].detach().float().cpu().numpy()

    return W_V_h, W_O_h


# ============================================================================
# Full-string logprob scoring (KEY FIX from Phase 579)
# ============================================================================

def compute_full_string_logprob(model, tokenizer, device, prompt, answer_str):
    """Compute log P(answer_str | prompt) by summing over all tokens in answer_str.

    This is the CORRECT way to score multi-token categories/values.
    Instead of looking at a single token's logit, we conditionally generate
    each token of the answer string and sum the log-probabilities.

    Returns:
        total_logprob: sum of log P(token_i | prompt + tokens_before_i)
        per_token_logprobs: list of individual token log-probs
    """
    # First get logits at the last position of the prompt
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
        # We need logits at positions where answer tokens are predicted
        # Efficient: one forward pass for the entire sequence
        full_input = torch.tensor([all_token_ids], device=device)
        outputs = model(input_ids=full_input, return_dict=True)
        logits = outputs.logits[0].float()  # [seq, vocab]

        # The answer starts at position len(input_ids[0])-1
        answer_start = len(input_ids[0]) - 1  # first answer token predicted from last prompt token

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
    """Compute full-string logprobs for multiple answer candidates.

    Returns:
        dict: {answer_str: (logprob, per_token_logprobs)}
    """
    results = {}
    for ans in answer_strings:
        lp, per = compute_full_string_logprob(model, tokenizer, device, prompt, ans)
        results[ans] = (lp, per)
    return results


# ============================================================================
# ORV Truth Table Construction
# ============================================================================

def build_orv_truth_tables(objects, relations, values, n_tables, seed=42):
    """Build Object-Relation-Value truth tables.

    Each table maps (object, relation) -> value.
    """
    rng = random.Random(seed)
    tables = []
    n_vals = len(values)

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
        # Fill remaining
        while idx < len(shuffled_objs):
            for r in relations:
                if idx < len(shuffled_objs):
                    mapping[(shuffled_objs[idx], r)] = rng.choice(values)
                    idx += 1
        tables.append(mapping)
    return tables


def build_oc_truth_tables(objects, categories, n_tables, seed=42):
    """Build Object-Category truth tables (same as Phase 577-579)."""
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


# ============================================================================
# Prompt Construction for ORV and OC tasks
# ============================================================================

def find_subsequence(full_ids, sub_ids, start=0):
    n = len(sub_ids)
    for i in range(start, len(full_ids) - n + 1):
        if full_ids[i:i + n] == sub_ids:
            return i
    return None


def find_symbol_in_full(tokenizer, full_ids, symbol, start=0):
    variants = [
        tokenizer.encode(symbol, add_special_tokens=False),
        tokenizer.encode(" " + symbol, add_special_tokens=False),
        tokenizer.encode("\n" + symbol, add_special_tokens=False),
    ]
    seen = set()
    unique_variants = []
    for v in variants:
        key = tuple(v)
        if key not in seen and v:
            seen.add(key)
            unique_variants.append(v)
    for v in unique_variants:
        pos = find_subsequence(full_ids, v, start)
        if pos is not None:
            return pos, pos + len(v)
    return None


def build_orv_prompt_with_positions(tokenizer, truth_table, query_object, query_relation,
                                     template="minimal", seed=42):
    """Build ORV prompt with token position tracking.

    template: syntax template name from SYNTAX_TEMPLATES
    """
    rng = random.Random(seed)
    rules = list(truth_table.items())
    rng.shuffle(rules)

    tmpl = SYNTAX_TEMPLATES.get(template, SYNTAX_TEMPLATES["minimal"])
    rule_lines = [tmpl.format(obj=obj, rel=rel, val=val) for (obj, rel), val in rules]

    prompt = "Rules:\n" + "\n".join(rule_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?\nAnswer:"

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)

    rule_info = []
    search_start = 0

    for (obj, rel), val in rules:
        obj_result = find_symbol_in_full(tokenizer, full_ids, obj, search_start)
        if obj_result is None:
            continue
        obj_pos, obj_end = obj_result

        rel_result = find_symbol_in_full(tokenizer, full_ids, rel, obj_end)
        if rel_result is None:
            search_start = obj_end
            continue
        rel_pos, rel_end = rel_result

        val_result = find_symbol_in_full(tokenizer, full_ids, val, rel_end)
        if val_result is None:
            search_start = rel_end
            continue
        val_pos, val_end = val_result

        is_correct = (obj == query_object and rel == query_relation)
        rule_info.append({
            "object": obj, "relation": rel, "value": val,
            "obj_positions": list(range(obj_pos, obj_end)),
            "rel_positions": list(range(rel_pos, rel_end)),
            "val_positions": list(range(val_pos, val_end)),
            "is_correct": is_correct,
        })
        search_start = val_end

    answer_pos = len(full_ids) - 1
    return prompt, full_ids, rule_info, answer_pos


def build_oc_prompt_with_positions(tokenizer, truth_table, query_object, seed=42):
    """Build OC prompt with token position tracking (same as Phase 577-579)."""
    rng = random.Random(seed)
    rules = list(truth_table.items())
    rng.shuffle(rules)

    rule_lines = [f"{obj} belongs to {cat}." for obj, cat in rules]
    prompt = "Rules:\n" + "\n".join(rule_lines)
    prompt += f"\n\nQuestion: {query_object} belongs to ?\nAnswer:"

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)

    rule_info = []
    search_start = 0
    for obj, cat in rules:
        obj_result = find_symbol_in_full(tokenizer, full_ids, obj, search_start)
        if obj_result is None:
            search_start += 1
            continue
        obj_pos, obj_end = obj_result
        cat_result = find_symbol_in_full(tokenizer, full_ids, cat, obj_end)
        if cat_result is None:
            search_start = obj_end
            continue
        cat_pos, cat_end = cat_result

        rule_info.append({
            "object": obj, "category": cat,
            "obj_positions": list(range(obj_pos, obj_end)),
            "cat_positions": list(range(cat_pos, cat_end)),
            "is_correct": obj == query_object,
        })
        search_start = cat_end

    answer_pos = len(full_ids) - 1
    return prompt, full_ids, rule_info, answer_pos


# ============================================================================
# Attention extraction and head scoring
# ============================================================================

def extract_attention_and_logits(model, tokenizer, device, prompt, answer_pos):
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)
    attn_rows = {}
    for l, attn in enumerate(outputs.attentions):
        if attn is not None:
            row = attn[0, :, answer_pos, :].detach().float().cpu().numpy()
            attn_rows[l] = row
    logits = outputs.logits[0, answer_pos, :].detach().float().cpu().numpy()
    return attn_rows, logits


def compute_orv_match_scores(attn_row, rule_info, n_heads):
    """Compute obj-match, rel-match, val-copy scores for ORV task."""
    scores = {}
    correct_rule = None
    wrong_rules = []
    for r in rule_info:
        if r["is_correct"]:
            correct_rule = r
        else:
            wrong_rules.append(r)

    if correct_rule is None or not wrong_rules:
        return scores

    correct_obj_pos = correct_rule["obj_positions"]
    correct_rel_pos = correct_rule["rel_positions"]
    correct_val_pos = correct_rule["val_positions"]
    wrong_obj_pos = [r["obj_positions"] for r in wrong_rules]
    wrong_rel_pos = [r["rel_positions"] for r in wrong_rules]
    wrong_val_pos = [r["val_positions"] for r in wrong_rules]

    for h in range(n_heads):
        attn_correct_obj = float(attn_row[h, correct_obj_pos].sum())
        attn_wrong_obj = np.mean([float(attn_row[h, p].sum()) for p in wrong_obj_pos]) if wrong_obj_pos else 0
        attn_correct_rel = float(attn_row[h, correct_rel_pos].sum())
        attn_wrong_rel = np.mean([float(attn_row[h, p].sum()) for p in wrong_rel_pos]) if wrong_rel_pos else 0
        attn_correct_val = float(attn_row[h, correct_val_pos].sum())
        attn_wrong_val = np.mean([float(attn_row[h, p].sum()) for p in wrong_val_pos]) if wrong_val_pos else 0

        total_obj = attn_correct_obj + attn_wrong_obj
        total_rel = attn_correct_rel + attn_wrong_rel
        total_val = attn_correct_val + attn_wrong_val

        obj_match = attn_correct_obj / (total_obj + 1e-10)
        rel_match = attn_correct_rel / (total_rel + 1e-10)
        val_copy = attn_correct_val / (total_val + 1e-10)

        scores[h] = {
            "obj_match": float(obj_match),
            "rel_match": float(rel_match),
            "val_copy": float(val_copy),
            "attn_mass_obj": attn_correct_obj,
            "attn_mass_rel": attn_correct_rel,
            "attn_mass_val": attn_correct_val,
        }
    return scores


def compute_oc_match_scores(attn_row, rule_info, n_heads):
    """Compute obj-match, cat-copy scores for OC task (same as Phase 578)."""
    scores = {}
    correct_rule = None
    wrong_rules = []
    for r in rule_info:
        if r["is_correct"]:
            correct_rule = r
        else:
            wrong_rules.append(r)

    if correct_rule is None or not wrong_rules:
        return scores

    correct_obj_pos = correct_rule["obj_positions"]
    correct_cat_pos = correct_rule["cat_positions"]
    wrong_obj_pos = [r["obj_positions"] for r in wrong_rules]
    wrong_cat_pos = [r["cat_positions"] for r in wrong_rules]

    for h in range(n_heads):
        attn_correct_obj = float(attn_row[h, correct_obj_pos].sum())
        attn_wrong_obj = np.mean([float(attn_row[h, p].sum()) for p in wrong_obj_pos])
        attn_correct_cat = float(attn_row[h, correct_cat_pos].sum())
        attn_wrong_cat = np.mean([float(attn_row[h, p].sum()) for p in wrong_cat_pos])

        obj_match = attn_correct_obj / (attn_correct_obj + attn_wrong_obj + 1e-10)
        cat_copy = attn_correct_cat / (attn_correct_cat + attn_wrong_cat + 1e-10)

        scores[h] = {
            "obj_match": float(obj_match),
            "cat_copy": float(cat_copy),
            "attn_mass_obj": attn_correct_obj,
            "attn_mass_cat": attn_correct_cat,
        }
    return scores


# ============================================================================
# Step 1: TRUE attention edge ablation (FIXED: dual mode)
# ============================================================================

def true_attn_edge_ablation_forward(model, tokenizer, device, prompt, answer_pos,
                                     target_positions, target_layers_heads,
                                     renorm=True):
    """Forward pass with TRUE attention edge ablation.

    Args:
        renorm: If True, renormalize after zeroing (Phase 579 style).
                If False, keep total attention mass reduced (zero-mass ablation).
                No-renorm reveals the true contribution of the ablated edge
                without compensation from other positions.
    """
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)

    attn_weights_all = outputs.attentions
    hidden_states_all = outputs.hidden_states
    base_logits = outputs.logits[0, answer_pos, :].detach().float().cpu().numpy()

    n_heads_cfg, d_head, d_model, n_kv_heads, kv_group_size = get_head_dims(model)
    W_U = get_W_U(model, model_name=None)

    modified_logits = base_logits.copy()

    for (l, h_list) in target_layers_heads.items():
        if l >= len(attn_weights_all) or attn_weights_all[l] is None:
            continue

        for h in h_list:
            attn = attn_weights_all[l][0, h, answer_pos, :].detach().float().cpu()
            h_input = hidden_states_all[l][0, :, :].detach().float().cpu()

            W_V_h, W_O_h = get_W_V_W_O(model, l, d_head, kv_group_size, h)
            h_input_np = h_input.numpy()

            V_all = (W_V_h @ h_input_np.T).T  # [seq, d_head]

            attn_np = attn.numpy()

            if attn_np.shape[0] != V_all.shape[0]:
                continue
            if V_all.shape[1] != W_O_h.shape[1]:
                continue

            original_weighted_V = attn_np[:, np.newaxis] * V_all
            original_head_out = W_O_h @ original_weighted_V.sum(axis=0)

            # Modified attention: zero target_positions
            modified_attn = attn_np.copy()
            for pos in target_positions:
                if pos < len(modified_attn):
                    modified_attn[pos] = 0.0

            if renorm:
                # Renormalize: redistribute attention mass
                sum_modified = modified_attn.sum()
                if sum_modified > 1e-10:
                    modified_attn = modified_attn / sum_modified
                else:
                    modified_head_out = np.zeros_like(original_head_out)
                    delta_head = modified_head_out - original_head_out
                    modified_logits += W_U @ delta_head
                    continue
            # else: no renorm — total attention mass is reduced
            # This means the head contributes less overall, revealing
            # the true magnitude of the ablated edge's contribution

            modified_weighted_V = modified_attn[:, np.newaxis] * V_all
            modified_head_out = W_O_h @ modified_weighted_V.sum(axis=0)

            delta_head = modified_head_out - original_head_out
            delta_logits = W_U @ delta_head
            modified_logits += delta_logits

    return modified_logits


# ============================================================================
# Step 3: Value vector logit attribution (FIXED: full-string scoring)
# ============================================================================

def compute_value_logit_attribution_orv(model, tokenizer, device, prompt, answer_pos,
                                         rule_info, top_heads, value_token_ids, d_head):
    """For each top retrieval head, compute logit contribution to correct value.

    Uses GQA-correct W_V/W_O slicing.
    """
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)

    attn_weights_all = outputs.attentions
    hidden_states_all = outputs.hidden_states
    W_U = get_W_U(model, model_name=None)
    n_heads_cfg, d_head_actual, d_model, n_kv_heads, kv_group_size = get_head_dims(model)

    results = []

    for l, h_list in top_heads.items():
        for h in h_list:
            if l >= len(attn_weights_all) or attn_weights_all[l] is None:
                continue

            attn = attn_weights_all[l][0, h, answer_pos, :].detach().float().cpu().numpy()
            h_input = hidden_states_all[l][0, :, :].detach().float().cpu().numpy()

            W_V_h, W_O_h = get_W_V_W_O(model, l, d_head_actual, kv_group_size, h)

            V_all = (W_V_h @ h_input.T).T  # [seq, d_head]
            weighted_V = attn[:, np.newaxis] * V_all
            head_output = W_O_h @ weighted_V.sum(axis=0)  # [d_model]

            # Logit contribution for each value token
            for val, tid in value_token_ids.items():
                logit_contrib = float(W_U[tid] @ head_output)
                results.append({
                    "layer": l, "head": h, "value": val,
                    "logit_contribution": logit_contrib,
                })

            # Position-specific contributions for correct rule
            correct_rule = None
            for r in rule_info:
                if r["is_correct"]:
                    correct_rule = r
                    break

            if correct_rule is not None:
                correct_val = correct_rule["value"]
                correct_tid = value_token_ids[correct_val]

                val_direct = 0.0
                for p in correct_rule["val_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        val_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                obj_direct = 0.0
                for p in correct_rule["obj_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        obj_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                rel_direct = 0.0
                for p in correct_rule["rel_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        rel_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                results.append({
                    "layer": l, "head": h, "type": "position_attribution",
                    "val_direct_logit_from_val": val_direct,
                    "val_direct_logit_from_obj": obj_direct,
                    "val_direct_logit_from_rel": rel_direct,
                    "attn_to_correct_obj": float(attn[correct_rule["obj_positions"]].sum()),
                    "attn_to_correct_rel": float(attn[correct_rule["rel_positions"]].sum()),
                    "attn_to_correct_val": float(attn[correct_rule["val_positions"]].sum()),
                    "correct_value": correct_val,
                })

    return results


# ============================================================================
# Step 4: Value-copy attribution for OC (FIXED with GQA)
# ============================================================================

def compute_value_logit_attribution_oc(model, tokenizer, device, prompt, answer_pos,
                                       rule_info, top_heads, cat_token_ids, d_head):
    """OC task value attribution — same as Phase 579 but with fixed GQA handling."""
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)

    attn_weights_all = outputs.attentions
    hidden_states_all = outputs.hidden_states
    W_U = get_W_U(model, model_name=None)
    n_heads_cfg, d_head_actual, d_model, n_kv_heads, kv_group_size = get_head_dims(model)

    results = []
    for l, h_list in top_heads.items():
        for h in h_list:
            if l >= len(attn_weights_all) or attn_weights_all[l] is None:
                continue

            attn = attn_weights_all[l][0, h, answer_pos, :].detach().float().cpu().numpy()
            h_input = hidden_states_all[l][0, :, :].detach().float().cpu().numpy()

            W_V_h, W_O_h = get_W_V_W_O(model, l, d_head_actual, kv_group_size, h)

            V_all = (W_V_h @ h_input.T).T
            weighted_V = attn[:, np.newaxis] * V_all
            head_output = W_O_h @ weighted_V.sum(axis=0)

            correct_rule = None
            for r in rule_info:
                if r["is_correct"]:
                    correct_rule = r
                    break

            if correct_rule is not None:
                correct_cat = correct_rule["category"]
                correct_tid = cat_token_ids[correct_cat]

                cat_direct = 0.0
                for p in correct_rule["cat_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        cat_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                obj_direct = 0.0
                for p in correct_rule["obj_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        obj_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                results.append({
                    "layer": l, "head": h, "type": "oc_position_attribution",
                    "cat_direct_logit_from_cat": cat_direct,
                    "cat_direct_logit_from_obj": obj_direct,
                    "attn_to_correct_obj": float(attn[correct_rule["obj_positions"]].sum()),
                    "attn_to_correct_cat": float(attn[correct_rule["cat_positions"]].sum()),
                    "correct_category": correct_cat,
                })

    return results


# ============================================================================
# Step 5: Compositional reasoning test (two-hop)
# ============================================================================

def build_compositional_prompt(tokenizer, oc_table, orv_table, query_object, query_relation,
                                seed=42):
    """Build two-hop prompt: O→C then (C,R)→V.

    Example:
    Rules:
    o17 belongs to c12.
    c12 r31 v05.
    c77 r64 v22.
    ...

    Question: o17 r31 ?
    Answer:

    This requires composing: o17→c12, then c12→r31→v05
    """
    rng = random.Random(seed)

    # First set of rules: object → category
    oc_rules = list(oc_table.items())
    rng.shuffle(oc_rules)
    oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]

    # Second set of rules: (category, relation) → value
    orv_rules = [((cat, rel), val) for (cat, rel), val in orv_table.items()
                 if cat in set(oc_table.values())]
    rng.shuffle(orv_rules)
    orv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in orv_rules]

    prompt = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(orv_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?\nAnswer:"

    correct_cat = oc_table.get(query_object, None)
    correct_val = orv_table.get((correct_cat, query_relation), None) if correct_cat else None

    return prompt, correct_cat, correct_val


# ============================================================================
# Step 6: Syntax template variation
# ============================================================================

def build_orv_prompt_with_template(tokenizer, truth_table, query_object, query_relation,
                                    template_name, seed=42):
    """Build ORV prompt using a specific syntax template."""
    return build_orv_prompt_with_positions(tokenizer, truth_table, query_object,
                                            query_relation, template=template_name, seed=seed)


# ============================================================================
# Step 7: Table-conditioned binding for ORV
# ============================================================================

def build_balanced_orv_conflicting_prompt(tokenizer, table_a, table_b, query_object,
                                           query_relation, table_context, order="A_first"):
    """Build ORV prompt with two conflicting rule tables, with controlled order."""
    rng = random.Random(42)
    rules_a = list(table_a.items())
    rules_b = list(table_b.items())
    rng.shuffle(rules_a)
    rng.shuffle(rules_b)

    lines_a = [f"{obj} {rel} {val}." for (obj, rel), val in rules_a]
    lines_b = [f"{obj} {rel} {val}." for (obj, rel), val in rules_b]

    if order == "A_first":
        prompt = "Table A:\n" + "\n".join(lines_a) + "\n\n"
        prompt += "Table B:\n" + "\n".join(lines_b) + "\n\n"
    else:
        prompt = "Table B:\n" + "\n".join(lines_b) + "\n\n"
        prompt += "Table A:\n" + "\n".join(lines_a) + "\n\n"

    prompt += f"Using Table {table_context}, {query_object} {query_relation} ?\nAnswer:"

    return prompt


def create_conflicting_orv_tables(truth_table, query_object, query_relation, values):
    """Create Table A (original) and Table B (conflicting value for query)."""
    table_a = truth_table.copy()
    correct_val = table_a.get((query_object, query_relation), None)
    if correct_val is None:
        return table_a, table_a.copy(), correct_val, correct_val

    rng = random.Random(456)
    other_vals = [v for v in values if v != correct_val]
    wrong_val = rng.choice(other_vals)

    table_b = truth_table.copy()
    table_b[(query_object, query_relation)] = wrong_val

    return table_a, table_b, correct_val, wrong_val


# ============================================================================
# Utility
# ============================================================================

def heads_to_groups(head_list):
    groups = {}
    for l, h in head_list:
        groups.setdefault(l, []).append(h)
    return groups


def create_corrupted_table_oc(truth_table, query_object, seed=42):
    rng = random.Random(seed)
    correct_cat = truth_table[query_object]
    other_cats = [c for c in set(truth_table.values()) if c != correct_cat]
    if not other_cats:
        return truth_table.copy()
    wrong_cat = rng.choice(other_cats)
    swap_obj = None
    for obj, cat in truth_table.items():
        if cat == wrong_cat and obj != query_object:
            swap_obj = obj
            break
    corrupted = truth_table.copy()
    if swap_obj:
        corrupted[query_object] = wrong_cat
        corrupted[swap_obj] = correct_cat
    return corrupted


# ============================================================================
# Main analysis
# ============================================================================

def run_model(args):
    global _safetensors_cache
    _safetensors_cache = {}

    model, tokenizer, device = load_model_eager(args.model)
    try:
        info = get_model_info(model, args.model)
        n_heads, d_head, d_model, n_kv_heads, kv_group_size = get_head_dims(model)
        n_layers = info.n_layers

        log(f"{args.model}: n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}, "
            f"d_model={d_model}, n_kv_heads={n_kv_heads}, kv_group_size={kv_group_size}")

        objects = CANDIDATE_OBJECTS[:8]
        categories = CANDIDATE_CATEGORIES[:4]
        relations = CANDIDATE_RELATIONS[:2]
        values = CANDIDATE_VALUES[:4]

        # FIX 1: Full-string token ID computation
        # For values: " v05" or "v05"
        value_token_seqs = {}
        for val in values:
            ids = tokenizer.encode(" " + val, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(val, add_special_tokens=False)
            value_token_seqs[val] = ids

        value_token_ids = {}
        for val, ids in value_token_seqs.items():
            value_token_ids[val] = ids[-1]  # Use last token (the distinguishing number)
        log(f"Value distinguishing tokens: {value_token_ids}")

        # For categories: " c12" or "c12"
        cat_token_seqs = {}
        for cat in categories:
            ids = tokenizer.encode(" " + cat, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(cat, add_special_tokens=False)
            cat_token_seqs[cat] = ids

        cat_token_ids = {}
        for cat, ids in cat_token_seqs.items():
            cat_token_ids[cat] = ids[-1]
        log(f"Category distinguishing tokens: {cat_token_ids}")

        n_tables = args.n_tables
        n_orv_samples = n_tables * min(len(objects), 4)  # Limit objects for speed

        result = {
            "phase": 580,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers, "n_heads": n_heads, "d_head": d_head,
            "d_model": d_model, "n_kv_heads": n_kv_heads, "kv_group_size": kv_group_size,
            "n_tables": n_tables,
            "value_token_ids": {k: int(v) for k, v in value_token_ids.items()},
            "cat_token_ids": {k: int(v) for k, v in cat_token_ids.items()},
        }

        # ================================================================
        # PART A: OC Task (fix Phase 579 issues)
        # ================================================================
        log("=" * 60)
        log("PART A: OC Task — Fix Phase 579 evaluation")
        log("=" * 60)

        oc_tables = build_oc_truth_tables(objects, categories, n_tables)
        log(f"Built {n_tables} OC truth tables")

        # Step A0: Identify top OC retrieval heads
        log("--- Step A0: Identify top OC retrieval heads ---")

        all_obj_scores = np.zeros((n_layers, n_heads))
        all_cat_scores = np.zeros((n_layers, n_heads))
        all_attn_mass_obj = np.zeros((n_layers, n_heads))
        all_attn_mass_cat = np.zeros((n_layers, n_heads))
        n_samples_scored = 0
        oc_sample_data = []

        total_oc_samples = n_tables * len(objects)
        for tt_idx, tt in enumerate(oc_tables):
            for obj in objects:
                prompt, full_ids, rule_info, answer_pos = build_oc_prompt_with_positions(
                    tokenizer, tt, obj, seed=tt_idx * 100 + hash(obj) % 1000)

                if not rule_info or answer_pos >= len(full_ids):
                    continue

                try:
                    attn_rows, logits = extract_attention_and_logits(
                        model, tokenizer, device, prompt, answer_pos)
                except Exception as e:
                    continue

                for l in range(n_layers):
                    if l in attn_rows:
                        scores = compute_oc_match_scores(attn_rows[l], rule_info, n_heads)
                        for h in range(n_heads):
                            if h in scores:
                                all_obj_scores[l, h] += scores[h]["obj_match"]
                                all_cat_scores[l, h] += scores[h]["cat_copy"]
                                all_attn_mass_obj[l, h] += scores[h]["attn_mass_obj"]
                                all_attn_mass_cat[l, h] += scores[h]["attn_mass_cat"]

                n_samples_scored += 1
                oc_sample_data.append({
                    "prompt": prompt, "rule_info": rule_info,
                    "answer_pos": answer_pos, "correct_cat": tt[obj],
                    "tt_idx": tt_idx, "obj": obj,
                })

                if n_samples_scored % 20 == 0:
                    log(f"  OC scored {n_samples_scored}/{total_oc_samples}")

        all_obj_scores /= max(1, n_samples_scored)
        all_cat_scores /= max(1, n_samples_scored)
        all_attn_mass_obj /= max(1, n_samples_scored)
        all_attn_mass_cat /= max(1, n_samples_scored)

        combined_scores = all_obj_scores + all_cat_scores
        flat_ranked = [(l, h, combined_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        flat_ranked.sort(key=lambda x: x[2], reverse=True)

        log(f"  Top-5 OC combined heads:")
        for l, h, s in flat_ranked[:5]:
            log(f"    L{l}H{h}: combined={s:.4f}, obj_match={all_obj_scores[l,h]:.4f}, "
                f"cat_copy={all_cat_scores[l,h]:.4f}, mass_cat={all_attn_mass_cat[l,h]:.6f}")

        cat_flat = [(l, h, all_cat_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        cat_flat.sort(key=lambda x: x[2], reverse=True)

        result["stepA0_top_heads"] = {
            "top_combined_50": [{"layer": l, "head": h, "combined": float(s),
                                "obj_match": float(all_obj_scores[l,h]),
                                "cat_copy": float(all_cat_scores[l,h]),
                                "mass_obj": float(all_attn_mass_obj[l,h]),
                                "mass_cat": float(all_attn_mass_cat[l,h])}
                               for l, h, s in flat_ranked[:50]],
        }

        # Step A1: TRUE edge ablation — DUAL MODE (renorm + no-renorm)
        log("--- Step A1: TRUE edge ablation (dual mode: renorm + no-renorm) ---")

        edge_samples = []
        for tt_idx in range(min(n_tables, 5)):
            for s in oc_sample_data:
                if s["tt_idx"] == tt_idx:
                    edge_samples.append(s)
                    break

        edge_configs = {
            "cat_edge_top10": heads_to_groups([(l, h) for l, h, _ in flat_ranked[:10]]),
            "cat_edge_top3_cat": heads_to_groups([(l, h) for l, h, _ in cat_flat[:3]]),
        }

        edge_results = {}

        for config_name, head_groups in edge_configs.items():
            for renorm_mode in [True, False]:
                mode_name = f"{config_name}_renorm" if renorm_mode else f"{config_name}_norenorm"
                log(f"  Edge config: {mode_name}")

                cat_logit_changes = []
                argmax_changes = 0
                n_tested = 0

                for i, s in enumerate(edge_samples):
                    correct_rule = None
                    for r in s["rule_info"]:
                        if r["is_correct"]:
                            correct_rule = r
                            break
                    if correct_rule is None:
                        continue

                    # FIX: Use full-string logprob scoring for baseline
                    baseline_lps = compute_full_string_logprob_batch(
                        model, tokenizer, device, s["prompt"], categories)
                    baseline_pred = max(baseline_lps, key=lambda c: baseline_lps[c][0])
                    baseline_correct_lp = baseline_lps[s["correct_cat"]][0]

                    # Edge ablation
                    target_positions = correct_rule["cat_positions"]
                    modified_logits = true_attn_edge_ablation_forward(
                        model, tokenizer, device, s["prompt"], s["answer_pos"],
                        target_positions, head_groups, renorm=renorm_mode)

                    # Score modified output using token-level comparison
                    cl = {cat: float(modified_logits[tid]) for cat, tid in cat_token_ids.items()}
                    ablated_pred = max(cl, key=cl.get)

                    logit_change = cl[s["correct_cat"]] - float(
                        extract_attention_and_logits(model, tokenizer, device, s["prompt"], s["answer_pos"])[1][cat_token_ids[s["correct_cat"]]])

                    cat_logit_changes.append(logit_change)
                    if ablated_pred != baseline_pred:
                        argmax_changes += 1

                    n_tested += 1
                    if i < 2:
                        log(f"    Sample {i}: correct={s['correct_cat']}, baseline={baseline_pred}, "
                            f"ablated={ablated_pred}, logit_change={logit_change:.4f}")

                edge_results[mode_name] = {
                    "n_tested": n_tested,
                    "mean_logit_change": float(np.mean(cat_logit_changes)) if cat_logit_changes else 0,
                    "argmax_change_rate": argmax_changes / max(1, n_tested),
                }
                r = edge_results[mode_name]
                log(f"    Summary: logit_change={r['mean_logit_change']:.4f}, "
                    f"argmax_change={r['argmax_change_rate']:.3f}")

        result["stepA1_true_edge_ablation_dual"] = edge_results

        # Step A2: Full-string logprob baseline (FIX)
        log("--- Step A2: Full-string logprob baseline (OC) ---")

        full_string_results = []
        for i, s in enumerate(oc_sample_data[:min(len(oc_sample_data), 20)]):
            lps = compute_full_string_logprob_batch(
                model, tokenizer, device, s["prompt"], categories)
            pred = max(lps, key=lambda c: lps[c][0])
            correct = pred == s["correct_cat"]
            full_string_results.append({
                "predicted": pred, "correct": s["correct_cat"],
                "is_correct": correct,
                "logprobs": {c: lp for c, (lp, _) in lps.items()},
            })

        fs_accuracy = sum(1 for r in full_string_results if r["is_correct"]) / max(1, len(full_string_results))
        log(f"  Full-string logprob accuracy (OC): {fs_accuracy:.3f} ({len(full_string_results)} samples)")

        result["stepA2_full_string_baseline"] = {
            "accuracy": fs_accuracy,
            "n_samples": len(full_string_results),
            "results": full_string_results,
        }

        # Step A3: OC value attribution (FIX with GQA)
        log("--- Step A3: OC value attribution (GQA-fixed) ---")

        top_5_combined = heads_to_groups([(l, h) for l, h, _ in flat_ranked[:5]])
        all_oc_attributions = []

        for i, s in enumerate(oc_sample_data[:5]):
            try:
                attrs = compute_value_logit_attribution_oc(
                    model, tokenizer, device, s["prompt"], s["answer_pos"],
                    s["rule_info"], top_5_combined, cat_token_ids, d_head)
                all_oc_attributions.extend(attrs)
            except Exception as e:
                log(f"  WARNING: OC attribution failed for sample {i}: {e}")

        # Compute advantage
        head_advantages = {}
        for attr in all_oc_attributions:
            if attr.get("type") == "oc_position_attribution":
                key = f"L{attr['layer']}H{attr['head']}"
                head_advantages[key] = {
                    "cat_direct": attr["cat_direct_logit_from_cat"],
                    "obj_direct": attr["cat_direct_logit_from_obj"],
                    "attn_cat": attr["attn_to_correct_cat"],
                    "attn_obj": attr["attn_to_correct_obj"],
                    "correct_category": attr["correct_category"],
                }

        result["stepA3_oc_value_attribution"] = {
            "head_advantages": head_advantages,
            "n_attributions": len(all_oc_attributions),
        }

        if head_advantages:
            log(f"  OC value attribution (top heads):")
            for key, vals in sorted(head_advantages.items()):
                log(f"    {key}: cat_direct={vals['cat_direct']:.4f}, "
                    f"obj_direct={vals['obj_direct']:.4f}, "
                    f"attn_cat={vals['attn_cat']:.4f}")

        # ================================================================
        # PART B: ORV Task (new)
        # ================================================================
        log("=" * 60)
        log("PART B: ORV Task — Object-Relation-Value retrieval")
        log("=" * 60)

        orv_tables = build_orv_truth_tables(objects, relations, values, n_tables)
        log(f"Built {n_tables} ORV truth tables")

        # Step B0: Identify top ORV retrieval heads
        log("--- Step B0: Identify top ORV retrieval heads ---")

        all_orv_obj_scores = np.zeros((n_layers, n_heads))
        all_orv_rel_scores = np.zeros((n_layers, n_heads))
        all_orv_val_scores = np.zeros((n_layers, n_heads))
        all_orv_mass_obj = np.zeros((n_layers, n_heads))
        all_orv_mass_rel = np.zeros((n_layers, n_heads))
        all_orv_mass_val = np.zeros((n_layers, n_heads))
        n_orv_scored = 0
        orv_sample_data = []

        total_orv = n_tables * len(objects) * len(relations)
        for tt_idx, tt in enumerate(orv_tables):
            for obj in objects[:4]:  # Limit for speed
                for rel in relations:
                    prompt, full_ids, rule_info, answer_pos = build_orv_prompt_with_positions(
                        tokenizer, tt, obj, rel, template="minimal",
                        seed=tt_idx * 100 + hash(obj) % 1000 + hash(rel) % 100)

                    if not rule_info or answer_pos >= len(full_ids):
                        continue

                    try:
                        attn_rows, logits = extract_attention_and_logits(
                            model, tokenizer, device, prompt, answer_pos)
                    except Exception as e:
                        continue

                    for l in range(n_layers):
                        if l in attn_rows:
                            scores = compute_orv_match_scores(attn_rows[l], rule_info, n_heads)
                            for h in range(n_heads):
                                if h in scores:
                                    all_orv_obj_scores[l, h] += scores[h]["obj_match"]
                                    all_orv_rel_scores[l, h] += scores[h]["rel_match"]
                                    all_orv_val_scores[l, h] += scores[h]["val_copy"]
                                    all_orv_mass_obj[l, h] += scores[h]["attn_mass_obj"]
                                    all_orv_mass_rel[l, h] += scores[h]["attn_mass_rel"]
                                    all_orv_mass_val[l, h] += scores[h]["attn_mass_val"]

                    n_orv_scored += 1
                    orv_sample_data.append({
                        "prompt": prompt, "rule_info": rule_info,
                        "answer_pos": answer_pos,
                        "correct_val": tt.get((obj, rel), None),
                        "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    })

                    if n_orv_scored % 16 == 0:
                        log(f"  ORV scored {n_orv_scored}/{total_orv}")

        all_orv_obj_scores /= max(1, n_orv_scored)
        all_orv_rel_scores /= max(1, n_orv_scored)
        all_orv_val_scores /= max(1, n_orv_scored)
        all_orv_mass_obj /= max(1, n_orv_scored)
        all_orv_mass_rel /= max(1, n_orv_scored)
        all_orv_mass_val /= max(1, n_orv_scored)

        orv_combined = all_orv_obj_scores + all_orv_rel_scores + all_orv_val_scores
        orv_flat_ranked = [(l, h, orv_combined[l, h]) for l in range(n_layers) for h in range(n_heads)]
        orv_flat_ranked.sort(key=lambda x: x[2], reverse=True)

        val_flat = [(l, h, all_orv_val_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        val_flat.sort(key=lambda x: x[2], reverse=True)

        rel_flat = [(l, h, all_orv_rel_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        rel_flat.sort(key=lambda x: x[2], reverse=True)

        log(f"  Top-5 ORV combined heads:")
        for l, h, s in orv_flat_ranked[:5]:
            log(f"    L{l}H{h}: combined={s:.4f}, obj_match={all_orv_obj_scores[l,h]:.4f}, "
                f"rel_match={all_orv_rel_scores[l,h]:.4f}, val_copy={all_orv_val_scores[l,h]:.4f}, "
                f"mass_val={all_orv_mass_val[l,h]:.6f}")

        result["stepB0_orv_top_heads"] = {
            "top_combined_20": [{"layer": l, "head": h, "combined": float(s),
                                  "obj_match": float(all_orv_obj_scores[l,h]),
                                  "rel_match": float(all_orv_rel_scores[l,h]),
                                  "val_copy": float(all_orv_val_scores[l,h]),
                                  "mass_obj": float(all_orv_mass_obj[l,h]),
                                  "mass_rel": float(all_orv_mass_rel[l,h]),
                                  "mass_val": float(all_orv_mass_val[l,h])}
                                 for l, h, s in orv_flat_ranked[:20]],
        }

        # Step B1: ORV full-string logprob baseline
        log("--- Step B1: Full-string logprob baseline (ORV) ---")

        orv_fs_results = []
        for i, s in enumerate(orv_sample_data[:min(len(orv_sample_data), 20)]):
            if s["correct_val"] is None:
                continue
            lps = compute_full_string_logprob_batch(
                model, tokenizer, device, s["prompt"], values)
            pred = max(lps, key=lambda c: lps[c][0])
            correct = pred == s["correct_val"]
            orv_fs_results.append({
                "predicted": pred, "correct": s["correct_val"],
                "is_correct": correct,
                "logprobs": {c: lp for c, (lp, _) in lps.items()},
            })

        orv_accuracy = sum(1 for r in orv_fs_results if r["is_correct"]) / max(1, len(orv_fs_results))
        log(f"  Full-string logprob accuracy (ORV): {orv_accuracy:.3f} ({len(orv_fs_results)} samples)")

        result["stepB1_orv_baseline"] = {
            "accuracy": orv_accuracy,
            "n_samples": len(orv_fs_results),
        }

        # Step B2: ORV value-copy attribution
        log("--- Step B2: ORV value-copy attribution ---")

        top_5_orv_combined = heads_to_groups([(l, h) for l, h, _ in orv_flat_ranked[:5]])
        all_orv_attributions = []

        for i, s in enumerate(orv_sample_data[:5]):
            if s["correct_val"] is None:
                continue
            try:
                attrs = compute_value_logit_attribution_orv(
                    model, tokenizer, device, s["prompt"], s["answer_pos"],
                    s["rule_info"], top_5_orv_combined, value_token_ids, d_head)
                all_orv_attributions.extend(attrs)
            except Exception as e:
                log(f"  WARNING: ORV attribution failed for sample {i}: {e}")

        orv_head_advantages = {}
        for attr in all_orv_attributions:
            if attr.get("type") == "position_attribution":
                key = f"L{attr['layer']}H{attr['head']}"
                orv_head_advantages[key] = {
                    "val_direct_from_val": attr["val_direct_logit_from_val"],
                    "val_direct_from_obj": attr["val_direct_logit_from_obj"],
                    "val_direct_from_rel": attr["val_direct_logit_from_rel"],
                    "attn_val": attr["attn_to_correct_val"],
                    "attn_obj": attr["attn_to_correct_obj"],
                    "attn_rel": attr["attn_to_correct_rel"],
                    "correct_value": attr["correct_value"],
                }

        # Compute advantage for ORV: correct value vs wrong values
        orv_advantages = []
        for attr in all_orv_attributions:
            if "value" in attr and "type" not in attr:
                key = f"L{attr['layer']}H{attr['head']}"
                # We'll aggregate by head below

        # Aggregate by head
        head_val_logits = {}
        for attr in all_orv_attributions:
            if "value" in attr and "type" not in attr:
                key = f"L{attr['layer']}H{attr['head']}"
                if key not in head_val_logits:
                    head_val_logits[key] = {}
                head_val_logits[key][attr["value"]] = attr["logit_contribution"]

        orv_advantage_list = []
        for key, vals in head_val_logits.items():
            # Find correct value from position_attribution
            if key in orv_head_advantages:
                correct_val = orv_head_advantages[key]["correct_value"]
                if correct_val in vals:
                    correct_logit = vals[correct_val]
                    wrong_logits = [v for k, v in vals.items() if k != correct_val]
                    if wrong_logits:
                        advantage = correct_logit - np.mean(wrong_logits)
                        orv_advantage_list.append({"head": key, "advantage": advantage})

        mean_orv_advantage = float(np.mean([a["advantage"] for a in orv_advantage_list])) if orv_advantage_list else 0
        log(f"  ORV mean value advantage: {mean_orv_advantage:.4f}")

        result["stepB2_orv_value_attribution"] = {
            "head_advantages": orv_head_advantages,
            "orv_advantage_list": orv_advantage_list,
            "mean_advantage": mean_orv_advantage,
        }

        # Step B3: ORV token corruption (value/obj/rel)
        log("--- Step B3: ORV token corruption ---")

        corruption_samples = []
        for tt_idx in range(min(n_tables, 5)):
            for s in orv_sample_data:
                if s["tt_idx"] == tt_idx and s["correct_val"] is not None:
                    corruption_samples.append(s)
                    break

        corruption_configs = ["val_only", "obj_only", "rel_only", "all_rules"]
        corruption_results = {}

        for config in corruption_configs:
            log(f"  Corruption config: {config}")
            n_correct_baseline = 0
            n_correct_corrupt = 0
            n_tested = 0

            for s in corruption_samples:
                correct_rule = None
                for r in s["rule_info"]:
                    if r["is_correct"]:
                        correct_rule = r
                        break
                if correct_rule is None:
                    continue

                # Baseline: generate from clean prompt
                input_ids = torch.tensor(
                    [tokenizer.encode(s["prompt"], add_special_tokens=False)], device=device)
                with torch.inference_mode():
                    gen = model.generate(input_ids, max_new_tokens=8, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                detected = [v for v in values if v.lower() in text.lower()]
                baseline_pred = detected[0] if len(detected) == 1 else "wrong"
                if baseline_pred == s["correct_val"]:
                    n_correct_baseline += 1

                # Corrupted prompt
                corrupt_prompt = s["prompt"]
                corrupt_ids = tokenizer.encode(corrupt_prompt, add_special_tokens=False)

                if config == "val_only":
                    for p in correct_rule["val_positions"]:
                        if p < len(corrupt_ids):
                            corrupt_ids[p] = tokenizer.encode(".", add_special_tokens=False)[0]
                elif config == "obj_only":
                    for p in correct_rule["obj_positions"]:
                        if p < len(corrupt_ids):
                            corrupt_ids[p] = tokenizer.encode(".", add_special_tokens=False)[0]
                elif config == "rel_only":
                    for p in correct_rule["rel_positions"]:
                        if p < len(corrupt_ids):
                            corrupt_ids[p] = tokenizer.encode(".", add_special_tokens=False)[0]
                elif config == "all_rules":
                    for r in s["rule_info"]:
                        for p in r["val_positions"] + r["obj_positions"] + r["rel_positions"]:
                            if p < len(corrupt_ids):
                                corrupt_ids[p] = tokenizer.encode(".", add_special_tokens=False)[0]

                corrupt_input = torch.tensor([corrupt_ids], device=device)
                with torch.inference_mode():
                    gen = model.generate(corrupt_input, max_new_tokens=8, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id)
                text = tokenizer.decode(gen[0][corrupt_input.shape[1]:], skip_special_tokens=True).strip()
                detected = [v for v in values if v.lower() in text.lower()]
                corrupt_pred = detected[0] if len(detected) == 1 else "wrong"
                if corrupt_pred == s["correct_val"]:
                    n_correct_corrupt += 1

                n_tested += 1

            corruption_results[config] = {
                "n_tested": n_tested,
                "baseline_correct": n_correct_baseline,
                "corrupt_correct": n_correct_corrupt,
                "corruption_effect": (n_correct_baseline - n_correct_corrupt) / max(1, n_tested),
            }
            log(f"    baseline={n_correct_baseline}/{n_tested}, corrupt={n_correct_corrupt}/{n_tested}")

        result["stepB3_orv_token_corruption"] = corruption_results

        # Step B4: ORV true edge ablation (dual mode)
        log("--- Step B4: ORV true edge ablation (dual mode) ---")

        orv_edge_configs = {
            "val_edge_top10": heads_to_groups([(l, h) for l, h, _ in orv_flat_ranked[:10]]),
            "val_edge_top3_val": heads_to_groups([(l, h) for l, h, _ in val_flat[:3]]),
        }

        orv_edge_results = {}

        for config_name, head_groups in orv_edge_configs.items():
            for renorm_mode in [True, False]:
                mode_name = f"{config_name}_renorm" if renorm_mode else f"{config_name}_norenorm"
                log(f"  ORV edge config: {mode_name}")

                argmax_changes = 0
                n_tested = 0

                for i, s in enumerate(corruption_samples):
                    correct_rule = None
                    for r in s["rule_info"]:
                        if r["is_correct"]:
                            correct_rule = r
                            break
                    if correct_rule is None or s["correct_val"] is None:
                        continue

                    # Baseline: use full-string logprob
                    baseline_lps = compute_full_string_logprob_batch(
                        model, tokenizer, device, s["prompt"], values)
                    baseline_pred = max(baseline_lps, key=lambda c: baseline_lps[c][0])

                    # Edge ablation
                    target_positions = correct_rule["val_positions"]
                    modified_logits = true_attn_edge_ablation_forward(
                        model, tokenizer, device, s["prompt"], s["answer_pos"],
                        target_positions, head_groups, renorm=renorm_mode)

                    cl = {v: float(modified_logits[tid]) for v, tid in value_token_ids.items()}
                    ablated_pred = max(cl, key=cl.get)

                    if ablated_pred != baseline_pred:
                        argmax_changes += 1
                    n_tested += 1

                orv_edge_results[mode_name] = {
                    "n_tested": n_tested,
                    "argmax_change_rate": argmax_changes / max(1, n_tested),
                }
                log(f"    Summary: argmax_change={orv_edge_results[mode_name]['argmax_change_rate']:.3f}")

        result["stepB4_orv_edge_ablation"] = orv_edge_results

        # ================================================================
        # PART C: Compositional reasoning (two-hop)
        # ================================================================
        log("=" * 60)
        log("PART C: Compositional reasoning (two-hop)")
        log("=" * 60)

        n_comp_samples = min(n_tables, 5)
        comp_results = []

        for tt_idx in range(n_comp_samples):
            oc_tt = oc_tables[tt_idx]
            orv_tt = orv_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]
            rel = relations[tt_idx % len(relations)]

            prompt, correct_cat, correct_val = build_compositional_prompt(
                tokenizer, oc_tt, orv_tt, obj, rel, seed=tt_idx * 300)

            if correct_val is None:
                continue

            # Generate answer
            input_ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False)], device=device)
            with torch.inference_mode():
                gen = model.generate(input_ids, max_new_tokens=15, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

            # Check if the model generated the correct value
            detected_vals = [v for v in values if v.lower() in text.lower()]
            pred_val = detected_vals[0] if len(detected_vals) == 1 else "wrong"

            # Also check intermediate category
            detected_cats = [c for c in categories if c.lower() in text.lower()]
            pred_cat = detected_cats[0] if len(detected_cats) == 1 else "wrong"

            # Full-string logprob scoring for values
            val_lps = compute_full_string_logprob_batch(
                model, tokenizer, device, prompt, values)
            logprob_pred = max(val_lps, key=lambda c: val_lps[c][0])

            comp_results.append({
                "tt_idx": tt_idx, "obj": obj, "rel": rel,
                "correct_cat": correct_cat, "correct_val": correct_val,
                "generated_text": text[:100],
                "pred_val": pred_val, "pred_cat": pred_cat,
                "logprob_pred_val": logprob_pred,
                "val_correct": pred_val == correct_val,
                "cat_correct": pred_cat == correct_cat,
                "logprob_correct": logprob_pred == correct_val,
            })

            log(f"  Sample {tt_idx}: obj={obj}, rel={rel}, correct_cat={correct_cat}, "
                f"correct_val={correct_val}, gen={text[:50]}, logprob_pred={logprob_pred}")

        val_correct_count = sum(1 for r in comp_results if r["val_correct"])
        logprob_correct_count = sum(1 for r in comp_results if r["logprob_correct"])
        cat_correct_count = sum(1 for r in comp_results if r["cat_correct"])

        log(f"  Compositional: val_gen_correct={val_correct_count}/{len(comp_results)}, "
            f"val_logprob_correct={logprob_correct_count}/{len(comp_results)}, "
            f"cat_correct={cat_correct_count}/{len(comp_results)}")

        result["stepC_compositional"] = {
            "n_samples": len(comp_results),
            "val_gen_accuracy": val_correct_count / max(1, len(comp_results)),
            "val_logprob_accuracy": logprob_correct_count / max(1, len(comp_results)),
            "cat_accuracy": cat_correct_count / max(1, len(comp_results)),
            "details": comp_results,
        }

        # ================================================================
        # PART D: Syntax template variation
        # ================================================================
        log("=" * 60)
        log("PART D: Syntax template variation")
        log("=" * 60)

        syntax_results = []
        template_names = list(SYNTAX_TEMPLATES.keys())

        for tt_idx in range(min(n_tables, 3)):
            orv_tt = orv_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]
            rel = relations[tt_idx % len(relations)]
            correct_val = orv_tt.get((obj, rel), None)
            if correct_val is None:
                continue

            for tmpl_name in template_names:
                prompt, full_ids, rule_info, answer_pos = build_orv_prompt_with_template(
                    tokenizer, orv_tt, obj, rel, tmpl_name, seed=tt_idx * 400)

                # Full-string logprob
                lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, prompt, values)
                pred = max(lps, key=lambda c: lps[c][0])
                correct = pred == correct_val

                syntax_results.append({
                    "tt_idx": tt_idx, "template": tmpl_name,
                    "predicted": pred, "correct": correct_val,
                    "is_correct": correct,
                    "correct_logprob": lps[correct_val][0],
                })

                log(f"  tt={tt_idx}, template={tmpl_name}: pred={pred}, "
                    f"correct={correct_val}, match={correct}")

        # Accuracy by template
        for tmpl_name in template_names:
            subset = [r for r in syntax_results if r["template"] == tmpl_name]
            acc = sum(1 for r in subset if r["is_correct"]) / max(1, len(subset))
            log(f"  Template {tmpl_name}: accuracy={acc:.3f} ({len(subset)} samples)")

        result["stepD_syntax_variation"] = {
            "results": syntax_results,
            "by_template": {
                tmpl: {
                    "accuracy": sum(1 for r in syntax_results if r["template"] == tmpl and r["is_correct"])
                              / max(1, sum(1 for r in syntax_results if r["template"] == tmpl)),
                    "n_samples": sum(1 for r in syntax_results if r["template"] == tmpl),
                }
                for tmpl in template_names
            },
        }

        # ================================================================
        # PART E: Balanced table binding for ORV
        # ================================================================
        log("=" * 60)
        log("PART E: Balanced table binding for ORV")
        log("=" * 60)

        binding_results = []

        for tt_idx in range(min(n_tables, 5)):
            orv_tt = orv_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]
            rel = relations[tt_idx % len(relations)]

            table_a, table_b, correct_val_a, wrong_val_b = create_conflicting_orv_tables(
                orv_tt, obj, rel, values)

            for order in ["A_first", "B_first"]:
                for context in ["A", "B"]:
                    expected_val = correct_val_a if context == "A" else wrong_val_b

                    prompt = build_balanced_orv_conflicting_prompt(
                        tokenizer, table_a, table_b, obj, rel,
                        table_context=context, order=order)

                    input_ids = torch.tensor(
                        [tokenizer.encode(prompt, add_special_tokens=False)], device=device)

                    with torch.inference_mode():
                        gen = model.generate(input_ids, max_new_tokens=10, do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id)

                    text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

                    detected = [v for v in values if v.lower() in text.lower()]
                    pred_val = detected[0] if len(detected) == 1 else ("none" if len(detected) == 0 else "ambiguous")

                    correct = pred_val == expected_val

                    binding_results.append({
                        "tt_idx": tt_idx, "obj": obj, "rel": rel,
                        "order": order, "context": context,
                        "expected_val": expected_val, "predicted_val": pred_val,
                        "correct": correct, "text": text[:100],
                    })

                    log(f"  tt={tt_idx}, order={order}, context={context}: "
                        f"expected={expected_val}, predicted={pred_val}, correct={correct}")

        # Recency bias analysis
        a_near = [b for b in binding_results if b["context"] == "A" and b["order"] == "A_first"]
        a_far = [b for b in binding_results if b["context"] == "A" and b["order"] == "B_first"]
        b_near = [b for b in binding_results if b["context"] == "B" and b["order"] == "B_first"]
        b_far = [b for b in binding_results if b["context"] == "B" and b["order"] == "A_first"]

        a_near_acc = sum(1 for b in a_near if b["correct"]) / max(1, len(a_near))
        a_far_acc = sum(1 for b in a_far if b["correct"]) / max(1, len(a_far))
        b_near_acc = sum(1 for b in b_near if b["correct"]) / max(1, len(b_near))
        b_far_acc = sum(1 for b in b_far if b["correct"]) / max(1, len(b_far))

        log(f"  ORV Recency bias analysis:")
        log(f"    A_context, A_near: {a_near_acc:.3f}")
        log(f"    A_context, A_far:  {a_far_acc:.3f}")
        log(f"    B_context, B_near: {b_near_acc:.3f}")
        log(f"    B_context, B_far:  {b_far_acc:.3f}")

        result["stepE_orv_binding"] = {
            "tests": binding_results,
            "a_near_acc": a_near_acc,
            "a_far_acc": a_far_acc,
            "b_near_acc": b_near_acc,
            "b_far_acc": b_far_acc,
            "recency_bias": a_near_acc - a_far_acc,
        }

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
    out_path = out_dir / f"phase580_{args.model}_orv_closure{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
