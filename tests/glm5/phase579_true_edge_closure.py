#!/usr/bin/env python3
"""
Phase 579: True Retrieval Edge Closure and Table-Binding Control
真检索边闭包与表绑定控制

Phase 578 proved:
- Token-level category corruption is causally effective (70-100% argmax change)
- mass_cat >> mass_obj (~100:1 for Qwen3)
- Qwen3 retrieval circuit highly redundant (top50 heads still 100% correct)
- B_context easier than A_context (recency bias)

Phase 579 completes the closure:
Step 1: TRUE attention edge ablation — modify A[l,h][q,j*] directly, not tokens
Step 2: Attention pattern patching — inject clean A patterns into corrupt run
Step 3: Value vector logit attribution — use get_W_U from safetensors for meta device
Step 4: Balanced table binding — test A-before-B AND B-before-A to remove position bias
Step 5: Retrieval-to-state transition — use get_W_U from safetensors for meta device

Requires eager attention (output_attentions=True).
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
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U  # noqa: E402

OUT_ROOT = Path("results/glm5_phase579_true_edge_closure")

CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Model loading with eager attention
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
# Truth table and prompt construction (same as Phase 577/578)
# ============================================================================

def build_truth_tables(objects, categories, n_tables, seed=42):
    rng = random.Random(seed)
    n_cats = len(categories)
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
        tokenizer.encode("." + symbol, add_special_tokens=False),
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


def build_prompt_with_positions(tokenizer, truth_table, query_object, seed=42):
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
            search_start = cat_end if 'cat_end' in dir() else obj_end
            continue
        cat_pos, cat_end = cat_result

        rule_info.append({
            "object": obj,
            "category": cat,
            "obj_positions": list(range(obj_pos, obj_end)),
            "cat_positions": list(range(cat_pos, cat_end)),
            "is_correct": obj == query_object,
        })
        search_start = cat_end

    # Find query object position (last occurrence)
    all_q = []
    search = 0
    while True:
        r = find_symbol_in_full(tokenizer, full_ids, query_object, search)
        if r is None:
            break
        all_q.append(r)
        search = r[1]
    query_obj_positions = None
    if all_q:
        q_pos, q_end = all_q[-1]
        query_obj_positions = list(range(q_pos, q_end))

    answer_pos = len(full_ids) - 1

    return prompt, full_ids, rule_info, query_obj_positions, answer_pos


# ============================================================================
# Step 0: Reuse Phase 577's attention extraction to identify top heads
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


def compute_rule_match_scores(attn_row, rule_info, n_heads):
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

        total_obj = attn_correct_obj + attn_wrong_obj
        total_cat = attn_correct_cat + attn_wrong_cat

        obj_match = attn_correct_obj / (total_obj + 1e-10)
        cat_copy = attn_correct_cat / (total_cat + 1e-10)

        scores[h] = {
            "obj_match": float(obj_match),
            "cat_copy": float(cat_copy),
            "attn_mass_obj": attn_correct_obj,
            "attn_mass_cat": attn_correct_cat,
        }
    return scores


def get_head_dims(model):
    """Get head dimensions, properly handling GQA (Grouped-Query Attention).

    In GQA models:
    - num_attention_heads = number of query heads
    - num_key_value_heads = number of KV heads (fewer than query heads)
    - Each query head group shares one KV head
    - d_head = head_dim (from config) or v_proj.shape[0] // num_key_value_heads
    - v_proj shape: [num_key_value_heads * d_head, d_model]
    - o_proj shape: [d_model, d_model] (all query heads output concatenated)
    """
    config = model.config
    n_heads = config.num_attention_heads
    d_model = config.hidden_size

    # Get actual d_head — use config.head_dim if available
    if hasattr(config, 'head_dim') and config.head_dim is not None:
        d_head = config.head_dim
    else:
        # Fallback: infer from v_proj weight shape
        n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
        layer0 = get_layers(model)[0]
        v_proj_shape = layer0.self_attn.v_proj.weight.shape
        d_head = v_proj_shape[0] // n_kv_heads

    n_kv_heads = getattr(config, 'num_key_value_heads', n_heads)
    kv_group_size = n_heads // n_kv_heads  # how many query heads share one KV head

    return n_heads, d_head, d_model, n_kv_heads, kv_group_size


# ============================================================================
# Step 1: TRUE attention edge ablation — modify A matrix directly
# ============================================================================

def true_attn_edge_ablation_forward(model, tokenizer, device, prompt, answer_pos,
                                     target_positions, target_layers_heads):
    """Forward pass with TRUE attention edge ablation.

    For each specified (layer, head), we zero the attention weight from
    answer_pos to each position in target_positions, then renormalize.

    This requires:
    1. First forward: get attention weights and value vectors
    2. Modify A[l,h][answer_pos, target_positions] = 0
    3. Renormalize: A' = A / sum(A) (excluding zeroed positions)
    4. Recompute head output: attn_output = modified_A @ V

    Implementation: we use a custom forward_pre_hook on the attention module
    to intercept and modify the attention computation.

    Actually, for eager attention in HuggingFace, the simplest approach is:
    - Run normal forward with output_attentions=True
    - For each target (l,h), modify the attention weights at [answer_pos, target_positions]
    - Renormalize
    - Recompute attn_output = modified_weights @ value_vectors
    - Subtract the original head output and add the modified one

    This requires access to value vectors, which we get from hidden_states + W_V.
    """
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    # Forward with attention + hidden states
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)

    # Get all data
    attn_weights_all = outputs.attentions  # tuple of [1, n_heads, seq, seq]
    hidden_states_all = outputs.hidden_states  # tuple of [1, seq, d_model]
    base_logits = outputs.logits[0, answer_pos, :].detach().float().cpu().numpy()

    layers_list = get_layers(model)
    n_heads_cfg = model.config.num_attention_heads
    d_model_cfg = model.config.hidden_size

    # Get actual d_head and GQA info
    _, d_head, _, n_kv_heads, kv_group_size = get_head_dims(model)

    # We need to modify the logits by subtracting original head contributions
    # and adding modified ones, for each target (l, h).

    # For each (l, h) in target_layers_heads:
    #   original: logits += W_U @ W_O_h @ (sum_j A[q,j] * W_V_h @ h_input[j])
    #   modified: logits += W_U @ W_O_h @ (sum_j A'[q,j] * W_V_h @ h_input[j])
    #   delta: logits += W_U @ W_O_h @ (sum_j (A'[q,j] - A[q,j]) * W_V_h @ h_input[j])

    # Get W_U using model_utils.get_W_U (handles meta device via safetensors)
    W_U = get_W_U(model, model_name=None)  # [vocab, d_model]

    modified_logits = base_logits.copy()

    for (l, h_list) in target_layers_heads.items():
        if l >= len(attn_weights_all) or attn_weights_all[l] is None:
            continue

        for h in h_list:
            # Get attention weights for this head at answer position: [seq]
            attn = attn_weights_all[l][0, h, answer_pos, :].detach().float().cpu()

            # Get hidden states input to this layer: [seq, d_model]
            h_input = hidden_states_all[l][0, :, :].detach().float().cpu()

            # GQA: query head h maps to KV head kv_h = h // kv_group_size
            kv_h = h // kv_group_size

            # Get W_V and W_O for this layer (handle meta device via safetensors)
            layer_obj = layers_list[l]
            W_V_weight = layer_obj.self_attn.v_proj.weight
            W_O_weight = layer_obj.self_attn.o_proj.weight

            if W_V_weight.is_meta or W_O_weight.is_meta:
                W_V_np, W_O_np = _load_attn_weights_from_safetensors(model, l)
                # GQA: W_V shape is [n_kv_heads * d_head, d_model]
                W_V_h = W_V_np[kv_h * d_head:(kv_h + 1) * d_head, :]  # [d_head, d_model]
                # W_O shape is [d_model, d_model] — slice by query head
                W_O_h = W_O_np[:, h * d_head:(h + 1) * d_head]   # [d_model, d_head]
            else:
                W_V_h = W_V_weight[kv_h * d_head:(kv_h + 1) * d_head, :].detach().float().cpu().numpy()
                W_O_h = W_O_weight[:, h * d_head:(h + 1) * d_head].detach().float().cpu().numpy()

            h_input_np = h_input.numpy()

            # Compute value vectors: V_j = W_V_h @ h_input[j] -> [d_head] for each j
            V_all = (W_V_h @ h_input_np.T).T  # [seq, d_head]

            # Original head output at answer_pos: sum_j A[q,j] * W_O_h @ V[j]
            # = W_O_h @ sum_j(A[q,j] * V[j])
            attn_np = attn.numpy()

            # Sanity check: dimensions must match
            if attn_np.shape[0] != V_all.shape[0]:
                # Attention seq_len differs from hidden_states seq_len
                # This can happen if answer_pos is out of range
                # Skip this head
                continue

            if V_all.shape[1] != W_O_h.shape[1]:
                log(f"  WARNING: shape mismatch at L{l}H{h}: V_all.shape[1]={V_all.shape[1]}, W_O_h.shape[1]={W_O_h.shape[1]}")
                continue

            original_weighted_V = attn_np[:, np.newaxis] * V_all  # [seq, d_head]
            original_head_out = W_O_h @ original_weighted_V.sum(axis=0)  # [d_model]

            # Modified attention: zero target_positions, renormalize
            modified_attn = attn_np.copy()
            for pos in target_positions:
                if pos < len(modified_attn):
                    modified_attn[pos] = 0.0

            # Renormalize
            sum_modified = modified_attn.sum()
            if sum_modified > 1e-10:
                modified_attn = modified_attn / sum_modified
            else:
                # All attention was on target positions — head output becomes zero
                modified_head_out = np.zeros_like(original_head_out)
                # Apply delta to logits
                for y_idx in range(len(modified_logits)):
                    delta = float(W_U[y_idx] @ (modified_head_out - original_head_out))
                    modified_logits[y_idx] += delta
                continue

            modified_weighted_V = modified_attn[:, np.newaxis] * V_all  # [seq, d_head]
            modified_head_out = W_O_h @ modified_weighted_V.sum(axis=0)  # [d_model]

            # Apply delta to logits: W_U[y] @ (modified_head_out - original_head_out)
            delta_head = modified_head_out - original_head_out  # [d_model]
            delta_logits = W_U @ delta_head  # [vocab]
            modified_logits += delta_logits

    return modified_logits


def _load_attn_weights_from_safetensors(model, layer_idx):
    """Load W_V and W_O from safetensors for a specific layer (handles meta device)."""
    # Try to find model path from MODEL_CONFIGS or model config
    model_path = None

    # Try from model config's _name_or_path (but it may be a HF repo ID, not local path)
    if hasattr(model, 'config'):
        name_or_path = getattr(model.config, '_name_or_path', None)
        if name_or_path and os.path.isdir(name_or_path):
            model_path = name_or_path

    # Try from MODEL_CONFIGS using the GPT5_TOOLS module
    if model_path is None:
        from model_registry import get_model_spec, MODEL_SPECS
        # Check which model matches by checking config attributes
        model_class = type(model).__name__
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
                return W_V, W_O

    raise ValueError(f"Cannot find layer {layer_idx} attention weights in safetensors at {model_path}")


# ============================================================================
# Step 2: Attention pattern patching — inject clean A into corrupt
# ============================================================================

def attn_pattern_patching_forward(model, tokenizer, device, corrupt_prompt, clean_prompt,
                                  corrupt_answer_pos, clean_answer_pos,
                                  target_layers_heads, rule_info_corrupt, rule_info_clean):
    """Patch attention patterns: inject clean attention weights into corrupt forward.

    For each (l, h) in target_layers_heads:
    - Get clean attention pattern A_clean[l,h][clean_answer_pos, :]
    - In corrupt forward, replace A_corrupt[l,h][corrupt_answer_pos, :] with A_clean pattern
      (mapped to corrupt positions)
    - Renormalize and recompute head output

    Since clean and corrupt have different prompts (different seq lengths, different rules),
    we can't directly copy the attention pattern. Instead, we do POSITION-LEVEL patching:
    - Copy attention to correct category positions from clean to corrupt
    - This means: for the corrupt query position, make it attend to the same relative
      category positions as in the clean prompt.

    Simpler practical approach:
    1. Run clean forward with attention output -> get A_clean[l,h][answer_pos, correct_cat_pos]
    2. Run corrupt forward with attention output -> get A_corrupt[l,h][answer_pos, :]
    3. Modify: increase A_corrupt at correct category positions, decrease at wrong positions
    4. Recompute head outputs and logits
    """
    # Clean forward
    clean_input_ids = torch.tensor([tokenizer.encode(clean_prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        clean_outputs = model(input_ids=clean_input_ids, output_attentions=True,
                              output_hidden_states=True, return_dict=True)

    # Corrupt forward
    corrupt_input_ids = torch.tensor([tokenizer.encode(corrupt_prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        corrupt_outputs = model(input_ids=corrupt_input_ids, output_attentions=True,
                                output_hidden_states=True, return_dict=True)

    # Get W_U (handles meta device)
    W_U = get_W_U(model, model_name=None)

    corrupt_base_logits = corrupt_outputs.logits[0, corrupt_answer_pos, :].detach().float().cpu().numpy()
    modified_logits = corrupt_base_logits.copy()

    layers_list = get_layers(model)
    n_heads, d_head, d_model, n_kv_heads, kv_group_size = get_head_dims(model)

    # Find correct category positions in corrupt prompt
    correct_rule_corrupt = None
    for r in rule_info_corrupt:
        if r["is_correct"]:
            correct_rule_corrupt = r
            break

    if correct_rule_corrupt is None:
        return corrupt_base_logits

    correct_cat_pos_corrupt = correct_rule_corrupt["cat_positions"]

    # Find correct category positions in clean prompt
    correct_rule_clean = None
    for r in rule_info_clean:
        if r["is_correct"]:
            correct_rule_clean = r
            break

    if correct_rule_clean is None:
        return corrupt_base_logits

    for (l, h_list) in target_layers_heads.items():
        if l >= len(corrupt_outputs.attentions) or corrupt_outputs.attentions[l] is None:
            continue
        if l >= len(clean_outputs.attentions) or clean_outputs.attentions[l] is None:
            continue

        # Load weights
        W_V_np, W_O_np = _load_attn_weights_from_safetensors(model, l) if \
            (layers_list[l].self_attn.v_proj.weight.is_meta or layers_list[l].self_attn.o_proj.weight.is_meta) \
            else (layers_list[l].self_attn.v_proj.weight.detach().float().cpu().numpy(),
                  layers_list[l].self_attn.o_proj.weight.detach().float().cpu().numpy())

        # Corrupt hidden states input to layer l
        h_input_corrupt = corrupt_outputs.hidden_states[l][0, :, :].detach().float().cpu().numpy()

        for h in h_list:
            kv_h = h // kv_group_size
            W_V_h = W_V_np[kv_h * d_head:(kv_h + 1) * d_head, :]  # [d_head, d_model] — GQA
            W_O_h = W_O_np[:, h * d_head:(h + 1) * d_head]  # [d_model, d_head]

            # Corrupt attention at answer_pos: [seq]
            attn_corrupt = corrupt_outputs.attentions[l][0, h, corrupt_answer_pos, :].detach().float().cpu().numpy()

            # Value vectors for corrupt
            V_corrupt = (W_V_h @ h_input_corrupt.T).T  # [seq, d_head]

            # Original head output
            original_weighted_V = attn_corrupt[:, np.newaxis] * V_corrupt
            original_head_out = W_O_h @ original_weighted_V.sum(axis=0)

            # STRATEGY: Boost attention to correct category positions in corrupt prompt
            # This simulates "if the corrupt query attended to the right category"
            # We multiply attention at correct_cat_positions by a boost factor
            # and redistribute the excess from other positions

            # Approach 1: Direct injection — set attention at correct cat positions
            # to the clean attention level at those positions, renormalize

            # Clean attention at answer_pos for this head
            attn_clean = clean_outputs.attentions[l][0, h, clean_answer_pos, :].detach().float().cpu().numpy()

            # Clean attention to correct category positions
            clean_cat_attn = float(attn_clean[correct_rule_clean["cat_positions"]].sum())

            # Modified corrupt: set cat positions to clean level, renormalize
            modified_attn = attn_corrupt.copy()

            # Zero out current cat position attention in corrupt
            for p in correct_cat_pos_corrupt:
                if p < len(modified_attn):
                    modified_attn[p] = 0.0

            # Add clean-level attention distributed over cat positions
            n_cat_pos = len(correct_cat_pos_corrupt)
            if n_cat_pos > 0 and clean_cat_attn > 0:
                per_pos_attn = clean_cat_attn / n_cat_pos
                for p in correct_cat_pos_corrupt:
                    if p < len(modified_attn):
                        modified_attn[p] = per_pos_attn

            # Renormalize
            sum_attn = modified_attn.sum()
            if sum_attn > 1e-10:
                modified_attn = modified_attn / sum_attn

            # Recompute head output
            modified_weighted_V = modified_attn[:, np.newaxis] * V_corrupt
            modified_head_out = W_O_h @ modified_weighted_V.sum(axis=0)

            # Apply delta to logits
            delta_head = modified_head_out - original_head_out
            delta_logits = W_U @ delta_head
            modified_logits += delta_logits

    return modified_logits


# ============================================================================
# Step 3: Value vector logit attribution (using get_W_U from safetensors)
# ============================================================================

def compute_value_logit_attribution(model, tokenizer, device, prompt, answer_pos,
                                    rule_info, top_heads, cat_token_ids, d_head):
    """For each top retrieval head, compute how much it contributes to correct category logit.

    Uses get_W_U from model_utils (handles meta device via safetensors).
    Properly handles GQA: query head h maps to KV head kv_h = h // kv_group_size.
    """
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)

    attn_weights_all = outputs.attentions
    hidden_states_all = outputs.hidden_states

    # Get W_U using model_utils (handles meta device via safetensors)
    W_U = get_W_U(model, model_name=None)  # [vocab, d_model]

    # GQA info
    n_heads, d_head_actual, d_model, n_kv_heads, kv_group_size = get_head_dims(model)

    results = []
    layers_list = get_layers(model)

    for l, h_list in top_heads.items():
        for h in h_list:
            if l >= len(attn_weights_all) or attn_weights_all[l] is None:
                continue

            attn = attn_weights_all[l][0, h, answer_pos, :].detach().float().cpu().numpy()
            h_input = hidden_states_all[l][0, :, :].detach().float().cpu().numpy()

            # GQA: query head h maps to KV head kv_h = h // kv_group_size
            kv_h = h // kv_group_size

            # Load W_V, W_O (handle meta device)
            if layers_list[l].self_attn.v_proj.weight.is_meta or layers_list[l].self_attn.o_proj.weight.is_meta:
                W_V_np, W_O_np = _load_attn_weights_from_safetensors(model, l)
                # GQA: W_V shape is [n_kv_heads * d_head, d_model]
                W_V_h = W_V_np[kv_h * d_head_actual:(kv_h + 1) * d_head_actual, :]  # [d_head, d_model]
                W_O_h = W_O_np[:, h * d_head_actual:(h + 1) * d_head_actual]  # [d_model, d_head]
            else:
                W_V_h = layers_list[l].self_attn.v_proj.weight[kv_h * d_head_actual:(kv_h + 1) * d_head_actual, :].detach().float().cpu().numpy()
                W_O_h = layers_list[l].self_attn.o_proj.weight[:, h * d_head_actual:(h + 1) * d_head_actual].detach().float().cpu().numpy()

            V_all = (W_V_h @ h_input.T).T  # [seq, d_head]

            # Total head output
            weighted_V = attn[:, np.newaxis] * V_all
            head_output = W_O_h @ weighted_V.sum(axis=0)  # [d_model]

            # Logit contribution for each category token
            for cat, tid in cat_token_ids.items():
                logit_contrib = float(W_U[tid] @ head_output)
                results.append({
                    "layer": l, "head": h, "category": cat,
                    "logit_contribution": logit_contrib,
                    "attn_mass_total": float(attn.sum()),
                })

            # Position-specific contributions for correct rule
            correct_rule = None
            for r in rule_info:
                if r["is_correct"]:
                    correct_rule = r
                    break

            if correct_rule is not None:
                correct_cat = correct_rule["category"]
                correct_tid = cat_token_ids[correct_cat]

                # Direct logit from cat positions: sum_j attn[j] * W_U[tid] @ W_O_h @ V[j]
                cat_direct = 0.0
                for p in correct_rule["cat_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        cat_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                obj_direct = 0.0
                for p in correct_rule["obj_positions"]:
                    if p < V_all.shape[0] and p < attn.shape[0]:
                        obj_direct += float(attn[p]) * float(W_U[correct_tid] @ W_O_h @ V_all[p])

                # Attention mass at specific positions
                attn_to_obj = float(attn[correct_rule["obj_positions"]].sum())
                attn_to_cat = float(attn[correct_rule["cat_positions"]].sum())

                results.append({
                    "layer": l, "head": h, "type": "position_attribution",
                    "attn_to_correct_obj": attn_to_obj,
                    "attn_to_correct_cat": attn_to_cat,
                    "cat_direct_logit_from_cat": cat_direct,
                    "cat_direct_logit_from_obj": obj_direct,
                    "correct_category": correct_cat,
                })

    return results


# ============================================================================
# Step 4: Balanced table binding — A-before-B AND B-before-A
# ============================================================================

def build_balanced_conflicting_prompt(tokenizer, table_a, table_b, query_object,
                                       table_context, order="A_first"):
    """Build prompt with two conflicting rule tables, with controlled order.

    order: "A_first" (Table A then Table B) or "B_first" (Table B then Table A)
    table_context: "A" or "B" (which table the query refers to)

    This removes the recency bias by testing both orders.
    """
    rng = random.Random(42)
    rules_a = list(table_a.items())
    rules_b = list(table_b.items())
    rng.shuffle(rules_a)
    rng.shuffle(rules_b)

    lines_a = [f"{obj} belongs to {cat}." for obj, cat in rules_a]
    lines_b = [f"{obj} belongs to {cat}." for obj, cat in rules_b]

    if order == "A_first":
        prompt = "Table A:\n" + "\n".join(lines_a) + "\n\n"
        prompt += "Table B:\n" + "\n".join(lines_b) + "\n\n"
    else:
        prompt = "Table B:\n" + "\n".join(lines_b) + "\n\n"
        prompt += "Table A:\n" + "\n".join(lines_a) + "\n\n"

    prompt += f"Using Table {table_context}, {query_object} belongs to ?\nAnswer:"

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Find positions — need to track which block each rule belongs to
    # This is more complex because order changes positions
    rule_info = []
    search_start = 0

    # First block (depends on order)
    first_rules = rules_a if order == "A_first" else rules_b
    first_table = "A" if order == "A_first" else "B"
    for obj, cat in first_rules:
        obj_result = find_symbol_in_full(tokenizer, full_ids, obj, search_start)
        if obj_result is None:
            continue
        obj_pos, obj_end = obj_result
        cat_result = find_symbol_in_full(tokenizer, full_ids, cat, obj_end)
        if cat_result is None:
            continue
        cat_pos, cat_end = cat_result

        is_correct = (obj == query_object and table_context == first_table)
        rule_info.append({
            "object": obj, "category": cat,
            "obj_positions": list(range(obj_pos, obj_end)),
            "cat_positions": list(range(cat_pos, cat_end)),
            "is_correct": is_correct,
            "table": first_table,
        })
        search_start = cat_end

    # Second block
    second_rules = rules_b if order == "A_first" else rules_a
    second_table = "B" if order == "A_first" else "A"
    for obj, cat in second_rules:
        obj_result = find_symbol_in_full(tokenizer, full_ids, obj, search_start)
        if obj_result is None:
            continue
        obj_pos, obj_end = obj_result
        cat_result = find_symbol_in_full(tokenizer, full_ids, cat, obj_end)
        if cat_result is None:
            continue
        cat_pos, cat_end = cat_result

        is_correct = (obj == query_object and table_context == second_table)
        rule_info.append({
            "object": obj, "category": cat,
            "obj_positions": list(range(obj_pos, obj_end)),
            "cat_positions": list(range(cat_pos, cat_end)),
            "is_correct": is_correct,
            "table": second_table,
        })
        search_start = cat_end

    answer_pos = len(full_ids) - 1

    return prompt, full_ids, rule_info, answer_pos


def create_conflicting_tables(truth_table, query_object, categories):
    """Create Table A (original) and Table B (conflicting category for query_object)."""
    table_a = truth_table.copy()
    correct_cat = table_a[query_object]

    other_cats = [c for c in categories if c != correct_cat]
    rng = random.Random(123)
    wrong_cat = rng.choice(other_cats)

    table_b = truth_table.copy()
    table_b[query_object] = wrong_cat

    return table_a, table_b, correct_cat, wrong_cat


# ============================================================================
# Step 5: Retrieval-to-state transition (using get_W_U from safetensors)
# ============================================================================

def probe_category_after_patching(model, tokenizer, device, layers, truth_table,
                                  query_object, patch_heads, d_head, probe_layers,
                                  categories, cat_token_ids, seed=42):
    """After patching retrieval heads, check if category probe accuracy changes at subsequent layers.

    Uses get_W_U from model_utils (handles meta device via safetensors).
    """
    correct_cat = truth_table[query_object]
    correct_tid = cat_token_ids[correct_cat]

    # Build clean and corrupt prompts
    clean_prompt, _, _, _, clean_answer_pos = build_prompt_with_positions(
        tokenizer, truth_table, query_object, seed=seed)

    corrupt_table = create_corrupted_table(truth_table, query_object, seed=seed + 1)
    corrupt_prompt, _, _, _, corrupt_answer_pos = build_prompt_with_positions(
        tokenizer, corrupt_table, query_object, seed=seed + 1)

    # Two-step prompts
    clean_2step = clean_prompt + " c"
    corrupt_2step = corrupt_prompt + " c"

    # Get W_U (handles meta device)
    W_U = get_W_U(model, model_name=None)

    def get_hidden_states_at_layers(prompt_text):
        input_ids = torch.tensor([tokenizer.encode(prompt_text, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
        states = {}
        for l in probe_layers:
            if l < len(outputs.hidden_states):
                hs = outputs.hidden_states[l][0, -1, :].detach().float().cpu().numpy()
                states[l] = hs
        return states

    # Clean hidden states
    clean_states = get_hidden_states_at_layers(clean_2step)
    # Corrupt hidden states
    corrupt_states = get_hidden_states_at_layers(corrupt_2step)

    # Patched: capture clean head inputs, then inject into corrupt
    clean_storage = {}
    capture_hooks = []
    for l, heads in patch_heads.items():
        if heads and l < len(layers):
            o_proj = layers[l].self_attn.o_proj
            ch = make_capture_hook(clean_storage, heads, d_head)
            capture_hooks.append(o_proj.register_forward_hook(ch))

    try:
        clean_input_ids = torch.tensor([tokenizer.encode(clean_prompt, add_special_tokens=False)], device=device)
        with torch.inference_mode():
            _ = model(input_ids=clean_input_ids, return_dict=True)
    finally:
        for h in capture_hooks:
            h.remove()

    # Corrupt + patch forward
    patch_hooks = []
    for l, heads in patch_heads.items():
        if heads and l < len(layers):
            o_proj = layers[l].self_attn.o_proj
            ph = make_patch_hook(clean_storage, heads, d_head)
            patch_hooks.append(o_proj.register_forward_hook(ph))

    try:
        patched_states = get_hidden_states_at_layers(corrupt_2step)
    finally:
        for h in patch_hooks:
            h.remove()

    # Compare states
    results = {}
    for l in probe_layers:
        if l in clean_states and l in corrupt_states and l in patched_states:
            clean_vec = clean_states[l]
            corrupt_vec = corrupt_states[l]
            patched_vec = patched_states[l]

            cos_cc = float(np.dot(clean_vec, corrupt_vec) / (np.linalg.norm(clean_vec) * np.linalg.norm(corrupt_vec) + 1e-10))
            cos_cp = float(np.dot(clean_vec, patched_vec) / (np.linalg.norm(clean_vec) * np.linalg.norm(patched_vec) + 1e-10))

            # Logits via W_U (only category tokens)
            clean_logits = {cat: float(W_U[cat_token_ids[cat]] @ clean_vec) for cat in categories}
            corrupt_logits = {cat: float(W_U[cat_token_ids[cat]] @ corrupt_vec) for cat in categories}
            patched_logits = {cat: float(W_U[cat_token_ids[cat]] @ patched_vec) for cat in categories}

            clean_pred = max(clean_logits, key=clean_logits.get)
            corrupt_pred = max(corrupt_logits, key=corrupt_logits.get)
            patched_pred = max(patched_logits, key=patched_logits.get)

            results[l] = {
                "cos_clean_corrupt": cos_cc,
                "cos_clean_patched": cos_cp,
                "clean_pred": clean_pred,
                "corrupt_pred": corrupt_pred,
                "patched_pred": patched_pred,
                "clean_logits": clean_logits,
                "corrupt_logits": corrupt_logits,
                "patched_logits": patched_logits,
                "correct_cat": correct_cat,
            }

    return results


# ============================================================================
# Helper hooks and functions
# ============================================================================

def make_capture_hook(storage, head_indices, d_head):
    def hook(module, input, output):
        x = input[0]
        for h in head_indices:
            storage[h] = x[:, :, h * d_head:(h + 1) * d_head].detach().cpu().clone()
    return hook


def make_patch_hook(clean_storage, head_indices, d_head):
    """Replace corrupt head inputs with clean versions."""
    def hook(module, input, output):
        x = input[0]
        x_new = x.clone()
        for h in head_indices:
            if h in clean_storage:
                clean_slice = clean_storage[h].to(x_new.device, x_new.dtype)
                seq_corrupt = x_new.shape[1]
                seq_clean = clean_slice.shape[1]
                n_patch = min(seq_corrupt, seq_clean)
                x_new[:, :n_patch, h * d_head:(h + 1) * d_head] = clean_slice[:, :n_patch, :]
        return module.forward(x_new)
    return hook


def create_corrupted_table(truth_table, query_object, seed=42):
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


def heads_to_groups(head_list):
    groups = {}
    for l, h in head_list:
        groups.setdefault(l, []).append(h)
    return groups


# ============================================================================
# Main analysis
# ============================================================================

def run_model(args):
    model, tokenizer, device = load_model_eager(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        n_layers = info.n_layers
        n_heads, d_head, d_model, n_kv_heads, kv_group_size = get_head_dims(model)

        log(f"{args.model}: n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}, d_model={d_model}, n_kv_heads={n_kv_heads}, kv_group_size={kv_group_size}")

        objects = CANDIDATE_OBJECTS[:8]
        categories = CANDIDATE_CATEGORIES[:4]

        # Category token IDs
        cat_token_seqs = {}
        for cat in categories:
            ids = tokenizer.encode(" " + cat, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(cat, add_special_tokens=False)
            cat_token_seqs[cat] = ids

        cat_token_ids = {}
        for cat, ids in cat_token_seqs.items():
            if len(ids) >= 2:
                cat_token_ids[cat] = ids[1]
            else:
                cat_token_ids[cat] = ids[0]
        log(f"Category distinguishing tokens: {cat_token_ids}")

        n_tables = args.n_tables
        truth_tables = build_truth_tables(objects, categories, n_tables)
        log(f"Built {n_tables} truth tables")

        result = {
            "phase": 579,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers, "n_heads": n_heads, "d_head": d_head, "d_model": d_model,
            "n_tables": n_tables, "cat_token_ids": {k: int(v) for k, v in cat_token_ids.items()},
        }

        # ================================================================
        # Step 0: Identify top retrieval heads
        # ================================================================
        log("=== Step 0: Identify top retrieval heads ===")

        all_obj_scores = np.zeros((n_layers, n_heads))
        all_cat_scores = np.zeros((n_layers, n_heads))
        all_attn_mass_obj = np.zeros((n_layers, n_heads))
        all_attn_mass_cat = np.zeros((n_layers, n_heads))
        n_samples_scored = 0

        sample_data = []
        sample_count = 0
        total_samples = n_tables * len(objects)

        for tt_idx, tt in enumerate(truth_tables):
            for obj in objects:
                prompt, full_ids, rule_info, q_pos, answer_pos = build_prompt_with_positions(
                    tokenizer, tt, obj, seed=tt_idx * 100 + hash(obj) % 1000)

                if not rule_info or answer_pos >= len(full_ids):
                    continue

                correct_cat = tt[obj]

                try:
                    attn_rows, logits = extract_attention_and_logits(
                        model, tokenizer, device, prompt, answer_pos)
                except Exception as e:
                    log(f"  WARNING: attention extraction failed: {e}")
                    continue

                for l in range(n_layers):
                    if l in attn_rows:
                        scores = compute_rule_match_scores(attn_rows[l], rule_info, n_heads)
                        for h in range(n_heads):
                            if h in scores:
                                all_obj_scores[l, h] += scores[h]["obj_match"]
                                all_cat_scores[l, h] += scores[h]["cat_copy"]
                                all_attn_mass_obj[l, h] += scores[h]["attn_mass_obj"]
                                all_attn_mass_cat[l, h] += scores[h]["attn_mass_cat"]

                n_samples_scored += 1
                sample_count += 1
                sample_data.append({
                    "prompt": prompt, "rule_info": rule_info,
                    "answer_pos": answer_pos, "correct_cat": correct_cat,
                    "tt_idx": tt_idx, "obj": obj,
                })

                if sample_count % 20 == 0:
                    log(f"  Processed {sample_count}/{total_samples} samples")

        all_obj_scores /= max(1, n_samples_scored)
        all_cat_scores /= max(1, n_samples_scored)
        all_attn_mass_obj /= max(1, n_samples_scored)
        all_attn_mass_cat /= max(1, n_samples_scored)

        combined_scores = all_obj_scores + all_cat_scores
        flat_ranked = [(l, h, combined_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        flat_ranked.sort(key=lambda x: x[2], reverse=True)

        log(f"  Top-5 combined heads:")
        for l, h, s in flat_ranked[:5]:
            log(f"    L{l}H{h}: combined={s:.4f}, obj_match={all_obj_scores[l,h]:.4f}, "
                f"cat_copy={all_cat_scores[l,h]:.4f}, mass_obj={all_attn_mass_obj[l,h]:.6f}, "
                f"mass_cat={all_attn_mass_cat[l,h]:.6f}")

        result["step0_top_heads"] = {
            "top_combined_50": [{"layer": l, "head": h, "combined": float(s),
                                "obj_match": float(all_obj_scores[l,h]),
                                "cat_copy": float(all_cat_scores[l,h]),
                                "mass_obj": float(all_attn_mass_obj[l,h]),
                                "mass_cat": float(all_attn_mass_cat[l,h])}
                               for l, h, s in flat_ranked[:50]],
        }

        # ================================================================
        # Step 1: TRUE attention edge ablation
        # ================================================================
        log("=== Step 1: TRUE attention edge ablation (modify A matrix) ===")

        edge_samples = []
        for tt_idx in range(min(n_tables, 10)):
            for s in sample_data:
                if s["tt_idx"] == tt_idx:
                    edge_samples.append(s)
                    break

        # Test configurations:
        # 1. baseline (no ablation)
        # 2. cat_edge_all: zero A[l,h][q, correct_cat_positions] for ALL top-10 heads
        # 3. cat_edge_top3: zero A for top-3 cat-copy heads only
        # 4. obj_edge_all: zero A[l,h][q, correct_obj_positions] for ALL top-10 heads
        # 5. cat_edge_top1_layer: zero A for single best head at its peak layer

        # Separate cat-copy heads and obj-match heads
        cat_flat = [(l, h, all_cat_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        cat_flat.sort(key=lambda x: x[2], reverse=True)
        obj_flat = [(l, h, all_obj_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        obj_flat.sort(key=lambda x: x[2], reverse=True)

        edge_configs = {
            "baseline": None,
            "cat_edge_top10": heads_to_groups([(l, h) for l, h, _ in flat_ranked[:10]]),
            "cat_edge_top3_cat": heads_to_groups([(l, h) for l, h, _ in cat_flat[:3]]),
            "obj_edge_top3_obj": heads_to_groups([(l, h) for l, h, _ in obj_flat[:3]]),
            "cat_edge_top1": heads_to_groups([(l, h) for l, h, _ in cat_flat[:1]]),
        }

        edge_results = {}

        for config_name, head_groups in edge_configs.items():
            log(f"  Edge config: {config_name}")

            cat_logit_changes = []
            argmax_changes = 0
            correct_baseline = 0
            correct_after = 0
            n_tested = 0

            for i, s in enumerate(edge_samples):
                correct_rule = None
                for r in s["rule_info"]:
                    if r["is_correct"]:
                        correct_rule = r
                        break

                if correct_rule is None:
                    continue

                # For all configs, we need to compare with a baseline that
                # can distinguish categories. The single-step prompt only predicts
                # "c" (shared by all categories), so we need the two-step prompt.
                # But edge ablation positions are calibrated to the original prompt.
                # Strategy: use the original prompt for edge ablation, then compare
                # category token logits (which are distinguishable even on single-step
                # because they include the second token of each category string).

                # Get baseline logits for this sample (single-step, for position calibration)
                baseline_logits_single = extract_attention_and_logits(
                    model, tokenizer, device, s["prompt"], s["answer_pos"])[1]
                baseline_cl_dict = {cat: float(baseline_logits_single[cat_token_ids[cat]])
                                    for cat, tid in cat_token_ids.items()}
                baseline_pred_dict = max(baseline_cl_dict, key=baseline_cl_dict.get)

                if config_name == "baseline":
                    correct_baseline += (1 if baseline_pred_dict == correct_cat else 0)
                else:
                    # TRUE edge ablation
                    if "cat" in config_name:
                        target_positions = correct_rule["cat_positions"]
                    elif "obj" in config_name:
                        target_positions = correct_rule["obj_positions"]
                    else:
                        target_positions = correct_rule["cat_positions"]

                    logits = true_attn_edge_ablation_forward(
                        model, tokenizer, device, s["prompt"], s["answer_pos"],
                        target_positions, head_groups)

                    cl = {cat: float(logits[tid]) for cat, tid in cat_token_ids.items()}
                    pred = max(cl, key=cl.get)

                    logit_change = cl[correct_cat] - baseline_cl_dict[correct_cat]
                    cat_logit_changes.append(logit_change)

                    if pred != baseline_pred_dict:
                        argmax_changes += 1
                    correct_after += (1 if pred == correct_cat else 0)

                    if i < 3:
                        log(f"    Sample {i}: correct={correct_cat}, baseline={baseline_pred_dict}, "
                            f"ablated={pred}, logit_change={logit_change:.4f}")

                n_tested += 1

            if config_name == "baseline":
                edge_results[config_name] = {
                    "n_tested": n_tested,
                    "correct_count": correct_baseline,
                    "accuracy": correct_baseline / max(1, n_tested),
                }
            else:
                edge_results[config_name] = {
                    "n_tested": n_tested,
                    "mean_logit_change": float(np.mean(cat_logit_changes)) if cat_logit_changes else 0,
                    "argmax_change_rate": argmax_changes / max(1, n_tested),
                    "correct_after": correct_after,
                    "accuracy_after": correct_after / max(1, n_tested),
                }
                r = edge_results[config_name]
                log(f"    Summary: logit_change={r['mean_logit_change']:.4f}, "
                    f"argmax_change={r['argmax_change_rate']:.3f}, "
                    f"accuracy_after={r['accuracy_after']:.3f}")

        result["step1_true_edge_ablation"] = edge_results

        # ================================================================
        # Step 2: Attention pattern patching
        # ================================================================
        log("=== Step 2: Attention pattern patching ===")

        patch_samples = []
        for tt_idx in range(min(n_tables, 5)):
            tt = truth_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]
            patch_samples.append((tt, obj, tt_idx * 200))

        patch_configs = {
            "top5_combined": heads_to_groups([(l, h) for l, h, _ in flat_ranked[:5]]),
            "top10_combined": heads_to_groups([(l, h) for l, h, _ in flat_ranked[:10]]),
        }

        patch_results = {}

        for config_name, head_groups in patch_configs.items():
            log(f"  Pattern patching config: {config_name}")

            recoveries = []
            kl_corrupt_list = []
            kl_patched_list = []

            for tt, obj, seed in patch_samples:
                correct_cat = tt[obj]
                correct_tid = cat_token_ids[correct_cat]

                corrupt_table = create_corrupted_table(tt, obj, seed=seed + 1)

                clean_prompt, _, clean_rule_info, _, clean_answer_pos = build_prompt_with_positions(
                    tokenizer, tt, obj, seed=seed)
                corrupt_prompt, _, corrupt_rule_info, _, corrupt_answer_pos = build_prompt_with_positions(
                    tokenizer, corrupt_table, obj, seed=seed + 1)

                # Get baseline logits
                clean_logits = extract_attention_and_logits(
                    model, tokenizer, device, clean_prompt, clean_answer_pos)[1]
                corrupt_logits = extract_attention_and_logits(
                    model, tokenizer, device, corrupt_prompt, corrupt_answer_pos)[1]

                # Pattern patching
                patched_logits = attn_pattern_patching_forward(
                    model, tokenizer, device, corrupt_prompt, clean_prompt,
                    corrupt_answer_pos, clean_answer_pos,
                    head_groups, corrupt_rule_info, clean_rule_info)

                # Analyze
                clean_cl = {cat: float(clean_logits[tid]) for cat, tid in cat_token_ids.items()}
                corrupt_cl = {cat: float(corrupt_logits[tid]) for cat, tid in cat_token_ids.items()}
                patched_cl = {cat: float(patched_logits[tid]) for cat, tid in cat_token_ids.items()}

                clean_pred = max(clean_cl, key=clean_cl.get)
                corrupt_pred = max(corrupt_cl, key=corrupt_cl.get)
                patched_pred = max(patched_cl, key=patched_cl.get)

                recovered = patched_pred == correct_cat
                recoveries.append(recovered)

                def softmax_4(cl):
                    vals = np.array(list(cl.values()))
                    e = np.exp(vals - vals.max())
                    return e / e.sum()

                clean_probs = softmax_4(clean_cl)
                corrupt_probs = softmax_4(corrupt_cl)
                patched_probs = softmax_4(patched_cl)

                kl_corrupt = float(np.sum(clean_probs * (np.log(clean_probs + 1e-10) - np.log(corrupt_probs + 1e-10))))
                kl_patched = float(np.sum(clean_probs * (np.log(clean_probs + 1e-10) - np.log(patched_probs + 1e-10))))
                kl_corrupt_list.append(kl_corrupt)
                kl_patched_list.append(kl_patched)

                log(f"    Sample: clean={clean_pred}, corrupt={corrupt_pred}, "
                    f"patched={patched_pred}, correct={correct_cat}, recovered={recovered}")

            n_recovered = sum(recoveries)
            patch_results[config_name] = {
                "n_recovered": n_recovered,
                "n_tests": len(patch_samples),
                "recovery_rate": n_recovered / max(1, len(patch_samples)),
                "mean_kl_corrupt": float(np.mean(kl_corrupt_list)),
                "mean_kl_patched": float(np.mean(kl_patched_list)),
                "kl_improvement": float(np.mean(kl_corrupt_list)) - float(np.mean(kl_patched_list)),
            }
            r = patch_results[config_name]
            log(f"    {config_name}: recovered={n_recovered}/{len(patch_samples)}, "
                f"KL: {r['mean_kl_corrupt']:.4f}→{r['mean_kl_patched']:.4f} "
                f"(improvement={r['kl_improvement']:.4f})")

        result["step2_pattern_patching"] = patch_results

        # ================================================================
        # Step 3: Value vector logit attribution
        # ================================================================
        log("=== Step 3: Value vector logit attribution ===")

        attribution_samples = edge_samples[:10]
        top_5_combined = heads_to_groups([(l, h) for l, h, _ in flat_ranked[:5]])
        top_10_combined = heads_to_groups([(l, h) for l, h, _ in flat_ranked[:10]])

        all_attributions = []
        for i, s in enumerate(attribution_samples[:5]):
            log(f"  Attribution sample {i}/5")
            try:
                attrs = compute_value_logit_attribution(
                    model, tokenizer, device, s["prompt"], s["answer_pos"],
                    s["rule_info"], top_5_combined, cat_token_ids, d_head)
                all_attributions.extend(attrs)
            except Exception as e:
                log(f"  WARNING: attribution failed for sample {i}: {e}")
                import traceback
                traceback.print_exc()

        # Summarize per-head
        head_contributions = {}
        for attr in all_attributions:
            if "type" not in attr:
                key = f"L{attr['layer']}H{attr['head']}"
                if key not in head_contributions:
                    head_contributions[key] = {}
                head_contributions[key][attr["category"]] = attr["logit_contribution"]
            elif attr["type"] == "position_attribution":
                key = f"L{attr['layer']}H{attr['head']}"
                if key not in head_contributions:
                    head_contributions[key] = {}
                head_contributions[key]["pos_attribution"] = attr

        log(f"  Value vector logit attribution (top-5 heads):")
        for key, vals in sorted(head_contributions.items()):
            if "pos_attribution" in vals:
                pa = vals["pos_attribution"]
                log(f"    {key}: cat_direct_logit_from_cat={pa['cat_direct_logit_from_cat']:.4f}, "
                    f"cat_direct_logit_from_obj={pa['cat_direct_logit_from_obj']:.4f}, "
                    f"attn_obj={pa['attn_to_correct_obj']:.4f}, "
                    f"attn_cat={pa['attn_to_correct_cat']:.4f}")

        # Check: does correct category get more logit than wrong categories?
        correct_vs_wrong = []
        for key, vals in head_contributions.items():
            if "pos_attribution" in vals:
                pa = vals["pos_attribution"]
                correct_cat_name = pa["correct_category"]
                # Find logit contributions for this head
                cat_logits = {c: vals.get(c, 0) for c in categories if c in vals}
                if correct_cat_name in cat_logits:
                    correct_logit = cat_logits[correct_cat_name]
                    wrong_logits = [cat_logits[c] for c in categories if c != correct_cat_name and c in cat_logits]
                    if wrong_logits:
                        correct_vs_wrong.append({
                            "head": key,
                            "correct_logit": correct_logit,
                            "mean_wrong_logit": float(np.mean(wrong_logits)),
                            "advantage": correct_logit - float(np.mean(wrong_logits)),
                        })

        if correct_vs_wrong:
            mean_advantage = float(np.mean([v["advantage"] for v in correct_vs_wrong]))
            log(f"  Mean correct-vs-wrong logit advantage: {mean_advantage:.4f}")

        result["step3_value_attribution"] = {
            "head_contributions": head_contributions,
            "correct_vs_wrong": correct_vs_wrong,
            "all_attributions_count": len(all_attributions),
        }

        # ================================================================
        # Step 4: Balanced table binding
        # ================================================================
        log("=== Step 4: Balanced table binding (A-first AND B-first) ===")

        binding_results = []

        # Test 4 conditions: {A_first, B_first} × {context_A, context_B}
        for tt_idx in range(min(n_tables, 5)):
            tt = truth_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]

            table_a, table_b, correct_cat_a, wrong_cat_b = create_conflicting_tables(tt, obj, categories)

            for order in ["A_first", "B_first"]:
                for context in ["A", "B"]:
                    expected_cat = correct_cat_a if context == "A" else wrong_cat_b

                    prompt, full_ids, rule_info, answer_pos = build_balanced_conflicting_prompt(
                        tokenizer, table_a, table_b, obj,
                        table_context=context, order=order)

                    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

                    with torch.inference_mode():
                        gen = model.generate(input_ids, max_new_tokens=10, do_sample=False,
                                             pad_token_id=tokenizer.pad_token_id)

                    text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

                    detected = [c for c in categories if c.lower() in text.lower()]
                    pred_cat = detected[0] if len(detected) == 1 else ("none" if len(detected) == 0 else "ambiguous")

                    correct = pred_cat == expected_cat

                    # Also extract attention to table blocks
                    try:
                        attn_rows, _ = extract_attention_and_logits(model, tokenizer, device, prompt, answer_pos)

                        # Compute attention to correct table's cat positions
                        attn_to_correct_table_cat = 0.0
                        attn_to_wrong_table_cat = 0.0
                        peak_layer = int(np.argmax(combined_scores.mean(axis=1)))

                        if peak_layer in attn_rows:
                            for r in rule_info:
                                cat_attn = float(attn_rows[peak_layer][:, r["cat_positions"]].sum())
                                if r["table"] == context:
                                    attn_to_correct_table_cat += cat_attn
                                else:
                                    attn_to_wrong_table_cat += cat_attn
                    except Exception as e:
                        attn_to_correct_table_cat = 0
                        attn_to_wrong_table_cat = 0

                    binding_results.append({
                        "tt_idx": tt_idx, "obj": obj,
                        "order": order, "context": context,
                        "expected_cat": expected_cat,
                        "predicted_cat": pred_cat,
                        "correct": correct,
                        "text": text,
                        "attn_to_correct_table_cat": attn_to_correct_table_cat,
                        "attn_to_wrong_table_cat": attn_to_wrong_table_cat,
                    })

                    log(f"    tt={tt_idx}, order={order}, context={context}: "
                        f"expected={expected_cat}, predicted={pred_cat}, correct={correct}")

        # Summarize by order and context
        for order in ["A_first", "B_first"]:
            for context in ["A", "B"]:
                subset = [b for b in binding_results if b["order"] == order and b["context"] == context]
                n_correct = sum(1 for b in subset if b["correct"])
                log(f"  order={order}, context={context}: {n_correct}/{len(subset)} correct")

        # Key comparison: A_context accuracy when A is near query vs when A is far
        a_near = [b for b in binding_results if b["context"] == "A" and b["order"] == "A_first"]
        a_far = [b for b in binding_results if b["context"] == "A" and b["order"] == "B_first"]
        b_near = [b for b in binding_results if b["context"] == "B" and b["order"] == "B_first"]
        b_far = [b for b in binding_results if b["context"] == "B" and b["order"] == "A_first"]

        a_near_acc = sum(1 for b in a_near if b["correct"]) / max(1, len(a_near))
        a_far_acc = sum(1 for b in a_far if b["correct"]) / max(1, len(a_far))
        b_near_acc = sum(1 for b in b_near if b["correct"]) / max(1, len(b_near))
        b_far_acc = sum(1 for b in b_far if b["correct"]) / max(1, len(b_far))

        log(f"  Recency bias analysis:")
        log(f"    A_context, A_near (Table A last): {a_near_acc:.3f}")
        log(f"    A_context, A_far  (Table A first): {a_far_acc:.3f}")
        log(f"    B_context, B_near (Table B last): {b_near_acc:.3f}")
        log(f"    B_context, B_far  (Table B first): {b_far_acc:.3f}")

        result["step4_balanced_binding"] = {
            "tests": binding_results,
            "a_near_acc": a_near_acc,
            "a_far_acc": a_far_acc,
            "b_near_acc": b_near_acc,
            "b_far_acc": b_far_acc,
            "recency_bias": a_near_acc - a_far_acc,
        }

        # ================================================================
        # Step 5: Retrieval-to-state transition
        # ================================================================
        log("=== Step 5: Retrieval-to-state transition ===")

        probe_layers = list(range(n_layers // 2, n_layers, 2))
        log(f"  Probe layers: {probe_layers}")

        transition_results = []
        for tt_idx in range(min(n_tables, 5)):
            tt = truth_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]

            try:
                tr = probe_category_after_patching(
                    model, tokenizer, device, layers, tt, obj,
                    patch_heads=patch_configs["top10_combined"],
                    d_head=d_head, probe_layers=probe_layers,
                    categories=categories, cat_token_ids=cat_token_ids,
                    seed=tt_idx * 200)
                transition_results.append(tr)

                for l, data in tr.items():
                    log(f"    L{l}: cos(clean,corrupt)={data['cos_clean_corrupt']:.4f}, "
                        f"cos(clean,patched)={data['cos_clean_patched']:.4f}, "
                        f"pred: clean={data['clean_pred']}, corrupt={data['corrupt_pred']}, "
                        f"patched={data['patched_pred']}")
            except Exception as e:
                log(f"  WARNING: transition test failed for tt={tt_idx}: {e}")
                import traceback
                traceback.print_exc()

        improvement_count = 0
        for tr in transition_results:
            for l, data in tr.items():
                if data["patched_pred"] == data["correct_cat"] and data["corrupt_pred"] != data["correct_cat"]:
                    improvement_count += 1

        total_probe_points = sum(len(tr) for tr in transition_results)
        log(f"  Transition: {improvement_count}/{total_probe_points} probe points recovered correct prediction")

        result["step5_transition"] = {
            "improvement_count": improvement_count,
            "total_probe_points": total_probe_points,
            "improvement_rate": improvement_count / max(1, total_probe_points),
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
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.n_tables = 4
        log("SMOKE TEST MODE: n_tables=4")

    t0 = time.time()
    result = run_model(args)
    result["total_time_min"] = round((time.time() - t0) / 60.0, 2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"phase579_{args.model}_true_edge_closure{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
