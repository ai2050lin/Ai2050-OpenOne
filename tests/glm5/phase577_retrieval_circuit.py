#!/usr/bin/env python3
"""
Phase 577: Micro-World Rule Retrieval Circuit and Binding Audit
微世界规则检索电路与绑定审计

Key shift from Phase 576:
  Phase 576 showed category subspace injection has ZERO effect on output.
  This means the model uses attention retrieval, not latent state variables.
  Phase 577 directly attacks the retrieval circuit:

Step 1: Attention retrieval graph — which heads attend from answer_pos to rules?
Step 2: Rule-match scoring — object-matchers vs category-copyers
Step 3: Head causal ablation — zero top heads, measure logit/text change
Step 4: Clean-to-corrupt patching — inject clean head outputs into corrupt run
Step 5: Logit-level analysis — target category logit, KL divergence, argmax change

NOTE: Requires eager attention (output_attentions=True). Flash/SDPA don't return
attention weights. Eager is slower but correct for circuit analysis.

Run:
  python tests/glm5/phase577_retrieval_circuit.py qwen3 --smoke
  python tests/glm5/phase577_retrieval_circuit.py qwen3
  python tests/glm5/phase577_retrieval_circuit.py glm4
  python tests/glm5/phase577_retrieval_circuit.py deepseek7b
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

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model  # noqa: E402

OUT_ROOT = Path("results/glm5_phase577_retrieval_circuit")

CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Model loading with eager attention
# ============================================================================

def load_model_eager(model_name: str):
    """Load model with eager attention (required for output_attentions)."""
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
# Truth table and prompt construction
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
    """Find first occurrence of sub_ids in full_ids starting from index start."""
    n = len(sub_ids)
    for i in range(start, len(full_ids) - n + 1):
        if full_ids[i:i + n] == sub_ids:
            return i
    return None


def find_all_subsequences(full_ids, sub_ids):
    """Find all starting positions of sub_ids in full_ids."""
    positions = []
    n = len(sub_ids)
    for i in range(len(full_ids) - n + 1):
        if full_ids[i:i + n] == sub_ids:
            positions.append(i)
    return positions


def find_symbol_in_full(tokenizer, full_ids, symbol, start=0):
    """Robustly find a symbol's token positions in full_ids.

    Tries multiple tokenization variants (with/without space, with newline).
    Returns (start_pos, end_pos) or None.
    """
    variants = [
        tokenizer.encode(symbol, add_special_tokens=False),
        tokenizer.encode(" " + symbol, add_special_tokens=False),
        tokenizer.encode("\n" + symbol, add_special_tokens=False),
        tokenizer.encode("." + symbol, add_special_tokens=False),
    ]
    # Deduplicate
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
    """Build prompt and track token positions of objects, categories, query, answer."""
    rng = random.Random(seed)
    rules = list(truth_table.items())
    rng.shuffle(rules)

    rule_lines = [f"{obj} belongs to {cat}." for obj, cat in rules]
    prompt = "Rules:\n" + "\n".join(rule_lines)
    prompt += f"\n\nQuestion: {query_object} belongs to ?\nAnswer:"

    full_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Find positions of each rule's object and category
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
            "object": obj,
            "category": cat,
            "obj_positions": list(range(obj_pos, obj_end)),
            "cat_positions": list(range(cat_pos, cat_end)),
            "is_correct": obj == query_object,
        })
        search_start = cat_end

    # Find query object position (last occurrence)
    query_result = find_symbol_in_full(tokenizer, full_ids, query_object, 0)
    query_obj_positions = None
    if query_result is not None:
        # Find ALL occurrences and take the last one
        all_q = []
        search = 0
        while True:
            r = find_symbol_in_full(tokenizer, full_ids, query_object, search)
            if r is None:
                break
            all_q.append(r)
            search = r[1]
        if all_q:
            q_pos, q_end = all_q[-1]
            query_obj_positions = list(range(q_pos, q_end))

    answer_pos = len(full_ids) - 1

    return prompt, full_ids, rule_info, query_obj_positions, answer_pos


def create_corrupted_table(truth_table, query_object, seed=42):
    """Create corrupted table where query_object's category is swapped."""
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
# Step 1-2: Attention extraction and rule-match scoring
# ============================================================================

def extract_attention_and_logits(model, tokenizer, device, prompt, answer_pos):
    """Forward pass with output_attentions=True, extract attention from answer_pos and logits."""
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_attentions=True, return_dict=True)

    # outputs.attentions: tuple of [1, n_heads, seq, seq] per layer
    # Extract row at answer_pos for each layer
    attn_rows = {}
    for l, attn in enumerate(outputs.attentions):
        if attn is not None:
            # [1, n_heads, seq, seq] -> [n_heads, seq]
            row = attn[0, :, answer_pos, :].detach().float().cpu().numpy()
            attn_rows[l] = row

    # Logits at answer_pos: [1, vocab] -> [vocab]
    logits = outputs.logits[0, answer_pos, :].detach().float().cpu().numpy()

    return attn_rows, logits


def compute_rule_match_scores(attn_row, rule_info, n_heads):
    """Compute object-match and category-copy scores for each head.

    attn_row: [n_heads, seq] — attention from answer_pos to all positions
    rule_info: list of {object, category, obj_positions, cat_positions, is_correct}

    Returns: {head: {obj_match: float, cat_copy: float}}
    """
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
        # Attention to correct rule's object
        attn_correct_obj = float(attn_row[h, correct_obj_pos].sum())
        # Attention to wrong rules' objects (mean per rule)
        attn_wrong_obj = np.mean([float(attn_row[h, p].sum()) for p in wrong_obj_pos])

        # Attention to correct rule's category
        attn_correct_cat = float(attn_row[h, correct_cat_pos].sum())
        # Attention to wrong rules' categories (mean per rule)
        attn_wrong_cat = np.mean([float(attn_row[h, p].sum()) for p in wrong_cat_pos])

        # Scores: ratio of correct to total
        total_obj = attn_correct_obj + attn_wrong_obj
        total_cat = attn_correct_cat + attn_wrong_cat

        obj_match = attn_correct_obj / (total_obj + 1e-10)
        cat_copy = attn_correct_cat / (total_cat + 1e-10)

        scores[h] = {
            "obj_match": float(obj_match),
            "cat_copy": float(cat_copy),
            "attn_correct_obj": attn_correct_obj,
            "attn_correct_cat": attn_correct_cat,
            "attn_wrong_obj": float(attn_wrong_obj),
            "attn_wrong_cat": float(attn_wrong_cat),
        }

    return scores


# ============================================================================
# Step 3: Head causal ablation
# ============================================================================

def get_head_dims(model):
    """Get n_heads, d_head from model config."""
    config = model.config
    n_heads = config.num_attention_heads
    d_model = config.hidden_size
    d_head = d_model // n_heads
    return n_heads, d_head, d_model


def make_ablation_hook(head_indices, d_head, o_proj_weight):
    """Create hook that ablates specific heads by subtracting their o_proj contribution.

    o_proj_weight: [d_model, n_heads * d_head]
    For head h: contribution = x[:, :, h*d_head:(h+1)*d_head] @ W[:, h*d_head:(h+1)*d_head].T
    """
    W = o_proj_weight.detach().float().cpu()
    head_W = {}
    for h in head_indices:
        head_W[h] = W[:, h * d_head:(h + 1) * d_head].T.numpy()  # [d_head, d_model]

    def hook(module, input, output):
        x = input[0]  # [batch, seq, n_heads * d_head]
        modified = output.clone()
        for h in head_indices:
            x_h = x[:, :, h * d_head:(h + 1) * d_head].detach().float().cpu().numpy()
            contribution = x_h @ head_W[h]  # [batch, seq, d_model]
            modified -= torch.tensor(contribution, device=output.device, dtype=output.dtype)
        return modified

    return hook


def forward_with_ablation(model, tokenizer, device, prompt, layers, head_groups, d_head):
    """Forward pass with head ablation, return logits at last position.

    head_groups: {layer_idx: [head_indices]} — which heads to ablate at each layer
    """
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    hooks = []
    for l, heads in head_groups.items():
        if heads and l < len(layers):
            o_proj = layers[l].self_attn.o_proj
            h = make_ablation_hook(heads, d_head, o_proj.weight)
            hooks.append(o_proj.register_forward_hook(h))

    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, return_dict=True)
        logits = outputs.logits[0, -1, :].detach().float().cpu().numpy()
    finally:
        for h in hooks:
            h.remove()

    return logits


def generate_with_ablation(model, tokenizer, device, prompt, layers, head_groups, d_head, max_new_tokens=5):
    """Generate text with head ablation."""
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)

    hooks = []
    for l, heads in head_groups.items():
        if heads and l < len(layers):
            o_proj = layers[l].self_attn.o_proj
            h = make_ablation_hook(heads, d_head, o_proj.weight)
            hooks.append(o_proj.register_forward_hook(h))

    try:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        for h in hooks:
            h.remove()

    return text


# ============================================================================
# Step 4: Clean-to-corrupt patching
# ============================================================================

def make_capture_hook(storage, head_indices, d_head):
    """Capture o_proj input for specified heads."""
    def hook(module, input, output):
        x = input[0]  # [batch, seq, n_heads * d_head]
        for h in head_indices:
            storage[h] = x[:, :, h * d_head:(h + 1) * d_head].detach().cpu().clone()
    return hook


def make_patch_hook(clean_storage, head_indices, d_head):
    """Replace corrupt head inputs with clean versions."""
    def hook(module, input, output):
        x = input[0]  # [batch, seq, n_heads * d_head] — corrupt
        x_new = x.clone()
        for h in head_indices:
            if h in clean_storage:
                clean_slice = clean_storage[h].to(x_new.device, x_new.dtype)
                x_new[:, :, h * d_head:(h + 1) * d_head] = clean_slice
        # Recompute output with patched input
        return module.forward(x_new)
    return hook


def patching_test(model, tokenizer, device, layers, truth_table, query_object,
                  top_heads, d_head, categories, cat_token_ids, seed=42):
    """Test if patching clean head outputs into corrupt run recovers correct answer.

    top_heads: {layer_idx: [head_indices]}
    """
    results = {}

    # Build clean and corrupt prompts
    clean_prompt, _, clean_rule_info, _, clean_answer_pos = build_prompt_with_positions(
        tokenizer, truth_table, query_object, seed=seed)
    corrupt_table = create_corrupted_table(truth_table, query_object, seed=seed + 1)
    corrupt_prompt, _, corrupt_rule_info, _, corrupt_answer_pos = build_prompt_with_positions(
        tokenizer, corrupt_table, query_object, seed=seed + 1)

    correct_cat = truth_table[query_object]
    corrupt_cat = corrupt_table[query_object]

    all_heads = []
    for l, heads in top_heads.items():
        all_heads.extend([(l, h) for h in heads])

    # Step 1: Clean run — capture head inputs (use original prompt for attention)
    # But for logit analysis, append " c" to get second-token logits
    clean_logit_prompt = clean_prompt + " c"
    corrupt_logit_prompt = corrupt_prompt + " c"
    
    clean_input_ids = torch.tensor([tokenizer.encode(clean_prompt, add_special_tokens=False)], device=device)
    clean_logit_ids = torch.tensor([tokenizer.encode(clean_logit_prompt, add_special_tokens=False)], device=device)
    clean_storage = {}
    capture_hooks = []
    for l, heads in top_heads.items():
        if heads and l < len(layers):
            o_proj = layers[l].self_attn.o_proj
            ch = make_capture_hook(clean_storage, heads, d_head)
            capture_hooks.append(o_proj.register_forward_hook(ch))

    try:
        # Capture head inputs from original prompt
        with torch.inference_mode():
            _ = model(input_ids=clean_input_ids, return_dict=True)
        # Get clean logits from two-step prompt (no hooks needed for this)
        with torch.inference_mode():
            clean_logit_out = model(input_ids=clean_logit_ids, return_dict=True)
        clean_logits = clean_logit_out.logits[0, -1, :].detach().float().cpu().numpy()
    finally:
        for h in capture_hooks:
            h.remove()

    # Step 2: Corrupt baseline (two-step prompt for logits)
    corrupt_logit_ids = torch.tensor([tokenizer.encode(corrupt_logit_prompt, add_special_tokens=False)], device=device)
    corrupt_input_ids = torch.tensor([tokenizer.encode(corrupt_prompt, add_special_tokens=False)], device=device)
    with torch.inference_mode():
        corrupt_logit_out = model(input_ids=corrupt_logit_ids, return_dict=True)
    corrupt_logits = corrupt_logit_out.logits[0, -1, :].detach().float().cpu().numpy()

    # Step 3: Corrupt + patch (use original prompt for head patching, then two-step for logits)
    # We need to patch during the forward pass, so we use the corrupt_logit_prompt
    # and apply patch hooks during that forward pass
    patch_hooks = []
    for l, heads in top_heads.items():
        if heads and l < len(layers):
            o_proj = layers[l].self_attn.o_proj
            ph = make_patch_hook(clean_storage, heads, d_head)
            patch_hooks.append(o_proj.register_forward_hook(ph))

    try:
        with torch.inference_mode():
            patched_out = model(input_ids=corrupt_logit_ids, return_dict=True)
        patched_logits = patched_out.logits[0, -1, :].detach().float().cpu().numpy()
    finally:
        for h in patch_hooks:
            h.remove()

    # Analyze logits for each category
    def get_cat_logits(logits):
        return {cat: float(logits[tid]) for cat, tid in cat_token_ids.items()}

    clean_cl = get_cat_logits(clean_logits)
    corrupt_cl = get_cat_logits(corrupt_logits)
    patched_cl = get_cat_logits(patched_logits)

    # Predicted category = argmax
    clean_pred = max(clean_cl, key=clean_cl.get)
    corrupt_pred = max(corrupt_cl, key=corrupt_cl.get)
    patched_pred = max(patched_cl, key=patched_cl.get)

    # Recovery: did patching move prediction towards clean?
    patch_recovered = patched_pred == correct_cat
    patch_changed = patched_pred != corrupt_pred

    # KL divergence (softmax)
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    clean_probs = softmax(np.array([clean_logits[tid] for tid in cat_token_ids.values()]))
    corrupt_probs = softmax(np.array([corrupt_logits[tid] for tid in cat_token_ids.values()]))
    patched_probs = softmax(np.array([patched_logits[tid] for tid in cat_token_ids.values()]))

    kl_corrupt = float(np.sum(clean_probs * (np.log(clean_probs + 1e-10) - np.log(corrupt_probs + 1e-10))))
    kl_patched = float(np.sum(clean_probs * (np.log(clean_probs + 1e-10) - np.log(patched_probs + 1e-10))))

    results = {
        "correct_cat": correct_cat,
        "corrupt_cat": corrupt_cat,
        "clean_pred": clean_pred,
        "corrupt_pred": corrupt_pred,
        "patched_pred": patched_pred,
        "patch_recovered": patch_recovered,
        "patch_changed": patch_changed,
        "clean_cat_logits": clean_cl,
        "corrupt_cat_logits": corrupt_cl,
        "patched_cat_logits": patched_cl,
        "kl_corrupt_vs_clean": kl_corrupt,
        "kl_patched_vs_clean": kl_patched,
        "n_heads_patched": len(all_heads),
    }

    log(f"    Patch: correct={correct_cat}, corrupt_pred={corrupt_pred}, patched_pred={patched_pred}, "
        f"recovered={patch_recovered}, KL: {kl_corrupt:.4f}→{kl_patched:.4f}")

    return results


# ============================================================================
# Main analysis
# ============================================================================

def run_model(args):
    model, tokenizer, device = load_model_eager(args.model)
    try:
        info = get_model_info(model, args.model)
        layers = get_layers(model)
        n_layers = info.n_layers
        n_heads, d_head, d_model = get_head_dims(model)

        log(f"{args.model}: n_layers={n_layers}, n_heads={n_heads}, d_head={d_head}, d_model={d_model}")

        objects = CANDIDATE_OBJECTS[:8]
        categories = CANDIDATE_CATEGORIES[:4]

        # Get token IDs for each category — find the FULL token sequence
        # Since categories share first token "c", we need the full sequence
        cat_token_seqs = {}
        for cat in categories:
            # Try with space (as it appears after "Answer: ")
            ids = tokenizer.encode(" " + cat, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(cat, add_special_tokens=False)
            cat_token_seqs[cat] = ids
        log(f"Category token sequences: {cat_token_seqs}")

        # For logit analysis, we use the SECOND token of each category
        # (first token "c" is shared, second token distinguishes: 12, 77, 33, 59)
        cat_token_ids = {}
        for cat, ids in cat_token_seqs.items():
            if len(ids) >= 2:
                cat_token_ids[cat] = ids[1]  # second token
            else:
                cat_token_ids[cat] = ids[0]
        log(f"Category distinguishing tokens (2nd): {cat_token_ids}")

        # Build truth tables
        n_tables = args.n_tables
        truth_tables = build_truth_tables(objects, categories, n_tables)
        log(f"Built {n_tables} truth tables, {len(objects)} objects, {len(categories)} categories")

        result = {
            "phase": 577,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers,
            "n_heads": n_heads,
            "d_head": d_head,
            "d_model": d_model,
            "n_tables": n_tables,
            "cat_token_ids": {k: int(v) for k, v in cat_token_ids.items()},
        }

        # ================================================================
        # Step 1-2: Attention extraction and rule-match scoring
        # ================================================================
        log("=== Step 1-2: Attention extraction and rule-match scoring ===")

        # Collect scores across all samples
        all_obj_scores = np.zeros((n_layers, n_heads))  # mean obj_match score
        all_cat_scores = np.zeros((n_layers, n_heads))  # mean cat_copy score
        n_samples_scored = 0

        # Also collect baseline logits
        baseline_logits_list = []
        sample_data = []  # (prompt, rule_info, answer_pos, correct_cat)

        sample_count = 0
        total_samples = n_tables * len(objects)
        for tt_idx, tt in enumerate(truth_tables):
            for obj in objects:
                prompt, full_ids, rule_info, q_pos, answer_pos = build_prompt_with_positions(
                    tokenizer, tt, obj, seed=tt_idx * 100 + hash(obj) % 1000)

                if not rule_info or answer_pos >= len(full_ids):
                    if sample_count == 0:
                        log(f"  DEBUG: First sample failed! rule_info={rule_info}, answer_pos={answer_pos}, len(full_ids)={len(full_ids)}")
                        log(f"  DEBUG: prompt='{prompt[:100]}...'")
                        log(f"  DEBUG: full_ids[:20]={full_ids[:20]}")
                    continue

                correct_cat = tt[obj]

                # Forward pass with attention
                try:
                    attn_rows, logits = extract_attention_and_logits(
                        model, tokenizer, device, prompt, answer_pos)
                except Exception as e:
                    log(f"  WARNING: attention extraction failed for tt={tt_idx}, obj={obj}: {e}")
                    continue

                # Compute scores for each layer
                for l in range(n_layers):
                    if l in attn_rows:
                        scores = compute_rule_match_scores(attn_rows[l], rule_info, n_heads)
                        for h in range(n_heads):
                            if h in scores:
                                all_obj_scores[l, h] += scores[h]["obj_match"]
                                all_cat_scores[l, h] += scores[h]["cat_copy"]

                n_samples_scored += 1
                sample_count += 1

                # Save for ablation
                baseline_logits_list.append(logits)
                sample_data.append({
                    "prompt": prompt,
                    "rule_info": rule_info,
                    "answer_pos": answer_pos,
                    "correct_cat": correct_cat,
                    "tt_idx": tt_idx,
                    "obj": obj,
                })

                if sample_count % 20 == 0:
                    log(f"  Processed {sample_count}/{total_samples} samples for attention extraction")

        all_obj_scores /= max(1, n_samples_scored)
        all_cat_scores /= max(1, n_samples_scored)
        log(f"  Attention extraction complete: {n_samples_scored} samples scored")

        # ================================================================
        # Step 2: Identify top heads
        # ================================================================
        log("=== Step 2: Identify top retrieval heads ===")

        # Flatten and rank
        obj_flat = [(l, h, all_obj_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]
        cat_flat = [(l, h, all_cat_scores[l, h]) for l in range(n_layers) for h in range(n_heads)]

        obj_flat.sort(key=lambda x: x[2], reverse=True)
        cat_flat.sort(key=lambda x: x[2], reverse=True)

        top_n = 10
        top_obj_heads = obj_flat[:top_n]
        top_cat_heads = cat_flat[:top_n]

        log(f"  Top-10 object-matching heads (score):")
        for l, h, s in top_obj_heads:
            log(f"    L{l}H{h}: obj_match={s:.4f}")

        log(f"  Top-10 category-copy heads (score):")
        for l, h, s in top_cat_heads:
            log(f"    L{l}H{h}: cat_copy={s:.4f}")

        # Also compute mean scores per layer
        mean_obj_per_layer = all_obj_scores.mean(axis=1)
        mean_cat_per_layer = all_cat_scores.mean(axis=1)
        peak_obj_layer = int(np.argmax(mean_obj_per_layer))
        peak_cat_layer = int(np.argmax(mean_cat_per_layer))
        log(f"  Peak object-match layer: L{peak_obj_layer} (mean={mean_obj_per_layer[peak_obj_layer]:.4f})")
        log(f"  Peak category-copy layer: L{peak_cat_layer} (mean={mean_cat_per_layer[peak_cat_layer]:.4f})")

        result["step2_top_heads"] = {
            "top_obj_heads": [{"layer": l, "head": h, "score": float(s)} for l, h, s in top_obj_heads],
            "top_cat_heads": [{"layer": l, "head": h, "score": float(s)} for l, h, s in top_cat_heads],
            "mean_obj_per_layer": mean_obj_per_layer.tolist(),
            "mean_cat_per_layer": mean_cat_per_layer.tolist(),
            "peak_obj_layer": peak_obj_layer,
            "peak_cat_layer": peak_cat_layer,
            "n_samples_scored": n_samples_scored,
        }

        # ================================================================
        # Step 3: Head causal ablation
        # ================================================================
        log("=== Step 3: Head causal ablation ===")

        # Define ablation configs
        def heads_to_groups(head_list):
            groups = {}
            for l, h in head_list:
                groups.setdefault(l, []).append(h)
            return groups

        configs = {
            "baseline": {},
            "top1_obj": heads_to_groups([(top_obj_heads[0][0], top_obj_heads[0][1])]),
            "top3_obj": heads_to_groups([(l, h) for l, h, _ in top_obj_heads[:3]]),
            "top5_obj": heads_to_groups([(l, h) for l, h, _ in top_obj_heads[:5]]),
            "top1_cat": heads_to_groups([(top_cat_heads[0][0], top_cat_heads[0][1])]),
            "top3_cat": heads_to_groups([(l, h) for l, h, _ in top_cat_heads[:3]]),
            "top5_cat": heads_to_groups([(l, h) for l, h, _ in top_cat_heads[:5]]),
            "top5_combined": heads_to_groups(
                [(l, h) for l, h, _ in top_obj_heads[:5]] + [(l, h) for l, h, _ in top_cat_heads[:5]]),
            "top10_combined": heads_to_groups(
                [(l, h) for l, h, _ in top_obj_heads[:10]] + [(l, h) for l, h, _ in top_cat_heads[:10]]),
        }

        # Select samples for ablation (1 per table)
        ablation_samples = []
        for tt_idx in range(min(n_tables, 10)):
            for s in sample_data:
                if s["tt_idx"] == tt_idx:
                    ablation_samples.append(s)
                    break

        log(f"  Testing {len(configs)} configs on {len(ablation_samples)} samples")

        ablation_results = {}
        for config_name, head_groups in configs.items():
            log(f"  Config: {config_name}")

            cat_logit_changes = []
            argmax_changes = 0
            kl_divergences = []
            correct_at_baseline = 0
            correct_after_ablation = 0
            n_tested = 0

            for i, s in enumerate(ablation_samples):
                # Use a two-step prompt for logit analysis:
                # Append " c" so the model predicts the SECOND token (which distinguishes categories)
                # First token "c" is shared by all categories, so we need the second token position
                logit_prompt = s["prompt"] + " c"

                if config_name == "baseline":
                    baseline_logits_step2 = forward_with_ablation(
                        model, tokenizer, device, logit_prompt, layers, {}, d_head)
                    ablated_logits = baseline_logits_step2
                else:
                    baseline_logits_step2 = forward_with_ablation(
                        model, tokenizer, device, logit_prompt, layers, {}, d_head)
                    ablated_logits = forward_with_ablation(
                        model, tokenizer, device, logit_prompt, layers, head_groups, d_head)

                # Analyze
                correct_cat = s["correct_cat"]
                correct_tid = cat_token_ids[correct_cat]

                # All category logits (at the second-token position)
                baseline_cl = {cat: float(baseline_logits_step2[tid]) for cat, tid in cat_token_ids.items()}
                ablated_cl = {cat: float(ablated_logits[tid]) for cat, tid in cat_token_ids.items()}

                baseline_pred = max(baseline_cl, key=baseline_cl.get)
                ablated_pred = max(ablated_cl, key=ablated_cl.get)

                # Logit change for correct category
                logit_change = ablated_cl[correct_cat] - baseline_cl[correct_cat]
                cat_logit_changes.append(float(logit_change))

                # Argmax change
                if baseline_pred != ablated_pred:
                    argmax_changes += 1

                # KL divergence (over category tokens)
                def softmax_4(x):
                    e = np.exp(np.array(list(x.values())))
                    return e / e.sum()

                bp = softmax_4(baseline_cl)
                ap = softmax_4(ablated_cl)
                kl = float(np.sum(bp * (np.log(bp + 1e-10) - np.log(ap + 1e-10))))
                kl_divergences.append(kl)

                if baseline_pred == correct_cat:
                    correct_at_baseline += 1
                if ablated_pred == correct_cat:
                    correct_after_ablation += 1

                n_tested += 1

                if i < 3 and config_name != "baseline":
                    log(f"    Sample {i}: correct={correct_cat}, baseline_pred={baseline_pred}, "
                        f"ablated_pred={ablated_pred}, logit_change={logit_change:.4f}, KL={kl:.6f}")

            ablation_results[config_name] = {
                "n_tested": n_tested,
                "mean_cat_logit_change": float(np.mean(cat_logit_changes)) if cat_logit_changes else 0,
                "mean_abs_cat_logit_change": float(np.mean(np.abs(cat_logit_changes))) if cat_logit_changes else 0,
                "argmax_change_rate": argmax_changes / max(1, n_tested),
                "mean_kl_divergence": float(np.mean(kl_divergences)) if kl_divergences else 0,
                "correct_at_baseline": correct_at_baseline,
                "correct_after_ablation": correct_after_ablation,
                "accuracy_change": (correct_after_ablation - correct_at_baseline) / max(1, n_tested),
            }

            r = ablation_results[config_name]
            log(f"    Summary: logit_change={r['mean_cat_logit_change']:.4f}, "
                f"argmax_change={r['argmax_change_rate']:.3f}, "
                f"KL={r['mean_kl_divergence']:.6f}, "
                f"acc={correct_at_baseline}→{correct_after_ablation}")

        result["step3_ablation"] = ablation_results

        # ================================================================
        # Step 3b: Text generation with ablation (subset)
        # ================================================================
        log("=== Step 3b: Text generation with ablation ===")

        gen_configs = ["baseline", "top5_obj", "top5_cat", "top5_combined", "top10_combined"]
        gen_samples = ablation_samples[:5]
        gen_results = {}

        for config_name in gen_configs:
            head_groups = configs[config_name]
            outputs = []
            for s in gen_samples:
                if config_name == "baseline":
                    text = generate_with_ablation(
                        model, tokenizer, device, s["prompt"], layers, {}, d_head, max_new_tokens=5)
                else:
                    text = generate_with_ablation(
                        model, tokenizer, device, s["prompt"], layers, head_groups, d_head, max_new_tokens=5)

                # Detect category in output
                text_lower = text.lower()
                detected = [c for c in categories if c.lower() in text_lower]
                detected_cat = detected[0] if len(detected) == 1 else ("none" if len(detected) == 0 else "ambiguous")

                outputs.append({
                    "correct_cat": s["correct_cat"],
                    "generated": text,
                    "detected_cat": detected_cat,
                    "correct": detected_cat == s["correct_cat"],
                })

            correct_count = sum(1 for o in outputs if o["correct"])
            gen_results[config_name] = {
                "outputs": outputs,
                "correct_rate": correct_count / max(1, len(outputs)),
            }
            log(f"    {config_name}: correct_rate={correct_count}/{len(outputs)}")

        result["step3b_generation"] = gen_results

        # ================================================================
        # Step 4: Clean-to-corrupt patching
        # ================================================================
        log("=== Step 4: Clean-to-corrupt patching ===")

        # Use top-5 combined heads for patching
        patch_head_groups = configs["top5_combined"]
        patch_samples = []
        for tt_idx in range(min(n_tables, 5)):
            tt = truth_tables[tt_idx]
            obj = objects[tt_idx % len(objects)]
            patch_samples.append((tt, obj, tt_idx * 200))

        patch_results = []
        for tt, obj, seed in patch_samples:
            pr = patching_test(
                model, tokenizer, device, layers, tt, obj,
                patch_head_groups, d_head, categories, cat_token_ids, seed=seed)
            patch_results.append(pr)

        n_recovered = sum(1 for pr in patch_results if pr["patch_recovered"])
        n_changed = sum(1 for pr in patch_results if pr["patch_changed"])
        result["step4_patching"] = {
            "tests": patch_results,
            "n_tests": len(patch_results),
            "n_recovered": n_recovered,
            "n_changed": n_changed,
            "recovery_rate": n_recovered / max(1, len(patch_results)),
            "change_rate": n_changed / max(1, len(patch_results)),
            "mean_kl_corrupt": float(np.mean([pr["kl_corrupt_vs_clean"] for pr in patch_results])),
            "mean_kl_patched": float(np.mean([pr["kl_patched_vs_clean"] for pr in patch_results])),
        }
        log(f"  Patching: recovered={n_recovered}/{len(patch_results)}, "
            f"changed={n_changed}/{len(patch_results)}, "
            f"KL: {result['step4_patching']['mean_kl_corrupt']:.4f}→{result['step4_patching']['mean_kl_patched']:.4f}")

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
    out_path = out_dir / f"phase577_{args.model}_retrieval_circuit{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
