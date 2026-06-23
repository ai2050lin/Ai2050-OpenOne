#!/usr/bin/env python3
"""
Phase 584: Causal Repair of Choice Gate, Value Retrieval Gate, and Polarity Gate
中间选择门、值检索门与极性读出门的因果修复

Phase 583 found three gate failures:
1. Qwen3: C_wrong_cat (wrong intermediate category selection)
2. DS7B: Val_fail (correct C but wrong V - value retrieval fails)
3. GLM4/DS7B: yes-bias (polarity readout bias)

Phase 584 attempts causal REPAIR:
  Part A: Choice gate repair - use gold-category injection + forced scaffold
  Part B: Value retrieval gate repair - enhance value-copy via prompt engineering
  Part C: Polarity gate repair - multiple answer formats + system instruction
  Part D: GLM4 bypass mechanism - compare O→C→V vs O→R→V direct path

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

OUT_ROOT = Path("results/glm5_phase584_gate_repair")

CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_RELATIONS = ["r31", "r64"]
CANDIDATE_VALUES = ["v05", "v91", "v22", "v48"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]

CATEGORY_OBJECTS = {
    "水果": ["苹果", "香蕉", "梨", "桃子", "葡萄"],
    "动物": ["老虎", "狗", "猫", "鸟", "鱼"],
    "天体": ["地球", "太阳", "月亮", "火星", "星星"],
    "工具": ["锤子", "剪刀", "斧头", "锯子", "钳子"],
    "家具": ["椅子", "桌子", "床", "柜子", "沙发"],
    "交通工具": ["汽车", "飞机", "船", "自行车", "火车"],
}
CATEGORY_NEGATIVES = {
    "水果": ["老虎", "地球", "汽车", "椅子", "石头"],
    "动物": ["苹果", "地球", "汽车", "椅子", "石头"],
    "天体": ["苹果", "老虎", "汽车", "椅子", "石头"],
    "工具": ["苹果", "老虎", "地球", "汽车", "椅子"],
    "家具": ["苹果", "老虎", "地球", "汽车", "锤子"],
    "交通工具": ["苹果", "老虎", "地球", "椅子", "锤子"],
}

ANSWER_FORMATS = {
    "single": ("是", "否"),
    "double": ("是的", "不是"),
    "belong": ("属于", "不属于"),
    "english": ("yes", "no"),
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl)
        log(f"Loaded {model_name} with flash_attention_2")
    except Exception:
        attn_impl = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager")

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Loaded {model_name}: class={type(model).__name__}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


def compute_full_string_logprob(model, tokenizer, device, prompt, answer_str):
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
    return {ans: compute_full_string_logprob(model, tokenizer, device, prompt, ans) for ans in answer_strings}


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


def build_cat_rel_truth_tables(categories, relations, values, seed=42):
    rng = random.Random(seed)
    mapping = {}
    for cat in categories:
        for rel in relations:
            mapping[(cat, rel)] = rng.choice(values)
    return mapping


def build_two_hop_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
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


def build_gold_cat_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    """Provide correct intermediate category explicitly."""
    rng = random.Random(seed)
    correct_cat = oc_table.get(query_object)
    if correct_cat is None:
        return None, None, None
    oc_rules = list(oc_table.items())
    rng.shuffle(oc_rules)
    oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]
    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]
    prompt = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(crv_lines)
    prompt += f"\n\n{query_object} belongs to {correct_cat}."
    prompt += f"\nQuestion: {query_object} {query_relation} ?\nAnswer:"
    correct_val = crv_table.get((correct_cat, query_relation))
    return prompt, correct_cat, correct_val


def build_proof_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    """Proof-style scaffold."""
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
# Part A: Choice gate repair
# ============================================================================

def run_choice_gate_repair(model, tokenizer, device, n_tables, objects, relations,
                            categories, values):
    """Repair C_wrong_cat by providing gold category and proof scaffold."""
    log("--- Part A: Choice gate repair ---")

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

                # Baseline: direct two-hop
                direct_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                direct_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, direct_prompt, values)
                direct_pred = max(direct_lps, key=lambda c: direct_lps[c][0])
                direct_correct = direct_pred == correct_val

                # Also check cat prediction
                cat_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, direct_prompt, categories)
                pred_cat = max(cat_lps, key=lambda c: cat_lps[c][0])
                cat_correct = pred_cat == correct_cat

                # Repair 1: Gold category
                gold_prompt, _, _ = build_gold_cat_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                if gold_prompt:
                    gold_lps = compute_full_string_logprob_batch(
                        model, tokenizer, device, gold_prompt, values)
                    gold_pred = max(gold_lps, key=lambda c: gold_lps[c][0])
                    gold_correct = gold_pred == correct_val
                else:
                    gold_pred = "none"
                    gold_correct = False

                # Repair 2: Proof scaffold
                proof_prompt, _, _ = build_proof_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                proof_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, proof_prompt, values)
                proof_pred = max(proof_lps, key=lambda c: proof_lps[c][0])
                proof_correct = proof_pred == correct_val

                # Only track cases where cat was wrong (the repair target)
                is_target = not cat_correct

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "pred_cat": pred_cat, "cat_correct": cat_correct,
                    "direct_pred": direct_pred, "direct_correct": direct_correct,
                    "gold_pred": gold_pred, "gold_correct": gold_correct,
                    "proof_pred": proof_pred, "proof_correct": proof_correct,
                    "is_repair_target": is_target,
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_direct = sum(1 for r in results if r["direct_correct"])
    n_gold = sum(1 for r in results if r["gold_correct"])
    n_proof = sum(1 for r in results if r["proof_correct"])

    # Repair targets: cat was wrong
    targets = [r for r in results if r["is_repair_target"]]
    n_target_direct = sum(1 for r in targets if r["direct_correct"])
    n_target_gold = sum(1 for r in targets if r["gold_correct"])
    n_target_proof = sum(1 for r in targets if r["proof_correct"])

    log(f"  Overall: direct={n_direct}/{len(results)}, gold={n_gold}/{len(results)}, proof={n_proof}/{len(results)}")
    log(f"  Repair targets (cat_wrong): {len(targets)} samples")
    if targets:
        log(f"    direct={n_target_direct}/{len(targets)}, gold={n_target_gold}/{len(targets)}, proof={n_target_proof}/{len(targets)}")

    return {
        "n_samples": len(results),
        "direct_accuracy": n_direct / max(1, len(results)),
        "gold_accuracy": n_gold / max(1, len(results)),
        "proof_accuracy": n_proof / max(1, len(results)),
        "n_repair_targets": len(targets),
        "target_direct_accuracy": n_target_direct / max(1, len(targets)),
        "target_gold_accuracy": n_target_gold / max(1, len(targets)),
        "target_proof_accuracy": n_target_proof / max(1, len(targets)),
        "results": results,
    }


# ============================================================================
# Part B: Value retrieval gate repair (DS7B focus)
# ============================================================================

def run_value_retrieval_repair(model, tokenizer, device, n_tables, objects, relations,
                                categories, values):
    """Repair Val_fail by providing relation context and structured prompts."""
    log("--- Part B: Value retrieval gate repair ---")

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

                # Baseline: gold category + direct query
                gold_prompt, _, _ = build_gold_cat_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                if gold_prompt is None:
                    continue
                gold_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, gold_prompt, values)
                gold_pred = max(gold_lps, key=lambda c: gold_lps[c][0])
                gold_correct = gold_pred == correct_val
                gold_margin = gold_lps[correct_val][0] - max(
                    v for k, v in [(k, lp[0]) for k, lp in gold_lps.items()] if k != correct_val)

                # Repair 1: Explicit relation emphasis
                rng = random.Random(tt_idx * 100)
                crv_rules = list(crv_tt.items())
                rng.shuffle(crv_rules)
                crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]
                repair1_prompt = "Rules:\n" + "\n".join(crv_lines)
                repair1_prompt += f"\n\n{obj} belongs to {correct_cat}."
                repair1_prompt += f"\nWhat is the value of {correct_cat} {rel}?"
                repair1_prompt += "\nAnswer:"
                r1_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, repair1_prompt, values)
                r1_pred = max(r1_lps, key=lambda c: r1_lps[c][0])
                r1_correct = r1_pred == correct_val

                # Repair 2: Only show relevant CRV rules (filter by relation)
                relevant_crv = [((c, r), v) for (c, r), v in crv_rules if r == rel]
                repair2_prompt = "Rules:\n" + "\n".join([f"{c} {r} {v}." for (c, r), v in relevant_crv])
                repair2_prompt += f"\n\n{obj} belongs to {correct_cat}."
                repair2_prompt += f"\nQuestion: {correct_cat} {rel} ?\nAnswer:"
                r2_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, repair2_prompt, values)
                r2_pred = max(r2_lps, key=lambda c: r2_lps[c][0])
                r2_correct = r2_pred == correct_val

                # Only track cases where gold was wrong (the repair target)
                is_target = not gold_correct

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "gold_pred": gold_pred, "gold_correct": gold_correct,
                    "gold_margin": gold_margin,
                    "repair1_pred": r1_pred, "repair1_correct": r1_correct,
                    "repair2_pred": r2_pred, "repair2_correct": r2_correct,
                    "is_repair_target": is_target,
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_gold = sum(1 for r in results if r["gold_correct"])
    n_r1 = sum(1 for r in results if r["repair1_correct"])
    n_r2 = sum(1 for r in results if r["repair2_correct"])

    targets = [r for r in results if r["is_repair_target"]]
    n_t_gold = sum(1 for r in targets if r["gold_correct"])
    n_t_r1 = sum(1 for r in targets if r["repair1_correct"])
    n_t_r2 = sum(1 for r in targets if r["repair2_correct"])

    log(f"  Overall: gold={n_gold}/{len(results)}, repair1={n_r1}/{len(results)}, repair2={n_r2}/{len(results)}")
    log(f"  Repair targets (gold_wrong): {len(targets)} samples")
    if targets:
        log(f"    gold={n_t_gold}/{len(targets)}, repair1={n_t_r1}/{len(targets)}, repair2={n_t_r2}/{len(targets)}")

    return {
        "n_samples": len(results),
        "gold_accuracy": n_gold / max(1, len(results)),
        "repair1_accuracy": n_r1 / max(1, len(results)),
        "repair2_accuracy": n_r2 / max(1, len(results)),
        "n_repair_targets": len(targets),
        "target_repair1_accuracy": n_t_r1 / max(1, len(targets)),
        "target_repair2_accuracy": n_t_r2 / max(1, len(targets)),
        "results": results,
    }


def crv_table_to_list(crv_table):
    return list(crv_table.items())


# ============================================================================
# Part C: Polarity gate repair
# ============================================================================

def run_polarity_gate_repair(model, tokenizer, device):
    """Repair yes-bias using multiple interventions."""
    log("--- Part C: Polarity gate repair ---")

    test_cases = []
    for cat in ["水果", "动物", "天体", "工具"]:
        for obj in CATEGORY_OBJECTS[cat][:2] + CATEGORY_NEGATIVES[cat][:2]:
            is_positive = obj in CATEGORY_OBJECTS[cat]
            test_cases.append((obj, cat, is_positive))

    results = []

    for obj, cat, is_positive in test_cases:
        expected_pos = is_positive

        # Baseline: single format
        base_prompt = f"Question: {obj}是不是{cat}？\nAnswer:"
        base_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, base_prompt, ["是", "否"])
        base_pred = max(base_lps, key=lambda c: base_lps[c][0])
        base_correct = (base_pred == "是") == expected_pos
        base_margin = base_lps["是"][0] - base_lps["否"][0]

        # Repair 1: Best format from Phase 583
        # For Qwen3 use english, for GLM4 use double, for DS7B use english
        best_fmt = "english"  # Will be overridden per model externally
        best_pos, best_neg = ANSWER_FORMATS[best_fmt]
        r1_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, base_prompt, [best_pos, best_neg])
        r1_pred = max(r1_lps, key=lambda c: r1_lps[c][0])
        r1_correct = (r1_pred == best_pos) == expected_pos
        r1_margin = r1_lps[best_pos][0] - r1_lps[best_neg][0]

        # Repair 2: System instruction to not default to "yes"
        r2_prompt = f"Instructions: Answer truthfully. Do not default to 'yes'.\nQuestion: {obj}是不是{cat}？\nAnswer:"
        r2_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, r2_prompt, ["是", "否"])
        r2_pred = max(r2_lps, key=lambda c: r2_lps[c][0])
        r2_correct = (r2_pred == "是") == expected_pos
        r2_margin = r2_lps["是"][0] - r2_lps["否"][0]

        # Repair 3: Explicit negation + best format
        if not is_positive:
            alt_cat = "动物" if cat == "水果" else "水果"
            r3_prompt = f"Rules:\n{obj} 属于 {alt_cat}, 不属于 {cat}.\n\nQuestion: {obj}是不是{cat}？\nAnswer:"
        else:
            r3_prompt = f"Rules:\n{obj} 属于 {cat}.\n\nQuestion: {obj}是不是{cat}？\nAnswer:"
        r3_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, r3_prompt, [best_pos, best_neg])
        r3_pred = max(r3_lps, key=lambda c: r3_lps[c][0])
        r3_correct = (r3_pred == best_pos) == expected_pos
        r3_margin = r3_lps[best_pos][0] - r3_lps[best_neg][0]

        # Repair 4: Forced choice format
        r4_prompt = f"Question: {obj}是不是{cat}？\nChoose one: yes or no\nAnswer:"
        r4_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, r4_prompt, ["yes", "no"])
        r4_pred = max(r4_lps, key=lambda c: r4_lps[c][0])
        r4_correct = (r4_pred == "yes") == expected_pos
        r4_margin = r4_lps["yes"][0] - r4_lps["no"][0]

        results.append({
            "object": obj, "category": cat, "is_positive": is_positive,
            "base_correct": base_correct, "base_margin": base_margin,
            "r1_correct": r1_correct, "r1_margin": r1_margin,  # best format
            "r2_correct": r2_correct, "r2_margin": r2_margin,  # system instruction
            "r3_correct": r3_correct, "r3_margin": r3_margin,  # explicit rule + best format
            "r4_correct": r4_correct, "r4_margin": r4_margin,  # forced choice
        })

    n_base = sum(1 for r in results if r["base_correct"])
    n_r1 = sum(1 for r in results if r["r1_correct"])
    n_r2 = sum(1 for r in results if r["r2_correct"])
    n_r3 = sum(1 for r in results if r["r3_correct"])
    n_r4 = sum(1 for r in results if r["r4_correct"])

    # Negatives only
    neg = [r for r in results if not r["is_positive"]]
    n_neg_base = sum(1 for r in neg if r["base_correct"])
    n_neg_r1 = sum(1 for r in neg if r["r1_correct"])
    n_neg_r2 = sum(1 for r in neg if r["r2_correct"])
    n_neg_r3 = sum(1 for r in neg if r["r3_correct"])
    n_neg_r4 = sum(1 for r in neg if r["r4_correct"])

    log(f"  Overall: base={n_base}/{len(results)}, fmt={n_r1}, sys={n_r2}, rule+fmt={n_r3}, forced={n_r4}")
    log(f"  Negatives: base={n_neg_base}/{len(neg)}, fmt={n_neg_r1}, sys={n_neg_r2}, rule+fmt={n_neg_r3}, forced={n_neg_r4}")

    return {
        "n_samples": len(results),
        "n_negatives": len(neg),
        "base_accuracy": n_base / max(1, len(results)),
        "format_accuracy": n_r1 / max(1, len(results)),
        "system_accuracy": n_r2 / max(1, len(results)),
        "rule_format_accuracy": n_r3 / max(1, len(results)),
        "forced_accuracy": n_r4 / max(1, len(results)),
        "neg_base_accuracy": n_neg_base / max(1, len(neg)),
        "neg_format_accuracy": n_neg_r1 / max(1, len(neg)),
        "neg_system_accuracy": n_neg_r2 / max(1, len(neg)),
        "neg_rule_format_accuracy": n_neg_r3 / max(1, len(neg)),
        "neg_forced_accuracy": n_neg_r4 / max(1, len(neg)),
        "results": results,
    }


# ============================================================================
# Part D: GLM4 bypass mechanism
# ============================================================================

def run_bypass_mechanism_test(model, tokenizer, device, n_tables, objects, relations,
                               categories, values):
    """Test if model uses direct O→V bypass vs O→C→V chain."""
    log("--- Part D: Bypass mechanism test ---")

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

                # Full two-hop (O→C→V)
                full_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                full_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, full_prompt, values)
                full_pred = max(full_lps, key=lambda c: full_lps[c][0])
                full_correct = full_pred == correct_val

                # Cat prediction
                cat_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, full_prompt, categories)
                pred_cat = max(cat_lps, key=lambda c: cat_lps[c][0])
                cat_correct = pred_cat == correct_cat

                # Direct O→V: only OC rules, no CRV rules, ask for V
                # This tests if model can retrieve V without seeing CRV rules
                # (should fail if model needs CRV rules)
                rng = random.Random(tt_idx * 100)
                oc_rules = list(oc_tt.items())
                rng.shuffle(oc_rules)
                oc_lines = [f"{obj2} belongs to {cat2}." for obj2, cat2 in oc_rules]
                no_crv_prompt = "Rules:\n" + "\n".join(oc_lines)
                no_crv_prompt += f"\n\nQuestion: {obj} {rel} ?\nAnswer:"
                no_crv_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, no_crv_prompt, values)
                no_crv_pred = max(no_crv_lps, key=lambda c: no_crv_lps[c][0])
                no_crv_correct = no_crv_pred == correct_val

                # Wrong cat but full rules: does model still get V right?
                # This is the bypass test
                wrong_cats = [c for c in categories if c != correct_cat]
                wrong_cat = rng.choice(wrong_cats)
                wrong_val = crv_tt.get((wrong_cat, rel))

                bypass_prompt = "Rules:\n" + "\n".join(
                    [f"{obj2} belongs to {cat2}." for obj2, cat2 in oc_rules])
                crv_rules = list(crv_tt.items())
                rng.shuffle(crv_rules)
                bypass_prompt += "\n" + "\n".join([f"{c} {r} {v}." for (c, r), v in crv_rules])
                bypass_prompt += f"\n\n{obj} belongs to {wrong_cat}."
                bypass_prompt += f"\nQuestion: {obj} {rel} ?\nAnswer:"
                bypass_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, bypass_prompt, values)
                bypass_pred = max(bypass_lps, key=lambda c: bypass_lps[c][0])
                bypass_matches_wrong = bypass_pred == wrong_val
                bypass_matches_correct = bypass_pred == correct_val

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "pred_cat": pred_cat, "cat_correct": cat_correct,
                    "full_pred": full_pred, "full_correct": full_correct,
                    "no_crv_pred": no_crv_pred, "no_crv_correct": no_crv_correct,
                    "wrong_cat": wrong_cat, "wrong_val": wrong_val,
                    "bypass_pred": bypass_pred,
                    "bypass_matches_wrong": bypass_matches_wrong,
                    "bypass_matches_correct": bypass_matches_correct,
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_full = sum(1 for r in results if r["full_correct"])
    n_no_crv = sum(1 for r in results if r["no_crv_correct"])
    n_bypass_wrong = sum(1 for r in results if r["bypass_matches_wrong"])
    n_bypass_correct = sum(1 for r in results if r["bypass_matches_correct"])

    log(f"  Full two-hop: {n_full}/{len(results)} = {n_full/max(1,len(results)):.3f}")
    log(f"  No CRV (should fail): {n_no_crv}/{len(results)} = {n_no_crv/max(1,len(results)):.3f}")
    log(f"  Force wrong cat -> matches wrong val: {n_bypass_wrong}/{len(results)} = {n_bypass_wrong/max(1,len(results)):.3f}")
    log(f"  Force wrong cat -> matches correct val: {n_bypass_correct}/{len(results)} = {n_bypass_correct/max(1,len(results)):.3f}")

    return {
        "n_samples": len(results),
        "full_accuracy": n_full / max(1, len(results)),
        "no_crv_accuracy": n_no_crv / max(1, len(results)),
        "bypass_matches_wrong_rate": n_bypass_wrong / max(1, len(results)),
        "bypass_matches_correct_rate": n_bypass_correct / max(1, len(results)),
        "results": results,
    }


# ============================================================================
# Main
# ============================================================================

def run_model(args):
    model, tokenizer, device = load_model_flash(args.model)
    try:
        info = get_model_info(model, args.model)
        n_layers = info.n_layers
        log(f"{args.model}: n_layers={n_layers}")

        objects = CANDIDATE_OBJECTS[:8]
        categories = CANDIDATE_CATEGORIES[:4]
        relations = CANDIDATE_RELATIONS[:2]
        values = CANDIDATE_VALUES[:4]
        n_tables = args.n_tables

        result = {
            "phase": 584,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers,
            "n_tables": n_tables,
        }

        result["partA_choice_gate_repair"] = run_choice_gate_repair(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        result["partB_value_retrieval_repair"] = run_value_retrieval_repair(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        result["partC_polarity_gate_repair"] = run_polarity_gate_repair(
            model, tokenizer, device)

        result["partD_bypass_mechanism"] = run_bypass_mechanism_test(
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
    out_path = out_dir / f"phase584_{args.model}_gate_repair{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
