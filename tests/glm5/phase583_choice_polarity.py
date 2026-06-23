#!/usr/bin/env python3
"""
Phase 583: Intermediate Choice Gate and Polarity Readout Decomposition
中间选择门与极性读出分解

Phase 582 found:
1. Composition failures are mainly C_wrong_cat (wrong intermediate category)
2. GLM4/DS7B have strong yes-bias in parametric judgment

Phase 583 decomposes both gates:
  Part A: Track intermediate category prediction in two-hop tasks
  Part B: Force correct/wrong intermediate category, measure V change
  Part C: Margin analysis for category competition
  Part D: yes/no readout with multiple answer formats
  Part E: Explicit negation rule control

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

OUT_ROOT = Path("results/glm5_phase583_choice_polarity")

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

# Multiple answer format pairs for yes/no calibration
ANSWER_FORMATS = {
    "single": ("是", "否"),
    "double": ("是的", "不是"),
    "belong": ("属于", "不属于"),
    "correct": ("正确", "错误"),
    "english": ("yes", "no"),
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Model loading
# ============================================================================

def load_model_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl,
        )
        log(f"Loaded {model_name} with flash_attention_2")
    except Exception:
        attn_impl = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log(f"Loaded {model_name}: class={type(model).__name__}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# ============================================================================
# Full-string logprob
# ============================================================================

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


def build_cat_rel_truth_tables(categories, relations, values, seed=42):
    rng = random.Random(seed)
    mapping = {}
    for cat in categories:
        for rel in relations:
            mapping[(cat, rel)] = rng.choice(values)
    return mapping


# ============================================================================
# Prompt construction
# ============================================================================

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


def build_forced_intermediate_prompt_with_category(tokenizer, oc_table, crv_table,
                                                     query_object, query_relation,
                                                     forced_category, seed=42):
    """Build two-hop prompt but explicitly state the intermediate category."""
    rng = random.Random(seed)
    oc_rules = list(oc_table.items())
    rng.shuffle(oc_rules)
    oc_lines = [f"{obj} belongs to {cat}." for obj, cat in oc_rules]
    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]
    prompt = "Rules:\n" + "\n".join(oc_lines) + "\n" + "\n".join(crv_lines)
    prompt += f"\n\nQuestion: {query_object} {query_relation} ?"
    prompt += f"\nIntermediate category: {forced_category}"
    prompt += "\nAnswer:"
    correct_val = crv_table.get((forced_category, query_relation))
    return prompt, forced_category, correct_val


# ============================================================================
# Part A: Track intermediate category prediction
# ============================================================================

def run_intermediate_category_tracking(model, tokenizer, device, n_tables, objects,
                                        relations, categories, values):
    """In two-hop tasks, track which intermediate category the model predicts.

    Method: After the two-hop prompt, score each candidate category using
    full-string logprob of "category_name" as the next token.
    """
    log("--- Part A: Intermediate category tracking ---")

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

                # Two-hop prompt
                twohop_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)

                # Score all candidate categories as intermediate
                cat_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, categories)
                predicted_cat = max(cat_lps, key=lambda c: cat_lps[c][0])

                # Also score values (final answer)
                val_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, values)
                predicted_val = max(val_lps, key=lambda c: val_lps[c][0])

                # Check if predicted_cat would give predicted_val
                expected_val_from_pred_cat = crv_tt.get((predicted_cat, rel))

                # Category margin: correct vs best competitor
                correct_cat_lp = cat_lps[correct_cat][0]
                wrong_cat_lps = [cat_lps[c][0] for c in categories if c != correct_cat]
                cat_margin = correct_cat_lp - max(wrong_cat_lps) if wrong_cat_lps else 0

                cat_correct = predicted_cat == correct_cat
                val_correct = predicted_val == correct_val

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "predicted_cat": predicted_cat, "cat_correct": cat_correct,
                    "predicted_val": predicted_val, "val_correct": val_correct,
                    "expected_val_from_pred_cat": expected_val_from_pred_cat,
                    "cat_margin": cat_margin,
                    "cat_logprobs": {c: lp for c, (lp, _) in cat_lps.items()},
                    "val_logprobs": {v: lp for v, (lp, _) in val_lps.items()},
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_cat_correct = sum(1 for r in results if r["cat_correct"])
    n_val_correct = sum(1 for r in results if r["val_correct"])
    # How often does wrong cat lead to wrong val?
    wrong_cat = [r for r in results if not r["cat_correct"]]
    wrong_cat_wrong_val = sum(1 for r in wrong_cat if not r["val_correct"])

    # Margin analysis
    success_margins = [r["cat_margin"] for r in results if r["cat_correct"]]
    fail_margins = [r["cat_margin"] for r in results if not r["cat_correct"]]

    log(f"  Cat prediction: {n_cat_correct}/{len(results)} = {n_cat_correct/max(1,len(results)):.3f}")
    log(f"  Val prediction: {n_val_correct}/{len(results)} = {n_val_correct/max(1,len(results)):.3f}")
    log(f"  Wrong cat -> wrong val: {wrong_cat_wrong_val}/{len(wrong_cat)} = {wrong_cat_wrong_val/max(1,len(wrong_cat)):.3f}")
    if success_margins:
        log(f"  Success cat_margin: mean={np.mean(success_margins):.3f}")
    if fail_margins:
        log(f"  Fail cat_margin: mean={np.mean(fail_margins):.3f}")

    return {
        "n_samples": len(results),
        "cat_accuracy": n_cat_correct / max(1, len(results)),
        "val_accuracy": n_val_correct / max(1, len(results)),
        "wrong_cat_leads_wrong_val": wrong_cat_wrong_val / max(1, len(wrong_cat)),
        "success_margin_mean": float(np.mean(success_margins)) if success_margins else 0,
        "fail_margin_mean": float(np.mean(fail_margins)) if fail_margins else 0,
        "results": results,
    }


# ============================================================================
# Part B: Force correct/wrong intermediate category
# ============================================================================

def run_forced_category_experiment(model, tokenizer, device, n_tables, objects,
                                    relations, categories, values):
    """Force correct or wrong intermediate category, measure V change."""
    log("--- Part B: Forced intermediate category experiment ---")

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

                # Baseline: no forced category
                base_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                base_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, base_prompt, values)
                base_pred = max(base_lps, key=lambda c: base_lps[c][0])
                base_correct = base_pred == correct_val

                # Force correct category
                correct_prompt, _, correct_forced_val = build_forced_intermediate_prompt_with_category(
                    tokenizer, oc_tt, crv_tt, obj, rel, correct_cat, seed=tt_idx * 100)
                correct_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, correct_prompt, values)
                correct_pred = max(correct_lps, key=lambda c: correct_lps[c][0])
                correct_forced_correct = correct_pred == correct_val

                # Force wrong category (pick a different one)
                wrong_cats = [c for c in categories if c != correct_cat]
                rng = random.Random(tt_idx * 100 + hash(obj) % 1000)
                wrong_cat = rng.choice(wrong_cats)
                wrong_forced_val = crv_tt.get((wrong_cat, rel))

                wrong_prompt, _, _ = build_forced_intermediate_prompt_with_category(
                    tokenizer, oc_tt, crv_tt, obj, rel, wrong_cat, seed=tt_idx * 100)
                wrong_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, wrong_prompt, values)
                wrong_pred = max(wrong_lps, key=lambda c: wrong_lps[c][0])
                wrong_forced_matches = wrong_pred == wrong_forced_val

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "base_pred": base_pred, "base_correct": base_correct,
                    "correct_forced_pred": correct_pred,
                    "correct_forced_correct": correct_forced_correct,
                    "wrong_cat": wrong_cat, "wrong_forced_val": wrong_forced_val,
                    "wrong_forced_pred": wrong_pred,
                    "wrong_forced_matches": wrong_forced_matches,
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_base = sum(1 for r in results if r["base_correct"])
    n_correct_forced = sum(1 for r in results if r["correct_forced_correct"])
    n_wrong_matches = sum(1 for r in results if r["wrong_forced_matches"])

    log(f"  Base (no force): {n_base}/{len(results)} = {n_base/max(1,len(results)):.3f}")
    log(f"  Force correct cat: {n_correct_forced}/{len(results)} = {n_correct_forced/max(1,len(results)):.3f}")
    log(f"  Force wrong cat -> val matches wrong: {n_wrong_matches}/{len(results)} = {n_wrong_matches/max(1,len(results)):.3f}")

    return {
        "n_samples": len(results),
        "base_accuracy": n_base / max(1, len(results)),
        "correct_forced_accuracy": n_correct_forced / max(1, len(results)),
        "wrong_forced_match_rate": n_wrong_matches / max(1, len(results)),
        "results": results,
    }


# ============================================================================
# Part C: Category competition margin analysis
# ============================================================================

def run_margin_analysis(model, tokenizer, device, n_tables, objects,
                         relations, categories, values):
    """Analyze margin between correct and best competing category."""
    log("--- Part C: Category competition margin analysis ---")

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

                twohop_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)

                cat_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, categories)
                val_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, values)

                predicted_cat = max(cat_lps, key=lambda c: cat_lps[c][0])
                predicted_val = max(val_lps, key=lambda c: val_lps[c][0])

                correct_cat_lp = cat_lps[correct_cat][0]
                wrong_cat_lps = {c: cat_lps[c][0] for c in categories if c != correct_cat}
                best_wrong_cat = max(wrong_cat_lps, key=wrong_cat_lps.get)
                best_wrong_lp = wrong_cat_lps[best_wrong_cat]
                cat_margin = correct_cat_lp - best_wrong_lp

                correct_val_lp = val_lps[correct_val][0]
                wrong_val_lps = {v: val_lps[v][0] for v in values if v != correct_val}
                best_wrong_val = max(wrong_val_lps, key=wrong_val_lps.get)
                best_wrong_val_lp = wrong_val_lps[best_wrong_val]
                val_margin = correct_val_lp - best_wrong_val_lp

                cat_correct = predicted_cat == correct_cat
                val_correct = predicted_val == correct_val

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "predicted_cat": predicted_cat, "best_wrong_cat": best_wrong_cat,
                    "cat_correct": cat_correct, "val_correct": val_correct,
                    "cat_margin": cat_margin, "val_margin": val_margin,
                    "correct_cat_lp": correct_cat_lp,
                    "best_wrong_cat_lp": best_wrong_lp,
                    "correct_val_lp": correct_val_lp,
                    "best_wrong_val_lp": best_wrong_val_lp,
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    # Compare margins: success vs failure
    cat_success = [r for r in results if r["cat_correct"]]
    cat_fail = [r for r in results if not r["cat_correct"]]
    val_success = [r for r in results if r["val_correct"]]
    val_fail = [r for r in results if not r["val_correct"]]

    log(f"  Cat margin: success={np.mean([r['cat_margin'] for r in cat_success]):.3f}, "
        f"fail={np.mean([r['cat_margin'] for r in cat_fail]):.3f}" if cat_fail else "")
    log(f"  Val margin: success={np.mean([r['val_margin'] for r in val_success]):.3f}, "
        f"fail={np.mean([r['val_margin'] for r in val_fail]):.3f}" if val_fail else "")

    return {
        "n_samples": len(results),
        "cat_success_margin": float(np.mean([r["cat_margin"] for r in cat_success])) if cat_success else 0,
        "cat_fail_margin": float(np.mean([r["cat_margin"] for r in cat_fail])) if cat_fail else 0,
        "val_success_margin": float(np.mean([r["val_margin"] for r in val_success])) if val_success else 0,
        "val_fail_margin": float(np.mean([r["val_margin"] for r in val_fail])) if val_fail else 0,
        "results": results,
    }


# ============================================================================
# Part D: yes/no readout with multiple answer formats
# ============================================================================

def run_yesno_readout_calibration(model, tokenizer, device):
    """Test if yes-bias is format-dependent using multiple answer pairs."""
    log("--- Part D: yes/no readout calibration ---")

    results = []

    for cat in ["水果", "动物", "天体", "工具"]:
        positives = CATEGORY_OBJECTS.get(cat, [])[:3]
        negatives = CATEGORY_NEGATIVES.get(cat, [])[:3]

        for obj in positives + negatives:
            is_positive = obj in positives
            expected_pos = is_positive

            prompt = f"Question: {obj}是不是{cat}？\nAnswer:"

            for fmt_name, (pos_ans, neg_ans) in ANSWER_FORMATS.items():
                lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, prompt, [pos_ans, neg_ans])
                pred = max(lps, key=lambda c: lps[c][0])

                if expected_pos:
                    correct = pred == pos_ans
                else:
                    correct = pred == neg_ans

                margin = lps[pos_ans][0] - lps[neg_ans][0]

                results.append({
                    "category": cat, "object": obj,
                    "is_positive": is_positive,
                    "format": fmt_name,
                    "pos_ans": pos_ans, "neg_ans": neg_ans,
                    "predicted": pred, "correct": correct,
                    "margin": margin,
                    "pos_lp": lps[pos_ans][0],
                    "neg_lp": lps[neg_ans][0],
                })

    # Summarize by format and pos/neg
    by_format = {}
    for r in results:
        f = r["format"]
        if f not in by_format:
            by_format[f] = {"n": 0, "correct": 0, "pos_n": 0, "pos_correct": 0,
                            "neg_n": 0, "neg_correct": 0, "margins": []}
        by_format[f]["n"] += 1
        by_format[f]["correct"] += 1 if r["correct"] else 0
        by_format[f]["margins"].append(r["margin"])
        if r["is_positive"]:
            by_format[f]["pos_n"] += 1
            by_format[f]["pos_correct"] += 1 if r["correct"] else 0
        else:
            by_format[f]["neg_n"] += 1
            by_format[f]["neg_correct"] += 1 if r["correct"] else 0

    log(f"  Readout calibration by format:")
    for f, s in by_format.items():
        log(f"    {f}: overall={s['correct']}/{s['n']}={s['correct']/max(1,s['n']):.3f}, "
            f"pos={s['pos_correct']}/{s['pos_n']}, neg={s['neg_correct']}/{s['neg_n']}, "
            f"mean_margin={np.mean(s['margins']):.3f}")

    return {
        "n_samples": len(results),
        "by_format": {f: {
            "overall_accuracy": s["correct"] / max(1, s["n"]),
            "pos_accuracy": s["pos_correct"] / max(1, s["pos_n"]),
            "neg_accuracy": s["neg_correct"] / max(1, s["neg_n"]),
            "mean_margin": float(np.mean(s["margins"])),
            "n": s["n"],
        } for f, s in by_format.items()},
        "results": results,
    }


# ============================================================================
# Part E: Explicit negation rule control
# ============================================================================

def run_explicit_negation_control(model, tokenizer, device):
    """Test if explicit negation rules can overcome yes-bias."""
    log("--- Part E: Explicit negation rule control ---")

    test_cases = [
        ("老虎", "水果", False),
        ("地球", "水果", False),
        ("汽车", "水果", False),
        ("苹果", "水果", True),
        ("香蕉", "水果", True),
        ("老虎", "动物", True),
        ("苹果", "动物", False),
        ("地球", "天体", True),
        ("苹果", "天体", False),
        ("锤子", "工具", True),
        ("苹果", "工具", False),
    ]

    results = []

    for obj, cat, is_positive in test_cases:
        expected = "是" if is_positive else "否"

        # No rule
        no_rule_prompt = f"Question: {obj}是不是{cat}？\nAnswer:"
        no_rule_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, no_rule_prompt, ["是", "否"])
        no_rule_pred = max(no_rule_lps, key=lambda c: no_rule_lps[c][0])
        no_rule_correct = no_rule_pred == expected
        no_rule_margin = no_rule_lps["是"][0] - no_rule_lps["否"][0]

        # With affirmative rule
        if is_positive:
            aff_rule = f"Rules:\n{obj} 属于 {cat}.\n\nQuestion: {obj}是不是{cat}？\nAnswer:"
        else:
            aff_rule = f"Rules:\n{obj} 不属于 {cat}.\n\nQuestion: {obj}是不是{cat}？\nAnswer:"

        aff_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, aff_rule, ["是", "否"])
        aff_pred = max(aff_lps, key=lambda c: aff_lps[c][0])
        aff_correct = aff_pred == expected
        aff_margin = aff_lps["是"][0] - aff_lps["否"][0]

        # With strong negation (for negatives only)
        if not is_positive:
            # Provide alternative category
            alt_cat = "动物" if cat == "水果" else "水果"
            strong_neg = f"Rules:\n{obj} 属于 {alt_cat}, 不属于 {cat}.\n\nQuestion: {obj}是不是{cat}？\nAnswer:"
            strong_lps = compute_full_string_logprob_batch(
                model, tokenizer, device, strong_neg, ["是", "否"])
            strong_pred = max(strong_lps, key=lambda c: strong_lps[c][0])
            strong_correct = strong_pred == expected
            strong_margin = strong_lps["是"][0] - strong_lps["否"][0]
        else:
            strong_pred = aff_pred
            strong_correct = aff_correct
            strong_margin = aff_margin

        results.append({
            "object": obj, "category": cat, "is_positive": is_positive,
            "expected": expected,
            "no_rule_pred": no_rule_pred, "no_rule_correct": no_rule_correct,
            "no_rule_margin": no_rule_margin,
            "aff_rule_pred": aff_pred, "aff_rule_correct": aff_correct,
            "aff_margin": aff_margin,
            "strong_neg_pred": strong_pred, "strong_neg_correct": strong_correct,
            "strong_margin": strong_margin,
        })

        log(f"  {obj}/{cat}({expected}): no_rule={no_rule_pred}({no_rule_correct}), "
            f"aff_rule={aff_pred}({aff_correct}), strong_neg={strong_pred}({strong_correct}), "
            f"margins: {no_rule_margin:.2f}->{aff_margin:.2f}->{strong_margin:.2f}")

    # Summarize
    n_no_rule = sum(1 for r in results if r["no_rule_correct"])
    n_aff = sum(1 for r in results if r["aff_rule_correct"])
    n_strong = sum(1 for r in results if r["strong_neg_correct"])

    # Only negatives
    neg_results = [r for r in results if not r["is_positive"]]
    n_neg_no = sum(1 for r in neg_results if r["no_rule_correct"])
    n_neg_aff = sum(1 for r in neg_results if r["aff_rule_correct"])
    n_neg_strong = sum(1 for r in neg_results if r["strong_neg_correct"])

    log(f"  Overall: no_rule={n_no_rule}/{len(results)}, aff={n_aff}/{len(results)}, strong={n_strong}/{len(results)}")
    log(f"  Negatives only: no_rule={n_neg_no}/{len(neg_results)}, aff={n_neg_aff}/{len(neg_results)}, strong={n_neg_strong}/{len(neg_results)}")

    return {
        "n_samples": len(results),
        "no_rule_accuracy": n_no_rule / max(1, len(results)),
        "aff_rule_accuracy": n_aff / max(1, len(results)),
        "strong_neg_accuracy": n_strong / max(1, len(results)),
        "neg_no_rule_accuracy": n_neg_no / max(1, len(neg_results)),
        "neg_aff_rule_accuracy": n_neg_aff / max(1, len(neg_results)),
        "neg_strong_neg_accuracy": n_neg_strong / max(1, len(neg_results)),
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
            "phase": 583,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers,
            "n_tables": n_tables,
        }

        result["partA_intermediate_tracking"] = run_intermediate_category_tracking(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        result["partB_forced_category"] = run_forced_category_experiment(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        result["partC_margin_analysis"] = run_margin_analysis(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        result["partD_yesno_calibration"] = run_yesno_readout_calibration(
            model, tokenizer, device)

        result["partE_negation_control"] = run_explicit_negation_control(
            model, tokenizer, device)

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
    out_path = out_dir / f"phase583_{args.model}_choice_polarity{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
