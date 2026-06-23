#!/usr/bin/env python3
"""
Phase 582: Relation Necessity, State-Bridge Failure Typing, and Parametric Category Judgment
关系必要性、状态桥接失败分型、与参数化类别判断

Three parts:
  Part A: Fix relation necessity audit (force R1≠R2 => V1≠V2)
  Part B: Composition failure decomposition (classify WHY two-hop fails)
  Part C: Parametric category judgment (fruit/animal/celestial membership)
          - Bridge from explicit micro-world retrieval to parametric knowledge
          - Test "苹果是不是水果？" type questions
          - Compare explicit-rule vs parametric-commonsense circuits

Model loading: BF16 + device_map="auto" + flash_attention_2 (no quantization)
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

OUT_ROOT = Path("results/glm5_phase582_relation_parametric")

# ============================================================================
# Constants
# ============================================================================

CANDIDATE_OBJECTS = ["o17", "o29", "o43", "o58", "o71", "o82", "o95", "o06"]
CANDIDATE_RELATIONS = ["r31", "r64"]
CANDIDATE_VALUES = ["v05", "v91", "v22", "v48"]
CANDIDATE_CATEGORIES = ["c12", "c77", "c33", "c59"]

# Parametric category judgment objects
CATEGORY_OBJECTS = {
    "水果": ["苹果", "香蕉", "梨", "桃子", "葡萄"],
    "动物": ["老虎", "狗", "猫", "鸟", "鱼"],
    "天体": ["地球", "太阳", "月亮", "火星", "星星"],
    "工具": ["锤子", "剪刀", "斧头", "锯子", "钳子"],
    "家具": ["椅子", "桌子", "床", "柜子", "沙发"],
    "交通工具": ["汽车", "飞机", "船", "自行车", "火车"],
}

# Negative examples (not in any of the above categories for this category)
CATEGORY_NEGATIVES = {
    "水果": ["老虎", "地球", "汽车", "椅子", "石头"],
    "动物": ["苹果", "地球", "汽车", "椅子", "石头"],
    "天体": ["苹果", "老虎", "汽车", "椅子", "石头"],
    "工具": ["苹果", "老虎", "地球", "汽车", "椅子"],
    "家具": ["苹果", "老虎", "地球", "汽车", "锤子"],
    "交通工具": ["苹果", "老虎", "地球", "椅子", "锤子"],
}


def log(msg: str = "") -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================================
# Model loading with flash attention
# ============================================================================

def load_model_flash(model_name: str):
    """Load model with flash_attention_2, fallback to eager if not available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attention_2 first, fallback to eager
    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=attn_impl,
        )
        log(f"Loaded {model_name} with flash_attention_2")
    except Exception as e:
        log(f"flash_attention_2 failed ({e}), falling back to eager")
        attn_impl = "eager"
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
    log(f"Loaded {model_name}: class={type(model).__name__}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# ============================================================================
# Full-string logprob (from Phase 580/581)
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
# Part A: Fixed relation necessity audit
# ============================================================================

def build_strong_orv_tables(objects, relations, values, n_tables, seed=42):
    """Build ORV tables where same object with different relations MUST have different values."""
    rng = random.Random(seed)
    tables = []
    n_vals = len(values)
    n_rels = len(relations)

    for t in range(n_tables):
        mapping = {}
        for obj in objects:
            # Assign different values to different relations for same object
            shuffled_vals = list(values)
            rng.shuffle(shuffled_vals)
            for ri, rel in enumerate(relations):
                mapping[(obj, rel)] = shuffled_vals[ri % n_vals]
        tables.append(mapping)
    return tables


def run_relation_necessity_audit(model, tokenizer, device, n_tables, objects, relations, values):
    """Test if model truly uses relation R.

    Key fix: force R1≠R2 => V1≠V2, so ignoring R would give wrong answer.
    """
    log("--- Part A: Relation necessity audit (fixed) ---")

    orv_tables = build_strong_orv_tables(objects, relations, values, n_tables)
    results = []

    for tt_idx in range(min(n_tables, 10)):
        tt = orv_tables[tt_idx]
        for obj in objects[:4]:
            vals_by_rel = {}
            for rel in relations:
                v = tt.get((obj, rel))
                if v:
                    vals_by_rel[rel] = v

            if len(vals_by_rel) < 2:
                continue
            if len(set(vals_by_rel.values())) < 2:
                continue  # Should not happen with strong tables

            for rel in relations:
                if rel not in vals_by_rel:
                    continue
                correct_val = vals_by_rel[rel]

                rng = random.Random(tt_idx * 100 + hash(obj) % 1000 + hash(rel) % 100)
                rules = list(tt.items())
                rng.shuffle(rules)
                rule_lines = [f"{o} {r} {v}." for (o, r), v in rules]
                prompt = "Rules:\n" + "\n".join(rule_lines)
                prompt += f"\n\nQuestion: {obj} {rel} ?\nAnswer:"

                lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, prompt, values)
                pred = max(lps, key=lambda c: lps[c][0])
                correct = pred == correct_val

                # Check: does model give different answers for different relations?
                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_val": correct_val,
                    "predicted": pred, "is_correct": correct,
                    "all_vals": dict(vals_by_rel),
                    "logprobs": {c: lp for c, (lp, _) in lps.items()},
                })

            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    n_correct = sum(1 for r in results if r["is_correct"])
    accuracy = n_correct / max(1, len(results))

    # Relation discrimination: does model give different answers for different relations?
    by_obj_tt = {}
    for r in results:
        key = (r["tt_idx"], r["obj"])
        if key not in by_obj_tt:
            by_obj_tt[key] = {}
        by_obj_tt[key][r["rel"]] = r["predicted"]

    rel_discrim = 0
    rel_discrim_total = 0
    correct_discrim = 0
    for key, preds in by_obj_tt.items():
        if len(preds) >= 2:
            rel_discrim_total += 1
            if len(set(preds.values())) >= 2:
                rel_discrim += 1
            # Check if both predictions match their correct values
            tt = orv_tables[key[0]]
            obj = key[1]
            both_correct = all(
                preds.get(rel) == tt.get((obj, rel))
                for rel in relations if rel in preds
            )
            if both_correct and len(set(preds.values())) >= 2:
                correct_discrim += 1

    discrim_rate = rel_discrim / max(1, rel_discrim_total)
    correct_discrim_rate = correct_discrim / max(1, rel_discrim_total)

    log(f"  Relation audit: accuracy={accuracy:.3f} ({n_correct}/{len(results)})")
    log(f"  Relation discrimination: {rel_discrim}/{rel_discrim_total} = {discrim_rate:.3f}")
    log(f"  Correct discrimination: {correct_discrim}/{rel_discrim_total} = {correct_discrim_rate:.3f}")

    return {
        "accuracy": accuracy,
        "n_samples": len(results),
        "relation_discrimination_rate": discrim_rate,
        "correct_discrimination_rate": correct_discrim_rate,
        "n_discrim_samples": rel_discrim_total,
        "results": results,
    }


# ============================================================================
# Part B: Composition failure decomposition
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


def build_gold_intermediate_prompt(tokenizer, oc_table, crv_table, query_object, query_relation, seed=42):
    rng = random.Random(seed)
    correct_cat = oc_table.get(query_object)
    if correct_cat is None:
        return None, None, None

    crv_rules = list(crv_table.items())
    rng.shuffle(crv_rules)
    crv_lines = [f"{cat} {rel} {val}." for (cat, rel), val in crv_rules]

    prompt = "Rules:\n" + "\n".join(crv_lines)
    prompt += f"\n\n{query_object} belongs to {correct_cat}."
    prompt += f"\nQuestion: {query_object} {query_relation} ?\nAnswer:"

    correct_val = crv_table.get((correct_cat, query_relation))
    return prompt, correct_cat, correct_val


def build_oc_prompt(tokenizer, truth_table, query_object, seed=42):
    rng = random.Random(seed)
    rules = list(truth_table.items())
    rng.shuffle(rules)
    rule_lines = [f"{obj} belongs to {cat}." for obj, cat in rules]
    prompt = "Rules:\n" + "\n".join(rule_lines)
    prompt += f"\n\nQuestion: {query_object} belongs to ?\nAnswer:"
    return prompt


def run_failure_decomposition(model, tokenizer, device, n_tables, objects, relations,
                               categories, values):
    """Decompose two-hop failures into types."""
    log("--- Part B: Composition failure decomposition ---")

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

                # Step A: O→C
                oc_prompt = build_oc_prompt(tokenizer, oc_tt, obj, seed=tt_idx * 100)
                oc_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, oc_prompt, categories)
                step_a_pred = max(oc_lps, key=lambda c: oc_lps[c][0])
                step_a_correct = step_a_pred == correct_cat

                # Step B: (C,R)→V gold
                gold_prompt, _, gold_val = build_gold_intermediate_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                if gold_prompt is None:
                    continue
                gold_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, gold_prompt, values)
                step_b_pred = max(gold_lps, key=lambda c: gold_lps[c][0])
                step_b_correct = step_b_pred == correct_val

                # Step C: Compose
                twohop_prompt, _, _ = build_two_hop_prompt(
                    tokenizer, oc_tt, crv_tt, obj, rel, seed=tt_idx * 100)
                twohop_lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, twohop_prompt, values)
                step_c_pred = max(twohop_lps, key=lambda c: twohop_lps[c][0])
                step_c_correct = step_c_pred == correct_val

                # Classify failure type
                failure_type = "success"
                if not step_c_correct:
                    if not step_a_correct:
                        failure_type = "A_fail"  # First step retrieval failed
                    elif not step_b_correct:
                        failure_type = "B_fail"  # Second step retrieval failed
                    else:
                        # Both A and B correct, but compose failed
                        # Check what the model predicted in compose
                        if step_c_pred == crv_tt.get((step_a_pred, rel)):
                            # Model used the correct intermediate category
                            # but value copy failed
                            failure_type = "bridge_fail"
                        else:
                            # Check if model used a wrong category
                            # Find which category would give this value
                            wrong_cat = None
                            for cat in categories:
                                if crv_tt.get((cat, rel)) == step_c_pred:
                                    wrong_cat = cat
                                    break
                            if wrong_cat and wrong_cat != correct_cat:
                                failure_type = "C_wrong_cat"  # Used wrong intermediate
                            else:
                                failure_type = "V_copy_fail"  # Value copy failed

                results.append({
                    "tt_idx": tt_idx, "obj": obj, "rel": rel,
                    "correct_cat": correct_cat, "correct_val": correct_val,
                    "step_a_pred": step_a_pred, "step_a_correct": step_a_correct,
                    "step_b_pred": step_b_pred, "step_b_correct": step_b_correct,
                    "step_c_pred": step_c_pred, "step_c_correct": step_c_correct,
                    "failure_type": failure_type,
                })

                if len(results) >= 40:
                    break
            if len(results) >= 40:
                break
        if len(results) >= 40:
            break

    # Summarize failure types
    failure_counts = {}
    for r in results:
        ft = r["failure_type"]
        failure_counts[ft] = failure_counts.get(ft, 0) + 1

    n_success = failure_counts.get("success", 0)
    n_total = len(results)

    log(f"  Failure decomposition ({n_total} samples):")
    for ft, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        log(f"    {ft}: {count}/{n_total} = {count/n_total:.3f}")

    return {
        "n_samples": n_total,
        "failure_counts": failure_counts,
        "success_rate": n_success / max(1, n_total),
        "results": results,
    }


# ============================================================================
# Part C: Parametric category judgment
# ============================================================================

def run_parametric_category_judgment(model, tokenizer, device):
    """Test parametric knowledge: '苹果是不是水果？' type questions."""
    log("--- Part C: Parametric category judgment ---")

    categories_to_test = ["水果", "动物", "天体", "工具", "家具", "交通工具"]
    results = []

    for cat in categories_to_test:
        positives = CATEGORY_OBJECTS.get(cat, [])
        negatives = CATEGORY_NEGATIVES.get(cat, [])

        for obj in positives + negatives:
            is_positive = obj in positives

            # Test multiple syntax templates
            templates = {
                "direct": f"{obj}是不是{cat}？",
                "formal": f"{obj}是否属于{cat}？",
                "simple": f"{obj}是{cat}吗？",
                "negative": f"{obj}不是{cat}吗？",
            }

            for tmpl_name, question in templates.items():
                prompt = f"Question: {question}\nAnswer:"
                lps = compute_full_string_logprob_batch(
                    model, tokenizer, device, prompt, ["是", "否"])
                pred = max(lps, key=lambda c: lps[c][0])

                # "是" is correct for positives, "否" for negatives
                # For negative template "不是...吗？", the polarity flips
                if tmpl_name == "negative":
                    # "苹果不是水果吗？" → answer "是" means "yes it is fruit"
                    expected = "是" if is_positive else "否"
                else:
                    expected = "是" if is_positive else "否"

                correct = pred == expected
                margin = lps["是"][0] - lps["否"][0]

                results.append({
                    "category": cat, "object": obj,
                    "is_positive": is_positive,
                    "template": tmpl_name,
                    "predicted": pred, "expected": expected,
                    "correct": correct,
                    "margin": margin,
                    "yes_logprob": lps["是"][0],
                    "no_logprob": lps["否"][0],
                })

    # Summarize
    by_template = {}
    for r in results:
        t = r["template"]
        if t not in by_template:
            by_template[t] = {"n": 0, "correct": 0}
        by_template[t]["n"] += 1
        by_template[t]["correct"] += 1 if r["correct"] else 0

    by_category = {}
    for r in results:
        c = r["category"]
        if c not in by_category:
            by_category[c] = {"n": 0, "correct": 0}
        by_category[c]["n"] += 1
        by_category[c]["correct"] += 1 if r["correct"] else 0

    by_posneg = {"positive": {"n": 0, "correct": 0}, "negative": {"n": 0, "correct": 0}}
    for r in results:
        key = "positive" if r["is_positive"] else "negative"
        by_posneg[key]["n"] += 1
        by_posneg[key]["correct"] += 1 if r["correct"] else 0

    log(f"  Parametric judgment results:")
    for t, s in by_template.items():
        log(f"    Template {t}: {s['correct']}/{s['n']} = {s['correct']/max(1,s['n']):.3f}")
    for c, s in by_category.items():
        log(f"    Category {c}: {s['correct']}/{s['n']} = {s['correct']/max(1,s['n']):.3f}")
    for k, s in by_posneg.items():
        log(f"    {k}: {s['correct']}/{s['n']} = {s['correct']/max(1,s['n']):.3f}")

    return {
        "n_samples": len(results),
        "by_template": {t: {"accuracy": s["correct"]/max(1,s["n"]), "n": s["n"]}
                        for t, s in by_template.items()},
        "by_category": {c: {"accuracy": s["correct"]/max(1,s["n"]), "n": s["n"]}
                        for c, s in by_category.items()},
        "by_posneg": {k: {"accuracy": s["correct"]/max(1,s["n"]), "n": s["n"]}
                      for k, s in by_posneg.items()},
        "results": results,
    }


# ============================================================================
# Part D: Explicit vs parametric comparison
# ============================================================================

def run_explicit_vs_parametric(model, tokenizer, device, categories, values):
    """Compare explicit rule vs parametric knowledge circuits.

    Test: '苹果是不是水果？' with explicit rules vs without.
    """
    log("--- Part D: Explicit vs parametric knowledge ---")

    results = []

    # Test a few fruit/animal objects
    test_cases = [
        ("苹果", "水果", "是"),
        ("老虎", "水果", "否"),
        ("香蕉", "水果", "是"),
        ("地球", "水果", "否"),
    ]

    for obj, cat, expected in test_cases:
        # Parametric (no rules)
        param_prompt = f"Question: {obj}是不是{cat}？\nAnswer:"
        param_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, param_prompt, ["是", "否"])
        param_pred = max(param_lps, key=lambda c: param_lps[c][0])
        param_correct = param_pred == expected
        param_margin = param_lps["是"][0] - param_lps["否"][0]

        # Explicit (with rules)
        explicit_rules = f"Rules:\n{obj} 属于 {cat}.\n" if expected == "是" else f"Rules:\n{obj} 不属于 {cat}.\n"
        explicit_prompt = explicit_rules + f"\nQuestion: {obj}是不是{cat}？\nAnswer:"
        explicit_lps = compute_full_string_logprob_batch(
            model, tokenizer, device, explicit_prompt, ["是", "否"])
        explicit_pred = max(explicit_lps, key=lambda c: explicit_lps[c][0])
        explicit_correct = explicit_pred == expected
        explicit_margin = explicit_lps["是"][0] - explicit_lps["否"][0]

        results.append({
            "object": obj, "category": cat, "expected": expected,
            "param_pred": param_pred, "param_correct": param_correct,
            "param_margin": param_margin,
            "explicit_pred": explicit_pred, "explicit_correct": explicit_correct,
            "explicit_margin": explicit_margin,
            "margin_diff": explicit_margin - param_margin,
        })

        log(f"  {obj}/{cat}: param={param_pred}({param_correct}), "
            f"explicit={explicit_pred}({explicit_correct}), "
            f"margin_diff={explicit_margin-param_margin:.3f}")

    return {
        "n_samples": len(results),
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
            "phase": 582,
            "model": args.model,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_layers": n_layers,
            "n_tables": n_tables,
        }

        # Part A: Relation necessity audit (fixed)
        result["partA_relation_audit"] = run_relation_necessity_audit(
            model, tokenizer, device, n_tables, objects, relations, values)

        # Part B: Failure decomposition
        result["partB_failure_decomposition"] = run_failure_decomposition(
            model, tokenizer, device, n_tables, objects, relations, categories, values)

        # Part C: Parametric category judgment
        result["partC_parametric_judgment"] = run_parametric_category_judgment(
            model, tokenizer, device)

        # Part D: Explicit vs parametric
        result["partD_explicit_vs_parametric"] = run_explicit_vs_parametric(
            model, tokenizer, device, categories, values)

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
    out_path = out_dir / f"phase582_{args.model}_relation_parametric{suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    log(f"Wrote {out_path}")
    log(f"Total time: {result['total_time_min']} min")

    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
