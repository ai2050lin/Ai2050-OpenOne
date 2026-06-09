"""
Phase 411: Vocabulary Frequency Control Experiment
===================================================

Phase 410发现level-gradient映射是跨属性存在的, 但R2只有0.4-0.5。
一个关键混淆变量: 候选词的词频(token frequency)是否影响gradient?

本实验: 对同一属性用不同频率的同义词组作为候选集, 观察level-gradient是否稳定。

测试属性: temperature (最强信号)
- 标准候选集: freezing/cold/cool/warm/hot/scorching (混合频率)
- 低频候选集: glacial/frigid/brisk/tepid/sweltering/incandescent (低频)
- 高频候选集: ice-cold/chilly/mild/boiling/burning/red-hot (高频短语)

如果换词后level-gradient correlation仍然显著, 说明编码机制独立于词汇频率。
如果大幅变化, 说明需要区分"语义编码"和"词汇频率编码"。

Usage:
  python tests/glm5/phase411_vocab_frequency.py qwen3
  python tests/glm5/phase411_vocab_frequency.py glm4
  python tests/glm5/phase411_vocab_frequency.py deepseek7b
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U

# ===== 同义词组定义 =====
CANDIDATE_SETS = OrderedDict({
    "standard": {
        "description": "标准候选集 (混合频率)",
        "candidates": OrderedDict([
            ("freezing", 1), ("cold", 2), ("cool", 3), ("warm", 4), ("hot", 5), ("scorching", 6),
        ]),
    },
    "low_freq": {
        "description": "低频候选集 (低频同义词)",
        "candidates": OrderedDict([
            ("glacial", 1), ("frigid", 2), ("brisk", 3), ("tepid", 4), ("sweltering", 5), ("incandescent", 6),
        ]),
    },
    "high_freq": {
        "description": "高频候选集 (常用短语/词)",
        "candidates": OrderedDict([
            ("icy", 1), ("chilly", 2), ("mild", 3), ("toasty", 4), ("boiling", 5), ("blazing", 6),
        ]),
    },
})

# 对象定义 (与Phase 410一致)
OBJECTS = OrderedDict({
    # Low level (1-2)
    "ice":         {"type": "substance",  "level": 1},
    "snow":        {"type": "substance",  "level": 1},
    "frost":       {"type": "substance",  "level": 1},
    "refrigerator":{"type": "object",     "level": 2},
    "freezer":     {"type": "object",     "level": 1},
    # Mid level (3-4)
    "spring":      {"type": "season",     "level": 3},
    "autumn":      {"type": "season",     "level": 3},
    # High level (5-6)
    "desert":      {"type": "place",      "level": 5},
    "volcano":     {"type": "place",      "level": 6},
    "oven":        {"type": "object",     "level": 5},
    "furnace":     {"type": "object",     "level": 5},
    "lava":        {"type": "substance",  "level": 6},
    "fire":        {"type": "substance",  "level": 5},
})


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            print(f"  Loaded with attn_implementation={impl}")
            break
        except Exception as e:
            print(f"  attn_implementation={impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_distribution_metrics(logits, candidate_ids, levels):
    """计算候选分布指标"""
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])

    if valid_mask.sum() < 2:
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0, "gradient": 0,
                "top_candidate": "", "prob_distribution": {}}

    max_logit = np.max(cand_logits[valid_mask])
    exp_logits = np.exp(cand_logits - max_logit)
    exp_logits[~valid_mask] = 0
    total = np.sum(exp_logits)
    probs = exp_logits / total if total > 0 else np.zeros_like(exp_logits)

    valid_probs = probs[valid_mask]
    valid_probs_pos = valid_probs[valid_probs > 0]
    entropy = -np.sum(valid_probs_pos * np.log(valid_probs_pos)) if len(valid_probs_pos) > 0 else 0

    variance = float(np.var(cand_logits[valid_mask])) if valid_mask.sum() > 1 else 0

    sorted_logits = np.sort(cand_logits[valid_mask])[::-1]
    top_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else 0

    valid_levels = np.array(levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]
    if len(valid_levels) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(valid_levels, valid_cand_logits)
        rank_corr = float(corr) if not np.isnan(corr) else 0
    else:
        rank_corr = 0

    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
        gradient = float(slope)
    else:
        gradient = 0

    # Top candidate
    cand_names_list = list(CANDIDATE_SETS.get("standard", {}).get("candidates", {}).keys())
    top_idx = np.argmax(cand_logits)
    top_candidate = cand_names_list[top_idx] if top_idx < len(cand_names_list) else ""

    # Probability distribution
    prob_dist = {}
    cand_names = list(CANDIDATE_SETS.get("standard", {}).get("candidates", {}).keys())
    for i in range(len(candidate_ids)):
        if valid_mask[i] and i < len(cand_names):
            prob_dist[cand_names[i]] = float(probs[i])

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "gradient": float(gradient),
    }


def compute_metrics_with_names(logits, candidate_ids, levels, cand_names):
    """计算候选分布指标(带候选名)"""
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])

    if valid_mask.sum() < 2:
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0, "gradient": 0,
                "prob_distribution": {}, "n_valid": int(valid_mask.sum())}

    max_logit = np.max(cand_logits[valid_mask])
    exp_logits = np.exp(cand_logits - max_logit)
    exp_logits[~valid_mask] = 0
    total = np.sum(exp_logits)
    probs = exp_logits / total if total > 0 else np.zeros_like(exp_logits)

    valid_probs = probs[valid_mask]
    valid_probs_pos = valid_probs[valid_probs > 0]
    entropy = -np.sum(valid_probs_pos * np.log(valid_probs_pos)) if len(valid_probs_pos) > 0 else 0

    variance = float(np.var(cand_logits[valid_mask])) if valid_mask.sum() > 1 else 0

    sorted_logits = np.sort(cand_logits[valid_mask])[::-1]
    top_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else 0

    valid_levels = np.array(levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]
    if len(valid_levels) > 2:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(valid_levels, valid_cand_logits)
        rank_corr = float(corr) if not np.isnan(corr) else 0
    else:
        rank_corr = 0

    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
        gradient = float(slope)
    else:
        gradient = 0

    # Probability distribution
    prob_dist = {}
    for i, cn in enumerate(cand_names):
        if valid_mask[i]:
            prob_dist[cn] = float(probs[i])

    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "gradient": float(gradient),
        "prob_distribution": prob_dist,
        "n_valid": int(valid_mask.sum()),
    }


def run_vocab_frequency_test(model, tokenizer, device, W_U_np):
    """运行词汇频率控制实验"""
    obj_names = sorted(OBJECTS.keys())

    all_results = {}

    for set_name, set_config in CANDIDATE_SETS.items():
        print(f"\n  --- Candidate Set: {set_name} ({set_config['description']}) ---")

        # Resolve token IDs
        candidate_ids = []
        levels = []
        cand_names = []
        for cand_name, level in set_config["candidates"].items():
            ids = tokenizer.encode(cand_name, add_special_tokens=False)
            tid = ids[0] if ids else None
            candidate_ids.append(tid)
            levels.append(level)
            cand_names.append(cand_name)

        print(f"  Candidates: {dict(zip(cand_names, candidate_ids))}")
        n_valid = sum(1 for cid in candidate_ids if cid is not None)
        print(f"  Valid tokens: {n_valid}/{len(candidate_ids)}")

        if n_valid < 3:
            print(f"  SKIP: Too few valid tokens")
            continue

        set_result = {"per_object": {}, "token_ids": dict(zip(cand_names, candidate_ids))}

        for obj_name in obj_names:
            obj_data = OBJECTS[obj_name]
            prompt = f"The {obj_name} is"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)

            final_logits = out.logits[0, -1].float().cpu().numpy()
            metrics = compute_metrics_with_names(final_logits, candidate_ids, levels, cand_names)

            set_result["per_object"][obj_name] = {
                "gradient": metrics["gradient"],
                "entropy": metrics["entropy"],
                "rank_corr": metrics["rank_corr"],
                "level": obj_data["level"],
                "type": obj_data["type"],
                "prob_distribution": metrics["prob_distribution"],
            }

            print(f"    {obj_name} (L{obj_data['level']}): "
                  f"grad={metrics['gradient']:+.4f}, corr={metrics['rank_corr']:+.4f}, "
                  f"entropy={metrics['entropy']:.4f}, n_valid={metrics['n_valid']}")

        # Aggregate
        all_gradients = [set_result["per_object"][n]["gradient"] for n in obj_names
                        if n in set_result["per_object"]]
        all_levels = [OBJECTS[n]["level"] for n in obj_names
                     if n in set_result["per_object"]]

        if len(all_gradients) > 2:
            from scipy.stats import spearmanr
            corr, _ = spearmanr(all_levels, all_gradients)
            level_gradient_corr = float(corr) if not np.isnan(corr) else 0
        else:
            level_gradient_corr = 0

        low_grads = [set_result["per_object"][n]["gradient"] for n in obj_names
                    if n in set_result["per_object"] and OBJECTS[n]["level"] <= 2]
        high_grads = [set_result["per_object"][n]["gradient"] for n in obj_names
                     if n in set_result["per_object"] and OBJECTS[n]["level"] >= 5]

        set_result["aggregate"] = {
            "level_gradient_corr": level_gradient_corr,
            "mean_gradient": float(np.mean(all_gradients)) if all_gradients else 0,
            "mean_entropy": float(np.mean([set_result["per_object"][n]["entropy"]
                                           for n in obj_names if n in set_result["per_object"]])),
            "low_mean_gradient": float(np.mean(low_grads)) if low_grads else 0,
            "high_mean_gradient": float(np.mean(high_grads)) if high_grads else 0,
            "high_low_delta": (float(np.mean(high_grads)) - float(np.mean(low_grads))) if (high_grads and low_grads) else 0,
        }

        agg = set_result["aggregate"]
        print(f"  >>> corr={agg['level_gradient_corr']:.4f}, "
              f"delta(H-L)={agg['high_low_delta']:+.4f}, "
              f"entropy={agg['mean_entropy']:.4f}")

        all_results[set_name] = set_result
        print(f"    {log_memory()}")

    return all_results


def run_phase411(model_name):
    """Phase 411主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 411: Vocabulary Frequency Control ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device

    # Get W_U
    W_U_np = get_W_U(model, model_name)
    print(f"  W_U: shape={W_U_np.shape}, n_layers={info.n_layers}")

    # Run test
    results = run_vocab_frequency_test(model, tokenizer, device, W_U_np)

    # ===== Cross-set comparison =====
    print(f"\n{'='*80}")
    print(f"=== Cross-Set Comparison ===")
    print(f"{'='*80}")

    print(f"\n1. Level-Gradient Correlation by Candidate Set:")
    for set_name, set_result in results.items():
        agg = set_result["aggregate"]
        print(f"  {set_name}: corr={agg['level_gradient_corr']:.4f}, "
              f"delta(H-L)={agg['high_low_delta']:+.4f}")

    print(f"\n2. Per-Object Gradient Comparison:")
    obj_names = sorted(OBJECTS.keys())
    header = f"  {'Object':<15} {'Level':>5}"
    for set_name in CANDIDATE_SETS.keys():
        if set_name in results:
            header += f" {set_name:>12}"
    print(header)
    print(f"  {'-'*60}")

    for obj_name in obj_names:
        line = f"  {obj_name:<15} {OBJECTS[obj_name]['level']:>5}"
        for set_name in CANDIDATE_SETS.keys():
            if set_name in results and obj_name in results[set_name]["per_object"]:
                grad = results[set_name]["per_object"][obj_name]["gradient"]
                line += f" {grad:>+12.4f}"
            else:
                line += f" {'N/A':>12}"
        print(line)

    print(f"\n3. Gradient Stability Analysis:")
    for obj_name in obj_names:
        grads = []
        for set_name in CANDIDATE_SETS.keys():
            if set_name in results and obj_name in results[set_name]["per_object"]:
                grads.append(results[set_name]["per_object"][obj_name]["gradient"])
        if len(grads) >= 2:
            sign_consistency = all(g > 0 for g in grads) or all(g < 0 for g in grads)
            grad_range = max(grads) - min(grads)
            print(f"  {obj_name}: sign_consistent={sign_consistency}, "
                  f"range={grad_range:.4f}, "
                  f"grads={[f'{g:+.3f}' for g in grads]}")

    # ===== Save results =====
    results_dir = ROOT / "results" / "phase411_vocab_frequency"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase411.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj

    output = {
        "model": model_name,
        "timestamp": timestamp,
        "phase": 411,
        "description": "Vocabulary Frequency Control",
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "results": convert(results),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return output


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase411(model_name)


if __name__ == "__main__":
    main()
