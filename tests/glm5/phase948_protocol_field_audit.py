#!/usr/bin/env python3
"""
Phase 948: Protocol场token组成全量审计 + 激活来源追踪
=====================================================
Route C 起点 — 全光谱审计:

1. UNSUPERVISED PROTOCOL DISCOVERY: 不使用预设类别, 让模型行为揭示哪些token是protocol token
   - 100+ 多样化prompt (QA/分类/解释/结构化输出)
   - 全量logit分布分析, 识别跨上下文一致性出现的token
   - 按激活模式聚类protocol token

2. COMPONENT-LEVEL ATTRIBUTION: 逐层归零attention/MLP, 测量对protocol logit的影响
   - 哪些层/组件驱动protocol token的logit?
   - attention vs MLP 贡献分解

3. CROSS-MODEL COMPARISON: qwen3 → GLM4 → DS7B
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U, get_sample_layers, MODEL_CONFIGS

PHASE = 948
MODELS = ["qwen3", "glm4", "deepseek7b"]
RESULT_DIR = Path("results/phase948_protocol_field_audit")

# ===== PROMPT TEMPLATES — 多样化语境 =====

# 1. QA prompts (答案前有句号)
QA_PROMPTS = [
    "What is the capital of France? The answer is",
    "Who wrote Romeo and Juliet? The answer is",
    "What is the chemical symbol for water? The answer is",
    "Which planet is closest to the Sun? The answer is",
    "How many continents are there? The answer is",
    "What is the largest ocean? The answer is",
    "Who painted the Mona Lisa? The answer is",
    "What is the speed of light? The answer is",
    "Which element has atomic number 79? The answer is",
    "What is the longest river? The answer is",
    "In what year did World War II end? The answer is",
    "What is the square root of 144? The answer is",
    "Which country has the largest population? The answer is",
    "What is the boiling point of water in Celsius? The answer is",
    "Who discovered penicillin? The answer is",
]

# 2. Classification prompts
CLASS_PROMPTS = [
    "Classify: 'apple' is a type of",
    "Category: 'dog' belongs to the category of",
    "Subclass: 'rose' is a kind of",
    "Category: 'hammer' is a",
    "Classify: 'Mars' is a",
    "Category: 'oxygen' is a",
    "Classify: 'Shakespeare' was a",
    "Subclass: 'diamond' is a type of",
    "Category: 'Python' (programming language) is a",
    "Classify: 'eagle' is a kind of",
    "Category: 'copper' is a",
    "Classify: 'novel' is a form of",
    "Subclass: 'bicycle' is a type of",
    "Category: 'triangle' is a",
    "Classify: 'Beethoven' was a",
]

# 3. Explanation prompts
EXPLAIN_PROMPTS = [
    "Explain why the sky is blue:",
    "What causes earthquakes?",
    "How does photosynthesis work?",
    "Why do we need sleep?",
    "What is artificial intelligence?",
    "How do vaccines work?",
    "Why is the ocean salty?",
    "What causes seasons?",
    "How does a computer work?",
    "Why do we have leap years?",
    "What is gravity?",
    "How do airplanes fly?",
]

# 4. Structured output prompts (列表/字段)
STRUCTURED_PROMPTS = [
    "List three primary colors:\n1.",
    "Name two types of energy:\n-",
    "List the planets in order:\n1.",
    "Name the components of blood:\n-",
    "List four cardinal directions:\n1.",
    "Name three states of matter:\n-",
    "List the noble gases:\n1.",
    "Name two types of cell division:\n-",
    "List the human senses:\n1.",
    "Name the parts of a plant:\n-",
]

# 5. Completion with period ending (协议场激活语境)
PROTOCOL_CONTEXT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "The Earth revolves around the Sun once per year.",
    "Water freezes at zero degrees Celsius.",
    "Shakespeare wrote many famous plays.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The Industrial Revolution began in Britain.",
    "DNA contains the genetic instructions for life.",
    "Gravity is one of the four fundamental forces.",
    "The Renaissance was a period of great cultural change.",
]

# 6. Mid-sentence continuations (非协议场对照)
MID_SENTENCE_PROMPTS = [
    "The quick brown fox",
    "Machine learning is",
    "The Earth revolves",
    "Water freezes",
    "Shakespeare wrote",
    "Photosynthesis converts",
    "The Industrial Revolution",
    "DNA contains",
    "Gravity is",
    "The Renaissance was",
]

ALL_PROMPT_GROUPS = {
    "qa": QA_PROMPTS,
    "classification": CLASS_PROMPTS,
    "explanation": EXPLAIN_PROMPTS,
    "structured": STRUCTURED_PROMPTS,
    "protocol_context": PROTOCOL_CONTEXT_PROMPTS,
    "mid_sentence": MID_SENTENCE_PROMPTS,
}

# Sampling layers for component attribution
ATTRIBUTION_LAYER_STRIDE = 4  # every 4 layers + final


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ===== PART 1: 全量logit分布分析 =====

def get_topk_logits(logits: torch.Tensor, k: int = 50) -> dict:
    """Get top-k token IDs and their logit values."""
    topk_values, topk_indices = torch.topk(logits.float(), k=k)
    indices = topk_indices.cpu().numpy().tolist()
    values = topk_values.cpu().numpy().tolist()
    return {"indices": indices, "logits": values}


def safe_decode(tokenizer, token_id: int) -> str:
    """Safely decode a token ID."""
    try:
        result = tokenizer.decode([token_id], skip_special_tokens=False)
        return result if result else f"<tok_{token_id}>"
    except Exception:
        return f"<tok_{token_id}>"


def capture_logit_distribution(model, tokenizer, device, prompts: list[str],
                                max_k: int = 100) -> dict:
    """
    Run all prompts, capture full top-K logit distributions.
    Returns: {prompt_group: {token_id: {count, mean_logit, mean_rank}}}
    """
    all_results = []
    total = len(prompts)
    
    for idx, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            logits_last = outputs.logits[0, -1].detach().float().cpu()
        
        topk = get_topk_logits(logits_last, k=max_k)
        
        row = {
            "prompt": prompt,
            "topk_indices": topk["indices"],
            "topk_logits": topk["logits"],
            "topk_tokens": [safe_decode(tokenizer, tid) for tid in topk["indices"]],
        }
        
        # Also get softmax probs
        probs = torch.softmax(logits_last, dim=-1)
        row["topk_probs"] = [float(probs[tid].item()) for tid in topk["indices"]]
        
        all_results.append(row)
        
        if (idx + 1) % 20 == 0:
            log(f"  processed {idx + 1}/{total} prompts")
    
    return {"prompts": all_results, "total_prompts": total}


def analyze_protocol_tokens(results: dict, tokenizer, min_cross_prompt: int = 3) -> dict:
    """
    Discover protocol tokens: tokens that consistently appear in top-K
    across many different prompts.
    """
    prompt_list = results["prompts"]
    n_prompts = len(prompt_list)
    
    # Aggregate: token_id -> {prompt_indices, ranks, logits}
    token_stats = defaultdict(lambda: {"indices": [], "ranks": [], "logits": []})
    
    for pi, pdata in enumerate(prompt_list):
        for rank_i, (tid, logit) in enumerate(zip(pdata["topk_indices"], pdata["topk_logits"])):
            token_stats[tid]["indices"].append(pi)
            token_stats[tid]["ranks"].append(rank_i + 1)
            token_stats[tid]["logits"].append(logit)
    
    # Filter: appear in >= min_cross_prompt different prompts
    protocol_candidates = {}
    for tid, stats in token_stats.items():
        unique_prompts = len(set(stats["indices"]))
        if unique_prompts >= min_cross_prompt:
            protocol_candidates[tid] = {
                "token_id": tid,
                "token_str": safe_decode(tokenizer, tid),
                "cross_prompt_count": unique_prompts,
                "cross_prompt_ratio": unique_prompts / n_prompts,
                "mean_rank": float(np.mean(stats["ranks"])),
                "median_rank": float(np.median(stats["ranks"])),
                "mean_logit": float(np.mean(stats["logits"])),
                "std_rank": float(np.std(stats["ranks"])),
            }
    
    # Sort by cross-prompt frequency
    sorted_candidates = sorted(protocol_candidates.values(),
                               key=lambda x: (-x["cross_prompt_count"], x["mean_rank"]))
    
    # Also compute per-prompt-group statistics
    prompt_groups = {}
    for pi, pdata in enumerate(prompt_list):
        pg_key = "unknown"
        for group_name, group_prompts in ALL_PROMPT_GROUPS.items():
            if pdata["prompt"] in group_prompts:
                pg_key = group_name
                break
        if pg_key not in prompt_groups:
            prompt_groups[pg_key] = {"indices": [], "n": 0}
        prompt_groups[pg_key]["indices"].append(pi)
        prompt_groups[pg_key]["n"] += 1
    
    return {
        "protocol_candidates": sorted_candidates,
        "total_candidates": len(sorted_candidates),
        "prompt_groups": {k: v["n"] for k, v in prompt_groups.items()},
    }


# ===== PART 2: COMPONENT-LEVEL ATTRIBUTION =====

def zero_last_token_output(output):
    """Zero-out the last token position in a layer's output."""
    if isinstance(output, tuple):
        if not output or not torch.is_tensor(output[0]):
            return output
        patched = output[0].clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = 0
        return (patched, *output[1:])
    if torch.is_tensor(output):
        patched = output.clone()
        if patched.ndim >= 3:
            patched[:, -1, :] = 0
        return patched
    return output


def get_component_module(model, layer_idx: int, component_kind: str):
    """Get attention or MLP module at given layer."""
    layers = get_layers(model)
    if not (0 <= int(layer_idx) < len(layers)):
        return None
    layer = layers[int(layer_idx)]
    if component_kind == "attention":
        return getattr(layer, "self_attn", None)
    if component_kind == "mlp":
        return getattr(layer, "mlp", None)
    return None


def run_attribution(model, tokenizer, device, prompts: list[str],
                    layer_indices: list[int], protocol_token_ids: list[int],
                    max_prompts: int = 20) -> dict:
    """
    For each prompt, zero out each layer's attention or MLP,
    measure the logit change for protocol tokens.
    """
    selected_prompts = prompts[:max_prompts]
    total_layers = len(layer_indices)
    
    # Results structure: {token_id: {f"L{li}_{comp}": delta}}
    all_attributions = {}

    for pi, prompt in enumerate(selected_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Baseline
        with torch.no_grad():
            baseline_output = model(input_ids=input_ids, use_cache=False)
            baseline_logits = baseline_output.logits[0, -1].detach().float().cpu()
        
        for li_idx, layer_idx in enumerate(layer_indices):
            for comp in ["attention", "mlp"]:
                module = get_component_module(model, layer_idx, comp)
                if module is None:
                    continue
                
                # Zero this component
                handle = module.register_forward_hook(
                    lambda _m, _in, out: zero_last_token_output(out)
                )
                with torch.no_grad():
                    try:
                        patched_output = model(input_ids=input_ids, use_cache=False)
                        patched_logits = patched_output.logits[0, -1].detach().float().cpu()
                    except Exception:
                        patched_logits = baseline_logits.clone()
                handle.remove()
                
                key = f"L{layer_idx}_{comp}"
                
                for tid in protocol_token_ids:
                    if tid not in all_attributions:
                        all_attributions[tid] = {}
                    delta = float(patched_logits[tid].item() - baseline_logits[tid].item())
                    all_attributions[tid][key] = delta
        
        if (pi + 1) % 5 == 0:
            log(f"  attribution: {pi + 1}/{len(selected_prompts)} prompts done")
    
    # Aggregate across prompts
    aggregated = {}
    for tid, deltas in all_attributions.items():
        by_key = defaultdict(list)
        for key, delta in deltas.items():
            by_key[key].append(delta)
        
        aggregated[str(tid)] = {
            key: {
                "mean_delta": float(np.mean(vals)),
                "std_delta": float(np.std(vals)),
                "neg_count": sum(1 for v in vals if v < 0),
                "total": len(vals),
                "neg_ratio": sum(1 for v in vals if v < 0) / max(len(vals), 1),
            }
            for key, vals in by_key.items()
        }
    
    return {
        "n_prompts": len(selected_prompts),
        "n_layers_tested": total_layers,
        "attributions": aggregated,
    }


# ===== PART 3: PROTOCOL TOKEN CLUSTERING =====

def cluster_protocol_tokens(attribution_data: dict, top_k_tokens: int = 30) -> dict:
    """
    Cluster protocol tokens by their attribution patterns.
    Compute cosine similarity between attribution vectors.
    """
    aggregated = attribution_data.get("attributions", {})
    
    # Build feature vectors for each token
    token_ids = []
    feature_matrix = []
    feature_keys = []
    
    for tid_str, comp_data in aggregated.items():
        tid = int(tid_str)
        keys = sorted(comp_data.keys())
        
        # Feature: mean delta for each component
        features = [comp_data[k]["mean_delta"] for k in keys if "mean_delta" in comp_data[k]]
        
        if len(features) > 0:
            token_ids.append(tid)
            feature_matrix.append(features)
            if not feature_keys:
                feature_keys = keys
    
    if not feature_matrix:
        return {"error": "No features extracted"}
    
    X = np.array(feature_matrix)  # [n_tokens, n_features]
    
    # Normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    X_norm = X / norms
    
    # Cosine similarity matrix
    cos_sim = X_norm @ X_norm.T  # [n_tokens, n_tokens]
    
    # Simple clustering: group tokens with cos > 0.7
    clusters = []
    used = set()
    for i in range(len(token_ids)):
        if i in used:
            continue
        cluster_members = [token_ids[i]]
        used.add(i)
        for j in range(i + 1, len(token_ids)):
            if j not in used and cos_sim[i, j] > 0.7:
                cluster_members.append(token_ids[j])
                used.add(j)
        if len(cluster_members) >= 2:
            clusters.append({
                "members": cluster_members,
                "size": len(cluster_members),
                "intra_cluster_sim": float(np.mean([cos_sim[i, j] for j in range(i + 1, len(token_ids))
                                                     if j in [token_ids.index(m) for m in cluster_members]])),
            })
    
    return {
        "n_tokens": len(token_ids),
        "n_features": len(feature_keys),
        "feature_keys": feature_keys,
        "n_clusters": len(clusters),
        "clusters": sorted(clusters, key=lambda c: -c["size"]),
    }


# ===== MAIN =====

def run_phase948(args: argparse.Namespace) -> None:
    ensure_dir(RESULT_DIR)
    
    models_to_run = [args.model] if args.model != "all" else MODELS
    
    for model_name in models_to_run:
        log(f"{'='*60}")
        log(f"Phase 948: Protocol Field Audit — Model: {model_name}")
        log(f"{'='*60}")
        
        # Load model
        model, tokenizer, device = load_model(model_name)
        info = get_model_info(model, model_name)
        log(f"Model loaded: {info.model_class}, {info.n_layers} layers, d={info.d_model}")
        
        model_result_dir = RESULT_DIR / model_name
        ensure_dir(model_result_dir)
        
        # ===== STEP 1: Logit Distribution =====
        log("STEP 1: Full logit distribution analysis...")
        all_prompts = []
        for group_name, prompts in ALL_PROMPT_GROUPS.items():
            all_prompts.extend(prompts)
        
        distribution = capture_logit_distribution(
            model, tokenizer, device, all_prompts, max_k=100
        )
        
        # Save full distribution
        dist_path = model_result_dir / "logit_distribution.json"
        dist_path.write_text(json.dumps(distribution, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Saved full distribution to {dist_path}, {len(all_prompts)} prompts")
        
        # ===== STEP 2: Protocol Token Discovery =====
        log("STEP 2: Protocol token discovery...")
        protocol_analysis = analyze_protocol_tokens(
            distribution, tokenizer, min_cross_prompt=args.min_cross_prompt
        )
        
        candidates = protocol_analysis["protocol_candidates"]
        log(f"  Found {len(candidates)} protocol candidates")
        log(f"  Top 20:")
        for c in candidates[:20]:
            log(f"    {c['token_str']:20s}  cross={c['cross_prompt_count']:3d}/{len(all_prompts)}  "
                f"mean_rank={c['mean_rank']:.1f}  median_rank={c['median_rank']:.1f}")
        
        # Save protocol discovery
        disc_path = model_result_dir / "protocol_discovery.json"
        disc_path.write_text(json.dumps(protocol_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Saved protocol discovery to {disc_path}")
        
        # ===== STEP 3: Component Attribution =====
        log("STEP 3: Component-level attribution...")
        
        # Top protocol tokens to trace
        top_k = min(args.top_protocol_tokens, len(candidates))
        protocol_ids = [c["token_id"] for c in candidates[:top_k]]
        
        layer_indices = get_sample_layers(info.n_layers, n_samples=max(1, info.n_layers // ATTRIBUTION_LAYER_STRIDE))
        log(f"  Attribution layers: {layer_indices} ({len(layer_indices)} layers)")
        log(f"  Protocol tokens to trace: {len(protocol_ids)}")
        log(f"  Attribution prompts: {min(args.attr_prompts, len(all_prompts))}")
        
        attribution = run_attribution(
            model, tokenizer, device,
            all_prompts, layer_indices, protocol_ids,
            max_prompts=args.attr_prompts
        )
        
        attr_path = model_result_dir / "component_attribution.json"
        attr_path.write_text(json.dumps(attribution, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Saved attribution to {attr_path}")
        
        # ===== STEP 4: Token Clustering =====
        log("STEP 4: Protocol token clustering...")
        clustering = cluster_protocol_tokens(attribution, top_k_tokens=top_k)
        
        cluster_path = model_result_dir / "token_clusters.json"
        cluster_path.write_text(json.dumps(clustering, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Found {clustering.get('n_clusters', 0)} clusters")
        
        # ===== STEP 5: Summary =====
        log("STEP 5: Generating summary...")
        
        # Compute key metrics
        # 1. Protocolness score distribution
        cross_prompt_ratios = [c["cross_prompt_ratio"] for c in candidates]
        mean_ranks = [c["mean_rank"] for c in candidates]
        
        # 2. Dominant component (attention vs MLP) analysis
        aggregated = attribution.get("attributions", {})
        attn_contrib = []
        mlp_contrib = []
        for tid_str, comp_data in aggregated.items():
            attn_deltas = [comp_data[k]["mean_delta"] for k in comp_data
                          if "attention" in k and "mean_delta" in comp_data[k]]
            mlp_deltas = [comp_data[k]["mean_delta"] for k in comp_data
                         if "mlp" in k and "mean_delta" in comp_data[k]]
            if attn_deltas:
                attn_contrib.append(np.mean(np.abs(attn_deltas)))
            if mlp_deltas:
                mlp_contrib.append(np.mean(np.abs(mlp_deltas)))
        
        summary = {
            "phase": PHASE,
            "model": model_name,
            "model_info": {
                "class": info.model_class,
                "n_layers": info.n_layers,
                "d_model": info.d_model,
                "vocab_size": info.vocab_size,
                "mlp_type": info.mlp_type,
            },
            "total_prompts": len(all_prompts),
            "prompt_groups": {k: len(v) for k, v in ALL_PROMPT_GROUPS.items()},
            "protocol_discovery": {
                "n_candidates": len(candidates),
                "n_high_cross": sum(1 for c in candidates if c["cross_prompt_ratio"] > 0.3),
                "n_very_high_cross": sum(1 for c in candidates if c["cross_prompt_ratio"] > 0.5),
                "mean_cross_prompt_ratio": float(np.mean(cross_prompt_ratios)) if cross_prompt_ratios else 0,
                "mean_rank": float(np.mean(mean_ranks)) if mean_ranks else 0,
                "top_20": [
                    {"token": c["token_str"], "cross_ratio": c["cross_prompt_ratio"],
                     "mean_rank": c["mean_rank"]}
                    for c in candidates[:20]
                ],
            },
            "attribution": {
                "n_protocol_tokens_traced": len(protocol_ids),
                "n_layers_tested": len(layer_indices),
                "n_prompts_used": attribution.get("n_prompts", 0),
                "mean_attn_contribution": float(np.mean(attn_contrib)) if attn_contrib else 0,
                "mean_mlp_contribution": float(np.mean(mlp_contrib)) if mlp_contrib else 0,
                "attn_vs_mlp_ratio": float(np.mean(attn_contrib) / max(np.mean(mlp_contrib), 1e-10)) if attn_contrib and mlp_contrib else 0,
            },
            "clustering": {
                "n_clusters": clustering.get("n_clusters", 0),
            },
            "tokenizer_eos": str(safe_decode(tokenizer, tokenizer.eos_token_id)),
        }
        
        summary_path = model_result_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"  Saved summary to {summary_path}")
        
        # Print key findings
        log("")
        log(f"===== KEY FINDINGS for {model_name} =====")
        log(f"Protocol candidates: {len(candidates)}")
        log(f"High-cross (>30%): {summary['protocol_discovery']['n_high_cross']}")
        log(f"Very-high-cross (>50%): {summary['protocol_discovery']['n_very_high_cross']}")
        log(f"Mean cross-prompt ratio: {summary['protocol_discovery']['mean_cross_prompt_ratio']:.3f}")
        log(f"Mean rank: {summary['protocol_discovery']['mean_rank']:.1f}")
        log(f"Attn contribution (mean abs): {summary['attribution']['mean_attn_contribution']:.4f}")
        log(f"MLP contribution (mean abs): {summary['attribution']['mean_mlp_contribution']:.4f}")
        log(f"Attn/MLP ratio: {summary['attribution']['attn_vs_mlp_ratio']:.3f}")
        log(f"Token clusters: {clustering.get('n_clusters', 0)}")
        log(f"Tokenizer EOS: '{safe_decode(tokenizer, tokenizer.eos_token_id)}'")
        
        # Release GPU memory
        release_model(model)
        log(f"Completed {model_name}")
    
    log("Phase 948 complete for all models!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 948: Protocol Field Audit")
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b", "all"],
                       help="Model to test")
    parser.add_argument("--min_cross_prompt", type=int, default=3,
                       help="Minimum cross-prompt count for protocol candidate")
    parser.add_argument("--top_protocol_tokens", type=int, default=30,
                       help="Top N protocol tokens to trace")
    parser.add_argument("--attr_prompts", type=int, default=15,
                       help="Number of prompts for attribution analysis")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log(f"Phase {PHASE} started, model={args.model}")
    run_phase948(args)
