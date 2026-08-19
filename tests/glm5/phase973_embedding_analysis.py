#!/usr/bin/env python3
"""
Phase 973: 词嵌入提取与基础语义族分析 (E001 WP0-WP1)
=====================================================
WP0: 提取三模型输入/输出嵌入矩阵, 记录元信息
WP1: 建立24个语义族清单, 计算族内/族间余弦相似度

不加载完整模型, 只读取embedding权重(safetensors), 避免GPU占用.
"""

from __future__ import annotations
import gc, json, sys, time, os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
sys.path.insert(0, str(ROOT / "tests" / "gpt5"))
from phase951_protocol_atlas import ensure_dir

PHASE = 973
RESULT_DIR = Path("results/phase973_large_validation/embedding_analysis")
MODEL_PATHS = {
    "qwen3": "D:/AI2050/Ai2050-OpenOne/models/hf/qwen3-4b",
    "glm4": "D:/AI2050/Ai2050-OpenOne/models/hf/glm4-9b-chat-hf",
    "deepseek7b": "D:/AI2050/Ai2050-OpenOne/models/hf/deepseek-r1-distill-qwen-7b",
}

# ============================================================
# 24个语义族 (英文, 单词元优先)
# ============================================================
SEMANTIC_FAMILIES = {
    "animals": ["cat", "dog", "lion", "tiger", "bird", "fish", "horse", "cow", "pig", "sheep",
                 "elephant", "monkey", "snake", "rabbit", "bear", "wolf", "deer", "mouse", "duck", "chicken"],
    "fruits": ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "cherry", "strawberry", "watermelon",
               "pineapple", "mango", "kiwi", "melon", "berry", "plum", "apricot", "fig", "coconut", "lime"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "brown",
               "gray", "grey", "gold", "silver", "violet", "cyan", "crimson", "scarlet", "amber", "teal"],
    "body_parts": ["hand", "eye", "heart", "brain", "ear", "nose", "mouth", "foot", "leg", "arm",
                   "finger", "head", "neck", "back", "stomach", "knee", "shoulder", "chest", "liver", "lung"],
    "vehicles": ["car", "bus", "train", "plane", "ship", "boat", "bike", "truck", "bicycle", "motorcycle",
                 "helicopter", "subway", "taxi", "tractor", "ambulance", "rocket", "yacht", "ferry", "jet", "tram"],
    "tools": ["hammer", "knife", "saw", "drill", "screwdriver", "wrench", "pliers", "scissors", "axe", "shovel",
              "rake", "chisel", "file", "clamp", "mallet", "plane", "measure", "level", "welder", "grinder"],
    "metals": ["iron", "gold", "silver", "copper", "steel", "aluminum", "tin", "lead", "zinc", "nickel",
               "mercury", "platinum", "titanium", "brass", "bronze", "chrome", "cobalt", "magnesium", "sodium", "calcium"],
    "emotions": ["happy", "sad", "angry", "fear", "love", "hate", "joy", "sorrow", "rage", "terror",
                 "delight", "grief", "excitement", "calm", "anxious", "proud", "ashamed", "hopeful", "despair", "envy"],
    "actions": ["run", "walk", "jump", "swim", "fly", "eat", "drink", "sleep", "think", "speak",
                "write", "read", "sing", "dance", "fight", "play", "work", "rest", "climb", "crawl"],
    "natures": ["tree", "flower", "grass", "mountain", "river", "ocean", "forest", "desert", "valley", "hill",
                "cloud", "rain", "snow", "wind", "storm", "sun", "moon", "star", "earth", "sky"],
    "buildings": ["house", "school", "hospital", "church", "bank", "factory", "office", "store", "library", "museum",
                  "hotel", "tower", "bridge", "castle", "palace", "cabin", "cottage", "barn", "garage", "shelter"],
    "clothes": ["shirt", "pants", "dress", "shoe", "hat", "coat", "jacket", "glove", "sock", "belt",
                "tie", "scarf", "skirt", "shorts", "sweater", "blouse", "vest", "cape", "robe", "uniform"],
    "food": ["bread", "rice", "meat", "egg", "milk", "cheese", "butter", "soup", "cake", "cookie",
             "pie", "pasta", "noodle", "sandwich", "salad", "steak", "chicken", "fish", "beef", "pork"],
    "professions": ["doctor", "teacher", "lawyer", "engineer", "farmer", "cook", "driver", "nurse", "pilot", "artist",
                    "writer", "singer", "actor", "farmer", "miner", "fisher", "hunter", "trader", "soldier", "police"],
    "places": ["city", "town", "village", "country", "island", "beach", "park", "zoo", "garden", "farm",
               "forest", "desert", "mountain", "valley", "lake", "river", "ocean", "sea", "harbor", "border"],
    "time": ["day", "night", "morning", "evening", "noon", "week", "month", "year", "hour", "minute",
             "second", "century", "decade", "season", "spring", "summer", "autumn", "winter", "today", "tomorrow"],
    "numbers": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "hundred", "thousand", "million", "billion", "zero", "first", "second", "third", "half", "quarter"],
    "prepositions": ["in", "on", "at", "to", "from", "with", "by", "for", "of", "about",
                     "under", "over", "through", "between", "against", "during", "before", "after", "above", "below"],
    "conjunctions": ["and", "but", "or", "so", "because", "although", "if", "unless", "since", "while",
                     "whereas", "therefore", "however", "moreover", "furthermore", "nevertheless", "thus", "hence", "yet", "nor"],
    "punctuation": [".", ",", "!", "?", ";", ":", "-", "(", ")", "\"", "'",
                    "/", "\\", "@", "#", "$", "%", "&", "*", "+"],
    "science": ["atom", "cell", "gene", "force", "energy", "mass", "light", "sound", "heat", "cold",
                "acid", "base", "salt", "gas", "liquid", "solid", "vapor", "wave", "particle", "field"],
    "math": ["add", "subtract", "multiply", "divide", "sum", "product", "ratio", "angle", "line", "point",
             "circle", "square", "triangle", "sphere", "cube", "curve", "slope", "area", "volume", "radius"],
    "weather": ["sunny", "cloudy", "rainy", "windy", "snowy", "foggy", "stormy", "hot", "cold", "warm",
                "cool", "dry", "wet", "humid", "clear", "overcast", "breezy", "frosty", "muggy", "frigid"],
    "family": ["father", "mother", "son", "daughter", "brother", "sister", "uncle", "aunt", "grandfather", "grandmother",
               "cousin", "nephew", "niece", "husband", "wife", "parent", "child", "baby", "twin", "relative"],
}


def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_embeddings(model_name: str) -> Dict[str, Any]:
    """WP0: 从safetensors提取嵌入矩阵, 不加载完整模型."""
    model_path = MODEL_PATHS[model_name]
    log(f"  Extracting embeddings for {model_name} from {model_path}")
    
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    
    # Load config and tokenizer
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
    
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    log(f"    vocab_size={vocab_size}, hidden_size={hidden_size}")
    
    # Find embedding weights in safetensors
    embed_weight = None
    lm_head_weight = None
    sf_files = sorted(Path(model_path).glob("*.safetensors"))
    log(f"    Scanning {len(sf_files)} safetensors files...")
    
    for sf_file in sf_files:
        with safe_open(str(sf_file), framework="pt", device="cpu") as sf:
            keys = list(sf.keys())
            for key in keys:
                if key in ["model.embed_tokens.weight", "model.embed_embeddings.weight",
                           "transformer.wte.weight", "embed_tokens.weight"]:
                    embed_weight = sf.get_tensor(key).float().numpy()
                    log(f"    Found input embedding: {key} shape={embed_weight.shape}")
                elif key in ["lm_head.weight", "embed_out.weight"]:
                    lm_head_weight = sf.get_tensor(key).float().numpy()
                    log(f"    Found output embedding: {key} shape={lm_head_weight.shape}")
    
    # Check if tied weights
    tied = (embed_weight is not None and lm_head_weight is not None and 
            np.array_equal(embed_weight, lm_head_weight))
    if embed_weight is not None and lm_head_weight is None:
        lm_head_weight = embed_weight
        tied = True
        log(f"    lm_head not found, assuming tied weights")
    
    if embed_weight is None:
        raise ValueError(f"Could not find embedding weights for {model_name}")
    
    # Numerical audit
    embed_norms = np.linalg.norm(embed_weight, axis=1)
    has_nan = np.isnan(embed_weight).any()
    has_inf = np.isinf(embed_weight).any()
    log(f"    Numerical: has_nan={has_nan}, has_inf={has_inf}")
    log(f"    Embed norms: mean={embed_norms.mean():.4f}, std={embed_norms.std():.4f}, "
        f"min={embed_norms.min():.4f}, max={embed_norms.max():.4f}")
    
    # Special tokens
    special_tokens = []
    for attr in ["bos_token", "eos_token", "pad_token", "unk_token"]:
        tok = getattr(tokenizer, attr, None)
        if tok:
            tid = tokenizer.convert_tokens_to_ids(tok)
            special_tokens.append({"name": attr, "token": tok, "id": tid})
    log(f"    Special tokens: {special_tokens}")
    
    return {
        "model_name": model_name,
        "config": {"vocab_size": vocab_size, "hidden_size": hidden_size, 
                   "model_type": getattr(config, "model_type", "unknown")},
        "embed_weight": embed_weight,
        "lm_head_weight": lm_head_weight,
        "tied": tied,
        "tokenizer": tokenizer,
        "embed_norms": embed_norms,
        "special_tokens": special_tokens,
    }


def wp1_semantic_family_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """WP1: 计算语义族内/族间余弦相似度."""
    model_name = data["model_name"]
    embed = data["embed_weight"]
    tokenizer = data["tokenizer"]
    vocab_size = embed.shape[0]
    
    log(f"  WP1: Semantic family analysis for {model_name}")
    
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embed_normed = embed / norms  # [vocab, hidden]
    
    # For each family, find token IDs
    family_tokens = {}
    family_stats = {}
    for family_name, words in SEMANTIC_FAMILIES.items():
        token_ids = []
        token_texts = []
        for word in words:
            # Try with leading space and without
            for variant in [word, f" {word}", word.capitalize(), f" {word.capitalize()}"]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1 and 0 <= ids[0] < vocab_size:
                    token_ids.append(ids[0])
                    token_texts.append(variant)
                    break
        family_tokens[family_name] = list(set(token_ids))  # dedupe
        
        if len(family_tokens[family_name]) < 3:
            log(f"    {family_name}: only {len(family_tokens[family_name])} single-token words found, skipping")
            continue
        
        # Compute intra-family cosine similarity
        ids = family_tokens[family_name]
        vecs = embed_normed[ids]  # [n, hidden]
        cos_matrix = vecs @ vecs.T  # [n, n]
        # Exclude diagonal
        n = len(ids)
        mask = ~np.eye(n, dtype=bool)
        intra_cos = cos_matrix[mask]
        intra_mean = float(intra_cos.mean())
        intra_std = float(intra_cos.std())
        
        family_stats[family_name] = {
            "n_tokens": n,
            "intra_cos_mean": intra_mean,
            "intra_cos_std": intra_std,
            "token_ids": ids,
            "token_texts": [tokenizer.decode([i]) for i in ids],
        }
        log(f"    {family_name}: {n} tokens, intra_cos={intra_mean:.4f}±{intra_std:.4f}")
    
    # Compute inter-family cosine similarity
    family_names = list(family_stats.keys())
    inter_matrix = {}
    for i, f1 in enumerate(family_names):
        for j, f2 in enumerate(family_names):
            if j <= i:
                continue
            ids1 = family_stats[f1]["token_ids"]
            ids2 = family_stats[f2]["token_ids"]
            vecs1 = embed_normed[ids1]
            vecs2 = embed_normed[ids2]
            cos_block = vecs1 @ vecs2.T  # [n1, n2]
            inter_mean = float(cos_block.mean())
            inter_matrix[f"{f1}__{f2}"] = inter_mean
    
    # Summary: intra vs inter
    intra_means = [family_stats[f]["intra_cos_mean"] for f in family_names]
    inter_means = list(inter_matrix.values())
    log(f"  Summary ({model_name}):")
    log(f"    Intra-family cos: mean={np.mean(intra_means):.4f} (range {np.min(intra_means):.4f}-{np.max(intra_means):.4f})")
    log(f"    Inter-family cos: mean={np.mean(inter_means):.4f} (range {np.min(inter_means):.4f}-{np.max(inter_means):.4f})")
    log(f"    Separation (intra-inter): {np.mean(intra_means)-np.mean(inter_means):.4f}")
    
    # Physical spectral analysis (top singular values)
    U, S, Vt = np.linalg.svd(embed_normed, full_matrices=False)
    log(f"    Top 10 singular values: {S[:10].round(3)}")
    # Effective rank
    p = S / S.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-10))))
    participation_ratio = float((S.sum()**2) / (S**2).sum())
    log(f"    Effective rank: {eff_rank:.1f}")
    log(f"    Participation ratio: {participation_ratio:.1f}")
    log(f"    Top-10 energy: {S[:10].sum()/S.sum():.4f}")
    
    return {
        "model_name": model_name,
        "family_stats": family_stats,
        "inter_matrix": inter_matrix,
        "intra_mean": float(np.mean(intra_means)),
        "inter_mean": float(np.mean(inter_means)),
        "separation": float(np.mean(intra_means) - np.mean(inter_means)),
        "singular_values": S[:50].tolist(),
        "effective_rank": eff_rank,
        "participation_ratio": participation_ratio,
        "top10_energy": float(S[:10].sum() / S.sum()),
    }


def main():
    ensure_dir(RESULT_DIR)
    log(f"Phase {PHASE}: Embedding Analysis (E001 WP0-WP1)")
    log(f"="*60)
    
    model_name = sys.argv[1] if len(sys.argv) > 1 else None
    models_to_run = [model_name] if model_name else ["qwen3", "glm4", "deepseek7b"]
    
    all_results = {}
    for m in models_to_run:
        try:
            log(f"\n{'='*60}\nProcessing {m}\n{'='*60}")
            data = extract_embeddings(m)
            wp1 = wp1_semantic_family_analysis(data)
            all_results[m] = wp1
            
            # Save per-model results
            save = {k: v for k, v in wp1.items() if k != "embed_weight"}
            (RESULT_DIR / f"{m}_embedding_analysis.json").write_text(
                json.dumps(save, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            
            # Free memory
            del data["embed_weight"]
            del data["lm_head_weight"]
            del data
            gc.collect()
            
        except Exception as e:
            log(f"  {m} FAILED: {e}")
            import traceback; traceback.print_exc()
    
    # Cross-model comparison
    if len(all_results) >= 2:
        log(f"\n{'='*60}\nCross-Model Comparison\n{'='*60}")
        log(f"{'Model':<12} {'Intra':>8} {'Inter':>8} {'Sep':>8} {'EffRank':>8} {'Top10%':>8}")
        for m, r in all_results.items():
            log(f"{m:<12} {r['intra_mean']:>8.4f} {r['inter_mean']:>8.4f} "
                f"{r['separation']:>8.4f} {r['effective_rank']:>8.1f} {r['top10_energy']:>8.4f}")
    
    # Save summary
    summary = {m: {k: v for k, v in r.items() if k != "family_stats" and k != "inter_matrix"}
               for m, r in all_results.items()}
    (RESULT_DIR / "cross_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"\nDone! Results saved to {RESULT_DIR}")


if __name__ == "__main__":
    main()
