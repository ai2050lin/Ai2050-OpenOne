"""
Phase 414: W_U Candidate Direction Structure Analysis
=====================================================

Phase 409-411发现: 温度/速度的up-reversal比down-reversal容易。
假设: 这种非对称性来自W_U中候选词方向的结构——"hot"方向比"cold"方向更强。

本实验直接检验这个假设:
1. 计算W_U中各候选词方向的范数、余弦相似度、子空间结构
2. 检验"正极性方向"(hot/fast/huge)是否比"负极性方向"(cold/slow/tiny)范数更大
3. 分析候选词方向之间的夹角和子空间结构
4. 跨模型比较W_U方向结构

如果hot方向确实比cold方向范数更大/投影更强,
则非对称反转有了数学解释: 规则更容易把对象推到W_U中的强方向。

Usage:
  python tests/glm5/phase414_wu_direction.py qwen3
  python tests/glm5/phase414_wu_direction.py glm4
  python tests/glm5/phase414_wu_direction.py deepseek7b
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

# ===== 候选词定义 =====
ATTRIBUTE_CANDIDATES = OrderedDict({
    "temperature": {
        "low_polarity": ["freezing", "cold", "cool"],      # 负极性(冷)
        "high_polarity": ["warm", "hot", "scorching"],      # 正极性(热)
        "all": ["freezing", "cold", "cool", "warm", "hot", "scorching"],
        "levels": {"freezing": 1, "cold": 2, "cool": 3, "warm": 4, "hot": 5, "scorching": 6},
    },
    "speed": {
        "low_polarity": ["sluggish", "slow", "steady"],     # 负极性(慢)
        "high_polarity": ["quick", "fast", "rapid", "swift"],  # 正极性(快)
        "all": ["sluggish", "slow", "steady", "quick", "fast", "rapid", "swift"],
        "levels": {"sluggish": 1, "slow": 2, "steady": 3, "quick": 5, "fast": 6, "rapid": 7, "swift": 8},
    },
    "size": {
        "low_polarity": ["microscopic", "tiny", "small"],   # 负极性(小)
        "high_polarity": ["large", "huge", "massive"],       # 正极性(大)
        "all": ["microscopic", "tiny", "small", "large", "huge", "massive"],
        "levels": {"microscopic": 1, "tiny": 2, "small": 3, "large": 5, "huge": 6, "massive": 7},
    },
})


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def analyze_wu_directions(W_U_np, tokenizer, model_name):
    """分析W_U中候选词方向的结构"""

    results = {"model": model_name, "attributes": {}}

    for attr_name, attr_config in ATTRIBUTE_CANDIDATES.items():
        print(f"\n{'='*60}")
        print(f"=== Attribute: {attr_name} ===")

        # 1. 获取所有候选词的token ID和W_U方向
        candidate_dirs = {}
        candidate_norms = {}
        candidate_ids = {}

        for cand_name in attr_config["all"]:
            ids = tokenizer.encode(cand_name, add_special_tokens=False)
            tid = ids[0] if ids else None
            candidate_ids[cand_name] = tid

            if tid is not None and tid < W_U_np.shape[0]:
                direction = W_U_np[tid].copy()
                norm = np.linalg.norm(direction)
                candidate_dirs[cand_name] = direction
                candidate_norms[cand_name] = norm
            else:
                print(f"  WARNING: {cand_name} (id={tid}) not in W_U")

        if len(candidate_dirs) < 4:
            print(f"  SKIP: Too few valid candidates")
            continue

        print(f"  Valid candidates: {list(candidate_dirs.keys())}")

        # 2. 方向范数分析
        print(f"\n  --- Direction Norms ---")
        norm_results = {}
        for cand_name, norm in sorted(candidate_norms.items(), key=lambda x: x[1], reverse=True):
            level = attr_config["levels"].get(cand_name, 0)
            polarity = "high" if cand_name in attr_config["high_polarity"] else "low"
            print(f"    {cand_name:<15} norm={norm:.4f}  level={level}  polarity={polarity}")
            norm_results[cand_name] = {"norm": float(norm), "level": level, "polarity": polarity}

        # 3. 极性范数比较
        low_norms = [candidate_norms[n] for n in attr_config["low_polarity"] if n in candidate_norms]
        high_norms = [candidate_norms[n] for n in attr_config["high_polarity"] if n in candidate_norms]

        low_mean_norm = float(np.mean(low_norms)) if low_norms else 0
        high_mean_norm = float(np.mean(high_norms)) if high_norms else 0
        norm_ratio = high_mean_norm / low_mean_norm if low_mean_norm > 0 else float('inf')

        print(f"\n  --- Polarity Norm Comparison ---")
        print(f"    Low polarity mean norm: {low_mean_norm:.4f} ({attr_config['low_polarity']})")
        print(f"    High polarity mean norm: {high_mean_norm:.4f} ({attr_config['high_polarity']})")
        print(f"    High/Low ratio: {norm_ratio:.4f}")
        print(f"    --> {'High polarity STRONGER' if norm_ratio > 1 else 'Low polarity STRONGER'}")

        # 4. 方向间余弦相似度
        print(f"\n  --- Cosine Similarity Matrix ---")
        cand_names = list(candidate_dirs.keys())
        n_cands = len(cand_names)
        cos_matrix = np.zeros((n_cands, n_cands))

        for i in range(n_cands):
            for j in range(n_cands):
                di = candidate_dirs[cand_names[i]]
                dj = candidate_dirs[cand_names[j]]
                ni = np.linalg.norm(di)
                nj = np.linalg.norm(dj)
                if ni > 0 and nj > 0:
                    cos_matrix[i, j] = float(np.dot(di, dj) / (ni * nj))

        # 打印缩略版
        print(f"    {'':>15}", end="")
        for cn in cand_names[:4]:
            print(f" {cn[:6]:>8}", end="")
        if len(cand_names) > 4:
            print(f" ...", end="")
        print()
        for i, cn in enumerate(cand_names):
            print(f"    {cn:>15}", end="")
            for j in range(min(4, n_cands)):
                print(f" {cos_matrix[i, j]:>+8.3f}", end="")
            if len(cand_names) > 4:
                print(f" ...", end="")
            print()

        # 5. 低极性内/高极性内/跨极性 平均余弦
        low_cos = []
        high_cos = []
        cross_cos = []

        for i in range(n_cands):
            for j in range(i+1, n_cands):
                pi = "high" if cand_names[i] in attr_config["high_polarity"] else "low"
                pj = "high" if cand_names[j] in attr_config["high_polarity"] else "low"
                if pi == "low" and pj == "low":
                    low_cos.append(cos_matrix[i, j])
                elif pi == "high" and pj == "high":
                    high_cos.append(cos_matrix[i, j])
                else:
                    cross_cos.append(cos_matrix[i, j])

        print(f"\n  --- Intra/Cross Polarity Cosine ---")
        print(f"    Low intra-polarity mean cos: {np.mean(low_cos):+.4f}" if low_cos else "    Low intra: N/A")
        print(f"    High intra-polarity mean cos: {np.mean(high_cos):+.4f}" if high_cos else "    High intra: N/A")
        print(f"    Cross-polarity mean cos: {np.mean(cross_cos):+.4f}" if cross_cos else "    Cross: N/A")

        # 6. 属性主方向 (PCA)
        from sklearn.decomposition import PCA

        dir_matrix = np.array([candidate_dirs[cn] for cn in cand_names])
        n_components = min(3, dir_matrix.shape[0] - 1)
        pca = PCA(n_components=n_components)
        pca.fit(dir_matrix)

        print(f"\n  --- PCA of Candidate Directions ---")
        for k in range(n_components):
            print(f"    PC{k+1}: explained_variance_ratio = {pca.explained_variance_ratio_[k]:.4f}")

        # 各候选词在PC1上的投影
        projections = pca.transform(dir_matrix)
        print(f"    PC1 projections:")
        for i, cn in enumerate(cand_names):
            level = attr_config["levels"].get(cn, 0)
            print(f"      {cn:<15} PC1={projections[i, 0]:+.4f}  level={level}")

        # PC1与level的spearman相关
        levels_list = [attr_config["levels"].get(cn, 0) for cn in cand_names]
        pc1_projs = projections[:, 0]
        if len(levels_list) > 2:
            from scipy.stats import spearmanr
            pc1_level_corr, _ = spearmanr(levels_list, pc1_projs)
            print(f"    PC1-level correlation: {pc1_level_corr:+.4f}")

        # 7. Norm与level的spearman相关
        norm_list = [candidate_norms[cn] for cn in cand_names]
        if len(levels_list) > 2:
            from scipy.stats import spearmanr
            norm_level_corr, _ = spearmanr(levels_list, norm_list)
            print(f"\n  --- Norm-Level Correlation ---")
            print(f"    Spearman corr(norm, level): {norm_level_corr:+.4f}")
            print(f"    --> {'High level = larger norm' if norm_level_corr > 0 else 'High level = smaller norm'}")

        # 保存结果
        results["attributes"][attr_name] = {
            "norm_results": norm_results,
            "low_mean_norm": low_mean_norm,
            "high_mean_norm": high_mean_norm,
            "norm_ratio": norm_ratio,
            "norm_level_corr": float(norm_level_corr) if len(levels_list) > 2 else 0,
            "low_intra_cos": float(np.mean(low_cos)) if low_cos else 0,
            "high_intra_cos": float(np.mean(high_cos)) if high_cos else 0,
            "cross_cos": float(np.mean(cross_cos)) if cross_cos else 0,
            "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
            "pc1_level_corr": float(pc1_level_corr) if len(levels_list) > 2 else 0,
            "pc1_projections": {cn: float(projections[i, 0]) for i, cn in enumerate(cand_names)},
        }

    return results


def run_phase414(model_name):
    """Phase 414主函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*80}")
    print(f"=== Phase 414: W_U Direction Structure Analysis ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")

    # Load model (only need W_U, not full inference)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )

    # Load model just for W_U
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

    # Get W_U
    W_U_np = get_W_U(model, model_name)
    info = get_model_info(model, model_name)
    print(f"  W_U: shape={W_U_np.shape}")

    # Run analysis
    results = analyze_wu_directions(W_U_np, tokenizer, model_name)
    results["timestamp"] = timestamp
    results["n_layers"] = info.n_layers
    results["d_model"] = info.d_model
    results["phase"] = 414

    # ===== Cross-attribute summary =====
    print(f"\n{'='*80}")
    print(f"=== Cross-Attribute Summary ({model_name}) ===")
    print(f"{'='*80}")

    print(f"\n  Polarity Norm Ratios (high/low):")
    for attr_name, attr_result in results["attributes"].items():
        ratio = attr_result["norm_ratio"]
        direction = "HIGH stronger" if ratio > 1 else "LOW stronger"
        print(f"    {attr_name}: ratio={ratio:.4f} ({direction})")

    print(f"\n  Norm-Level Correlation:")
    for attr_name, attr_result in results["attributes"].items():
        corr = attr_result["norm_level_corr"]
        direction = "higher level = larger norm" if corr > 0 else "higher level = smaller norm"
        print(f"    {attr_name}: corr={corr:+.4f} ({direction})")

    print(f"\n  PC1-Level Correlation:")
    for attr_name, attr_result in results["attributes"].items():
        corr = attr_result["pc1_level_corr"]
        print(f"    {attr_name}: corr={corr:+.4f}")

    # ===== Save results =====
    results_dir = ROOT / "results" / "phase414_wu_direction"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"{model_name}_phase414.json"

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

    results = convert(results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Release model
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase414(model_name)


if __name__ == "__main__":
    main()
