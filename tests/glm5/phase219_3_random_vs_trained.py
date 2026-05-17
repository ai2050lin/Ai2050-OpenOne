"""
Phase 219-3: 训练vs随机模型的MLP约束传播对照

Phase 219-2的核心发现:
  - MLP有"双重角色": 某些层增强约束,某些层抑制约束
  - 形成约4-6层周期的振荡模式
  - L0 MLP贡献最大(53.6%), L27 MLP抑制最强(-40.1%)

关键问题: 这种振荡是训练效应还是架构效应?

如果训练模型有振荡而随机模型没有 → 振荡是训练习得的
如果两者都有振荡 → 振荡是架构的数学必然

同时进行多约束实验:
  - 数约束 (sg/pl)
  - 时态约束 (present/past)
  - 看它们是否有独立的传播路径

执行时间: 2026-05-17 20:55
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
import json
import time
from pathlib import Path

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SVA_PAIRS = [
    ("The cat chases", "The cats chase"),
    ("The dog runs", "The dogs run"),
    ("The bird sings", "The birds sing"),
    ("The girl reads", "The girls read"),
    ("The boy walks", "The boys walk"),
    ("The tree falls", "The trees fall"),
    ("The car moves", "The cars move"),
    ("The child plays", "The children play"),
    ("The woman writes", "The women write"),
    ("The man speaks", "The men speak"),
]

TENSE_PAIRS = [
    ("The cat chases", "The cat chased"),
    ("The dog runs", "The dog ran"),
    ("The bird sings", "The bird sang"),
    ("The girl reads", "The girl read"),
    ("The boy walks", "The boy walked"),
]

def compute_kl(p, q, eps=1e-10):
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum()
    q = q / q.sum()
    return (0.5 * (p * (p/q).log()).sum() + 0.5 * (q * (q/p).log()).sum()).item()

def analyze_mlp_ablation(model, pairs, label=""):
    """分析MLP逐层ablation"""
    print(f"\n=== MLP Ablation: {label} ===")
    
    n_layers = model.cfg.n_layers
    
    # 基线KL
    baseline_kl = []
    for s1, s2 in pairs:
        with torch.no_grad():
            _, cache_s1 = model.run_with_cache(s1)
            _, cache_s2 = model.run_with_cache(s2)
        
        pair_kl = []
        for l in range(n_layers):
            h1 = cache_s1["resid_post", l][0, -1]
            h2 = cache_s2["resid_post", l][0, -1]
            logits1 = h1.float() @ model.W_U.float()
            logits2 = h2.float() @ model.W_U.float()
            probs1 = torch.softmax(logits1, dim=-1)
            probs2 = torch.softmax(logits2, dim=-1)
            kl = compute_kl(probs1, probs2)
            pair_kl.append(kl)
        baseline_kl.append(pair_kl)
    
    baseline_kl = np.array(baseline_kl)
    
    # MLP Ablation
    ablation_reductions = []
    for l in range(n_layers):
        kl_after = []
        for s1, s2 in pairs:
            with torch.no_grad():
                _, cache_s1 = model.run_with_cache(s1)
                _, cache_s2 = model.run_with_cache(s2)
            
            h1_ablated = cache_s1["resid_post", l][0, -1] - cache_s1["mlp_out", l][0, -1]
            h2_ablated = cache_s2["resid_post", l][0, -1] - cache_s2["mlp_out", l][0, -1]
            
            logits1 = h1_ablated.float() @ model.W_U.float()
            logits2 = h2_ablated.float() @ model.W_U.float()
            probs1 = torch.softmax(logits1, dim=-1)
            probs2 = torch.softmax(logits2, dim=-1)
            kl = compute_kl(probs1, probs2)
            kl_after.append(kl)
        
        mean_kl = np.mean(kl_after)
        baseline = baseline_kl.mean(axis=0)[l]
        reduction = 1.0 - mean_kl / max(baseline, 1e-10)
        ablation_reductions.append(reduction * 100)
    
    # 打印
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            sign = "+" if ablation_reductions[l] > 0 else ""
            print(f"  L{l:2d}: {sign}{ablation_reductions[l]:.1f}%")
    
    return ablation_reductions, baseline_kl.mean(axis=0).tolist()

def analyze_constraint_direction_correlation(model, sva_pairs, tense_pairs):
    """
    分析不同约束的传播方向相关性
    如果数约束和时态约束在不同子空间传播 → 支持组合性
    """
    print("\n=== 约束方向相关性 ===")
    
    n_layers = model.cfg.n_layers
    
    correlations = []
    
    for l in range(n_layers):
        delta_num = []  # 数约束方向
        delta_tense = []  # 时态约束方向
        
        # 数约束: sg vs pl
        for sg, pl in sva_pairs[:5]:
            with torch.no_grad():
                _, cache_sg = model.run_with_cache(sg)
                _, cache_pl = model.run_with_cache(pl)
            
            d = cache_sg["resid_post", l][0, -1] - cache_pl["resid_post", l][0, -1]
            delta_num.append(d.float())
        
        # 时态约束: present vs past
        for pres, past in tense_pairs:
            with torch.no_grad():
                _, cache_pres = model.run_with_cache(pres)
                _, cache_past = model.run_with_cache(past)
            
            d = cache_pres["resid_post", l][0, -1] - cache_past["resid_post", l][0, -1]
            delta_tense.append(d.float())
        
        # 计算数约束和时态约束方向的平均cosine similarity
        cos_sims = []
        for dn in delta_num:
            for dt in delta_tense:
                cs = torch.nn.functional.cosine_similarity(dn.unsqueeze(0), dt.unsqueeze(0)).item()
                cos_sims.append(cs)
        
        mean_cos = np.mean(cos_sims)
        correlations.append(mean_cos)
    
    print(f"\n  Layer | Num-Tense cos_sim")
    print("  " + "-" * 30)
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            print(f"  L{l:2d}   | {correlations[l]:.4f}")
    
    return correlations

def analyze_mlp_increment_angle(model, pairs, label=""):
    """分析MLP增量与前一层的角度"""
    print(f"\n=== MLP增量角度: {label} ===")
    
    n_layers = model.cfg.n_layers
    angles = []
    
    for s1, s2 in pairs[:5]:
        with torch.no_grad():
            _, cache_s1 = model.run_with_cache(s1)
            _, cache_s2 = model.run_with_cache(s2)
        
        pair_angles = []
        prev_delta = None
        
        for l in range(n_layers):
            delta_mlp = (cache_s1["mlp_out", l][0, -1] - cache_s2["mlp_out", l][0, -1]).float()
            
            if prev_delta is not None and prev_delta.norm() > 1e-8 and delta_mlp.norm() > 1e-8:
                cs = torch.nn.functional.cosine_similarity(delta_mlp.unsqueeze(0), prev_delta.unsqueeze(0)).item()
                pair_angles.append(cs)
            else:
                pair_angles.append(0.0)
            
            prev_delta = (cache_s1["resid_post", l][0, -1] - cache_s2["resid_post", l][0, -1]).float()
        
        angles.append(pair_angles)
    
    angles = np.array(angles)
    mean_angles = angles.mean(axis=0)
    
    for l in range(n_layers):
        if l % 4 == 0 or l == n_layers - 1:
            print(f"  L{l:2d}: angle={mean_angles[l]:.4f}")
    
    return mean_angles.tolist()

def main():
    print("=" * 70)
    print("Phase 219-3: 训练vs随机模型的MLP约束传播对照")
    print("=" * 70)
    print(f"执行时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"设备: {DEVICE}")
    
    # ===== 加载训练模型 =====
    print("\n加载训练模型 Qwen2.5-1.5B...")
    model_trained = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    n_layers = model_trained.cfg.n_layers
    
    # ===== 加载随机模型 =====
    print("\n加载随机模型...")
    model_random = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    # 重新初始化所有权重
    for name, param in model_random.named_parameters():
        if param.requires_grad:
            param.data = torch.randn_like(param) * 0.02
    
    # ===== 实验1: 训练vs随机的MLP Ablation =====
    trained_reductions, trained_kl = analyze_mlp_ablation(model_trained, SVA_PAIRS, "训练模型-SVA")
    random_reductions, random_kl = analyze_mlp_ablation(model_random, SVA_PAIRS, "随机模型-SVA")
    
    # ===== 实验2: 约束方向相关性 =====
    num_tense_corr = analyze_constraint_direction_correlation(model_trained, SVA_PAIRS, TENSE_PAIRS)
    
    # ===== 实验3: MLP增量角度 =====
    trained_angles = analyze_mlp_increment_angle(model_trained, SVA_PAIRS, "训练模型")
    random_angles = analyze_mlp_increment_angle(model_random, SVA_PAIRS, "随机模型")
    
    # ===== 综合分析 =====
    print("\n" + "=" * 70)
    print("综合分析")
    print("=" * 70)
    
    # 1. 训练vs随机的MLP Ablation模式
    print("\n--- 训练vs随机: MLP Ablation模式对比 ---")
    print(f"{'Layer':>6} | {'Trained':>10} | {'Random':>10} | {'差异':>10}")
    print("-" * 45)
    for l in range(n_layers):
        if l % 2 == 0 or l == n_layers - 1:
            diff = trained_reductions[l] - random_reductions[l]
            print(f"L{l:4d}  | {trained_reductions[l]:+9.1f}% | {random_reductions[l]:+9.1f}% | {diff:+9.1f}%")
    
    # 2. 振荡模式对比
    trained_sign_changes = sum(1 for i in range(1, len(trained_reductions)) 
                              if (trained_reductions[i] > 0) != (trained_reductions[i-1] > 0))
    random_sign_changes = sum(1 for i in range(1, len(random_reductions)) 
                             if (random_reductions[i] > 0) != (random_reductions[i-1] > 0))
    
    print(f"\n训练模型符号变化次数: {trained_sign_changes}")
    print(f"随机模型符号变化次数: {random_sign_changes}")
    
    if trained_sign_changes > random_sign_changes:
        print("→ 训练模型有更多振荡 → 振荡部分是训练效应")
    elif trained_sign_changes < random_sign_changes:
        print("→ 随机模型有更多振荡 → 训练反而减少了振荡")
    else:
        print("→ 振荡次数相似 → 可能是架构效应")
    
    # 3. 约束独立性
    low_corr_layers = [l for l in range(n_layers) if abs(num_tense_corr[l]) < 0.3]
    high_corr_layers = [l for l in range(n_layers) if abs(num_tense_corr[l]) > 0.5]
    print(f"\n数-时态约束低相关层(|cos|<0.3): {low_corr_layers}")
    print(f"数-时态约束高相关层(|cos|>0.5): {high_corr_layers}")
    if len(low_corr_layers) > len(high_corr_layers):
        print("→ 约束方向大部分正交 → 支持约束独立性")
    else:
        print("→ 约束方向大部分相关 → 不支持约束独立性")
    
    # ===== 保存结果 =====
    output = {
        "experiment": "Phase219-3_Trained_vs_Random_MLP",
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "model": "Qwen2.5-1.5B",
        "n_layers": n_layers,
        "trained_mlp_ablation": trained_reductions,
        "random_mlp_ablation": random_reductions,
        "trained_kl_by_layer": trained_kl,
        "random_kl_by_layer": random_kl,
        "num_tense_correlation": num_tense_corr,
        "trained_mlp_angles": trained_angles,
        "random_mlp_angles": random_angles,
        "trained_sign_changes": trained_sign_changes,
        "random_sign_changes": random_sign_changes,
        "low_corr_layers": low_corr_layers,
        "high_corr_layers": high_corr_layers,
    }
    
    result_file = OUTPUT_DIR / "phase219_3_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    # 释放GPU
    del model_trained
    del model_random
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
