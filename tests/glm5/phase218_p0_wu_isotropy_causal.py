"""
Phase 218-P0: W_U各向异性因果必要性验证

这是整个项目当前最重要的实验，决定项目方向：
- 如果W_U各向同性化后语言能力崩溃 → W_U各向异性是语言能力的必要条件
- 如果W_U各向同性化后语言能力基本不变 → W_U各向异性只是副产品

实验设计:
1. 对训练好的Qwen2.5-1.5B的W_U做SVD分解
2. 构造各向同性版本: W_U_iso = U @ diag(S_mean) @ Vh
3. 测试原始W_U和各向同性W_U的语言能力对比

关键测试:
- SVA准确率 (主谓一致) — 用logits直接判断
- 输出分布锐度 (entropy比较)
- 约束KL敏感性 (sg vs pl)

执行时间: 2026-05-17 20:08
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
import json
import time
import sys
from pathlib import Path

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== SVA测试对 =====
SVA_SG_PL_PAIRS = [
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
    ("The king rules", "The kings rule"),
    ("The horse gallops", "The horses gallop"),
    ("The fish swims", "The fish swim"),
    ("The student studies", "The students study"),
    ("The lamp shines", "The lamps shine"),
    ("The clock ticks", "The clocks tick"),
    ("The river flows", "The rivers flow"),
    ("The wind blows", "The winds blow"),
    ("The star glows", "The stars glow"),
    ("The bell rings", "The bells ring"),
]

# ===== 辅助函数 =====

def compute_kl(p, q, eps=1e-10):
    """对称KL散度"""
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum()
    q = q / q.sum()
    return (0.5 * (p * (p/q).log()).sum() + 0.5 * (q * (q/p).log()).sum()).item()

def get_logits_from_h(model, h):
    """从hidden state计算logits"""
    return h.float() @ model.W_U.float()

def compute_entropy(probs, eps=1e-10):
    """计算分布的entropy"""
    p = probs.float() + eps
    p = p / p.sum()
    return -(p * p.log()).sum().item()

def make_isotropic_wu(w_u):
    """
    创建各向同性版本的W_U
    W_U = U @ diag(S) @ Vh
    W_U_iso = U @ diag(S_mean) @ Vh
    保留行空间和列空间，只把奇异值均匀化
    """
    print("  SVD分解W_U...")
    U, S, Vh = torch.linalg.svd(w_u.float(), full_matrices=False)
    
    s_mean = S.mean()
    print(f"  原始奇异值: min={S.min():.3f}, max={S.max():.3f}, mean={s_mean:.3f}")
    print(f"  条件数: {S.max()/S.min():.3f}")
    print(f"  前10个奇异值: {S[:10].tolist()}")
    print(f"  后10个奇异值: {S[-10:].tolist()}")
    
    # 各向同性化: 所有奇异值设为均值
    S_iso = torch.ones_like(S) * s_mean
    w_u_iso = U @ torch.diag(S_iso) @ Vh
    
    # 验证
    U_v, S_v, Vh_v = torch.linalg.svd(w_u_iso.float(), full_matrices=False)
    print(f"  各向同性奇异值: min={S_v.min():.3f}, max={S_v.max():.3f}")
    print(f"  各向同性条件数: {S_v.max()/S_v.min():.3f}")
    
    return w_u_iso, S, S_v

# ===== 实验函数 =====

def experiment_sva_constraint_kl(model):
    """
    实验1: SVA约束KL敏感性
    R_f = KL(P(·|h(sg)), P(·|h(pl))) 在原始 vs 各向同性W_U中的差异
    这是Phase 217的核心指标
    """
    print("\n>>> 实验1: SVA约束KL敏感性")
    
    w_u_original = model.W_U.clone().detach()
    w_u_iso, S_orig, S_iso = make_isotropic_wu(w_u_original)
    
    results_original = []
    results_isotropic = []
    
    for sg, pl in SVA_SG_PL_PAIRS:
        # 获取hidden states
        with torch.no_grad():
            _, cache_sg = model.run_with_cache(sg)
            _, cache_pl = model.run_with_cache(pl)
        
        for layer in range(model.cfg.n_layers):
            h_sg = cache_sg["resid_post", layer][0, -1]  # [d_model]
            h_pl = cache_pl["resid_post", layer][0, -1]  # [d_model]
            
            # 原始W_U的logits和KL
            logits_sg_orig = get_logits_from_h(model, h_sg)
            logits_pl_orig = get_logits_from_h(model, h_pl)
            probs_sg_orig = torch.softmax(logits_sg_orig, dim=-1)
            probs_pl_orig = torch.softmax(logits_pl_orig, dim=-1)
            kl_orig = compute_kl(probs_sg_orig, probs_pl_orig)
            
            # 各向同性W_U的logits和KL
            logits_sg_iso = h_sg.float() @ w_u_iso.float()
            logits_pl_iso = h_pl.float() @ w_u_iso.float()
            probs_sg_iso = torch.softmax(logits_sg_iso, dim=-1)
            probs_pl_iso = torch.softmax(logits_pl_iso, dim=-1)
            kl_iso = compute_kl(probs_sg_iso, probs_pl_iso)
            
            results_original.append({"layer": layer, "pair": (sg, pl), "kl": kl_orig})
            results_isotropic.append({"layer": layer, "pair": (sg, pl), "kl": kl_iso})
    
    # 按层聚合
    n_layers = model.cfg.n_layers
    kl_by_layer_orig = []
    kl_by_layer_iso = []
    
    for l in range(n_layers):
        kl_vals_orig = [r["kl"] for r in results_original if r["layer"] == l]
        kl_vals_iso = [r["kl"] for r in results_isotropic if r["layer"] == l]
        kl_by_layer_orig.append(np.mean(kl_vals_orig))
        kl_by_layer_iso.append(np.mean(kl_vals_iso))
    
    print("\n  Layer | KL_orig  | KL_iso   | Ratio(iso/orig)")
    print("  " + "-" * 55)
    for l in range(n_layers):
        ratio = kl_by_layer_iso[l] / max(kl_by_layer_orig[l], 1e-10)
        if l % 4 == 0 or l == n_layers - 1:
            print(f"  L{l:2d}   | {kl_by_layer_orig[l]:.6f} | {kl_by_layer_iso[l]:.6f} | {ratio:.4f}")
    
    # 有效层范围(L0-L20)的统计
    eff_orig = np.mean(kl_by_layer_orig[:21])
    eff_iso = np.mean(kl_by_layer_iso[:21])
    ratio_eff = eff_iso / max(eff_orig, 1e-10)
    
    print(f"\n  L0-L20均值: orig={eff_orig:.6f}, iso={eff_iso:.6f}, ratio={ratio_eff:.4f}")
    
    return {
        "kl_by_layer_original": kl_by_layer_orig,
        "kl_by_layer_isotropic": kl_by_layer_iso,
        "effective_range_ratio": ratio_eff,
        "results_original": results_original,
        "results_isotropic": results_isotropic,
    }

def experiment_output_entropy(model):
    """
    实验2: 输出分布锐度
    各向同性化是否削弱模型的"锐化"能力(entropy增加)
    """
    print("\n>>> 实验2: 输出分布锐度(entropy)")
    
    w_u_original = model.W_U.clone().detach()
    w_u_iso, _, _ = make_isotropic_wu(w_u_original)
    
    entropy_orig = []
    entropy_iso = []
    
    for sg, pl in SVA_SG_PL_PAIRS[:10]:  # 取前10对
        with torch.no_grad():
            _, cache = model.run_with_cache(sg)
        
        for layer in [0, 7, 14, 20, 27]:
            h = cache["resid_post", layer][0, -1]
            
            # 原始
            logits_orig = get_logits_from_h(model, h)
            probs_orig = torch.softmax(logits_orig, dim=-1)
            ent_o = compute_entropy(probs_orig)
            
            # 各向同性
            logits_iso = h.float() @ w_u_iso.float()
            probs_iso = torch.softmax(logits_iso, dim=-1)
            ent_i = compute_entropy(probs_iso)
            
            entropy_orig.append({"layer": layer, "entropy": ent_o})
            entropy_iso.append({"layer": layer, "entropy": ent_i})
    
    # 按层聚合
    layers_reported = [0, 7, 14, 20, 27]
    print("\n  Layer | Ent_orig  | Ent_iso   | Delta")
    print("  " + "-" * 50)
    for l in layers_reported:
        vals_o = [r["entropy"] for r in entropy_orig if r["layer"] == l]
        vals_i = [r["entropy"] for r in entropy_iso if r["layer"] == l]
        mean_o = np.mean(vals_o)
        mean_i = np.mean(vals_i)
        delta = mean_i - mean_o
        print(f"  L{l:2d}   | {mean_o:.4f}   | {mean_i:.4f}   | {delta:+.4f}")
    
    return {
        "entropy_original": entropy_orig,
        "entropy_isotropic": entropy_iso,
    }

def experiment_top1_accuracy(model):
    """
    实验3: Top-1预测准确性
    各向同性化是否改变模型的top-1预测
    """
    print("\n>>> 实验3: Top-1预测变化")
    
    w_u_original = model.W_U.clone().detach()
    w_u_iso, _, _ = make_isotropic_wu(w_u_original)
    
    changes_by_layer = []
    
    test_sentences = [sg for sg, pl in SVA_SG_PL_PAIRS[:10]]
    
    for layer in [0, 7, 14, 20, 27]:
        same_count = 0
        total_count = 0
        
        for sent in test_sentences:
            with torch.no_grad():
                _, cache = model.run_with_cache(sent)
            
            h = cache["resid_post", layer][0, -1]
            
            # 原始top-1
            logits_orig = get_logits_from_h(model, h)
            top1_orig = logits_orig.argmax().item()
            
            # 各向同性top-1
            logits_iso = h.float() @ w_u_iso.float()
            top1_iso = logits_iso.argmax().item()
            
            if top1_orig == top1_iso:
                same_count += 1
            total_count += 1
        
        agreement = same_count / total_count
        changes_by_layer.append({"layer": layer, "agreement": agreement, "same": same_count, "total": total_count})
        print(f"  L{layer:2d}: top-1一致率={agreement:.3f} ({same_count}/{total_count})")
    
    return {"changes_by_layer": changes_by_layer}

def experiment_logit_magnitude(model):
    """
    实验4: Logit幅度变化
    各向同性化对logit分布幅度的影响
    """
    print("\n>>> 实验4: Logit幅度变化")
    
    w_u_original = model.W_U.clone().detach()
    w_u_iso, _, _ = make_isotropic_wu(w_u_original)
    
    results = []
    
    for sg, _ in SVA_SG_PL_PAIRS[:5]:
        with torch.no_grad():
            _, cache = model.run_with_cache(sg)
        
        for layer in [0, 7, 14, 20, 27]:
            h = cache["resid_post", layer][0, -1]
            
            logits_orig = get_logits_from_h(model, h)
            logits_iso = h.float() @ w_u_iso.float()
            
            results.append({
                "layer": layer,
                "sentence": sg,
                "logit_max_orig": logits_orig.max().item(),
                "logit_max_iso": logits_iso.max().item(),
                "logit_std_orig": logits_orig.std().item(),
                "logit_std_iso": logits_iso.std().item(),
                "logit_range_orig": (logits_orig.max() - logits_orig.min()).item(),
                "logit_range_iso": (logits_iso.max() - logits_iso.min()).item(),
            })
    
    # 按层聚合
    layers_reported = [0, 7, 14, 20, 27]
    print("\n  Layer | Range_orig | Range_iso  | Std_orig  | Std_iso")
    print("  " + "-" * 60)
    for l in layers_reported:
        vals = [r for r in results if r["layer"] == l]
        range_o = np.mean([r["logit_range_orig"] for r in vals])
        range_i = np.mean([r["logit_range_iso"] for r in vals])
        std_o = np.mean([r["logit_std_orig"] for r in vals])
        std_i = np.mean([r["logit_std_iso"] for r in vals])
        print(f"  L{l:2d}   | {range_o:9.2f}  | {range_i:9.2f}  | {std_o:9.2f} | {std_i:9.2f}")
    
    return {"logit_magnitude": results}

def experiment_null_space_effect(model):
    """
    实验5: Null space效应
    各向同性化消除了null space → 扰动应不再被null space吸收
    """
    print("\n>>> 实验5: Null space效应(扰动敏感性)")
    
    w_u_original = model.W_U.clone().detach()
    w_u_iso, _, _ = make_isotropic_wu(w_u_original)
    
    scales = [0.01, 0.05, 0.1, 0.5, 1.0]
    n_perturb = 10  # 每个scale的扰动次数
    
    results_orig = []
    results_iso = []
    
    for sent in ["The cat chases"]:
        with torch.no_grad():
            _, cache = model.run_with_cache(sent)
        
        for layer in [14, 20]:  # 中间层和后层
            h = cache["resid_post", layer][0, -1]
            
            # 原始logits
            logits_orig = get_logits_from_h(model, h)
            probs_orig = torch.softmax(logits_orig, dim=-1)
            
            # 各向同性logits
            logits_iso_base = h.float() @ w_u_iso.float()
            probs_iso_base = torch.softmax(logits_iso_base, dim=-1)
            
            for scale in scales:
                kl_vals_orig = []
                kl_vals_iso = []
                
                for _ in range(n_perturb):
                    delta = torch.randn_like(h) * scale
                    h_perturbed = h + delta
                    
                    # 原始W_U
                    logits_p_o = get_logits_from_h(model, h_perturbed)
                    probs_p_o = torch.softmax(logits_p_o, dim=-1)
                    kl_o = compute_kl(probs_orig, probs_p_o)
                    kl_vals_orig.append(kl_o)
                    
                    # 各向同性W_U
                    logits_p_i = h_perturbed.float() @ w_u_iso.float()
                    probs_p_i = torch.softmax(logits_p_i, dim=-1)
                    kl_i = compute_kl(probs_iso_base, probs_p_i)
                    kl_vals_iso.append(kl_i)
                
                results_orig.append({
                    "layer": layer, "scale": scale,
                    "kl_mean": np.mean(kl_vals_orig), "kl_std": np.std(kl_vals_orig)
                })
                results_iso.append({
                    "layer": layer, "scale": scale,
                    "kl_mean": np.mean(kl_vals_iso), "kl_std": np.std(kl_vals_iso)
                })
    
    print("\n  Layer | Scale | KL_orig   | KL_iso    | Ratio(iso/orig)")
    print("  " + "-" * 60)
    for r_o, r_i in zip(results_orig, results_iso):
        ratio = r_i["kl_mean"] / max(r_o["kl_mean"], 1e-10)
        print(f"  L{r_o['layer']:2d}   | {r_o['scale']:5.2f} | {r_o['kl_mean']:9.6f} | {r_i['kl_mean']:9.6f} | {ratio:.4f}")
    
    return {"perturbation_orig": results_orig, "perturbation_iso": results_iso}

# ===== 主函数 =====

def main():
    print("=" * 70)
    print("Phase 218-P0: W_U各向异性因果必要性验证")
    print("=" * 70)
    print(f"执行时间: {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"设备: {DEVICE}")
    
    # 加载模型
    print("\n加载Qwen2.5-1.5B模型...")
    model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    print(f"模型层数: {model.cfg.n_layers}")
    print(f"d_model: {model.cfg.d_model}")
    print(f"d_vocab: {model.cfg.d_vocab}")
    print(f"W_U形状: {model.W_U.shape}")
    
    # ===== 执行5个实验 =====
    all_results = {}
    
    # 实验1: SVA约束KL敏感性 (最重要)
    all_results["sva_kl"] = experiment_sva_constraint_kl(model)
    
    # 实验2: 输出分布锐度
    all_results["entropy"] = experiment_output_entropy(model)
    
    # 实验3: Top-1预测变化
    all_results["top1"] = experiment_top1_accuracy(model)
    
    # 实验4: Logit幅度变化
    all_results["logit_mag"] = experiment_logit_magnitude(model)
    
    # 实验5: Null space效应
    all_results["null_space"] = experiment_null_space_effect(model)
    
    # ===== 综合判决 =====
    print("\n" + "=" * 70)
    print("综合判决")
    print("=" * 70)
    
    # 关键指标
    kl_ratio = all_results["sva_kl"]["effective_range_ratio"]
    top1_data = all_results["top1"]["changes_by_layer"]
    top1_l20 = [r for r in top1_data if r["layer"] == 20][0]["agreement"]
    
    print(f"\n关键指标:")
    print(f"  1. SVA KL比率(iso/orig, L0-L20): {kl_ratio:.4f}")
    print(f"     → <0.1: 各向同性大幅削弱约束传播能力")
    print(f"     → 0.1-0.5: 部分削弱")
    print(f"     → >0.5: 基本不削弱")
    print(f"  2. Top-1一致率(L20): {top1_l20:.3f}")
    print(f"     → <0.5: 各向同性大幅改变预测")
    print(f"     → 0.5-0.8: 中等改变")
    print(f"     → >0.8: 基本不变")
    
    # 判决逻辑
    kl_severe = kl_ratio < 0.1
    kl_moderate = 0.1 <= kl_ratio < 0.5
    top1_severe = top1_l20 < 0.5
    top1_moderate = 0.5 <= top1_l20 < 0.8
    
    print(f"\n=== 判决结果 ===")
    if kl_severe and top1_severe:
        conclusion = "情况A: W_U各向异性是语言能力的必要条件"
        print(f"  {conclusion}")
        print(f"  → 各向同性化严重削弱约束传播(KL比率={kl_ratio:.4f})")
        print(f"  → 各向同性化严重改变预测(一致率={top1_l20:.3f})")
        print(f"  → 理论方向: 研究W_U各向异性如何支持语言能力")
    elif kl_moderate or top1_moderate:
        conclusion = "情况C: W_U各向异性对某些能力必要"
        print(f"  {conclusion}")
        print(f"  → 各向同性化部分削弱(KL比率={kl_ratio:.4f}, 一致率={top1_l20:.3f})")
        print(f"  → 需要更精细的分析: 哪些能力依赖各向异性？")
    else:
        conclusion = "情况B: W_U各向异性是副产品"
        print(f"  {conclusion}")
        print(f"  → 各向同性化影响很小(KL比率={kl_ratio:.4f}, 一致率={top1_l20:.3f})")
        print(f"  → 需要完全重新寻找surviving信号")
    
    # ===== 保存结果 =====
    # 转换numpy为python类型
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    output = {
        "experiment": "Phase218-P0_WU_Isotropy_Causal",
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "model": "Qwen2.5-1.5B",
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "d_vocab": model.cfg.d_vocab,
        "conclusion": conclusion,
        "key_metrics": {
            "kl_ratio_L0_L20": float(kl_ratio),
            "top1_agreement_L20": float(top1_l20),
        },
        "detailed_results": convert(all_results),
    }
    
    result_file = OUTPUT_DIR / "phase218_p0_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    return output

if __name__ == "__main__":
    results = main()
