"""Phase 217-3: 控制h_norm的等价类分析

关键问题: "等价类变厚"是尺度效应还是结构性效应？
方法: 对每层h_l归一化后再注入扰动，消除h_norm增长的影响

如果归一化后KL仍随层下降 → 存在结构性扰动抑制
如果归一化后KL不再随层下降 → 纯粹是尺度效应
"""
import torch
import numpy as np
import json
import time
from pathlib import Path
from transformer_lens import HookedTransformer

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
DEVICE = "cuda"

def main():
    print("="*60)
    print("Phase 217-3: 控制h_norm的等价类分析")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # === 加载训练模型 ===
    print("\n>>> Loading trained model...")
    model_t = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    n_layers = model_t.cfg.n_layers
    
    test_sentence = "The cat chases"
    tokens = model_t.to_tokens(test_sentence, prepend_bos=True)
    _, cache_t = model_t.run_with_cache(tokens, remove_batch_dim=True)
    
    # === 实验: 归一化后的扰动敏感性 ===
    print("\n=== 训练模型: 原始 vs 归一化 ===")
    trained_results = []
    
    for layer in range(n_layers):
        h = cache_t["resid_post", layer][-1].float()
        h_norm = h.norm().item()
        h_normed = h / h_norm  # 归一化
        
        # 原始h的logits
        logits_orig = h @ model_t.W_U.float()
        p_orig = torch.softmax(logits_orig, dim=-1)
        
        # 归一化h的logits
        logits_normed = h_normed @ model_t.W_U.float()
        p_normed = torch.softmax(logits_normed, dim=-1)
        
        # 扰动测试: 在原始h上注入scale=1.0的扰动
        torch.manual_seed(42)
        delta = torch.randn_like(h) * 1.0
        
        # 原始h+扰动
        h_perturbed = h + delta
        logits_perturbed = h_perturbed @ model_t.W_U.float()
        p_perturbed = torch.softmax(logits_perturbed, dim=-1)
        
        # 归一化h+同方向扰动（但scale相对于归一化后的h）
        delta_normed = delta / h_norm  # 归一化扰动
        h_normed_perturbed = h_normed + delta_normed
        logits_normed_perturbed = h_normed_perturbed @ model_t.W_U.float()
        p_normed_perturbed = torch.softmax(logits_normed_perturbed, dim=-1)
        
        # 计算KL
        eps = 1e-10
        def sym_kl(p1, p2):
            p1 = p1 + eps; p2 = p2 + eps
            p1 = p1 / p1.sum(); p2 = p2 / p2.sum()
            return (0.5 * (p1 * (p1/p2).log()).sum() + 0.5 * (p2 * (p2/p1).log()).sum()).item()
        
        kl_orig = sym_kl(p_orig, p_perturbed)
        kl_normed = sym_kl(p_normed, p_normed_perturbed)
        
        trained_results.append({
            "layer": layer,
            "h_norm": h_norm,
            "logits_norm": logits_orig.norm().item(),
            "kl_original": kl_orig,
            "kl_normalized": kl_normed,
        })
        
        if layer in [0, 7, 14, 21, 27]:
            print(f"  L{layer:2d}: h_norm={h_norm:.1f}, logits_norm={logits_orig.norm().item():.1f}, "
                  f"KL_orig={kl_orig:.4f}, KL_normed={kl_normed:.4f}")
    
    results["trained"] = trained_results
    
    del model_t
    torch.cuda.empty_cache()
    
    # === 随机模型 ===
    print("\n>>> Loading random model...")
    model_r = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    for name, param in model_r.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.02)
        elif param.dim() == 1:
            if 'b_' in name or 'bias' in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, std=0.02)
    
    _, cache_r = model_r.run_with_cache(tokens, remove_batch_dim=True)
    
    print("\n=== 随机模型: 原始 vs 归一化 ===")
    random_results = []
    
    for layer in range(n_layers):
        h = cache_r["resid_post", layer][-1].float()
        h_norm = h.norm().item()
        h_normed = h / h_norm
        
        logits_orig = h @ model_r.W_U.float()
        p_orig = torch.softmax(logits_orig, dim=-1)
        
        logits_normed = h_normed @ model_r.W_U.float()
        p_normed = torch.softmax(logits_normed, dim=-1)
        
        torch.manual_seed(42)
        delta = torch.randn_like(h) * 1.0
        
        h_perturbed = h + delta
        logits_perturbed = h_perturbed @ model_r.W_U.float()
        p_perturbed = torch.softmax(logits_perturbed, dim=-1)
        
        delta_normed = delta / h_norm
        h_normed_perturbed = h_normed + delta_normed
        logits_normed_perturbed = h_normed_perturbed @ model_r.W_U.float()
        p_normed_perturbed = torch.softmax(logits_normed_perturbed, dim=-1)
        
        eps = 1e-10
        def sym_kl(p1, p2):
            p1 = p1 + eps; p2 = p2 + eps
            p1 = p1 / p1.sum(); p2 = p2 / p2.sum()
            return (0.5 * (p1 * (p1/p2).log()).sum() + 0.5 * (p2 * (p2/p1).log()).sum()).item()
        
        kl_orig = sym_kl(p_orig, p_perturbed)
        kl_normed = sym_kl(p_normed, p_normed_perturbed)
        
        random_results.append({
            "layer": layer,
            "h_norm": h_norm,
            "logits_norm": logits_orig.norm().item(),
            "kl_original": kl_orig,
            "kl_normalized": kl_normed,
        })
        
        if layer in [0, 7, 14, 21, 27]:
            print(f"  L{layer:2d}: h_norm={h_norm:.1f}, logits_norm={logits_orig.norm().item():.1f}, "
                  f"KL_orig={kl_orig:.4f}, KL_normed={kl_normed:.4f}")
    
    results["random"] = random_results
    
    # === 核心对比 ===
    print("\n" + "="*60)
    print("核心对比: 归一化后的KL变化")
    print("="*60)
    print(f"\n{'Layer':>5} | {'T_KL_orig':>9} | {'T_KL_norm':>9} | {'R_KL_orig':>9} | {'R_KL_norm':>9}")
    print("-"*60)
    for l in range(n_layers):
        t = trained_results[l]
        r = random_results[l]
        if l in [0, 4, 7, 10, 14, 18, 21, 24, 27]:
            print(f"L{l:2d}   | {t['kl_original']:9.4f} | {t['kl_normalized']:9.4f} | "
                  f"{r['kl_original']:9.4f} | {r['kl_normalized']:9.4f}")
    
    # 关键判断
    print("\n" + "="*60)
    print("关键判断")
    print("="*60)
    
    # 训练模型归一化后KL的趋势
    trained_normed_kls = [r["kl_normalized"] for r in trained_results]
    if trained_normed_kls[0] > trained_normed_kls[-1]:
        print(f"\n训练模型归一化KL: L0={trained_normed_kls[0]:.4f} → L27={trained_normed_kls[-1]:.4f}")
        print("  → 归一化后KL仍随层下降 → 存在结构性扰动抑制！")
    else:
        print(f"\n训练模型归一化KL: L0={trained_normed_kls[0]:.4f} → L27={trained_normed_kls[-1]:.4f}")
        print("  → 归一化后KL不再随层下降 → '等价类变厚'完全是尺度效应")
    
    random_normed_kls = [r["kl_normalized"] for r in random_results]
    print(f"随机模型归一化KL: L0={random_normed_kls[0]:.4f} → L27={random_normed_kls[-1]:.4f}")
    
    # 保存
    output_file = OUTPUT_DIR / "phase217_3_normalized_kl_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    
    print(f"\nPhase 217-3 完成! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
