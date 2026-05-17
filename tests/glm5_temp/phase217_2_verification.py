"""Phase 217-2: 验证实验——确认等价类宽度发现的真实性

问题1: 训练模型L27的KL=0.00是否真实？
  → 可能是LayerNorm导致输出被"锁定"
  → 检查L27的hidden state norm和logits分布

问题2: 随机模型的低KL是否因为输出是均匀分布？
  → 检查随机模型输出的entropy
  → 与均匀分布对比

问题3: 有效null space的精确定义
  → 基于KL阈值定义"对输出影响<ε的方向比例"
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
    print("Phase 217-2: 等价类宽度验证实验")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    # === 加载训练模型 ===
    print("\n>>> Loading trained model...")
    model_trained = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    n_layers = model_trained.cfg.n_layers
    
    test_sentence = "The cat chases"
    tokens = model_trained.to_tokens(test_sentence, prepend_bos=True)
    _, cache = model_trained.run_with_cache(tokens, remove_batch_dim=True)
    
    # === 实验1: 训练模型每层的hidden state norm和logits特征 ===
    print("\n=== 训练模型: 逐层hidden state和logits分析 ===")
    trained_layer_analysis = []
    for layer in range(n_layers):
        h = cache["resid_post", layer][-1].float()  # 最后位置
        logits = h @ model_trained.W_U.float()
        p = torch.softmax(logits, dim=-1)
        
        # hidden state统计
        h_norm = h.norm().item()
        h_std = h.std().item()
        
        # logits统计
        logits_norm = logits.norm().item()
        logits_max = logits.max().item()
        logits_min = logits.min().item()
        logits_std = logits.std().item()
        
        # 概率分布统计
        p_max = p.max().item()
        entropy = -(p * p.log()).sum().item()
        max_entropy = np.log(p.shape[0])  # 均匀分布的熵
        
        # 扰动测试: scale=0.5
        torch.manual_seed(42)
        delta = torch.randn_like(h) * 0.5
        h_perturbed = h + delta
        logits_perturbed = h_perturbed @ model_trained.W_U.float()
        p_perturbed = torch.softmax(logits_perturbed, dim=-1)
        kl = (0.5 * (p * (p/p_perturbed).log()).sum() + 0.5 * (p_perturbed * (p_perturbed/p).log()).sum()).item()
        
        # 扰动对logits的影响
        delta_logits = logits_perturbed - logits
        delta_logits_norm = delta_logits.norm().item()
        delta_logits_relative = delta_logits_norm / max(logits_norm, 1e-10)
        
        # delta_h的范数
        delta_h_norm = delta.norm().item()
        delta_h_relative = delta_h_norm / max(h_norm, 1e-10)
        
        trained_layer_analysis.append({
            "layer": layer,
            "h_norm": h_norm,
            "h_std": h_std,
            "logits_norm": logits_norm,
            "logits_max": logits_max,
            "logits_std": logits_std,
            "p_max": p_max,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "entropy_ratio": entropy / max_entropy,
            "kl_scale05": kl,
            "delta_h_relative": delta_h_relative,
            "delta_logits_relative": delta_logits_relative,
        })
        
        if layer in [0, 7, 14, 21, 27]:
            print(f"  L{layer:2d}: h_norm={h_norm:.2f}, logits_norm={logits_norm:.1f}, "
                  f"p_max={p_max:.4f}, entropy/MaxEnt={entropy/max_entropy:.4f}, "
                  f"KL(scale=0.5)={kl:.4f}, Δlogits_rel={delta_logits_relative:.4f}")
    
    results["trained_layer_analysis"] = trained_layer_analysis
    
    del model_trained
    torch.cuda.empty_cache()
    
    # === 加载随机模型 ===
    print("\n>>> Loading random model...")
    model_random = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    for name, param in model_random.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.02)
        elif param.dim() == 1:
            if 'b_' in name or 'bias' in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, std=0.02)
    
    _, cache_r = model_random.run_with_cache(tokens, remove_batch_dim=True)
    
    # === 实验2: 随机模型同样的分析 ===
    print("\n=== 随机模型: 逐层hidden state和logits分析 ===")
    random_layer_analysis = []
    for layer in range(n_layers):
        h = cache_r["resid_post", layer][-1].float()
        logits = h @ model_random.W_U.float()
        p = torch.softmax(logits, dim=-1)
        
        h_norm = h.norm().item()
        logits_norm = logits.norm().item()
        logits_max = logits.max().item()
        logits_std = logits.std().item()
        p_max = p.max().item()
        entropy = -(p * p.log()).sum().item()
        max_entropy = np.log(p.shape[0])
        
        torch.manual_seed(42)
        delta = torch.randn_like(h) * 0.5
        h_perturbed = h + delta
        logits_perturbed = h_perturbed @ model_random.W_U.float()
        p_perturbed = torch.softmax(logits_perturbed, dim=-1)
        kl = (0.5 * (p * (p/p_perturbed).log()).sum() + 0.5 * (p_perturbed * (p_perturbed/p).log()).sum()).item()
        
        delta_logits = logits_perturbed - logits
        delta_logits_norm = delta_logits.norm().item()
        delta_logits_relative = delta_logits_norm / max(logits_norm, 1e-10)
        delta_h_norm = delta.norm().item()
        delta_h_relative = delta_h_norm / max(h_norm, 1e-10)
        
        random_layer_analysis.append({
            "layer": layer,
            "h_norm": h_norm,
            "logits_norm": logits_norm,
            "logits_max": logits_max,
            "logits_std": logits_std,
            "p_max": p_max,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "entropy_ratio": entropy / max_entropy,
            "kl_scale05": kl,
            "delta_h_relative": delta_h_relative,
            "delta_logits_relative": delta_logits_relative,
        })
        
        if layer in [0, 7, 14, 21, 27]:
            print(f"  L{layer:2d}: h_norm={h_norm:.2f}, logits_norm={logits_norm:.1f}, "
                  f"p_max={p_max:.6f}, entropy/MaxEnt={entropy/max_entropy:.4f}, "
                  f"KL(scale=0.5)={kl:.4f}, Δlogits_rel={delta_logits_relative:.4f}")
    
    results["random_layer_analysis"] = random_layer_analysis
    
    # === 核心对比 ===
    print("\n" + "="*60)
    print("核心对比: 训练 vs 随机")
    print("="*60)
    
    print(f"\n{'Layer':>5} | {'T_EntRatio':>10} | {'R_EntRatio':>10} | {'T_KL05':>8} | {'R_KL05':>8} | {'T_dLogRel':>9} | {'R_dLogRel':>9}")
    print("-"*75)
    for l in range(n_layers):
        t = trained_layer_analysis[l]
        r = random_layer_analysis[l]
        if l in [0, 7, 14, 21, 27]:
            print(f"L{l:2d}   | {t['entropy_ratio']:10.4f} | {r['entropy_ratio']:10.4f} | "
                  f"{t['kl_scale05']:8.4f} | {r['kl_scale05']:8.4f} | "
                  f"{t['delta_logits_relative']:9.4f} | {r['delta_logits_relative']:9.4f}")
    
    # 保存
    output_file = OUTPUT_DIR / "phase217_2_verification_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    
    print(f"\nPhase 217-2 完成! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
