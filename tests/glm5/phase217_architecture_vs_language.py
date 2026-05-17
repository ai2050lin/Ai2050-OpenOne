"""
Phase 217: 架构vs语言分离——随机权重Transformer对照实验

核心问题: 所有已发现的"等价类保持"和"约束方向放大"，
         是训练习得的语言特异性结构，还是架构的数学必然？

实验1B: 约束方向KL敏感性 (最关键实验)
  R_f(l) = KL(P(·|h_l(sg)), P(·|h_l(pl))) 在训练 vs 随机模型中的差异

实验1A: 续写分布等价类宽度
  注入扰动后KL(P(·|h), P(·|h+δ)) 在训练 vs 随机模型中的差异

实验1C: W_U的null space比例

模型: Qwen3 (先做Qwen3确认方法)
"""

import torch
import numpy as np
import json
import time
import sys
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

PERTURBATION_SCALES = [0.01, 0.05, 0.1, 0.5, 1.0]

def compute_kl(p, q, eps=1e-10):
    """对称KL散度"""
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum()
    q = q / q.sum()
    return (0.5 * (p * (p/q).log()).sum() + 0.5 * (q * (q/p).log()).sum()).item()

def get_logits_from_h(model, h):
    """从hidden state计算logits"""
    return h.float() @ model.W_U.float()  # [d_model] @ [d_model, d_vocab] = [d_vocab]

def analyze_W_U_null_space(model):
    """分析W_U的null space结构（使用thin SVD避免OOM）"""
    W_U = model.W_U.float()  # [d_model, d_vocab]
    d_model = W_U.shape[0]
    
    # 使用thin SVD: 只计算min(d_model, d_vocab)个奇异值
    # W_U: [1536, 151936] → thin SVD给出1536个奇异值
    U, S, Vh = torch.linalg.svd(W_U, full_matrices=False)
    # U: [d_model, min], S: [min], Vh: [min, d_vocab]
    
    # 有效秩
    effective_rank = (S > S[0] * 1e-5).sum().item()
    null_space_dim = d_model - effective_rank
    
    # 前20个奇异值
    top_sv = S[:20].tolist()
    
    # 累积能量
    total_energy = (S**2).sum().item()
    cum_energy = np.cumsum((S**2).detach().cpu().numpy()) / total_energy
    rank_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
    rank_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
    rank_99 = int(np.searchsorted(cum_energy, 0.99)) + 1
    
    return {
        "d_model": d_model,
        "d_vocab": W_U.shape[1],
        "effective_rank": effective_rank,
        "null_space_dim": null_space_dim,
        "null_space_frac": null_space_dim / d_model,
        "top_20_sv": top_sv,
        "rank_90": rank_90,
        "rank_95": rank_95,
        "rank_99": rank_99,
        "condition_number": (S[0] / S[min(effective_rank-1, len(S)-1)]).item() if effective_rank > 0 else float('inf'),
    }

def run_experiment_1B(model, model_name):
    """
    实验1B: 约束方向KL敏感性
    R_f(l) = KL(P(·|h_l(sg)), P(·|h_l(pl)))
    """
    print(f"\n{'='*60}")
    print(f"实验1B: 约束方向KL敏感性 [{model_name}]")
    print(f"{'='*60}")
    
    n_layers = model.cfg.n_layers
    results = {}
    
    for pair_idx, (sg_sent, pl_sent) in enumerate(SVA_PAIRS):
        tokens_sg = model.to_tokens(sg_sent, prepend_bos=True)
        tokens_pl = model.to_tokens(pl_sent, prepend_bos=True)
        
        _, cache_sg = model.run_with_cache(tokens_sg, remove_batch_dim=True)
        _, cache_pl = model.run_with_cache(tokens_pl, remove_batch_dim=True)
        
        layer_data = []
        for layer in range(n_layers):
            h_sg = cache_sg["resid_post", layer][-1]  # 最后位置
            h_pl = cache_pl["resid_post", layer][-1]
            
            logits_sg = get_logits_from_h(model, h_sg)
            logits_pl = get_logits_from_h(model, h_pl)
            
            p_sg = torch.softmax(logits_sg, dim=-1)
            p_pl = torch.softmax(logits_pl, dim=-1)
            
            kl = compute_kl(p_sg, p_pl)
            
            # hidden state距离
            h_dist = (h_sg - h_pl).norm().item()
            
            # Δh在W_U row space和null space的分布
            # 用logits差异作为代理: 如果约束改变logits→row space分量大
            delta_h = (h_sg - h_pl).float()
            delta_logits = (logits_sg - logits_pl).float()
            logits_change_norm = delta_logits.norm().item()
            delta_h_norm = delta_h.norm().item()
            
            # 简化: 用logits变化量/h变化量作为"row space投影比"的代理
            # 如果delta_h完全在null space → logits不变 → ratio≈0
            # 如果delta_h完全在row space → logits变化大 → ratio高
            row_space_proxy = logits_change_norm / max(delta_h_norm, 1e-10)
            
            layer_data.append({
                "layer": layer,
                "kl_constraint": kl,
                "h_distance": delta_h_norm,
                "row_space_proxy": row_space_proxy,
            })
        
        results[f"pair_{pair_idx}"] = layer_data
        print(f"  Pair {pair_idx}: '{sg_sent}' vs '{pl_sent}' - "
              f"KL: L0={layer_data[0]['kl_constraint']:.4f}, "
              f"L{n_layers-1}={layer_data[-1]['kl_constraint']:.4f}")
    
    return results

def run_experiment_1A(model, model_name):
    """
    实验1A: 续写分布等价类宽度
    注入随机扰动，测量KL(P(·|h), P(·|h+δ))
    """
    print(f"\n{'='*60}")
    print(f"实验1A: 续写分布等价类宽度 [{model_name}]")
    print(f"{'='*60}")
    
    n_layers = model.cfg.n_layers
    test_sentence = "The cat chases"
    tokens = model.to_tokens(test_sentence, prepend_bos=True)
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    
    results = {}
    
    # 测试5个关键层: 首层、1/4、1/2、3/4、末层
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for layer in key_layers:
        h_clean = cache["resid_post", layer][-1].float()
        logits_clean = get_logits_from_h(model, h_clean)
        p_clean = torch.softmax(logits_clean, dim=-1)
        
        scale_results = []
        for scale in PERTURBATION_SCALES:
            kl_values = []
            for seed in range(10):
                torch.manual_seed(seed + 5000)
                delta = torch.randn_like(h_clean) * scale
                h_perturbed = h_clean + delta
                logits_perturbed = get_logits_from_h(model, h_perturbed)
                p_perturbed = torch.softmax(logits_perturbed, dim=-1)
                kl = compute_kl(p_clean, p_perturbed)
                kl_values.append(kl)
            
            scale_results.append({
                "scale": scale,
                "avg_kl": float(np.mean(kl_values)),
                "std_kl": float(np.std(kl_values)),
            })
        
        results[f"layer_{layer}"] = scale_results
        print(f"  Layer {layer}: " + 
              ", ".join([f"scale={s['scale']:.2f}→KL={s['avg_kl']:.4f}" for s in scale_results[:3]]))
    
    return results

def main():
    print("="*60)
    print("Phase 217: 架构vs语言分离——随机权重对照实验")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("="*60)
    
    all_results = {}
    
    # ============ 加载训练模型并运行所有实验 ============
    print("\n>>> Loading trained Qwen2.5-1.5B...")
    from transformer_lens import HookedTransformer
    model_trained = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    n_layers = model_trained.cfg.n_layers
    print(f"  Layers={n_layers}, d_model={model_trained.cfg.d_model}")
    
    # 实验1B (训练模型)
    all_results["1B_trained"] = run_experiment_1B(model_trained, "trained")
    
    # 实验1A (训练模型)
    all_results["1A_trained"] = run_experiment_1A(model_trained, "trained")
    
    # 实验1C: W_U分析 (训练模型)
    print(f"\n{'='*60}")
    print("实验1C: W_U null space分析 [trained]")
    print(f"{'='*60}")
    all_results["1C_trained"] = analyze_W_U_null_space(model_trained)
    print(f"  有效秩: {all_results['1C_trained']['effective_rank']}")
    print(f"  null space维: {all_results['1C_trained']['null_space_dim']}")
    print(f"  null space比例: {all_results['1C_trained']['null_space_frac']:.4f}")
    print(f"  rank_90/95/99: {all_results['1C_trained']['rank_90']}/{all_results['1C_trained']['rank_95']}/{all_results['1C_trained']['rank_99']}")
    
    # 保存训练模型的结果
    del model_trained
    torch.cuda.empty_cache()
    print("\n  Trained model released. GPU cache cleared.")
    
    # ============ 加载随机模型 ============
    print("\n>>> Loading random Qwen2.5-1.5B...")
    model_random = HookedTransformer.from_pretrained("Qwen/Qwen2.5-1.5B", device=DEVICE)
    # 随机初始化
    for name, param in model_random.named_parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.02)
        elif param.dim() == 1:
            if 'b_' in name or 'bias' in name:
                torch.nn.init.zeros_(param)
            else:
                torch.nn.init.normal_(param, std=0.02)
    print("  Random initialization complete.")
    
    # 实验1B (随机模型)
    all_results["1B_random"] = run_experiment_1B(model_random, "random")
    
    # 实验1A (随机模型)
    all_results["1A_random"] = run_experiment_1A(model_random, "random")
    
    # 实验1C: W_U分析 (随机模型)
    print(f"\n{'='*60}")
    print("实验1C: W_U null space分析 [random]")
    print(f"{'='*60}")
    all_results["1C_random"] = analyze_W_U_null_space(model_random)
    print(f"  有效秩: {all_results['1C_random']['effective_rank']}")
    print(f"  null space维: {all_results['1C_random']['null_space_dim']}")
    print(f"  null space比例: {all_results['1C_random']['null_space_frac']:.4f}")
    
    del model_random
    torch.cuda.empty_cache()
    
    # ============ 保存结果 ============
    output_file = OUTPUT_DIR / "phase217_architecture_vs_language_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    
    # ============ 核心对比分析 ============
    print("\n" + "="*60)
    print("核心判断: 训练效应 vs 架构效应")
    print("="*60)
    
    # 1B: 约束方向KL
    print("\n--- 实验1B: 约束方向KL (R_f(l)) ---")
    for model_key in ["1B_trained", "1B_random"]:
        model_label = model_key.replace("1B_", "")
        layer_kls = {}
        for pair_key in all_results[model_key]:
            for lr in all_results[model_key][pair_key]:
                l = lr["layer"]
                if l not in layer_kls:
                    layer_kls[l] = []
                layer_kls[l].append(lr["kl_constraint"])
        
        key_l = sorted(layer_kls.keys())
        print(f"\n  {model_label} model:")
        for l in [key_l[0], key_l[len(key_l)//2], key_l[-1]]:
            if l in layer_kls:
                print(f"    Layer {l}: avg_KL = {np.mean(layer_kls[l]):.6f}")
        
        # KL增长
        first_kl = np.mean(layer_kls[key_l[0]]) if key_l[0] in layer_kls else 1e-10
        last_kl = np.mean(layer_kls[key_l[-1]]) if key_l[-1] in layer_kls else 1e-10
        growth = last_kl / max(first_kl, 1e-10)
        print(f"    KL增长 (L0→L{key_l[-1]}): {growth:.2f}x")
    
    # 1A: 等价类宽度
    print("\n--- 实验1A: 续写分布等价类宽度 (avg KL per scale) ---")
    for model_key in ["1A_trained", "1A_random"]:
        model_label = model_key.replace("1A_", "")
        print(f"\n  {model_label} model:")
        for layer_key in sorted(all_results[model_key].keys()):
            for sr in all_results[model_key][layer_key]:
                if sr["scale"] in [0.1, 1.0]:
                    print(f"    {layer_key}: scale={sr['scale']:.1f} → KL={sr['avg_kl']:.4f}")
    
    # 1C: W_U对比
    print("\n--- 实验1C: W_U null space结构 ---")
    for model_key in ["1C_trained", "1C_random"]:
        model_label = model_key.replace("1C_", "")
        r = all_results[model_key]
        print(f"  {model_label}: eff_rank={r['effective_rank']}, "
              f"null_frac={r['null_space_frac']:.4f}, "
              f"rank_95={r['rank_95']}")
    
    # 最终判断
    print("\n" + "="*60)
    print("最终判断")
    print("="*60)
    
    trained_kls = []
    random_kls = []
    for pair_key in all_results.get("1B_trained", {}):
        for lr in all_results["1B_trained"][pair_key]:
            trained_kls.append(lr["kl_constraint"])
    for pair_key in all_results.get("1B_random", {}):
        for lr in all_results["1B_random"][pair_key]:
            random_kls.append(lr["kl_constraint"])
    
    if trained_kls and random_kls:
        avg_t = np.mean(trained_kls)
        avg_r = np.mean(random_kls)
        print(f"\n约束方向KL (所有层平均):")
        print(f"  训练模型: {avg_t:.6f}")
        print(f"  随机模型: {avg_r:.6f}")
        print(f"  比率(训练/随机): {avg_t/max(avg_r,1e-10):.2f}")
        
        # 最后5层的对比（关键层）
        n_l = len(all_results["1B_trained"].get("pair_0", []))
        last5_trained = []
        last5_random = []
        for pair_key in all_results["1B_trained"]:
            for lr in all_results["1B_trained"][pair_key][-5:]:
                last5_trained.append(lr["kl_constraint"])
        for pair_key in all_results["1B_random"]:
            for lr in all_results["1B_random"][pair_key][-5:]:
                last5_random.append(lr["kl_constraint"])
        
        if last5_trained and last5_random:
            avg_t_last5 = np.mean(last5_trained)
            avg_r_last5 = np.mean(last5_random)
            print(f"\n最后5层约束方向KL:")
            print(f"  训练模型: {avg_t_last5:.6f}")
            print(f"  随机模型: {avg_r_last5:.6f}")
            print(f"  比率(训练/随机): {avg_t_last5/max(avg_r_last5,1e-10):.2f}")
            
            if avg_t_last5 > avg_r_last5 * 2:
                print("\n  ★ 结论: 约束方向放大是训练习得的语言特异性结构！")
                print("  → 商空间动力学假说获得关键支持")
            elif avg_t_last5 < avg_r_last5 * 0.5:
                print("\n  ★ 结论: 训练实际上抑制了约束方向的KL差异")
                print("  → 需要修正理论（训练使模型更等价类保持）")
            else:
                print("\n  ★ 结论: 约束方向KL差异在训练和随机模型间不显著")
                print("  → 可能是架构效应，需要寻找其他语言特异性结构")
    
    print(f"\nPhase 217 实验完成! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
