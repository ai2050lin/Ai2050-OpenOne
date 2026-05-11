"""Phase 126 三模型汇总分析"""
import json
import numpy as np

models = ["qwen3", "deepseek7b", "glm4"]

for model_name in models:
    path = f"tests/glm5_temp/phase126_{model_name}_circuit_topology.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    info = data.get("model_info", {})
    print(f"  Class: {info.get('class')}, L={info.get('n_layers')}, "
          f"d={info.get('d_model')}, n_heads={info.get('n_heads')}, "
          f"head_dim={info.get('head_dim')}")
    
    # Exp 1: Head功能分化
    exp1 = data.get("exp1", {})
    print(f"\n  --- Exp 1: Head功能分化 ---")
    target_layers = exp1.get("target_layers", [])
    for li in target_layers:
        ldata = exp1.get(f"L{li}", {})
        if ldata:
            print(f"    L{li}: n_high_corr={ldata.get('n_high_corr_pairs', 'N/A')}, "
                  f"clusters={ldata.get('n_clusters', 'N/A')}, "
                  f"sizes={ldata.get('cluster_sizes_top5', 'N/A')}, "
                  f"top_selective={ldata.get('top5_selective_heads', [{}])[0].get('best_category', 'N/A') if ldata.get('top5_selective_heads') else 'N/A'}")
    
    # Exp 2: Head协同激活
    exp2 = data.get("exp2", {})
    print(f"\n  --- Exp 2: Head协同激活 ---")
    for li in exp2.get("target_layers", []):
        ldata = exp2.get(f"L{li}", {})
        if ldata:
            print(f"    L{li}: mean|r|={ldata.get('mean_abs_corr', 'N/A'):.3f}, "
                  f"frac>0.5={ldata.get('frac_gt05', 'N/A'):.3f}, "
                  f"clusters={ldata.get('n_clusters', 'N/A')}, "
                  f"sizes={ldata.get('cluster_sizes_top5', 'N/A')}")
    
    # Exp 3: 条件轨迹分叉
    exp3 = data.get("exp3", {})
    print(f"\n  --- Exp 3: 条件轨迹分叉 ---")
    layer_sens = exp3.get("layer_context_sensitivity", {})
    for li_str, vals in sorted(layer_sens.items()):
        print(f"    L{li_str}: sensitivity={vals.get('mean', 'N/A'):.3f} ± {vals.get('std', 'N/A'):.3f}")
    
    top_heads = exp3.get("top_context_sensitive_heads", [])[:5]
    if top_heads:
        print(f"    Top context-sensitive heads: {[(h['layer'], h['head'], round(h['mean_js'], 3)) for h in top_heads]}")
    
    # Exp 5: 回路消融 vs 方向消融
    exp5 = data.get("exp5", {})
    print(f"\n  --- Exp 5: 回路消融 vs 方向消融 ---")
    
    dir_abl = exp5.get("direction_ablation", {})
    print(f"    Direction (PCA) ablation:")
    for k, v in sorted(dir_abl.items()):
        print(f"      top-{k}: KL={v.get('mean_kl', 'N/A'):.4f}")
    
    circ_abl = exp5.get("circuit_ablation", {})
    print(f"    Circuit (head) ablation:")
    for k, v in sorted(circ_abl.items()):
        print(f"      k={k}: random={v.get('random', {}).get('mean_kl', 'N/A'):.4f}, "
              f"same_layer={v.get('same_layer', {}).get('mean_kl', 'N/A'):.4f}, "
              f"cross_layer={v.get('cross_layer', {}).get('mean_kl', 'N/A'):.4f}")
    
    mlp_abl = exp5.get("mlp_ablation", {})
    print(f"    MLP ablation:")
    for k, v in sorted(mlp_abl.items()):
        print(f"      k={k}: KL={v.get('mean_kl', 'N/A'):.4f}")
    
    # 关键比较: MLP vs Attention vs PCA
    print(f"\n  === 核心比较 (k=5) ===")
    mlp5 = mlp_abl.get("5", {}).get("mean_kl", 0)
    attn5 = circ_abl.get("5", {}).get("cross_layer", {}).get("mean_kl", 0)
    pca5 = dir_abl.get("5", {}).get("mean_kl", 0)
    print(f"    MLP k=5: {mlp5:.4f}")
    print(f"    Attention cross-layer k=5: {attn5:.4f}")
    print(f"    PCA top-5 dirs: {pca5:.4f}")
    if mlp5 > 0 and attn5 > 0:
        print(f"    MLP/Attn ratio: {mlp5/attn5:.1f}x")
    if mlp5 > 0 and pca5 > 0:
        print(f"    MLP/PCA ratio: {mlp5/pca5:.1f}x")
