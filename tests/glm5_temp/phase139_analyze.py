#!/usr/bin/env python3
"""Phase 139 结果分析脚本"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

for model_name, fn in [
    ("Qwen3", "tests/glm5_temp/phase139_qwen3_jacobian_geometry_20260512_1350.json"),
    ("GLM4", "tests/glm5_temp/phase139_glm4_jacobian_geometry_20260512_1410.json"),
]:
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Exp A 关键结果
    print("\n--- Exp A: 传播比(非归一化) ---")
    expA = data["expA"]
    for perturb_key in sorted(expA.get("perturbation_layers", {}).keys()):
        pdata = expA["perturbation_layers"][perturb_key]
        if "error" in pdata:
            continue
        eff_rank = pdata["mean_effective_rank_ratio"]
        # 最后一个观察层的传播比
        last_obs = None
        for obs_key in sorted(pdata.get("propagation_ratios_summary", {}).keys()):
            last_obs = obs_key
        if last_obs:
            last_stats = pdata["propagation_ratios_summary"][last_obs]
            print(f"  {perturb_key} -> {last_obs}: mean_prop={last_stats['mean']:.1f}, "
                  f"pct>1={last_stats['pct_above_1']:.2f}, eff_rank_ratio={eff_rank:.3f}")
    
    # Exp B 关键结果
    print("\n--- Exp B: 语义vs随机扰动 ---")
    expB = data["expB"]
    for op_name in ["negation_analysis", "tense_analysis"]:
        print(f"\n  {op_name}:")
        for layer in sorted(expB.get(op_name, {}).keys()):
            d = expB[op_name][layer]
            sem = d["semantic_logit_shift_mean"]
            rand = d["random_logit_shift_mean"]
            ratio = d["shift_ratio"]
            print(f"    {layer}: sem={sem:.2f}, rand={rand:.2f}, ratio(rand/sem)={ratio:.2f}")
    
    # Exp C 关键结果
    print("\n--- Exp C: 归一化传播比 & 方向保持 ---")
    expC = data["expC"]
    for perturb_key in sorted(expC.get("layer_results", {}).keys()):
        layer_data = expC["layer_results"][perturb_key]
        print(f"\n  扰动@{perturb_key}:")
        for obs_key in sorted(layer_data.keys()):
            s = layer_data[obs_key]
            nr = s["norm_ratio_mean"]
            dp = s["direction_preserve_mean"]
            te = s["top10pct_energy_ratio_mean"]
            print(f"    -> {obs_key}: norm_ratio={nr:.3f}, dir_preserve={dp:.3f}, top10%_energy={te:.3f}")
