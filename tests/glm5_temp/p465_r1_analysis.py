"""Phase 465 R1 结果提取与分析"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

results = {}
for model in ['qwen3', 'deepseek7b', 'glm4']:
    path = f'results/glm5/phase465_{model}_r1.json'
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            results[model] = json.load(f)

print("=" * 80)
print("Phase 465 R1 关键结果对比")
print("=" * 80)

# ===== Exp1: 自然流形兼容性 =====
print("\n### Exp1: 自然流形兼容性 - 关键指标对比 ###")
print(f"\n{'模型':<12} {'层':<5} {'类别':<10} {'beta':<5} {'norm_ratio':<12} {'KL散度':<10} {'top5_overlap':<14} {'selectivity':<12}")
print("-" * 80)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    if model not in results or 'exp1_manifold' not in results[model]:
        continue
    exp1 = results[model]['exp1_manifold']
    for layer_key in sorted(exp1.keys()):
        layer_data = exp1[layer_key]
        for cat in ['fruit', 'animal', 'vehicle']:
            if cat not in layer_data:
                continue
            for beta_key in ['beta_5.0', 'beta_10.0']:
                if beta_key not in layer_data[cat]:
                    continue
                d = layer_data[cat][beta_key]
                print(f"{model:<12} {layer_key:<5} {cat:<10} {beta_key[5:]:<5} "
                      f"{d['norm_ratio']:<12.4f} {d['kl_div']:<10.4f} {d['top5_overlap']:<14.2f} {d['selectivity']:<12.4f}")

# ===== Exp2: 一维轴真假验证 =====
print("\n\n### Exp2: 一维轴真假验证 - eff_rank对比 ###")
print(f"\n{'模型':<12} {'层':<5} {'eff_rank_raw':<14} {'top1_ratio':<12} {'eff_rank_whitened':<18} {'remove_top1':<14} {'remove_top3':<14}")
print("-" * 80)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    if model not in results or 'exp2_axis_verify' not in results[model]:
        continue
    exp2 = results[model]['exp2_axis_verify']
    for layer_key in sorted(exp2.keys()):
        d = exp2[layer_key]
        rm1 = d.get('remove_top_k', {}).get('remove_top_1', {}).get('eff_rank', 'N/A')
        rm3 = d.get('remove_top_k', {}).get('remove_top_3', {}).get('eff_rank', 'N/A')
        whitened = d.get('eff_rank_whitened', 'N/A')
        print(f"{model:<12} {layer_key:<5} {d['eff_rank_raw']:<14.4f} {d['top1_ratio']:<12.4f} "
              f"{whitened if isinstance(whitened, str) else f'{whitened:<18.4f}'} "
              f"{rm1 if isinstance(rm1, str) else f'{rm1:<14.4f}'} "
              f"{rm3 if isinstance(rm3, str) else f'{rm3:<14.4f}'}")

# ===== Exp3: vehicle反向分析 =====
print("\n\n### Exp3: vehicle反向分析 - 差分方向与读出方向的cos ###")
for model in ['qwen3', 'deepseek7b', 'glm4']:
    if model not in results or 'exp3_vehicle' not in results[model]:
        continue
    exp3 = results[model]['exp3_vehicle']
    print(f"\n{model}:")
    for layer_key in sorted(exp3.keys()):
        d = exp3[layer_key]
        veh_readout = d.get('vehicle_readout_cos', {})
        cross_lang = d.get('cross_lang_vehicle_cos', 'N/A')
        cat_diff_cos = d.get('cat_diff_cos', {})
        
        # vehicle vs other categories的cos
        veh_vs_tool = cat_diff_cos.get('vehicle_vs_tool', 'N/A')
        veh_vs_furniture = cat_diff_cos.get('vehicle_vs_furniture', 'N/A')
        
        print(f"  {layer_key}: cross_lang_veh_cos={cross_lang if isinstance(cross_lang, str) else f'{cross_lang:.4f}'}, "
              f"veh_vs_tool={veh_vs_tool if isinstance(veh_vs_tool, str) else f'{veh_vs_tool:.4f}'}, "
              f"veh_vs_furniture={veh_vs_furniture if isinstance(veh_vs_furniture, str) else f'{veh_vs_furniture:.4f}'}")
        
        # W_U readout cos
        if veh_readout:
            avg_cos = sum(veh_readout.values()) / len(veh_readout) if veh_readout else 0
            print(f"    W_U vehicle readout cos: {veh_readout} avg={avg_cos:.4f}")

# ===== Exp4: 中文候选族读出 =====
print("\n\n### Exp4: 中文候选族读出 - 中英文selectivity对比 ###")
print(f"\n{'模型':<12} {'类别':<10} {'beta':<5} {'EN_sel':<10} {'ZH_sel_old':<12} {'ZH_sel_new':<12}")
print("-" * 60)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    if model not in results or 'exp4_zh_readout' not in results[model]:
        continue
    exp4 = results[model]['exp4_zh_readout']
    for cat in ['fruit', 'animal', 'vehicle']:
        if cat not in exp4:
            continue
        for beta_key in ['beta_5.0', 'beta_10.0']:
            if beta_key not in exp4[cat]:
                continue
            d = exp4[cat][beta_key]
            print(f"{model:<12} {cat:<10} {beta_key[5:]:<5} "
                  f"{d['en_selectivity']:<10.4f} {d['zh_selectivity_old']:<12.4f} {d['zh_selectivity_new']:<12.4f}")

# ===== Exp5: 残差可写性 =====
print("\n\n### Exp5: 残差可写性 - 6类holdout selectivity ###")
print(f"\n{'模型':<12} {'类别':<10} {'层':<5} {'sel_beta5':<12} {'sel_beta10':<12}")
print("-" * 55)

for model in ['qwen3', 'deepseek7b', 'glm4']:
    if model not in results or 'exp5_writability' not in results[model]:
        continue
    exp5 = results[model]['exp5_writability']
    for cat in ['fruit', 'animal', 'vehicle', 'clothing', 'furniture', 'tool']:
        if cat not in exp5:
            continue
        for layer_key in sorted(exp5[cat].keys()):
            d = exp5[cat][layer_key]
            avg = d.get('avg_selectivity', {})
            b5 = avg.get('beta_5.0', 'N/A')
            b10 = avg.get('beta_10.0', 'N/A')
            print(f"{model:<12} {cat:<10} {layer_key:<5} "
                  f"{b5 if isinstance(b5, str) else f'{b5:<12.4f}'} "
                  f"{b10 if isinstance(b10, str) else f'{b10:<12.4f}'}")

print("\n" + "=" * 80)
print("Phase 465 R1 分析完成")
