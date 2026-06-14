"""Phase 487结果提取和分析"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase487_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n{"="*70}')
    print(f'  {model.upper()} - Phase 487 Results')
    print(f'{"="*70}')
    
    # Exp1: 正交成分因果测试
    print(f'\n--- Exp1: Orthogonal Component Causal Test ---')
    exp1 = data.get('exp1_orthogonal_component_causal', {})
    if 'error' in exp1:
        print(f'  ERROR: {exp1["error"][:100]}')
    else:
        for cat, d in exp1.items():
            if not isinstance(d, dict) or 'ablation_results' not in d:
                continue
            dir_remove = d.get('direction_remove_target', 0)
            print(f'\n  {cat} (dir_remove_target={dir_remove:.2f}):')
            for layer_key, layer_abl in d['ablation_results'].items():
                print(f'    {layer_key}:')
                for abl_type in ['proj_bc', 'orth_bc', 'full_mlp']:
                    if abl_type in layer_abl:
                        a = layer_abl[abl_type]
                        print(f'      {abl_type:10s}: target_D={a["target_delta"]:+.2f}, '
                              f'amp={a["amplitude_ratio"]:.1%}, cos={a["cos_with_direction_remove"]:.3f}')
    
    # Exp2: 连续段消融
    print(f'\n--- Exp2: Segment Ablation ---')
    exp2 = data.get('exp2_segment_ablation', {})
    if 'error' in exp2:
        print(f'  ERROR: {exp2["error"][:100]}')
    else:
        for cat, d in exp2.items():
            if not isinstance(d, dict) or 'segments' not in d:
                continue
            dir_remove = d.get('direction_remove_target', 0)
            print(f'\n  {cat} (dir_remove_target={dir_remove:.2f}):')
            for seg_name, seg_data in d['segments'].items():
                abl = seg_data.get('ablation_results', {})
                if not abl:
                    continue
                parts = []
                for abl_type in ['full_mlp', 'proj_bc', 'orth_bc']:
                    if abl_type in abl:
                        a = abl[abl_type]
                        parts.append(f'{abl_type}: D={a["target_delta"]:+.2f} amp={a["amplitude_ratio"]:.0%}')
                print(f'    {seg_name:10s}: {" | ".join(parts)}')
    
    # Exp3: 反对层因果验证
    print(f'\n--- Exp3: Opposition Layer Causal ---')
    exp3 = data.get('exp3_opposition_layer_causal', {})
    if 'error' in exp3:
        print(f'  ERROR: {exp3["error"][:100]}')
    else:
        for cat, d in exp3.items():
            if not isinstance(d, dict) or 'intervention_results' not in d:
                continue
            print(f'\n  {cat}:')
            print(f'    attn_opp: {d.get("attn_opp_layers", [])}, '
                  f'mlp_opp: {d.get("mlp_opp_layers", [])}, '
                  f'support: {d.get("support_layers", [])}')
            for action_name, action_data in d['intervention_results'].items():
                print(f'    {action_name:25s}: target_D={action_data["target_delta"]:+.2f} '
                      f'(layers={action_data["layers"]})')
    
    # Exp4: 格式-语义分离
    print(f'\n--- Exp4: Format-Semantic Separation ---')
    exp4 = data.get('exp4_format_semantic_separation', {})
    if 'error' in exp4:
        print(f'  ERROR: {exp4["error"][:100]}')
    else:
        for cat, d in exp4.items():
            if not isinstance(d, dict) or 'format_semantic_profile' not in d:
                continue
            print(f'\n  {cat}:')
            profile = d['format_semantic_profile']
            # 只显示最佳层和最高层
            key_layers = sorted(profile.keys(), key=int)
            for l in key_layers:
                p = profile[l]
                print(f'    L{l}: cos_fmt={p["cos_bc_format_top1"]:.3f}, '
                      f'cos_sem={p["cos_bc_semantic_top1"]:.3f}, '
                      f'proj_fmt={p["proj_ratio_format"]:.3f}, '
                      f'proj_sem={p["proj_ratio_semantic"]:.3f}')
