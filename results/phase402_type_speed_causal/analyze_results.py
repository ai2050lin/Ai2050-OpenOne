"""
Phase 402 Results Analysis
分析三个模型的因果分解结果
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_results(model_name):
    """加载模型结果"""
    path = Path(__file__).parent / f"{model_name}_phase402_optimized.json"
    with open(path, 'r') as f:
        return json.load(f)

def analyze_model(results):
    """分析单个模型结果"""
    model_name = results['model']
    print(f"\n{'='*60}")
    print(f"模型: {model_name}")
    print(f"{'='*60}")
    
    per_layer = results['per_layer']
    for layer_str, layer_data in per_layer.items():
        li = int(layer_str)
        agg = layer_data['aggregate']
        
        print(f"\n层 L{li}:")
        print(f"  Full方向: within-type={agg['full_within_odd']:.4f}, across-type={agg['full_across_odd']:.4f}, diff={agg['full_within_odd']-agg['full_across_odd']:.4f}")
        print(f"  TYPE成分: within-type={agg['type_within_odd']:.4f}, across-type={agg['type_across_odd']:.4f}, diff={agg['type_within_odd']-agg['type_across_odd']:.4f}")
        print(f"  SPEED成分: within-type={agg['speed_within_odd']:.4f}, across-type={agg['speed_across_odd']:.4f}, diff={agg['speed_within_odd']-agg['speed_across_odd']:.4f}")
        print(f"  SPEED成分 (速度相似性): same-speed={agg['speed_samespeed_odd']:.4f}, diff-speed={agg['speed_diffspeed_odd']:.4f}, diff={agg['speed_samespeed_odd']-agg['speed_diffspeed_odd']:.4f}")
        
        # 计算成分贡献比例
        decomp = layer_data['decomposition_norms']
        total_type = sum(v['type_comp_norm'] for v in decomp.values())
        total_speed = sum(v['speed_comp_norm'] for v in decomp.values())
        total_residual = sum(v['residual_norm'] for v in decomp.values())
        total = total_type + total_speed + total_residual
        
        print(f"  成分比例: TYPE={total_type/total:.1%}, SPEED={total_speed/total:.1%}, RESIDUAL={total_residual/total:.1%}")

def compare_models():
    """比较三个模型的结果"""
    models = ['qwen3', 'deepseek7b', 'glm4']
    all_results = {}
    
    for model in models:
        try:
            results = load_results(model)
            all_results[model] = results
        except FileNotFoundError:
            print(f"警告: 未找到 {model} 的结果文件")
            continue
    
    print(f"\n{'='*80}")
    print(f"三模型对比分析")
    print(f"{'='*80}")
    
    # 深层结果对比
    print(f"\n深层 (late layer) 结果对比:")
    print(f"{'模型':<12} {'层':<6} {'Full(within-across)':<20} {'TYPE(within-across)':<20} {'SPEED(within-across)':<20}")
    print(f"{'-'*80}")
    
    for model in models:
        if model not in all_results:
            continue
            
        results = all_results[model]
        per_layer = results['per_layer']
        
        # 获取深层（最后一个层）
        layer_keys = sorted(per_layer.keys(), key=int)
        if not layer_keys:
            continue
            
        last_layer = layer_keys[-1]
        layer_data = per_layer[last_layer]
        agg = layer_data['aggregate']
        
        full_diff = agg['full_within_odd'] - agg['full_across_odd']
        type_diff = agg['type_within_odd'] - agg['type_across_odd']
        speed_diff = agg['speed_within_odd'] - agg['speed_across_odd']
        
        print(f"{model:<12} L{last_layer:<4} {full_diff:>+10.4f}{' ':<10} {type_diff:>+10.4f}{' ':<10} {speed_diff:>+10.4f}")
    
    print(f"\n{'='*80}")
    print(f"关键发现:")
    print(f"{'='*80}")
    
    # 分析每个模型的关键特征
    for model in models:
        if model not in all_results:
            continue
            
        results = all_results[model]
        per_layer = results['per_layer']
        
        # 获取早期和深层
        layer_keys = sorted(per_layer.keys(), key=int)
        early = per_layer[layer_keys[0]]
        late = per_layer[layer_keys[-1]]
        
        early_agg = early['aggregate']
        late_agg = late['aggregate']
        
        print(f"\n{model}:")
        
        # 1. Full方向效应
        early_full = early_agg['full_within_odd'] - early_agg['full_across_odd']
        late_full = late_agg['full_within_odd'] - late_agg['full_across_odd']
        
        # 2. TYPE成分效应
        early_type = early_agg['type_within_odd'] - early_agg['type_across_odd']
        late_type = late_agg['type_within_odd'] - late_agg['type_across_odd']
        
        # 3. SPEED成分效应
        early_speed = early_agg['speed_within_odd'] - early_agg['speed_across_odd']
        late_speed = late_agg['speed_within_odd'] - late_agg['speed_across_odd']
        
        # 4. SPEED成分速度相似性
        early_speed_sim = early_agg['speed_samespeed_odd'] - early_agg['speed_diffspeed_odd']
        late_speed_sim = late_agg['speed_samespeed_odd'] - late_agg['speed_diffspeed_odd']
        
        print(f"  Full方向: 早期{early_full:+.4f} → 深层{late_full:+.4f}")
        print(f"  TYPE成分: 早期{early_type:+.4f} → 深层{late_type:+.4f}")
        print(f"  SPEED成分: 早期{early_speed:+.4f} → 深层{late_speed:+.4f}")
        print(f"  速度相似性: 早期{early_speed_sim:+.4f} → 深层{late_speed_sim:+.4f}")
        
        # 判断模式
        if late_full > 0.05:
            print(f"  → Full方向: within-type > across-type (类型聚类明显)")
        elif late_full < -0.05:
            print(f"  → Full方向: within-type < across-type (跨类型传递更强)")
        else:
            print(f"  → Full方向: within-type ≈ across-type (无显著类型效应)")
            
        if late_type > 0.05:
            print(f"  → TYPE成分: 在类型聚类中起主要作用")
        elif abs(late_type) < 0.05:
            print(f"  → TYPE成分: 作用微弱")
            
        if late_speed > 0.05:
            print(f"  → SPEED成分: 在类型聚类中起主要作用")
        elif abs(late_speed) < 0.05:
            print(f"  → SPEED成分: 作用微弱")
            
        if abs(late_speed_sim) > 0.1:
            if late_speed_sim > 0:
                print(f"  → 速度相似性: 同速对象间传递更强 (speed-level效应)")
            else:
                print(f"  → 速度相似性: 异速对象间传递更强 (反直觉)")
        else:
            print(f"  → 速度相似性: 无显著speed-level效应")

def main():
    """主分析函数"""
    print("Phase 402 因果分解结果分析")
    print("="*60)
    
    # 分析每个模型
    for model in ['qwen3', 'deepseek7b', 'glm4']:
        try:
            results = load_results(model)
            analyze_model(results)
        except FileNotFoundError:
            print(f"\n模型 {model} 结果文件未找到")
    
    # 对比分析
    compare_models()
    
    print(f"\n{'='*80}")
    print(f"总结与结论")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()