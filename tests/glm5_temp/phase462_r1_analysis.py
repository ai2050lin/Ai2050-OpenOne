"""Phase 462 R1结果分析"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, numpy as np
from pathlib import Path

def load_r1(model):
    path = f"results/glm5/phase462_{model}_r1.json"
    with open(path, encoding='utf-8') as f:
        return json.load(f)

for model in ['qwen3', 'glm4', 'deepseek7b']:
    d = load_r1(model)
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}")
    
    # Exp1: 跨语言探针
    exp1 = d.get('exp1_cross_lang_probe', {})
    en_zh = exp1.get('en_zh_acc', {})
    en_en = exp1.get('en_en_acc', {})
    center_cos = exp1.get('center_cosines', {})
    
    print(f"\n--- Exp1: Cross-language probe ---")
    for lk in sorted(en_zh.keys(), key=lambda x: int(x[1:])):
        acc = en_zh[lk]['acc']
        n = en_zh[lk]['n']
        en_acc = en_en.get(lk, {}).get('acc', 0)
        cos_avg = np.mean(list(center_cos.get(lk, {}).values())) if center_cos.get(lk) else 0
        # Per-category breakdown
        per_cat = en_zh[lk].get('per_cat', {})
        cat_str = " | ".join([f"{k[:3]}={v:.0%}" for k, v in per_cat.items()])
        print(f"  {lk}: EN→EN={en_acc:.2f}, EN→ZH={acc:.2f} (n={n}), cos={cos_avg:.3f} | {cat_str}")
    
    # Exp2b: Activation Patch
    exp2b = d.get('exp2b_residual_patch', {})
    print(f"\n--- Exp2b: Cross-language Activation Patch ---")
    
    # 按层汇总
    layer_effects = {}
    for key, val in exp2b.items():
        li = val['patch_layer']
        if li not in layer_effects:
            layer_effects[li] = []
        layer_effects[li].append(val['delta_en_margin'])
    
    for li in sorted(layer_effects.keys()):
        deltas = layer_effects[li]
        avg_d = np.mean(deltas)
        pos_count = sum(1 for d in deltas if d > 0)
        print(f"  L{li}: avg_Δ={avg_d:+.2f}, positive={pos_count}/{len(deltas)}")
    
    # Exp3: 翻译方向分解
    exp3 = d.get('exp3_translate_decomposition', {})
    print(f"\n--- Exp3: Translate direction decomposition ---")
    for lk in sorted(exp3.keys(), key=lambda x: int(x[1:])):
        dd = exp3[lk]
        tc = dd.get('translate_diff_cos', 'N/A')
        tn = dd.get('target_lang_vs_surface_lang_cos', 'N/A')
        cc = dd.get('content_vs_translate_cos', 'N/A')
        tdn = dd.get('target_lang_diff_norm', 'N/A')
        cdfn = dd.get('content_diff_fruit_animal_norm', 'N/A')
        print(f"  {lk}: translate_cos={tc if isinstance(tc, str) else f'{tc:.3f}'}, "
              f"target_vs_surface={tn if isinstance(tn, str) else f'{tn:.3f}'}, "
              f"content_vs_translate={cc if isinstance(cc, str) else f'{cc:.3f}'}")
    
    # Exp4: 写入向量 vs 残差差分
    exp4 = d.get('exp4_write_vs_residual', {})
    print(f"\n--- Exp4: Write vector vs Residual diff ---")
    for key, val in exp4.items():
        if 'error' in val:
            print(f"  {key}: ERROR={val['error']}")
            continue
        inj = val.get('injections', {})
        b10 = inj.get('beta10', {})
        print(f"  {key}: alignment={val.get('alignment_write_vs_diff',0):.3f}, "
              f"b10: residual_sel={b10.get('residual_selectivity',0):.2f}, "
              f"write_sel={b10.get('write_vec_selectivity',0):.2f}")
