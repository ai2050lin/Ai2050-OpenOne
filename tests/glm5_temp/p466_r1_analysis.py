"""Phase 466 R1结果分析"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

def load_r(model, r):
    path = f"results/glm5/phase466_{model}_r{r}.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)

print("=" * 80)
print("### Exp1: 白化方向 vs 原始方向 vs 去主轴方向 ###")
print("=" * 80)
for model in ['qwen3', 'deepseek7b', 'glm4']:
    d = load_r(model, 1)
    if not d or 'error' in d.get('exp1_whitened_injection', {}):
        print(f"\n{model}: EXP1 ERROR")
        continue
    exp1 = d['exp1_whitened_injection']
    print(f"\n{model}:")
    for lk in sorted(exp1.keys()):
        ld = exp1[lk]
        print(f"  {lk}:")
        # 方向余弦
        cos_info = ld.get('animal', ld.get('vehicle', {})).get('direction_cosine', {})
        if cos_info:
            print(f"    direction_cosine: raw_vs_whitened={cos_info.get('raw_vs_whitened','N/A')}, "
                  f"raw_vs_nopc1={cos_info.get('raw_vs_nopc1','N/A')}")
        for cat in ['animal', 'vehicle', 'fruit']:
            if cat not in ld:
                continue
            cd = ld[cat]
            # 比较3种方法在ratio=1时的selectivity
            raw_sel = cd.get('raw', {}).get('ratio_1.0', {}).get('selectivity', 'N/A')
            white_sel = cd.get('whitened_back', {}).get('ratio_1.0', {}).get('selectivity', 'N/A')
            nopc1_sel = cd.get('no_pc1', {}).get('ratio_1.0', {}).get('selectivity', 'N/A')
            raw_kl = cd.get('raw', {}).get('ratio_1.0', {}).get('kl_div', 'N/A')
            nopc1_kl = cd.get('no_pc1', {}).get('ratio_1.0', {}).get('kl_div', 'N/A')
            print(f"    {cat}: raw_sel={raw_sel}, white_sel={white_sel}, nopc1_sel={nopc1_sel}, "
                  f"raw_kl={raw_kl}, nopc1_kl={nopc1_kl}")

print("\n\n" + "=" * 80)
print("### Exp2: 自适应beta校准 — 最佳norm_ratio ###")
print("=" * 80)
for model in ['qwen3', 'deepseek7b', 'glm4']:
    d = load_r(model, 1)
    if not d or 'error' in d.get('exp2_adaptive_beta', {}):
        print(f"\n{model}: EXP2 ERROR")
        continue
    exp2 = d['exp2_adaptive_beta']
    print(f"\n{model}:")
    for lk in sorted(exp2.keys()):
        ld = exp2[lk]
        print(f"  {lk}:")
        for cat in ['animal', 'vehicle', 'fruit', 'clothing']:
            if cat not in ld:
                continue
            cd = ld[cat]
            best_ratio = None
            best_sel = -999
            for rk in ['ratio_0.25', 'ratio_0.5', 'ratio_1.0', 'ratio_2.0', 'ratio_4.0']:
                if rk not in cd:
                    continue
                sel = cd[rk]['selectivity']
                kl = cd[rk]['kl_div']
                if sel > best_sel and kl < 0.5:
                    best_sel = sel
                    best_ratio = rk
            # Print ratio=1.0 and best
            r1 = cd.get('ratio_1.0', {})
            r05 = cd.get('ratio_0.5', {})
            print(f"    {cat}: ratio0.5_sel={r05.get('selectivity','N/A'):.4f} kl={r05.get('kl_div','N/A'):.4f}, "
                  f"ratio1_sel={r1.get('selectivity','N/A'):.4f} kl={r1.get('kl_div','N/A'):.4f}, "
                  f"best={best_ratio}_sel={best_sel:.4f}")

print("\n\n" + "=" * 80)
print("### Exp3: 类别混叠剥离 — vehicle/tool/furniture ###")
print("=" * 80)
for model in ['qwen3', 'deepseek7b', 'glm4']:
    d = load_r(model, 1)
    if not d or 'error' in d.get('exp3_disentangle', {}):
        print(f"\n{model}: EXP3 ERROR")
        continue
    exp3 = d['exp3_disentangle']
    print(f"\n{model}:")
    for lk in sorted(exp3.keys()):
        ld = exp3[lk]
        print(f"  {lk}:")
        for cat in ['vehicle', 'tool', 'furniture']:
            if cat not in ld:
                continue
            cd = ld[cat]
            raw = cd.get('raw_selectivity', 'N/A')
            dis = cd.get('disentangle_selectivity', 'N/A')
            rnd = cd.get('random_selectivity', 'N/A')
            ploss = cd.get('projection_loss_ratio', 'N/A')
            raw_cos = cd.get('raw_cos_with_others', {})
            dis_cos = cd.get('disentangle_cos_with_others', {})
            print(f"    {cat}: raw_sel={raw}, disentangle_sel={dis}, random_sel={rnd}, proj_loss={ploss}")
            if raw_cos:
                print(f"      raw_cos: {raw_cos}")
            if dis_cos:
                print(f"      dis_cos: {dis_cos}")

print("\n\n" + "=" * 80)
print("### Exp4: clothing候选族修复 ###")
print("=" * 80)
for model in ['qwen3', 'deepseek7b', 'glm4']:
    d = load_r(model, 1)
    if not d or 'error' in d.get('exp4_clothing_fix', {}):
        print(f"\n{model}: EXP4 ERROR")
        continue
    exp4 = d['exp4_clothing_fix']
    print(f"\n{model}:")
    # Clothing tokenization
    ct = exp4.get('clothing_tokenization', {})
    found = [w for w, v in ct.items() if v.get('found', False)]
    not_found = [w for w, v in ct.items() if not v.get('found', False)]
    print(f"  clothing found: {found}")
    print(f"  clothing NOT found: {not_found}")
    cs = exp4.get('clothing_selectivity', {})
    if cs:
        print(f"  standard_sel={cs.get('standard')}, alt_sel={cs.get('alt_families')}")

print("\n\n" + "=" * 80)
print("### Exp5: 生成质量 ###")
print("=" * 80)
for model in ['qwen3', 'deepseek7b', 'glm4']:
    d = load_r(model, 1)
    if not d or 'error' in d.get('exp5_generation_quality', {}):
        print(f"\n{model}: EXP5 ERROR")
        continue
    exp5 = d['exp5_generation_quality']
    print(f"\n{model}:")
    for cat in ['fruit', 'animal', 'vehicle']:
        if cat not in exp5:
            continue
        cd = exp5[cat]
        base = cd.get('gen_base', '')[:80]
        r1 = cd.get('gen_patch_ratio1', '')[:80]
        r2 = cd.get('gen_patch_ratio2', '')[:80]
        print(f"  {cat}:")
        print(f"    base:   {base}")
        print(f"    ratio1: {r1}")
        print(f"    ratio2: {r2}")
