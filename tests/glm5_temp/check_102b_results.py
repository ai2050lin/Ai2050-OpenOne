import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('tests/glm5_temp/phase102b_exp1b_qwen3_delta_h_intervention.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看L33, alpha=229.0的结果
l33 = data['intervention']['L33']
alpha_key = 'alpha_229.0'
if alpha_key in l33:
    for zh in ['龙', '云', '光', '冰', '沙', '梦']:
        print(f'=== {zh} ===')
        
        # baseline
        base = data['baseline'].get(zh, {})
        en = base.get('en_translation', '?')
        print(f'  Baseline: P({en})={base.get("en_prob", 0):.8f}, en_in_top20={base.get("en_tokens_in_top20", 0)}')
        
        # delta_h intervention
        dh = l33[alpha_key]['delta_h_intervention'].get(zh, {})
        if 'top5' in dh:
            top5_str = ', '.join([f'{t["token"]}({t["prob"]:.3f})' for t in dh['top5']])
            print(f'  Dh干预 top5: {top5_str}')
            print(f'  Dh P({en})={dh.get("en_prob", 0):.8f}, Δ={dh.get("en_prob_change", 0):.8f}')
        
        # contextual intervention
        ctx = l33[alpha_key]['contextual_intervention'].get(zh, {})
        if 'top5' in ctx:
            top5_str = ', '.join([f'{t["token"]}({t["prob"]:.3f})' for t in ctx['top5']])
            print(f'  Ctx干预 top5: {top5_str}')
            print(f'  Ctx P({en})={ctx.get("en_prob", 0):.8f}, Δ={ctx.get("en_prob_change", 0):.8f}')
        print()

# 也看L9 alpha=21.8的结果
print("=== L9 alpha=21.8 ===")
l9 = data['intervention']['L9']
alpha_key = 'alpha_21.8'
if alpha_key in l9:
    for zh in ['龙', '云', '光', '冰']:
        base = data['baseline'].get(zh, {})
        en = base.get('en_translation', '?')
        ctx = l9[alpha_key]['contextual_intervention'].get(zh, {})
        if 'top5' in ctx:
            top5_str = ', '.join([f'{t["token"]}({t["prob"]:.3f})' for t in ctx['top5']])
            print(f'  {zh}({en}) Ctx top5: {top5_str}, ΔP={ctx.get("en_prob_change", 0):.8f}')
