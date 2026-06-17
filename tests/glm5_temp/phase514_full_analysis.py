import sys; sys.stdout.reconfigure(encoding='utf-8')
import json

d = json.load(open('d:/AI2050/Ai2050-OpenOne/results/glm5_phase514_path_value/phase514_qwen3_path_value.json','r',encoding='utf-8'))

print(f"=== Qwen3 Phase 514 Full Results ===")
print(f"Steps: {d['steps']}")
print(f"Test objects: {d['test_objects']}")
print(f"Categories: {d['categories']}")

for cat in d['categories']:
    cr = d['category_results'][cat]
    pv = cr['path_value']
    pvs = pv['path_values']
    
    print(f"\n--- {cat} ---")
    print(f"  n_prompts: {pv['n_prompts']}")
    
    # Group summary - these are floats directly
    gs = pv['group_summary']
    print(f"  Group V_c(semantic):")
    for gname, gval in sorted(gs.items(), key=lambda x: -x[1]):
        if gval > 0:
            print(f"    {gname}: {gval:.4f}")
    
    # Logit vs Value
    lv = pv['logit_vs_value']
    print(f"  Match rate: {lv['match_rate']}")
    
    # Path values - sorted by V_c_semantic
    print(f"  Top-10 path values:")
    sorted_pv = sorted(pvs.items(), key=lambda x: -x[1].get('V_c_semantic',0))[:10]
    for name, info in sorted_pv:
        print(f"    {name}: V_sem={info.get('V_c_semantic',0):.4f}, V_lex={info.get('V_c_lexical',0):.4f}, logit_rank={info.get('avg_logit_rank','?')}, group={info.get('group','?')}")
    
    # Hub effects
    he = cr.get('hub_hidden_state', {}).get('hub_effects', {})
    print(f"  Hub effects on cat_logit:")
    for hname, heff in sorted(he.items()):
        delta = heff['cat_logit_delta']
        base = heff['cat_logit_base']
        after = heff['cat_logit_after_hub']
        std = heff.get('cat_logit_delta_std', 'N/A')
        print(f"    {hname}: base={base:.2f} -> after={after:.2f}, delta={delta:+.3f} (std={std})")
    
    # Intervention
    interv = cr.get('intervention', {})
    for cond in ['clean', 'best_hub', 'boost_cat_logit', 'best_hub_plus_boost']:
        ci = interv.get(cond, {})
        if ci:
            bh_name = ci.get('hub_name', ci.get('name', '?'))
            print(f"    {cond} ({bh_name}): lex={ci.get('lexical_hit_rate','?')}, nat={ci.get('natural_hit_rate','?')}, sem={ci.get('semantic_hit_rate','?')}")

# Now GLM4
d2 = json.load(open('d:/AI2050/Ai2050-OpenOne/results/glm5_phase514_path_value/phase514_glm4_path_value.json','r',encoding='utf-8'))
print(f"\n=== GLM4 Phase 514 ===")
for cat in d2['categories']:
    cr = d2['category_results'][cat]
    pv = cr['path_value']
    gs = pv['group_summary']
    print(f"\n--- {cat} ---")
    print(f"  Group V_c(semantic):")
    for gname, gval in sorted(gs.items(), key=lambda x: -x[1]):
        if gval > 0:
            print(f"    {gname}: {gval:.4f}")
    lv = pv['logit_vs_value']
    print(f"  Match rate: {lv['match_rate']}")
    he = cr.get('hub_hidden_state', {}).get('hub_effects', {})
    print(f"  Hub effects on cat_logit:")
    for hname, heff in sorted(he.items()):
        delta = heff['cat_logit_delta']
        print(f"    {hname}: delta={delta:+.3f}")
    interv = cr.get('intervention', {})
    for cond in ['clean', 'best_hub', 'boost_cat_logit', 'best_hub_plus_boost']:
        ci = interv.get(cond, {})
        if ci:
            bh_name = ci.get('hub_name', ci.get('name', '?'))
            print(f"    {cond} ({bh_name}): lex={ci.get('lexical_hit_rate','?')}, nat={ci.get('natural_hit_rate','?')}, sem={ci.get('semantic_hit_rate','?')}")

# DS7B
d3 = json.load(open('d:/AI2050/Ai2050-OpenOne/results/glm5_phase514_path_value/phase514_deepseek7b_path_value.json','r',encoding='utf-8'))
print(f"\n=== DS7B Phase 514 ===")
for cat in d3['categories']:
    cr = d3['category_results'][cat]
    pv = cr['path_value']
    gs = pv['group_summary']
    print(f"\n--- {cat} ---")
    print(f"  Group V_c(semantic):")
    for gname, gval in sorted(gs.items(), key=lambda x: -x[1]):
        if gval > 0:
            print(f"    {gname}: {gval:.4f}")
    lv = pv['logit_vs_value']
    print(f"  Match rate: {lv['match_rate']}")
    he = cr.get('hub_hidden_state', {}).get('hub_effects', {})
    print(f"  Hub effects on cat_logit:")
    for hname, heff in sorted(he.items()):
        delta = heff['cat_logit_delta']
        print(f"    {hname}: delta={delta:+.3f}")
    interv = cr.get('intervention', {})
    for cond in ['clean', 'best_hub', 'boost_cat_logit', 'best_hub_plus_boost']:
        ci = interv.get(cond, {})
        if ci:
            bh_name = ci.get('hub_name', ci.get('name', '?'))
            print(f"    {cond} ({bh_name}): lex={ci.get('lexical_hit_rate','?')}, nat={ci.get('natural_hit_rate','?')}, sem={ci.get('semantic_hit_rate','?')}")