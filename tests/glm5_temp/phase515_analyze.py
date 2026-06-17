import sys; sys.stdout.reconfigure(encoding='utf-8')
import json

d = json.load(open('d:/AI2050/Ai2050-OpenOne/results/glm5_phase515_trajectory_subspace/phase515_qwen3_trajectory_subspace.json','r',encoding='utf-8'))

print("=== Qwen3 Phase 515 Results ===")
for cat, cr in d['category_results'].items():
    print(f"\n--- {cat} ---")
    
    # Exp1: U_trajectory
    exp1 = cr['utraj_discovery']
    qd = exp1['quality_dist']
    print(f"  Quality dist: {qd}")
    dirs = exp1['traj_directions']
    for li, dinfo in dirs.items():
        print(f"  L{li}: norm={dinfo['direction_norm']:.2f}, suc_n={dinfo['suc_n']}, fail_n={dinfo['fail_n']}")
        if dinfo.get('suc_cat_logit_mean'):
            print(f"    suc_cat_logit={dinfo['suc_cat_logit_mean']:.2f}, fail_cat_logit={dinfo['fail_cat_logit_mean']:.2f}")
    
    # Exp2: Multi-step hub
    exp2 = cr['multistep_hub']
    for hub, hdata in exp2.items():
        traj = hdata['avg_cat_logit_trajectory']
        qd2 = hdata['quality_dist']
        print(f"  Hub '{hub}': cat_logit_traj={traj}, quality={qd2}")
    
    # Exp4: Intervention
    exp4 = cr.get('utraj_intervention', {})
    if exp4.get('note') != 'no_directions_found':
        for key, val in exp4.items():
            print(f"  Intervention {key}: add_delta={val.get('delta_add','N/A')}, remove_delta={val.get('delta_remove','N/A')}")
    else:
        print(f"  Intervention: skipped (no directions)")

# Exp3: Action templates
exp3 = d['action_template_comparison']
print(f"\n--- Action Template Comparison ---")
for tmpl_type, data in exp3.items():
    print(f"  {tmpl_type}: action_hit={data['action_word_hit_rate']}, cat_hit={data['category_word_hit_rate']}, quality={data['quality_dist']}")
    # Show examples
    for ex in data['examples'][:3]:
        print(f"    '{ex['prompt']}' -> quality={ex['quality']}")
        print(f"    gen: '{ex['gen_text'][:100]}'")