import json, numpy as np

SLOT_ATTRS = {
    'apple': {'color': ['red', 'green', 'yellow'], 'taste': ['sweet', 'sour', 'juicy'], 'part': ['seed', 'skin', 'core', 'stem'], 'category': ['fruit', 'food', 'produce'], 'non_category': ['animal', 'tool', 'vehicle'], 'material': ['organic', 'fresh', 'natural'], 'random': ['square', 'loud', 'electric', 'digital']},
    'dog': {'color': ['brown', 'black', 'white'], 'part': ['leg', 'tail', 'fur', 'ear'], 'category': ['animal', 'pet', 'mammal'], 'non_category': ['fruit', 'tool', 'vehicle'], 'random': ['square', 'sweet', 'metallic', 'digital']},
    'knife': {'color': ['silver', 'gray', 'metallic'], 'part': ['blade', 'handle', 'edge', 'tip'], 'category': ['tool', 'weapon', 'instrument'], 'non_category': ['fruit', 'animal', 'vehicle'], 'material': ['metal', 'steel', 'iron'], 'random': ['sweet', 'furry', 'organic', 'digital']},
}

for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/glm5/phase448_{model}_r1.json') as f:
        data = json.load(f)
    
    print(f'\n{"="*60}')
    print(f'{model} Exp1: SlotMediation Decomposition')
    print(f'{"="*60}')
    exp1 = data.get('exp1_slot_decomposition', {})
    
    for obj_name in ['apple', 'dog', 'knife']:
        if obj_name not in exp1:
            continue
        obj_attrs = SLOT_ATTRS.get(obj_name, {})
        obj_data = exp1[obj_name]
        
        all_priors = []
        all_obj_conds = []
        all_conflict_deltas = []
        
        slot_scores = {}
        for slot_name, slot_data in obj_data.items():
            no_obj = slot_data.get('no_obj', {})
            with_obj = slot_data.get('with_obj', {})
            conflict = slot_data.get('conflict', {})
            
            slot_priors = []
            slot_conds = []
            slot_confs = []
            
            for group_name in obj_attrs:
                if not obj_attrs[group_name]:
                    continue
                no_v = no_obj.get(group_name)
                with_v = with_obj.get(group_name)
                conf_v = conflict.get(group_name)
                
                if no_v is not None:
                    all_priors.append(no_v)
                    slot_priors.append(no_v)
                if no_v is not None and with_v is not None:
                    oc = with_v - no_v
                    all_obj_conds.append(oc)
                    slot_conds.append(oc)
                if no_v is not None and conf_v is not None:
                    cd = conf_v - no_v
                    all_conflict_deltas.append(cd)
                    slot_confs.append(cd)
            
            if slot_priors:
                slot_scores[slot_name] = {
                    'prior': float(np.mean(slot_priors)),
                    'obj_cond': float(np.mean(slot_conds)) if slot_conds else 0,
                    'conflict_delta': float(np.mean(slot_confs)) if slot_confs else 0,
                }
        
        avg_prior = float(np.mean(all_priors)) if all_priors else 0
        avg_obj_cond = float(np.mean(all_obj_conds)) if all_obj_conds else 0
        avg_conflict = float(np.mean(all_conflict_deltas)) if all_conflict_deltas else 0
        
        prior_ratio = abs(avg_prior) / (abs(avg_prior) + abs(avg_obj_cond) + 1e-8)
        obj_cond_ratio = abs(avg_obj_cond) / (abs(avg_prior) + abs(avg_obj_cond) + 1e-8)
        conflict_resilience = avg_conflict / (avg_obj_cond + 1e-8) if abs(avg_obj_cond) > 0.01 else 0
        
        print(f'\n  {obj_name}: avg_prior={avg_prior:.3f}, avg_obj_cond={avg_obj_cond:.3f}, avg_conflict={avg_conflict:.3f}')
        print(f'    PriorScore={prior_ratio:.3f}, ObjCondScore={obj_cond_ratio:.3f}, ConflictResilience={conflict_resilience:.3f}')
        
        for slot_name in sorted(slot_scores.keys()):
            s = slot_scores[slot_name]
            p = s['prior']
            oc = s['obj_cond']
            cd = s['conflict_delta']
            print(f'    {slot_name}: prior={p:.2f}, obj_cond={oc:.2f}, conflict_delta={cd:.2f}')

    print(f'\n--- {model} Exp3: Alpha Regime ---')
    exp3 = data.get('exp3_alpha_regime', {})
    for obj_name in ['apple', 'dog', 'knife']:
        if obj_name in exp3:
            regimes = exp3[obj_name].get('regimes', {})
            items = sorted(regimes.items(), key=lambda x: float(x[0]))
            regime_str = ', '.join(f'a={a}={r}' for a, r in items)
            print(f'  {obj_name}: {regime_str}')
