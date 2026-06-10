"""Phase 449 R1 Analysis"""
import json, numpy as np

TEST_OBJECTS = {
    'apple':  {'category': 'fruit',  'opp_category': 'animal', 'slot_words': {'category': ['fruit', 'food', 'produce'], 'color': ['red', 'green', 'yellow'], 'taste': ['sweet', 'sour', 'juicy'], 'part': ['seed', 'skin', 'core'], 'material': ['organic', 'fresh', 'natural']}},
    'dog':    {'category': 'animal', 'opp_category': 'fruit',  'slot_words': {'category': ['animal', 'pet', 'mammal'], 'color': ['brown', 'black', 'white'], 'part': ['leg', 'tail', 'fur']}},
    'knife':  {'category': 'tool',   'opp_category': 'fruit',  'slot_words': {'category': ['tool', 'weapon', 'instrument'], 'color': ['silver', 'gray', 'metallic'], 'part': ['blade', 'handle', 'edge'], 'material': ['metal', 'steel', 'iron']}},
    'orange': {'category': 'fruit',  'opp_category': 'animal', 'slot_words': {'category': ['fruit', 'food', 'citrus'], 'color': ['orange', 'yellow'], 'taste': ['sweet', 'sour', 'juicy']}},
    'hammer': {'category': 'tool',   'opp_category': 'animal', 'slot_words': {'category': ['tool', 'instrument', 'equipment'], 'color': ['brown', 'gray'], 'part': ['head', 'handle', 'nail'], 'material': ['metal', 'wood', 'steel']}},
    'cat':    {'category': 'animal', 'opp_category': 'tool',   'slot_words': {'category': ['animal', 'pet', 'mammal'], 'color': ['black', 'white', 'orange'], 'part': ['leg', 'tail', 'fur']}},
}

for model in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/glm5/phase449_{model}_r1.json') as f:
        data = json.load(f)

    print(f'\n=== {model} Exp1: Unlock Gate ===')
    exp1 = data.get('exp1_unlock_gate', {})

    # 按类别汇总
    cat_metrics = {'fruit': {}, 'animal': {}, 'tool': {}}
    for obj_name, obj_cfg in TEST_OBJECTS.items():
        cat = obj_cfg['category']
        metrics = exp1.get(obj_name, {}).get('metrics', {})
        for slot_name, m in metrics.items():
            if slot_name not in cat_metrics[cat]:
                cat_metrics[cat][slot_name] = {'unlock': [], 'conflict': [], 'repeat': [], 'pos': [], 'replace': []}
            cat_metrics[cat][slot_name]['unlock'].append(m['unlock_score'])
            cat_metrics[cat][slot_name]['conflict'].append(m['conflict_recovery'])
            cat_metrics[cat][slot_name]['repeat'].append(m['obj_repeat_ctrl'])
            cat_metrics[cat][slot_name]['pos'].append(m['position_ctrl'])
            cat_metrics[cat][slot_name]['replace'].append(m['obj_replace_ctrl'])

    # 总体平均
    all_unlock, all_conflict, all_repeat, all_pos, all_replace = [], [], [], [], []
    for cat, slots in cat_metrics.items():
        for slot, vals in slots.items():
            all_unlock.extend(vals['unlock'])
            all_conflict.extend(vals['conflict'])
            all_repeat.extend(vals['repeat'])
            all_pos.extend(vals['pos'])
            all_replace.extend(vals['replace'])

    print(f'  Overall: unlock={np.mean(all_unlock):+.2f}, conflict={np.mean(all_conflict):+.2f}, repeat={np.mean(all_repeat):+.2f}, pos={np.mean(all_pos):+.2f}, replace={np.mean(all_replace):+.2f}')

    # 按类别
    for cat in ['fruit', 'animal', 'tool']:
        if cat not in cat_metrics or not cat_metrics[cat]:
            continue
        u = [v for vals in cat_metrics[cat].values() for v in vals['unlock']]
        c = [v for vals in cat_metrics[cat].values() for v in vals['conflict']]
        r = [v for vals in cat_metrics[cat].values() for v in vals['repeat']]
        print(f'  {cat}: unlock={np.mean(u):+.2f}, conflict={np.mean(c):+.2f}, repeat_ctrl={np.mean(r):+.2f}')

    # 判断机制类型
    avg_unlock = np.mean(all_unlock)
    avg_conflict = np.mean(all_conflict)
    avg_repeat = np.mean(all_repeat)
    if avg_unlock > 3 and abs(avg_repeat) < 1:
        unlock_type = 'template-prior-driven'
    elif avg_unlock > 3 and avg_conflict > 0.5:
        unlock_type = 'object-unlock-driven'
    elif avg_unlock < 2 and abs(avg_repeat) > 1:
        unlock_type = 'repeat-sensitive'
    else:
        unlock_type = 'mixed'
    print(f'  => Mechanism type: {unlock_type}')

    # Exp3: Shared/Private Causal
    print(f'\n=== {model} Exp3: Shared/Private Causal ===')
    exp3 = data.get('exp3_causal_injection', {})
    inj = exp3.get('injection_results', {})

    print(f'  Layer | shared->cat | private->cat | private/cat_ratio | shared->color | private->color')
    for layer_key in sorted(inj.keys(), key=lambda x: int(x[1:])):
        lr = inj[layer_key]
        s = lr.get('shared_only', {})
        p = lr.get('private_only', {})
        sc = s.get('cat_delta', 0) or 0
        pc = p.get('cat_delta', 0) or 0
        scol = s.get('color_delta', 0) or 0
        pcol = p.get('color_delta', 0) or 0
        ratio = abs(pc) / (abs(sc) + abs(pc) + 1e-6)
        print(f'  {layer_key:5s} | {sc:+.3f}      | {pc:+.3f}       | {ratio:.3f}             | {scol:+.3f}      | {pcol:+.3f}')

    # 判断哪个模型private逐渐主导
    if inj:
        first_key = sorted(inj.keys(), key=lambda x: int(x[1:]))[0]
        last_key = sorted(inj.keys(), key=lambda x: int(x[1:]))[-1]
        first_p = abs(inj[first_key].get('private_only', {}).get('cat_delta', 0) or 0)
        last_p = abs(inj[last_key].get('private_only', {}).get('cat_delta', 0) or 0)
        first_s = abs(inj[first_key].get('shared_only', {}).get('cat_delta', 0) or 0)
        last_s = abs(inj[last_key].get('shared_only', {}).get('cat_delta', 0) or 0)
        print(f'  => Private growth: first={first_p:.3f}, last={last_p:.3f}, growth={last_p/(first_p+1e-6):.2f}x')
        print(f'  => Shared decay: first={first_s:.3f}, last={last_s:.3f}, ratio={last_s/(first_s+1e-6):.2f}x')
