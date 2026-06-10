import json
with open('results/phase429_layer_probe_direction/qwen3_phase429_r1.json') as f:
    d = json.load(f)

print('=== DIRECTION COMPARISON at alpha=4.0, category task ===')
for obj in d['per_object']:
    od = d['per_object'][obj]
    base = od['baselines']['category']['level']
    base_top = od['baselines']['category']['top']
    print(f'\n{obj} (base_level={base:.3f}, top={base_top}):')
    
    for dir_type in ['embed_dir', 'layer_probe', 'random']:
        key = f'category_{dir_type}'
        if key not in od['perturbations']:
            continue
        pd = od['perturbations'][key]
        
        for pos in ['obj', 'last']:
            if pos not in pd:
                continue
            for layer, curve in pd[pos].items():
                if '4.0' in curve:
                    c = curve['4.0']
                    delta = c['delta']
                    top = c['top']
                    fh = c['full_entropy']
                    conf = c['confidence']
                    print(f'  {dir_type}@{layer}/{pos} a=4: D={delta:+.3f} top={top[:5]} H={fh:.1f} c={conf:.3f}')

print('\n=== POSITION ROUTING: obj vs last (layer_probe, alpha=4.0) ===')
for obj in d['per_object']:
    od = d['per_object'][obj]
    key = 'category_layer_probe'
    if key not in od['perturbations']:
        continue
    pd = od['perturbations'][key]
    
    for pos in ['obj', 'last']:
        if pos not in pd:
            continue
        best_delta = 0
        best_layer = ''
        for layer, curve in pd[pos].items():
            if '4.0' in curve:
                c = curve['4.0']
                if abs(c['delta']) > abs(best_delta):
                    best_delta = c['delta']
                    best_layer = layer
    print(f'  {obj}: obj best|D| vs last best|D|')

print('\n=== LAYER-BY-LAYER (layer_probe, last pos, alpha=4.0, category) ===')
for obj in d['per_object']:
    od = d['per_object'][obj]
    key = 'category_layer_probe'
    if key not in od['perturbations']:
        continue
    pd = od['perturbations'].get(key, {})
    last_data = pd.get('last', {})
    obj_data = pd.get('obj', {})
    
    print(f'\n  {obj} (layer_probe):')
    for pos in ['obj', 'last']:
        data = obj_data if pos == 'obj' else last_data
        if not data:
            continue
        for layer in sorted(data.keys()):
            curve = data[layer]
            if '4.0' in curve:
                c = curve['4.0']
                delta = c['delta']
                top = c['top']
                fh = c['full_entropy']
                conf = c['confidence']
                print(f'    {layer}/{pos} a=4: D={delta:+.3f} top={top[:5]} H={fh:.1f} c={conf:.3f}')

print('\n=== DIRECTION SIMILARITY ===')
for sim_key in d.get('direction_similarity', {}):
    sim = d['direction_similarity'][sim_key]
    print(f'  {sim_key}:')
    for k, v in sorted(sim.items()):
        print(f'    {k}: {v:.4f}')
