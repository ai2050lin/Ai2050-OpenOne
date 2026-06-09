import json
for model in ['qwen3','glm4','deepseek7b']:
    with open(f'results/phase415_fictional_objects/{model}_phase415.json') as f:
        data = json.load(f)
    print(f'\n=== {model} L0 baseline ===')
    for attr in ['temperature','speed','size']:
        fict = data['attributes'][attr]['fictional_objects']
        print(f'  {attr}:')
        for obj_name, obj_data in fict.items():
            bl = obj_data['L0_baseline']['expected_level']
            mx = obj_data['L0_baseline']['max_candidate']
            mp = obj_data['L0_baseline']['max_prob']
            print(f'    {obj_name}: level={bl:.3f} max={mx}({mp:.3f})')
