import json
for model in ['qwen3','glm4','deepseek7b']:
    with open(f'results/phase415_fictional_objects/{model}_phase415.json') as f:
        data = json.load(f)
    print(f'\n=== {model} ===')
    for attr in ['temperature','speed','size']:
        asym = data['attributes'][attr]['asymmetry_analysis']
        print(f'  {attr}:')
        for rl in ['L1_mild','L2_definition','L4_qa']:
            a = asym[rl]
            ra = a['real_asymmetry']
            fa = a['fictional_asymmetry']
            d = a['asymmetry_difference']
            print(f'    {rl}: real={ra:+.3f} fict={fa:+.3f} diff={d:+.3f}')
