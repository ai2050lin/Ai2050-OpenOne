import json
for model in ['qwen3','glm4','deepseek7b']:
    with open(f'results/phase416_neutral_control/{model}_phase416.json') as f:
        data = json.load(f)
    print(f'\n=== {model} ===')
    for attr in ['temperature','speed']:
        asym = data['attributes'][attr]['asymmetry']
        print(f'  {attr}:')
        print(f'    Real:      asym={asym["real"]["asymmetry"]:+.3f}')
        print(f'    Fictional: asym={asym["fictional"]["asymmetry"]:+.3f}')
        print(f'    Random:    asym={asym["random_token"]["asymmetry"]:+.3f}')
        print(f'    Knowledge anchoring: {asym["knowledge_anchoring_effect"]:+.3f}')
        print(f'    Embedding bias:      {asym["embedding_bias_effect"]:+.3f}')
        print(f'    Base asymmetry:      {asym["base_asymmetry"]:+.3f}')
