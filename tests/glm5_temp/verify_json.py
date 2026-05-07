import json
d = json.load(open('d:/ai2050/TransformerLens-Project/frontend/public/data/forward_pass_demo.json', 'r', encoding='utf-8'))
print(f'Total layers: {len(d["layers"])}')
for l in d['layers']:
    acts = [n["activation"] for n in l["neuron_activations"]]
    print(f'L{l["layer"]:2d}: {l["label"]:20s} neurons={len(acts)} acts={[round(a,2) for a in acts]}')
