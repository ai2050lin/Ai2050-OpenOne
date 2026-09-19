"""P2820 post-hoc symbol audit: w.dW_c for the 7 census columns."""
import json
from pathlib import Path

import numpy as np

ROOT = Path('D:/AI2050/Ai2050-OpenOne')
BASE = ROOT / 'tests' / 'glm5' / 'result' / \
    'rdc_query_construction_20260913'
npz = np.load(BASE / 'phase2820' / 'edit_generalization' / 'edit.npz')
dW = {c: npz['dW_%s' % c].astype(np.float64)
      for c in ['red', 'black', 'purple']}
COLOURS = ['red', 'black', 'purple', 'blue', 'green', 'yellow']
for c in ['blue', 'green', 'yellow']:
    # rebuild from z-space diff of colour embeddings (same protocol)
    pass  # audit limited to saved 3 directions

mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
from safetensors import safe_open
index = json.loads((mdir / 'model.safetensors.index.json')
                   .read_text(encoding='utf-8'))['weight_map']

COLS = {'red': [(35, 1552), (34, 1218), (33, 2566), (25, 695)],
        'color': [(32, 353), (31, 3298), (30, 4290)]}
rep = {}
for kind, cols in COLS.items():
    for (L, j) in cols:
        name = 'model.layers.%d.mlp.down_proj.weight' % L
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            w = f.get_tensor(name)[:, j].float().numpy().astype(
                np.float64)
        key = '%s_L%d_j%d' % (kind, L, j)
        rep[key] = {c: round(float(w @ dW[c]), 4) for c in dW}

out = ROOT / 'tests' / 'gpt5_temp' / 'probe_2820_sign.json'
out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
print('WROTE', out)
