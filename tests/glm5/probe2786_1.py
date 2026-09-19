# -*- coding: utf-8 -*-
import sys

import numpy as np

B = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
P2761 = B + r'\phase2761\qwen4_kc_fault'
P2763 = B + r'\phase2763\qwen4_debias_repair'
P2774 = B + r'\phase2774\qwen4_pull_validation'

z61 = np.load(P2761 + r'\fault_scores.npz', allow_pickle=False)
wrong_idx = np.array(sorted(int(i) for i in z61['wrong_idx']))
zbd = np.load(P2763 + r'\bias_dirs.npz', allow_pickle=False)
v_rows, v_norms = zbd['v_rows'], zbd['v_norms']
zp = np.load(P2774 + r'\pull_stats.npz', allow_pickle=False)
pull, bsub_ok = zp['pull'], zp['bsub'].astype(bool)

sel = [k for k in range(len(wrong_idx)) if pull[k] < 0 and bool(bsub_ok[k])]
row_ids = [int(wrong_idx[k]) for k in sel]
V = np.stack([v_rows[i] / v_norms[i] for i in row_ids])
U, S, Vh = np.linalg.svd(V.astype(np.float64), full_matrices=False)
e0 = Vh[0] / np.linalg.norm(Vh[0])

# rebuild model head only? too heavy. Instead load W_U from safetensors
# via safetensors lib on the 4B shard (tied lm_head = embed_tokens? not
# for qwen4? 2775 used model.lm_head). Use safe_open on shards.
from safetensors.torch import safe_open
import glob, json
mdir = r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-4B'
cfg = json.load(open(mdir + r'\config.json', encoding='utf-8'))
tied = cfg.get('tie_word_embeddings', False)
print('tied', tied)
W_U = None
for shard in sorted(glob.glob(mdir + r'\*.safetensors')):
    with safe_open(shard, framework='pt') as f:
        if 'lm_head.weight' in f.keys():
            W_U = f.get_tensor('lm_head.weight').float().numpy().astype(np.float64)
            print('lm_head from', shard.split('\\')[-1])
            break
        if tied and 'model.embed_tokens.weight' in f.keys():
            W_U = f.get_tensor('model.embed_tokens.weight').float().numpy().astype(np.float64)
            print('embed (tied) from', shard.split('\\')[-1])
            break
assert W_U is not None

import phase2747_rdc_material as mat
material, data = mat.freeze()
crows = [x for x in data['diagnostic'] if x['kind'] == 'controlled_relation']
tgt = {i: crows[i]['target'] for i in row_ids}
rivals = {int(wrong_idx[k]): int(zp['rival_ids'][k]) for k in sel}

cos_u_ro = []
cos_v = []
cos_riv = []
cos_tgt = []
for k, i in enumerate(row_ids):
    v = V[k]
    u_ro = W_U[tgt[i]] - W_U[rivals[i]]
    u_ro /= np.linalg.norm(u_ro)
    cos_u_ro.append(float(v @ u_ro))
    cos_v.append(float(v @ e0))
    cos_riv.append(float(e0 @ W_U[rivals[i]] /
                         np.linalg.norm(W_U[rivals[i]])))
    cos_tgt.append(float(e0 @ W_U[tgt[i]] /
                         np.linalg.norm(W_U[tgt[i]])))
print('cos(e0, v_row): mean %.4f  min %.4f  max %.4f'
      % (np.mean(cos_v), np.min(cos_v), np.max(cos_v)))
print('cos(v_row, u_ro): mean %.4f (this is pull/|u||v| sanity)'
      % np.mean(cos_u_ro))
print('cos(e0, W_U[rival]): mean %.4f  min %.4f  max %.4f'
      % (np.mean(cos_riv), np.min(cos_riv), np.max(cos_riv)))
print('cos(e0, W_U[target]): mean %.4f  min %.4f  max %.4f'
      % (np.mean(cos_tgt), np.min(cos_tgt), np.max(cos_tgt)))
# per-projection weights: v.e0 (how much of each row sits on e0)
w = V @ e0
print('v.e0 weights: mean %.4f min %.4f max %.4f'
      % (w.mean(), w.min(), w.max()))
# is e0 close to mean of v rows?
vm = V.mean(axis=0)
vm /= np.linalg.norm(vm)
print('cos(e0, mean_v) %.4f' % float(e0 @ vm))
# energy of e0 within each row: (v.e0)^2 vs |v|^2=1
print('energy share per row: mean %.4f min %.4f'
      % (float((w ** 2).mean()), float((w ** 2).min())))
