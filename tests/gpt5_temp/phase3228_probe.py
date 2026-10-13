# -*- coding: utf-8 -*-
"""Phase 3228 pre-arm probe: verify on-disk objects for the compute+seal arm."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUT = []


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


import numpy as np

# 1) 3154 caches
for tag, rel in [('4b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\collect.npz'),
                 ('14b', r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\collect.npz')]:
    p = os.path.join(SRC_DIR, rel)
    z = np.load(p)
    OUT.append('3154 %s sha8=%s keys=%s' % (tag, sha8_file(p), sorted(z.files)))
    for k in z.files:
        OUT.append('  %s %s %s' % (k, z[k].shape, z[k].dtype))
    g = z['g']
    OUT.append('  g uniq=%s l uniq=%s s uniq=%s ti uniq=%s ci uniq=%s ho uniq=%s'
               % (np.unique(g), np.unique(z['l']), np.unique(z['s']),
                  np.unique(z['ti']), np.unique(z['ci']), np.unique(z['ho'])))

# 2) 3169 caches
for tag, rel in [('4b', r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
                 ('14b', r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz')]:
    p = os.path.join(SRC_DIR, rel)
    z = np.load(p)
    OUT.append('3169 %s sha8=%s keys=%s' % (tag, sha8_file(p), sorted(z.files)))
    for k in z.files:
        OUT.append('  %s %s %s' % (k, z[k].shape, z[k].dtype))

# 3) 3225 u* npz
p = os.path.join(SRC_DIR, r'phase3225\g5g25_a1_behavioral_band_compute\a1_behavioral_band.npz')
z = np.load(p)
OUT.append('3225 npz sha8=%s keys=%s' % (sha8_file(p), sorted(z.files)))
for k in sorted(z.files):
    OUT.append('  %s %s %s' % (k, z[k].shape, z[k].dtype))

# 4) 3183 anchors
r83 = json.load(open(os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'), encoding='utf-8'))
OUT.append('3183 res_sha8=%s seal=%s' % (r83['res_sha8'], r83['seal_sha8']))
for m in ('qwen3-4b', 'qwen3-14b'):
    OUT.append('  %s: %s' % (m, {k: r83['per_model'][m][k] for k in
                                 ('cos_rl', 'norm_dv', 'frac_full', 'pc_share', 'med_cos')}))

# 5) tokenizers: single-token check for shi/you + class first tokens
import glob
from tokenizers import Tokenizer
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
SNAP = {'qwen3-4b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base', 'snapshots', '*'))[0],
        'qwen3-14b': glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]}
for m, sp in SNAP.items():
    tk = Tokenizer.from_file(os.path.join(sp, 'tokenizer.json'))
    for w in ['是', '有', '是一种', '没有', '不是一种']:
        ids = tk.encode(w, add_special_tokens=False).ids
        OUT.append('tok %s %r -> %s' % (m, w, ids))
    CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色', '乐器', '天气', '运动', '电器']
    ct = [tk.encode(c, add_special_tokens=False).ids[0] for c in CLASSES]
    OUT.append('tok %s CLS_TOK=%s distinct=%s' % (m, ct, len(set(ct)) == 10))

# 6) ledger
LP = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
led = json.load(open(LP, encoding='utf-8'))
OUT.append('ledger n=%d post=%s last_phase=%s'
           % (len(led['measurements']), led['ledger_sha256_8'],
              led['measurements'][-1].get('phase')))

# 7) 3157 cache keys (direction recipe basis)
p = os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz')
z = np.load(p)
OUT.append('3157 4b sha8=%s keys=%s H=%s' % (sha8_file(p), sorted(z.files), z['H'].shape))

# 8) 3154/3169 result anchors for layer indices
r54 = json.load(open(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\result.json'), encoding='utf-8'))
OUT.append('3154 4b NL=%s hidden=%s readout=%s kstar=%s'
           % (r54['NL'], r54['hidden'], r54['readout'], r54['kstar']))
r54b = json.load(open(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle\qwen3-14b\result.json'), encoding='utf-8'))
OUT.append('3154 14b NL=%s hidden=%s readout=%s kstar=%s'
           % (r54b['NL'], r54b['hidden'], r54b['readout'], r54b['kstar']))
r69 = json.load(open(os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'), encoding='utf-8'))
for m in ('qwen3-4b', 'qwen3-14b'):
    OUT.append('3169 %s readout=%s H_shape=%s' % (m, r69['per_model'][m]['readout'], r69['per_model'][m]['H_shape']))

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3228_probe_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('probe done')
