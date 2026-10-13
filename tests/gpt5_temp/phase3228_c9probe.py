# -*- coding: utf-8 -*-
"""Phase 3228 C9 probe: is cached entry NH-1 the POST-NORM pre-logit state?"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94

OUT = []
from safetensors import safe_open
from tokenizers import Tokenizer
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
SNAP4 = glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-4B-Base', 'snapshots', '*'))[0]
SNAP14 = glob.glob(os.path.join(HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色', '乐器', '天气', '运动', '电器']

for m, snap, wkey, D, fin in (('qwen3-4b', SNAP4, 'model.embed_tokens.weight', 2560, 36),
                              ('qwen3-14b', SNAP14, 'lm_head.weight', 5120, 40)):
    tk = Tokenizer.from_file(os.path.join(snap, 'tokenizer.json'))
    cls_tok = [tk.encode(c, add_special_tokens=False).ids[0] for c in CLASSES]
    cfg = json.load(open(os.path.join(snap, 'config.json'), encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))
    if m == 'qwen3-4b':
        with safe_open(os.path.join(snap, 'model-00001-of-00003.safetensors'), framework='pt') as f:
            W = f.get_tensor('model.embed_tokens.weight').float().numpy()
        with safe_open(os.path.join(snap, 'model-00003-of-00003.safetensors'), framework='pt') as f:
            gamma = f.get_tensor('model.norm.weight').float().numpy().reshape(-1)
    else:
        with safe_open(os.path.join(snap, 'model-00008-of-00008.safetensors'), framework='pt') as f:
            W = f.get_tensor('lm_head.weight').float().numpy()
            gamma = f.get_tensor('model.norm.weight').float().numpy().reshape(-1)
    z = np.load(p94.SRC['p3169_4b' if m == 'qwen3-4b' else 'p3169_14b'])
    MARG = z['marg'].reshape(2190, 10)
    Hfin = z['H'][:, :, fin, :].reshape(2190, D).astype(np.float32)
    Hpre = z['H'][:, :, fin - 1, :].reshape(2190, D).astype(np.float32)
    del z
    # RMS of the cached final states (post-norm should be O(gamma scale), pre-norm O(10+))
    OUT.append('%s: RMS(h[%d]) mean=%.4f | RMS(h[%d]) mean=%.4f'
               % (m, fin, float(np.sqrt((Hfin ** 2).mean(1)).mean()),
                  fin - 1, float(np.sqrt((Hpre ** 2).mean(1)).mean())))

    def lnf(X):
        return (X / np.sqrt((X ** 2).mean(axis=-1, keepdims=True) + eps)) * gamma

    c_direct = Hfin @ W[cls_tok].T           # NO lnf
    c_double = lnf(Hfin) @ W[cls_tok].T      # lnf applied (old map)
    c_frompre = lnf(Hpre) @ W[cls_tok].T     # lnf of the pre-norm last-layer INPUT slot
    am_c = MARG.argmax(1)
    for tag, c in (('direct W@h_fin', c_direct), ('lnf(h_fin) double', c_double),
                   ('lnf(h_fin-1)', c_frompre)):
        am = c.argmax(1)
        g = np.sort(c, axis=1)
        mygap = g[:, 9] - g[:, 8]
        OUT.append('  %s: argmax agree=%.4f | centered-cos=%.4f'
                   % (tag, float((am == am_c).mean()),
                      float(np.mean([float(np.corrcoef(c[i] - c[i].mean(),
                                                        MARG[i])[0, 1]) for i in range(0, 2190, 8)]))))

with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3228_c9_out.txt', 'w',
          encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('c9 probe done')
