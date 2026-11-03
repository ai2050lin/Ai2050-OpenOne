# -*- coding: utf-8 -*-
"""r2_rawfreq 折划分方差诊断：多种子 5 折，看主值与复核值是否都在波动范围内。"""
import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
from pathlib import Path
sys.path.insert(0, 'tools')
import pat_q8b as B

doc = json.load(open('results/2026-11-01/190627_pat_q8e.json', encoding='utf-8'))
for name, mp in (('qwen3-4b', 'D:/develop/model/Qwen/Qwen3-4B'),
                 ('qwen3-14b', 'D:/develop/model/Qwen/Qwen3-14B')):
    B.MODEL_PATH = mp; B.SMOKE = False
    B.RNG = np.random.default_rng(20261101)
    tok, E, _ = B.load_E()
    pools = B.build_pools(tok, E)
    ic, zc = pools['char']; iw, zw = pools['word']
    idx = np.concatenate([ic, iw]); z = np.concatenate([zc, zw])
    mu = E.mean(0); X = (E[idx]-mu).astype(np.float32)
    freq = np.power(10.0, z-9.0); yf = freq - freq.mean()
    vals = []
    for seed in range(5):
        B.RNG = np.random.default_rng(777+seed)
        folds = B.folds_of(len(idx))
        r2, _ = B.cv_fast(X, yf, folds, ns='%s|diag%d' % (name, seed))
        vals.append(float(r2))
    ref = doc['payload']['models'][name]['r2_rawfreq']
    print('%s: 5 seeds = %s  mean=%.4f range=%.3f | recorded=%.4f' %
          (name, ['%.3f'%v for v in vals], np.mean(vals), np.ptp(vals), ref))
    del E, X
    import gc; gc.collect()
