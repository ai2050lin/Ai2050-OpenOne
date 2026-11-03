# -*- coding: utf-8 -*-
"""Q8-D 剂量标定：‖u_freq‖ 到底多大，δ 一个 zipf 单位相当于 token 范数的几成。"""
import os, sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import pat_q8b as B

OUT = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUT.append(s)

for name, p in (('Qwen', 'D:/develop/model/Qwen/Qwen3-4B'),
                ('GLM', 'D:/develop/model/ZhipuAI/glm-4-9b-chat-hf')):
    B.MODEL_PATH = p
    tok, E, tied = B.load_E()
    pools = B.build_pools(tok, E)
    ic, zc = pools['char']; iw, zw = pools['word']
    idx = np.concatenate([ic, iw]); y = np.concatenate([zc, zw])
    mu = E.mean(0)
    X = (E[idx] - mu).astype(np.float32)
    yl = y - y.mean()
    G, b = B.gram_xy(X, yl)
    w = B.solve_w(G, b)
    ww = float(w @ w)
    u = w / ww
    nu = float(np.linalg.norm(u))
    nrm = np.linalg.norm(E[idx], axis=1)
    med = float(np.median(nrm))
    # 线性预测的 zipf 标准差（检验 δ 与预测尺度是否同一量纲）
    t = X @ w
    say('%s d=%d n=%d  ‖w‖=%.4f  ‖u‖=%.5f' % (name, E.shape[1], len(idx),
                                              float(np.linalg.norm(w)), nu))
    say('   median‖e‖=%.4f  mean‖e‖=%.4f  ‖μ‖=%.4f' % (med, float(nrm.mean()),
                                                      float(np.linalg.norm(mu))))
    say('   线性预测 t: std=%.3f  min=%.2f max=%.2f（真 zipf 范围 %.2f–%.2f）'
        % (float(t.std()), float(t.min()), float(t.max()), float(y.min()), float(y.max())))
    for d in (0.25, 0.5, 1.0, 2.0):
        say('   δ=%.2f zipf -> 位移 %.4f = %.1f%% of median‖e‖'
            % (d, d * nu, 100.0 * d * nu / med))
    say('   位移达 median‖e‖ 的 10%%/20%%/40%% 对应 δ = %.2f / %.2f / %.2f'
        % (0.1 * med / nu, 0.2 * med / nu, 0.4 * med / nu))
    del E, X, G, b
    import gc; gc.collect()

Path('tests/gpt5_temp/q8d_scale.txt').write_text('\n'.join(OUT) + '\n',
                                                 encoding='utf-8', newline='')
say('WROTE')
