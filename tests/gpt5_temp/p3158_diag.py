# -*- coding: utf-8 -*-
# p3158_diag.py: 协议 logits vs LG 逐变体诊断（3156 row zh/A/k0）
import os, json
import numpy as np
import torch
from safetensors import safe_open

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R6 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                  'phase3156', 'g3p1_position_shift_family')
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
z6 = np.load(os.path.join(R6, 'qwen3-4b', 'collect.npz'))
H6 = z6['H']
LG6 = z6['LG']
langs = [str(x) for x in z6['lang']]
arms = [str(x) for x in z6['arm']]
ks = [int(x) for x in z6['k']]
ntg = [int(x) for x in z6['n_tgt']]
IDX = {(langs[s], arms[s], ks[s]): s for s in range(len(langs))}
s = IDX[('zh', 'A', 0)]
h = H6[s, ntg[s] - 1, 36].astype(np.float64)
lg = LG6[s].astype(np.float64)
with safe_open(os.path.join(MDIR, 'model-00001-of-00003.safetensors'), framework='pt') as f:
    W = f.get_tensor('model.embed_tokens.weight')
with safe_open(os.path.join(MDIR, 'model-00003-of-00003.safetensors'), framework='pt') as f:
    gamma = f.get_tensor('model.norm.weight').float().numpy().astype(np.float64)
Wt = W.to('cuda', torch.float32)
out = []
out.append('h: norm=%.3f rms=%.3f max=%.1f  lg: norm=%.3f max=%.2f argmax=%d' % (
    np.linalg.norm(h), np.sqrt((h ** 2).mean()), np.abs(h).max(),
    np.linalg.norm(lg), np.abs(lg).max(), int(np.argmax(lg))))
def proto(vec):
    v = torch.from_numpy(vec.astype(np.float32)).cuda()
    lp = (v @ Wt.T).cpu().numpy().astype(np.float64)
    return lp
rms = np.sqrt((h ** 2).mean() + 1e-6)
variants = {
    'norm+gamma': h / rms * gamma,
    'norm_only': h / rms,
    'raw': h,
    'norm+gamma_f32eps': h / np.sqrt((h ** 2).mean() + 1e-6) * gamma,
}
for name, p in variants.items():
    lp = proto(p)
    cos = float(lp @ lg / (np.linalg.norm(lp) * np.linalg.norm(lg) + 1e-18))
    out.append('%-18s |lp|=%9.2f cos_vs_LG=%.4f argmax=%d top1_hit=%s maxdiff=%.2f' % (
        name, np.linalg.norm(lp), cos, int(np.argmax(lp)),
        int(np.argmax(lp)) == int(np.argmax(lg)), np.abs(lp - lg).max()))
out.append('LG top5: %s' % np.argsort(lg)[::-1][:5].tolist())
lp0 = proto(variants['norm+gamma'])
out.append('proto top5: %s' % np.argsort(lp0)[::-1][:5].tolist())
out.append('LG[:8]=%s' % np.round(lg[:8], 3).tolist())
out.append('proto[:8]=%s' % np.round(lp0[:8], 3).tolist())
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3158_diag.txt'), 'w', encoding='utf-8').write(chr(10).join(out))
print('written')
