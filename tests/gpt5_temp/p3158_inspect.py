# -*- coding: utf-8 -*-
# p3158_inspect.py: 3157/3156 npz 结构 + safetensors 张量位置探查
import os, json, glob
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R7 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                  'phase3157', 'g2p2_transform_algebra_commutator')
R6 = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                  'phase3156', 'g3p1_position_shift_family')
out = []

def npz_info(path, name):
    if not os.path.exists(path):
        out.append('%s: MISSING %s' % (name, path))
        return None
    z = np.load(path, allow_pickle=False)
    lines = ['%s: %s' % (name, os.path.basename(path))]
    for k in z.files:
        a = z[k]
        lines.append('  %s %s %s' % (k, a.shape, a.dtype))
    out.extend(lines)
    return z

for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    p7 = os.path.join(R7, m, 'collect.npz')
    if m == 'qwen3-4b':
        npz_info(p7, '3157/' + m)
        r7 = json.load(open(os.path.join(R7, m, 'result.json'), encoding='utf-8'))
        out.append('3157 %s: readout=%s nl=%s res=%s' % (m, r7.get('readout'), r7.get('nl'), r7.get('res_sha8')))
        p6 = None
        for cand in ('qwen3-4b', os.path.join('qwen3-4b')):
            for fn in ('collect.npz',):
                q = os.path.join(R6, 'qwen3-4b', fn)
                if os.path.exists(q):
                    p6 = q
        if p6:
            npz_info(p6, '3156/qwen3-4b')
        else:
            out.append('3156/qwen3-4b collect.npz MISSING; listing dir:')
            base6 = os.path.join(R6, 'qwen3-4b')
            for f in os.listdir(base6):
                out.append('  3156 dir: %s' % f)
            sm = os.path.join(base6, 'smoke')
            if os.path.isdir(sm):
                for f in os.listdir(sm):
                    out.append('  3156 smoke: %s' % f)
    else:
        npz_info(p7, '3157/' + m)

# 模型目录与 lm_head / norm 张量位置
try:
    import torch
    out.append('torch %s cuda=%s' % (torch.__version__, torch.cuda.is_available()))
except Exception as e:
    out.append('torch import fail: %r' % e)

from safetensors import safe_open
MD = {
    'qwen3-4b': r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-4B',
    'qwen3-14b': r'D:\AI2050\Ai2050-OpenOne\models\hf\Qwen3-14B',
    'glm4': r'D:\AI2050\Ai2050-OpenOne\models\hf\glm-4-9b-hf',
}
# 修正: 先探测真实目录
for k in list(MD):
    if not os.path.isdir(MD[k]):
        out.append('MODEL DIR MISSING: %s -> %s' % (k, MD[k]))
cfg = json.load(open(os.path.join(ROOT, 'tests', 'glm5', 'phase3157_g2p2_transform_algebra.py'), encoding='utf-8')) if False else None

out.append('--- model dirs probe ---')
import re
src = open(os.path.join(ROOT, 'tests', 'glm5', 'phase3157_g2p2_transform_algebra.py'), encoding='utf-8').read()
mm = re.search(r'MODEL_MAP\s*=\s*\{(.*?)\}', src, re.S)
out.append('MODEL_MAP raw: %s' % (mm.group(1)[:400] if mm else 'NOT FOUND'))

open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3158_inspect.txt'), 'w', encoding='utf-8').write(chr(10).join(out))
print('written')
