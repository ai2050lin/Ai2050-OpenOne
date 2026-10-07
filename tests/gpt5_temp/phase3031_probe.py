# -*- coding: utf-8 -*-
"""3030 前置盘点: 列出 3028/3029/3022/3020 npz 的键与形状 (3031 预注册输入)."""
import numpy as np
import json
import io
import os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
TARGETS = [
    ('phase3028', 'omega_p2v_dose_symmetry_qwen'),
    ('phase3029', 'omega_p2w_recruitment_decomp_qwen'),
    ('phase3022', 'omega_p2p_l3_relay_neurons_qwen'),
    ('phase3020', 'omega_p2n_readout_specificity_qwen'),
]

out = []
for ph, name in TARGETS:
    p = os.path.join(BASE, ph, name, name + '.npz')
    z = np.load(p, allow_pickle=True)
    out.append('=== %s %s ===' % (ph, name))
    for k in z.files:
        a = z[k]
        try:
            out.append('  %-40s %-18s %s' % (k, str(a.shape), str(a.dtype)))
        except Exception as e:
            out.append('  %s ERR %s' % (k, e))

# 3028 result.json 逐 tag 字段
rp = os.path.join(BASE, 'phase3028', 'omega_p2v_dose_symmetry_qwen', 'result.json')
r = json.load(io.open(rp, encoding='utf-8'))
out.append('=== 3028 result.json top keys ===')
out.append(str(list(r.keys())))

rep = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3031_probe.txt'
with io.open(rep, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('WROTE', rep)
