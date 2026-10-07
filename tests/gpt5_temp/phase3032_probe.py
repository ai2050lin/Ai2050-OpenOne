# -*- coding: utf-8 -*-
"""3032 预注册前探针: 深层带超额剖面形状 (3030 npz) + 3020 traj L31 次峰."""
import numpy as np
import io

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
out = []

z30 = np.load(BASE + r'\phase3030\omega_p2x_readout_convexity_qwen'
              r'\omega_p2x_readout_convexity_qwen.npz', allow_pickle=True)
tags = [str(t) for t in z30['tags']]
t0 = z30['traj_alpha0']
t1 = z30['traj_alpha1']
t2 = z30['traj_alpha2']
theta_l = z30['theta_l']
layers = np.arange(4, 36)
out.append('shapes %s %s %s theta %s' % (t0.shape, t1.shape, t2.shape, theta_l.shape))
excess = t2 - 2.0 * t1 + t0
counted = t2 > 2.0 * theta_l
# band indices: idx 0-3 = L4-7 seed, 4-16 = L8-20 mid, 17-30 = L21-34 deep, 31 = xfin
for i in range(11):
    e = excess[i]
    c = counted[i]
    deep_idx = [j for j in range(17, 31) if c[j]]
    late_idx = [j for j in range(27, 31) if c[j]]  # L31-34
    pos_deep = sum(max(e[j], 0.0) for j in range(17, 31) if c[j])
    pos_late = sum(max(e[j], 0.0) for j in range(27, 31) if c[j])
    lstar30 = int(z30['lstar'][i])
    # 局部峰检测: counted 层内的正超额序列是否存在内部极大值 (非端点)
    prof = [(j, float(e[j])) for j in deep_idx]
    peaks = []
    for a in range(1, len(prof) - 1):
        if prof[a][1] >= prof[a - 1][1] and prof[a][1] >= prof[a + 1][1] and prof[a][1] > 0:
            peaks.append(prof[a][0])
    out.append('t%-6s lstar30=%2d deepcounted=%s late_share=%.3f peaks=%s'
               % (tags[i], lstar30, str(deep_idx),
                  (pos_late / pos_deep if pos_deep > 0 else -1),
                  str(peaks)))

# 3020 traj 深层局部峰 (js_l 剖面, alpha=1 擦除链)
z20 = np.load(BASE + r'\phase3020\omega_p2n_readout_specificity_qwen'
              r'\omega_p2n_readout_specificity_qwen.npz', allow_pickle=True)
keys20 = [str(t) for t in z20['traj_keys_logic']]
tags20 = [str(t) for t in z20['tags']]
idx = [keys20.index(t) for t in tags20]
tr = z20['traj_logic'][idx, :]
out.append('--- 3020 js_l profile deep local max (idx>=17) ---')
for i in range(11):
    js = tr[i]
    seg = js[17:31]
    j = int(np.argmax(seg)) + 17
    is_peak = (j > 17 and j < 30 and js[j] >= js[j - 1] and js[j] >= js[j + 1])
    out.append('t%-6s deep_argmax_idx=%2d(L%d) interior_peak=%s js=%.5f'
               % (tags20[i], j, j + 4, is_peak, js[j]))

rep = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase3032_probe.txt'
io.open(rep, 'w', encoding='utf-8').write('\n'.join(out))
print('WROTE')
