# -*- coding: utf-8 -*-
# 3160 探针: 确认 (a) 3159 npz 键与 SHARE 形状/分位统计; (b) 3156 result/npz 中
#          rank-1 massive 轴的存储形态; (c) 3157 H 可用性。结果写 txt 供 Read。
import os, json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_probe_out.txt')
out = []

def dump_npz(tag, p):
    z = np.load(p, allow_pickle=False)
    out.append('[%s] %s' % (tag, os.path.relpath(p, ROOT)))
    for k in z.files:
        a = z[k]
        out.append('  %-22s %-14s %s' % (k, str(a.shape), a.dtype))

# ---- 3159: SHARE 曲线 (top 臂 = index 2) ----
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    d9 = os.path.join(RDIR, 'phase3159', 'g4p2_equivalence_dynamics', m)
    dump_npz('3159:' + m, os.path.join(d9, 'collect.npz'))
    r9 = json.load(open(os.path.join(d9, 'result.json'), encoding='utf-8'))
    z9 = np.load(os.path.join(d9, 'collect.npz'))
    SHARE = z9['SHARE'].astype(np.float64)
    L_MID = int(z9['l_mid'])
    top_share = SHARE[2].mean(axis=(0, 1, 2))       # top 臂逐层均值 (NH,)
    null_share = SHARE[0].mean(axis=(0, 1, 2))
    out.append('  L_mid=%d top_share[L_mid:L_mid+6]=%s' % (
        L_MID, np.round(top_share[L_MID:L_MID + 6], 4).tolist()))
    out.append('  top_share[NL-3:]=%s null_share[L_mid:L_mid+3]=%s' % (
        np.round(top_share[-3:], 4).tolist(), np.round(null_share[L_MID:L_MID + 3], 4).tolist()))
    # 分位曲线: top 臂逐层分位 (over anchors x dirs x alphas)
    q = np.percentile(SHARE[2].reshape(-1, SHARE.shape[4]).T,
                      [10, 50, 90], axis=1)  # (3q, NH)
    out.append('  top q10/q50/q90 at L_mid..L_mid+4: %s' % np.round(q[:, L_MID:L_MID + 5], 4).tolist())

# ---- 3156: rank-1 massive 轴 ----
found6 = {}
for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, 'tests', 'glm5', 'result')):
    for fn in filenames:
        if 'phase3156' in dirpath and fn in ('result.json', 'collect.npz'):
            found6.setdefault(fn, []).append(os.path.join(dirpath, fn))
out.append('[3156 files] %d result.json, %d collect.npz' %
           (len(found6.get('result.json', [])), len(found6.get('collect.npz', []))))
for p in found6.get('result.json', []):
    out.append('  %s' % os.path.relpath(p, ROOT))
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    cands = [p for p in found6.get('result.json', []) if m in p]
    if not cands:
        out.append('[3156:%s] NOT FOUND' % m)
        continue
    r6 = json.load(open(cands[0], encoding='utf-8'))
    out.append('[3156:%s] result keys: %s' % (m, sorted(r6.keys())))
    def walk(o, pre=''):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    walk(v, pre + k + '.')
                elif isinstance(v, (int, float)) and ('rank1' in k.lower() or 'massive' in k.lower()):
                    out.append('  %s%s = %s' % (pre, k, v))
        elif isinstance(o, list):
            pass
    walk(r6)
    ncands = [p for p in found6.get('collect.npz', []) if m in p]
    if ncands:
        dump_npz('3156:' + m, ncands[0])
    else:
        out.append('[3156:%s] collect.npz NOT FOUND' % m)

# ---- 3157 H 可用性 ----
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    z7 = np.load(os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator', m, 'collect.npz'))
    out.append('[3157:%s] H %s %s' % (m, z7['H'].shape, z7['H'].dtype))

with open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('PROBE DONE')
