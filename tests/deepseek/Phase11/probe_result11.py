# -*- coding: utf-8 -*-
"""Phase 11 结构探针：把 result_phase11.json 的键/类型/形状 dump 到 txt 供 Read。"""
import json, os, sys
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11', 'result_phase11.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11', '_probe_result11.txt')
lines = []
R = json.load(open(SRC, encoding='utf-8'))
lines.append('TOP KEYS (%d): %s' % (len(R), sorted(R.keys())))
lines.append('')


def fmt(v, depth=0):
    if isinstance(v, dict):
        if depth >= 2:
            return 'dict(k=%d, keys=%s)' % (len(v), list(v.keys())[:14])
        return 'dict(k=%d)' % len(v)
    if isinstance(v, list):
        a = np.asarray(v)
        return 'list(n=%d) np.shape=%s np.dtype=%s head=%s' % (len(v), a.shape, a.dtype, a.ravel()[:6].tolist())
    return '%r' % (v,)


for k in ['E0', 'E0b', 'full_L6', 'bit_replication', 'dose_coord', 'verdict', 'elapsed_s', 'drift_flags']:
    lines.append('== %s :: %s' % (k, fmt(R.get(k))))
    lines.append('   raw=%s' % (str(R.get(k))[:600],))
    lines.append('')

for k in ['profile_abs', 'profile_rel', 'profile_R_ext', 'E1_pairs', 'E1b_pairs', 'E3_pairs', 'E3', 'E3_verdict',
          'bootstrap_band', 'permutation_null', 'E4', 'E5', 'floors']:
    v = R.get(k)
    lines.append('== %s :: %s' % (k, fmt(v)))
    if isinstance(v, dict):
        for kk in sorted(v.keys()):
            lines.append('     .%s :: %s' % (kk, fmt(v[kk])))
    elif isinstance(v, list):
        if v and isinstance(v[0], dict):
            lines.append('     item0 keys=%s' % (sorted(v[0].keys()),))
            lines.append('     item0=%s' % (json.dumps(v[0], ensure_ascii=False)[:500],))
            if len(v) > 1:
                lines.append('     item1=%s' % (json.dumps(v[1], ensure_ascii=False)[:500],))
    lines.append('')

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines))
print('WROTE', OUT, len(lines), 'lines')
