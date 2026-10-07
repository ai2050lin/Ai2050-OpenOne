# -*- coding: utf-8 -*-
"""Phase 11 结构探针 2：E1/E1b/E3 rows、band 子字典、锚字段。"""
import json, os
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11', 'result_phase11.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11', '_probe_result11b.txt')
R = json.load(open(SRC, encoding='utf-8'))
L = []


def show(tag, v, maxlen=900):
    L.append('== %s :: %s' % (tag, type(v).__name__))
    L.append('   ' + json.dumps(v, ensure_ascii=False, default=str)[:maxlen])
    L.append('')


for k in ['E1', 'E1b', 'E6', 'E5', 'base', 'decisions', 'subspace', 'sites', 'panel', 'x_star_rel',
          'off_manifold_alphas', 'full_ref_phase9', 'layers', 'phase', 'name', 'model', 'smoke',
          'exec_sha8', 'seal_sha8', 'E3_verdict']:
    v = R.get(k)
    if isinstance(v, dict):
        L.append('== %s :: dict(k=%d) keys=%s' % (k, len(v), list(v.keys())[:24]))
        for kk in list(v.keys())[:2]:
            show('%s[%s]' % (k, kk), v[kk])
    else:
        show(k, v)

# E3 rows 单例
if isinstance(R.get('E3'), dict):
    k0 = sorted(R['E3'].keys(), key=lambda z: int(z))[0]
    show('E3[%s]' % k0, R['E3'][k0])

# bootstrap_band 子结构
bb = R.get('bootstrap_band', {})
for kk in ['abs', 'rel', 'own', 'rho_hat']:
    show('bootstrap_band.%s' % kk, bb.get(kk))
show('bootstrap_band.J_site_ci', {k: bb['J_site_ci'][k] for k in list(bb.get('J_site_ci', {}).keys())[:3]})
show('bootstrap_band.indistinguishable_pairs', bb.get('indistinguishable_pairs'))

# profile_abs 单例
pa = R.get('profile_abs', {})
k0 = '6'
show('profile_abs[%s]' % k0, pa.get(k0))

# Phase10 result
P10 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10', 'result_phase10.json')
L.append('== Phase10 result exists :: %s' % os.path.exists(P10))
if os.path.exists(P10):
    R10 = json.load(open(P10, encoding='utf-8'))
    L.append('   Phase10 keys = %s' % sorted(R10.keys()))
    E1k = R10.get('E1')
    if isinstance(E1k, dict):
        kk = list(E1k.keys())[:2]
        L.append('   Phase10.E1 keys=%d sample=%s' % (len(E1k), kk))
        for k2 in kk:
            show('Phase10.E1[%s]' % k2, E1k[k2])

open(OUT, 'w', encoding='utf-8').write('\n'.join(L))
print('WROTE', OUT, len(L), 'lines')
