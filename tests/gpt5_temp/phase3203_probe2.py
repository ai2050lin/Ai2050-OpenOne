# -*- coding: utf-8 -*-
"""Probe exact scalar field paths in sealed GAP-4 chain results."""
import io, json

SD = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
def load(ph, sub):
    return json.load(io.open(SD + '\\phase%s\\%s\\result.json' % (ph, sub), encoding='utf-8'))

r = load('3169', 'g5a6_oov_panel')
print('3169 keys:', list(r.keys()))
print('3169 arm/ratio fields:', {k: v for k, v in r.items() if not isinstance(v, (dict, list))})
for k in ('per_model', 'arm', 'summary'):
    if k in r: print('3169', k, json.dumps(r[k], ensure_ascii=False)[:400])

r = load('3172', 'g5a9_port_calibration')
print('---3172 keys:', list(r.keys()))
for k in ('per_model', 'arm'):
    if k in r: print('3172', k, json.dumps(r[k], ensure_ascii=False)[:400])

r = load('3175', 'g5a12_residual_arms')
print('---3175 keys:', list(r.keys()))
for k in ('per_model', 'arm'):
    if k in r: print('3175', k, json.dumps(r[k], ensure_ascii=False)[:400])
