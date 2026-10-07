# -*- coding: utf-8 -*-
"""探针：result.arms[*] 的键集 + disk_verify 依赖键的存在性。"""
import io
import os
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
OUT = os.path.join(P21T, '_probe_keys.txt')
o = []


def w(s=''):
    o.append(str(s))


R = json.load(io.open(os.path.join(P21T, 'result_phase21.json'), encoding='utf-8'))
o_ = []
def w2(s=''):
    o_.append(str(s))


NEED = ['n_fw', 'F4_dims_ok', 'F5_o_proj_ok', 'G1_core', 'G1_full', 'model', 'scheme', 'primary_layer',
        'M1', 'M2', 'M3', 'confirmation', 'E0_selfcheck', 'U']
for a in R['arm_order']:
    rec = R['arms'][a]
    w('--- %s ---' % a)
    w('  keys: %s' % sorted(rec.keys()))
    for k in NEED:
        w('  %-16s %s' % (k, 'OK' if k in rec else '** MISSING **'))
    if 'n_fw' in rec:
        w('  n_fw = %s' % rec['n_fw'])
    if 'M3' in rec:
        w('  M3 keys: %s' % sorted(rec['M3'].keys()))
    if 'M1' in rec and 'share_v' in rec['M1']:
        w('  M1.share_v type=%s len=%d' % (type(rec['M1']['share_v']).__name__, len(rec['M1']['share_v'])))
    w('')

w('top-level result keys: %s' % sorted(R.keys()))
w('calibration keys: %s' % sorted(R['calibration'].keys()))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('PROBE OK')
