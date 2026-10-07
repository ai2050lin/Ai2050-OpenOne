# -*- coding: utf-8 -*-
"""补丁 16：present 的 P9 域分解块改用 6 位小数（使 0.006171 vs 0.003713 / P16 标定值的匹配可见）。"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D20 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')
P = os.path.join(D20, 'gen_present_phase20.py')
s = io.open(P, encoding='utf-8').read()
LOG = []


def rep(old, new, tag, count=1):
    global s
    n = s.count(old)
    assert n == count, '[%s] 期望 %d 处，实为 %d' % (tag, count, n)
    s = s.replace(old, new)
    LOG.append('  OK  %s' % tag)


rep("+ fp(PH['p16_calib_max_abs_dxh']) + '</code>）。<br>'",
    "+ f(PH['p16_calib_max_abs_dxh'], 6) + '</code>）。<br>'", 'E2 note 6dp')

rep("% (k.replace('|', ' | '), fp(r['as_coded_max_abs_dxh']),\n"
    "                pill(r['as_coded_pass'], 'PASS', 'FAIL'), fp(r['reach_domain_max_abs_dxh']),\n"
    "                pill(r['reach_domain_pass'], 'PASS', 'FAIL'), fp(r['shallow_max_abs_dxh']),\n",
    "% (k.replace('|', ' | '), f(r['as_coded_max_abs_dxh'], 6),\n"
    "                pill(r['as_coded_pass'], 'PASS', 'FAIL'), f(r['reach_domain_max_abs_dxh'], 6),\n"
    "                pill(r['reach_domain_pass'], 'PASS', 'FAIL'), f(r['shallow_max_abs_dxh'], 6),\n",
    'E2 表 6dp')

rep("+ fp(PHD['A0_nf4|A0_bf16']['as_coded_max_abs_dxh'])\n",
    "+ f(PHD['A0_nf4|A0_bf16']['as_coded_max_abs_dxh'], 6)\n", 'E3 note as-coded 6dp')

rep("+ fp(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh']) + '</code>，比 &ell;=1 小 '",
    "+ f(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 6) + '</code>，比 &ell;=1 小 '",
    'E3 note reach 6dp')

rep("'（A0 <code>' + fp(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh']) + '</code> / A1 <code>'\n"
    "         + fp(PHD['A1_nf4|A1_bf16']['reach_domain_max_abs_dxh']) + '</code>，两对皆 PASS，'",
    "'（A0 <code>' + f(PHD['A0_nf4|A0_bf16']['reach_domain_max_abs_dxh'], 6) + '</code> / A1 <code>'\n"
    "         + f(PHD['A1_nf4|A1_bf16']['reach_domain_max_abs_dxh'], 6) + '</code>，两对皆 PASS，'",
    'E3 note A0/A1 6dp')

rep("+ fp(PH['p16_calib_max_abs_dxh']) + '</code>）。<br>'",
    "+ f(PH['p16_calib_max_abs_dxh'], 6) + '</code>）。<br>'", 'E2 note 6dp (dup idempotent)',
    0) if False else None

io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
io.open(os.path.join(D20, '_patch_p20_16_present6dp.log'), 'w', encoding='utf-8', newline='\n')\
    .write('\n'.join(LOG) + '\n')
print('\n'.join(LOG))
print('patched:', len(LOG))
