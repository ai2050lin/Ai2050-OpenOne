# -*- coding: utf-8 -*-
"""p3157 digest: extract key fields from four results"""
import json

RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3157\g2p2_transform_algebra_commutator'
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4']
lines = []
rs = {}
for m in MODELS:
    r = json.load(open(RDIR + '\\' + m + '\\result.json', encoding='utf-8'))
    rs[m] = r
    ko = r['anova_kout']
    lines.append('%s: KOUT[E=%.3f R=%.3f N=%.3f C=%.3f ExR=%.3f ExN=%.3f ExC=%.3f RxN=%.4f RxC=%.3f NxC=%.3f resid=%.3f] rxn_argmax=L%d' % (
        m, ko['E'], ko['R'], ko['N'], ko['C'], ko['ExR'], ko['ExN'], ko['ExC'],
        ko['RxN'], ko['RxC'], ko['NxC'], ko['residual'], r['rxn_argmax_layer']))
    lines.append('  exch: C0 R=%.3f N=%.3f | C1 R=%.3f N=%.3f | null95=%.3f sig=%s | tc_cos=%.3f | det=%s npz=%s res=%s seal=%s rt=%ss' % (
        r['exch']['exchR_c0'], r['exch']['exchN_c0'], r['exch']['exchR_c1'], r['exch']['exchN_c1'],
        r['shuffle_null95'], r['exch_significant'], r['tc_mean_cos'],
        r['determinism_note'][:12], r['npz_sha8'], r['res_sha8'], r['seal_sha8'], r['runtime_s']))
s = json.load(open(RDIR + '\\summary\\result_summary.json', encoding='utf-8'))
lines.append('SUMMARY: %s' % s['verdict'])
lines.append('  fp: %s' % {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in s['fp_pairs'].items()})
lines.append('  shares_mean_kout: %s' % {k: round(v, 4) for k, v in s['shares_mean_kout'].items()})
lines.append('  summary res=%s seal=%s' % (s['res_sha8'], s['seal_sha8']))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3157_digest.txt', 'w', encoding='utf-8') as f:
    f.write(chr(10).join(lines))
print('written')
