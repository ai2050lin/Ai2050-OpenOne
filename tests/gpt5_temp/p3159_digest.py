# -*- coding: utf-8 -*-
# 3159 digest: extract key numbers from result.json / result_summary.json
import json, os

RDIR = os.path.join(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913',
                    'phase3159', 'g4p2_equivalence_dynamics')
lines = []
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    r = json.load(open(os.path.join(RDIR, m, 'result.json'), encoding='utf-8'))
    bm = r['budget_median']
    pt = r['passthrough_mid_over_kout']
    ree = r['re_emergence']
    am = r['anchor_meta']
    det = r['det']
    lines.append('%s: L_mid=%d ratio=%.3f med[null=%.3f rand=%.3f top=%.3f] cens=%s' % (
        m, r['model']['L_mid'], r['ratio_mid'], bm['null'], bm['rand'], bm['top'],
        {k: round(v, 2) for k, v in r['censored_frac'].items()}))
    lines.append('  re: bot %.4f->%.4f (gain %.4f) | rand gain %.4f | top gain %.4f | pnorm %.1f-%.1f' % (
        ree['share_mid_bottom'], ree['share_NL_bottom'], ree['re_gain_bottom'],
        ree['re_gain_rand'], ree['re_gain_top'],
        min(a['pnorm'] for a in am), max(a['pnorm'] for a in am)))
    lines.append('  passthrough mid/kout: null %.2f rand %.2f top %.2f | ctx anchors %d/16' % (
        pt['null'], pt['rand'], pt['top'], sum(1 for a in am if a['ctx'] == 1)))
    lines.append('  det: base1_bitwise=%s anchor3157_bitwise=%s b16rel=%.2e pre_slot=%.2e orth=%.1e' % (
        det['base1_bitwise'], det['anchor_bitwise_3157'],
        det['batch6_vs_batch1_rel_max'], det['pre_slot_rel_max'], det['dir_orth']))
    lines.append('  res=%s seal=%s npz=%s runtime=%ss' % (
        r['res_sha8'], r['seal_sha8'], r['npz_sha8'], r['runtime_s']))
s = json.load(open(os.path.join(RDIR, 'summary', 'result_summary.json'), encoding='utf-8'))
lines.append('SUMMARY: %s' % s['verdict'])
lines.append('  kl_pairs=%s' % {k: round(v, 4) for k, v in s['kl_pairs'].items()})
lines.append('  re_pairs=%s' % {k: round(v, 4) for k, v in s['re_pairs'].items()})
open(os.path.join(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp', 'p3159_digest.txt'),
     'w', encoding='utf-8').write('\n'.join(lines))
print('written')
