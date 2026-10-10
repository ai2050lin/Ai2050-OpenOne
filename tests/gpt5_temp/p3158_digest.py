# -*- coding: utf-8 -*-
# p3158_digest.py: 提取 3158 结果关键字段
import json, os
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                 'phase3158', 'g4p1_output_equivalence_class')
lines = []
rs = {}
for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    r = json.load(open(os.path.join(R, m, 'result.json'), encoding='utf-8'))
    rs[m] = r
    p1, p2 = r['part1'], r['part2']
    lines.append('%s: sig=[%.4g..%.4g] e90=%d e99=%d snull=%d PR=%.1f' % (
        m, p1['sigma_min'], p1['sigma_max'], p1['e90_dim'], p1['e99_dim'],
        p1['soft_null_dim'], p1['participation_ratio']))
    lines.append('  budget: null=%.4f rand=%.4f top=%.4f ratio=%.3f cls=%s pnorm=%.1f(cv=%.2f)' % (
        p2['budget_median']['null'], p2['budget_median']['rand'], p2['budget_median']['top'],
        r['ratio'], r['quotient_class'], p2['p_norm_mean'], p2['p_norm_cv']))
    lines.append('  spec_head=%s spec_tail=%s' % (
        ['%.3g' % x for x in p1['spec_head']], ['%.3g' % x for x in p1['spec_tail']]))
    p3 = r.get('part3_3156')
    if p3:
        d = p3['diam']
        lines.append('  part3: zh_B dmax(KL<0.01)=%.4f klmax=%.5f | zh_A kl=[%.3f..%.3f] | en_B dmax=%.4f' % (
            d['zh_B']['dmax_kl001'], d['zh_B']['kl_max'], d['zh_A']['kl_min'], d['zh_A']['kl_max'],
            d['en_B']['dmax_kl001']))
        lines.append('  verify=%s relerr=%s' % (p3['verify_top1'], ['%.4f' % x for x in p3['verify_relerr']]))
    lines.append('  res=%s seal=%s npz=%s runtime=%ss' % (
        r['res_sha8'], r['seal_sha8'], r['npz_sha8'], r['runtime_s']))
s = json.load(open(os.path.join(R, 'summary', 'result_summary.json'), encoding='utf-8'))
lines.append('SUMMARY: %s' % s['verdict'])
lines.append('  spec_pairs=%s' % {k: round(v, 4) for k, v in s['spec_pairs'].items()})
lines.append('  curve_pairs=%s' % {k: round(v, 4) for k, v in s['curve_pairs'].items()})
lines.append('  fpmin_spec=%.4f fpmin_curve=%.4f gates=%s' % (
    s['fpmin_spec'], s['fpmin_curve'], s['gates']))
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3158_digest.txt'), 'w', encoding='utf-8').write(chr(10).join(lines))
print('written')
