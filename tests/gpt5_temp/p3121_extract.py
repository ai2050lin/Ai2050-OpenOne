# -*- coding: utf-8 -*-
"""Extract full-precision key values from the 3121
result.json (arg1: result dir, defaults to the real
run dir)."""
import io
import json
import sys

F = sys.argv[1] if len(sys.argv) > 1 else (
    r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
    r'\rdc_query_construction_20260913'
    r'\phase3121'
    r'\omega_p119_repl_causality_erase_'
    r'polarity_recon\result.json')
OUTF = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3121_values_full.txt')
r = json.load(io.open(F, encoding='utf-8'))
pa = r['part_a']
pb = r['part_b']
pc = r['part_c']
pd = r['part_d']
o = []
o.append('verdict=%r' % r['verdict'])
o.append('smoke=%r n_pairs=%r np_a=%r np_g=%r'
         % (r['smoke'], r['n_pairs'], r['np_a'],
            r['np_g']))
o.append('runtime_s=%r' % r['runtime_s'])
o.append('-- part_a --')
o.append('repro=%r max_diff=%r'
         % (pa['repro']['verdict'],
            pa['repro']['max_diff']))
o.append('n_span_P=%r n_span_A1=%r'
         % (pa['n_span_P'], pa['n_span_A1']))
for d in ('P', 'A1'):
    e = pa['effects'][d]
    o.append('eff[%s]=%s' % (d, json.dumps(e)))
o.append('gates=%s' % json.dumps(pa['gates']))
o.append('-- part_b --')
o.append('yes_clean=%r yes_joint=%r delta=%r'
         % (pb['yes_rate_clean'],
            pb['yes_rate_joint'],
            pb['yes_delta']))
o.append('joint_verdict=%r' % pb['joint_verdict'])
o.append('fam_r=%r fam_verdict=%r'
         % (pb['fam_r'], pb['fam_verdict']))
for k, v in pb['first_token'].items():
    o.append('fork[%s]=%s' % (k, json.dumps(v)))
o.append('-- part_c --')
o.append('r2_two=%r r2_lin=%r r2_pers=%r d_r2=%r'
         % (pc['r2_two'], pc['r2_lin'],
            pc['r2_persistence'], pc['d_r2']))
o.append('fit_verdict=%r' % pc['fit_verdict'])
o.append('auc_r=%r auc_r_lin=%r curve=%r'
         % (pc['auc_r'], pc['auc_r_lin'],
            pc['curve_verdict']))
o.append('sigma=%r coverage=%r mc=%r'
         % (pc['sigma'], pc['coverage'],
            pc['mc_verdict']))
o.append('-- part_d --')
o.append('verdict=%r' % pd['verdict'])
for k in ('syntax_min_P', 'syntax_max_P',
          'syntax_min_A1', 'syntax_max_A1'):
    o.append('%s=%r' % (k, pd[k]))
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
