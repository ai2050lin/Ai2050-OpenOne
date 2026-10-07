# -*- coding: utf-8 -*-
"""Extract phase3120 result.json key values for closeout asserts."""
import io
import json

F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3120'
     r'\omega_p118_content_attr_amplifier_'
     r'behavior_opshape\result.json')
OUTF = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3120_values.txt')
r = json.load(io.open(F, encoding='utf-8'))
o = []
o.append('verdict=%r' % r['verdict'])
pa = r['part_a']
o.append('A verdict=%r' % pa['verdict'])
o.append('A factP=%s factA1=%s gap=%s'
         % (pa['fact_margin_P_up'],
            pa['fact_margin_A1_down'],
            pa['gap_attribution_confirmed']))
for k in ('gate_P', 'gate_A1', 'gate_gap'):
    g = pa[k]
    o.append('%s pooled=%r rate=%r nvalid=%d'
             % (k, g.get('pooled_diff',
                         g.get('dgap_ff')),
                g.get('unit_rate'),
                g['n_valid_units']))
o.append('A fact_rate P=%r A1=%r'
         % (pa['fact_token_rate']['P'],
            pa['fact_token_rate']['A1']))
o.append('A clsP=%r' % json.dumps(
    {k: (v['n'], v['mean_dm'])
     for k, v in pa['class_table']['P'].items()}))
o.append('A clsA1=%r' % json.dumps(
    {k: (v['n'], v['mean_dm'])
     for k, v in pa['class_table']['A1'].items()}))
o.append('A subP=%r' % json.dumps(
    {k: (v['n'], v['mean_dm'])
     for k, v in pa['subtag_table']['P'].items()}))
o.append('A subA1=%r' % json.dumps(
    {k: (v['n'], v['mean_dm'])
     for k, v in pa['subtag_table']['A1'].items()}))
o.append('A ynyn=%d' % pa['yes_no_at_content_steps'])
o.append('A per_step_factP=%r'
         % ['%.3f' % x for x in
            pa['per_step_fact_rate']['P']])
o.append('A per_step_factA1=%r'
         % ['%.3f' % x for x in
            pa['per_step_fact_rate']['A1']])

pb = r['part_b']
o.append('B verdict=%r beh=%r samp=%r repro=%r'
         % (pb['verdict'], pb['beh_verdict'],
            pb['samp_verdict'],
            pb['repro_verdict']))
o.append('B yes_greedy=%r'
         % json.dumps(pb['yes_rate_greedy']))
o.append('B agree_greedy=%r'
         % json.dumps(pb['agree_greedy']))
o.append('B beh_max=%r' % pb['beh_max'])
o.append('B yes_samp=%r'
         % json.dumps(pb['yes_rate_sampled']))
o.append('B seq_agree=%r'
         % json.dumps(pb['seq_agree_sampled']))
o.append('B samp_max=%r' % pb['samp_max'])
o.append('B state=%r' % json.dumps(pb['state_ratio']))
o.append('B closed=%r' % json.dumps(
    pb['closed_ratio']))
o.append('B rechk_g=%r rechk_s=%r'
         % (json.dumps(pb['recheck_greedy']),
            json.dumps(pb['recheck_sampled'])))
o.append('B repro_max=%r' % pb['repro_max_diff'])

pc = r['part_c']
o.append('C verdict=%r mono=%r quad=%r gate=%r'
         % (pc['verdict'], pc['mono_verdict'],
            pc['quad_verdict'],
            pc['gate_verdict']))
p = pc['primary']
o.append('C prim sp=%.10f r2l=%.10f r2q=%.10f '
         'r2pw=%.10f b=%r dq=%.10f dpw=%.10f '
         'slope=%.10f n=%d'
         % (p['spearman'], p['r2_lin'],
            p['r2_quad'], p['r2_pw_best'],
            p['pw_breakpoint'], p['d_quad'],
            p['d_pw'], p['slope_lin'], p['n']))
rp = pc['raw_pooled']
o.append('C raw sp=%.10f r2l=%.10f dq=%.10f '
         'dpw=%.10f fixed=%r'
         % (rp['spearman'], rp['r2_lin'],
            rp['d_quad'], rp['d_pw'],
            pc['fixed_point_raw_lin']))
o.append('C t2 sp=%.10f r2l=%.10f flip=%r'
         % (pc['t2_sensitivity']['spearman'],
            pc['t2_sensitivity']['r2_lin'],
            json.dumps(pc['t2_flip'])))
o.append('C centering=%r'
         % json.dumps(pc['centering']))
bc = pc['bin_curve']
o.append('C bins y=%r'
         % ['%.4f' % b['y_mean'] for b in bc])
o.append('C bins x=%r'
         % ['%.3f' % b['x_center'] for b in bc])
o.append('C bins n=%r' % [b['n'] for b in bc])
o.append('created=%s' % r['created'])
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('extracted')
