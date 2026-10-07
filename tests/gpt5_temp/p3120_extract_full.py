# -*- coding: utf-8 -*-
"""Full-precision repr of key 3120 values."""
import io
import json

F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3120'
     r'\omega_p118_content_attr_amplifier_'
     r'behavior_opshape\result.json')
OUTF = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3120_values_full.txt')
r = json.load(io.open(F, encoding='utf-8'))
pa = r['part_a']
pb = r['part_b']
pc = r['part_c']
o = []
o.append('dgap_nn=%r' % pa['gate_gap']['dgap_nn'])
o.append('contrast=%r' % pa['gate_gap']['contrast'])
o.append('gap_rate=%r' % pa['gate_gap']['unit_rate'])
o.append('sp=%r' % pc['primary']['spearman'])
o.append('r2l=%r' % pc['primary']['r2_lin'])
o.append('r2q=%r' % pc['primary']['r2_quad'])
o.append('r2pw=%r' % pc['primary']['r2_pw_best'])
o.append('dq=%r' % pc['primary']['d_quad'])
o.append('dpw=%r' % pc['primary']['d_pw'])
o.append('slope=%r' % pc['primary']['slope_lin'])
o.append('t2sp=%r'
         % pc['t2_sensitivity']['spearman'])
o.append('t2r2l=%r'
         % pc['t2_sensitivity']['r2_lin'])
o.append('clsP_qr_mean=%r'
         % pa['class_table']['P']['query_rest']
         ['mean_dm'])
o.append('clsA1_other_mean=%r'
         % pa['class_table']['A1']['other']
         ['mean_dm'])
o.append('clsP_other_mean=%r'
         % pa['class_table']['P']['other']
         ['mean_dm'])
o.append('bin_y0=%r' % pc['bin_curve'][0]['y_mean'])
o.append('bin_y19=%r'
         % pc['bin_curve'][19]['y_mean'])
o.append('beh_max=%r' % pb['beh_max'])
o.append('samp_max=%r' % pb['samp_max'])
o.append('state_L30_ratio=%r'
         % pb['state_ratio']['L30']['ratio'])
o.append('closed_L30_ratio=%r'
         % pb['closed_ratio']['L30']['ratio'])
o.append('yes_L30=%r'
         % pb['yes_rate_greedy']['abl_L30'])
o.append('yes_L32=%r'
         % pb['yes_rate_greedy']['abl_L32'])
o.append('yes_samp_L30=%r'
         % pb['yes_rate_sampled']['abl_L30'])
o.append('yes_samp_L32=%r'
         % pb['yes_rate_sampled']['abl_L32'])
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('ok')
