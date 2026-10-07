# -*- coding: utf-8 -*-
"""Phase 3127 vals extractor: reads the FINAL
result.json and emits p3127_vals.txt with all
closeout constants (fill the closeout script
from this file)."""
import io
import json

OUTD = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
        r'\result\rdc_query_construction_'
        r'20260913\phase3127'
        r'\omega_p125_writechain_port_'
        r'crossmodel_a1closure_fullregen')
VF = OUTD + r'\result.json'
OF = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
      r'\p3127_vals.txt')

r = json.load(io.open(VF, encoding='utf-8'))
o = []
o.append('VERDICT=%s' % r['verdict'])
o.append('RUNTIME=%.1f' % r['runtime_s'])
o.append('SMOKE=%s' % r['smoke'])

pa = r['part_a']
ac = pa['a1_closure']
for dc in ('P', 'A1'):
    o.append('A1CLO_%s_GAIN46=%.6f'
             % (dc, ac[dc]['gain46']))
    o.append('A1CLO_%s_Z=%.4f'
             % (dc, ac[dc]['z46']))
    o.append('A1CLO_%s_GAINTR=%.6f'
             % (dc, ac[dc]['gain_transfer']))
    o.append('A1CLO_%s_PERM=%.4f/%.4f'
             % (dc, ac[dc]['perm_mean'],
                ac[dc]['perm_std']))
    o.append('A1CLO_%s_G46_0_0=%.4f'
             % (dc, ac[dc]['G46'][0][0]))
cs2 = pa['cross_second']
for dc in ('P', 'A1'):
    o.append('CROSS2_%s=%.6f'
             % (dc, cs2[dc]['cross_gain']))
pca = pa['pca']
o.append('PCA_PC1=%.6f' % pca['pc1_share'])
o.append('PCA_CORR_M=%.4f' % pca['corr_pc1_m'])
o.append('PCA_CORR_DIR=%.4f'
         % pca['corr_pc1_dir'])
o.append('PCA_TOP5=%s'
         % ['%.4f' % v
            for v in pca['var_top5']])
sp = pa['sparse_events']
o.append('SPARSE_RATE=%.6f' % sp['rate'])
o.append('SPARSE_TOP=%.4f' % sp['top_share'])
o.append('SPARSE_COUNTS=%s' % sp['counts'])
o.append('SPARSE_LAYERS=%s' % sp['layers'])

pb = r['part_b']
o.append('B_DREP=%.6e' % pb['repro_max_abs'])
o.append('B_PATH_R=%.8f' % pb['path_r_min'])
o.append('B_GATES=%s'
         % {k: round(v, 4)
            for k, v in pb['gates'].items()})
o.append('B_SPEC=%.4f' % pb['spec_corr'])
o.append('B_CTRL_MED=%.6f' % pb['ctrl_median'])

pc = r['part_c']
o.append('C_DREP=%.6e' % pc['repro_max_abs'])
o.append('C_PATH_R=%.8f' % pc['path_r_min'])
o.append('C_GATES=%s'
         % {k: round(v, 4)
            for k, v in pc['gates'].items()})
o.append('C_SPEC=%.4f' % pc['spec_corr'])
o.append('C_CTRL_MED=%.6f' % pc['ctrl_median'])

dp = r['depth']
o.append('DEPTH_CORR=%.4f' % dp['corr'])
o.append('DEPTH_XS_Q=%s'
         % ['%.3f' % v for v in dp['xs_q']])
o.append('DEPTH_YS_Q=%s'
         % ['%.2f' % v for v in dp['ys_q']])
o.append('DEPTH_YS_G=%s'
         % ['%.2f' % v for v in dp['ys_g']])

pd_ = r['part_d']
o.append('D_S0_AGREE=%.4f' % pd_['s0_probe_agree'])
o.append('D_BIT_MISM=%s' % pd_['bit_mismatch'])
o.append('D_SHIFT_MIN=%.4f'
         % pd_['shift_min_full'])
o.append('D_FLIP_SEP=%.4f' % pd_['flip_sep_max'])
o.append('D_NO_CAP_LOW=%s/%s'
         % (pd_['n_no_cap'], pd_['n_no_low']))
o.append('D_NP_REG=%s' % pd_['np_reg'])
for dc in ('P', 'A1'):
    for c in ('s1', 's2', 's3'):
        st = pd_['stats'][dc][c]
        o.append('D_%s_%s_AGREE=%.4f '
                 'FDIV=%.2f FLIP=%.4f '
                 'MFLIP=%.4f'
                 % (dc, c, st['agree_mean'],
                    st['first_div_mean'],
                    st['flip_rate'],
                    st['multi_flip_rate']))

f = io.open(OF, 'w', encoding='utf-8')
f.write('\n'.join(o) + '\n')
f.close()
print('VALS_OK %d lines -> %s' % (len(o), OF))
