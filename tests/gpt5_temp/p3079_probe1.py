import json
import io

import numpy as np

RDIR = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
z76 = np.load(
    RDIR + r'\phase3076'
    r'\omega_p73_cross_prompt_family'
    r'\omega_p73_cross_prompt_family.npz')
P74 = (RDIR + r'\phase3074'
       r'\omega_p71_capacity_law')
z74 = np.load(P74 + r'\omega_p71_capacity_law.npz')

out = []
# CS / R_S / A_S / MASKS structure
for fk in ('A', 'B', 'C'):
    out.append(
        'CS_%s %s %s | R_S %s | A_S %s | '
        'MASKS %s'
        % (fk, z76['CS_' + fk].shape,
           z76['CS_' + fk].dtype,
           z76['R_S_' + fk].shape,
           z76['A_S_' + fk].shape,
           z76['MASKS_' + fk].shape))
m = {fk: z76['MASKS_' + fk]
     for fk in ('A', 'B', 'C')}
out.append('MASKS A==B: %s A==C: %s'
           % (bool(np.array_equal(m['A'], m['B'])),
              bool(np.array_equal(m['A'], m['C']))))
out.append('MASKS[0:8] A: %s'
           % m['A'][:8].tolist())
out.append('MASKS sorted? %s | min %d max %d'
           % (bool(np.all(np.diff(m['A']) > 0)),
              int(m['A'].min()),
              int(m['A'].max())))
# CS orientation probe: CS_A[0, :5]
out.append('CS_A[0,:5]=%s'
           % z76['CS_A'][0, :5].tolist())
out.append('CS_A[:,0][:5]=%s'
           % z76['CS_A'][:5, 0].tolist())
out.append('R_S_A[:5]=%s'
           % z76['R_S_A'][:5].tolist())
out.append('A_S_A[:5]=%s'
           % z76['A_S_A'][:5].tolist())
out.append('A_S_A vs 3074 max|d|=%s'
           % float(np.max(np.abs(
               z76['A_S_A']
               - z74['A_S'].astype(np.float64)))))
# CS1H structure
for fk in ('A', 'B', 'C'):
    out.append('CS1H_%s %s med_row0[:4]=%s'
               % (fk, z76['CS1H_' + fk].shape,
                  np.median(
                      z76['CS1H_' + fk],
                      axis=1)[:4].tolist()))
# SP_R1 replay data
R1 = {fk: z76['R1_ALL32_' + fk]
      .astype(np.float64)
      for fk in ('A', 'B', 'C')}


def sp(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    return float((ra * rb).sum() / den)


out.append('SP_R1 replay AB %.6f (npz %.6f)'
           % (sp(R1['A'], R1['B']),
              float(z76['SP_R1_AB'])))
out.append('SP_R1 replay AC %.6f (npz %.6f)'
           % (sp(R1['A'], R1['C']),
              float(z76['SP_R1_AC'])))
out.append('SP_R1 replay BC %.6f (npz %.6f)'
           % (sp(R1['B'], R1['C']),
              float(z76['SP_R1_BC'])))
# 255-spectrum family similarity preview
for fa, fb in (('A', 'B'), ('A', 'C'),
               ('B', 'C')):
    out.append(
        'sp(A_S_%s, A_S_%s)=%.4f  '
        'sp(R_S_%s, R_S_%s)=%.4f'
        % (fa, fb, sp(z76['A_S_' + fa],
                      z76['A_S_' + fb]),
           fa, fb, sp(z76['R_S_' + fa],
                      z76['R_S_' + fb])))
# TT norms per pair
for fk in ('A', 'B', 'C'):
    tn = np.linalg.norm(
        z76['TT_' + fk].astype(np.float64),
        axis=1)
    out.append('TT_%s norms med=%.2f '
               'min=%.2f max=%.2f'
               % (fk, float(np.median(tn)),
                  float(tn.min()),
                  float(tn.max())))
# TT cross-family per-pair spearman preview
TT = {fk: z76['TT_' + fk]
      .astype(np.float64)
      for fk in ('A', 'B', 'C')}
for fa, fb in (('A', 'B'), ('A', 'C'),
               ('B', 'C')):
    spp = [sp(TT[fa][k], TT[fb][k])
           for k in range(24)]
    out.append('TT sp per-pair %s%s: '
               'med=%.4f min=%.4f max=%.4f'
               % (fa, fb,
                  float(np.median(spp)),
                  float(min(spp)),
                  float(max(spp))))
# 3074 npz keys
out.append('z74 keys: %s'
           % sorted(z74.files)[:20])
p = (r'D:\AI2050\Ai2050-OpenOne'
     r'\tests\gpt5_temp\p3079_probe1.txt')
io.open(p, 'w', encoding='utf-8').write(
    '\n'.join(out) + '\n')
print('OK')
