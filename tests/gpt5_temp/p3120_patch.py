# -*- coding: utf-8 -*-
"""Patch phase3120 script on real disk (Edit tool
unreliable this session).  Idempotent: each patch
applied only if old pattern present; report to txt."""
import io

F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3120_omega_p118_content_attr_'
     r'amplifier_behavior_opshape.py')
RPT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3120_patch_report.txt')
src = io.open(F, encoding='utf-8').read()
o = []

P1_OLD = r'''    dm = (mD[:, 1:] - mD[:, :-1]) \
        .astype(np.float64)
    nt = dm.shape[1]
    x = mD[:, :-1].astype(np.float64)[:, 1:]
    fv = factmask[:, 1:]
    edges = np.quantile(x.ravel(),
                        [0.2, 0.4, 0.6, 0.8])
    qb = np.searchsorted(edges, x.ravel(),
                         side='right')
    rows = []
    diffs = []
    dP = dm.ravel()
    for t in range(nt):
        for b in range(5):
            sel = (qb == b) \
                & (np.arange(nt)[None, :]
                   == t).ravel()
'''
P1_NEW = r'''    dm = (mD[:, 1:] - mD[:, :-1]) \
        .astype(np.float64)[:, 1:]
    nt = dm.shape[1]
    x = mD[:, :-1].astype(np.float64)[:, 1:]
    fv = factmask[:, 1:]
    edges = np.quantile(x.ravel(),
                        [0.2, 0.4, 0.6, 0.8])
    qb = np.searchsorted(edges, x.ravel(),
                         side='right')
    tarr = np.tile(np.arange(nt), x.shape[0])
    rows = []
    diffs = []
    dP = dm.ravel()
    for t in range(nt):
        for b in range(5):
            sel = (qb == b) & (tarr == t)
'''

P2_OLD = r'''        sel = (qb == b) \
            & (np.arange(nt)[None, :] == t).ravel()
        sf_all |= (sel & fv_r)'''
P2_NEW = r'''        sel = (qb == b) & (tarr == t)
        sf_all |= (sel & fv_r)'''

P3_OLD = r'''dmP_full = (mP[:, 1:] - mP[:, :-1]) \
    .astype(np.float64)
dmA_full = (mA[:, 1:] - mA[:, :-1]) \
    .astype(np.float64)
'''
P3_NEW = ''

P4_OLD = r'''    sel = (qbg == b) \
        & (np.arange(nt)[None, :] == t).ravel()
    sfF |= (sel & ffv)'''
P4_NEW = r'''    sel = (qbg == b) & (tarrg == t)
    sfF |= (sel & ffv)'''

P5_OLD = r'''    'gate_P': {k: gATE_P[k] for k in
               ('pooled_diff', 'unit_rate',
                'n_valid_units', 'n_units')},
    'gate_A1': {k: gATE_A[k] for k in
                ('pooled_diff', 'unit_rate',
                 'n_valid_units', 'n_units')},'''
P5_NEW = r'''    'gate_P': {**{k: gATE_P[k] for k in
                  ('pooled_diff', 'unit_rate',
                   'n_valid_units',
                   'n_units')},
               'units': gATE_P['units']},
    'gate_A1': {**{k: gATE_A[k] for k in
                   ('pooled_diff', 'unit_rate',
                    'n_valid_units',
                    'n_units')},
                'units': gATE_A['units']},'''

P6_OLD = r'''    'gate_gap': {'dgap_ff': pdg_ff,
                 'dgap_nn': pdg_nn,
                 'contrast': contrast_pool,
                 'unit_rate': rate_g,
                 'n_valid_units': nval_g},'''
P6_NEW = r'''    'gate_gap': {'dgap_ff': pdg_ff,
                 'dgap_nn': pdg_nn,
                 'contrast': contrast_pool,
                 'unit_rate': rate_g,
                 'n_valid_units': nval_g,
                 'units': rows_g},'''

PATCHES = [('P1_margin_gate_head', P1_OLD, P1_NEW),
           ('P2_margin_pooling', P2_OLD, P2_NEW),
           ('P3_drop_dmP_full', P3_OLD, P3_NEW),
           ('P4_gap_pooling', P4_OLD, P4_NEW),
           ('P5_gate_units_P_A1', P5_OLD, P5_NEW),
           ('P6_gate_gap_units', P6_OLD, P6_NEW)]
for (nm, old, new) in PATCHES:
    cnt = src.count(old)
    if cnt == 1:
        src = src.replace(old, new)
        o.append('%s: applied (1 match)' % nm)
    elif cnt == 0 and new in src:
        o.append('%s: already applied' % nm)
    else:
        o.append('%s: ERROR count=%d' % (nm, cnt))

# post conditions
chk = [('no legacy sel margin', 'np.arange(nt)['
        '[None' not in src),
       ('tarr present', 'tarr = np.tile' in src),
       ('tarrg present', 'tarrg = np.tile' in src),
       ('dm slice 11col', '.astype(np.float64)'
        '[:, 1:]\n    nt = dm.shape[1]' in src),
       ('torch_dtype ok',
        'torch_dtype=torch.bfloat16,' in src),
       ('no torch.dtype kwarg',
        'torch.dtype=' not in src),
       ('dmP_full gone', 'dmP_full' not in src)]
for (nm, ok) in chk:
    o.append('CHK %s: %s' % (nm, 'OK' if ok
                             else 'FAIL'))
ok_all = all(ok for (_, ok) in chk) and \
    all('ERROR' not in line for line in o)
if ok_all:
    with io.open(F, 'w', encoding='utf-8') as f:
        f.write(src)
    o.append('FILE WRITTEN')
else:
    o.append('FILE NOT WRITTEN (errors above)')
with io.open(RPT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('patch done ok_all=%s' % ok_all)
