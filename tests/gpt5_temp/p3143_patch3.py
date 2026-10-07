# -*- coding: utf-8 -*-
"""p3143 patch3: PC[29] v1 sign
disambiguation protocol.

Finding: b_pc1_l29_d2.0 with +v1 gave
chg=0.4062 != preregistered 3142 anchor
0.203125. np.linalg.svd singular-vector
SIGNS are not stable across sessions
(LAPACK); 3141->3142 agreement was
incidental. Protocol: the preregistered
behavior anchor resolves the free sign
parameter. If neither sign reproduces
the anchor, the SVD drifted numerically
and we hard-fail for manual inspection.
Records pc1_sign in result; captures and
projections then use the resolved
vector. ckpt keeps the +v1 trial (valid
data point either way)."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\glm5\phase3143_omega_'
       r'p141_d19field_readout_topk_'
       r'newI.py')

txt = io.open(SRC, encoding='utf-8').read()

old1 = '''v1 = PC[29]['V'][0]
mn29 = PC[29]['mnorm']
dv_pc1 = np.tile(
    (v1 * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_dv29 = (dvec[29][:NCAP] * 2.0) \\
    .astype(np.float32)'''
new1 = '''v1 = PC[29]['V'][0]
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (v1 * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
pc1_sign = None  # resolved below
dv_dv29 = (dvec[29][:NCAP] * 2.0) \\
    .astype(np.float32)'''
n1 = txt.count(old1)
assert n1 == 1, 'old1 count %d' % n1
txt = txt.replace(old1, new1)

old2 = '''_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1, 1.0, 'allstep',
               rows_scan, base12_P)
_run_vec_trial('b_dvec29_l29_d2.0', 29,
               dv_dv29, 1.0, 'allstep',
               rows_scan, base12_P)
n_bitB = 0
bit_anchors_B = {}
if not SMOKE:
    for tn, v in (('b_pc1_l29_d2.0',
                   PC1_D2),
                  ('b_dvec29_l29_d2.0',
                   DVEC29_D2)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitB += int(m)
        bit_anchors_B[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('D2 3142 repro: %d/2 bit-match'
        % n_bitB)
else:
    log('D2 3142 repro SKIPPED (smoke)')'''
new2 = '''_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P)
# sign disambiguation: the preregistered
# 3142 anchor resolves the SVD sign free
# parameter (LAPACK signs are not stable
# across sessions). If neither sign
# matches, the SVD drifted numerically
# -> hard fail.
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - PC1_D2) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg'] - PC1_D2) < 1e-9:
        pc1_sign = -1
    else:
        raise AssertionError(
            'pc1 sign unresolved: pos '
            '%.6f neg %.6f want %.6f'
            % (E_res['b_pc1_l29_d2.0']
               ['chg'],
               E_res['b_pc1neg_l29_d2.0']
               ['chg'], PC1_D2))
else:
    if not SMOKE:
        pc1_sign = 1
if pc1_sign == -1:
    dv_pc1 = dv_pc1_neg
    log('D2 pc1 sign resolved: -1 '
        '(neg reproduces 3142 anchor '
        '%.6f)' % PC1_D2)
elif pc1_sign == 1:
    log('D2 pc1 sign resolved: +1 '
        '(pos reproduces 3142 anchor)')
_run_vec_trial('b_dvec29_l29_d2.0', 29,
               dv_dv29, 1.0, 'allstep',
               rows_scan, base12_P)
n_bitB = 0
bit_anchors_B = {}
if not SMOKE:
    for tn, v in (('b_pc1_l29_d2.0',
                   PC1_D2),
                  ('b_dvec29_l29_d2.0',
                   DVEC29_D2)):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bitB += int(m)
        bit_anchors_B[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('D2 3142 repro: %d/2 bit-match'
        % n_bitB)
else:
    log('D2 3142 repro SKIPPED (smoke)')'''
n2 = txt.count(old2)
assert n2 == 1, 'old2 count %d' % n2
txt = txt.replace(old2, new2)

old3 = '''    'part_readout': {
        'bit_anchors_3142': bit_anchors_B,'''
new3 = '''    'part_readout': {
        'pc1_sign': pc1_sign,
        'bit_anchors_3142': bit_anchors_B,'''
n3 = txt.count(old3)
assert n3 == 1, 'old3 count %d' % n3
txt = txt.replace(old3, new3)

io.open(SRC, 'w', encoding='utf-8').write(txt)
txt2 = io.open(SRC, encoding='utf-8').read()
assert txt2.count(new1) == 1
assert txt2.count(new2) == 1
assert txt2.count(new3) == 1
assert "pc1_sign = None" in txt2
py_compile.compile(SRC, doraise=True)
print('patch3 OK, compiled')
