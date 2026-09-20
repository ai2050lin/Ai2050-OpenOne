# -*- coding: utf-8 -*-
"""Ledger M2939 registration + verification."""
import hashlib
import json

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2939_ledger_update_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d = json.load(open(P, encoding='utf-8'))
n_before = len(d['measurements'])
assert d['measurements'][-1]['meas_id'] == 'M2938_subspace_angles'
d['measurements'].append({
    'meas_id': 'M2939_rotation_target',
    'type': 'rotation_target',
    'verdict': 'rotation_target_identified - ONE run 14 s, '
               'NO ablation, ANCHORS 6/6: a1 dirs rebuild '
               '2.17e-08 (15th consecutive forward '
               'anchoring); a2 determinism 0.00e+00; a3 '
               '7.21e-06 / a4 6.26e-06 vs 2935 s_base; a5 '
               'alpha_k vs 2938 align_k 3.61e-16 (near '
               'bit-level); a6 func sep 185.70. DESIGN: '
               '2938 protocol verbatim + NEW per-word '
               'coordinates c(w,k) = x_w . v_k in the top-8 '
               'SVD basis. P1 STRUCTURE RETENTION: per-basis '
               'Spearman rho_med over 4 null sets - dir35 '
               '0.8809 (rank structure kept even where '
               'magnitude halves), v2 0.8186, v5 0.7940, '
               'v1 0.7504 vs v6 0.3864 (ONLY basis losing '
               'word structure; share smallest 0.005-0.012). '
               'P2 ENERGY FLOW: k* = v3, delta_e_med +0.1047 '
               '(share 0.181 -> 0.290), label-swap perm p '
               '9.999e-04; inflow also v5 +0.0689, v7 '
               '+0.0223, v4 +0.0212; OUTFLOW v1 -0.1463, v2 '
               '-0.0739. SEAL MECHANISM: v1/v2 are dir35 '
               'decomposition (coords vs dir35 proj '
               'Spearman -0.96/+0.97) while v3 is weakly '
               'coupled (-0.38) - energy moves from the '
               'dir35-parallel component to near-orthogonal '
               'v3/v5: the quantitative shape of '
               're-encoding. P3 CLASS DISPLACEMENT: delta_c '
               'cross-null cos(null0 vs null1/2/3) 0.997-'
               '1.000 - the re-encoding direction is a '
               'FIXED context-independent direction '
               '(context-statistics driven, not token '
               'identity); same context gives a DIFFERENT '
               'displacement pattern. Word displacement '
               '||dc||/||c_func|| median 0.341, max word '
               'war (2937 rewrite outlier consistent). '
               'CONCLUSION: null-context re-encoding = '
               'linear-structured energy transfer from the '
               'language-readout axis to a fixed '
               'near-orthogonal in-subspace direction v3, '
               'with word-rank structure largely preserved '
               '(dir35 rho 0.88). Lesson: projection-'
               'magnitude collapse and coordinate-rank '
               'structure are different claims - report '
               'both.',
    'source': ('tests/glm5/phase2939_rotation_target.py '
               'sha256_8=f14811a3; outputs execution '
               '45f76e71 result 11183c5c npz 8bae7be6; '
               'sources 2927 84fec594 / 2938 5f1bd256')})
L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
L['connects'].append('M2939_rotation_target')
L['phase_updated'] = 2939
with open(P, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

d2 = json.load(open(P, encoding='utf-8'))
rep = ['n_meas %d -> %d' % (n_before, len(d2['measurements'])),
       'last meas_id: %s'
       % d2['measurements'][-1]['meas_id'],
       'L14 connects len: %d tail1: %s'
       % (len(d2['linkage'][-1]['connects']),
          d2['linkage'][-1]['connects'][-1]),
       'ledger sha8: %s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write(
    chr(10).join(rep) + chr(10))
print('ledger done')
