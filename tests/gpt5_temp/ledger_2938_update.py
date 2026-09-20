# -*- coding: utf-8 -*-
"""Ledger M2938 registration + verification."""
import hashlib
import json

P = r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas\atlas_ledger.json'
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2938_ledger_update_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d = json.load(open(P, encoding='utf-8'))
n_before = len(d['measurements'])
assert d['measurements'][-1]['meas_id'] == 'M2937_scale_collapse'
d['measurements'].append({
    'meas_id': 'M2938_subspace_angles',
    'type': 'subspace_angles',
    'verdict': 'subspace_rotation_retained - ONE run 14 s, '
               'NO ablation, ANCHORS 7/7: a1 dirs rebuild '
               '2.17e-08 (14th consecutive forward anchoring); '
               'a2 determinism 0.00e+00; a3 7.21e-06 / a4 '
               '6.26e-06 vs 2935 s_base; a5 SVD orthonormality '
               '2.11e-15; a6 func sep 185.70; a7 align_dir35 '
               'vs 2937 |proj|/fin_norm 0.00e+00 (cross-phase '
               'bit-level). DESIGN: 2937 protocol verbatim + '
               'SVD of stacked dirs_word (36 x 2560, top-8 '
               'energy 91.9 percent - language subspace is '
               'low-rank); alignment alpha_k = ||Vt[:k] x||/'
               '||x|| scale-invariant, grid k {1,4,8,16,36} + '
               'dir35 single direction. P1 DISSOCIATION: rho '
               '(null/func median alignment) dir35 0.5091 vs '
               'PC1 0.8339 vs k=8 0.9991 vs k=36 0.9957 - '
               'collapse is STRICTLY confined to the single '
               'direction dirs_word[35]; per-word alpha_8 '
               'null0 min 0.173, 0/57 words leave the '
               'subspace. P2 paired sign-flip permutation '
               'func vs null0 alpha_8 median diff -0.0036, p '
               '0.66 - no detectable subspace-alignment '
               'difference at all. P3 LAYER PROFILE: dir '
               'ratio collapses to 0.162 at L18 while alpha8 '
               'same layer 0.942; alpha8 ratio >= 0.77 all '
               'layers, dir ratio < 0.7 at L6-L11 - mid-layer '
               'attention re-encodes WITHIN the language '
               'subspace. SEAL: alignment gradient monotone '
               '(dir35 0.51 < PC1 0.83 < k4 0.97 < k8 1.00); '
               'L20 dir ratio 1.187 > 1 (partial deep '
               'recovery/inversion). MECHANISM: 2937 '
               'direction collapse = rotation WITHIN the '
               'language subspace, i.e. re-encoding of the '
               'word-term structure into other subspace '
               'directions under null context; the readout '
               'failure is a single-direction projection '
               'artifact, not loss of language information. '
               'Lesson: single-direction cos claims must be '
               'checked against the spanned subspace before '
               'naming a collapse.',
    'source': ('tests/glm5/phase2938_subspace_angles.py '
               'sha256_8=b7f9bf71; outputs execution '
               '56632231 result 17d6ea93 npz 5f1bd256; '
               'sources 2927 84fec594 / 2937 518cb922')})
L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
L['connects'].append('M2938_subspace_angles')
L['phase_updated'] = 2938
with open(P, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

# verify on disk
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
