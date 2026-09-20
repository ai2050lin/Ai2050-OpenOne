# -*- coding: utf-8 -*-
"""Ledger update: M2940 (v3_decode)."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5'
     r'\atlas\atlas_ledger.json')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2940_ledger_update_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


d = json.load(open(P, encoding='utf-8'))
assert d['measurements'][-1]['meas_id'] == \
    'M2939_rotation_target'
verdict_text = (
    'v3_decoder_not_established - ZERO forward 1.5 s, '
    'ANCHORS 5/5 after run1 correction_note (a1 Vt8 '
    'rebuild 3.04e-08, threshold corrected 1e-10 -> '
    '1e-6: cross-phase dirs_word bf16 noise 2.17e-08 '
    'propagates into SVD; a2 sing_vals 4.81e-09; a3 '
    'bit-level; a4 vs 2937 proj 0.00e+00; a5 2.21e-04, '
    'threshold corrected 1e-9 -> 5.1e-4: 2939 stores P3 '
    'rounded to 3 decimals). DESIGN: v3 word-level '
    'decode from 2939 npz coords - P1 SVD layer profile, '
    'P2 displacement decoding 4 ways (class axis label-'
    'swap / concept ICC / scale lock / rewrite link, '
    '10000 perms each), P3 v3 coordinate semantics. P1 '
    'LAYER PROFILE: v3 = mid-layer L14-L18 positive '
    '(+0.39..+0.62) + early L1-L10 and deep L24-L35 all '
    'negative (-0.4..-0.5) bipolar direction, effective '
    'layers 16.1 - v3 is owned by the mid-layer rewrite '
    'battlefield (2937 sep-collapse L8-L16, 2938 L18 '
    'dir ratio 0.162). P2 ALL DECODES FAIL: class axis '
    'obs +5.22 p 0.296; concept ICC 0.389 p 0.420; '
    'scale lock rho 0.127 p 0.352; rewrite link rho '
    '0.099 p 0.461 - the v3 displacement is NOT a '
    'function of any word attribute: a homogeneous '
    'fixed-direction push. P2 per-null class diffs all '
    'same-sign (+3.3..+7.1) but within-word variance '
    'dominates. P3 c3(func) class separation '
    'significant (p 7.0e-04) BUT language confound: '
    'rho(c3, lab) within en only 0.0000 (n=22) - the '
    'separation is language-group driven, not a '
    'semantic axis. CONCLUSION: v3 is the fixed '
    'propulsion direction of the re-encoding machine '
    '(mid-layer owned, word-attribute-blind), not a '
    'decodable semantic axis; the null re-encoding '
    'moves all words along v3 regardless of class/'
    'concept/scale. Lessons: (1) anchor thresholds '
    'must be checked for reachability against the '
    'comparison object precision (2939 P3 rounded to 3 '
    'decimals; SVD propagates 2e-08 input noise), (2) '
    'significant group separation on a confounded '
    'word set requires within-group correlation check '
    'before claiming a semantic axis.')

d['measurements'].append({
    'meas_id': 'M2940_v3_decode',
    'type': 'v3_decode',
    'verdict': verdict_text,
    'source': ('tests/glm5/phase2940_v3_decode.py '
               'sha256_8=02247d1e; outputs execution '
               '1fda5eb9 result 11dd5d1f npz 4f5d8fa8; '
               'sources 2887 e4835a87 / 2927 84fec594 / '
               '2937 518cb922 / 2939 8bae7be6')})

L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
L['connects'].append('M2940_v3_decode')
L['phase_updated'] = 2940

with open(P, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)

rep = ['measurements now: %d' % len(d['measurements']),
       'last meas_id: %s'
       % d['measurements'][-1]['meas_id'],
       'linkage connects now: %d' % len(L['connects']),
       'tail: %s' % L['connects'][-2:],
       'ledger sha8: %s' % sha8(P)]
with open(REP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
print('OK ledger 2940', flush=True)
