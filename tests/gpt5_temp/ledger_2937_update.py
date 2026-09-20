# -*- coding: utf-8 -*-
"""Ledger update: append M2937 to atlas_ledger.json."""
import hashlib
import json
import os

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUTD = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913\phase2937'
        r'\scale_collapse')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2937_ledger_update_report.txt')


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


verdict_text = (
    'scale_collapse_rewrite - ONE run 14 s, NO ablation, '
    'ANCHORS 6/6: a1 dirs rebuild 2.17e-08 (13th consecutive '
    'forward anchoring); a2 determinism 0.00e+00; a3 '
    'proj_func vs 2935 s_base 7.21e-06; a4 proj_null0 vs '
    '2935 6.26e-06; a5 masks 469/382/295; a6 func sep '
    '185.70. DESIGN: baseline-only 5 conditions (func/same/'
    'null0-3) x batch57, per-layer pos-1 attn-input capture '
    '+ final pre-norm residual + zero-forward embedding '
    'lookup. P1 ENERGY DOES NOT COLLAPSE: per-layer residual-'
    'norm ratio (null/func) 0.94-1.07 all layers, no '
    'crossing < 0.8 - the 2936 scale collapse is NOT energy. '
    'P3 DIRECTION COLLAPSES: cos(final, dirs_word[35]) '
    'median 0.045 (func) -> 0.002-0.024 (null), ratio '
    '0.05-0.24, all 4 sets direction-dominant (|log norm| < '
    '|log cos|); absolute cos shows dirs_word[35] explains '
    'only ~4.5% of the final residual even under func. P2 '
    'REWRITE: s_null = beta*s_func + gamma fits give beta '
    '0.43-0.55 (median 0.4886), gamma -11.6..-16.1, R2 '
    '0.74-0.86; same beta 0.60. P4 embedding norms: words '
    '1.085+-0.073 vs null 1.071+-0.215, p 0.64 - token-'
    'identity norm explanation rejected. SEAL: (1) sep ratio '
    'profile is U-SHAPED - language-signal collapse deepest '
    'at L8-L16 (0.11-0.36), partial recovery deep (L32-L35 '
    '0.41-0.61): mid-layer attention is the rewrite arena, '
    'where func sep itself peaks (L6-L10: 3.5-7.3); (2) '
    'REWRITE IS CLASS-ASYMMETRIC: within-lab1 readout '
    'structure highly retained (spearman 0.75-0.84) while '
    'within-lab0 rewritten (0.13-0.24) - connects to 2928 '
    'L+ class asymmetry; (3) rewrite outliers include high-'
    'func-CI words (light 194->131, war 198->18, city '
    '202->33). REGISTERED: context-statistics effect on the '
    'baseline is a mid-layer attention-driven ROTATION of '
    'the final residual away from the language direction '
    'with ~50% word-term retention (beta) + negative offset '
    '(gamma -11..-16), NOT energy scaling; magnitude of '
    '2936 scale collapse = direction collapse in a small-'
    'cos component. Lesson: readout-magnitude claims must '
    'decompose norm vs cos before naming a mechanism.')

d = json.load(open(P, encoding='utf-8'))
exec37 = json.load(open(os.path.join(OUTD, 'execution.json'),
                        encoding='utf-8'))
src_str = ' / '.join('%s %s' % (k, v)
                     for k, v in exec37['sources'].items())
d['measurements'].append({
    'meas_id': 'M2937_scale_collapse',
    'type': 'scale_collapse',
    'verdict': verdict_text,
    'source': ('tests/glm5/phase2937_scale_collapse.py '
               'sha256_8=%s; outputs execution %s result %s '
               'npz %s; sources %s'
               % (exec37['script_sha256_8'],
                  sha8(os.path.join(OUTD, 'execution.json')),
                  sha8(os.path.join(OUTD, 'result.json')),
                  sha8(os.path.join(OUTD, 'scale_collapse.npz')),
                  src_str))})
L = d['linkage'][-1]
assert L['link_id'] == 'L14_readout_spectrum_cross_model'
L['connects'].append('M2937_scale_collapse')
L['phase_updated'] = 2937
json.dump(d, open(P, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)

d2 = json.load(open(P, encoding='utf-8'))
rep = ['n_meas=%d' % len(d2['measurements']),
       'last id=%s' % d2['measurements'][-1]['meas_id'],
       'connects_len=%d tail=%s'
       % (len(d2['linkage'][-1]['connects']),
          d2['linkage'][-1]['connects'][-2:]),
       'ledger sha8=%s' % sha8(P)]
open(REP, 'w', encoding='utf-8').write('\n'.join(rep) + '\n')
print('OK ledger 2937', flush=True)
