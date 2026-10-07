# -*- coding: utf-8 -*-
"""Phase 2997: Omega-F2b registry REGRADE -- card-set v2.1
registration (plan v4 P0 continuation; 2996 audit verdict
`registry_robust_across_calibers` supplies the evidence).

Question (preregistered, frozen BEFORE observation): does the
2996 product-level evidence chain warrant the frozen regrade
mapping
    ('T3_M2989', 'not_replicated') -> 'replicated'
on the 2995 omega_f1_glm4_panel grade sheet, with the
registration invariants holding (exactly ONE regraded card;
unregraded cards untouched; basis hashes consistent across
2996 seal, ledger tail, and this registration)?

Tasks.
  T1 (degeneracy reproduced at artifact level, independent of
      any narrative):
      a1 2995 npz dirs_attn AND dirs_mlp: all 40 rows
         max|row - row0| == 0 (bit-level degeneracy).
      a2 2995 result grades dict EXACTLY equals the frozen
         forensics {'T1_M2963': 'replicated',
                     'T2_M2947': 'replicated',
                     'T3_M2989': 'not_replicated'}.
      a3 2995 T3 per-layer z at [7,10,13] equals the frozen
         registered values [-1.75, -2.25, -1.89] within 1e-9.
  T2 (audit evidence chain):
      a4 2996 result final_verdict ==
         'registry_robust_across_calibers' AND seal
         result_sha256_8 == 'd03a47b0' AND ledger tail entry
         phase == 2996 with the SAME result hash AND the SAME
         verdict string.
      a5 2996 T3.k2 z at [7,10,13] equals the frozen values
         [30.55, 10.41, 10.15] within 5e-3 (result stores 2
         decimals) AND T2.glm_sig is True AND share-leg p at
         glm L10 == 0.0 AND qwen L6 == 0.0.
  T3 (regrade registration + invariants):
      a6 apply the frozen mapping to the 2995 grade sheet ->
         cards_v21; invariants: exactly one card changed;
         changed card old grade == 'not_replicated' and new
         grade == 'replicated' (inside the allowed grade
         vocabulary); the two unregraded cards keep their
         grades bit-identical; every basis field cites the
         2996 result hash (8 hex) and the artifact-level
         degeneracy value 0.0.

Verdict (explicit branches):
  all anchors pass            -> regrade_replicated_registered
  a1..a5 any fail             -> evidence_chain_broken
  a6 fails (invariant)        -> regrade_mapping_invalid

Tags: plan v4 P0 domain -- lang axis / len-2 / snapshot only
(causal ablation untested) / glm4-9b panel grades; card-set
v2.1 adds basis-hash provenance column (audit trail rule:
no regrade without a sealed basis).
"""
import hashlib
import json
import os
import re
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2995 = os.path.join(BASE, 'phase2995',
                        'omega_f1_glm4_panel',
                        'omega_f1_glm4_panel.npz')
RES_2995 = os.path.join(BASE, 'phase2995',
                        'omega_f1_glm4_panel', 'result.json')
RES_2996 = os.path.join(BASE, 'phase2996',
                        'omega_f2a_registry_caliber_audit',
                        'result.json')
SEAL_2996 = os.path.join(BASE, 'phase2996',
                         'omega_f2a_registry_caliber_audit',
                         'seal.json')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
OUT = os.path.join(BASE, 'phase2997',
                   'omega_f2b_registry_regrade')

# frozen forensics (PREREG; recorded before this run)
FROZEN_2995_GRADES = {
    'T1_M2963': 'replicated',
    'T2_M2947': 'replicated',
    'T3_M2989': 'not_replicated'}
FROZEN_2995_T3_Z = {'7': -1.75, '10': -2.25, '13': -1.89}
FROZEN_2996_T3_Z = {'7': 30.55, '10': 10.41, '13': 10.15}
FROZEN_2996_VERDICT = 'registry_robust_across_calibers'
FROZEN_2996_RESULT_SHA8 = 'd03a47b0'
REGRADE_MAP = {('T3_M2989', 'not_replicated'): 'replicated'}
GRADE_VOCAB = {'replicated', 'directionally_replicated',
               'partial', 'not_replicated', 'artifact'}
Z_TOL_2995 = 1e-9
Z_TOL_2996 = 5e-3


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    prereg = {
        'question': 'does the 2996 product-level evidence '
                    'chain warrant the frozen regrade '
                    '(T3_M2989 not_replicated -> replicated) '
                    'with registration invariants holding?',
        'T1': 'artifact-level degeneracy reproduction: 2995 '
              'npz dirs_attn/dirs_mlp all 40 rows identical '
              '(== 0); 2995 grades dict == frozen forensics; '
              '2995 T3 z == frozen [-1.75,-2.25,-1.89] '
              '(tol 1e-9)',
        'T2': 'audit chain: 2996 verdict + seal hash '
              'd03a47b0 + ledger tail agree (phase 2996, '
              'same hash, same verdict); 2996 T3.k2 z == '
              'frozen [30.55,10.41,10.15] (tol 5e-3) AND '
              'glm_sig AND share p==0 at glm L10 and qwen '
              'L6',
        'T3': 'regrade registration: frozen mapping applied '
              'to 2995 grades -> cards_v21; invariants: '
              'exactly one regrade, old==not_replicated, '
              'new==replicated (in vocab), others untouched, '
              'basis cites 2996 result hash + degeneracy 0.0',
        'verdict': 'anchor fail => evidence_chain_broken '
                   '(a1..a5) or regrade_mapping_invalid '
                   '(a6); else regrade_replicated_registered',
        'anchors': {
            'a1': '2995 npz dirs degeneracy == 0 (both '
                  'arrays, bit-level)',
            'a2': '2995 grades dict == frozen forensics',
            'a3': '2995 T3 z vs frozen |d| < 1e-9',
            'a4': '2996 verdict+seal+ledger-tail agreement',
            'a5': '2996 T3.k2 z vs frozen |d| < 5e-3 AND '
                  'glm_sig AND share p==0 (glm L10, qwen L6)',
            'a6': 'regrade invariants (one card, old->new '
                  'as frozen, basis hash cited)'},
        'tags': 'plan v4 P0: lang axis / len-2 / snapshot '
                'only / glm4-9b panel / card-set v2.1 adds '
                'basis-hash provenance; no regrade without '
                'a sealed basis',
    }

    # ---------- execution.json freeze ----------
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2997,
                   'name': 'omega_f2b_registry_regrade',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2995npz': sha8(SRC_2995),
                       's2995res': sha8(RES_2995),
                       's2996res': sha8(RES_2996),
                       's2996seal': sha8(SEAL_2996),
                       'ledger': sha8(LEDGER)},
                   'prereg': prereg},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    verdict = None
    a = {}

    # ---------- T1: artifact-level degeneracy ----------
    z95 = np.load(SRC_2995, allow_pickle=True)
    da = z95['dirs_attn'].astype(np.float64)
    dm = z95['dirs_mlp'].astype(np.float64)
    deg_attn = float(np.abs(da - da[0]).max())
    deg_mlp = float(np.abs(dm - dm[0]).max())
    a['a1_degen_attn'] = deg_attn
    a['a1_degen_mlp'] = deg_mlp
    a1_ok = bool(deg_attn == 0.0 and deg_mlp == 0.0)
    log('a1 2995 npz dirs degeneracy attn=%.3e mlp=%.3e '
        'ok=%s' % (deg_attn, deg_mlp, a1_ok), lines)

    r95 = json.load(open(RES_2995, encoding='utf-8'))
    grades95 = r95.get('grades', {})
    a2_ok = bool(grades95 == FROZEN_2995_GRADES)
    a['a2_grades_match'] = a2_ok
    log('a2 2995 grades == frozen forensics: %s (%s)'
        % (a2_ok, json.dumps(grades95)), lines)

    z95_t3 = {k: v['z'] for k, v in
              r95['T3']['per_layer'].items()}
    a3_diff = max(abs(z95_t3[k] - FROZEN_2995_T3_Z[k])
                  for k in FROZEN_2995_T3_Z)
    a3_ok = bool(a3_diff < Z_TOL_2995)
    a['a3_z_diff'] = a3_diff
    log('a3 2995 T3 z vs frozen max|d|=%.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    # ---------- T2: audit evidence chain ----------
    r96 = json.load(open(RES_2996, encoding='utf-8'))
    s96 = json.load(open(SEAL_2996, encoding='utf-8'))
    led = json.load(open(LEDGER, encoding='utf-8'))
    tail = led['measurements'][-1]
    v96 = r96['final_verdict']
    h96 = s96['result_sha256_8']
    a4_ok = bool(
        v96 == FROZEN_2996_VERDICT
        and h96 == FROZEN_2996_RESULT_SHA8
        and tail.get('phase') == 2996
        and tail.get('hashes', {}).get('result_sha256_8')
        == FROZEN_2996_RESULT_SHA8
        and tail.get('verdict') == FROZEN_2996_VERDICT)
    a['a4_chain_ok'] = a4_ok
    log('a4 2996 verdict=%s seal=%s ledger tail phase=%s '
        'hash=%s verdict-match=%s ok=%s'
        % (v96, h96, tail.get('phase'),
           tail.get('hashes', {}).get('result_sha256_8'),
           tail.get('verdict') == FROZEN_2996_VERDICT,
           a4_ok), lines)

    z96_t3 = {k: v['z'] for k, v in
              r96['T3']['k2'].items()}
    a5_zdiff = max(abs(z96_t3[k] - FROZEN_2996_T3_Z[k])
                   for k in FROZEN_2996_T3_Z)
    glm_sig = bool(r96['T2']['glm_sig'])
    p_glm10 = float(r96['T2']['k1']['10']['p_share'])
    p_qwen6 = float(r96['T1']['k1']['6']['p_share'])
    a5_ok = bool(a5_zdiff < Z_TOL_2996 and glm_sig
                 and p_glm10 == 0.0 and p_qwen6 == 0.0)
    a['a5_z_diff'] = a5_zdiff
    a['a5_glm_sig'] = glm_sig
    a['a5_p_glm10'] = p_glm10
    a['a5_p_qwen6'] = p_qwen6
    log('a5 2996 T3.k2 z vs frozen max|d|=%.2e glm_sig=%s '
        'p_share glmL10=%.4f qwenL6=%.4f ok=%s'
        % (a5_zdiff, glm_sig, p_glm10, p_qwen6, a5_ok),
        lines)

    # ---------- T3: regrade registration ----------
    a6_ok = False
    cards_v21 = None
    if a1_ok and a2_ok and a3_ok and a4_ok and a5_ok:
        cards_v21 = []
        n_regraded = 0
        ok_map = True
        for card, old in FROZEN_2995_GRADES.items():
            key = (card, old)
            new = REGRADE_MAP.get(key, old)
            changed = new != old
            n_regraded += int(changed)
            if changed:
                if key not in REGRADE_MAP \
                        or new not in GRADE_VOCAB:
                    ok_map = False
            cards_v21.append({
                'card': card,
                'grade_old': old,
                'grade_new': new,
                'regraded': changed,
                'basis': {
                    'audit_phase': 2996,
                    'result_sha256_8':
                        FROZEN_2996_RESULT_SHA8,
                    'degeneracy_maxdiff': min(deg_attn,
                                              deg_mlp),
                    'audit_verdict': FROZEN_2996_VERDICT}})
        a6_ok = bool(ok_map and n_regraded == 1)
        a['a6_n_regraded'] = n_regraded
        a['a6_map_valid'] = ok_map
        log('a6 regrade n=%d map_valid=%s ok=%s'
            % (n_regraded, ok_map, a6_ok), lines)
    else:
        a['a6_n_regraded'] = None
        log('a6 skipped: evidence chain broken', lines)

    # ---------- verdict (explicit branches) ----------
    if not (a1_ok and a2_ok and a3_ok and a4_ok and a5_ok):
        verdict = 'evidence_chain_broken'
    elif not a6_ok:
        verdict = 'regrade_mapping_invalid'
    else:
        verdict = 'regrade_replicated_registered'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    # ---------- persist ----------
    res = {
        'phase': 2997,
        'final_verdict': verdict,
        'anchor_all_ok': bool(a1_ok and a2_ok and a3_ok
                              and a4_ok and a5_ok and a6_ok),
        'anchors': a,
        'degeneracy_finding': 'reproduced at artifact level: '
                              '2995 npz dirs_attn/dirs_mlp '
                              'all 40 rows identical '
                              '(max|diff|=%.1e / %.1e)'
                              % (deg_attn, deg_mlp),
        'cards_v21': cards_v21,
        'grade_vocab': sorted(GRADE_VOCAB),
        'regrade_map': {'%s:%s' % k: v for k, v
                        in REGRADE_MAP.items()},
        'tags': prereg['tags'],
        'elapsed_s': round(elapsed, 2),
        'correction_note': 'first run',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # registration file (card-set v2.1 sheet)
    with open(os.path.join(OUT, 'cards_v21.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'version': 'v2.1',
                   'based_on': 'phase2995 grades + '
                               'phase2996 audit',
                   'basis_sha256_8':
                       FROZEN_2996_RESULT_SHA8,
                   'cards': cards_v21},
                  f, indent=2, ensure_ascii=False)

    # small npz: artifact-level degeneracy evidence
    npz_path = os.path.join(
        OUT, 'omega_f2b_registry_regrade.npz')
    np.savez_compressed(
        npz_path,
        dirs_attn_row0=da[0].astype(np.float32),
        dirs_mlp_row0=dm[0].astype(np.float32),
        deg_attn=np.float64(deg_attn),
        deg_mlp=np.float64(deg_mlp),
        z95_t3=np.array([FROZEN_2995_T3_Z[k]
                         for k in ('7', '10', '13')]),
        z96_t3=np.array([FROZEN_2996_T3_Z[k]
                         for k in ('7', '10', '13')]))

    seal = {
        'npz_sha256_8': sha8(npz_path),
        'result_sha256_8': sha8(
            os.path.join(OUT, 'result.json')),
        'exec_sha256_8': sha8(
            os.path.join(OUT, 'execution.json')),
    }
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(seal, f, indent=2)
    log('sealed %s' % json.dumps(seal), lines)
    log('elapsed %.2fs' % elapsed, lines)


if __name__ == '__main__':
    main()
