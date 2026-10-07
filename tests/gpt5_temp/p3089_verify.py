# -*- coding: utf-8 -*-
"""Phase 3089 independent verify (post-closeout,
read-only except its own output file).

Recomputes and cross-checks:
  - seal 4x sha8 (npz / result / execution / script)
  - npz scalars (L_INJ=38, L_POST=39, SMOKE=False,
    SETUP_OK, REPRO_OK, FORWARDS>=20000, TIED=False)
  - family stats vs result.json (n_neg / med_c /
    r_all / top8) + b anchors all zero-diff OK
  - REPRO anchor independently re-derived from the
    SAME 3087 A1 npz L38 keys (MED_C_L38_/N_NEG_L38_/
    TOP8_L38_/CS1H_L38_/R_ALL_L38_), diff <= 1e-9,
    stored REPRO_* keys bit-equal to recomputation
  - spectrum class re-derivation (0.9 / 0.5)
  - E3 f2_cTT 6 tests: spearman + perm_p recomputed
    bit-level from stored arrays (frozen source
    extracted from the authoritative script)
  - G_DS gate + Stouffer z recomputed (clamped
    inv_cdf, order T_AB,T_AC,T_BC,U_AB,U_AC,U_BC)
  - SP_UT / MIG / SP_F1F2 recomputed bit-level
  - mig_state + verdict re-derivation from stored
    gates; G1 re-derivation
  - ledger n=228 / L14=196 / meas3089 hashes /
    ledger_sha256_8 recomputed
  - MEMO Phase 3089 (order after 3088, verdict,
    created, forwards, seal hashes, menu tail)
  - audit 51, wlog, workspace MEMORY max=3089,
    closeout log, run_log, stdout RUN_COMPLETE,
    stderr clean
  - reverse checks: 3087 L37 npz, 3088 verdict,
    smoke dir, A1 npz L38 keys

Usage:
  python p3089_verify.py              (full verify)
  python p3089_verify.py --selftest   (extract +
    toy-test spearman/perm_p only, no artifacts)
Writes p3089_verify_out.txt; exit 0 iff all PASS.
"""
import hashlib
import io
import json
import os
import sys
import traceback

import numpy as np
from statistics import NormalDist

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3089'
       r'\omega_p87_glm4_l38_full_arbitration')
R_A1 = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3087'
        r'\omega_p84_glm4_layer_scan')
R_87 = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3087'
        r'\omega_p85_glm4_l37_full_arbitration')
R_88 = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3088'
        r'\omega_p86_continuum_n12')
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3089_omega_p87_glm4_l38_'
          r'full_arbitration.py')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
AUDIT = (ROOT + r'\research\gpt5\docs'
         r'\hdmcc_knowledge_map_review_'
         r'20260921.md')
WLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-22.md')
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'
OUTF = ROOT + r'\tests\gpt5_temp' \
       r'\p3089_verify_out.txt'

fails = []
_n = [0]
o = []


def chk(cond, label):
    _n[0] += 1
    ok = bool(cond)
    o.append('[%02d] %s %s'
             % (_n[0], 'PASS' if ok else 'FAIL',
                label))
    if not ok:
        fails.append(_n[0])


def rd(path):
    return io.open(path,
                   encoding='utf-8').read()


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


def extract_stats_fns():
    """Extract the frozen spearman + perm_p
    source from the authoritative script and
    exec them in a controlled namespace."""
    src = rd(SCRIPT)
    i0 = src.index('def spearman(a, b):')
    i1 = src.index('def perm_p(a, b')
    i2 = src.index('\ndef ', i1 + 10)
    ns = {'np': np, 'N_PERM': 20000,
          'SEED': 3089}
    exec(src[i0:i2], ns)  # noqa: S102
    return ns['spearman'], ns['perm_p']


if '--selftest' in sys.argv:
    sp_fn, pp_fn = extract_stats_fns()
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 4.0, 7.0, 9.0, 11.0])
    assert abs(float(sp_fn(a, b)) - 1.0) < 1e-12
    p = float(pp_fn(a, b, n_perm=200))
    assert 0.0 <= p <= 1.0
    c = np.array([9.0, 7.0, 4.0, 2.0, 0.0])
    assert abs(float(sp_fn(a, c)) + 1.0) < 1e-12
    print('SELFTEST_OK')
    sys.exit(0)


def main():
    # ---------- A. identity & seal ----------
    res = json.load(io.open(
        RES + r'\result.json',
        encoding='utf-8'))
    seal = json.load(io.open(
        RES + r'\seal.json', encoding='utf-8'))
    exe = json.load(io.open(
        RES + r'\execution.json',
        encoding='utf-8'))
    npz_p = (RES + r'\omega_p87_glm4_l38_'
             r'full_arbitration.npz')
    z = np.load(npz_p, allow_pickle=False)
    verdict = res['verdict']
    fw = int(res['forwards'])
    created = res['created']

    chk(seal['phase'] == 3089
        and seal['name'] ==
        'omega_p87_glm4_l38_full_arbitration'
        and seal['setup_ok'] is True
        and seal['verdict'] == verdict,
        'seal.json phase/name/setup_ok/verdict')
    chk(sha8(npz_p) == seal['npz_sha256_8'],
        'sha8(npz) == seal.npz_sha256_8 (%s)'
        % seal['npz_sha256_8'])
    chk(sha8(RES + r'\result.json')
        == seal['result_sha256_8'],
        'sha8(result.json) == seal.result8')
    chk(sha8(RES + r'\execution.json')
        == seal['exec_sha256_8'],
        'sha8(execution.json) == seal.exec8')
    chk(sha8(SCRIPT) == seal['script_sha256_8'],
        'sha8(script) == seal.script8')
    chk(exe['created'] == created
        and seal['created'] == created
        and exe['smoke'] is False
        and exe['phase'] == 3089,
        'created consistent; execution smoke=False')
    chk(res['phase'] == 3089
        and res['name'] ==
        'omega_p87_glm4_l38_full_arbitration'
        and res['run'].startswith(
            'run1 authoritative'),
        'result.json phase/name/authoritative run')
    chk(str(z['VERDICT']) == verdict
        and verdict.startswith('fourth_l38_'),
        'npz VERDICT == result verdict (%s)'
        % verdict)
    chk(bool(z['SMOKE']) is False,
        'npz SMOKE False')
    chk(int(z['FORWARDS']) == fw and fw >= 20000,
        'npz FORWARDS == result.forwards (%d) '
        'and >= 20000' % fw)
    chk(int(z['L_INJ']) == 38
        and int(z['L_POST']) == 39,
        'npz L_INJ=38 / L_POST=39')
    chk(bool(z['SETUP_OK'])
        and bool(z['REPRO_OK']),
        'npz SETUP_OK and REPRO_OK True')
    chk(not bool(z['TIED']),
        'npz TIED False (untied embeddings)')
    chk(abs(float(z['ELAPSED'])
            - float(res['elapsed'])) < 2.0,
        'npz ELAPSED ~= result.elapsed')

    # smoke dir reverse check
    sm_res_p = (RES + r'\smoke' + r'\result.json')
    sm_ok = os.path.exists(sm_res_p)
    if sm_ok:
        sm = json.load(io.open(
            sm_res_p, encoding='utf-8'))
        sm_npz = (RES + r'\smoke'
                  r'\omega_p87_glm4_l38_full_'
                  r'arbitration.npz')
        smz = np.load(sm_npz,
                      allow_pickle=False)
        sm_ok = (sm['verdict'] == 'smoke_pending'
                 and bool(smz['SMOKE']) is True)
    chk(sm_ok,
        'smoke dir: verdict=smoke_pending, '
        'npz SMOKE True')

    # ---------- B. family stats + b anchors --
    for fk in 'ABC':
        rf = res['stats']['families'][fk]
        chk(int(rf['n_neg'])
            == int(z['N_NEG_' + fk])
            and float(rf['med_c'])
            == float(z['MED_C_' + fk])
            and float(rf['r_all'])
            == float(z['R_ALL_' + fk]),
            'family %s: result.json n_neg/med_c/'
            'r_all bit == npz' % fk)
        chk(list(rf['top8'])
            == [int(h) for h
                in z['TOP8_' + fk]],
            'family %s: top8 list == npz' % fk)
        oks = all(bool(z[k + fk])
                  for k in ('B0_OK_', 'B1_OK_',
                            'B3_OK_', 'B4_OK_',
                            'B6_OK_', 'B7A_OK_',
                            'B8_OK_'))
        zeros = all(float(z[k + fk])
                    == 0.0
                    for k in ('B0_DIFF_',
                              'B1_DIFF_',
                              'B4_DIFF_',
                              'B6_DIFF_',
                              'B7A_DIFF_',
                              'B8_DIFF_'))
        chk(oks and zeros,
            'family %s: b anchors all OK, '
            'diffs all bit-0' % fk)

    # ---------- C. REPRO vs 3087 A1 L38 -----
    za1 = np.load(
        R_A1 + r'\omega_p84_glm4_layer_scan.npz',
        allow_pickle=False)
    chk(all(('MED_C_L38_' + fk) in za1.files
            and ('N_NEG_L38_' + fk) in za1.files
            and ('TOP8_L38_' + fk) in za1.files
            and ('CS1H_L38_' + fk) in za1.files
            and ('R_ALL_L38_' + fk) in za1.files
            for fk in 'ABC'),
        'A1 npz has all five L38 key families')
    if 'REPRO_MEDC_DIFF_A' in z.files:
        for fk in 'ABC':
            d_mc = abs(float(z['MED_C_' + fk])
                       - float(
                           za1['MED_C_L38_'
                               + fk]))
            chk(d_mc
                == float(z['REPRO_MEDC_DIFF_'
                           + fk])
                and d_mc <= 1e-9,
                'repro %s: med_c diff %.3e '
                'bit==stored, <=1e-9'
                % (fk, d_mc))
            nn = (int(z['N_NEG_' + fk])
                  == int(za1['N_NEG_L38_'
                             + fk]))
            chk(nn
                and bool(z['REPRO_NNEG_OK_'
                           + fk]),
                'repro %s: n_neg exact == A1 L38'
                % fk)
            t8 = ([int(h) for h
                   in z['TOP8_' + fk]]
                  == [int(h) for h
                      in za1['TOP8_L38_'
                             + fk]])
            chk(t8
                and bool(z['REPRO_TOP8_OK_'
                           + fk]),
                'repro %s: top8 exact == A1 L38'
                % fk)
            d_cs = float(np.max(np.abs(
                z['CS1H_' + fk]
                - za1['CS1H_L38_' + fk])))
            chk(d_cs
                == float(z['REPRO_CS1H_DIFF_'
                           + fk])
                and d_cs <= 1e-9,
                'repro %s: CS1H diff %.3e '
                'bit==stored, <=1e-9'
                % (fk, d_cs))
            d_ra = abs(float(z['R_ALL_' + fk])
                       - float(
                           za1['R_ALL_L38_'
                               + fk]))
            chk(d_ra
                == float(z['REPRO_RALL_DIFF_'
                           + fk])
                and d_ra <= 1e-9,
                'repro %s: R_ALL diff %.3e '
                'bit==stored, <=1e-9'
                % (fk, d_ra))
            rr = res['stats']['repro'][fk]
            chk(float(rr['medc_diff']) == d_mc
                and float(rr['cs1h_diff'])
                == d_cs
                and float(rr['rall_diff'])
                == d_ra
                and rr['ok'] is True,
                'repro %s: result.json stats '
                'bit == npz REPRO_*' % fk)
    else:
        chk(False, 'REPRO keys present in npz')

    # ---------- D. cross statistics --------
    has_cross = 'T_AB' in z.files
    DEGEN = verdict in (
        'fourth_l38_setup_failed',
        'fourth_l38_top8_degenerate')
    if DEGEN:
        chk(res['stats']['cross'] is None,
            'degenerate verdict: cross stats '
            'None in result.json')
        chk(not bool(z['TOP8_ALL_OK']),
            'degenerate verdict: TOP8_ALL_OK '
            'False')
    else:
        chk(has_cross
            and bool(z['TOP8_ALL_OK']),
            'non-degenerate: cross keys present, '
            'TOP8_ALL_OK True')
    if has_cross and not DEGEN:
        sp_fn, pp_fn = extract_stats_fns()
        # spectrum class re-derivation
        t3s = {fk: float(z['E3_TOP3_CS_' + fk])
               for fk in 'ABC'}
        if min(t3s.values()) >= 0.9:
            cls = 'trunk'
        elif max(t3s.values()) <= 0.5:
            cls = 'dispersed'
        else:
            cls = 'mixed'
        chk(str(z['SPEC_CLASS']) == cls
            and res['stats']['cross']
            ['spec_class'] == cls
            and res['gates']['spec_class'] == cls,
            'spec_class re-derived %s (top3 '
            '%.4f/%.4f/%.4f)'
            % (cls, t3s['A'], t3s['B'],
               t3s['C']))
        for fk in 'ABC':
            chk(float(res['gates']['spectrum']
                      [fk]['top3'])
                == t3s[fk]
                and float(res['stats']['cross']
                          ['spectrum'][fk]
                          ['top3']) == t3s[fk],
                'family %s: spectrum top3 bit '
                'consistent (gates+stats)' % fk)
        # E3 f2_cTT 6 tests recomputed
        e2 = {}
        for rn in ('T', 'U'):
            for key in ('AB', 'AC', 'BC'):
                fa = z['F2_CTT_' + key]
                rb = z[rn + '_' + key]
                s_mine = float(
                    sp_fn(fa, rb))
                p_mine = float(
                    pp_fn(fa, rb))
                k = rn + '_' + key
                e2[k] = (s_mine, p_mine)
                chk(s_mine
                    == float(z['E3_F2_CTT_'
                               + k]),
                    'E3 f2_cTT~%s sp bit '
                    'recomputed (%+.4f)'
                    % (k, s_mine))
                chk(p_mine
                    == float(z['E3P_F2_CTT_'
                               + k]),
                    'E3 f2_cTT~%s perm_p bit '
                    'recomputed (%.5f)'
                    % (k, p_mine))
        # G_DS gate + Stouffer
        cnt = sum(1 for s_, p_ in e2.values()
                  if s_ > 0 and p_ < 0.05)
        n_bonf = sum(1 for _, p_ in
                     e2.values() if p_
                     < 0.05 / 6)
        min_sp = min(s_ for s_, _ in
                     e2.values())
        g_ds = cnt >= 4 and min_sp > 0
        zl = [NormalDist().inv_cdf(
            min(1.0 - 1e-12, 1.0 - p_))
            for s_, p_ in e2.values()]
        stz = float(np.sum(zl)
                    / np.sqrt(len(zl)))
        chk(int(z['GDS_COUNT']) == cnt,
            'GDS_COUNT bit recomputed (%d/6)'
            % cnt)
        chk(int(z['GDS_N_BONF']) == n_bonf,
            'GDS_N_BONF bit recomputed (%d)'
            % n_bonf)
        chk(float(z['GDS_MIN_SP']) == min_sp,
            'GDS_MIN_SP bit recomputed '
            '(%.4f)' % min_sp)
        chk(float(z['STOUFFER_Z']) == stz,
            'STOUFFER_Z bit recomputed '
            '(%.4f)' % stz)
        g = res['gates']
        chk(g['G_DS'] == g_ds
            and g['count_sig_pos'] == cnt
            and g['n_bonf'] == n_bonf
            and float(g['min_sp']) == min_sp
            and float(g['stouffer_z']) == stz,
            'result.gates G_DS block bit '
            'consistent')
        chk(float(g['f2_TAB_sp'])
            == e2['T_AB'][0]
            and float(g['f2_TAB_p'])
            == e2['T_AB'][1],
            'gates f2_TAB sp/p bit consistent')
        # SP_UT / MIG / SP_F1F2 recompute
        for key in ('AB', 'AC', 'BC'):
            chk(float(z['SP_UT_' + key])
                == float(sp_fn(z['U_' + key],
                               z['T_' + key])),
                'SP_UT_%s bit recomputed'
                % key)
        for fa, fb in (('A', 'B'), ('A', 'C'),
                       ('B', 'C')):
            key = fa + fb
            chk(float(z['MIG_' + key])
                == float(sp_fn(
                    z['R1_ALLNH_' + fa],
                    z['R1_ALLNH_' + fb])),
                'MIG_%s bit recomputed' % key)
            chk(float(z['SP_F1F2_' + key])
                == float(sp_fn(
                    z['F1_STT_' + key],
                    z['F2_CTT_' + key])),
                'SP_F1F2_%s bit recomputed'
                % key)
        # verdict re-derivation
        p_tab = e2['T_AB'][1]
        if g_ds and p_tab < 0.05:
            mig_state = 'locked'
        elif g_ds:
            mig_state = 'partial'
        else:
            mig_state = 'absent'
        if cls == 'mixed':
            v2 = 'fourth_l38_mixed_' + mig_state
        elif mig_state == 'absent':
            v2 = ('fourth_l38_trunk_no_migrate'
                  if cls == 'trunk' else
                  'fourth_l38_dispersed_no_'
                  'migrate')
        else:
            v2 = ('fourth_l38_trunk_migrates'
                  if cls == 'trunk' else
                  'fourth_l38_dispersed_'
                  'migrates')
        chk(v2 == verdict,
            'verdict re-derived from stored '
            'gates: %s' % v2)
        # G1 re-derivation
        best = None
        for key in ('AB', 'AC', 'BC'):
            for rn in ('T', 'U'):
                s_ = float(z['E3_F1_STT_%s_%s'
                             % (rn, key)])
                p_ = float(z['E3P_F1_STT_%s_%s'
                             % (rn, key)])
                cand = (abs(s_), s_, p_, rn,
                        key)
                if best is None \
                        or cand[0] > best[0]:
                    best = cand
        g1 = best[0] >= 0.5 and best[2] < 0.05
        f1_pos = g1 and best[1] > 0
        chk(float(g['G1_sp']) == best[1]
            and float(g['G1_p']) == best[2]
            and g['G1_resp'] == best[3]
            and g['G1_pair'] == best[4]
            and g['G1'] == g1
            and g['f1_sign_positive'] == f1_pos,
            'G1 block re-derived (best f1 '
            'sp=%+.4f on %s_%s)'
            % (best[1], best[3], best[4]))

    # ---------- E. ledger ----------
    led = json.load(io.open(LEDGER,
                            encoding='utf-8'))
    stored_sha = led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    calc = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = stored_sha
    chk(calc == stored_sha,
        'ledger_sha256_8 recomputed (%s)'
        % stored_sha)
    chk(len(led['measurements']) == 228,
        'ledger measurements n=228 (got %d)'
        % len(led['measurements']))
    ms = [m for m in led['measurements']
          if isinstance(m, dict)
          and m.get('phase') == 3089]
    chk(len(ms) == 1
        and ms[0]['meas_id']
        == 'meas3089_omega_p87_glm4_l38_replica'
        and ms[0]['verdict'] == verdict,
        'meas3089 present once, verdict match')
    chk(ms[0]['hashes']['npz_sha256_8']
        == seal['npz_sha256_8']
        and ms[0]['hashes']['result_sha256_8']
        == seal['result_sha256_8']
        and ms[0]['hashes']['script_sha256_8']
        == seal['script_sha256_8'],
        'meas3089 hashes == seal')
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model']
    chk(len(l14) == 1
        and len(l14[0]['connects']) == 196,
        'L14 connects n=196 (got %d)'
        % (len(l14[0]['connects'])
           if l14 else -1))
    if l14 and l14[0]['connects']:
        last = l14[0]['connects'][-1]
        chk(last.get('phase') == 3089
            and last.get('meas_id')
            == 'meas3089_omega_p87_glm4_l38_'
               'replica',
            'L14 last connect is meas3089')

    # ---------- F. MEMO ----------
    memo = rd(MEMO)
    i3088 = memo.index('## Phase 3088:')
    i3089 = memo.index('## Phase 3089:')
    chk(i3089 > i3088,
        'MEMO Phase 3089 after 3088 (append-'
        'only order)')
    chk('## Phase 3090:' not in memo,
        'MEMO has no premature Phase 3090')
    tail = memo[i3089:]
    chk(verdict in tail
        and created in tail
        and str(fw) in tail,
        'MEMO 3089 section has verdict/created/'
        'forwards')
    chk(seal['npz_sha256_8'] in tail
        and seal['script_sha256_8'] in tail
        and seal['exec_sha256_8'] in tail,
        'MEMO 3089 section has seal hashes')
    chk(('| A |' in tail) and ('| AB |' in tail)
        and tail.rstrip().endswith(
            '第五谱点。'),
        'MEMO 3089 tables + menu tail intact')

    # ---------- G. audit / wlog / MEMORY ---
    aud = rd(AUDIT)
    i50 = aud.index('## 五十、')
    i51 = aud.index('## 五十一、3089')
    chk(i51 > i50 and verdict in aud[i51:],
        'audit 51 appended after 50, verdict '
        'present')
    chk('Phase 3089' in rd(WLOG)
        and verdict in rd(WLOG),
        'workspace daily log has Phase 3089')
    chk('max=3089' in rd(WMEM),
        'workspace MEMORY max=3089')

    # ---------- H. logs ----------
    clog = rd(RES + r'\closeout_log.txt')
    chk('ledger appended n=228' in clog
        or 'ledger already upserted n=228'
        in clog, 'closeout: ledger step logged')
    chk('memo +' in clog
        or 'memo already appended' in clog,
        'closeout: memo step logged')
    chk('audit addendum +' in clog
        or 'audit already appended' in clog,
        'closeout: audit step logged')
    chk('wlog appended' in clog
        or 'wlog already' in clog,
        'closeout: wlog step logged')
    chk('memory updated' in clog
        or 'memory already max=3089' in clog,
        'closeout: memory step logged')
    rl = rd(RES + r'\run_log.txt')
    chk(('VERDICT: ' + verdict) in rl
        and ('sealed npz8='
             + seal['npz_sha256_8']) in rl,
        'run_log has VERDICT + sealed line')
    chk(rl.count('vs 3087A1 L38') == 3,
        'run_log has 3 repro lines (A/B/C)')
    so = rd(ROOT + r'\tests\gpt5_temp'
            r'\p3089_run_stdout.txt')
    chk(('RUN_COMPLETE ' + verdict) in so,
        'stdout ends with RUN_COMPLETE')
    se = rd(ROOT + r'\tests\gpt5_temp'
            r'\p3089_run_stderr.txt')
    chk('Traceback' not in se
        and 'Error' not in se.replace(
            'Errors', ''),
        'stderr free of Traceback/Error')

    # ---------- I. reverse checks ----------
    res87 = json.load(io.open(
        R_87 + r'\result.json',
        encoding='utf-8'))
    chk(res87['verdict'].startswith('fourth_'),
        '3087 L37 verdict is fourth_* (%s)'
        % res87['verdict'])
    z87 = np.load(
        R_87 + r'\omega_p85_glm4_l37_full_'
        r'arbitration.npz',
        allow_pickle=False)
    chk('T_AB' in z87.files
        and 'MED_C_A' in z87.files,
        '3087 L37 npz has comparison keys')
    res88 = json.load(io.open(
        R_88 + r'\result.json',
        encoding='utf-8'))
    chk(res88['verdict']
        == 'cross_architecture_confirmed',
        '3088 continuum verdict intact')


try:
    main()
    o.append('')
    o.append('TOTAL %d checks, %d FAIL'
             % (_n[0], len(fails)))
    o.append('VERIFY_ALL_PASS'
             if not fails else
             'VERIFY_FAILED at %s' % fails)
except Exception:
    o.append('VERIFY_CRASHED:')
    o.append(traceback.format_exc())
    fails.append(-1)

with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('VERIFY_DONE fails=%d' % len(fails))
sys.exit(0 if not fails else 1)
