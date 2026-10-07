# -*- coding: utf-8 -*-
"""Phase 3082 independent verify (no GPU, no RNG):
1) seal 4xsha8 + input-sha8 refreeze;
2) npz replay of ALL analytic quantities from
   the sealed source npz (E1 health, E2 top8
   overlap + hypergeometric SF, E3 SVD PR/keff/
   top3, E4 dispersion, E5 f2 med / sp(f1,f2) /
   TT norm, E6, SP_AS bit, TREE, verdict);
3) Ledger sha + counts + meas3082 content;
4) MEMO/audit/wlog/MEMORY presence.
Writes report; stdout unreliable on this host.
"""
import hashlib
import io
import json
from math import comb, sqrt

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = BASE + (r'\phase3082\omega_p79_'
            r'ds7b_negative_anatomy')
NPZ81 = BASE + (r'\phase3081\omega_p78_'
                r'ds7b_crossmodel'
                r'\omega_p78_ds7b_crossmodel.npz')
NPZ76 = BASE + (r'\phase3076\omega_p73_'
                r'cross_prompt_family'
                r'\omega_p73_cross_prompt_family'
                r'.npz')
NPZ79 = BASE + (r'\phase3079\omega_p76_'
                r'migration_lock'
                r'\omega_p76_migration_lock.npz')
NPZ80 = BASE + (r'\phase3080\omega_p77_'
                r'ab_anatomy'
                r'\omega_p77_ab_anatomy.npz')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-22.md'
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'
REP = (ROOT + r'\tests\gpt5_temp'
       r'\p3082_verify_report.txt')
o = []
fails = []


def chk(name, cond, detail=''):
    o.append('%s %s %s'
             % ('PASS' if cond else 'FAIL',
                name, detail))
    if not cond:
        fails.append(name)


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def hypergeom_sf(x, N, K, n):
    lo = max(x, 0)
    hi = min(K, n)
    tot = comb(N, n)
    s = 0
    for k in range(lo, hi + 1):
        s += comb(K, k) * comb(N - K, n - k)
    return s / tot


def spectrum_struct(M):
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    lam = s * s
    tot = float(lam.sum())
    pr = float(lam.sum() ** 2
               / (lam * lam).sum())
    p = lam / tot
    p = p[p > 0]
    keff = float(np.exp(
        -(p * np.log(p)).sum()))
    top3 = float(lam[:3].sum() / tot)
    return pr, keff, top3


# ---------- 1) hashes ----------
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
for key, path in (
        ('npz_sha256_8', R + '\\' +
         'omega_p79_ds7b_negative_anatomy'
         '.npz'),
        ('result_sha256_8',
         R + r'\result.json'),
        ('script_sha256_8',
         ROOT + r'\tests\glm5\phase3082_omega_'
         r'p79_ds7b_negative_anatomy.py'),
        ('exec_sha256_8',
         R + r'\execution.json')):
    h = sha8(path)
    chk('seal_' + key, h == seal[key],
        '%s vs %s' % (h, seal[key]))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
in_sha = {
    'npz3081_ds7b': sha8(NPZ81),
    'npz3076_4b': sha8(NPZ76),
    'npz3079_4b': sha8(NPZ79),
    'npz3080_4b': sha8(NPZ80),
}
chk('input_sha_refreeze',
    exe['prereg']['input_sha8'] == in_sha)

# ---------- 2) npz replay ----------
res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
Z = np.load(R + r'\omega_p79_ds7b_negative_'
            r'anatomy.npz', allow_pickle=True)
Z81 = np.load(NPZ81, allow_pickle=True)
Z76 = np.load(NPZ76, allow_pickle=True)
Z79 = np.load(NPZ79, allow_pickle=True)
Z80 = np.load(NPZ80, allow_pickle=True)
chk('verdict', str(Z['VERDICT'])
    == res['verdict'] == 'ds7b_decorrelated',
    str(Z['VERDICT']))
chk('forwards0', int(res['forwards']) == 0)
chk('tree',
    {'D': bool(Z['TREE_D']),
     'R1': bool(Z['TREE_R1']),
     'R2': bool(Z['TREE_R2']),
     'R3': bool(Z['TREE_R3'])}
    == {'D': True, 'R1': True, 'R2': True,
        'R3': True})

FK = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
HDR = {'DS7B': 28, '4B': 32}
ZS = {'DS7B': Z81, '4B': Z76}
S = res['stats']


def close(name, a, b, tol=0.0):
    if abs(a - b) <= tol:
        chk(name, True, 'bit')
    else:
        chk(name, False,
            '%r vs %r' % (a, b))


# E1 + E2 + E3 replay per model/family
for mdl in ('DS7B', '4B'):
    z = ZS[mdl]
    h = HDR[mdl]
    for fk in FK:
        r1 = np.asarray(
            z['R1_ALL%d_%s' % (h, fk)],
            dtype=np.float64)
        medc = float(
            z[('MED_C_' if mdl == 'DS7B'
               else 'MED_C_34_') + fk])
        e1 = S['E1'][mdl + '_' + fk]
        close('E1_r1min_%s_%s' % (mdl, fk),
              float(r1.min()),
              e1['r1_min'])
        close('E1_medc_%s_%s' % (mdl, fk),
              medc, e1['med_c'])
        close('E1_depth_%s_%s' % (mdl, fk),
              abs(float(r1.min())) / medc,
              e1['depth'], 1e-12)
        close('E1_cap8_%s_%s' % (mdl, fk),
              float(z['CAPTURE8_' + fk]),
              e1['capture8'])
        close('E1_rall_%s_%s' % (mdl, fk),
              float(z['R_ALL_' + fk]),
              e1['R_ALL'])
        n_neg = int((r1 < 0).sum())
        chk('E1_nneg_%s_%s' % (mdl, fk),
            n_neg == e1['n_neg']
            == int(z['N_NEG_' + fk]))
        top8 = [int(v) for v in
                np.argsort(r1)[:min(8, n_neg)]]
        chk('E2_top8replay_%s_%s'
            % (mdl, fk),
            top8 == [int(v) for v
                     in z['TOP8_' + fk]])
        close('E3_PR_CS_%s_%s' % (mdl, fk),
              spectrum_struct(
                  np.asarray(z['CS_' + fk],
                             dtype=np.float64))[0],
              S['E3'][mdl + '_CS_' + fk]['PR'],
              1e-12)
        close('E3_TOP3_CS_%s_%s' % (mdl, fk),
              spectrum_struct(
                  np.asarray(z['CS_' + fk],
                             dtype=np.float64))[2],
              S['E3'][mdl + '_CS_'
                      + fk]['top3'], 1e-12)
        close('E3_KEFF_CS1H_%s_%s'
              % (mdl, fk),
              spectrum_struct(np.asarray(
                  z['CS1H_' + fk],
                  dtype=np.float64))[1],
              S['E3'][mdl + '_CS1H_'
                      + fk]['keff'], 1e-12)
        for fa, fb in CPAIRS:
            key = fa + fb
            if fk == 'A':
                ov = len(set(top8) & set(
                    int(v) for v in
                    z['TOP8_' + fb]))
            # (overlap handled below per pair)

# E2 overlap + hypergeom replay per model/pair
for mdl in ('DS7B', '4B'):
    z = ZS[mdl]
    h = HDR[mdl]
    for fa, fb in CPAIRS:
        key = fa + fb
        t8a = [int(v) for v
               in z['TOP8_' + fa]]
        t8b = [int(v) for v
               in z['TOP8_' + fb]]
        ov = len(set(t8a) & set(t8b))
        p = hypergeom_sf(ov, h, 8, 8)
        close('E2_ov_%s_%s' % (mdl, key),
              float(ov),
              float(Z['E2_OV_' + mdl + '_'
                      + key]))
        close('E2_p_%s_%s' % (mdl, key), p,
              float(Z['E2_P_' + mdl + '_'
                      + key]), 1e-15)
        chk('E2_stats_%s_%s' % (mdl, key),
            S['E2'][mdl + '_' + key]
            ['overlap'] == ov
            and abs(S['E2'][mdl + '_'
                            + key]['sf_p']
                    - p) < 1e-15)

# E4 dispersion replay
NOISE = 1.0 / sqrt(23.0)
for tag, zp in (('4B', Z79), ('DS7B', Z81)):
    for rn in ('T', 'U'):
        for fa, fb in CPAIRS:
            key = fa + fb
            arr = np.asarray(
                zp[rn + '_' + key],
                dtype=np.float64)
            sd = float(arr.std(ddof=1))
            close('E4_std_%s_%s_%s'
                  % (tag, rn, key), sd,
                  float(Z['E4_STD_%s_%s_%s'
                          % (rn, tag, key)]),
                  1e-15)
            e4 = S['E4']['%s_%s_%s'
                         % (tag, rn, key)]
            chk('E4_snr_%s_%s_%s'
                % (tag, rn, key),
                abs(sd / NOISE
                    - e4['snr']) < 1e-12)

# E5 replay
for tag, zp, fk2 in (('4B', Z79, None),
                     ('DS7B', Z81, None)):
    for fa, fb in CPAIRS:
        key = fa + fb
        f2 = np.asarray(
            zp['F2_CTT_' + key],
            dtype=np.float64)
        f1 = np.asarray(
            zp['F1_STT_' + key],
            dtype=np.float64)
        close('E5_medf2_%s_%s' % (tag, key),
              float(np.median(f2)),
              float(Z['E5_MED_F2_' + tag
                      + '_' + key]), 1e-15)
        close('E5_spf1f2_%s_%s'
              % (tag, key),
              spearman(f1, f2),
              float(Z['E5_SPF1F2_' + tag
                      + '_' + key]), 1e-12)
for tag, zp, tkey in (('4B', Z80,
                       'TT_NORM_MED_'),
                      ('DS7B', Z81,
                       'MED_TT_NORM_')):
    for fk in FK:
        close('E5_ttnorm_%s_%s' % (tag, fk),
              float(zp[tkey + fk]),
              float(Z['E5_TTNORM_' + tag
                      + '_' + fk]), 1e-9)

# E6 replay from source npz
close('E6_sp_4B',
      float(Z79['E3_F2_CTT_U_AB']),
      float(Z['E6_SP_4B']), 0.0)
close('E6_p_4B',
      float(Z79['E3P_F2_CTT_U_AB']),
      float(Z['E6_P_4B']), 0.0)
close('E6_sp_DS7B',
      float(Z81['E3_F2_CTT_U_AB']),
      float(Z['E6_SP_DS7B']), 0.0)
close('E6_p_DS7B',
      float(Z81['E3P_F2_CTT_U_AB']),
      float(Z['E6_P_DS7B']), 0.0)
chk('E6_both_sig',
    S['E6']['both_significant'] is True)

# SP_AS bit replay (A_S sources: 4B from
# 3076 npz, DS7B from 3081 npz)
for mdl, zp in (('DS7B', Z81), ('4B', Z76)):
    for fa, fb in CPAIRS:
        key = fa + fb
        v = spearman(
            np.asarray(zp['A_S_' + fa],
                       dtype=np.float64),
            np.asarray(zp['A_S_' + fb],
                       dtype=np.float64))
        close('SP_AS_%s_%s' % (mdl, key), v,
              float(Z['SP_AS_' + mdl + '_'
                      + key]), 0.0)

# tree re-evaluation from replayed numbers
D = all(
    spectrum_struct(np.asarray(
        ZS['DS7B']['CS_' + fk],
        dtype=np.float64))[0]
    - spectrum_struct(np.asarray(
        ZS['4B']['CS_' + fk],
        dtype=np.float64))[0] >= 0.15
    for fk in FK)
R1 = all(hypergeom_sf(
    len(set(int(v) for v in ZS['DS7B']
            ['TOP8_' + fa])
        & set(int(v) for v in ZS['DS7B']
              ['TOP8_' + fb])), 28, 8, 8)
    > 0.05 for fa, fb in CPAIRS) and any(
    hypergeom_sf(len(set(int(v) for v in
                         ZS['4B']['TOP8_'
                         + fa])
                      & set(int(v) for v in
                            ZS['4B']['TOP8_'
                            + fb])), 32, 8, 8)
    < 0.05 for fa, fb in CPAIRS)
R2 = all(float(Z['SP_AS_DS7B_' + fa + fb])
         < 0.2 for fa, fb in CPAIRS) and all(
    float(Z['SP_AS_4B_' + fa + fb]) > 0.4
    for fa, fb in CPAIRS)
R3 = all(float(ZS['DS7B']['CAPTURE8_' + fk])
         >= 0.5 for fk in FK)
chk('tree_replay',
    D and R1 and R2 and R3,
    'D=%s R1=%s R2=%s R3=%s'
    % (D, R1, R2, R3))
vr = ('ds7b_decorrelated' if D else
      'ds7b_family_specific_routing'
      if (R1 and R2 and R3)
      else 'ds7b_mixed_anatomy')
chk('verdict_replay',
    vr == 'ds7b_decorrelated', vr)

# ---------- 3) Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('ledger_n', len(led['measurements']) == 221,
    str(len(led['measurements'])))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
chk('ledger_l14', len(l14['connects']) == 189,
    str(len(l14['connects'])))
m82 = [m for m in led['measurements']
       if isinstance(m, dict)
       and m.get('phase') == 3082]
chk('ledger_meas3082_present',
    len(m82) == 1, str(len(m82)))
if m82:
    chk('meas3082_verdict',
        m82[0]['verdict']
        == 'ds7b_decorrelated')
    chk('meas3082_hashes',
        m82[0]['hashes']['npz_sha256_8']
        == seal['npz_sha256_8'])
l14c = [c for c in l14['connects']
        if isinstance(c, dict)
        and c.get('phase') == 3082]
chk('l14_conn3082', len(l14c) == 1)
ref_sha = led.get('ledger_sha256_8')
led2 = json.loads(json.dumps(
    led, ensure_ascii=False))
led2.pop('ledger_sha256_8', None)
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
h = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
chk('ledger_sha', h == ref_sha == '6e4060d0',
    '%s vs %s' % (h, ref_sha))

# ---------- 4) docs ----------
memo = io.open(MEMO, encoding='utf-8').read()
chk('memo_phase3082',
    '## Phase 3082:' in memo
    and 'ds7b_decorrelated' in memo
    and '[2026-09-22 00:58:12]' in memo)
chk('memo_menu3083', '3083' in memo)
aud = io.open(AUDIT, encoding='utf-8').read()
chk('audit_44',
    u'四十四、3082' in aud
    and 'ds7b_decorrelated' in aud)
wl = io.open(WLOG, encoding='utf-8').read()
chk('wlog_3082', 'Phase 3082' in wl)
wm = io.open(WMEM, encoding='utf-8').read()
chk('wmem_max3082', 'max=3082' in wm)
chk('wmem_len', len(wm) < 3000,
    str(len(wm)))

# ---------- summary ----------
n_pass = len([x for x in o
              if x.startswith('PASS')])
o.append('')
o.append('TOTAL %d PASS %d FAIL'
         % (n_pass, len(fails)))
if fails:
    o.append('FAILED: ' + ', '.join(fails))
    o.append('VERIFY_FAIL')
else:
    o.append('VERIFY_OK')
io.open(REP, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
