# -*- coding: utf-8 -*-
"""Phase 3081 independent verify (no GPU):
1) seal 4xsha8 recomputed from disk bytes;
2) npz bit replay: r1_nh, A_S=-R_S, R_S median,
   top8, capture8, T/U/F1/F2/F3/F4/F5, mig,
   sp_ut, sp_f1f2, tt_med;
3) E3 sp replay (all 30) + perm replay (2 tests,
   seed 3081, n_perm 20000) + partial + G_DS/G1/
   verdict replay;
4) Ledger sha + counts + meas3081 content;
5) MEMO/audit/wlog/MEMORY presence.
Writes report to tests/gpt5_temp/
p3081_verify_report.txt (stdout unreliable)."""
import hashlib
import io
import json
import numpy as np
from statistics import NormalDist

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3081'
     r'\omega_p78_ds7b_crossmodel')
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
       r'\p3081_verify_report.txt')
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


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


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


def perm_p(a, b, n_perm=20000, seed=3081):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (rb * ra[None, :]).sum(axis=1)
    den = np.sqrt(
        (rb * rb).sum(axis=1)
        * float((ra * ra).sum()))
    den[den == 0] = 1.0
    stats = np.abs(num / den)
    return float((stats >= obs - 1e-12)
                 .mean())


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()
                  * (b * b).sum())
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)


def ranks(x):
    x = np.asarray(x, np.float64)
    return np.argsort(np.argsort(x)) \
        .astype(np.float64)


def partial_sp(x, y, z):
    rx = ranks(x)
    ry = ranks(y)
    rz = ranks(z)
    r_xy = pearson(rx, ry)
    r_xz = pearson(rx, rz)
    r_yz = pearson(ry, rz)
    den = np.sqrt(
        (1.0 - r_xz * r_xz)
        * (1.0 - r_yz * r_yz))
    if den == 0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / den)


def partial_sp_perm_p(x, y, z,
                      n_perm=20000, seed=3081):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    obs = abs(partial_sp(x, y, z))
    Y = np.tile(y, (n_perm, 1))
    Y = rng.permuted(Y, axis=1)
    rx = ranks(x)
    rz = ranks(z)
    r_xz = pearson(rx, rz)
    cnt = 0
    for i in range(n_perm):
        ry = ranks(Y[i])
        r_xy = pearson(rx, ry)
        r_yz = pearson(ry, rz)
        den = np.sqrt(
            (1.0 - r_xz * r_xz)
            * (1.0 - r_yz * r_yz))
        val = 0.0 if den == 0 else \
            (r_xy - r_xz * r_yz) / den
        if abs(val) >= obs - 1e-12:
            cnt += 1
    # 3081 main script returns cnt/n_perm
    # (no +1 smoothing; verify must match bit)
    return float(cnt) / n_perm


# ---------- 1) seal hashes ----------
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
seal_items = [
    ('npz_sha256_8',
     R + r'\omega_p78_ds7b_crossmodel.npz'),
    ('result_sha256_8', R + r'\result.json'),
    ('script_sha256_8',
     ROOT + r'\tests\glm5\phase3081_omega_'
     r'p78_ds7b_crossmodel.py'),
    ('exec_sha256_8',
     R + r'\execution.json'),
]
for key, path in seal_items:
    h = sha8(path)
    chk('seal_' + key, h == seal[key],
        '%s vs %s' % (h, seal[key]))

# ---------- 2) identity ----------
res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
exe = json.load(io.open(R + r'\execution.json',
                        encoding='utf-8'))
Z = np.load(R + r'\omega_p78_ds7b_crossmodel'
            r'.npz', allow_pickle=True)
chk('verdict', str(Z['VERDICT'])
    == res['verdict'] == 'ds7b_cos_absent',
    str(Z['VERDICT']))
chk('forwards', int(Z['FORWARDS']) == 20634
    == res['forwards'])
chk('smoke', bool(Z['SMOKE']) is False)
chk('setup_ok_all', bool(Z['SETUP_OK']) is True)
chk('top8_all_ok',
    bool(Z['TOP8_ALL_OK']) is True)

# ---------- 3) npz bit replay ----------
FK = ('A', 'B', 'C')
MED_C = {f: float(Z['MED_C_' + f]) for f in FK}
CS1H = {f: np.asarray(Z['CS1H_' + f],
                      dtype=np.float64) for f in FK}
CS = {f: np.asarray(Z['CS_' + f],
                    dtype=np.float64) for f in FK}
TT = {f: np.asarray(Z['TT_' + f],
                    dtype=np.float64) for f in FK}
LG = {f: np.asarray(Z['LG_' + f],
                    dtype=np.float64) for f in FK}


def bits(name, a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        chk(name, False,
            'shape %s vs %s'
            % (a.shape, b.shape))
        return
    if (a == b).all():
        chk(name, True, 'bit')
    else:
        d = float(np.max(np.abs(a - b)))
        chk(name, d < 1e-12, 'maxdiff=%.3e' % d)


R1 = {}
for f in FK:
    r1 = np.median(CS1H[f], axis=1) - MED_C[f]
    R1[f] = r1
    bits('r1_nh_' + f, r1, Z['R1_ALL28_' + f])
    n_neg = int((r1 < 0).sum())
    chk('n_neg_' + f, n_neg
        == int(Z['N_NEG_' + f]), str(n_neg))
    order = np.argsort(r1)
    topk = min(8, n_neg)
    top8 = [int(h) for h in order[:topk]]
    sel_ok = bool(topk == 8
                  and bool((r1[top8] < 0).all()))
    chk('top8_' + f,
        top8 == [int(h) for h
                 in Z['TOP8_' + f]],
        str(top8))
    chk('top8_sel_ok_' + f, sel_ok
        == bool(Z['TOP8_SEL_OK_' + f]))
    cap = abs(float(r1[top8].sum())) \
        / abs(float(r1[r1 < 0].sum()))
    d = abs(cap - float(Z['CAPTURE8_' + f]))
    chk('capture8_' + f, d < 1e-10,
        'diff=%.2e' % d)
    rs = np.asarray(Z['R_S_' + f],
                    dtype=np.float64)
    bits('A_S_eq_negRS_' + f,
         -rs, Z['A_S_' + f])
    bits('R_S_median_' + f,
         np.median(CS[f], axis=1) - MED_C[f], rs)

CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
CIDX = np.array([k // 8 + 1 for k in range(24)])
BIDX = np.array([k % 8 for k in range(24)])
PREF_K = BIDX * 4 + CIDX
BASE_KX = BIDX * 4
T = {}
U = {}
F = {}
for fa, fb in CPAIRS:
    key = fa + fb
    T[key] = np.array([
        spearman(CS1H[fa][:, k], CS1H[fb][:, k])
        for k in range(24)])
    U[key] = np.array([
        spearman(CS[fa][:, k], CS[fb][:, k])
        for k in range(24)])
    F['f1_sTT' + key] = np.array([
        spearman(TT[fa][k], TT[fb][k])
        for k in range(24)])
    F['f2_cTT' + key] = np.array([
        cosv(TT[fa][k], TT[fb][k])
        for k in range(24)])
    F['f3_sLGp' + key] = np.array([
        spearman(LG[fa][PREF_K[k]],
                 LG[fb][PREF_K[k]])
        for k in range(24)])
    F['f4_sLGb' + key] = np.array([
        spearman(LG[fa][BASE_KX[k]],
                 LG[fb][BASE_KX[k]])
        for k in range(24)])
    na = np.linalg.norm(TT[fa], axis=1)
    nb = np.linalg.norm(TT[fb], axis=1)
    F['f5_amp' + key] = np.minimum(
        na, nb) / np.maximum(na, nb)
    bits('T_' + key, T[key], Z['T_' + key])
    bits('U_' + key, U[key], Z['U_' + key])
    bits('F1_' + key, F['f1_sTT' + key],
         Z['F1_STT_' + key])
    bits('F2_' + key, F['f2_cTT' + key],
         Z['F2_CTT_' + key])
    bits('F3_' + key, F['f3_sLGp' + key],
         Z['F3_SLGP_' + key])
    bits('F4_' + key, F['f4_sLGb' + key],
         Z['F4_SLGB_' + key])
    bits('F5_' + key, F['f5_amp' + key],
         Z['F5_AMP_' + key])
    mig = spearman(R1[fa], R1[fb])
    d = abs(mig - float(Z['MIG_' + key]))
    chk('mig_' + key, d < 1e-10,
        '%.6f' % mig)
    sut = spearman(U[key], T[key])
    d = abs(sut - float(Z['SP_UT_' + key]))
    chk('sp_ut_' + key, d < 1e-10,
        '%.6f' % sut)
    sf12 = spearman(F['f1_sTT' + key],
                    F['f2_cTT' + key])
    d = abs(sf12
            - float(Z['SP_F1F2_' + key]))
    chk('sp_f1f2_' + key, d < 1e-10)
    tm = float(np.median([
        spearman(TT[fa][k], TT[fb][k])
        for k in range(24)]))
    d = abs(tm - float(Z['TT_MED_' + key]))
    chk('tt_med_' + key, d < 1e-10)

# ---------- 4) E3 + gates replay ----------
RESP = {'T': T, 'U': U}
FKEYMAP = {'f1_sTT': 'F1_STT',
           'f2_cTT': 'F2_CTT',
           'f3_sLGp': 'F3_SLGP',
           'f4_sLGb': 'F4_SLGB',
           'f5_amp': 'F5_AMP'}
E3SP = {}
for rn in ('T', 'U'):
    for fid in FKEYMAP:
        for key in ('AB', 'AC', 'BC'):
            sk = ('%s~%s_%s'
                  % (fid, rn, key))
            s_ = spearman(F[fid + key],
                          RESP[rn][key])
            E3SP[sk] = s_
            zk = ('E3_%s_%s_%s'
                  % (FKEYMAP[fid], rn, key))
            d = abs(s_ - float(Z[zk]))
            chk('e3sp_' + sk, d < 1e-10,
                '%+.6f' % s_)

# perm replay (2 decisive tests)
p1 = perm_p(F['f2_cTTAB'], U['AB'])
z1 = float(Z['E3P_F2_CTT_U_AB'])
chk('perm_f2_cTT~U_AB',
    abs(p1 - z1) < 1e-12,
    '%.5f vs %.5f' % (p1, z1))
p2 = perm_p(F['f4_sLGbAB'], T['AB'])
z2 = float(Z['E3P_F4_SLGB_T_AB'])
chk('perm_f4_sLGb~T_AB',
    abs(p2 - z2) < 1e-12,
    '%.5f vs %.5f' % (p2, z2))

# G_DS replay
E2 = []
for rn in ('T', 'U'):
    for key in ('AB', 'AC', 'BC'):
        s_ = E3SP['f2_cTT~%s_%s'
                  % (rn, key)]
        pk = perm_p(F['f2_cTT' + key],
                    RESP[rn][key])
        E2.append((s_, pk))
cnt_pos = sum(1 for s_, p_ in E2
              if s_ > 0 and p_ < 0.05)
n_bonf = sum(1 for _, p_ in E2
             if p_ < 0.05 / 6)
min_sp2 = min(s_ for s_, _ in E2)
g_ds = bool(cnt_pos >= 4 and min_sp2 > 0)
zlist = [NormalDist().inv_cdf(
    min(1.0 - 1e-12, 1.0 - p_))
    for _, p_ in E2]
stouffer = float(np.sum(zlist)
                 / np.sqrt(len(zlist)))
chk('gds_count', cnt_pos
    == int(Z['GDS_COUNT']) == 1, str(cnt_pos))
chk('gds_min_sp',
    abs(min_sp2 - float(Z['GDS_MIN_SP']))
    < 1e-10, '%.4f' % min_sp2)
chk('gds_n_bonf', n_bonf
    == int(Z['GDS_N_BONF']), str(n_bonf))
chk('gds_gate', g_ds is False)
d = abs(stouffer - float(Z['STOUFFER_Z']))
chk('stouffer', d < 1e-9, '%.4f' % stouffer)
chk('gates_res_G_DS',
    res['gates']['G_DS'] is False)
chk('gates_res_count',
    res['gates']['count_sig_pos'] == 1)
chk('gates_res_min_sp',
    abs(res['gates']['min_sp'] - min_sp2)
    < 1e-10)

# G1 replay
E1 = {}
for key in ('AB', 'AC', 'BC'):
    for rn in ('T', 'U'):
        s_ = E3SP['f1_sTT~%s_%s'
                  % (rn, key)]
        pk = perm_p(F['f1_sTT' + key],
                    RESP[rn][key])
        E1[(rn, key)] = (s_, pk)
best = None
for (rn, key), (s_, pk) in E1.items():
    cand = (abs(s_), s_, pk, rn, key)
    if best is None or cand[0] > best[0]:
        best = cand
g1 = bool(best[0] >= 0.5 and best[2] < 0.05)
chk('g1_gate', g1 is False,
    'best %+.4f p=%.5f on %s_%s'
    % (best[1], best[2], best[3], best[4]))
chk('gates_res_G1',
    res['gates']['G1'] is False)
d = abs(best[1] - res['gates']['G1_sp'])
chk('g1_sp_match', d < 1e-10)
d = abs(best[2] - res['gates']['G1_p'])
chk('g1_p_match', d < 1e-12)

# verdict replay (gate tree, non-degenerate
# branch since setup/top8 ok)
verdict_replay = 'ds7b_cos_locked' \
    if g_ds else 'ds7b_cos_absent'
chk('verdict_replay',
    verdict_replay == 'ds7b_cos_absent',
    verdict_replay)

# partial replay (f2g1_U_AB)
psp = partial_sp(F['f2_cTTAB'], U['AB'],
                 F['f1_sTTAB'])
d = abs(psp - float(Z['PSP_F2G1_U_AB']))
chk('partial_f2g1_U_AB', d < 1e-10,
    '%.6f' % psp)
pp = partial_sp_perm_p(F['f2_cTTAB'], U['AB'],
                       F['f1_sTTAB'])
d = abs(pp - float(Z['PSP_F2G1P_U_AB']))
chk('partial_p_f2g1_U_AB', d < 1e-12,
    '%.5f vs %.5f'
    % (pp, float(Z['PSP_F2G1P_U_AB'])))

chk('f2uab_narrative',
    E3SP['f2_cTT~U_AB'] > 0.5 and z1 < 0.01,
    'sp=%.4f p=%.5f'
    % (E3SP['f2_cTT~U_AB'], z1))

# ---------- 5) Ledger ----------
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('ledger_n', len(led['measurements']) == 220,
    str(len(led['measurements'])))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
chk('ledger_l14', len(l14['connects']) == 188,
    str(len(l14['connects'])))
m81 = [m for m in led['measurements']
       if m.get('phase') == 3081]
chk('ledger_meas3081_present',
    len(m81) == 1, str(len(m81)))
if m81:
    m = m81[0]
    chk('meas3081_verdict',
        m['verdict'] == 'ds7b_cos_absent')
    chk('meas3081_hashes',
        m['hashes']['npz_sha256_8']
        == seal['npz_sha256_8']
        and m['hashes']['result_sha256_8']
        == seal['result_sha256_8'])
l14c = [c for c in l14['connects']
        if isinstance(c, dict)
        and c.get('phase') == 3081]
chk('l14_conn3081', len(l14c) == 1)
ref_sha = led.get('ledger_sha256_8')
led2 = json.loads(json.dumps(
    led, ensure_ascii=False))
led2.pop('ledger_sha256_8', None)
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
h = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
chk('ledger_sha', h == ref_sha == '76672367',
    '%s vs %s' % (h, ref_sha))

# ---------- 6) docs ----------
memo = io.open(MEMO, encoding='utf-8').read()
chk('memo_phase3081',
    '## Phase 3081:' in memo
    and 'ds7b_cos_absent' in memo
    and '[2026-09-22 00:08:55]' in memo)
chk('memo_menu3082', '3082' in memo)
aud = io.open(AUDIT, encoding='utf-8').read()
chk('audit_43',
    u'四十三、3081' in aud
    and 'ds7b_cos_absent' in aud)
wl = io.open(WLOG, encoding='utf-8').read()
chk('wlog_3081', 'Phase 3081' in wl)
wm = io.open(WMEM, encoding='utf-8').read()
chk('wmem_max3081', 'max=3081' in wm)
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
