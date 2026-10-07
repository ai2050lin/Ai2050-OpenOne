# -*- coding: utf-8 -*-
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3048_omega_p45_kvpos_full_replay_qwen.py')
s = io.open(P, encoding='utf-8').read()

# 1) forward_run: allow partial-position repl
#    (rows = number of masked positions)
old1 = """        rk = torch.tensor(np.ascontiguousarray(
            replK.reshape(NL, n, 8, HDIM)),
            dtype=torch.float32, device='cuda')"""
new1 = """        rk = torch.tensor(np.ascontiguousarray(
            replK.reshape(NL, -1, 8, HDIM)),
            dtype=torch.float32, device='cuda')"""
assert s.count(old1) == 1, ('rk', s.count(old1))
s = s.replace(old1, new1)

# 2) K integrity: scatter onto masked rows
old2 = """            if rk is not None:
                comb = torch.where(
                    mk[:, None, None], rk[li],
                    capKn[li]['orig'])
                e = max(e, float(
                    (capKn[li]['mod'] - comb)
                    .abs().max()))"""
new2 = """            if rk is not None:
                comb = capKn[li]['orig'].clone()
                comb[mk] = rk[li]
                e = max(e, float(
                    (capKn[li]['mod'] - comb)
                    .abs().max()))"""
assert s.count(old2) == 1, ('ik', s.count(old2))
s = s.replace(old2, new2)

# 3) V integrity: same scatter
old3 = """            if rv is not None:
                comb = torch.where(
                    mv[:, None], rv[li],
                    capV[li]['orig'])
                e = max(e, float(
                    (capV[li]['mod'] - comb)
                    .abs().max()))"""
new3 = """            if rv is not None:
                comb = capV[li]['orig'].clone()
                comb[mv] = rv[li]
                e = max(e, float(
                    (capV[li]['mod'] - comb)
                    .abs().max()))"""
assert s.count(old3) == 1, ('iv', s.count(old3))
s = s.replace(old3, new3)

# 4) collect per-pair REP fields for the null
old4 = """a130_fail = 0
rec_kind = []
rec_body = []
rec_dlg = []"""
new4 = """a130_fail = 0
rec_kind = []
rec_body = []
rec_dlg = []
repK_list = []
repV_list = []"""
assert s.count(old4) == 1, ('lists', s.count(old4))
s = s.replace(old4, new4)

old5 = """    repV = VP[pref_i, :, off:off + nb, :].copy()
    m_b = np.ones(nb, dtype=bool)
    res = forward_run(ids_b, replK=repK, maskK=m_b,
                      replV=repV, maskV=m_b,
                      integ=True)"""
new5 = """    repV = VP[pref_i, :, off:off + nb, :].copy()
    repK_list.append(repK)
    repV_list.append(repV)
    m_b = np.ones(nb, dtype=bool)
    res = forward_run(ids_b, replK=repK, maskK=m_b,
                      replV=repV, maskV=m_b,
                      integ=True)"""
assert s.count(old5) == 1, ('collect', s.count(old5))
s = s.replace(old5, new5)

# 5) null MC: preregistered per-mc 24-pair median
old6 = """# null MC for REP
log('=== null MC (R=%d random replacement) ==='
    % R_NULL)
NK = np.linalg.norm(
    repK.reshape(NL, nb, HDIM * 8), axis=2)
NV_ = np.linalg.norm(repV, axis=2)
null_cos = np.zeros(R_NULL)
null_frac = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL + mc)
    RK = rng.standard_normal((NL, nb, HDIM * 8)) \\
        .astype(np.float32)
    RK *= (NK / np.linalg.norm(RK, axis=2)
           )[:, :, None]
    RV = rng.standard_normal((NL, nb, HDIM * 8)) \\
        .astype(np.float32)
    RV *= (NV_ / np.linalg.norm(RV, axis=2)
           )[:, :, None]
    res = forward_run(ids_b, replK=RK, maskK=m_b,
                      replV=RV, maskV=m_b)
    r = res['lg'] - LG[base_i]
    nr = float(np.linalg.norm(r))
    csm = float(r @ t) / (nr * nt) \\
        if nr > 1e-12 and nt > 1e-12 else 0.0
    null_cos[mc] = csm
    null_frac[mc] = nr / nt if nt > 1e-12 \\
        else float('nan')
    rec_kind.append(3)
    rec_body.append(b)
    rec_dlg.append(nr)
    if (mc + 1) % 50 == 0:
        log('  mc %d/%d' % (mc + 1, R_NULL))"""
new6 = """# null MC for REP (per-mc 24-pair median,
# preregistered statistic)
log('=== null MC (R=%d random replacement, '
    '24-pair median) ===' % R_NULL)
null_cos = np.zeros(R_NULL)
null_frac = np.zeros(R_NULL)
for mc in range(R_NULL):
    rng = np.random.default_rng(SEED_NULL + mc)
    csm = np.zeros(NP_)
    frm = np.zeros(NP_)
    for k in range(NP_):
        b = int(bidx[k])
        c = int(cidx[k])
        base_i = idx_of[(0, b, False)]
        ids_b = assembled[base_i]['ids']
        nb = len(ids_b)
        repK = repK_list[k]
        repV = repV_list[k]
        m_b = np.ones(nb, dtype=bool)
        NK = np.linalg.norm(repK, axis=2)
        NV_ = np.linalg.norm(repV, axis=2)
        RK = rng.standard_normal(
            (NL, nb, HDIM * 8)).astype(np.float32)
        RK *= (NK / np.linalg.norm(RK, axis=2)
               )[:, :, None]
        RV = rng.standard_normal(
            (NL, nb, HDIM * 8)).astype(np.float32)
        RV *= (NV_ / np.linalg.norm(RV, axis=2)
               )[:, :, None]
        res = forward_run(ids_b, replK=RK,
                          maskK=m_b, replV=RV,
                          maskV=m_b)
        r = res['lg'] - LG[base_i]
        nr = float(np.linalg.norm(r))
        t = t_targets[(b, c)]
        nt = float(np.linalg.norm(t))
        csm[k] = float(r @ t) / (nr * nt) \\
            if nr > 1e-12 and nt > 1e-12 else 0.0
        frm[k] = nr / nt if nt > 1e-12 \\
            else float('nan')
        rec_kind.append(3)
        rec_body.append(b)
        rec_dlg.append(nr)
    null_cos[mc] = float(np.median(csm))
    null_frac[mc] = float(np.nanmedian(frm))
    if (mc + 1) % 25 == 0:
        log('  mc %d/%d (null med cos=%.4f)'
            % (mc + 1, R_NULL, null_cos[mc]))"""
assert s.count(old6) == 1, ('null', s.count(old6))
s = s.replace(old6, new6)

# 6) corrections: cover run1 + run2
old7 = """    'corrections': 'run1 crashed pre-anchor on '
                   'the prompt-assembly alignment '
                   'assertion: BPE leading-space '
                   'boundary effect (first body '
                   'token is the no-space variant in '
                   'the base prompt but the '
                   'leading-space variant inside the '
                   'prefix prompt, so naive '
                   'subsequence matching fails); '
                   'alignment redefined as tail '
                   'matching (pref ids end with base '
                   'ids[1:]) plus a first-token '
                   'semantic strip-equality check; '
                   'offset = length difference; '
                   'statistics unchanged; run2 '
                   'authoritative',"""
new7 = """    'corrections': 'run1 crashed pre-anchor on '
                   'the prompt-assembly alignment '
                   'assertion: BPE leading-space '
                   'boundary effect (first body '
                   'token is the no-space variant in '
                   'the base prompt but the '
                   'leading-space variant inside the '
                   'prefix prompt, so naive '
                   'subsequence matching fails); '
                   'alignment redefined as tail '
                   'matching (pref ids end with base '
                   'ids[1:]) plus a first-token '
                   'semantic strip-equality check; '
                   'offset = length difference. '
                   'run2 crashed mid-T4 on a '
                   'replacement-array length '
                   'mismatch (partial-position repl '
                   'passed where a full-length '
                   'reshape was assumed; '
                   'forward_run now reshapes to '
                   '(-1,8,128) and integrity scatters '
                   'onto masked rows); the run2 null '
                   'printout (p=0.00498) was found '
                   'to be single-pair (accidental '
                   'reuse of the last T2 iteration) '
                   'instead of the preregistered '
                   'per-mc 24-pair median - null '
                   'implementation corrected to the '
                   'preregistered statistic; ADD/REP '
                   'observed statistics are '
                   'frozen-seed deterministic and '
                   're-derived unchanged; run3 '
                   'authoritative',"""
assert s.count(old7) == 1, ('corr', s.count(old7))
s = s.replace(old7, new7)

# 7) run label -> run3
old8 = ("'run': 'run2 authoritative (fp32; run1 "
        "crashed pre-anchor on a tokenization "
        "alignment assertion, see corrections)',")
new8 = ("'run': 'run3 authoritative (fp32; run1 "
        "crashed pre-anchor on a tokenization "
        "alignment assertion; run2 crashed mid-T4 "
        "on a repl length mismatch and its null "
        "was single-pair instead of the "
        "preregistered 24-pair median; see "
        "corrections)',")
assert s.count(old8) == 1, ('run', s.count(old8))
s = s.replace(old8, new8)

io.open(P, 'w', encoding='utf-8').write(s)
py_compile.compile(P, doraise=True)
io.open(r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\patch3048c_result.txt', 'w',
        encoding='utf-8').write(
    'patched ok; compile ok\n')
print('ok')
