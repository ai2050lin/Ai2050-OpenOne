# -*- coding: utf-8 -*-
"""Patch phase3122 main script: fix 3 known bugs.
R1: forward_wrec stacked 4D -> squeeze(1) for einsum 3D.
R2: remove dead `sim` init block.
R3: remove first (pseudo-PIT) simulation loop + dead auc loop.
R4: insert rank-based PIT computed on the SAME snap simulation.
R5: remove old KS block (now computed in R4).
All strings ASCII; raw triple-quotes keep literal backslash-newline.
"""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3122_omega_p120_write_content_readout_'
     r'sentence_causal_dist_recon.py')
src = io.open(P, encoding='utf-8').read()
rep_log = []


def rep(tag, old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, ('COUNT %d != %d for [%s] head=%r'
                    % (c, n, tag, old[:70]))
    src = src.replace(old, new)
    rep_log.append('%s: replaced %d' % (tag, c))


# ---- R1: squeeze(1) after torch.stack ----
rep('R1',
    r"""        stacked = torch.stack(
            [WREC['store'][L] for L in range(NL)])
""",
    r"""        stacked = torch.stack(
            [WREC['store'][L]
             for L in range(NL)]).squeeze(1)
""")

# ---- R2: remove dead sim init block ----
rep('R2',
    r"""rng_mc = np.random.default_rng(3122)
sim = {}
for dcode in ('P', 'A1'):
    seq = MSEQ[dcode]
    mh = np.zeros((NP_A, N_REPS),
                  dtype=np.float64)
    mh[:] = seq[:, 0][:, None]
    sim[dcode] = mh
auc_sim = np.zeros(N_NEW + 1,
""",
    r"""rng_mc = np.random.default_rng(3122)
auc_sim = np.zeros(N_NEW + 1,
""")

# ---- R3: remove first sim loop + dead auc loop +
#          NOTE comment (keep snap block) ----
rep('R3',
    r"""for dcode_i, dcode in enumerate(('P', 'A1')):
    seq = MSEQ[dcode]
    mh = sim[dcode]
    col_all = ANN[dcode].astype(np.int64)
    for k in range(N_NEW):
        noise = res_pool[rng_mc.integers(
            0, len(res_pool),
            size=(NP_A, N_REPS))]
        step = S * (mh - MS) \
            + content[dcode][col_all[:, k], k][:, None] \
            + noise
        mh = mh + step
        sim[dcode] = mh
        lo = np.quantile(mh, 0.025, axis=1)
        hi = np.quantile(mh, 0.975, axis=1)
        pit[dcode_i, :, k + 1] = np.clip(
            (seq[:, k + 1] - lo)
            / np.maximum(hi - lo, 1e-9), 0, 1)

for t in range(N_NEW + 1):
    auc_sim[t] = auc_mw(
        sim['P'][:, :].ravel()
        if t == 0 else sim['P'].ravel() * 0
        + sim['P'][:, :] * 0 + sim['P'].ravel(),
        sim['A1'].ravel()) if False else \
        auc_mw(np.concatenate(
            [sim['P'][:, :1].ravel()]
            if t == 0 else
            [np.zeros(0)]),
            np.zeros(1)) if False else 0.0

# NOTE: the loop above is replaced by a clean
# per-step AUC on state snapshots; we re-simulate
# snapshots properly here.
snap = {d: [np.zeros((NP_A, N_REPS))""",
    r"""# single simulation pass below: per-step
# snapshots; auc_sim and rank-based PIT are
# both computed on THIS simulation (seed 3122,
# one continuous rng stream, no re-simulation).
snap = {d: [np.zeros((NP_A, N_REPS))""")

# ---- R4: rank-based PIT on same snap simulation ----
rep('R4',
    r"""for t in range(N_NEW + 1):
    auc_sim[t] = auc_mw(
        snap['P'][t].ravel(),
        snap['A1'][t].ravel())
r_dist = float(np.corrcoef(auc_sim, auc18)[0, 1])
""",
    r"""for t in range(N_NEW + 1):
    auc_sim[t] = auc_mw(
        snap['P'][t].ravel(),
        snap['A1'][t].ravel())
# rank-based PIT on the SAME simulation:
# u = (#{sim<true} + 0.5*#{sim==true}) / N_REPS;
# t=0 is the deterministic anchor (u=0.5),
# KS is computed over t>=1 only.
for dcode_i, dcode in enumerate(('P', 'A1')):
    seq = MSEQ[dcode]
    pit[dcode_i, :, 0] = 0.5
    for t in range(1, N_NEW + 1):
        s_i = snap[dcode][t]
        tr = seq[:, t][:, None]
        pit[dcode_i, :, t] = (
            (s_i < tr).sum(axis=1)
            + 0.5 * (s_i == tr).sum(axis=1)) \
            / float(N_REPS)
u = pit[:, :, 1:].ravel()
u_sorted = np.sort(u)
n_u = len(u_sorted)
ecdf = np.arange(1, n_u + 1) / float(n_u)
ks = float(np.max(np.abs(ecdf - u_sorted)))
r_dist = float(np.corrcoef(auc_sim, auc18)[0, 1])
""")

# ---- R5: remove old KS block ----
rep('R5',
    r"""u = np.concatenate([pit[0].ravel(),
                    pit[1].ravel()])
u_sorted = np.sort(u)
n_u = len(u_sorted)
ecdf = np.arange(1, n_u + 1) / float(n_u)
ks = float(np.max(np.abs(ecdf - u_sorted)))
c_pit_v = ('pit_calibrated' if ks < 0.05 else
""",
    r"""c_pit_v = ('pit_calibrated' if ks < 0.05 else
""")

# ---- post assertions ----
assert "sim['P']" not in src, 'leftover sim[P]'
assert 'sim[dcode]' not in src, 'leftover sim[dcode]'
assert 'sim = {}' not in src, 'leftover sim init'
assert 'if False else' not in src, 'leftover dead code'
assert src.count('.squeeze(1)') == 1, 'squeeze count'
assert 'u = pit[:, :, 1:].ravel()' in src, 'rank PIT'
assert 'npz_out[' + chr(39) + 'pit' + chr(39) + ']' \
    in src, 'npz pit missing'
assert src.count('rng_mc.integers') == 1, \
    'rng draw count (snap loop only; R3 removed first)'

py_compile.compile(P, doraise=True)
io.open(P, 'w', encoding='utf-8').write(src)
rep_log.append('COMPILE_OK; lines=%d'
               % len(src.splitlines()))
rep = chr(10).join(rep_log)
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3122_patch_out.txt', 'w',
        encoding='utf-8').write(rep + chr(10))
print(rep)
