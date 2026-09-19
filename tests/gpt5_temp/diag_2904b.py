# -*- coding: utf-8 -*-
"""Phase 2904b diagnostic: analytic behavior of margin under
(a) class-covariance difference (audit-2nd construction) and
(b) class-skewness / asymmetric-subcluster construction.

Expectation: (a) gives E[cos same] - E[cos diff] = 0 (first
moment of a covariance anisotropy vanishes in the cosine), so
margin_within cannot detect it BY CONSTRUCTION - the audit-2nd
failure in 2904 is a construction error, not detector failure.
(b) keeps class means at zero but puts most mass on one side,
giving a positive first-moment contribution to same-label
cosines; margin_within should detect it.

Part A: pairwise-sampling estimate of E[cos] (independent pairs).
Part B: full n=57 pipeline (within-center, margin, 1000-perm
p95) repeated; detection rate = P(margin_within > p95_within).
"""
import numpy as np

OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\diag_2904b.txt')
N_PAIRS = 400000
N_REP = 100
N_PERM = 1000
SEED = 29040


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def skew_rows(rng, n, w, big=2.0, tail=18.0, pf=0.9):
    Z = rng.normal(size=(n, len(w)))
    side = rng.random(n) < pf
    mu = np.where(side, big, -tail)[:, None] * w[None, :]
    return Z + mu          # E[row] = 0 by construction


def cov_rows(rng, n, w, k=3.0):
    Z = rng.normal(size=(n, len(w)))
    return Z + k * np.outer(Z @ w, w)   # 4x std along w


def pair_cos_expect(gen, lab, rng, n_pairs):
    cs = []
    labs = []
    for _ in range(n_pairs):
        i = int(rng.integers(0, len(lab)))
        j = int(rng.integers(0, len(lab)))
        if i == j:
            continue
        xi = gen(lab[i])
        xj = gen(lab[j])
        cs.append(float(xi @ xj)
                  / max(float(np.linalg.norm(xi)
                              * np.linalg.norm(xj)), 1e-30))
        labs.append(int(lab[i] == lab[j]))
    cs = np.array(cs)
    labs = np.array(labs, dtype=bool)
    return float(cs[labs].mean()), float(cs[~labs].mean())


def main():
    lines = []
    rng = np.random.default_rng(SEED)

    # ---------- Part A: E[cos same] - E[cos diff] ----------
    for name, gen0 in (('cov4x', lambda c: (
            (lambda w: (lambda x: x)(w))(unit(rng.normal(size=10)))
    )),):
        pass
    for name in ('cov4x', 'skew'):
        e_s0, e_d0, e_s1, e_d1 = [], [], [], []
        for rep in range(4):
            w0 = unit(rng.normal(size=10))
            w1 = unit(rng.normal(size=10)
                      - (rng.normal(size=10) @ w0) * w0)
            if name == 'cov4x':
                def gen(c, w0=w0, w1=w1):
                    return cov_rows(rng, 1, w0 if c == 0 else w1)[0]
            else:
                def gen(c, w0=w0, w1=w1):
                    return skew_rows(rng, 1,
                                     w0 if c == 0 else w1)[0]
            prng = np.random.default_rng(SEED + 1 + rep)
            lab = np.zeros(400000, dtype=int)
            lab[200000:] = 1
            es, ed = pair_cos_expect(gen, lab, prng, N_PAIRS)
            e_s0.append(es)
            e_d0.append(ed)
        diff = float(np.mean(e_s0) - np.mean(e_d0))
        lines.append('A %s: E[cos|same]=%.5f E[cos|diff]=%.5f '
                     'gap=%+.5f' % (name, float(np.mean(e_s0)),
                                    float(np.mean(e_d0)), diff))

    # ---------- Part B: full n=57 pipeline ----------
    n, d, n0 = 57, 10, 22
    prng_master = np.random.default_rng(SEED + 100)
    for name in ('cov4x', 'skew', 'null_iid'):
        det = 0
        det_full = 0
        mw_list = []
        for rep in range(N_REP):
            rg = np.random.default_rng([SEED, rep, hash(name) % 997])
            lab = np.array([0] * n0 + [1] * (n - n0))
            if name == 'null_iid':
                B = rg.normal(size=(n, d))
            else:
                w0 = unit(rg.normal(size=d))
                w1 = unit(rg.normal(size=d)
                          - (rg.normal(size=d) @ w0) * w0)
                B = np.empty((n, d))
                m0 = lab == 0
                m1 = lab == 1
                if name == 'cov4x':
                    B[m0] = cov_rows(rg, int(m0.sum()), w0)
                    B[m1] = cov_rows(rg, int(m1.sum()), w1)
                else:
                    B[m0] = skew_rows(rg, int(m0.sum()), w0)
                    B[m1] = skew_rows(rg, int(m1.sum()), w1)

            def mw_of(Bm, lb):
                Bw = Bm.copy()
                for c in (0, 1):
                    mm = lb == c
                    if mm.any():
                        Bw[mm] -= Bw[mm].mean(0, keepdims=True)
                U = Bw / np.maximum(np.linalg.norm(
                    Bw, axis=1, keepdims=True), 1e-30)
                Sm = U @ U.T
                eye = np.eye(len(lb), dtype=bool)
                same = (lb[:, None] == lb[None, :]) & (~eye)
                diff = (~eye) & (~same)
                return float(Sm[same].mean() - Sm[diff].mean())

            mw_obs = mw_of(B, lab)
            pl = np.random.default_rng([SEED, rep, 77])
            nw = [mw_of(B, pl.permutation(lab))
                  for _ in range(N_PERM)]
            p95 = float(np.percentile(nw, 95))
            mw_list.append(mw_obs)
            if mw_obs > p95:
                det += 1
            if name == 'skew':
                U = B / np.maximum(np.linalg.norm(
                    B, axis=1, keepdims=True), 1e-30)
                Sm = U @ U.T
                eye = np.eye(n, dtype=bool)
                same = (lab[:, None] == lab[None, :]) & (~eye)
                diff = (~eye) & (~same)
                if float(Sm[same].mean()
                         - Sm[diff].mean()) > 0:
                    det_full += 1
        lines.append('B %s: detection rate within=%.2f (%d/%d) '
                     'median margin_within=%+.4f'
                     % (name, det / N_REP, det, N_REP,
                        float(np.median(mw_list))))

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('diag ok')


if __name__ == '__main__':
    main()
