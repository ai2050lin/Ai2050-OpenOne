# -*- coding: utf-8 -*-
"""Phase 2876: growth-curve point 2 re-test at 42 words (2873 protocol
on the phase2875 expanded attr census).

Prereg (frozen before any readout; execution.json written first):
  Same protocol as 2873: column z-score + unit rows, LOO-NN cosine
  retrieval of axis label; selection = per-head one-way eta2 F-test
  p<0.01 (scipy.stats.f.sf, correct tail); null = full-pipeline
  mirrored label permutation (N=200, SEED=2876).
  D1  sparse sufficiency at 42 words: acc(sig2)>acc(full) AND
      acc(sig2)>null p95 => sparse_sufficiency_replicates.
  D2  cross-axis overlap: |sig2 AND class43| hypergeometric p<0.05
      => shared_core, else axis_specific.  Growth datum: n_new.
  D3  transfer: acc(class43 on attr42) vs acc(full on attr42);
      reuse_signal iff greater.
  D4  power-resolution curve point: X5 re-check (acc_full vs null)
      recorded with word count 42 (window 28-80 from 2863/2873).
Growth ledger point: components_axis2 = |sig2|, shared, new.
"""
import hashlib
import json
import os
import time

import numpy as np
from scipy import stats

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2875 = os.path.join(BASE, 'phase2875', 'attr_census_v2',
                        'attr_census_v2.npz')
SRC_2868 = os.path.join(BASE, 'phase2868', 'growth_v2', 'growth_v2.npz')
OUT = os.path.join(BASE, 'phase2876', 'growth_axis2_v2')

SEED = 2876
N_NULL = 200
P_SIG = 0.01

PREREG = {
    'D1': 'sparse_sufficiency_replicates iff acc(sig2)>acc(full) and '
          'acc(sig2)>null_p95 (full-pipeline-mirrored, N=200)',
    'D2': 'shared_core iff hypergeom p < 0.05, else axis_specific; '
          'n_new = |sig2 \\ class43|',
    'D3': 'reuse_signal iff acc(class43 on attr42) > acc(full on attr42)',
    'D4': 'power curve point at 42 words (2863:8 fail, 2873:28 fail, '
          'here: ?)',
    'null_seed': SEED, 'n_null': N_NULL, 'p_sig': P_SIG,
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def zmat(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.maximum(sd, 1e-30)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-30)


def loo_acc(C, lab):
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def eta2_p(x, lab):
    labs = np.unique(lab)
    grand = x.mean()
    ss_b = 0.0
    ss_w = 0.0
    for g in labs:
        m = lab == g
        d = x[m]
        ss_b += m.sum() * (d.mean() - grand) ** 2
        ss_w += ((d - d.mean()) ** 2).sum()
    ss_t = ((x - grand) ** 2).sum()
    et = ss_b / max(ss_t, 1e-30)
    k, n = len(labs), len(x)
    df_b, df_w = k - 1, n - k
    if df_w <= 0 or ss_w <= 1e-30:
        p = 0.0 if ss_b > 1e-30 else 1.0
    else:
        p = float(stats.f.sf((ss_b / df_b) / (ss_w / df_w), df_b, df_w))
    return float(et), p


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)

    script = os.path.abspath(__file__)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2876, 'name': 'growth_axis2_v2',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'sources': {'attr_census_v2': sha8(SRC_2875),
                               'growth_v2': sha8(SRC_2868)},
                   'prereg': PREREG}, f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    z75 = np.load(SRC_2875, allow_pickle=True)
    zg = np.load(SRC_2868, allow_pickle=True)

    words = list(z75['target_list'])
    X = z75['drops'].astype(np.float64).reshape(len(words), -1)
    axes = list(z75['axis_names'])
    lab = np.array([axes.index(w.split(':')[0]) for w in words])
    n_words = len(words)
    class_sig = np.where(zg['sig_mask'].astype(bool))[0]
    log('words=%d axes=%d class43=%d' % (n_words, len(axes),
                                         len(class_sig)))

    rng = np.random.default_rng(SEED)

    # ---------- observed ----------
    acc_full = loo_acc(zmat(X), lab)
    ps = np.full(X.shape[1], 1.0)
    keep = []
    for j in range(X.shape[1]):
        _, p = eta2_p(X[:, j], lab)
        ps[j] = p
        if p < P_SIG:
            keep.append(j)
    sig2 = sorted(keep)
    log('sig2=%d  acc_full=%.4f' % (len(sig2), acc_full))

    acc_sig2 = loo_acc(zmat(X[:, sig2]), lab) if sig2 else float('nan')
    acc_43 = loo_acc(zmat(X[:, class_sig]), lab)

    # ---------- null (full-pipeline mirrored) ----------
    null_full, null_sig = [], []
    for it in range(N_NULL):
        pl = rng.permutation(lab)
        null_full.append(loo_acc(zmat(X), pl))
        ks, _ = [], None
        ks = [j for j in range(X.shape[1])
              if eta2_p(X[:, j], pl)[1] < P_SIG]
        null_sig.append(loo_acc(zmat(X[:, ks]), pl)
                        if ks else 0.0)
        if (it + 1) % 50 == 0:
            log('  null %d/%d' % (it + 1, N_NULL))
    nf, ns = np.array(null_full), np.array(null_sig)
    p95_sig = float(np.percentile(ns, 95))

    # ---------- D2 overlap ----------
    s2, c43 = set(sig2), set(class_sig.tolist())
    inter = s2 & c43
    p_over = (float(stats.hypergeom.sf(len(inter) - 1, 1152, 43, len(s2)))
              if s2 else float('nan'))

    # ---------- verdicts ----------
    d1 = bool(sig2 and acc_sig2 > acc_full and acc_sig2 > p95_sig)
    d2 = bool(s2 and p_over < 0.05)
    d3 = bool(acc_43 > acc_full)
    d4 = bool(acc_full > float(np.percentile(nf, 95)))

    res = {
        'phase': 2876, 'prereg': PREREG,
        'n_words': n_words, 'n_axes': len(axes),
        'acc_ladder': {'full_1152': round(acc_full, 4),
                       'sig2_p001': round(acc_sig2, 4) if sig2 else None,
                       'class43_on_attr': round(acc_43, 4)},
        'D1': {'verdict': 'sparse_sufficiency_replicates' if d1 else
                          ('no_signal' if not sig2 else 'not_replicated'),
               'acc_sig2': round(acc_sig2, 4) if sig2 else None,
               'acc_full': round(acc_full, 4),
               'null_sig_p95': round(p95_sig, 4),
               'null_sig_mean': round(float(ns.mean()), 4)},
        'D2': {'verdict': 'shared_core' if d2 else 'axis_specific',
               'intersect': len(inter), 'hypergeom_p': round(p_over, 5)
               if s2 else None,
               'n_new': len(s2) - len(inter)},
        'D3': {'verdict': 'reuse_signal' if d3 else 'novel_dominant',
               'acc_class43': round(acc_43, 4),
               'acc_full': round(acc_full, 4)},
        'D4': {'verdict': 'axis_signal_detected' if d4
                          else 'axis_signal_absent',
               'acc_full': round(acc_full, 4),
               'null_full_p95': round(float(np.percentile(nf, 95)), 4),
               'null_full_mean': round(float(nf.mean()), 4)},
        'growth_point_axis2_v2': {
            'components_axis2': len(s2),
            'shared_with_class43': len(inter),
            'new': len(s2) - len(inter),
            'acc_best': round(max([a for a in [acc_sig2, acc_43, acc_full]
                                   if a == a]), 4)},
        'nulls': {'N': N_NULL, 'seed': SEED},
        'sig2_headlist': sig2,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT, 'growth_axis2_v2.npz'),
                        X_attr=X.astype(np.float32), labels=lab,
                        sig2=np.array(sig2, dtype=np.int64),
                        class43=class_sig.astype(np.int64),
                        null_sig=ns.astype(np.float32),
                        null_full=nf.astype(np.float32))

    log('==== VERDICTS ====')
    log('acc ladder: full=%.4f sig2=%s class43=%.4f'
        % (acc_full, ('%.4f' % acc_sig2) if sig2 else 'NA', acc_43))
    log('D1=%s D2=%s(inter=%d p=%.5f new=%d) D3=%s D4=%s'
        % (res['D1']['verdict'], res['D2']['verdict'], len(inter),
           p_over if s2 else float('nan'), len(s2) - len(inter),
           res['D3']['verdict'], res['D4']['verdict']))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
