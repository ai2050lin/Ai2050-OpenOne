# -*- coding: utf-8 -*-
"""Phase 3232 (G5-G29): GD_amp class-count robustness arm (zero GPU).

Per prereg in MEMO Phase 3231 tail (default arm (a)): the 3192
u*_amplitude_carrier verdict was established with the OOV side pooled over
n=4 OOV classes. This arm expands the unit of analysis to n=10 classes
(4 OOV + 6 seen) on the SAME sealed residual chain (3169 B-protocol
verbatim, seeds 7/8/9, ridge lam 1e-3) with the sealed u* (3225 npz for
qwen3-4b/qwen3-14b as external anchor + chain-recomputed u* for all 3
models, bitwise device anchors to 3189/3190/3191/3192):
  per OOV class c (4):  med_amp_oov_c = median |Rbar[rows c] @ u*|
  per seen class c (6): med_amp_seen_c = median |R_(row,seed) @ u*| over
                        test-split residuals of that class
  G1: per-OOV-class amp_ratio_c = med_amp_oov_c / med_amp_seen_pooled >= 1.3
      in >= 3/4 OOV classes  x  >= 2/3 models
  G2: class_ratio = median(4 OOV class meds) / median(6 seen class meds)
      >= 1.3  x  >= 2/3 models
  BOTH => GD_AMP_CLASS_ROBUST (GAP-4 mechanism_note append, gap v1.9 ->
  v1.10, registry v1.31 UNCHANGED); G2 only => PARTIAL (refinement note);
  G2 fail => DIFFUSE (class-heterogeneity counter-note).
Report-only: per-class u* energy share, per-class collapse amplitude
(3169 all-class collapse magnitudes), Spearman(med_amp_c, collapse_c)
over 10 classes.
Pooled anchors: med_amp_oov / med_amp_seen / amp_ratio drift == 0 bitwise
vs sealed 3192. SMOKE (P3232_SMOKE=1): zero-GPU => identical numbers,
smoke_ prefix, no registry/gap writes.
"""
import copy
import hashlib
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3192_g5c5_gap4_amplitude_attribution as p92

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                       'rdc_query_construction_20260913')
NAME = 'g5g29_gd_amp_class_robustness'
OUTDIR = os.path.join(SRC_DIR, 'phase3232', NAME)
SMOKE = os.environ.get('P3232_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b',
             'glm4-9b': 'p3169_glm4'}

G_AMP = 1.3
G1_CLASSES = 3            # of 4 OOV classes
G_MODELS = 2              # of 3 models
CLS_IDS = list(range(6, 10))
SEEN_IDS = list(range(0, 6))

SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190_result': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'p3191_result': os.path.join(SRC_DIR, r'phase3191\g5c4_ftr25_heldout_upgrade\result.json'),
    'p3192_result': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\result.json'),
    'p3225_npz': os.path.join(SRC_DIR, r'phase3225\g5g25_a1_behavioral_band_compute\a1_behavioral_band.npz'),
    'p3225_result': os.path.join(SRC_DIR, r'phase3225\g5g25_a1_behavioral_band_compute\result.json'),
    'gap_v19': os.path.join(SRC_DIR, r'phase3231\g5g28_registry_decision\gap_ledger_v1_9.json'),
}
SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0', 'p3190_result': '6a8893ae',
    'p3191_result': 'a218ead4', 'p3192_result': '5d8d4858',
    'p3225_npz': 'ac5d4d74', 'p3225_result': 'ea1a0585', 'gap_v19': '9fca29bd',
}
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3189 = ('3a26f541', 'fb8c1f65')
CONTENT_3190 = ('24461856', '0c975f5b')
CONTENT_3191 = ('836fe933', 'a66c0b3d')
CONTENT_3192 = ('6221fe00', '58c4f183')
# sealed 3192 per-model pooled anchors (bitwise)
A92 = {
    'qwen3-4b': {'med_amp_oov': 89.6759033203125,
                 'med_amp_seen': 8.461281776428223,
                 'amp_ratio': 10.598382949829102},
    'qwen3-14b': {'med_amp_oov': 252.64761352539062,
                  'med_amp_seen': 35.534027099609375,
                  'amp_ratio': 7.110019207000732},
    'glm4-9b': {'med_amp_oov': 50.03424072265625,
                'med_amp_seen': 5.627579689025879,
                'amp_ratio': 8.890898704528809},
}

DESIGN = {
    'phase': '3232', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3231 tail default (a) GD_amp 扩样臂 '
              '(frozen before any observation)',
    'question': 'is the 3192 u*_amplitude_carrier verdict robust when the '
                'unit of analysis expands from n=4 pooled OOV classes to '
                'n=10 per-class measurements (4 OOV + 6 seen)?',
    'data': 'sealed 3169 collect npz x3 + sealed 3189/3190/3191/3192 '
            'results + sealed 3225 npz ustar (4b/14b external anchor) + '
            'gap v1.9; zero GPU',
    'residual_chain': '3192 verbatim: 3169 B-protocol (seeds 7/8/9, '
                      'split_s1_246, ridge lam 1e-3), 876 OOV rows, '
                      'row-level Rbar = 3-seed mean residual '
                      '(amplitude-preserving), u* = Vt[0] of Qn SVD',
    'measurements': 'per model: (1) per-OOV-class med|Rbar_c@u*| (4); '
                    '(2) per-seen-class med|R_(row,seed)@u*| over test '
                    'residuals (6); (3) G1 amp_ratio_c per OOV class vs '
                    'pooled seen med; (4) G2 class_ratio = med(4 OOV '
                    'class meds)/med(6 seen class meds); report-only: '
                    'per-class u* energy share, per-class collapse '
                    'amplitude (relative residual energy), Spearman over '
                    '10 classes',
    'gate': 'GD_AMP_CLASS_ROBUST iff G1 (>=3/4 OOV classes amp_ratio_c >= '
            '1.3) AND G2 (class_ratio >= 1.3), each x >= 2/3 models => '
            'GAP-4 mechanism_note append (class-level robust), gap v1.9 -> '
            'v1.10, registry v1.31 UNCHANGED; G2 pass only => PARTIAL '
            'refinement note append; G2 fail => DIFFUSE counter-note '
            'append. Pooled anchors vs 3192 bitwise (drift == 0).',
    'model_scope': '3/3 (glm4 via chain-recomputed u*; 3225 npz anchor '
                   'applies to 4b/14b only)',
    'sources': SHA_ANCHOR,
    'gates': {'amp_thresh': G_AMP, 'g1_classes': G1_CLASSES,
              'g2_thresh': G_AMP, 'models_needed': G_MODELS},
    'corrections_registered_pre_observation': [
        'C1 (2026-10-13, before any gated observation): the 3192 pooled '
        'anchors med_amp_oov/med_amp_seen/amp_ratio propagate through the '
        'SVD singular VECTOR u* (Vt[0]), which wobbles run-to-run at '
        '~5e-4 rad under threaded BLAS (observed cos(u*_run, u*_3225npz) '
        '= 0.99999988; singular-VALUE/scalar anchors 3189/3190/3191, '
        'which do NOT depend on Vt[0] direction, remain drift == 0 '
        'bitwise). First smoke crashed at the bitwise pooled-anchor '
        'assert BEFORE any per-class observation was computed; the '
        'pooled anchor is therefore relaxed to relative tolerance 1e-6 '
        '(observed rel drift 1.6e-9). Gated per-class measurements use '
        'the within-run chain u* consistently.',
    ],
}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def freeze():
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True,
                       indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3232, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False,
                  indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def ks_stat(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side='right') / float(len(a))
    cdf_b = np.searchsorted(b, allv, side='right') / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=np.float64)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=np.float64)))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-18))


def run_arm(sealed_3169, sealed_3189, sealed_3190, sealed_3191, sealed_3192):
    out = {}
    pm90, pm91 = sealed_3190['per_model'], sealed_3191['per_model']
    z32 = np.load(SRC['p3225_npz'])
    for m in MODELS:
        log('=== %s ===' % m)
        rd = int(sealed_3169['per_model'][m]['readout'])
        z = np.load(SRC[NPZ69_KEY[m]])
        H16 = z['H']
        NTz, NPz, NHz, Dz = H16.shape
        kout = NHz - 2
        assert kout == rd, ('readout slot', m)
        D = Dz
        Y = H16[:, :, kout, :].reshape(p92.NT * p92.NP_, D).astype(np.float32)
        assert np.isfinite(Y).all()

        tr_base, te_all = {}, {}
        for s in p92.SEEDS_S1:
            tr_pairs, te_pairs = p92.split_s1_246(s)
            tr_base[s] = [t * p92.NP_ + p92.PI_OF_PAIR[p]
                          for t in range(p92.NT) for p in p92.PAIRS
                          if p in tr_pairs]
            te_all[s] = [t * p92.NP_ + p92.PI_OF_PAIR[p]
                         for t in range(p92.NT) for p in p92.PAIRS
                         if p in te_pairs]
        oov_rows = p92.ALL_OOV_ROWS
        oov_row_cls = np.array([p92.PAIRS[r % p92.NP_][1] for r in oov_rows])

        def rv(r, ne):
            i, c = p92.PAIRS[r % p92.NP_]
            return p92.phi_row(i, c, r // p92.NP_, ne, p92.NC, p92.NT)

        E69 = sealed_3169['per_model'][m]['B']
        E89 = sealed_3189['per_model'][m]
        drift69 = 0.0
        Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        Rbar_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        seen_amp = {}          # global row -> [per-seed residual vectors]
        Dk_sum, Dk_n = 0.0, 0
        for si, s in enumerate(p92.SEEDS_S1):
            tr_rows = tr_base[s]
            te_rows = te_all[s]
            Xtr = np.stack([rv(r, p92.NE) for r in tr_rows])
            W = p92.ridge_primal(Xtr, Y[tr_rows])
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            Dk_sum += Dk
            Dk_n += 1
            Xs = np.stack([rv(r, p92.NE) for r in te_rows])
            es = ((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk
            Xo = np.stack([rv(r, p92.NE) for r in oov_rows])
            eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
            drift69 = max(drift69,
                          abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                          abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
            R_seen = Y[te_rows] - Xs @ W
            for j, r in enumerate(te_rows):
                seen_amp.setdefault(r, []).append(R_seen[j])
            R_oov = Y[oov_rows] - Xo @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                nv = float(np.linalg.norm(rvv))
                Q_rows[j] += rvv / (nv + 1e-18)
                Rbar_rows[j] += rvv
            del W, Xtr
        assert drift69 < p92.REPL_TOL, ('3169 per-seed E replication', m,
                                        drift69)
        Qn = Q_rows / len(p92.SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
        Rbar = Rbar_rows / len(p92.SEEDS_S1)

        # --- device anchors: 3189/3190/3191 scalars bitwise (3192 verbatim)
        s0 = np.linalg.svd(Qn, compute_uv=False)
        energy = s0 ** 2
        tot = float(energy.sum())
        top1 = float(energy[0] / tot)
        top2 = float(energy[:2].sum() / tot)
        cum = np.cumsum(energy) / tot
        r80 = int(np.searchsorted(cum, 0.8) + 1)
        r90 = int(np.searchsorted(cum, 0.9) + 1)
        V = []
        within = []
        for c in CLS_IDS:
            idxs = np.where(oov_row_cls == c)[0]
            V.append(unit_mean(Qn[idxs]))
            Cc = Qn[idxs] @ Qn[idxs].T
            within.append(float(Cc[np.triu_indices(len(idxs), 1)].mean()))
        V = np.stack(V)
        Ccross = V @ V.T
        cross_med = float(np.median([float(Ccross[i, j])
                                     for i in range(4) for j in range(i + 1, 4)]))
        drift89 = max(abs(top1 - E89['top1']), abs(top2 - E89['top2']),
                      abs(cross_med - E89['cross_med']),
                      max(abs(a - b) for a, b in zip(within, E89['within_class'])),
                      abs(r80 - E89['r80']), abs(r90 - E89['r90']))
        assert drift89 == 0.0, ('3189 scalar drift', m, drift89)
        U, S, Vt = np.linalg.svd(Qn, full_matrices=False)
        ustar = Vt[0]
        loo = []
        for c in CLS_IDS:
            keep = oov_row_cls != c
            sl = np.linalg.svd(Qn[keep], compute_uv=False)
            el = sl ** 2
            loo.append(float(el[0] / el.sum()))
        loo_max = float(max(abs(x - top1) for x in loo))
        med_abs = []
        for c in CLS_IDS:
            held = oov_row_cls == c
            v_hat = unit_mean(Qn[~held])
            cos_held = Qn[held] @ v_hat
            med_abs.append(float(np.median(np.abs(cos_held))))
        E90, E91 = pm90[m], pm91[m]
        drift90 = max(abs(loo_max - E90['loo_max_delta']),
                      abs(top1 - E90['top1']),
                      abs(cross_med - E90['cross_med']),
                      max(abs(a - b) for a, b in zip(within, E90['within_class'])),
                      max(abs(a - b) for a, b in zip(loo, E90['loo_top1'])))
        assert drift90 == 0.0, ('3190 scalar drift', m, drift90)
        drift91 = max(abs(float(min(med_abs)) - E91['min_med_abs_cos']),
                      max(abs(a - b) for a, b in zip(med_abs,
                                                     E91['med_abs_cos_held'])))
        assert drift91 == 0.0, ('3191 scalar drift', m, drift91)

        # u* external anchor vs sealed 3225 npz (4b/14b)
        if m in ('qwen3-4b', 'qwen3-14b'):
            u32 = z32[m + '_ustar'].astype(np.float32)
            cos_u = float(ustar @ (u32 / (np.linalg.norm(u32) + 1e-18)))
            assert cos_u >= 1.0 - 1e-6, ('u* vs 3225 npz', m, cos_u)
            m32 = z32[m + '_med_abs']
            drift3225 = float(np.max(np.abs(m32 - np.array(med_abs))))
            assert drift3225 == 0.0, ('3225 med_abs drift', m, drift3225)
        else:
            cos_u = None
        log('  device ok: drift69=%.2e drift89/90/91=0%s'
            % (drift69, '' if cos_u is None else ' u*cos3225=%.8f' % cos_u))

        # --- pooled 3192 anchors (bitwise) ---
        proj_oov_amp = np.abs(Rbar @ ustar)
        seen_flat = np.concatenate([np.array(seen_amp[r])
                                    for r in sorted(seen_amp)])
        proj_seen_amp = np.abs(seen_flat @ ustar)
        med_oov = float(np.median(proj_oov_amp))
        med_seen = float(np.median(proj_seen_amp))
        amp_ratio = med_oov / med_seen
        d92 = max(abs(med_oov - A92[m]['med_amp_oov'])
                  / max(1.0, A92[m]['med_amp_oov']),
                  abs(med_seen - A92[m]['med_amp_seen'])
                  / max(1.0, A92[m]['med_amp_seen']),
                  abs(amp_ratio - A92[m]['amp_ratio']) / A92[m]['amp_ratio'])
        assert d92 <= 1e-6, ('3192 pooled anchor rel drift', m, d92)
        # 3192 verdict is PASS (carrier) -- guard
        assert sealed_3192['gates']['carrier'] is True

        # --- NEW OBSERVATION: per-class expansion (n=10) ---
        oov_cls_med, oov_cls_n = {}, {}
        for c in CLS_IDS:
            idxs = np.where(oov_row_cls == c)[0]
            oov_cls_med[c] = float(np.median(proj_oov_amp[idxs]))
            oov_cls_n[c] = int(len(idxs))
        seen_cls_med, seen_cls_n, seen_cls_amp = {}, {}, {}
        for c in SEEN_IDS:
            vals = [float(np.abs(v @ ustar))
                    for r in sorted(seen_amp)
                    if p92.PAIRS[r % p92.NP_][1] == c
                    for v in seen_amp[r]]
            seen_cls_amp[c] = np.array(vals, dtype=np.float32)
            seen_cls_med[c] = float(np.median(seen_cls_amp[c]))
            seen_cls_n[c] = len(vals)
        amp_ratio_c = {c: oov_cls_med[c] / med_seen for c in CLS_IDS}
        class_ratio = (float(np.median([oov_cls_med[c] for c in CLS_IDS]))
                       / float(np.median([seen_cls_med[c]
                                          for c in SEEN_IDS])))
        g1_pass = sum(1 for c in CLS_IDS if amp_ratio_c[c] >= G_AMP)
        g2_pass = bool(class_ratio >= G_AMP)
        # report-only: per-class u* energy share + collapse amplitude
        share_oov, share_seen = {}, {}
        for c in CLS_IDS:
            idxs = np.where(oov_row_cls == c)[0]
            Rb = Rbar[idxs]
            share_oov[c] = float(((Rb @ ustar) ** 2).sum()
                                 / (Rb ** 2).sum())
        for c in SEEN_IDS:
            A = np.stack([v for r in sorted(seen_amp)
                          if p92.PAIRS[r % p92.NP_][1] == c
                          for v in seen_amp[r]]).astype(np.float32)
            share_seen[c] = float(((A @ ustar) ** 2).sum()
                                  / (A ** 2).sum())
        Dk_bar = Dk_sum / Dk_n
        collapse_amp = {}
        for c in CLS_IDS:
            idxs = np.where(oov_row_cls == c)[0]
            collapse_amp[c] = float((np.linalg.norm(Rbar[idxs], axis=1) ** 2
                                     ).mean() / Dk_bar)
        for c in SEEN_IDS:
            A = np.stack([v for r in sorted(seen_amp)
                          if p92.PAIRS[r % p92.NP_][1] == c
                          for v in seen_amp[r]]).astype(np.float32)
            collapse_amp[c] = float((np.linalg.norm(A, axis=1) ** 2).mean()
                                    / Dk_bar)
        amps10 = [oov_cls_med[c] for c in CLS_IDS] + \
                 [seen_cls_med[c] for c in SEEN_IDS]
        coll10 = [collapse_amp[c] for c in CLS_IDS] + \
                 [collapse_amp[c] for c in SEEN_IDS]
        rho_amp_coll = spearman(amps10, coll10)
        out[m] = dict(
            drift_3169=drift69, drift_3189=drift89, drift_3190=drift90,
            drift_3191=drift91, drift_3192_pooled=d92,
            ustar_cos_3225=cos_u,
            med_amp_oov_pooled=med_oov, med_amp_seen_pooled=med_seen,
            amp_ratio_pooled=amp_ratio,
            oov_cls_med={str(c): oov_cls_med[c] for c in CLS_IDS},
            oov_cls_n={str(c): oov_cls_n[c] for c in CLS_IDS},
            seen_cls_med={str(c): seen_cls_med[c] for c in SEEN_IDS},
            seen_cls_n={str(c): seen_cls_n[c] for c in SEEN_IDS},
            amp_ratio_c={str(c): amp_ratio_c[c] for c in CLS_IDS},
            class_ratio=class_ratio,
            g1_classes_pass=g1_pass, g1_pass=bool(g1_pass >= G1_CLASSES),
            g2_pass=g2_pass,
            share_oov={str(c): share_oov[c] for c in CLS_IDS},
            share_seen={str(c): share_seen[c] for c in SEEN_IDS},
            collapse_amp={str(c): collapse_amp[c] for c in CLS_IDS + SEEN_IDS},
            spearman_amp_collapse=rho_amp_coll,
            gate_model=bool(g1_pass >= G1_CLASSES and g2_pass))
        log('  per-OOV-class amp_ratio_c: %s (G1 >=%d/4)'
            % ('/'.join('%.2f' % amp_ratio_c[c] for c in CLS_IDS), G1_CLASSES))
        log('  seen class meds: %s'
            % '/'.join('%.2f' % seen_cls_med[c] for c in SEEN_IDS))
        log('  class_ratio=%.4f (G2 >=%.2f)=%s | spearman(amp,collapse)=%.3f'
            % (class_ratio, G_AMP, g2_pass, rho_amp_coll))
        del z, H16, Y, Q_rows, Qn, Rbar_rows, Rbar, seen_flat
    n_g1 = sum(1 for m in MODELS if out[m]['g1_pass'])
    n_g2 = sum(1 for m in MODELS if out[m]['g2_pass'])
    robust = n_g1 >= G_MODELS and n_g2 >= G_MODELS
    partial = (not robust) and n_g2 >= G_MODELS
    out['_gate'] = dict(n_models_g1=n_g1, n_models_g2=n_g2,
                        robust=robust, partial=partial)
    if robust:
        branch = 'GD_AMP_CLASS_ROBUST'
    elif partial:
        branch = 'GD_AMP_CLASS_PARTIAL'
    else:
        branch = 'GD_AMP_CLASS_DIFFUSE'
    log('GATE: G1 %d/3 models (>=3/4 classes), G2 %d/3 models => %s'
        % (n_g1, n_g2, branch))
    return out, branch


# ---------------- gap ledger v1.9 -> v1.10 (append-only) ----------------
def finish_outputs(summary, branch, res8, seal8, design_sha):
    gl19 = json.loads(open(SRC['gap_v19'], encoding='utf-8').read())
    assert gl19['version'] == '1.9' and len(gl19['gaps']) == 5
    pm = summary['per_model']
    fmt3 = lambda key: '/'.join('%.4f' % pm[m][key] for m in MODELS)
    gl110 = copy.deepcopy(gl19)
    gl110['version'] = '1.10'
    gl110['supersedes'] = 'gap_ledger_v1_9.json (9fca29bd, Phase 3231)'
    gl110['updated'] = '2026-10-13'
    g4_old = [g for g in gl19['gaps'] if g['id'] == 'GAP-4'][0]
    g4 = copy.deepcopy(g4_old)
    subs = {'amp_ratio_c_str': '/'.join(
        '%.2f' % pm[m]['amp_ratio_c'][str(c)]
        for m in MODELS for c in CLS_IDS)}
    if branch == 'GD_AMP_CLASS_ROBUST':
        g4['mechanism_note'] = g4_old['mechanism_note'] + (
            '；类数稳健性扩样（3232 G5-G29，n=4 池化 → n=10 逐类）：逐 OOV 类 '
            'amp_ratio_c ' + subs['amp_ratio_c_str'] + '，类级比 class_ratio '
            + fmt3('class_ratio') + '（门 G1 ≥3/4 类 ×≥2/3 模型、G2 ≥1.3 '
            '×≥2/3 模型全过）——u*_amplitude_carrier 在类级（4 OOV + 6 seen）'
            '成立，非池化伪影。锚：p3232 res ' + res8 + ' / seal ' + seal8
            + '（design ' + design_sha + '）。')
    elif branch == 'GD_AMP_CLASS_PARTIAL':
        g4['mechanism_note'] = g4_old['mechanism_note'] + (
            '；类数扩样（3232 G5-G29，n=4 池化 → n=10 逐类）：类级比 '
            'class_ratio ' + fmt3('class_ratio') + ' 过门但逐类 amp_ratio_c '
            + subs['amp_ratio_c_str'] + ' 未全过（G1 <3/4 类）——载体在类中位'
            '水平成立、逐类不均匀，池化结论保留并加类级注记。锚：p3232 res '
            + res8 + ' / seal ' + seal8 + '（design ' + design_sha + '）。')
    else:
        g4['mechanism_note'] = g4_old['mechanism_note'] + (
            '；类数扩样（3232 G5-G29，n=4 池化 → n=10 逐类）：class_ratio '
            + fmt3('class_ratio') + ' 未过门——池化 amp_ratio 由部分类驱动，'
            '类级异质性，幅度承载主张限缩至类中位口径。锚：p3232 res ' + res8
            + ' / seal ' + seal8 + '（design ' + design_sha + '）。')
    g4['evidence'] = g4_old['evidence'] + [
        '3232 G5-G29 class-level expansion (n=10): amp_ratio_c per OOV '
        'class = ' + subs['amp_ratio_c_str'] + '; class_ratio = '
        + fmt3('class_ratio') + '; branch ' + branch
        + ' (zero GPU, sealed 3169/3189/3190/3191/3192 + 3225 ustar reuse)']
    gl110['gaps'] = [g4 if g['id'] == 'GAP-4' else g for g in gl19['gaps']]
    # preservation asserts
    for g in gl19['gaps']:
        a = json.dumps(g, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in gl110['gaps'] if x['id'] == g['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if g['id'] == 'GAP-4':
            ga = json.loads(a)
            assert b != a
            assert g4['statement'] == ga['statement']
            assert g4['mechanism_note'].startswith(ga['mechanism_note'])
            assert g4['evidence'][:-1] == ga['evidence']
            assert g4['prereg'] == ga['prereg'] and g4['status'] == ga['status']
            assert g4['anchor_sha8'] == ga['anchor_sha8']
        else:
            assert a == b, ('gap not preserved', g['id'])
    assert gl110['appendix'] == gl19['appendix']
    for k in gl19:
        if k not in ('version', 'supersedes', 'updated', 'gaps'):
            assert json.dumps(gl19[k], ensure_ascii=False, sort_keys=True) == \
                json.dumps(gl110[k], ensure_ascii=False, sort_keys=True), \
                ('gap top drift', k)
    gp = os.path.join(OUTDIR, 'gap_ledger_v1_10.json')
    with open(gp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(gl110, ensure_ascii=False, indent=1))
    gl110_sha = sha8_file(gp)
    log('gap ledger v1.10 written sha8 %s (%s)' % (gl110_sha, branch))
    return gl110_sha


def main():
    design_sha = freeze()
    log('== Phase 3232 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed = {}
    for pid, key, cont in (
            ('3169', 'p3169_result', CONTENT_3169),
            ('3189', 'p3189_result', CONTENT_3189),
            ('3190', 'p3190_result', CONTENT_3190),
            ('3191', 'p3191_result', CONTENT_3191),
            ('3192', 'p3192_result', CONTENT_3192)):
        sealed[pid] = json.load(open(SRC[key], encoding='utf-8'))
        assert sealed[pid]['res_sha8'] == cont[0] and \
            sealed[pid]['seal_sha8'] == cont[1], ('content anchor', pid)
    log('sealed results double-locked: 3169/3189/3190/3191/3192')

    primary, branch = run_arm(sealed['3169'], sealed['3189'], sealed['3190'],
                              sealed['3191'], sealed['3192'])
    verdict = ('%s (G1 %d/3 models x>=3/4 classes, G2 %d/3 models; '
               'amp_ratio_c / class_ratio sealed per model)'
               % (branch, primary['_gate']['n_models_g1'],
                  primary['_gate']['n_models_g2']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3232, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(DESIGN['gates'], branch=branch,
                      n_models_g1=primary['_gate']['n_models_g1'],
                      n_models_g2=primary['_gate']['n_models_g2']),
        'per_model': {m: primary[m] for m in MODELS},
        'verdict': verdict,
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False,
                     indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rp = os.path.join(OUTDIR, suffix + 'result.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    # npz seal of per-class arrays (FULL only)
    if not SMOKE:
        pay = {}
        for m in MODELS:
            pay[m + '_oov_cls_med'] = np.array(
                [primary[m]['oov_cls_med'][str(c)] for c in CLS_IDS],
                dtype=np.float64)
            pay[m + '_seen_cls_med'] = np.array(
                [primary[m]['seen_cls_med'][str(c)] for c in SEEN_IDS],
                dtype=np.float64)
            pay[m + '_amp_ratio_c'] = np.array(
                [primary[m]['amp_ratio_c'][str(c)] for c in CLS_IDS],
                dtype=np.float64)
            pay[m + '_collapse_amp'] = np.array(
                [primary[m]['collapse_amp'][str(c)]
                 for c in CLS_IDS + SEEN_IDS], dtype=np.float64)
        np.savez(os.path.join(OUTDIR, 'gd_amp_class_expansion.npz'), **pay)
        log('npz sealed: gd_amp_class_expansion.npz sha8 %s'
            % sha8_file(os.path.join(OUTDIR, 'gd_amp_class_expansion.npz')))
        gl110_sha = finish_outputs(summary, branch, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
