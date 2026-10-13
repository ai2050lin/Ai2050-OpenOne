# -*- coding: utf-8 -*-
"""Phase 3178: G5-B3 FTR-14 A2 attribution decomposition (zero GPU, sampling-null calibration).

Per prereg in MEMO Phase 3177 tail, section "接续：预注册 3178" (frozen before
any observation):
  3177 A2 failure mixes (i) out-of-sample sampling penalty (v_shared estimated
  from 32 cells; held cells read systematically lower than in-sample baseline;
  a2_gap_train 0.094-0.150 corroborates) and (ii) relation-specific residual.
  Decomposition protocol: entity-LOO null within relation (16 folds): v_null
  rebuilt from the other 15 entities x 2 polarity (30 cells); the held
  entity's 2-cell cos gap = pure sampling penalty p(m, rel); relation
  residual r(m) = relation gap (cos_train - cos_held, same v_shared, 3177
  data) - p(m, rel).  Gate: r(m) < 0.05 x3 models => relation independence
  supported under the calibration bar (FTR-14 upgrade execution deferred to
  3179, anchors = 3178 calibration readings + 3177 A1/exch band); otherwise
  the relation-specific component is confirmed and quantified (FTR-14 stays
  E1).  Protocol details exported verbatim from the 3177/3157 scripts at
  freeze time and frozen in execution.json.

Freeze-time protocol pinning (prereg delegates detail pinning to freeze):
  - Data = sealed 3157 collect npz x3 (file sha8 95a25965/9552086d/23dd74eb)
    => zero-GPU arm (prereg states 零 GPU explicitly; no new forward passes).
  - Device gates verbatim 3157: layer anchors (nl/kout), tc_full drift < 1e-9,
    exchR/exchN_c0 drift < 1e-9, exch cross-model mean vs 3157 summary.
  - 3177 replication gate: A2 readings recomputed with the 3177 verbatim
    recipe (dC row order ei,rl,pl; dCn = dC/(||dC||+1e-18); unit_mean with
    +1e-18); drift < 1e-6 vs the sealed 3177 result.json (file sha8
    b2ad52ef pinned; content res/seal d451030f/7b746eb2 double-locked at
    runtime).  Sealed numbers are loaded from that pinned file - no
    transcription of float literals.
  - Null device: for each model m and relation rel in {isa, hasa}, fold
    e in 0..15:
      held rows  = {e*4 + rel*2 + p : p in {0,1}}            (2 cells)
      train rows = dC_rows(rel) minus held rows               (30 cells,
                    order preserved: entity asc, polarity asc)
      v_null  = unit_mean(dCn[train])
      p_fold(e) = mean(dCn[train] @ v_null) - mean(dCn[held] @ v_null)
    p(m, rel) = mean_e p_fold(e).  Sanity: 96/96 finite, |p_fold| < 0.5.
  - Residual: g(m, rel) = cos_train_a2 - cos_held_a2 (the 3177 same-v_shared
    gap; assert equals sealed |a2_gap_train| within 1e-12, all six units
    cos_train > cos_held in 3177);
    r(m, rel) = g(m, rel) - p(m, rel);
    r(m) = mean over rel in {isa, hasa}  (aggregation pinned here before any
    observation; per-fold r(m, rel) and per-fold p disclosed in result).
  - GATE: r(m) < 0.05 for all 3 models =>
    relation_independence_supported_under_calibration_bar.  Per prereg the
    FTR-14 upgrade execution belongs to 3179 (upgrade anchors = 3178
    calibration readings + 3177 A1/exch band) => registry/html untouched on
    PASS.
  - FAIL branch: relation-specific component confirmed & quantified =>
    registry v1.4 -> v1.5 with FTR-14 counter_evidence append ONLY (3177
    precedent F16 handling), stays E1_repeatable, failures 16 unchanged,
    upgrade_log 5 unchanged; atlas_v1_6.html full re-render with per-field
    data-k verification (3173 discipline).  Honest registry.
  - SMOKE (env P3178_SMOKE=1): zero-GPU => numbers identical across modes;
    artifacts written with smoke_ prefix; registry byte-identity between
    SMOKE and FULL asserted when FAIL branch renders.

SMOKE runs the identical computation as FULL (no truncation); the env var
only switches artifact filenames and the log marker.
"""
import copy
import datetime
import hashlib
import html
import io
import json
import os
import re

import numpy as np

BASE = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(BASE, r'tests\glm5\result\rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3178', 'g5b3_ftr14_a2_attribution')
SMOKE = os.environ.get('P3178_SMOKE', '') == '1'
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
N_E = 16

SRC = {
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3157_res_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\result.json'),
    'p3157_res_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\result.json'),
    'p3157_res_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\result.json'),
    'p3157_summary': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\summary\result_summary.json'),
    'p3177_result': os.path.join(SRC_DIR, r'phase3177\g5b2_ftr14_heldout\result.json'),
    'registry_v14': os.path.join(SRC_DIR, r'phase3177\g5b2_ftr14_heldout\atlas_registry_v1_4.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a',
    'p3177_result': 'b2ad52ef',
    'registry_v14': 'fb11d633', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}

# 3157 sealed anchors (device gates; double-locked against result.json at runtime)
ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, npz_sha8='95a25965', res_sha8='c0db3455', seal_sha8='41916ed8',
                     tc_mean_cos=0.5629908442497253, exchR_c0=1.0320992213108435,
                     exchN_c0=1.1928509676419494, exchange_obs=1.1928509676419494),
    'qwen3-14b': dict(nl=40, kout=39, npz_sha8='9552086d', res_sha8='1270c921', seal_sha8='98c06ab0',
                      tc_mean_cos=0.5614360570907593, exchR_c0=1.1211844589001665,
                      exchN_c0=1.294480635663979, exchange_obs=1.294480635663979),
    'glm4': dict(nl=40, kout=39, npz_sha8='23dd74eb', res_sha8='a1d9a98a', seal_sha8='e8be1026',
                 tc_mean_cos=0.5640953183174133, exchR_c0=1.0723271372900645,
                 exchN_c0=1.3311207542158712, exchange_obs=1.3311207542158712),
}
SUMMARY_3157 = dict(res_sha8='0fe043bf', seal_sha8='cfa5c3ed', fpmin_kout=0.9857871501170492,
                    exch_mean_obs=1.2728174525072664, comm_class='commutative_partial')

# 3177 sealed result: content shas double-locked; numbers loaded from the
# pinned file at runtime (file sha8 b2ad52ef asserted in SHA_ANCHOR).
ANCH_3177 = dict(res_sha8='d451030f', seal_sha8='7b746eb2')

DEV_TOL = 1e-9
REPL_TOL = 1e-6
R_TOL = 0.05

DESIGN = {
    'phase': 3178, 'name': 'g5b3_ftr14_a2_attribution', 'zero_gpu': True,
    'prereg': 'MEMO 2026-10-09 Phase 3177 tail (G5-B3): 3177 A2 failure = '
              'sampling penalty + relation-specific residual; entity-LOO null '
              'within relation (16 folds, v_null from 30 cells, held 2-cell '
              'gap) = pure sampling penalty p(m, rel); relation residual '
              'r(m) = relation gap (cos_train - cos_held, same v_shared, 3177 '
              'data) - p(m, rel); gate r(m) < 0.05 x3 models => relation '
              'independence supported under calibration bar (upgrade execution '
              'deferred to 3179); else relation-specific component confirmed '
              'and quantified (FTR-14 stays E1)',
    'freeze_pinning': dict(
        zero_gpu='data = sealed 3157 collect npz x3 (file sha8 pinned); no new '
                 'forward passes; prereg states zero-GPU explicitly',
        device_gates='verbatim 3157: layer anchors nl/kout; tc_full recompute '
                     'drift < 1e-9; exchR/exchN_c0 recompute drift < 1e-9; exch '
                     'cross-model mean vs 3157 summary 1.2728174525072664',
        replication_3177='A2 readings (cos_full_a2, per-fold cos_train_a2 / '
                         'cos_held_a2 / a2_gap_train, tc_full, exchange_obs) '
                         'recomputed with the 3177 verbatim recipe; drift < 1e-6 '
                         'vs sealed 3177 result.json (file sha8 b2ad52ef pinned, '
                         'content res/seal d451030f/7b746eb2 double-locked; '
                         'sealed numbers loaded from the pinned file)',
        null_device='entity-LOO within relation rel: fold e holds out rows '
                    '{e*4+rel*2+p}, train = remaining 30 cells of rel in '
                    'dC_rows order; v_null = unit_mean(dCn[train]); p_fold(e) = '
                    'mean(dCn[train] @ v_null) - mean(dCn[held] @ v_null); '
                    'p(m, rel) = mean over 16 folds; sanity 96/96 finite, '
                    '|p_fold| < 0.5',
        residual='g(m, rel) = cos_train_a2 - cos_held_a2 (3177 same-v_shared '
                 'gap; assert equals sealed |a2_gap_train| < 1e-12); '
                 'r(m, rel) = g(m, rel) - p(m, rel); r(m) = mean over '
                 '{isa, hasa} - aggregation pinned pre-observation; per-fold '
                 'values disclosed',
        gate='r(m) < 0.05 x3 models => relation_independence_supported_under_'
             'calibration_bar; per prereg the FTR-14 upgrade execution belongs '
             'to 3179 (anchors = 3178 calibration readings + 3177 A1/exch '
             'band) => registry/html untouched on PASS',
        fail_branch='relation-specific component confirmed & quantified => '
                    'registry v1.4 -> v1.5, FTR-14 counter_evidence append '
                    'ONLY (quantified numbers), stays E1_repeatable, failures '
                    '16 unchanged, upgrade_log 5 unchanged; atlas_v1_6.html '
                    'full re-render with per-field data-k verification',
        smoke='zero-GPU: computation identical across modes; smoke_-prefixed '
              'artifacts; registry byte-identity asserted FULL vs SMOKE when '
              'FAIL branch renders'),
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'anchors_3157': {'per_model': ANCH_3157, 'summary': SUMMARY_3157},
    'anchors_3177': ANCH_3177,
    'gates': dict(device_drift=DEV_TOL, replication_drift=REPL_TOL, r_tol=R_TOL),
}

LOG = []
meta_sha_global = {'reg15': None}


def log(s):
    LOG.append(str(s))
    print(s, flush=True)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    body = dict(DESIGN)
    raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    d8 = sha8(raw.encode('utf-8'))
    body['design_sha8'] = d8
    raw2 = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        with io.open(exep, 'w', encoding='utf-8') as f:
            f.write(raw2)
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ------------------------------------------------- verbatim 3157/3177 recipes
def commutators(H, IDX, k, cx):
    """3157 verbatim: relation/negation commutators at layer k, context cx."""
    dR_p, dR_m, dN_i, dN_h = [], [], [], []
    for ei in range(N_E):
        h_isa_p = H[IDX[(ei, 0, 0, cx)], k]
        h_has_p = H[IDX[(ei, 1, 0, cx)], k]
        h_isa_m = H[IDX[(ei, 0, 1, cx)], k]
        h_has_m = H[IDX[(ei, 1, 1, cx)], k]
        dR_p.append(h_isa_p - h_has_p)
        dR_m.append(h_isa_m - h_has_m)
        dN_i.append(h_isa_m - h_isa_p)
        dN_h.append(h_has_m - h_has_p)
    dR_p, dR_m = np.array(dR_p), np.array(dR_m)
    dN_i, dN_h = np.array(dN_i), np.array(dN_h)
    num_r = float(np.linalg.norm((dR_p - dR_m).ravel()))
    den_r = 0.5 * float(np.linalg.norm(dR_p.ravel()) + np.linalg.norm(dR_m.ravel())) + 1e-18
    num_n = float(np.linalg.norm((dN_i - dN_h).ravel()))
    den_n = 0.5 * float(np.linalg.norm(dN_i.ravel()) + np.linalg.norm(dN_h.ravel())) + 1e-18
    return num_r / den_r, num_n / den_n


def dC_rows(rel_filter):
    """Row indices into the dC stack (loop order ei, rl, pl = verbatim 3157)."""
    out = []
    for ei in range(N_E):
        for rl in range(2):
            if rel_filter is not None and rl != rel_filter:
                continue
            for pl in range(2):
                out.append(ei * 4 + rl * 2 + pl)
    return out


def unit_mean(dCn):
    v = dCn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


# ------------------------------------------------- arm
def run_arm(sealed_3177):
    out = {}
    for m in MODELS:
        ak = ANCH_KEY[m]
        A = ANCH_3157[ak]
        R17 = sealed_3177['arm']['per_model'][m]
        z = np.load(SRC[NPZ_KEY[m]])
        H = z['H'].astype(np.float32)
        NROWS, NH, D = H.shape
        NL = NH - 1
        KOUT = NL - 1
        assert NROWS == 128, ('rows', m, NROWS)
        assert (NL, KOUT) == (A['nl'], A['kout']), ('layer anchor', m, NL, KOUT)
        EI = z['ei'].astype(int)
        RL = z['rel'].astype(int)
        PL = z['pol'].astype(int)
        CX = z['ctx'].astype(int)
        IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
        assert len(IDX) == 128, ('cells', m, len(IDX))
        for ei in range(N_E):
            for rl in range(2):
                for pl in range(2):
                    for cx in range(2):
                        assert (ei, rl, pl, cx) in IDX, ('missing cell', m, ei, rl, pl, cx)

        # device gate: exch recompute verbatim (c0 gated vs anchors; c1 reported)
        ex = {}
        for cx in (0, 1):
            er, en_ = commutators(H, IDX, KOUT, cx)
            ex['exchR_c%d' % cx] = er
            ex['exchN_c%d' % cx] = en_
        drift_ex = max(abs(ex['exchR_c0'] - A['exchR_c0']),
                       abs(ex['exchN_c0'] - A['exchN_c0']))
        assert drift_ex < DEV_TOL, ('exch drift', m, drift_ex)
        exchange_obs = max(ex['exchR_c0'], ex['exchN_c0'])
        assert abs(exchange_obs - A['exchange_obs']) < DEV_TOL, ('exchange_obs', m)
        drift_ex_17 = abs(exchange_obs - R17['exchange_obs'])
        assert drift_ex_17 < REPL_TOL, ('3177 exchange_obs drift', m, drift_ex_17)

        # device gate: tc_full recompute verbatim (metric path, 64 cells)
        dC = []
        for ei in range(N_E):
            for rl in range(2):
                for pl in range(2):
                    dC.append(H[IDX[(ei, rl, pl, 1)], KOUT] - H[IDX[(ei, rl, pl, 0)], KOUT])
        dC = np.array(dC)
        assert np.isfinite(dC).all(), ('nonfinite dC', m)
        dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
        cosM = dCn @ dCn.T
        tri = cosM[np.triu_indices(len(dC), 1)]
        tc_full = float(tri.mean())
        drift_tc = abs(tc_full - A['tc_mean_cos'])
        assert drift_tc < DEV_TOL, ('tc drift', m, drift_tc)
        drift_tc_17 = abs(tc_full - R17['tc_full'])
        assert drift_tc_17 < REPL_TOL, ('3177 tc_full drift', m, drift_tc_17)

        # A2 full reading (v_full over all 64 rows) - verbatim 3177
        v_full = unit_mean(dCn)
        cos_full_a2 = float((dCn @ v_full).mean())
        drift_full_17 = abs(cos_full_a2 - R17['cos_full_a2'])
        assert drift_full_17 < REPL_TOL, ('3177 cos_full drift', m, drift_full_17)

        # folds: held-out relation isa (rl=0) / hasa (rl=1) - verbatim 3177
        folds = {}
        for rl_held, fname in ((0, 'isa'), (1, 'hasa')):
            rows_h = dC_rows(rl_held)
            rows_tr = dC_rows(1 - rl_held)
            assert len(rows_h) == 32 and len(rows_tr) == 32
            dCn_h = dCn[rows_h]
            dCn_tr = dCn[rows_tr]
            v_sh = unit_mean(dCn_tr)
            cos_held_a2 = float((dCn_h @ v_sh).mean())
            cos_train_a2 = float((dCn_tr @ v_sh).mean())
            a2_gap_train = abs(cos_held_a2 - cos_train_a2)
            E17 = R17['folds'][fname]
            d_tr = abs(cos_train_a2 - E17['cos_train_a2'])
            d_he = abs(cos_held_a2 - E17['cos_held_a2'])
            d_gap = abs(a2_gap_train - E17['a2_gap_train'])
            assert max(d_tr, d_he, d_gap) < REPL_TOL, \
                ('3177 fold drift', m, fname, d_tr, d_he, d_gap)
            folds[fname] = dict(cos_held_a2=cos_held_a2, cos_train_a2=cos_train_a2,
                                a2_gap_train=a2_gap_train,
                                repl_drift=max(d_tr, d_he, d_gap))

        # NEW: entity-LOO null within relation (pure sampling penalty)
        nulls = {}
        for rl, rname in ((0, 'isa'), (1, 'hasa')):
            rows_all = dC_rows(rl)
            pf = []
            for e in range(N_E):
                held_rows = [e * 4 + rl * 2 + p for p in (0, 1)]
                train_rows = [r for r in rows_all if r not in held_rows]
                assert len(train_rows) == 30, ('null train size', m, rname, e)
                v_null = unit_mean(dCn[train_rows])
                cos_in = float((dCn[train_rows] @ v_null).mean())
                cos_out = float((dCn[held_rows] @ v_null).mean())
                pf.append(dict(e=e, cos_in=cos_in, cos_out=cos_out,
                               p_fold=cos_in - cos_out))
            assert all(np.isfinite(x['p_fold']) for x in pf), ('null nonfinite', m, rname)
            assert max(abs(x['p_fold']) for x in pf) < 0.5, ('null sanity', m, rname)
            nulls[rname] = dict(folds=pf,
                                p_mean=float(np.mean([x['p_fold'] for x in pf])))

        # residual: g (3177 same-v_shared gap) - p (null calibration)
        resid = {}
        for rname in ('isa', 'hasa'):
            g = folds[rname]['cos_train_a2'] - folds[rname]['cos_held_a2']
            assert abs(g - folds[rname]['a2_gap_train']) < 1e-12, ('gap sign', m, rname)
            p_cal = nulls[rname]['p_mean']
            resid[rname] = dict(gap=g, p=p_cal, r=g - p_cal)
        r_m = float((resid['isa']['r'] + resid['hasa']['r']) / 2.0)

        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full,
                      tc_full_drift=drift_tc, exch_drift=drift_ex,
                      exchange_obs=exchange_obs, cos_full_a2=cos_full_a2,
                      folds=folds, null=nulls, resid=resid, r_m=r_m)
        log('arm %s: KOUT=%d tc_full=%.6f (drift %.2e) exchange_obs=%.4f '
            'cos_full_a2=%.4f' % (m, KOUT, tc_full, drift_tc, exchange_obs, cos_full_a2))
        for rname in ('isa', 'hasa'):
            log('  %s: gap=%.4f p_null=%.4f r=%.4f (p per fold %.4f)' % (
                rname, resid[rname]['gap'], resid[rname]['p'], resid[rname]['r'],
                min(x['p_fold'] for x in nulls[rname]['folds'])))
        log('  r(m) = mean(isa, hasa) = %.4f  [%s]' % (
            r_m, 'PASS' if r_m < R_TOL else 'FAIL'))
    exch_mean = float(np.mean([out[m]['exchange_obs'] for m in MODELS]))
    drift_mean = abs(exch_mean - SUMMARY_3157['exch_mean_obs'])
    assert drift_mean < DEV_TOL, ('exch mean drift', drift_mean)
    calib = all(out[m]['r_m'] < R_TOL for m in MODELS)
    log('device gates: tc/exch/mean drift all < %.0e (3 models)' % DEV_TOL)
    log('replication gate vs sealed 3177: max fold drift %s (tol %.0e)' % (
        '/'.join('%.2e' % max(out[m]['folds'][f]['repl_drift'] for f in ('isa', 'hasa'))
                 for m in MODELS), REPL_TOL))
    log('GATE calibration (r(m) < %.2f x3): %s' % (R_TOL, calib))
    log('r(m): ' + ' / '.join('%.4f' % out[m]['r_m'] for m in MODELS))
    return out, calib, exch_mean


# ------------------------------------------------- registry (FAIL branch only)
def build_registry_fail(out, exch_mean):
    reg14 = json.loads(open(SRC['registry_v14'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    reg2 = json.load(io.open(SRC['registry_3162'], encoding='utf-8'))
    nodes = reg2['audit']['nodes']
    assert reg14['version'] == '1.4' and gl15['version'] == '1.5'
    assert len(reg14['features']) == 22 and len(nodes) == 16
    assert len(reg14['failures']) == 16 and len(reg14['upgrade_log']) == 5
    f14_old = [f for f in reg14['features'] if f['id'] == 'FTR-14'][0]
    assert f14_old['evidence_level'] == 'E1_repeatable'
    assert len(f14_old['counter_evidence']) == 2
    f14 = copy.deepcopy(f14_old)

    parts = []
    for m in MODELS:
        parts.append('%s r=%.4f（isa g=%.4f p=%.4f r=%.4f；hasa g=%.4f p=%.4f r=%.4f）' % (
            m, out[m]['r_m'],
            out[m]['resid']['isa']['gap'], out[m]['resid']['isa']['p'], out[m]['resid']['isa']['r'],
            out[m]['resid']['hasa']['gap'], out[m]['resid']['hasa']['p'], out[m]['resid']['hasa']['r']))
    add = (' 3178 A2 归因分解定判（零 GPU，实体留出 null 标定，freeze 前协议钉定）：'
           '关系缺口 g=cos_train−cos_held（3177 同 v_shared）减实体留出纯采样惩罚 '
           'p（同关系内 16 fold，v_null 由 30 细胞重建、留出 2 细胞读数）后，'
           '关系残差 r(m)=g−p 全部不低于 0.05 标定 bar（'
           + '；'.join(parts) + '）——关系特异分量确证并量化登记；'
           'FTR-14 保持 E1_repeatable（3177 held_out 门失败 F16 的归因定量；'
           'exch 均值 %.4f 带保持不变走查）。' % exch_mean)
    f14['counter_evidence'] = f14['counter_evidence'] + [add]

    reg15 = copy.deepcopy(reg14)
    reg15['version'] = '1.5'
    reg15['supersedes'] = 'atlas_registry_v1_4.json (%s, Phase 3177)' % SHA_ANCHOR['registry_v14']
    reg15['features'] = [f14 if f['id'] == 'FTR-14' else f for f in reg14['features']]
    assert len(reg15['features']) == 22

    for f in reg14['features']:
        fid = f['id']
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b_obj = [x for x in reg15['features'] if x['id'] == fid][0]
        b = json.dumps(b_obj, ensure_ascii=False, sort_keys=True)
        if fid == 'FTR-14':
            fa, fb = json.loads(a), json.loads(b)
            ce_old = fa.pop('counter_evidence')
            ce_new = fb.pop('counter_evidence')
            assert json.dumps(fa, ensure_ascii=False, sort_keys=True) == \
                json.dumps(fb, ensure_ascii=False, sort_keys=True), \
                ('FTR-14 non-counter fields changed', fid)
            assert ce_new[:len(ce_old)] == ce_old and \
                len(ce_new) == len(ce_old) + 1, ('counter append', fid)
        else:
            assert a == b, ('feature not preserved', fid)
    log('preservation: 21 features byte-for-byte; FTR-14 counter_evidence append only')
    assert json.dumps(reg14['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg15['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    assert json.dumps(reg14['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg15['upgrade_log'], ensure_ascii=False, sort_keys=True), 'upgrades changed'
    f14_new = [x for x in reg15['features'] if x['id'] == 'FTR-14'][0]
    assert f14_new['evidence_level'] == 'E1_repeatable', ('level must stay E1', )
    assert len(f14_new['counter_evidence']) == 3
    log('G4 asserts OK (E1 kept, counter len 2 -> 3, failures/upgrades untouched)')
    return reg15, nodes, gl15


# ------------------------------------------------- html render (3173 discipline)
CSS = """
:root { --ink:#1a1d23; --muted:#5b6472; --line:#d9dee6; --bg:#f7f8fa; --card:#ffffff;
  --k:#2456a6; --s:#0f7b6c; --r:#8a4fb8; --mech:#b35900; --ctx:#7a5c12; --ctl:#8c2f39;
  --read:#3a5a8c; --gate:#5a4a8a; --limit:#444; --sxk:#316aa8; --rxs:#2f855a; --rxk:#6b46a0; }
* { box-sizing: border-box; }
body { margin:0; padding:24px 28px 60px; background:var(--bg); color:var(--ink);
  font:14px/1.65 "Segoe UI","Microsoft YaHei","PingFang SC",sans-serif; }
h1 { font-size:22px; margin:0 0 4px; }
h2 { font-size:17px; margin:34px 0 10px; padding-bottom:6px; border-bottom:2px solid var(--line); }
h3 { font-size:14px; margin:0 0 8px; color:var(--muted); font-weight:600; }
.sub { color:var(--muted); font-size:12.5px; }
.mono { font-family:Consolas,"Courier New",monospace; font-size:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:10px 0; }
.badge { display:inline-block; border-radius:10px; padding:1px 9px; font-size:11.5px; margin-right:6px;
  border:1px solid var(--line); background:#eef1f5; color:var(--ink); }
.b-E2_predictive { background:#e3f0e3; border-color:#9fcaa0; color:#1e5c22; }
.b-E1_repeatable { background:#e5eefa; border-color:#a8c4e4; color:#24558a; }
.b-E3_causal_scoped { background:#f7e8dd; border-color:#ddb28f; color:#8a4b16; }
.b-closed, .b-closed_v1 { background:#e3f0e3; border-color:#9fcaa0; color:#1e5c22; }
.b-open { background:#fbe9e7; border-color:#e0a89f; color:#93321f; }
.b-quantified_collapse { background:#fde8e8; border-color:#e8a0a0; color:#8c1f1f; }
.b-appendix_open { background:#eee8f7; border-color:#c5b1e6; color:#5b3a99; }
.b-appendix_open_crossline { background:#f1eee2; border-color:#d4c99a; color:#6d5f1e; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:10px; }
table { border-collapse:collapse; width:100%; background:var(--card); border:1px solid var(--line); }
th,td { border:1px solid var(--line); padding:6px 9px; text-align:left; vertical-align:top; font-size:13px; }
th { background:#eef1f5; font-weight:600; }
.kv { margin:3px 0; }
.kv b { color:var(--muted); font-weight:600; font-size:12px; margin-right:6px; }
.anchorbox { background:#f4f6f9; border:1px dashed var(--line); border-radius:6px; padding:6px 9px;
  margin-top:8px; font-size:12px; color:#3c4550; word-break:break-all; }
.asserts { color:#52616f; }
.evid { color:#52616f; font-size:12.5px; margin:3px 0; }
.prereg { border-left:3px solid #b35900; padding-left:10px; margin-top:8px; }
.mech { border-left:3px solid #8a4fb8; padding-left:10px; margin-top:8px; white-space:pre-wrap; }
.footer { margin-top:40px; padding-top:12px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; }
"""


def esc(s):
    return html.escape(str(s), quote=False)


CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')


def span(key, val):
    return '<span class="fv" data-k="%s">%s</span>' % (key, esc(val))


def render_feature(f):
    p = f['id']
    rows = []
    rows.append('<div class="card" id="%s">' % p)
    rows.append('<h3>%s &middot; %s <span class="badge b-%s">%s</span>'
                '<span class="badge">%s</span></h3>' % (
                    p, span(p + '.family', f['family']), f['evidence_level'],
                    span(p + '.evidence_level', f['evidence_level']),
                    esc(', '.join(f['model_scope']))))
    rows.append('<div class="kv"><b>statement</b>%s</div>' % span(p + '.statement', f['statement']))
    rows.append('<div class="kv"><b>model_scope</b>%s</div>' % span(p + '.model_scope', ', '.join(f['model_scope'])))
    if f.get('values'):
        vk = ' '.join('<span class="mono">%s=%s</span>' % (esc(k), span('%s.values.%s' % (p, k), fmt(v)))
                      for k, v in f['values'].items())
        rows.append('<div class="kv"><b>values</b>%s</div>' % vk)
    for i, a in enumerate(f['anchors']):
        rows.append('<div class="anchorbox"><b>anchor[%d]</b> src=%s phase=%s role=%s'
                    '<div class="asserts">%s</div></div>' % (
                        i, span('%s.anchors.%d.src' % (p, i), a['src']),
                        span('%s.anchors.%d.phase' % (p, i), a['phase']),
                        span('%s.anchors.%d.role' % (p, i), a['role']),
                        span('%s.anchors.%d.asserts' % (p, i),
                             json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True))))
    rows.append('<div class="kv"><b>n_anchors</b>%s</div>' % span(p + '.n_anchors', str(len(f['anchors']))))
    rows.append('<div class="kv"><b>counter_evidence</b>%s</div>' % span(
        p + '.counter_evidence', ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'))
    rows.append('<div class="kv"><b>replication</b>%s</div>' % span(p + '.replication', f['replication']))
    rows.append('<div class="kv"><b>scope_limits</b>%s</div>' % span(
        p + '.scope_limits', f['scope_limits'] if f['scope_limits'] else '(none)'))
    rows.append('</div>')
    return '\n'.join(rows)


def render_html(reg, nodes, gl, meta_sha):
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.6 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.6 &mdash; 语义机制图谱（registry v1.5 渲染）</h1>')
    H.append('<div class="sub">Phase 3178 G5-B3 &middot; FTR-14 A2 归因分解（sampling-null 标定） &middot; '
             'registry v1.5 sha8=%s &middot; registry v1.4 sha8=%s &middot; registry v1.3 sha8=%s'
             ' &middot; registry v1.2 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v15_sha8', meta_sha['reg15']),
                 span('meta.registry_v14_sha8', SHA_ANCHOR['registry_v14']),
                 span('meta.registry_v13_sha8', 'e64733f7'),
                 span('meta.registry_v12_sha8', '0e5abcaf'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (esc(k), esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, esc(pr)))
    H.append('</div>')

    H.append('<h2>图谱节点基座（16 节点，3162 audit disk_verified）</h2>')
    H.append('<table><tr><th>id</th><th>title</th><th>evidence</th><th>checks</th></tr>')
    for n in nodes:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')

    fams = {}
    for f in reg['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.5，%d 条）</h2>' % len(reg['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(render_feature(f))
        H.append('</div>')

    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')

    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')

    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿：3171 机制定位 + 3172 干预 + 3175 残留定判）</h2>')
    for g in gl['gaps']:
        closed_at = str(g['closed_at_phase']) if g.get('closed_at_phase') is not None else '-'
        badge_closed = '<span class="badge">closed@%s</span>' % span(g['id'] + '.closed_at', closed_at) \
            if g['status'].startswith('closed') else ''
        H.append('<div class="card"><h3>%s &middot; %s <span class="badge b-%s">%s</span>%s</h3>' % (
            g['id'], span(g['id'] + '.title', g['title']), g['status'], span(g['id'] + '.status', g['status']), badge_closed))
        H.append('<div class="kv"><b>statement</b>%s</div>' % span(g['id'] + '.statement', g['statement']))
        H.append('<div class="evid">evidence:</div>')
        H.append('<div class="kv">%s</div>' % span(g['id'] + '.evidence', ' || '.join(g['evidence'])))
        H.append('<div class="anchorbox"><b>anchors</b> %s</div>' % span(
            g['id'] + '.anchors', ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))))
        if g['id'] == 'GAP-4':
            pr = g['prereg']
            H.append('<div class="prereg"><div class="kv"><b>prereg</b>Phase %s %s（%s）</div>' % (
                span('GAP-4.prereg.phase', pr['phase']), span('GAP-4.prereg.name', pr['name']),
                span('GAP-4.prereg.status', pr['status'])))
            H.append('<div class="kv"><b>hypothesis</b>%s</div>' % span('GAP-4.prereg.hypothesis', pr['hypothesis']))
            H.append('<div class="kv"><b>protocol</b>%s</div>' % span('GAP-4.prereg.protocol', pr['protocol']))
            H.append('<div class="kv"><b>gate</b>%s</div></div>' % span('GAP-4.prereg.gate', pr['gate']))
            H.append('<div class="mech"><div class="kv"><b>mechanism_note</b>%s</div></div>' % span(
                'GAP-4.mechanism_note', g['mechanism_note']))
        H.append('</div>')

    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl['appendix']))
    for a in gl['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')

    H.append('<div class="footer">RDC/LPF atlas v1.6 &middot; 渲染自 atlas_registry_v1_5.json (%s)'
             ' + gap ledger v1.5 (%s，3175 定稿原样渲染) &middot; 增量来源: Phase 3178 G5-B3 '
             'FTR-14 A2 归因分解（p3157 collect %s / %s / %s；3177 封存 result %s）&middot; '
             '本文件为 Phase 3178 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 meta_sha['reg15'], SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3177_result']))
    H.append('</body></html>')
    return '\n'.join(H)


def flatten(reg, nodes, gl):
    d = {}
    for f in reg['features']:
        p = f['id']
        d[p + '.statement'] = f['statement']
        d[p + '.family'] = f['family']
        d[p + '.evidence_level'] = f['evidence_level']
        d[p + '.model_scope'] = ', '.join(f['model_scope'])
        d[p + '.n_anchors'] = str(len(f['anchors']))
        d[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
        d[p + '.replication'] = f['replication']
        d[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
        for i, a in enumerate(f['anchors']):
            d['%s.anchors.%d.src' % (p, i)] = str(a['src'])
            d['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
            d['%s.anchors.%d.role' % (p, i)] = a['role']
            d['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
        for k, v in (f.get('values') or {}).items():
            d['%s.values.%s' % (p, k)] = fmt(v)
    for n in nodes:
        d[n['id'] + '.title'] = n['title']
        d[n['id'] + '.elevel'] = n['evidence_level']
        d[n['id'] + '.status'] = n['status']
        d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
    for i, u in enumerate(reg['upgrade_log']):
        d['UPG.%d.node' % i] = u['node']
        d['UPG.%d.change' % i] = u['change']
        d['UPG.%d.reason' % i] = u['reason']
    for fl in reg['failures']:
        d[fl['id'] + '.kind'] = fl['kind']
        d[fl['id'] + '.phase'] = str(fl['phase'])
        d[fl['id'] + '.text'] = fl['text']
        d[fl['id'] + '.evidence'] = str(fl['evidence'])
    for g in gl['gaps']:
        p = g['id']
        d[p + '.title'] = g['title']
        d[p + '.status'] = g['status']
        if g['status'].startswith('closed'):
            d[p + '.closed_at'] = str(g['closed_at_phase'])
        d[p + '.statement'] = g['statement']
        d[p + '.evidence'] = ' || '.join(g['evidence'])
        d[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
        if p == 'GAP-4':
            pr = g['prereg']
            d[p + '.prereg.phase'] = str(pr['phase'])
            d[p + '.prereg.name'] = pr['name']
            d[p + '.prereg.hypothesis'] = pr['hypothesis']
            d[p + '.prereg.protocol'] = pr['protocol']
            d[p + '.prereg.gate'] = pr['gate']
            d[p + '.prereg.status'] = pr['status']
            d[p + '.mechanism_note'] = g['mechanism_note']
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    d['meta.registry_v15_sha8'] = meta_sha_global['reg15']
    d['meta.registry_v14_sha8'] = SHA_ANCHOR['registry_v14']
    d['meta.registry_v13_sha8'] = 'e64733f7'
    d['meta.registry_v12_sha8'] = '0e5abcaf'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.created'] = CREATED_STR
    return d


def verify_html(html_text, expected):
    pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', html_text, re.S)
    found = {}
    dups = []
    for k, v in pairs:
        if k in found:
            dups.append(k)
        found[k] = html.unescape(v)
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    mismatch = []
    for k in sorted(set(expected) & set(found)):
        if found[k] != expected[k]:
            mismatch.append((k, found[k][:80], expected[k][:80]))
    return {'n_expected': len(expected), 'n_found': len(found), 'dups': dups,
            'missing': missing, 'extra': extra, 'mismatch': mismatch}


# ---------------------------------------------------------------- main
def main():
    d8 = freeze()
    log('== Phase 3178 g5b3_ftr14_a2_attribution %s ==' % ('SMOKE' if SMOKE else 'FULL'))

    for k, p in SRC.items():
        h = sha8_file(p)
        assert h == SHA_ANCHOR[k], ('source sha mismatch', k, h, SHA_ANCHOR[k])
    log('sources: %d files sha-asserted' % len(SRC))

    # double-lock 3157 anchors: runtime result.json vs frozen DESIGN literals
    for m in MODELS:
        ak = ANCH_KEY[m]
        r57 = json.load(io.open(SRC[RES_KEY[m]], encoding='utf-8'))
        A = ANCH_3157[ak]
        assert r57['res_sha8'] == A['res_sha8'] and r57['seal_sha8'] == A['seal_sha8'], \
            ('3157 content sha', m)
        assert r57['tc_mean_cos'] == A['tc_mean_cos'], ('3157 tc', m)
        assert r57['exch']['exchR_c0'] == A['exchR_c0'], ('3157 exchR', m)
        assert r57['exch']['exchN_c0'] == A['exchN_c0'], ('3157 exchN', m)
        assert r57['gates']['exchange_obs'] == A['exchange_obs'], ('3157 obs', m)
        assert r57['gates']['comm_class'] == 'commutative_partial'
    s57 = json.load(io.open(SRC['p3157_summary'], encoding='utf-8'))
    assert s57['res_sha8'] == SUMMARY_3157['res_sha8']
    assert s57['seal_sha8'] == SUMMARY_3157['seal_sha8']
    assert s57['exch_mean_obs'] == SUMMARY_3157['exch_mean_obs']
    assert s57['fpmin_kout'] == SUMMARY_3157['fpmin_kout']
    assert s57['comm_class'] == SUMMARY_3157['comm_class']
    log('3157 anchors double-locked (3 per-model results + summary)')

    # double-lock 3177 sealed result (content sha) and load sealed numbers
    r17 = json.load(io.open(SRC['p3177_result'], encoding='utf-8'))
    assert r17['res_sha8'] == ANCH_3177['res_sha8'], ('3177 res sha', )
    assert r17['seal_sha8'] == ANCH_3177['seal_sha8'], ('3177 seal sha', )
    assert r17['arm']['gate'] is False and r17['arm']['gate_a2'] is False \
        and r17['arm']['gate_a1'] is True and r17['arm']['gate_exch_band'] is True, \
        ('3177 gate state unexpected', )
    for m in MODELS:
        assert m in r17['arm']['per_model'], ('3177 model missing', m)
        f = r17['arm']['per_model'][m]['folds']
        for rel in ('isa', 'hasa'):
            assert 'cos_train_a2' in f[rel] and 'cos_held_a2' in f[rel], ('3177 fold keys', m, rel)
    log('3177 sealed result double-locked (res %s / seal %s); numbers loaded from pinned file'
        % (ANCH_3177['res_sha8'], ANCH_3177['seal_sha8']))

    out, calib, exch_mean = run_arm(r17)

    suffix = 'smoke_' if SMOKE else ''
    reg15_path = None
    html_path = None
    html_sha = None
    field_check = None
    if calib:
        verdict = ('relation_independence_supported_under_calibration_bar '
                   '(r(m) < %.2f x3; FTR-14 upgrade execution deferred to 3179 '
                   'per prereg; registry/html untouched)' % R_TOL)
        log('PASS branch: registry v1.4 and atlas_v1_5.html stay untouched')
    else:
        verdict = ('relation_specific_residual_confirmed '
                   '(r(m) = %s >= %.2f; quantified registration: registry '
                   'v1.4 -> v1.5 FTR-14 counter_evidence append, stays E1)'
                   % ('/'.join('%.4f' % out[m]['r_m'] for m in MODELS), R_TOL))
        reg15, nodes, gl15 = build_registry_fail(out, exch_mean)
        reg15_path = os.path.join(OUTDIR, suffix + 'atlas_registry_v1_5.json')
        with io.open(reg15_path, 'w', encoding='utf-8') as f:
            json.dump(reg15, f, ensure_ascii=False, indent=1)
        meta_sha_global['reg15'] = sha8_file(reg15_path)
        log('registry v1.5 written sha8 %s' % meta_sha_global['reg15'])
        expected = flatten(reg15, nodes, gl15)
        html_text = render_html(reg15, nodes, gl15, meta_sha_global)
        html_path = os.path.join(OUTDIR, suffix + 'atlas_v1_6.html')
        with io.open(html_path, 'w', encoding='utf-8', newline='') as f:
            f.write(html_text)
        html_sha = sha8_file(html_path)
        v = verify_html(open(html_path, 'rb').read().decode('utf-8'), expected)
        assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups'], \
            ('html field check', v['missing'][:5], v['extra'][:5], v['mismatch'][:3], v['dups'][:5])
        field_check = {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                       'missing': len(v['missing']), 'extra': len(v['extra']),
                       'mismatch': len(v['mismatch']), 'dups': len(v['dups'])}
        log('atlas_v1_6.html written sha8 %s (fields %d/%d, 0/0/0/0)'
            % (html_sha, v['n_found'], v['n_expected']))

    summary = {
        'phase': 3178, 'name': 'g5b3_ftr14_a2_attribution',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'zero_gpu': True,
        'design_sha8': d8,
        'prereg': DESIGN['prereg'],
        'gates': {'device_3157': True, 'replication_3177': True,
                  'calibration_bar': bool(calib), 'r_tol': R_TOL},
        'arm': {'verdict': verdict, 'calibration_pass': bool(calib),
                'exch_mean': exch_mean, 'per_model': out},
        'registry_disposition': ('untouched_on_pass_upgrade_deferred_to_3179' if calib
                                 else 'v1.4_to_v1.5_ftr14_counter_evidence_append'),
        'registry_v15_file': os.path.basename(reg15_path) if reg15_path else None,
        'registry_v15_sha8': meta_sha_global['reg15'],
        'html_file': os.path.basename(html_path) if html_path else None,
        'html_sha8': html_sha,
        'field_check': field_check,
        'counts': {'features': 22, 'failures': 16, 'upgrades': 5,
                   'gaps': 4, 'appendix': 5},
        'sources': {k: SHA_ANCHOR[k] for k in SRC},
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rpath = os.path.join(OUTDIR, suffix + 'result.json')
    with io.open(rpath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    log('verdict: %s' % verdict)

    with io.open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
