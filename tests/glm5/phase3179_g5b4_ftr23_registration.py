# -*- coding: utf-8 -*-
"""Phase 3179: G5-B4 FTR-23 registration (zero GPU).

Per prereg in MEMO Phase 3178 tail, section "接续：预注册 3179" (frozen before
any observation): the T_C relation-specific residual currently lives only
inside FTR-14 counter_evidence strings; it satisfies the registration bar
(3177+3178 two-phase independent measurement, 3-model same direction, complete
prereg gate history F16 gate fail -> 3178 attribution quantification) =>
promote to standalone feature FTR-23 (E1_repeatable), registry v1.5 -> v1.6 +
atlas_v1_7.html re-render (3173 discipline).

Freeze-time protocol pinning:
  - Zero-GPU registration phase: NO new scientific observation.  All numbers
    rendered into FTR-23 are loaded at runtime from the sealed p3178/p3177
    result.json (no float transcription).  Device gates below only re-derive
    the sealed numbers from the sealed 3157 collect npz (repeatable proof).
  - No-fail-branch phase: gate failure = device/environment defect (repair &
    rerun), NOT a scientific verdict.  The scientific verdict (relation-
    specific residual confirmed) is already sealed in p3178.
  - Sources (file sha8 pinned): p3178 result abae296e (content 4f521d9b /
    1f166645), p3177 result b2ad52ef (d451030f / 7b746eb2), registry v1.5
    e1942d03, gap v1.5 4fb3f37d, registry 3162 00f15e98, p3157 collect npz x3
    95a25965/9552086d/23dd74eb + per-model results ca863153/e797adf9/4030081f
    + summary 71959b6a.
  - Repeatable recompute gate (verbatim 3178 run_arm protocol): device exch/
    tc_full drift < 1e-9 vs 3157 anchors; 3177 fold/cos_full drift < 1e-6;
    3178 null p(m,rel) and r(m,rel)/r(m) recompute drift < 1e-6; exch mean
    drift < 1e-9 vs 3157 summary.
  - Statement rendered at runtime from sealed numbers; range asserts vs the
    MEMO draft (round-3 boundaries, tol 5e-4): p in [0.019, 0.026], per-fold
    r in [0.072, 0.131], r(m) in [0.092, 0.129], tc_held in [0.581, 0.623],
    tc_full in [0.561, 0.564]; r(m) >= 0.05 x3 (calibration FAIL x3 is the
    registration fact, re-asserted).
  - Registry upgrade: v1.5 -> v1.6 by features APPEND FTR-23 ONLY; the 22
    existing features byte-for-byte preserved (sort_keys dump equality);
    failures 16 / upgrade_log 5 / lineage metadata (created/phase/provenance/
    schema/...) unchanged; supersedes updated.  FTR-23: family context (same
    family as FTR-14), evidence_level E1_repeatable, model_scope 3 models,
    anchors x3 (p3178 primary_attribution / p3177 directional_fold_evidence /
    p3157 device_base), values 6 keys (p_m/r_m per model), counter_evidence 3
    honest-limitation entries (all numbers runtime-loaded).
  - HTML: atlas_v1_7.html full re-render with per-field data-k verification;
    meta chain gains registry_v16_sha8 (written before render, sha8 of the
    v1.6 file); meta.created whitelist as before.
  - SMOKE (env P3179_SMOKE=1): zero-GPU => numbers identical across modes;
    smoke_-prefixed artifacts; registry v1.6 byte-identity SMOKE vs FULL
    asserted at closeout.

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
OUTDIR = os.path.join(SRC_DIR, 'phase3179', 'g5b4_ftr23_registration')
SMOKE = os.environ.get('P3179_SMOKE', '') == '1'
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
    'p3178_result': os.path.join(SRC_DIR, r'phase3178\g5b3_ftr14_a2_attribution\result.json'),
    'registry_v15': os.path.join(SRC_DIR, r'phase3178\g5b3_ftr14_a2_attribution\atlas_registry_v1_5.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a',
    'p3177_result': 'b2ad52ef', 'p3178_result': 'abae296e',
    'registry_v15': 'e1942d03', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}

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
ANCH_3177 = dict(res_sha8='d451030f', seal_sha8='7b746eb2')
ANCH_3178 = dict(res_sha8='4f521d9b', seal_sha8='1f166645')

DEV_TOL = 1e-9
REPL_TOL = 1e-6
R_TOL = 0.05
# MEMO draft round-3 range boundaries (tol covers rounding)
RANGE = dict(p=(0.019, 0.026), r_fold=(0.072, 0.131), r_m=(0.092, 0.129),
             tc_held=(0.581, 0.623), tc_full=(0.561, 0.564))
RANGE_TOL = 5e-4

DESIGN = {
    'phase': 3179, 'name': 'g5b4_ftr23_registration', 'zero_gpu': True,
    'prereg': 'MEMO 2026-10-10 Phase 3178 tail (G5-B4): T_C relation-specific '
              'residual promoted from FTR-14 counter_evidence strings to '
              'standalone feature FTR-23 (E1_repeatable, family context); '
              'registration bar = 3177+3178 two-phase independent measurement '
              '+ 3-model same direction + complete prereg gate history (F16 '
              'gate fail -> 3178 attribution quantification); registry v1.5 '
              '-> v1.6 features-append-only + atlas_v1_7.html re-render',
    'freeze_pinning': dict(
        phase_kind='no-fail-branch registration phase: gate failure = device/'
                   'environment defect (repair & rerun), NOT a scientific '
                   'verdict; the scientific verdict (relation-specific '
                   'residual confirmed) is sealed in p3178 already',
        sources='p3178 result abae296e (content 4f521d9b/1f166645), p3177 '
                'result b2ad52ef (d451030f/7b746eb2), registry v1.5 e1942d03, '
                'gap v1.5 4fb3f37d, registry 3162 00f15e98, p3157 collect x3 '
                '95a25965/9552086d/23dd74eb + results ca863153/e797adf9/'
                '4030081f + summary 71959b6a',
        repeatable_gate='verbatim 3178 run_arm recompute from sealed 3157 '
                        'collect npz x3: device exch/tc_full drift < 1e-9 vs '
                        '3157 anchors; 3177 folds/cos_full drift < 1e-6; 3178 '
                        'null p(m,rel) and r(m,rel)/r(m) recompute drift '
                        '< 1e-6; exch mean drift < 1e-9 vs summary',
        statement='rendered at runtime from sealed p3178/p3177 numbers (no '
                  'float transcription); range asserts vs MEMO draft round-3 '
                  'boundaries (tol 5e-4): p [0.019,0.026], r_fold [0.072,'
                  '0.131], r_m [0.092,0.129], tc_held [0.581,0.623], tc_full '
                  '[0.561,0.564]; r(m) >= 0.05 x3 re-asserted (registration '
                  'fact)',
        registry_upgrade='v1.5 -> v1.6 features APPEND FTR-23 ONLY; 22 '
                         'existing features byte-for-byte (sort_keys dump); '
                         'failures 16 / upgrade_log 5 / lineage metadata '
                         'unchanged; supersedes updated; FTR-23 = family '
                         'context, E1_repeatable, 3 models, anchors x3 '
                         '(p3178/p3177/p3157, asserts runtime-built), values '
                         '6 keys (p_m/r_m), counter_evidence 3 honest '
                         'limitations (numbers runtime-loaded)',
        html='atlas_v1_7.html full re-render, per-field data-k verification '
             '(3173 discipline); meta chain gains registry_v16_sha8; '
             'meta.created whitelist',
        smoke='zero-GPU: computation identical across modes; smoke_-prefixed '
              'artifacts; registry v1.6 byte-identity SMOKE vs FULL asserted '
              'at closeout'),
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'anchors': {'3157': {'per_model': ANCH_3157, 'summary': SUMMARY_3157},
                '3177': ANCH_3177, '3178': ANCH_3178},
    'gates': dict(device_drift=DEV_TOL, recompute_drift=REPL_TOL, r_tol=R_TOL,
                  range_tol=RANGE_TOL, draft_ranges=RANGE),
}

LOG = []
meta_sha_global = {'reg16': None}


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


# ------------------------------------------------- verbatim 3178 recipes
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


def recompute_arm(sealed_3177, sealed_3178):
    """Verbatim 3178 run_arm recompute (repeatable gate).  Asserts device
    drifts vs 3157 anchors and recompute drifts vs sealed 3177/3178 numbers.
    Returns per-model summary dict (no new observation)."""
    out = {}
    for m in MODELS:
        ak = ANCH_KEY[m]
        A = ANCH_3157[ak]
        R17 = sealed_3177['arm']['per_model'][m]
        R18 = sealed_3178['arm']['per_model'][m]
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

        # device gate: exch recompute verbatim
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
        assert abs(exchange_obs - R17['exchange_obs']) < REPL_TOL, ('3177 exch', m)

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
        assert abs(tc_full - R17['tc_full']) < REPL_TOL, ('3177 tc_full', m)

        # A2 full reading - verbatim 3177
        v_full = unit_mean(dCn)
        cos_full_a2 = float((dCn @ v_full).mean())
        assert abs(cos_full_a2 - R17['cos_full_a2']) < REPL_TOL, ('3177 cos_full', m)

        # folds: held-out relation - verbatim 3177
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
            # A1 metric: within-held-relation pairwise cos (FTR-23 directional evidence)
            Mh = dCn_h @ dCn_h.T
            tc_held_a1 = float(Mh[np.triu_indices(len(rows_h), 1)].mean())
            E17 = R17['folds'][fname]
            d_tr = abs(cos_train_a2 - E17['cos_train_a2'])
            d_he = abs(cos_held_a2 - E17['cos_held_a2'])
            d_a1 = abs(tc_held_a1 - E17['tc_held_a1'])
            assert max(d_tr, d_he, d_a1) < REPL_TOL, \
                ('3177 fold drift', m, fname, d_tr, d_he, d_a1)
            folds[fname] = dict(cos_held_a2=cos_held_a2, cos_train_a2=cos_train_a2,
                                tc_held_a1=tc_held_a1,
                                repl_drift=max(d_tr, d_he, d_a1))

        # entity-LOO null within relation - verbatim 3178
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
                pf.append(cos_in - cos_out)
            assert all(np.isfinite(pf)) and max(abs(x) for x in pf) < 0.5, \
                ('null sanity', m, rname)
            p_mean = float(np.mean(pf))
            E18 = R18['null'][rname]['p_mean']
            d_p = abs(p_mean - E18)
            assert d_p < REPL_TOL, ('3178 p_mean drift', m, rname, d_p)
            nulls[rname] = dict(p_mean=p_mean, drift=d_p)

        # residual: g (3177 same-v_shared gap) - p (null calibration)
        resid = {}
        for rname in ('isa', 'hasa'):
            g = folds[rname]['cos_train_a2'] - folds[rname]['cos_held_a2']
            p_cal = nulls[rname]['p_mean']
            r = g - p_cal
            E18 = R18['resid'][rname]
            assert abs(g - E18['gap']) < REPL_TOL, ('3178 gap drift', m, rname)
            assert abs(r - E18['r']) < REPL_TOL, ('3178 r drift', m, rname)
            resid[rname] = dict(gap=g, p=p_cal, r=r)
        r_m = float((resid['isa']['r'] + resid['hasa']['r']) / 2.0)
        assert abs(r_m - R18['r_m']) < REPL_TOL, ('3178 r_m drift', m)

        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full,
                      tc_full_drift=drift_tc, exch_drift=drift_ex,
                      exchange_obs=exchange_obs, cos_full_a2=cos_full_a2,
                      folds=folds, null=nulls, resid=resid, r_m=r_m)
        log('recompute %s: tc_full drift %.2e, exch drift %.2e, '
            'p_mean drift %.2e/%.2e (vs sealed 3178)' % (
                m, drift_tc, drift_ex, nulls['isa']['drift'], nulls['hasa']['drift']))
    exch_mean = float(np.mean([out[m]['exchange_obs'] for m in MODELS]))
    drift_mean = abs(exch_mean - SUMMARY_3157['exch_mean_obs'])
    assert drift_mean < DEV_TOL, ('exch mean drift', drift_mean)
    log('repeatable gate: device %.0e + recompute %.0e all green (3 models)'
        % (DEV_TOL, REPL_TOL))
    return out, exch_mean


# ------------------------------------------------- FTR-23 construction
def build_ftr23(sealed_3177, sealed_3178, exch_mean):
    """All numbers runtime-loaded from sealed results (no transcription)."""
    A8 = sealed_3178['arm']
    A7 = sealed_3177['arm']
    assert A8['calibration_pass'] is False, ('3178 calibration state', )
    assert A7['gate'] is False and A7['gate_a1'] is True and \
        A7['gate_a2'] is False and A7['gate_exch_band'] is True, ('3177 gate state', )

    r_m = {m: A8['per_model'][m]['r_m'] for m in MODELS}
    p_m = {m: float(np.mean([A8['per_model'][m]['resid'][rel]['p']
                             for rel in ('isa', 'hasa')])) for m in MODELS}
    p_rel = {(m, rel): A8['per_model'][m]['resid'][rel]['p']
             for m in MODELS for rel in ('isa', 'hasa')}
    r_rel = {(m, rel): A8['per_model'][m]['resid'][rel]['r']
             for m in MODELS for rel in ('isa', 'hasa')}
    tc_full = {m: A7['per_model'][m]['tc_full'] for m in MODELS}
    # A1 metric (within-held-relation pairwise cos) = the MEMO draft "tc_held"
    tc_held = {(m, rel): A7['per_model'][m]['folds'][rel]['tc_held_a1']
               for m in MODELS for rel in ('isa', 'hasa')}
    tc_train = {(m, rel): A7['per_model'][m]['folds'][rel]['cos_train_a2']
                for m in MODELS for rel in ('isa', 'hasa')}
    exch_obs = {m: A7['per_model'][m]['exchange_obs'] for m in MODELS}

    # range asserts vs MEMO draft (round 3, tol 5e-4)
    lo, hi = RANGE['p']
    assert lo - RANGE_TOL <= min(p_rel.values()) and max(p_rel.values()) <= hi + RANGE_TOL, \
        ('p range', min(p_rel.values()), max(p_rel.values()))
    lo, hi = RANGE['r_fold']
    assert lo - RANGE_TOL <= min(r_rel.values()) and max(r_rel.values()) <= hi + RANGE_TOL, \
        ('r_fold range', min(r_rel.values()), max(r_rel.values()))
    lo, hi = RANGE['r_m']
    assert lo - RANGE_TOL <= min(r_m.values()) and max(r_m.values()) <= hi + RANGE_TOL, \
        ('r_m range', min(r_m.values()), max(r_m.values()))
    lo, hi = RANGE['tc_held']
    assert lo - RANGE_TOL <= min(tc_held.values()) and max(tc_held.values()) <= hi + RANGE_TOL, \
        ('tc_held range', min(tc_held.values()), max(tc_held.values()))
    lo, hi = RANGE['tc_full']
    assert lo - RANGE_TOL <= min(tc_full.values()) and max(tc_full.values()) <= hi + RANGE_TOL, \
        ('tc_full range', min(tc_full.values()), max(tc_full.values()))
    assert all(r_m[m] >= R_TOL for m in MODELS), ('registration fact r>=0.05 x3', r_m)
    assert all(tc_held[k] > tc_full[k[0]] for k in tc_held), \
        ('direction tc_held > tc_full 6/6', )
    log('range asserts vs MEMO draft: all green (p/r_fold/r_m/tc_held/tc_full)')

    stmt = ('T_C 共享分量携带关系特异残差（3178 sampling-null 归因定判 + 3177 关系留出'
            '方向性发现）：实体留出 null 标定下纯采样惩罚 p %.3f-%.3f，关系残差 r %.3f-%.3f '
            '×3 模型（标定 bar %.2f，r(m) = %s 全过 bar 判关系无关失败）；关系内两两 cos '
            'tc_held %.3f-%.3f > 全量 tc_full %.3f-%.3f（6/6 fold×模型单元）——T_C 几何'
            '按关系分解，关系内自洽、跨关系系统性偏移；关系残差由 3177 F16 门失败归因定量'
            '闭合（F16 = gate fail 记录，本特征 = 残差本体的正面登记）' % (
                min(p_rel.values()), max(p_rel.values()),
                min(r_rel.values()), max(r_rel.values()), R_TOL,
                '/'.join('%.4f' % r_m[m] for m in MODELS),
                min(tc_held.values()), max(tc_held.values()),
                min(tc_full.values()), max(tc_full.values())))

    anchor_3178 = {
        'src': 'p3178', 'phase': 3178, 'role': 'primary_attribution',
        'tags': ['cross_model', 'held_out'],
        'asserts': dict(
            **{('r_m.%s' % m): r_m[m] for m in MODELS},
            **{('p_m.%s' % m): p_m[m] for m in MODELS},
            exch_mean=A8['exch_mean'],
            calibration_pass=False,
            res_sha8=ANCH_3178['res_sha8'], seal_sha8=ANCH_3178['seal_sha8'])}
    anchor_3177 = {
        'src': 'p3177', 'phase': 3177, 'role': 'directional_fold_evidence',
        'tags': ['cross_model', 'held_out'],
        'asserts': dict(
            **{('tc_full.%s' % m): tc_full[m] for m in MODELS},
            **{('tc_held_a1.%s.isa' % m): tc_held[(m, 'isa')] for m in MODELS},
            **{('tc_held_a1.%s.hasa' % m): tc_held[(m, 'hasa')] for m in MODELS},
            **{('exchange_obs.%s' % m): exch_obs[m] for m in MODELS},
            gate_a1=True, gate_a2=False,
            res_sha8=ANCH_3177['res_sha8'], seal_sha8=ANCH_3177['seal_sha8'])}
    anchor_3157 = {
        'src': 'p3157', 'phase': 3157, 'role': 'device_base_measurement',
        'tags': ['device', 'cross_model'],
        'asserts': dict(
            **{('tc_mean_cos.%s' % ANCH_KEY[m]): ANCH_3157[ANCH_KEY[m]]['tc_mean_cos']
               for m in MODELS},
            fpmin_kout=SUMMARY_3157['fpmin_kout'],
            exch_mean_obs=SUMMARY_3157['exch_mean_obs'],
            comm_class=SUMMARY_3157['comm_class'],
            res_sha8=SUMMARY_3157['res_sha8'], seal_sha8=SUMMARY_3157['seal_sha8'])}

    ce = [
        'r 幅度按关系不同：14b hasa %.4f vs isa %.4f（同模型内近 2 倍差）⇒ 残差非关系间'
        '均匀，r(m) 聚合口径掩盖关系间幅度差' % (r_rel[('qwen3-14b', 'hasa')],
                                                r_rel[('qwen3-14b', 'isa')]),
        'null p fold 级存在负值（3178 封存 min %.4f）⇒ 纯采样惩罚在该几何中可 ≈0 甚至为负'
        '（单实体方向自洽），标定 bar 的解释以 p(m,rel) 聚合均值为准' % (
            min(min(A8['per_model'][m]['null'][rel]['folds'],
                    key=lambda x: x['p_fold'])['p_fold']
                for m in MODELS for rel in ('isa', 'hasa')),),
        'held_out 升级未定：对关系特异特征「held_out 关系」本身即测量构造，E2 升级设计'
        '挂账（3177 F16 门的 0.1 bar 对重建型读数不可达根源即本特征的关系几何）',
    ]

    ftr23 = {
        'id': 'FTR-23', 'family': 'context', 'statement': stmt,
        'evidence_level': 'E1_repeatable',
        'model_scope': list(MODELS),
        'anchors': [anchor_3178, anchor_3177, anchor_3157],
        'counter_evidence': ce,
        'replication': 'zero GPU：3178 sampling-null r_m/p_m + 3177 relation-LOO '
                       'tc_held/tc_full 封存数字运行时加载；3179 SMOKE/FULL null '
                       '复算 drift<1e-6 ×3 模型 + registry 字节恒等',
        'scope_limits': 'T_C 几何限于 3157 面板（2 关系 isa/hasa ×16 实体 ×2 极性 ×2 语境）；'
                        'r 幅度关系间不同；对其他关系族泛化未测；标定 bar 0.05 = 3178 '
                        'freeze 钉定聚合口径',
        'values': dict(
            **{('r_m.%s' % m): r_m[m] for m in MODELS},
            **{('p_m.%s' % m): p_m[m] for m in MODELS}),
    }
    log('FTR-23 built: statement %d chars, %d anchors, %d values, %d counter'
        % (len(stmt), len(ftr23['anchors']), len(ftr23['values']), len(ce)))
    return ftr23


# ------------------------------------------------- registry v1.6
def build_registry_v16(ftr23):
    reg15 = json.loads(open(SRC['registry_v15'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    reg2 = json.load(io.open(SRC['registry_3162'], encoding='utf-8'))
    nodes = reg2['audit']['nodes']
    assert reg15['version'] == '1.5' and gl15['version'] == '1.5'
    assert len(reg15['features']) == 22 and len(nodes) == 16
    assert len(reg15['failures']) == 16 and len(reg15['upgrade_log']) == 5
    assert not [f for f in reg15['features'] if f['id'] == 'FTR-23'], 'FTR-23 exists'
    reg16 = copy.deepcopy(reg15)
    reg16['version'] = '1.6'
    reg16['supersedes'] = 'atlas_registry_v1_5.json (%s, Phase 3178)' % SHA_ANCHOR['registry_v15']
    reg16['features'] = reg15['features'] + [ftr23]
    assert len(reg16['features']) == 23
    # byte-for-byte preservation of the 22 existing features
    for f in reg15['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg16['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        assert a == b, ('feature not preserved', f['id'])
    for k in ('failures', 'upgrade_log', 'evidence_level_taxonomy', 'principles',
              'schema', 'models_mainline', 'models_pending', 'provenance'):
        assert json.dumps(reg15[k], ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg16[k], ensure_ascii=False, sort_keys=True), (k, 'changed')
    for k in ('created', 'phase', 'name', 'design_sha8'):
        assert reg15[k] == reg16[k], (k, 'changed')
    f23 = reg16['features'][-1]
    assert f23['evidence_level'] == 'E1_repeatable' and f23['family'] == 'context'
    assert len(f23['anchors']) == 3 and f23['counter_evidence'] and f23['values']
    log('preservation: 22 features byte-for-byte; failures/upgrade_log/lineage '
        'untouched; FTR-23 appended (23 features)')
    return reg16, nodes, gl15


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
    H.append('<title>Atlas v1.7 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.7 &mdash; 语义机制图谱（registry v1.6 渲染）</h1>')
    H.append('<div class="sub">Phase 3179 G5-B4 &middot; FTR-23 立项入册（T_C 关系特异残差） &middot; '
             'registry v1.6 sha8=%s &middot; registry v1.5 sha8=%s &middot; registry v1.4 sha8=%s'
             ' &middot; registry v1.3 sha8=%s &middot; registry v1.2 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3178 %s + p3177 %s &middot; p3157 collect %s/%s/%s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v16_sha8', meta_sha['reg16']),
                 span('meta.registry_v15_sha8', SHA_ANCHOR['registry_v15']),
                 span('meta.registry_v14_sha8', 'fb11d633'),
                 span('meta.registry_v13_sha8', 'e64733f7'),
                 span('meta.registry_v12_sha8', '0e5abcaf'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3177_result', SHA_ANCHOR['p3177_result']),
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
    H.append('<h2>跨模型稳定特征登记表（registry v1.6，%d 条）</h2>' % len(reg['features']))
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

    H.append('<div class="footer">RDC/LPF atlas v1.7 &middot; 渲染自 atlas_registry_v1_6.json (%s)'
             ' + gap ledger v1.5 (%s，3175 定稿原样渲染) &middot; 增量来源: Phase 3179 G5-B4 '
             'FTR-23 立项入册（T_C 关系特异残差，数据源 p3178 %s + p3177 %s + p3157 collect '
             '%s / %s / %s）&middot; 本文件为 Phase 3179 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 meta_sha['reg16'], SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3178_result'], SHA_ANCHOR['p3177_result'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4']))
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
    d['meta.registry_v16_sha8'] = meta_sha_global['reg16']
    d['meta.registry_v15_sha8'] = SHA_ANCHOR['registry_v15']
    d['meta.registry_v14_sha8'] = 'fb11d633'
    d['meta.registry_v13_sha8'] = 'e64733f7'
    d['meta.registry_v12_sha8'] = '0e5abcaf'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.p3177_result'] = SHA_ANCHOR['p3177_result']
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
    log('== Phase 3179 g5b4_ftr23_registration %s ==' % ('SMOKE' if SMOKE else 'FULL'))

    for k, p in SRC.items():
        h = sha8_file(p)
        assert h == SHA_ANCHOR[k], ('source sha mismatch', k, h, SHA_ANCHOR[k])
    log('sources: %d files sha-asserted' % len(SRC))

    # double-lock sealed results (content sha) and load sealed numbers
    r18 = json.load(io.open(SRC['p3178_result'], encoding='utf-8'))
    assert r18['res_sha8'] == ANCH_3178['res_sha8'] and r18['seal_sha8'] == ANCH_3178['seal_sha8'], \
        ('3178 content sha', )
    assert r18['arm']['calibration_pass'] is False, ('3178 gate state', )
    assert r18['arm']['verdict'].startswith('relation_specific_residual_confirmed'), \
        ('3178 verdict', r18['arm']['verdict'][:60])
    r17 = json.load(io.open(SRC['p3177_result'], encoding='utf-8'))
    assert r17['res_sha8'] == ANCH_3177['res_sha8'] and r17['seal_sha8'] == ANCH_3177['seal_sha8'], \
        ('3177 content sha', )
    assert r17['arm']['gate'] is False and r17['arm']['gate_a1'] is True \
        and r17['arm']['gate_a2'] is False and r17['arm']['gate_exch_band'] is True, \
        ('3177 gate state', )
    # 3157 content double-lock
    for m in MODELS:
        ak = ANCH_KEY[m]
        r57 = json.load(io.open(SRC[RES_KEY[m]], encoding='utf-8'))
        A = ANCH_3157[ak]
        assert r57['res_sha8'] == A['res_sha8'] and r57['seal_sha8'] == A['seal_sha8'], \
            ('3157 content sha', m)
    s57 = json.load(io.open(SRC['p3157_summary'], encoding='utf-8'))
    assert s57['res_sha8'] == SUMMARY_3157['res_sha8'] and s57['seal_sha8'] == SUMMARY_3157['seal_sha8']
    assert s57['exch_mean_obs'] == SUMMARY_3157['exch_mean_obs']
    log('sealed content double-locked: p3178 %s/%s, p3177 %s/%s, p3157 x3 + summary'
        % (ANCH_3178['res_sha8'], ANCH_3178['seal_sha8'],
           ANCH_3177['res_sha8'], ANCH_3177['seal_sha8']))

    # repeatable recompute gate (verbatim 3178 protocol, 3 models)
    out, exch_mean = recompute_arm(r17, r18)

    # FTR-23 construction (numbers runtime-loaded from sealed results)
    ftr23 = build_ftr23(r17, r18, exch_mean)

    # registry v1.6
    reg16, nodes, gl15 = build_registry_v16(ftr23)
    suffix = 'smoke_' if SMOKE else ''
    reg16_path = os.path.join(OUTDIR, suffix + 'atlas_registry_v1_6.json')
    with io.open(reg16_path, 'w', encoding='utf-8') as f:
        json.dump(reg16, f, ensure_ascii=False, indent=1)
    meta_sha_global['reg16'] = sha8_file(reg16_path)
    log('registry v1.6 written sha8 %s' % meta_sha_global['reg16'])

    # html v1.7
    expected = flatten(reg16, nodes, gl15)
    html_text = render_html(reg16, nodes, gl15, meta_sha_global)
    html_path = os.path.join(OUTDIR, suffix + 'atlas_v1_7.html')
    with io.open(html_path, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(html_path)
    v = verify_html(open(html_path, 'rb').read().decode('utf-8'), expected)
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups'], \
        ('html field check', v['missing'][:5], v['extra'][:5], v['mismatch'][:3], v['dups'][:5])
    log('atlas_v1_7.html written sha8 %s (fields %d/%d, 0/0/0/0)'
        % (html_sha, v['n_found'], v['n_expected']))

    verdict = ('ftr23_registered_E1_repeatable (family context, 3 models, '
               'anchors p3178/p3177/p3157; repeatable gate drift %.0e/%.0e '
               'green; registration facts re-asserted: r(m) = %s >= %.2f x3, '
               'tc_held > tc_full 6/6)'
               % (DEV_TOL, REPL_TOL,
                  '/'.join('%.4f' % out[m]['r_m'] for m in MODELS), R_TOL))

    summary = {
        'phase': 3179, 'name': 'g5b4_ftr23_registration',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'zero_gpu': True,
        'design_sha8': d8,
        'prereg': DESIGN['prereg'],
        'gates': {'sources_sha': True, 'content_double_lock': True,
                  'repeatable_recompute': True, 'range_asserts': True,
                  'device_drift': DEV_TOL, 'recompute_drift': REPL_TOL,
                  'r_tol': R_TOL},
        'arm': {'verdict': verdict,
                'exch_mean': exch_mean,
                'per_model': {m: dict(kout=out[m]['kout'], nl=out[m]['nl'],
                                      tc_full=out[m]['tc_full'],
                                      tc_full_drift=out[m]['tc_full_drift'],
                                      exch_drift=out[m]['exch_drift'],
                                      exchange_obs=out[m]['exchange_obs'],
                                      cos_full_a2=out[m]['cos_full_a2'],
                                      r_m=out[m]['r_m'],
                                      p_isa=out[m]['resid']['isa']['p'],
                                      p_hasa=out[m]['resid']['hasa']['p'],
                                      r_isa=out[m]['resid']['isa']['r'],
                                      r_hasa=out[m]['resid']['hasa']['r'],
                                      g_isa=out[m]['resid']['isa']['gap'],
                                      g_hasa=out[m]['resid']['hasa']['gap'])
                              for m in MODELS}},
        'ftr23': {'id': 'FTR-23', 'family': 'context',
                  'evidence_level': 'E1_repeatable',
                  'n_anchors': 3, 'n_values': 6, 'n_counter': 3,
                  'statement': ftr23['statement']},
        'registry_disposition': 'v1.5_to_v1.6_features_append_ftr23_only',
        'registry_v16_file': os.path.basename(reg16_path),
        'registry_v16_sha8': meta_sha_global['reg16'],
        'html_file': os.path.basename(html_path),
        'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'counts': {'features': 23, 'failures': 16, 'upgrades': 5,
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
