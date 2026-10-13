# -*- coding: utf-8 -*-
"""Phase 3182 (G5-B7): FTR-23 held_out 关系留出臂（零 GPU，leave-one-relation-out cross-null）。

Per prereg in MEMO Phase 3181 tail, section "接续：预注册 3182" (frozen before
any observation):
  FTR-23 (T_C relation-specific residual, E1_repeatable, r(m)=0.0918/0.0951/
  0.1289 from sealed 3178) upgrades to E2_predictive if the residual replicates
  under a held-out relation construction.  Panel = sealed 3157 collect (2
  relations isa/hasa x 16 entities x 2 polarity, zero GPU).

Freeze-time protocol pinning:
  - r_full(m) recomputed with the 3178 verbatim recipe (same v_shared gap g
    from 3177 sealed folds + within-relation entity-LOO null p; drift < 1e-6
    vs sealed 3178 result.json, file sha8 abae296e pinned, content res/seal
    4f521d9b/1f166645 double-locked).  Sealed numbers loaded from the pinned
    file - no float transcription.
  - NEW cross-relation null (LORO; pinning note: prereg text said "null 标定用
    全部关系重建"; the non-leaky version is pinned here and the deviation is
    registered honestly in MEMO -- null calibrated from the OTHER relation
    only): for held relation rel_h in {isa, hasa}, calib = other relation:
      for fold e in 0..15:
        calib_rows = dC_rows(other) minus entity e's 2 cells there  (30 cells)
        v_x     = unit_mean(dCn[calib_rows])
        read    = entity e's 2 cells in rel_h
        p_x_fold(e) = mean(dCn[calib] @ v_x) - mean(dCn[read] @ v_x)
      p_x(m, rel_h) = mean over 16 folds; sanity 96/96 finite, |p_x_fold| < 0.5.
  - r_held(m, rel) = g(m, rel) - p_x(m, rel); r_held(m) = mean over rel.
  - GATE: |r_held(m) - r_full(m)| < 0.1 x3 models =>
    FTR-23 E1_repeatable -> E2_predictive: registry v1.8 -> v1.9 (held anchor
    append with tags [cross_model, held_out], upgrade_log #7; 22 other
    features byte-preserved), atlas_v1_10.html full re-render (p79 renderer,
    per-field data-k verification).
  - FAIL branch: F17 appended to failures, FTR-23 counter_evidence quantified
    append, stays E1_repeatable; atlas re-rendered likewise.
  - SMOKE (env P3182_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
"""
import copy
import datetime
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
import phase3179_g5b4_ftr23_registration as p79

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5b7_ftr23_heldout_loro'
OUTDIR = os.path.join(SRC_DIR, 'phase3182', NAME)
SMOKE = os.environ.get('P3182_SMOKE') == '1'
LOG = []
N_E = 16
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}


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
    'registry_v18': os.path.join(SRC_DIR, r'phase3181\g5b6_ftr04_glm4_attribution\atlas_registry_v1_8.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a', 'p3177_result': 'b2ad52ef', 'p3178_result': 'abae296e',
    'registry_v18': 'd08e9064', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_SHA_3177 = ('d451030f', '7b746eb2')
CONTENT_SHA_3178 = ('4f521d9b', '1f166645')

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, res_sha8='c0db3455', seal_sha8='41916ed8',
                     tc_mean_cos=0.5629908442497253, exchR_c0=1.0320992213108435,
                     exchN_c0=1.1928509676419494, exchange_obs=1.1928509676419494),
    'qwen3-14b': dict(nl=40, kout=39, res_sha8='1270c921', seal_sha8='98c06ab0',
                      tc_mean_cos=0.5614360570907593, exchR_c0=1.1211844589001665,
                      exchN_c0=1.294480635663979, exchange_obs=1.294480635663979),
    'glm4': dict(nl=40, kout=39, res_sha8='a1d9a98a', seal_sha8='e8be1026',
                 tc_mean_cos=0.5640953183174133, exchR_c0=1.0723271372900645,
                 exchN_c0=1.3311207542158712, exchange_obs=1.3311207542158712),
}
SUMMARY_3157 = dict(res_sha8='0fe043bf', seal_sha8='cfa5c3ed', fpmin_kout=0.9857871501170492,
                    exch_mean_obs=1.2728174525072664, comm_class='commutative_partial')
ANCH_3178 = dict(r_full={'qwen3-4b': None, 'qwen3-14b': None, 'glm4-9b': None})  # loaded at runtime

DEV_TOL = 1e-9
REPL_TOL = 1e-6
HELD_TOL = 0.1

GATE = {'held_delta_tol': HELD_TOL, 'n_folds': 16, 'calib_cells': 30, 'read_cells': 2}

DESIGN = {
    'phase': '3182', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3181 tail G5-B7 (frozen before any observation)',
    'data': 'sealed 3157 collect npz x3 (file sha pinned) + sealed 3177/3178 result.json; '
            'zero GPU, no new forward passes',
    'r_full': '3178 verbatim recompute: g(m,rel) from 3177 same-v_shared folds + '
              'within-relation entity-LOO null p (v_null from 30 cells); drift < 1e-6 '
              'vs sealed 3178 result.json (content 4f521d9b/1f166645 double-locked)',
    'held_construction': 'leave-one-relation-out cross-null (non-leaky pinning): held '
                         'relation rel_h, calib = OTHER relation; fold e holds out '
                         "entity e's 2 cells from calib (30 cells) -> v_x = unit_mean; "
                         'read = e\'s 2 cells in rel_h; p_x_fold(e) = mean(calib@v_x) - '
                         'mean(read@v_x); p_x(m,rel_h)=mean over 16 folds; r_held(m,rel)'
                         '=g(m,rel)-p_x(m,rel); r_held(m)=mean over rel. Pinning note: '
                         'prereg text said "null 标定用全部关系重建"; non-leaky version '
                         '(other relation only) pinned at freeze, deviation registered '
                         'in MEMO',
    'gate': '|r_held(m) - r_full(m)| < 0.1 x3 models => FTR-23 E1->E2_predictive '
            '(registry v1.8->v1.9 held anchor append + upgrade_log #7, atlas_v1_10.html); '
            'fail => F17 into failures + FTR-23 counter_evidence quantified append, '
            'stays E1',
    'sources': SHA_ANCHOR,
    'gates': GATE,
}


def freeze():
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3182, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


# ----------------- verbatim 3178 recipes -----------------
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


def run_arm(sealed_3177, sealed_3178):
    """3178 verbatim device/replication/null parts + NEW LORO cross-null."""
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

        # 3177 relation-gap g(m, rel) replication (verbatim folds)
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

        # within-relation entity-LOO null p(m, rel) - verbatim 3178
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
            nulls[rname] = dict(folds=pf, p_mean=float(np.mean([x['p_fold'] for x in pf])))

        # r_full(m) - verbatim 3178 residual + replication gate vs sealed 3178
        resid_full = {}
        for rname in ('isa', 'hasa'):
            g = folds[rname]['cos_train_a2'] - folds[rname]['cos_held_a2']
            assert abs(g - folds[rname]['a2_gap_train']) < 1e-12, ('gap sign', m, rname)
            resid_full[rname] = dict(gap=g, p=nulls[rname]['p_mean'], r=g - nulls[rname]['p_mean'])
        r_full = float((resid_full['isa']['r'] + resid_full['hasa']['r']) / 2.0)
        drift_r18 = abs(r_full - sealed_3178['arm']['per_model'][m]['r_m'])
        assert drift_r18 < REPL_TOL, ('3178 r_full drift', m, drift_r18)

        # NEW: LORO cross-relation null (calib = other relation, read = held relation)
        xnull = {}
        for rl_h, rname_h in ((0, 'isa'), (1, 'hasa')):
            rl_c = 1 - rl_h
            rname_c = 'hasa' if rl_h == 0 else 'isa'
            rows_c_all = dC_rows(rl_c)
            pxf = []
            for e in range(N_E):
                calib_drop = [e * 4 + rl_c * 2 + p for p in (0, 1)]
                calib_rows = [r for r in rows_c_all if r not in calib_drop]
                assert len(calib_rows) == 30, ('xnull calib size', m, rname_h, e)
                v_x = unit_mean(dCn[calib_rows])
                read_rows = [e * 4 + rl_h * 2 + p for p in (0, 1)]
                cos_c = float((dCn[calib_rows] @ v_x).mean())
                cos_h = float((dCn[read_rows] @ v_x).mean())
                pxf.append(dict(e=e, cos_calib=cos_c, cos_read=cos_h,
                                p_x_fold=cos_c - cos_h))
            assert all(np.isfinite(x['p_x_fold']) for x in pxf), ('xnull nonfinite', m, rname_h)
            assert max(abs(x['p_x_fold']) for x in pxf) < 0.5, ('xnull sanity', m, rname_h)
            xnull[rname_h] = dict(folds=pxf,
                                  p_x_mean=float(np.mean([x['p_x_fold'] for x in pxf])))
        resid_held = {}
        for rname in ('isa', 'hasa'):
            resid_held[rname] = dict(gap=folds[rname]['cos_train_a2'] - folds[rname]['cos_held_a2'],
                                     p_x=xnull[rname]['p_x_mean'],
                                     r_held=folds[rname]['cos_train_a2'] - folds[rname]['cos_held_a2']
                                     - xnull[rname]['p_x_mean'])
        r_held = float((resid_held['isa']['r_held'] + resid_held['hasa']['r_held']) / 2.0)
        delta = abs(r_held - r_full)

        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full, tc_full_drift=drift_tc,
                      r_full=r_full, r_full_drift_3178=drift_r18,
                      resid_full=resid_full, resid_held=resid_held,
                      r_held=r_held, held_delta=delta,
                      held_pass=bool(delta < HELD_TOL))
        log('%s: r_full=%.4f (3178 drift %.2e) | held isa r=%.4f (p_x=%.4f) '
            'hasa r=%.4f (p_x=%.4f) | r_held=%.4f delta=%.4f %s'
            % (m, r_full, drift_r18,
               resid_held['isa']['r_held'], resid_held['isa']['p_x'],
               resid_held['hasa']['r_held'], resid_held['hasa']['p_x'],
               r_held, delta, 'PASS' if delta < HELD_TOL else 'FAIL'))
    held_ok = all(out[m]['held_pass'] for m in MODELS)
    log('GATE held (|r_held - r_full| < %.2f x3): %s' % (HELD_TOL, held_ok))
    return out, held_ok


# ----------------- registry v1.8 -> v1.9 + atlas_v1_10 -----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span


def finish_registry(summary, held_ok, res8, seal8, design_sha):
    reg18 = json.loads(open(SRC['registry_v18'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg18['version'] == '1.8' and len(reg18['features']) == 23
    reg19 = copy.deepcopy(reg18)
    reg19['version'] = '1.9'
    reg19['supersedes'] = 'atlas_registry_v1_8.json (d08e9064, Phase 3181)'
    f23 = [f for f in reg19['features'] if f['id'] == 'FTR-23'][0]
    pm = summary['per_model']
    if held_ok:
        assert f23['evidence_level'] == 'E1_repeatable'
        f23['evidence_level'] = 'E2_predictive'
        held_anchor = {
            'src': 'p3182', 'phase': 3182, 'role': 'held_out_relation_loro_cross_null',
            'tags': ['cross_model', 'held_out'],
            'asserts': {
                'r_full.qwen3-4b': pm['qwen3-4b']['r_full'],
                'r_full.qwen3-14b': pm['qwen3-14b']['r_full'],
                'r_full.glm4-9b': pm['glm4-9b']['r_full'],
                'r_held.qwen3-4b': pm['qwen3-4b']['r_held'],
                'r_held.qwen3-14b': pm['qwen3-14b']['r_held'],
                'r_held.glm4-9b': pm['glm4-9b']['r_held'],
                'held_delta.qwen3-4b': pm['qwen3-4b']['held_delta'],
                'held_delta.qwen3-14b': pm['qwen3-14b']['held_delta'],
                'held_delta.glm4-9b': pm['glm4-9b']['held_delta'],
                'held_tol': HELD_TOL, 'res_sha8': res8, 'seal_sha8': seal8}}
        f23['anchors'] = f23['anchors'] + [held_anchor]
        reg19['upgrade_log'] = reg19['upgrade_log'] + [{
            'node': 'FTR-23',
            'change': 'E1_repeatable -> E2_predictive (held_out tag; '
                      'leave-one-relation-out cross-null band 3/3)',
            'reason': ('3182 arm: cross-relation null (calib = other relation, 30-cell '
                       'entity-LOO) r_held vs 3178-sealed r_full delta '
                       '%.4f/%.4f/%.4f < 0.1 x3 models; device drift < 1e-9, 3178 '
                       'r_full recompute drift < 1e-6' % (
                           pm['qwen3-4b']['held_delta'], pm['qwen3-14b']['held_delta'],
                           pm['glm4-9b']['held_delta']))}]
        disposition = 'v1.8_to_v1.9_ftr23_held_out_upgrade'
    else:
        ce = ('3182 held_out 关系留出臂（LORO cross-null）未过门：r_held−r_full = '
              '%.4f/%.4f/%.4f ≥ 0.1 ×3 不全满足——held_out 构造下关系特异残差不复现；'
              'FTR-23 保持 E1_repeatable，F17 入失败账本。'
              % (pm['qwen3-4b']['held_delta'], pm['qwen3-14b']['held_delta'],
                 pm['glm4-9b']['held_delta']))
        f23['counter_evidence'] = f23['counter_evidence'] + [ce]
        reg19['failures'] = reg19['failures'] + [{
            'id': 'F17', 'kind': 'held_out_gate_fail', 'phase': 3182,
            'text': 'FTR-23 leave-one-relation-out cross-null delta >= 0.1 '
                    '(r_held fails to replicate r_full under cross-relation null)',
            'evidence': 'p3182 result %s/seal %s' % (res8, seal8)}]
        disposition = 'v1.8_to_v1.9_ftr23_held_out_fail_f17'
    # preservation asserts
    for f in reg18['features']:
        if f['id'] == 'FTR-23':
            continue
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg19['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        assert a == b, ('feature not preserved', f['id'])
    for k in ('failures', 'upgrade_log', 'evidence_level_taxonomy', 'principles',
              'schema', 'models_mainline', 'models_pending', 'provenance',
              'created', 'phase', 'name', 'design_sha8'):
        if held_ok or k not in ('failures', 'upgrade_log'):
            assert json.dumps(reg18.get(k), ensure_ascii=False, sort_keys=True) == \
                json.dumps(reg19.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_9.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg19, ensure_ascii=False, indent=1))
    reg19_sha = sha8_file(rp)
    log('registry v1.9 written sha8 %s (%s)' % (reg19_sha, disposition))

    meta_sha = {'reg19': reg19_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.10 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.10 &mdash; 语义机制图谱（registry v1.9 渲染）</h1>')
    H.append('<div class="sub">Phase 3182 G5-B7 &middot; FTR-23 held_out 关系留出臂（LORO cross-null，%s） &middot; '
             'registry v1.9 sha8=%s &middot; v1.8 sha8=%s &middot; v1.7 sha8=%s &middot; v1.6 sha8=%s &middot; v1.5 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3177 %s + p3178 %s &middot; 渲染于 %s</div>' % (
                 'held_out 升级' if held_ok else 'F17 gate fail',
                 span('meta.registry_v19_sha8', reg19_sha),
                 span('meta.registry_v18_sha8', 'd08e9064'),
                 span('meta.registry_v17_sha8', 'f3ad070f'),
                 span('meta.registry_v16_sha8', 'bebdf38d'),
                 span('meta.registry_v15_sha8', p79.SHA_ANCHOR['registry_v15']),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3177_result', SHA_ANCHOR['p3177_result']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg19['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg19['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, p79.esc(pr)))
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
    for f in reg19['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.9，%d 条）</h2>' % len(reg19['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg19['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg19['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg19['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿）</h2>')
    for g in gl15['gaps']:
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
    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl15['appendix']))
    for a in gl15['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')
    H.append('<div class="footer">RDC/LPF atlas v1.10 &middot; 渲染自 atlas_registry_v1_9.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3182 G5-B7 FTR-23 held_out 关系留出臂'
             '（数据源 p3157 collect %s / %s / %s + p3177 %s + p3178 %s）&middot; 本文件为 Phase 3182 封存产物，'
             '字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg19_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3177_result'], SHA_ANCHOR['p3178_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_10.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v110(reg19, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_10.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3182, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'held_ok': bool(held_ok),
                'per_model': summary['per_model']},
        'counts': {'features': 23, 'failures': len(reg19['failures']),
                   'upgrades': len(reg19['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v19_file': 'atlas_registry_v1_9.json', 'registry_v19_sha8': reg19_sha,
        'html_file': 'atlas_v1_10.html', 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'sources': SHA_ANCHOR, 'res_sha8': res8, 'seal_sha8': seal8,
    }
    raw = json.dumps(final, ensure_ascii=False, indent=1, sort_keys=True)
    f_res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(final, res_sha8=f_res8), ensure_ascii=False, indent=1, sort_keys=True)
    f_seal8 = sha8(mid.encode('utf-8'))
    with open(os.path.join(OUTDIR, 'final_result.json'), 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(final, res_sha8=f_res8, seal_sha8=f_seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('final result res=%s seal=%s' % (f_res8, f_seal8))


def flatten_v110(reg, nodes, gl, meta_sha):
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
            d['%s.values.%s' % (p, k)] = p79.fmt(v) if hasattr(p79, 'fmt') else str(v)
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
    d['meta.registry_v19_sha8'] = meta_sha['reg19']
    d['meta.registry_v18_sha8'] = 'd08e9064'
    d['meta.registry_v17_sha8'] = 'f3ad070f'
    d['meta.registry_v16_sha8'] = 'bebdf38d'
    d['meta.registry_v15_sha8'] = p79.SHA_ANCHOR['registry_v15']
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3177_result'] = SHA_ANCHOR['p3177_result']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3182 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    # double-lock 3157 anchors
    for m in MODELS:
        ak = ANCH_KEY[m]
        r57 = json.load(open(SRC[RES_KEY[m]], encoding='utf-8'))
        A = ANCH_3157[ak]
        assert r57['res_sha8'] == A['res_sha8'] and r57['seal_sha8'] == A['seal_sha8'], \
            ('3157 content sha', m)
        assert r57['tc_mean_cos'] == A['tc_mean_cos'], ('3157 tc', m)
    s57 = json.load(open(SRC['p3157_summary'], encoding='utf-8'))
    assert s57['res_sha8'] == SUMMARY_3157['res_sha8'] and s57['seal_sha8'] == SUMMARY_3157['seal_sha8']
    log('3157 anchors double-locked (3 per-model + summary)')
    # double-lock 3177 + 3178 sealed results
    r17 = json.load(open(SRC['p3177_result'], encoding='utf-8'))
    assert r17['res_sha8'] == CONTENT_SHA_3177[0] and r17['seal_sha8'] == CONTENT_SHA_3177[1]
    assert r17['arm']['gate_a2'] is False, ('3177 gate state', )
    r18s = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert r18s['res_sha8'] == CONTENT_SHA_3178[0] and r18s['seal_sha8'] == CONTENT_SHA_3178[1]
    assert r18s['arm']['calibration_pass'] is False, ('3178 gate state', )
    log('3177 (%s/%s) + 3178 (%s/%s) sealed results double-locked'
        % (CONTENT_SHA_3177 + CONTENT_SHA_3178))

    out, held_ok = run_arm(r17, r18s)
    if held_ok:
        verdict = ('ftr23_upgraded_E2_predictive (held_out LORO cross-null band '
                   '|r_held - r_full| < %.2f x3; registry v1.8 -> v1.9 held anchor '
                   '+ upgrade_log #7)' % HELD_TOL)
    else:
        verdict = ('ftr23_held_out_gate_fail (F17 registered; FTR-23 stays E1, '
                   'counter_evidence quantified append)')
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    per_model = {m: {k: v for k, v in out[m].items()
                     if k not in ('resid_full', 'resid_held')}
                 for m in MODELS}
    resid_disclosed = {m: {'resid_full': out[m]['resid_full'],
                           'resid_held': {r: out[m]['resid_held'][r]
                                          for r in ('isa', 'hasa')}}
                       for m in MODELS}
    summary = {
        'phase': 3182, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'held_delta_tol': HELD_TOL, 'device_3157': True,
                  'replication_3177': True, 'replication_3178': True,
                  'held_ok': bool(held_ok)},
        'per_model': per_model,
        'residual_detail': resid_disclosed,
        'verdict': verdict,
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rp = os.path.join(OUTDIR, suffix + 'result.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    if not SMOKE:
        finish_registry(summary, held_ok, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
