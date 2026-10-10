# -*- coding: utf-8 -*-
"""Phase 3177: G5-B2 FTR-14 held_out relation-LOO T_C generalization (zero GPU).

Per prereg in MEMO Phase 3176 tail (frozen 2026-10-09 before any observation):
  3157 collect (16 ent x 2 rel x 2 ctx x 2 pol) relation leave-one-out fold:
  hold out relation r, rebuild the T_C shared component from the other
  relation (3157 recipe verbatim), measure the held-out relation T_C cos
  drift and exch band preservation.  Gate: held-out T_C cos vs full < 0.1 AND
  exch mean in 1.0-1.3 partial band, x3 models -> held_out anchor into the
  registry, FTR-14 upgraded to E2.  Protocol details exported verbatim from
  the phase3157 script at freeze time and frozen in execution.json.

Freeze-time protocol pinning (prereg delegates detail pinning to freeze):
  - Data = sealed 3157 collect npz x3 models (128 rows each).  The
    preregistered protocol body operates entirely on this sealed collect,
    i.e. NO new forward passes => zero-GPU arm.  The prereg title label
    "GPU 臂" is resolved in favor of the protocol body at freeze time,
    before any observation (precedent: 3176 held_out arms on sealed npz).
  - dC recipe verbatim 3157: dC(e,r,p) = H[IDX[(e,r,p,1)],KOUT] -
    H[IDX[(e,r,p,0)],KOUT] with KOUT = NL-1; dCn = dC/(||dC||+1e-18);
    tc = mean pairwise cos over triu(k=1); dtype flow fp16->fp32 verbatim.
  - T_C shared component reconstruction in fold r (train = other relation,
    32 dCn rows = 16 ent x 2 pol): v_shared = unit(mean(dCn_train)).
    A2 held reading  = mean(dCn_held @ v_shared).
    A2 full reading  = mean(dCn_all  @ v_full), v_full = unit(mean(dCn_all)).
  - GATE A (prereg "留出关系 T_C cos 与全量差 < 0.1"), frozen pre-observation
    as a conservative composite of BOTH sub-readings:
      A1: |tc_held_pairwise - tc_full| < 0.1          (subset statistic)
      A2: |cos_held_vs_shared - cos_full_vs_shared| < 0.1 (reconstruction)
    for all 2 folds x 3 models.  Descriptive extras: tc_train pairwise,
    a2_gap_train (same component, train vs held cells).
  - GATE B (prereg "exch 均值落 1.0-1.3 partial 带"): the commutator needs
    both relations, so no fold-level exch exists under 2-relation LOO; the
    band is re-asserted on the same full panel: exch recomputed verbatim
    3157 (device drift < 1e-9 vs sealed values), then cross-model
    mean(exchange_obs) in [1.0, 1.3].  Per-model exchange_obs reported
    (glm4 1.3311 slightly above 1.3 is reported; the gate is on the mean
    per the prereg wording "均值").
  - Verdict pass -> FTR-14 E1_repeatable -> E2_predictive, held_out anchor
    into registry v1.3 -> v1.4 + atlas_v1_5.html re-render (per-field
    data-k verification).  Fail -> counter_evidence append + F16, stays E1;
    render and versioning proceed regardless (honest registry).

SMOKE (env P3177_SMOKE=1): zero-GPU arm always FULL (numbers identical
across modes); only the html render subset differs.  DESIGN fully static.
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
OUTDIR = os.path.join(SRC_DIR, 'phase3177', 'g5b2_ftr14_heldout')
SMOKE = os.environ.get('P3177_SMOKE', '') == '1'
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
    'registry_v13': os.path.join(SRC_DIR, r'phase3176\g5b1_elev_upgrade\atlas_registry_v1_3.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a',
    'registry_v13': 'e64733f7', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
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
A_TOL = 0.1
EXCH_LO, EXCH_HI = 1.0, 1.3
DEV_TOL = 1e-9

DESIGN = {
    'phase': 3177, 'name': 'g5b2_ftr14_heldout', 'zero_gpu': True,
    'prereg': 'MEMO 2026-10-09 (Phase 3176 tail, G5-B2): 3157 collect (16 ent x 2 rel '
              'x 2 pol x 2 ctx) relation-LOO fold; held-out relation T_C reading drift '
              'vs full < 0.1 AND exch mean in 1.0-1.3 partial band x3 models -> '
              'held_out anchor into registry, FTR-14 upgraded to E2; protocol details '
              'exported verbatim from the phase3157 script at freeze time, frozen in '
              'execution.json before any observation',
    'freeze_pinning': dict(
        zero_gpu_resolution='prereg body pins the data source to the sealed 3157 '
            'collect npz (no new forward passes) => zero-GPU arm; prereg title label '
            'GPU-arm resolved in favor of the protocol body at freeze time, before '
            'any observation (precedent: 3176 held_out arms on sealed npz)',
        dC_recipe='verbatim 3157: dC(e,r,p) = H[IDX[(e,r,p,1)],KOUT] - '
            'H[IDX[(e,r,p,0)],KOUT], KOUT = NL-1 (readout); dCn = dC/(||dC||+1e-18); '
            'tc = mean pairwise cos over triu(k=1); dtype flow fp16 load -> fp32 '
            'compute, identical op order to 3157',
        shared_component='fold r (held-out relation r, train = other relation, 32 '
            'dCn rows = 16 ent x 2 pol): v_shared = unit(mean(dCn_train)); A2 held '
            'reading = mean(dCn_held @ v_shared); A2 full reading = mean(dCn_all @ '
            'v_full) with v_full = unit(mean(dCn_all over 64 rows))',
        gate_a_composite='prereg gate (held-out T_C cos vs full < 0.1) frozen as BOTH '
            'sub-readings, conservative, pre-observation: A1 |tc_held_pairwise - '
            'tc_full| < 0.1 AND A2 |cos_held_vs_shared - cos_full_vs_shared| < 0.1, '
            'all 2 folds x 3 models = 6 checks each; descriptive extras: tc_train '
            'pairwise, a2_gap_train (same component v_shared, train vs held cells)',
        exch_band='commutator needs both relations => no fold-level exch exists under '
            '2-relation LOO; exch band re-asserted on the same full panel: exch '
            'recomputed verbatim 3157 (device drift < 1e-9 vs sealed exchR_c0/exchN_c0/'
            'exchange_obs), then cross-model mean(exchange_obs) in [1.0, 1.3]; '
            'per-model exchange_obs reported (glm4 1.3311 slightly above 1.3 '
            'reported; gate on the mean per prereg wording "均值")'),
    'sources': {k: v.replace(BASE, '') for k, v in SRC.items()},
    'anchors_3157': {'per_model': ANCH_3157, 'summary': SUMMARY_3157},
    'gates': dict(a1_tol=A_TOL, a2_tol=A_TOL, exch_band=[EXCH_LO, EXCH_HI],
                  device_drift=DEV_TOL),
    'upgrade': dict(
        feature='FTR-14', path='E1_repeatable -> E2_predictive (held_out)',
        registry='v1.3 -> v1.4: FTR-14 upgraded on pass (statement + held_out anchor '
                 '+ values extended, scope_limits extended); other 21 features + '
                 'failures 15 + upgrade_log first 5 byte-for-byte; on fail: '
                 'counter_evidence append only + F16 failure entry, stays E1; render '
                 'and versioning proceed regardless (honest registry)',
        html='atlas_v1_5.html full re-render (22 features, 16 nodes, GAP-4 incl '
             'mechanism_note, failures F1-F15[/F16]) with per-field data-k '
             'verification against flattened v1.4/v1.5 sources'),
    'smoke': {'features': ['FTR-14', 'FTR-04', 'FTR-22'], 'nodes': 4, 'failures': 2,
              'arm': 'full (no truncation; numbers identical across modes)'},
}

LOG = []


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


# ------------------------------------------------- verbatim 3157 recipes
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
def run_arm():
    out = {}
    for m in MODELS:
        ak = ANCH_KEY[m]
        A = ANCH_3157[ak]
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

        # A2 full reading (v_full over all 64 rows)
        v_full = unit_mean(dCn)
        cos_full_a2 = float((dCn @ v_full).mean())

        # folds: held-out relation isa (rl=0) / hasa (rl=1)
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
            cosM_h = dCn_h @ dCn_h.T
            tri_h = cosM_h[np.triu_indices(len(rows_h), 1)]
            tc_held_a1 = float(tri_h.mean())
            cosM_tr = dCn_tr @ dCn_tr.T
            tri_tr = cosM_tr[np.triu_indices(len(rows_tr), 1)]
            tc_train_a1 = float(tri_tr.mean())
            a1_diff = abs(tc_held_a1 - tc_full)
            a2_diff = abs(cos_held_a2 - cos_full_a2)
            folds[fname] = dict(
                tc_held_a1=tc_held_a1, tc_train_a1=tc_train_a1,
                a1_diff=a1_diff, a1_pass=bool(a1_diff < A_TOL),
                cos_held_a2=cos_held_a2, cos_train_a2=cos_train_a2,
                a2_diff=a2_diff, a2_pass=bool(a2_diff < A_TOL),
                a2_gap_train=abs(cos_held_a2 - cos_train_a2))
            log('arm %s fold(%s held): A1 tc_held=%.4f vs full %.4f (diff %.4f, %s) | '
                'A2 cos_held=%.4f vs full %.4f (diff %.4f, %s) | tc_train=%.4f '
                'a2_gap_train=%.4f'
                % (m, fname, tc_held_a1, tc_full, a1_diff,
                   'PASS' if folds[fname]['a1_pass'] else 'FAIL',
                   cos_held_a2, cos_full_a2, a2_diff,
                   'PASS' if folds[fname]['a2_pass'] else 'FAIL',
                   tc_train_a1, folds[fname]['a2_gap_train']))
        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full, tc_full_drift=drift_tc,
                      exch=ex, exch_drift=drift_ex, exchange_obs=exchange_obs,
                      cos_full_a2=cos_full_a2, folds=folds)
        log('arm %s: KOUT=%d tc_full=%.6f (drift %.2e) exchange_obs=%.4f '
            'cos_full_a2=%.4f' % (m, KOUT, tc_full, drift_tc, exchange_obs, cos_full_a2))
    gate_a1 = all(out[m]['folds'][f]['a1_pass'] for m in MODELS for f in ('isa', 'hasa'))
    gate_a2 = all(out[m]['folds'][f]['a2_pass'] for m in MODELS for f in ('isa', 'hasa'))
    exch_mean = float(np.mean([out[m]['exchange_obs'] for m in MODELS]))
    drift_mean = abs(exch_mean - SUMMARY_3157['exch_mean_obs'])
    assert drift_mean < DEV_TOL, ('exch mean drift', drift_mean)
    gate_exch = bool(EXCH_LO <= exch_mean <= EXCH_HI)
    gate = bool(gate_a1 and gate_a2 and gate_exch)
    log('GATE A1 (|tc_held - tc_full| < %.2f, 2 folds x 3 models): %s' % (A_TOL, gate_a1))
    log('GATE A2 (|cos_held - cos_full| < %.2f, 2 folds x 3 models): %s' % (A_TOL, gate_a2))
    log('GATE B (mean exchange_obs %.4f in [%.1f, %.1f], drift %.2e): %s'
        % (exch_mean, EXCH_LO, EXCH_HI, drift_mean, gate_exch))
    log('per-model exchange_obs: ' + ' / '.join(
        '%s %.4f' % (m, out[m]['exchange_obs']) for m in MODELS))
    log('ARM GATE (A1 and A2 and exch band): %s' % gate)
    return out, gate_a1, gate_a2, gate_exch, exch_mean, gate


# ------------------------------------------------- registry upgrade
def build_registry(out, gate_a1, gate_a2, gate_exch, exch_mean, gate):
    reg13 = json.loads(open(SRC['registry_v13'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], 'rb').read().decode('utf-8'))
    reg2 = json.load(io.open(SRC['registry_3162'], encoding='utf-8'))
    nodes = reg2['audit']['nodes']
    assert reg13['version'] == '1.3' and gl15['version'] == '1.5'
    assert len(reg13['features']) == 22 and len(nodes) == 16
    assert len(reg13['failures']) == 15 and len(reg13['upgrade_log']) == 5
    f14_old = [f for f in reg13['features'] if f['id'] == 'FTR-14'][0]
    assert f14_old['evidence_level'] == 'E1_repeatable'
    f14 = copy.deepcopy(f14_old)

    a1isa = {m: out[m]['folds']['isa']['a1_diff'] for m in MODELS}
    a1hasa = {m: out[m]['folds']['hasa']['a1_diff'] for m in MODELS}
    a2isa = {m: out[m]['folds']['isa']['a2_diff'] for m in MODELS}
    a2hasa = {m: out[m]['folds']['hasa']['a2_diff'] for m in MODELS}
    a2held_isa = {m: out[m]['folds']['isa']['cos_held_a2'] for m in MODELS}
    a2held_hasa = {m: out[m]['folds']['hasa']['cos_held_a2'] for m in MODELS}
    upg_events = []
    f16 = None

    if gate:
        add = (
            ' 3177 held-out 定判（零 GPU，3157 collect 关系留出 fold，freeze 前协议钉定）：'
            '留出关系 r、以另一关系 32 个 dCn 行重建 T_C 共享分量 v_shared=unit(mean(dCn_train))。'
            'A2（重建型读数）留出关系对共享分量 cos 与全量口径差：isa '
            + '/'.join('%.4f' % a2isa[m] for m in MODELS) + '、hasa '
            + '/'.join('%.4f' % a2hasa[m] for m in MODELS) + '（门 <0.1，全过）；'
            '留出 cos 读数 isa ' + '/'.join('%.4f' % a2held_isa[m] for m in MODELS)
            + '、hasa ' + '/'.join('%.4f' % a2held_hasa[m] for m in MODELS) + '。'
            'A1（子集统计）留出关系内两两 cos 与全量差：isa '
            + '/'.join('%.4f' % a1isa[m] for m in MODELS) + '、hasa '
            + '/'.join('%.4f' % a1hasa[m] for m in MODELS) + '（门 <0.1，全过）；'
            '全量 tc = ' + '/'.join('%.4f' % out[m]['tc_full'] for m in MODELS) + '。'
            'A1 x A2 = 2 fold x 3 模型 = 6/6 全过。'
            'exch 带保持：cross-model mean(exchange_obs) = %.4f 落 [1.0, 1.3] partial 带'
            '（per-model ' % exch_mean + '/'.join('%.4f' % out[m]['exchange_obs'] for m in MODELS)
            + '；对易子需双关系，2 关系 LOO 下无 fold 级 exch，带在原面板重算 drift 0）。'
            'held_out 锚入表，FTR-14 升 E2_predictive。')
        f14['statement'] = f14['statement'] + add
        f14['evidence_level'] = 'E2_predictive'
        f14['anchors'] = f14['anchors'] + [{
            'src': 'p3157', 'phase': 3177, 'role': 'held_out_relation_loo_tcos',
            'tags': ['held_out'],
            'asserts': {
                'a1_diff_isa_qwen3_4b': a1isa['qwen3-4b'],
                'a1_diff_isa_qwen3_14b': a1isa['qwen3-14b'],
                'a1_diff_isa_glm4_9b': a1isa['glm4-9b'],
                'a1_diff_hasa_qwen3_4b': a1hasa['qwen3-4b'],
                'a1_diff_hasa_qwen3_14b': a1hasa['qwen3-14b'],
                'a1_diff_hasa_glm4_9b': a1hasa['glm4-9b'],
                'a2_diff_isa_qwen3_4b': a2isa['qwen3-4b'],
                'a2_diff_isa_qwen3_14b': a2isa['qwen3-14b'],
                'a2_diff_isa_glm4_9b': a2isa['glm4-9b'],
                'a2_diff_hasa_qwen3_4b': a2hasa['qwen3-4b'],
                'a2_diff_hasa_qwen3_14b': a2hasa['qwen3-14b'],
                'a2_diff_hasa_glm4_9b': a2hasa['glm4-9b'],
                'a2_held_isa_qwen3_4b': a2held_isa['qwen3-4b'],
                'a2_held_isa_qwen3_14b': a2held_isa['qwen3-14b'],
                'a2_held_isa_glm4_9b': a2held_isa['glm4-9b'],
                'a2_held_hasa_qwen3_4b': a2held_hasa['qwen3-4b'],
                'a2_held_hasa_qwen3_14b': a2held_hasa['qwen3-14b'],
                'a2_held_hasa_glm4_9b': a2held_hasa['glm4-9b'],
                'tc_full_qwen3_4b': out['qwen3-4b']['tc_full'],
                'tc_full_qwen3_14b': out['qwen3-14b']['tc_full'],
                'tc_full_glm4_9b': out['glm4-9b']['tc_full'],
                'exch_obs_qwen3_4b': out['qwen3-4b']['exchange_obs'],
                'exch_obs_qwen3_14b': out['qwen3-14b']['exchange_obs'],
                'exch_obs_glm4_9b': out['glm4-9b']['exchange_obs'],
                'exch_mean': exch_mean,
                'gate_a_tol': A_TOL, 'exch_band': [EXCH_LO, EXCH_HI],
                'src_3157_content_sha8': {
                    'qwen3-4b': ANCH_3157['qwen3-4b']['res_sha8'],
                    'qwen3-14b': ANCH_3157['qwen3-14b']['res_sha8'],
                    'glm4': ANCH_3157['glm4']['res_sha8'],
                    'summary': SUMMARY_3157['res_sha8']},
                'src_file_sha8': {'p3157_4b': SHA_ANCHOR['p3157_4b'],
                                  'p3157_14b': SHA_ANCHOR['p3157_14b'],
                                  'p3157_glm4': SHA_ANCHOR['p3157_glm4'],
                                  'registry_v13': SHA_ANCHOR['registry_v13'],
                                  'gap_v15': SHA_ANCHOR['gap_v15']}}}]
        f14['values'] = dict(f14['values'],
                             loo_a2_held_isa_qwen3_4b=a2held_isa['qwen3-4b'],
                             loo_a2_held_isa_qwen3_14b=a2held_isa['qwen3-14b'],
                             loo_a2_held_isa_glm4_9b=a2held_isa['glm4-9b'],
                             exch_mean_recomputed=exch_mean)
        f14['scope_limits'] = f14['scope_limits'] + \
            ' 留出泛化限 2 关系 fold（isa/hasa 互为留出）；exch 带门=跨模型均值（per-model 见锚）；' \
            '对易子需双关系，LOO 下无 fold 级 exch。'
        upg_events.append({
            'node': 'FTR-14',
            'change': 'E1_repeatable -> E2_predictive (held_out tag; relation-LOO '
                      'T_C generalization A1/A2 6/6 + exch band)',
            'reason': '3177: T_C shared component rebuilt from the other relation '
                      'predicts the held-out relation readings (A2 diff %.4f/%.4f/%.4f '
                      'isa, %.4f/%.4f/%.4f hasa; A1 diff %.4f/%.4f/%.4f isa, '
                      '%.4f/%.4f/%.4f hasa; tol 0.1) and exch mean %.4f stays in the '
                      'partial band' % (
                          a2isa['qwen3-4b'], a2isa['qwen3-14b'], a2isa['glm4-9b'],
                          a2hasa['qwen3-4b'], a2hasa['qwen3-14b'], a2hasa['glm4-9b'],
                          a1isa['qwen3-4b'], a1isa['qwen3-14b'], a1isa['glm4-9b'],
                          a1hasa['qwen3-4b'], a1hasa['qwen3-14b'], a1hasa['glm4-9b'],
                          exch_mean)})
        log('FTR-14 upgraded: E1_repeatable -> E2_predictive (held_out)')
    else:
        parts = []
        if not gate_a1:
            parts.append('A1 留出-全量差超门 ' + '/'.join(
                '%.4f' % max(a1isa[m], a1hasa[m]) for m in MODELS))
        if not gate_a2:
            parts.append('A2 留出-全量差超门 ' + '/'.join(
                '%.4f' % max(a2isa[m], a2hasa[m]) for m in MODELS))
        if not gate_exch:
            parts.append('exch 均值 %.4f 出 [1.0, 1.3] 带' % exch_mean)
        f14['counter_evidence'] = f14['counter_evidence'] + [
            '3177 arm FAILED gate: ' + '；'.join(parts) + ' — held_out 升级未达成']
        f16 = {'id': 'F16', 'kind': 'upgrade_gate_failed', 'phase': 3177,
               'text': 'E1->E2 升级门未全过：FTR-14 关系留出 fold（' + '；'.join(parts)
                       + '）。相应特征保持 E1_repeatable，held_out 锚不入表（诚实登记）。',
               'evidence': 'p3177 run (this phase); gates A1/A2/exch = %s/%s/%s'
                           % (gate_a1, gate_a2, gate_exch)}
        log('FTR-14 stays E1: ' + '；'.join(parts))

    # assembly
    reg14 = copy.deepcopy(reg13)
    reg14['version'] = '1.4'
    reg14['supersedes'] = 'atlas_registry_v1_3.json (%s, Phase 3176)' % SHA_ANCHOR['registry_v13']
    new_feats = []
    for f in reg13['features']:
        new_feats.append(f14 if f['id'] == 'FTR-14' else f)
    reg14['features'] = new_feats
    assert len(reg14['features']) == 22

    # preservation checks
    for f in reg13['features']:
        fid = f['id']
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b_obj = [x for x in reg14['features'] if x['id'] == fid][0]
        b = json.dumps(b_obj, ensure_ascii=False, sort_keys=True)
        if fid == 'FTR-14':
            if gate:
                assert a != b, ('upgrade did not change feature', fid)
            else:
                fa, fb = json.loads(a), json.loads(b)
                ce_old = fa.pop('counter_evidence')
                ce_new = fb.pop('counter_evidence')
                assert json.dumps(fa, ensure_ascii=False, sort_keys=True) == \
                    json.dumps(fb, ensure_ascii=False, sort_keys=True), \
                    ('FTR-14 non-counter fields changed on failed arm', fid)
                assert ce_new[:len(ce_old)] == ce_old and \
                    len(ce_new) == len(ce_old) + 1, ('counter append', fid)
        else:
            assert a == b, ('feature not preserved', fid)
    log('preservation: 21 features byte-for-byte; FTR-14 %s'
        % ('upgraded' if gate else 'counter_evidence append only'))

    fails = list(reg13['failures'])
    if f16 is not None:
        fails = fails + [f16]
    reg14['failures'] = fails
    for i in range(15):
        a = json.dumps(reg13['failures'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg14['failures'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('failure not preserved', i)
    upgs = list(reg13['upgrade_log']) + upg_events
    reg14['upgrade_log'] = upgs
    for i in range(5):
        a = json.dumps(reg13['upgrade_log'][i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg14['upgrade_log'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('old upgrade event changed', i)
    assert len(reg14['upgrade_log']) == 5 + len(upg_events)
    log('upgrade_log %d -> %d (first 5 byte-for-byte)'
        % (5, len(reg14['upgrade_log'])))

    # G4 asserts
    f14_new = [x for x in reg14['features'] if x['id'] == 'FTR-14'][0]
    if gate:
        assert f14_new['evidence_level'] == 'E2_predictive', ('G4 level', )
        tags = [t for a in f14_new['anchors'] for t in a.get('tags', [])]
        assert 'held_out' in tags, ('G4 tag missing', )
        assert len(f14_new['anchors']) == 3, ('G4 n_anchors', )
        av = f14_new['anchors'][-1]['asserts']
        assert av['a2_diff_isa_qwen3_4b'] == a2isa['qwen3-4b'], ('G4 value', )
        assert av['exch_mean'] == exch_mean, ('G4 value', )
        assert av['src_file_sha8']['p3157_4b'] == SHA_ANCHOR['p3157_4b']
    else:
        assert f14_new['evidence_level'] == 'E1_repeatable', ('G4 fallback level', )
        assert reg14['failures'][-1]['id'] == 'F16'
    log('G4 asserts OK (level/tags/values consistent with arm verdict)')
    return reg14, nodes, gl15, upg_events, f16


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


def render_html(reg, nodes, gl, feat_subset, node_subset, fail_subset, upg_subset, meta_sha):
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.5 - RDC/LPF 语义图谱</title><style>%s</style></head>' % CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.5 &mdash; 语义机制图谱（registry v1.4 渲染）</h1>')
    H.append('<div class="sub">Phase 3177 G5-B2 &middot; FTR-14 held_out 关系留出泛化 &middot; '
             'registry v1.4 sha8=%s &middot; registry v1.3 sha8=%s &middot; registry v1.2 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v14_sha8', meta_sha['reg14']),
                 span('meta.registry_v13_sha8', SHA_ANCHOR['registry_v13']),
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
    for n in node_subset:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')

    fams = {}
    for f in feat_subset:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.4，%d 条）</h2>' % len(reg['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(render_feature(f))
        H.append('</div>')

    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in upg_subset:
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')

    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in fail_subset:
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

    H.append('<div class="footer">RDC/LPF atlas v1.5 &middot; 渲染自 atlas_registry_v1_4.json (%s)'
             ' + gap ledger v1.5 (%s，3175 定稿原样渲染) &middot; 增量来源: Phase 3177 G5-B2 '
             'FTR-14 held_out 关系留出泛化（p3157 collect %s / %s / %s）&middot; 本文件为 Phase 3177 '
             '封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 meta_sha['reg14'], SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4']))
    H.append('</body></html>')
    return '\n'.join(H)


meta_sha_global = {'reg14': None}


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
    d['meta.registry_v14_sha8'] = meta_sha_global['reg14']
    d['meta.registry_v13_sha8'] = SHA_ANCHOR['registry_v13']
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
    log('== Phase 3177 g5b2_ftr14_heldout %s ==' % ('SMOKE' if SMOKE else 'FULL'))

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

    out, gate_a1, gate_a2, gate_exch, exch_mean, gate = run_arm()

    reg14, nodes, gl15, upg_events, f16 = build_registry(
        out, gate_a1, gate_a2, gate_exch, exch_mean, gate)

    suffix = 'smoke_' if SMOKE else ''
    reg14_path = os.path.join(OUTDIR, suffix + 'atlas_registry_v1_4.json')
    with io.open(reg14_path, 'w', encoding='utf-8') as f:
        json.dump(reg14, f, ensure_ascii=False, indent=1)
    meta_sha_global['reg14'] = sha8_file(reg14_path)
    log('registry v1.4 written sha8 %s' % meta_sha_global['reg14'])

    feats_all = reg14['features']
    feats = feats_all
    fails_sub = reg14['failures']
    upgs_sub = list(enumerate(reg14['upgrade_log']))
    nodes_sub = nodes[:]
    if SMOKE:
        keep_ids = tuple(DESIGN['smoke']['features'])
        feats = [f for f in feats_all if f['id'] in keep_ids]
        nodes_sub = nodes[:DESIGN['smoke']['nodes']]
        fails_sub = fails_sub[:DESIGN['smoke']['failures']]

    flat = flatten(dict(reg14, features=feats_all, failures=reg14['failures'],
                        upgrade_log=reg14['upgrade_log']),
                   nodes_sub, gl15)
    if SMOKE:
        keep = set()
        for f in feats:
            keep |= set(k for k in flat if k.startswith(f['id'] + '.'))
        for n in nodes_sub:
            keep |= set(k for k in flat if k.startswith(n['id'] + '.'))
        for i, u in upgs_sub:
            keep |= set(k for k in flat if k.startswith('UPG.%d.' % i))
        for fl in fails_sub:
            keep |= set(k for k in flat if k.startswith(fl['id'] + '.'))
        keep |= set(k for k in flat if k.startswith(('GAP-', 'APPX-', 'meta.')))
        flat = {k: v for k, v in flat.items() if k in keep}

    html_text = render_html(reg14, nodes_sub, gl15, feats, nodes_sub, fails_sub,
                            upgs_sub, meta_sha_global)

    assert '<link' not in html_text and '<script' not in html_text, 'external resource found'
    assert 'http://' not in html_text and 'https://' not in html_text, 'external url found'
    log('G6 no-external OK')

    v = verify_html(html_text, flat)
    log('G1 field check: expected=%d found=%d missing=%d extra=%d mismatch=%d dups=%d' % (
        v['n_expected'], v['n_found'], len(v['missing']), len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'], ('missing keys', v['missing'][:10])
    assert not v['extra'], ('extra keys', v['extra'][:10])
    assert not v['mismatch'], ('mismatch', v['mismatch'][:5])
    assert not v['dups'], ('dup keys', v['dups'][:5])

    html_path = os.path.join(OUTDIR, suffix + 'atlas_v1_5.html')
    with io.open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_text)
    html_sha = sha8_file(html_path)
    log('html written %s (%d B, sha8 %s)' % (os.path.basename(html_path),
                                             len(html_text.encode('utf-8')), html_sha))

    gtag = 'pass' if gate else 'fail'
    verdict = ('g5b2_ftr14_heldout|gate_%s|a1_%s|a2_%s|exch_band_%s|exch_mean_%.4f|'
               'upgrades_%d|features_22|failures_%d|registry_v14|html_fields_%d_ok') % (
        gtag, '6of6' if gate_a1 else 'broken', '6of6' if gate_a2 else 'broken',
        'keep' if gate_exch else 'break', exch_mean, len(upg_events),
        len(reg14['failures']), v['n_found'])

    summary = {
        'phase': 3177, 'name': 'g5b2_ftr14_heldout', 'smoke': SMOKE,
        'design_sha8': d8, 'verdict': verdict,
        'registry_v14_file': os.path.basename(reg14_path),
        'registry_v14_sha8': meta_sha_global['reg14'],
        'html_file': os.path.basename(html_path), 'html_sha8': html_sha,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'arm': {'gate': gate, 'gate_a1': gate_a1, 'gate_a2': gate_a2,
                'gate_exch_band': gate_exch, 'exch_mean': exch_mean,
                'per_model': out},
        'upgrades': upg_events, 'n_upgrades': len(upg_events),
        'failure_added': f16,
        'counts': {'features': len(feats_all), 'nodes': len(nodes_sub),
                   'failures': len(reg14['failures']),
                   'upgrades': len(reg14['upgrade_log']),
                   'gaps': len(gl15['gaps']), 'appendix': len(gl15['appendix'])},
        'gap_status': {g['id']: g['status'] for g in gl15['gaps']},
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
