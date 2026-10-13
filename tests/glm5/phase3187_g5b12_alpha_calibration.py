# -*- coding: utf-8 -*-
"""Phase 3187 (G5-B12): R_rel mean-level alpha calibration (T2 closing arm,
zero GPU).

Per prereg in MEMO Phase 3186 tail, section "接续：预注册 3187" (frozen before
any observation):
  After the 3186 mean-level contraction, calibrate the single amplitude
  parameter alpha with one-parameter least squares: model
    g_e = gbar - kappa * alpha * gam_e        (3185 prediction-A form with
  the fixed coefficient 1 replaced by free alpha), per model per held
  relation fold (2 folds x 3 models = 6 fits):
    alpha_hat = -sum((g_e - gbar) * gam_e) / (kappa * sum(gam_e^2))
    (= <Delta-v residual projection, g component> / ||.||^2, prereg wording)
  - explained rate = R2 of the one-parameter fit (alpha free; equals
    squared corr(g_e, gam_e) by LS algebra; cross-checked numerically).
  - p_x side (3182 xnull): one-parameter alpha_px fit, REPORT-ONLY
    descriptive (prereg pinned g-side only).
  - Device/replication gates identical to 3186 (3157 tc <1e-9; 3177 folds /
    3178 r_full / 3182 p_x / 3183 dv / 3185 full-panel r_A,R2 <1e-9).
  - GATE (pinned pre-observation): STABLE iff per model sign(alpha_hat_isa)
    == sign(alpha_hat_hasa) AND |alpha_hat| < 100 sanity x3 AND cross-model
    band of |alpha_hat_m| (mean of two folds) has max/min <= 3
    => alpha calibrated: alpha_hat x3 + explained rates into FTR-24 anchor
    (role=rrel_alpha_calibration), T2 mean-level parameterization finalized
    in statement (registry v1.13 -> v1.14, 23 legacy byte-preserved),
    atlas_v1_15.html.
    Otherwise UNSTABLE => T2 downgraded to direction-level statement (keep
    Delta-v direction only, amplitude claims withdrawn): FTR-24 statement
    append only (registry v1.13 -> v1.14 likewise).
  - SMOKE (env P3187_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
NAME = 'g5b12_alpha_calibration'
OUTDIR = os.path.join(SRC_DIR, 'phase3187', NAME)
SMOKE = os.environ.get('P3187_SMOKE') == '1'
LOG = []
N_E = 16
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}

G_BAND = 3.0
ALPHA_SANITY = 100.0
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'band_max_min': G_BAND, 'alpha_sanity': ALPHA_SANITY,
        'repl_tol': REPL_TOL, 'dev_tol': DEV_TOL}

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
    'p3182_result': os.path.join(SRC_DIR, r'phase3182\g5b7_ftr23_heldout_loro\result.json'),
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'p3185_result': os.path.join(SRC_DIR, r'phase3185\g5b10_rrel_predictive\result.json'),
    'registry_v113': os.path.join(SRC_DIR, r'phase3186\g5b11_second_order\atlas_registry_v1_13.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a', 'p3177_result': 'b2ad52ef', 'p3178_result': 'abae296e',
    'p3182_result': '15b16f37', 'p3183_result': '0ab70579', 'p3185_result': '386f6197',
    'registry_v113': '730729f5', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_SHA_3177 = ('d451030f', '7b746eb2')
CONTENT_SHA_3178 = ('4f521d9b', '1f166645')
CONTENT_SHA_3182 = ('33d73196', 'ed51ca22')
CONTENT_SHA_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_SHA_3185 = ('df958cc9', '7559b27b')

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, res_sha8='c0db3455', seal_sha8='41916ed8',
                     tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, res_sha8='1270c921', seal_sha8='98c06ab0',
                      tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, res_sha8='a1d9a98a', seal_sha8='e8be1026',
                 tc_mean_cos=0.5640953183174133),
}

DESIGN = {
    'phase': '3187', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3186 tail G5-B12 (frozen before any observation)',
    'data': 'sealed 3157 collect npz x3 + sealed 3177/3178/3182/3183/3185 '
            'result.json; zero GPU',
    'alpha_definition': 'per model per held-relation fold: one-parameter LS on 3185 '
                        'prediction-A form g_e = gbar - kappa*alpha*gam_e; '
                        'alpha_hat = -sum((g_e-gbar)*gam_e)/(kappa*sum(gam_e^2)); '
                        'explained rate = R2 (alpha free); p_x-side alpha_px '
                        'one-parameter fit report-only',
    'measurements': 'alpha_hat x6 (2 folds x 3 models), explained rate x6, '
                    'alpha_px report-only x6, cross-fold sign consistency, '
                    'cross-model band max/min of |alpha_hat_m|',
    'gate': 'STABLE iff per model sign(alpha_isa)==sign(alpha_hasa) AND '
            '|alpha|<100 sanity x3 AND cross-model band max/min <= 3 => alpha '
            'calibrated: alpha_hat x3 + explained rates into FTR-24 anchor '
            '(role=rrel_alpha_calibration), T2 mean-level finalized in statement '
            '(registry v1.13->v1.14, 23 legacy byte-preserved, atlas_v1_15.html); '
            'else UNSTABLE => T2 downgraded to direction-level statement, FTR-24 '
            'statement append only',
    'sources': SHA_ANCHOR,
    'gates': GATE,
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
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3187, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


# ----------------- verbatim helpers (3178/3182/3183/3185) -----------------
def dC_rows(rel_filter):
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


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def r2(meas, pred):
    meas = np.asarray(meas, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(((meas - pred) ** 2).sum())
    ss_tot = float(((meas - meas.mean()) ** 2).sum())
    if ss_tot < 1e-300:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def xnull_folds(dCn, rl_h):
    """3182 LORO cross-null verbatim: per-entity folds."""
    rl_c = 1 - rl_h
    rows_c_all = dC_rows(rl_c)
    pxf = []
    for e in range(N_E):
        calib_drop = [e * 4 + rl_c * 2 + p for p in (0, 1)]
        calib_rows = [r for r in rows_c_all if r not in calib_drop]
        assert len(calib_rows) == 30, ('xnull calib size', rl_h, e)
        v_x = unit_mean(dCn[calib_rows])
        read_rows = [e * 4 + rl_h * 2 + p for p in (0, 1)]
        cos_c = float((dCn[calib_rows] @ v_x).mean())
        cos_h = float((dCn[read_rows] @ v_x).mean())
        pxf.append(dict(e=e, v_x=v_x, cos_calib=cos_c, cos_read=cos_h,
                        p_x_fold=cos_c - cos_h))
    return pxf


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

        # device gate: tc_full verbatim
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
        assert abs(tc_full - A['tc_mean_cos']) < DEV_TOL, ('tc drift', m)

        # 3183 replication (dv quantities)
        v_isa = unit_mean(dCn[dC_rows(0)])
        v_hasa = unit_mean(dCn[dC_rows(1)])
        cos_rl = float(v_isa @ v_hasa)
        dv = v_isa - v_hasa
        norm_dv = float(np.linalg.norm(dv))
        v_full = unit_mean(dCn)
        frac_full = float((dv @ v_full) ** 2 / max(norm_dv ** 2, 1e-300))
        U = []
        for e in range(N_E):
            a_e = dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0)
            b_e = dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)
            U.append(unit(a_e - b_e))
        U = np.stack(U)
        s0 = np.linalg.svd(U, compute_uv=False)
        pc_share = float(s0[0] ** 2 / (s0 ** 2).sum())
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
        C = Un @ Un.T
        med_cos = float(np.median(C[np.triu_indices(N_E, 1)]))
        E83 = sealed_3183['per_model'][m]
        d83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']),
                  abs(frac_full - E83['frac_full']), abs(pc_share - E83['pc_share']),
                  abs(med_cos - E83['med_cos']))
        assert d83 < REPL_TOL, ('3183 dv replication', m, d83)
        dvu = unit(dv)

        # 3177/3178/3182 replications (compact verbatim, as 3186)
        folds_rep = {}
        nulls = {}
        for rl_h, fname in ((0, 'isa'), (1, 'hasa')):
            rows_h = dC_rows(rl_h)
            v_c = unit_mean(dCn[dC_rows(1 - rl_h)])
            cos_train = float((dCn[dC_rows(1 - rl_h)] @ v_c).mean())
            cos_held = float((dCn[rows_h] @ v_c).mean())
            E77 = sealed_3177['arm']['per_model'][m]['folds'][fname]
            d77 = max(abs(cos_train - E77['cos_train_a2']),
                      abs(cos_held - E77['cos_held_a2']))
            assert d77 < REPL_TOL, ('3177 fold replication', m, fname, d77)
            folds_rep[fname] = dict(cos_train_a2=cos_train, cos_held_a2=cos_held)
            rows_all = dC_rows(rl_h)
            pf = []
            for e in range(N_E):
                held_rows = [e * 4 + rl_h * 2 + p for p in (0, 1)]
                train_rows = [r for r in rows_all if r not in held_rows]
                v_null = unit_mean(dCn[train_rows])
                pf.append(float((dCn[train_rows] @ v_null).mean() -
                                (dCn[held_rows] @ v_null).mean()))
            nulls[fname] = float(np.mean(pf))
        r_full = float(sum((folds_rep[r]['cos_train_a2'] - folds_rep[r]['cos_held_a2']
                            - nulls[r]) for r in ('isa', 'hasa')) / 2.0)
        d78 = abs(r_full - sealed_3178['arm']['per_model'][m]['r_m'])
        assert d78 < REPL_TOL, ('3178 r_full replication', m, d78)
        px_rep = {}
        for rl_h, rname in ((0, 'isa'), (1, 'hasa')):
            pxf = xnull_folds(dCn, rl_h)
            E82 = sealed_3182['residual_detail'][m]['resid_held'][rname]['p_x']
            p_rep = abs(float(np.mean([x['p_x_fold'] for x in pxf])) - E82)
            assert p_rep < REPL_TOL, ('3182 p_x replication', m, rname, p_rep)
            px_rep[rname] = p_rep

        # ---- alpha calibration (g side, per fold) + p_x side report-only ----
        alpha = {}
        for rl_h, rname in ((0, 'isa'), (1, 'hasa')):
            rows_h = dC_rows(rl_h)
            v_c = unit_mean(dCn[dC_rows(1 - rl_h)])
            cos_train = float((dCn[dC_rows(1 - rl_h)] @ v_c).mean())
            vh_t = dCn[rows_h].mean(0)
            gbar = cos_train - float(vh_t @ v_c)
            kap = float(dvu @ v_c)
            gam = np.zeros(N_E)
            g_meas = np.zeros(N_E)
            for e in range(N_E):
                u_e = dCn[[e * 4 + rl_h * 2 + p for p in (0, 1)]].mean(0)
                gam[e] = float((u_e - vh_t) @ dvu)
                g_meas[e] = cos_train - float(u_e @ v_c)
            # identities (3185)
            assert abs(g_meas.mean() - gbar) < 1e-6, ('gbar identity', m, rname)
            assert abs(gam.mean()) < 1e-6, ('gam zero-mean identity', m, rname)
            denom = kap * float((gam ** 2).sum())
            assert abs(denom) > 1e-12, ('alpha denominator', m, rname)
            a_hat = -float(((g_meas - gbar) * gam).sum()) / denom
            g_pred = gbar - kap * a_hat * gam
            R2 = r2(g_meas, g_pred)
            # LS algebra cross-check: R2 == corr(g, gam)^2
            cc = float(np.corrcoef(g_meas, gam)[0, 1])
            assert abs(R2 - cc ** 2) < 1e-9, ('R2 == r^2 algebra', m, rname, R2, cc ** 2)

            # p_x side one-parameter fit (report-only)
            pxf = xnull_folds(dCn, rl_h)
            p_meas = np.array([x['p_x_fold'] for x in pxf])
            p_base = np.zeros(N_E)
            p_slope = np.zeros(N_E)
            for e in range(N_E):
                x = pxf[e]
                u_e = dCn[[e * 4 + rl_h * 2 + p for p in (0, 1)]].mean(0)
                gam_e = float((u_e - vh_t) @ dvu)
                p_base[e] = x['cos_calib'] - float(vh_t @ x['v_x'])
                p_slope[e] = float(dvu @ x['v_x']) * gam_e
            a_px = -float(((p_meas - p_base.mean()) * p_slope).sum()) / \
                float((p_slope ** 2).sum())
            p_pred = p_base - a_px * p_slope
            R2_px = r2(p_meas, p_pred)

            alpha[rname] = dict(alpha_hat=a_hat, explained_rate=R2,
                                kap=kap, gam_var=float((gam ** 2).mean()),
                                gbar=gbar, alpha_px=a_px, r2_px_report=R2_px,
                                g_meas=[float(x) for x in g_meas],
                                g_pred=[float(x) for x in g_pred])
            log('  %s rel(%s): alpha_hat=%+.4f explained=%.4f (kap=%+.4f) | '
                'alpha_px=%+.4f R2_px=%.4f (report-only)'
                % (m, rname, a_hat, R2, kap, a_px, R2_px))
        a_m = float((alpha['isa']['alpha_hat'] + alpha['hasa']['alpha_hat']) / 2.0)
        sign_ok = bool(np.sign(alpha['isa']['alpha_hat']) ==
                       np.sign(alpha['hasa']['alpha_hat']))
        sanity_ok = bool(abs(alpha['isa']['alpha_hat']) < ALPHA_SANITY and
                         abs(alpha['hasa']['alpha_hat']) < ALPHA_SANITY)
        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full,
                      dv_rep=dict(cos_rl=cos_rl, norm_dv=norm_dv, frac_full=frac_full,
                                  pc_share=pc_share, med_cos=med_cos, drift=d83),
                      folds_rep=folds_rep, r_full_rep=r_full, r_full_drift=d78,
                      px_rep=px_rep, alpha=alpha, alpha_m=a_m,
                      sign_ok=sign_ok, sanity_ok=sanity_ok)
        log('%s: alpha_m=%+.4f sign_ok=%s sanity_ok=%s'
            % (m, a_m, sign_ok, sanity_ok))

    gate_sign = all(out[m]['sign_ok'] and out[m]['sanity_ok'] for m in MODELS)
    bands = [abs(out[m]['alpha_m']) for m in MODELS]
    band_ratio = float(max(bands) / max(min(bands), 1e-300))
    gate_band = bool(band_ratio <= G_BAND)
    gate_all = bool(gate_sign and gate_band)
    log('GATE sign+sanity x3: %s | band |alpha_m| %s ratio %.2f (gate <= %.1f): %s'
        % (gate_sign, '/'.join('%.4f' % b for b in bands), band_ratio, G_BAND, gate_band))
    log('GATE STABLE: %s' % gate_all)
    return out, gate_all, band_ratio


# ----------------- registry v1.13 -> v1.14 + atlas_v1_15 -----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span


def finish_registry(summary, gate_all, res8, seal8, design_sha):
    reg23 = json.loads(open(SRC['registry_v113'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg23['version'] == '1.13' and len(reg23['features']) == 24
    assert len(reg23['failures']) == 19 and len(reg23['upgrade_log']) == 6
    reg24 = copy.deepcopy(reg23)
    reg24['version'] = '1.14'
    reg24['supersedes'] = 'atlas_registry_v1_13.json (730729f5, Phase 3186)'
    pm = summary['per_model']
    fmt3 = lambda k: '/'.join('%.4f' % pm[m][k] for m in MODELS)
    f24_old = [f for f in reg23['features'] if f['id'] == 'FTR-24'][0]
    f24 = copy.deepcopy(f24_old)
    if gate_all:
        anchor = {
            'src': 'p3187', 'phase': 3187,
            'role': 'rrel_alpha_calibration',
            'tags': ['cross_model'],
            'asserts': {
                'alpha_hat.qwen3-4b': pm['qwen3-4b']['alpha_m'],
                'alpha_hat.qwen3-14b': pm['qwen3-14b']['alpha_m'],
                'alpha_hat.glm4-9b': pm['glm4-9b']['alpha_m'],
                'explained_rate.qwen3-4b': float(
                    (pm['qwen3-4b']['alpha']['isa']['explained_rate'] +
                     pm['qwen3-4b']['alpha']['hasa']['explained_rate']) / 2),
                'explained_rate.qwen3-14b': float(
                    (pm['qwen3-14b']['alpha']['isa']['explained_rate'] +
                     pm['qwen3-14b']['alpha']['hasa']['explained_rate']) / 2),
                'explained_rate.glm4-9b': float(
                    (pm['glm4-9b']['alpha']['isa']['explained_rate'] +
                     pm['glm4-9b']['alpha']['hasa']['explained_rate']) / 2),
                'gate_band_max_min': G_BAND,
                'res_sha8': res8, 'seal_sha8': seal8}}
        f24['anchors'] = f24_old['anchors'] + [anchor]
        f24['statement'] = f24_old['statement'] + (
            '；3187 G5-B12 α 定标定稿：单参数最小二乘 α̂ = ' + fmt3('alpha_m') +
            '（同模型两 fold 符号一致、跨模型带比 ≤3），解释率 ' +
            '/'.join('%.4f' % ((pm[m]['alpha']['isa']['explained_rate'] +
                                pm[m]['alpha']['hasa']['explained_rate']) / 2)
                     for m in MODELS) +
            '——T2 均值级参数化 ΔM_rel = T_C + αΔv 正式定稿（α̂ 入锚；'
            '实体级预测主张维持撤销，见 3186 收缩）。')
        reg24['features'] = [f24 if f['id'] == 'FTR-24' else f for f in reg23['features']]
        disposition = 'v1.13_to_v1.14_alpha_calibrated'
    else:
        f24['statement'] = f24_old['statement'] + (
            '；3187 G5-B12 α 定标不稳定：α̂ = ' + fmt3('alpha_m') +
            '（' + '；'.join('%s isa %+.4f / hasa %+.4f' %
                            (m, pm[m]['alpha']['isa']['alpha_hat'],
                             pm[m]['alpha']['hasa']['alpha_hat']) for m in MODELS) +
            '，跨模型带比 %.2f）——T2 降级为方向级陈述：FTR-24 只保留 Δv 方向主张，'
            '幅度参数 α 撤销（实体级主张已于 3186 撤销）。' % summary['gates']['band_ratio'])
        reg24['features'] = [f24 if f['id'] == 'FTR-24' else f for f in reg23['features']]
        disposition = 'v1.13_to_v1.14_direction_level_downgrade'
    # preservation asserts
    for f in reg23['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg24['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-24':
            assert b != a, ('FTR-24 must change',)
            fa = json.loads(a)
            assert f24['statement'].startswith(fa['statement']), 'statement not append-only'
            assert f24['counter_evidence'] == fa['counter_evidence'], 'ce changed'
            if gate_all:
                assert json.dumps(f24['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True), \
                    'anchors not prefix-preserved'
            else:
                assert f24['anchors'] == fa['anchors'], 'anchors changed on fail'
        else:
            assert a == b, ('feature not preserved', f['id'])
    for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_14.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg24, ensure_ascii=False, indent=1))
    reg24_sha = sha8_file(rp)
    log('registry v1.14 written sha8 %s (%s)' % (reg24_sha, disposition))

    meta_sha = {'reg24': reg24_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.15 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.15 &mdash; 语义机制图谱（registry v1.14 渲染）</h1>')
    H.append('<div class="sub">Phase 3187 G5-B12 &middot; R_rel 均值级 α 拟合定标（%s） &middot; '
             'registry v1.14 sha8=%s &middot; v1.13 sha8=%s &middot; v1.12 sha8=%s &middot; v1.11 sha8=%s &middot; v1.10 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3177 %s + p3178 %s + p3182 %s + p3183 %s + p3185 %s &middot; 渲染于 %s</div>' % (
                 'α̂ 定稿入锚' if gate_all else 'T2 降级方向级陈述',
                 span('meta.registry_v114_sha8', reg24_sha),
                 span('meta.registry_v113_sha8', '730729f5'),
                 span('meta.registry_v112_sha8', '269a37c0'),
                 span('meta.registry_v111_sha8', '84aecc21'),
                 span('meta.registry_v110_sha8', '4cf49743'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3177_result', SHA_ANCHOR['p3177_result']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3182_result', SHA_ANCHOR['p3182_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.p3185_result', SHA_ANCHOR['p3185_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg24['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg24['principles']):
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
    for f in reg24['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.14，%d 条）</h2>' % len(reg24['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg24['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg24['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg24['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.15 &middot; 渲染自 atlas_registry_v1_14.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3187 G5-B12 R_rel 均值级 α 拟合定标'
             '（数据源 p3157 collect %s / %s / %s + p3177 %s + p3178 %s + p3182 %s + p3183 %s + p3185 %s）'
             '&middot; 本文件为 Phase 3187 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg24_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3177_result'], SHA_ANCHOR['p3178_result'],
                 SHA_ANCHOR['p3182_result'], SHA_ANCHOR['p3183_result'],
                 SHA_ANCHOR['p3185_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_15.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v115(reg24, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_15.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3187, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_all': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg24['features']), 'failures': len(reg24['failures']),
                   'upgrades': len(reg24['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v114_file': 'atlas_registry_v1_14.json', 'registry_v114_sha8': reg24_sha,
        'html_file': 'atlas_v1_15.html', 'html_sha8': html_sha,
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


def flatten_v115(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v114_sha8'] = meta_sha['reg24']
    d['meta.registry_v113_sha8'] = '730729f5'
    d['meta.registry_v112_sha8'] = '269a37c0'
    d['meta.registry_v111_sha8'] = '84aecc21'
    d['meta.registry_v110_sha8'] = '4cf49743'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3177_result'] = SHA_ANCHOR['p3177_result']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.p3182_result'] = SHA_ANCHOR['p3182_result']
    d['meta.p3183_result'] = SHA_ANCHOR['p3183_result']
    d['meta.p3185_result'] = SHA_ANCHOR['p3185_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    global sealed_3177, sealed_3178, sealed_3182, sealed_3183, sealed_3185
    design_sha = freeze()
    log('== Phase 3187 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    for m in MODELS:
        ak = ANCH_KEY[m]
        r57 = json.load(open(SRC[RES_KEY[m]], encoding='utf-8'))
        A = ANCH_3157[ak]
        assert r57['res_sha8'] == A['res_sha8'] and r57['seal_sha8'] == A['seal_sha8'], \
            ('3157 content sha', m)
        assert r57['tc_mean_cos'] == A['tc_mean_cos'], ('3157 tc', m)
    log('3157 anchors double-locked (3 per-model)')
    sealed_3177 = json.load(open(SRC['p3177_result'], encoding='utf-8'))
    assert sealed_3177['res_sha8'] == CONTENT_SHA_3177[0] and sealed_3177['seal_sha8'] == CONTENT_SHA_3177[1]
    sealed_3178 = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert sealed_3178['res_sha8'] == CONTENT_SHA_3178[0] and sealed_3178['seal_sha8'] == CONTENT_SHA_3178[1]
    sealed_3182 = json.load(open(SRC['p3182_result'], encoding='utf-8'))
    assert sealed_3182['res_sha8'] == CONTENT_SHA_3182[0] and sealed_3182['seal_sha8'] == CONTENT_SHA_3182[1]
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    assert sealed_3183['res_sha8'] == CONTENT_SHA_3183[0] and sealed_3183['seal_sha8'] == CONTENT_SHA_3183[1]
    sealed_3185 = json.load(open(SRC['p3185_result'], encoding='utf-8'))
    assert sealed_3185['res_sha8'] == CONTENT_SHA_3185[0] and sealed_3185['seal_sha8'] == CONTENT_SHA_3185[1]
    log('sealed results double-locked: 3177 %s/%s 3178 %s/%s 3182 %s/%s 3183 %s/%s 3185 %s/%s'
        % (CONTENT_SHA_3177 + CONTENT_SHA_3178 + CONTENT_SHA_3182 +
           CONTENT_SHA_3183 + CONTENT_SHA_3185))

    # 3185 replication: alpha=1 prediction A readings vs sealed r_A
    out, gate_all, band_ratio = run_arm()
    # 3185 replication: LS algebra identity explained_rate == sealed r_A^2 (x6)
    for m in MODELS:
        for rname in ('isa', 'hasa'):
            E85 = sealed_3185['per_model'][m]['predA'][rname]['r_A']
            R2_87 = out[m]['alpha'][rname]['explained_rate']
            assert abs(R2_87 - E85 ** 2) < 1e-9, ('3185 r^2 replication', m, rname, R2_87, E85 ** 2)
    log('3185 replication: explained_rate == sealed r_A^2 (x6, <1e-9)')

    if gate_all:
        verdict = ('rrel_alpha_calibrated (sign-consistent x3, band ratio %.2f <= 3; '
                   'alpha_hat %s into FTR-24 anchor, T2 mean-level finalized: '
                   'Delta-M_rel = T_C + alpha*Delta-v, registry v1.13 -> v1.14)'
                   % (band_ratio, '/'.join('%.4f' % out[m]['alpha_m'] for m in MODELS)))
    else:
        verdict = ('rrel_alpha_unstable (sign/band gate fail, band ratio %.2f; T2 '
                   'downgraded to direction-level statement, FTR-24 statement '
                   'append, registry v1.13 -> v1.14)' % band_ratio)
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3187, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'band_max_min': G_BAND, 'alpha_sanity': ALPHA_SANITY,
                  'repl_tol': REPL_TOL, 'device_3157': True,
                  'band_ratio': band_ratio, 'gate_all': bool(gate_all)},
        'per_model': out,
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
        finish_registry(summary, gate_all, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
