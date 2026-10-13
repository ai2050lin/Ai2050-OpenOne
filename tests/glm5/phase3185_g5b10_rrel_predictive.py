# -*- coding: utf-8 -*-
"""Phase 3185 (G5-B10): R_rel low-rank model quantitative predictive test
(T2 main arm, zero GPU).

Per prereg in MEMO Phase 3184 tail, section "接续：预注册 3185" (frozen before
any observation):
  Test the predictive power of the FTR-24 low-rank geometry: parameterize
  R_rel by Delta-v = v_isa - v_hasa (sealed 3183/3184 direction, 3178 dC
  recipe verbatim) and predict, entity by entity:
  (a) the 3177 relation gap g_e (Delta-v projection prediction vs measured
      per-entity cos gap), and
  (b) the 3182 cross-null penalty p_x entity-level distribution (analytic
      v_x x Delta-v prediction vs measured p_x_fold).
  Freeze-time protocol pinning (recipes verbatim-exported from 3177/3182/
  3183/3184 scripts before any observation):
  - Panel: 64-cell dC panel (16 ent x 2 rel x 2 pol), dCn, tc_full device
    gate drift < 1e-9 vs sealed 3157 (3178 recipe verbatim).
  - Replication gates (all < 1e-9): 3177 folds cos_train_a2/cos_held_a2;
    3183 dv quantities (cos_rl, norm_dv, frac_full, pc_share, med_cos);
    3178 r_full; 3182 xnull p_x_mean per relation.
  - Prediction A (g_e, per held-relation fold):
      u_e  = mean of e's 2 held cells (unnormalized)
      vh_t = mean of all 32 held cells (unnormalized)
      v_c  = unit_mean(train 32 cells)          [3177 v_sh verbatim]
      g_e  = cos_train - (u_e @ v_c)            [measured; cos_train =
              mean(dCn_train @ v_c) = 3177 cos_train_a2 verbatim]
      dv   = v_isa - v_hasa; dvu = dv/(||dv||+1e-18)   [3183 verbatim]
      gam_e = (u_e - vh_t) @ dvu ; kap = dvu @ v_c
      g_pred_e = gbar - kap * gam_e,  gbar = cos_train - vh_t @ v_c
      (exact identities: mean(g_e) = gbar and mean(gam_e) = 0, so the
       analytic prediction preserves the fold mean by construction).
      r_A = Pearson corr(g_e, g_pred_e);
      R2_A = 1 - SS_res/SS_tot (fit-free: zero free parameters).
  - Prediction B (p_x, per held relation, 16 LORO folds verbatim 3182):
      v_x(e) = unit_mean(calib 30 rows minus e); p_x(e) = cos_c(e) - cos_h(e)
      p_pred(e) = b_e - gam_e * kap_x(e),  b_e = cos_c(e) - vh_t @ v_x(e),
      kap_x(e) = dvu @ v_x(e).   (fit-free)
      R2_B = 1 - SS_res/SS_tot (centered on measured mean); r_B report-only.
  - Per model: r_g(m) = mean(r_A over isa/hasa folds); r2_px(m) = mean(R2_B).
  - GATE (pinned pre-observation): R_rel low-rank parameterization CONFIRMED
    iff r_g(m) >= 0.5 x3 models AND r2_px(m) >= 0.5 x3 models =>
    FTR-24 predictive anchor appended + T2 mainline pinned in statement
    (registry v1.11 -> v1.12, other 23 features byte-preserved, failures /
    upgrade_log untouched), atlas_v1_13.html.  Otherwise => linear
    superposition rejected: FTR-24 counter_evidence append + F19
    (rrel_linear_superposition_rejected) into failures (append-only),
    T2 route revised (nonlinear superposition).
  - Residual spectrum: per fold per-entity residuals, max-abs entity;
    report-only cross-check vs 3184 outlier entity e=12 (qwen3-14b).
  - SMOKE (env P3185_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
NAME = 'g5b10_rrel_predictive'
OUTDIR = os.path.join(SRC_DIR, 'phase3185', NAME)
SMOKE = os.environ.get('P3185_SMOKE') == '1'
LOG = []
N_E = 16
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}

G_RG = 0.5
G_R2 = 0.5
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'r_g_min': G_RG, 'r2_px_min': G_R2, 'repl_tol': REPL_TOL, 'dev_tol': DEV_TOL}

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
    'p3184_result': os.path.join(SRC_DIR, r'phase3184\g5b9_14b_marginal_loo\result.json'),
    'registry_v111': os.path.join(SRC_DIR, r'phase3184\g5b9_14b_marginal_loo\atlas_registry_v1_11.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a', 'p3177_result': 'b2ad52ef', 'p3178_result': 'abae296e',
    'p3182_result': '15b16f37', 'p3183_result': '0ab70579', 'p3184_result': 'c88898a0',
    'registry_v111': '84aecc21', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_SHA_3177 = ('d451030f', '7b746eb2')
CONTENT_SHA_3178 = ('4f521d9b', '1f166645')
CONTENT_SHA_3182 = ('33d73196', 'ed51ca22')
CONTENT_SHA_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_SHA_3184 = ('54e2b9d9', '55706292')

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, res_sha8='c0db3455', seal_sha8='41916ed8',
                     tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, res_sha8='1270c921', seal_sha8='98c06ab0',
                      tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, res_sha8='a1d9a98a', seal_sha8='e8be1026',
                 tc_mean_cos=0.5640953183174133),
}

DESIGN = {
    'phase': '3185', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3184 tail G5-B10 (frozen before any observation)',
    'data': 'sealed 3157 collect npz x3 + sealed 3177/3178/3182/3183/3184 result.json; '
            'zero GPU',
    'parameterization': 'R_rel = alpha * Delta-v; dv = v_isa - v_hasa (3183 verbatim: '
                        'unit_mean over each relation own 32 cells); dvu = unit(dv)',
    'prediction_A': 'per held-relation fold: g_e = cos_train - u_e@v_c (3177 verbatim '
                    'components); analytic fit-free prediction g_pred = gbar - kap*gam_e '
                    'with gam_e = (u_e - vh_t)@dvu, kap = dvu@v_c, gbar = cos_train - '
                    'vh_t@v_c; identities mean(g_e)=gbar, mean(gam_e)=0 exact',
    'prediction_B': 'per held relation, 16 LORO folds verbatim 3182 (v_x = unit_mean '
                    'calib 30 rows minus e): p_pred(e) = b_e - gam_e*kap_x(e), b_e = '
                    'cos_c(e) - vh_t@v_x(e), kap_x(e) = dvu@v_x(e); fit-free',
    'measurements': 'r_A = corr(g_e, g_pred); R2_A fit-free; R2_B fit-free centered; '
                    'r_B report-only; residual spectrum per fold (max-abs entity, '
                    'report-only cross-check vs 3184 e=12)',
    'gate': 'R_rel low-rank parameterization CONFIRMED iff r_g >= 0.5 x3 AND r2_px '
            '>= 0.5 x3 => FTR-24 predictive anchor append + T2 mainline statement '
            'append (registry v1.11->v1.12, other features byte-preserved, failures/'
            'upgrade_log untouched, atlas_v1_13.html); else F19 '
            'rrel_linear_superposition_rejected + FTR-24 counter_evidence append, '
            'T2 nonlinear-superposition revision',
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
        json.dump({'phase': 3185, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


# ----------------- verbatim helpers (3178/3182/3183) -----------------
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


def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    sa, sb = a.std(), b.std()
    if sa < 1e-300 or sb < 1e-300:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def r2(meas, pred):
    meas = np.asarray(meas, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(((meas - pred) ** 2).sum())
    ss_tot = float(((meas - meas.mean()) ** 2).sum())
    if ss_tot < 1e-300:
        return float('nan')
    return 1.0 - ss_res / ss_tot


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

        # device gate: tc_full recompute verbatim (3178 recipe)
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

        # 3183 replication: dv quantities verbatim
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
        s = np.linalg.svd(U, compute_uv=False)
        pc_share = float(s[0] ** 2 / (s ** 2).sum())
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
        C = Un @ Un.T
        med_cos = float(np.median(C[np.triu_indices(N_E, 1)]))
        E83 = sealed_3183['per_model'][m]
        d83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']),
                  abs(frac_full - E83['frac_full']), abs(pc_share - E83['pc_share']),
                  abs(med_cos - E83['med_cos']))
        assert d83 < REPL_TOL, ('3183 dv replication', m, d83)
        dvu = unit(dv)

        # 3177 folds replication (verbatim components) + Prediction A
        predA = {}
        folds_rep = {}
        for rl_h, fname in ((0, 'isa'), (1, 'hasa')):
            rows_h = dC_rows(rl_h)
            rows_c = dC_rows(1 - rl_h)
            v_c = unit_mean(dCn[rows_c])
            cos_train = float((dCn[rows_c] @ v_c).mean())
            cos_held = float((dCn[rows_h] @ v_c).mean())
            E77 = sealed_3177['arm']['per_model'][m]['folds'][fname]
            d77 = max(abs(cos_train - E77['cos_train_a2']),
                      abs(cos_held - E77['cos_held_a2']))
            assert d77 < REPL_TOL, ('3177 fold replication', m, fname, d77)
            folds_rep[fname] = dict(cos_train_a2=cos_train, cos_held_a2=cos_held,
                                    repl_drift=d77)

            vh_t = dCn[rows_h].mean(0)              # unnormalized held mean
            gbar = cos_train - float(vh_t @ v_c)
            gam = np.zeros(N_E)
            g_meas = np.zeros(N_E)
            g_pred = np.zeros(N_E)
            for e in range(N_E):
                u_e = dCn[[e * 4 + rl_h * 2 + p for p in (0, 1)]].mean(0)
                gam[e] = float((u_e - vh_t) @ dvu)
                g_meas[e] = cos_train - float(u_e @ v_c)
                g_pred[e] = gbar - float(dvu @ v_c) * gam[e]
            # exact identities
            assert abs(g_meas.mean() - gbar) < 1e-6, ('gbar identity', m, fname)
            assert abs(gam.mean()) < 1e-6, ('gam zero-mean identity', m, fname)
            r_A = corr(g_meas, g_pred)
            R2_A = r2(g_meas, g_pred)
            res_A = g_meas - g_pred
            e_max = int(np.abs(res_A).argmax())
            predA[fname] = dict(r_A=r_A, R2_A=R2_A, gbar=gbar,
                                kap=float(dvu @ v_c),
                                gam_mean=float(gam.mean()),
                                resid_max=float(np.abs(res_A).max()),
                                resid_max_entity=e_max,
                                resid_rms=float(np.sqrt((res_A ** 2).mean())),
                                g_meas=[float(x) for x in g_meas],
                                g_pred=[float(x) for x in g_pred])
            log('  A %s fold(%s): r=%.4f R2=%.4f kap=%.4f resid_max=%.4f@e%d'
                % (m, fname, r_A, R2_A, predA[fname]['kap'],
                   predA[fname]['resid_max'], e_max))

        # 3178 r_full replication (verbatim 3182 path)
        nulls = {}
        for rl, rname in ((0, 'isa'), (1, 'hasa')):
            rows_all = dC_rows(rl)
            pf = []
            for e in range(N_E):
                held_rows = [e * 4 + rl * 2 + p for p in (0, 1)]
                train_rows = [r for r in rows_all if r not in held_rows]
                v_null = unit_mean(dCn[train_rows])
                cos_in = float((dCn[train_rows] @ v_null).mean())
                cos_out = float((dCn[held_rows] @ v_null).mean())
                pf.append(dict(e=e, p_fold=cos_in - cos_out))
            nulls[rname] = float(np.mean([x['p_fold'] for x in pf]))
        r_full = float(sum((folds_rep[r]['cos_train_a2'] - folds_rep[r]['cos_held_a2']
                            - nulls[r]) for r in ('isa', 'hasa')) / 2.0)
        d78 = abs(r_full - sealed_3178['arm']['per_model'][m]['r_m'])
        assert d78 < REPL_TOL, ('3178 r_full replication', m, d78)

        # 3182 LORO xnull verbatim recompute + Prediction B
        predB = {}
        px_rep = {}
        for rl_h, rname_h in ((0, 'isa'), (1, 'hasa')):
            rl_c = 1 - rl_h
            rows_c_all = dC_rows(rl_c)
            rows_h = dC_rows(rl_h)
            vh_t = dCn[rows_h].mean(0)
            p_meas = np.zeros(N_E)
            p_pred = np.zeros(N_E)
            pxf = []
            for e in range(N_E):
                calib_drop = [e * 4 + rl_c * 2 + p for p in (0, 1)]
                calib_rows = [r for r in rows_c_all if r not in calib_drop]
                assert len(calib_rows) == 30, ('xnull calib size', m, rname_h, e)
                v_x = unit_mean(dCn[calib_rows])
                read_rows = [e * 4 + rl_h * 2 + p for p in (0, 1)]
                cos_c = float((dCn[calib_rows] @ v_x).mean())
                cos_h = float((dCn[read_rows] @ v_x).mean())
                u_e = dCn[read_rows].mean(0)
                gam_e = float((u_e - vh_t) @ dvu)
                p_meas[e] = cos_c - cos_h
                p_pred[e] = (cos_c - float(vh_t @ v_x)) - gam_e * float(dvu @ v_x)
                pxf.append(dict(e=e, p_x_fold=cos_c - cos_h))
            assert max(abs(x['p_x_fold']) for x in pxf) < 0.5, ('xnull sanity', m, rname_h)
            E82 = sealed_3182['residual_detail'][m]['resid_held'][rname_h]['p_x']
            p_rep = abs(float(np.mean([x['p_x_fold'] for x in pxf])) - E82)
            assert p_rep < REPL_TOL, ('3182 p_x replication', m, rname_h, p_rep)
            px_rep[rname_h] = p_rep
            R2_B = r2(p_meas, p_pred)
            r_B = corr(p_meas, p_pred)
            res_B = p_meas - p_pred
            e_max = int(np.abs(res_B).argmax())
            predB[rname_h] = dict(R2_B=R2_B, r_B=r_B,
                                  resid_max=float(np.abs(res_B).max()),
                                  resid_max_entity=e_max,
                                  resid_rms=float(np.sqrt((res_B ** 2).mean())),
                                  p_meas=[float(x) for x in p_meas],
                                  p_pred=[float(x) for x in p_pred])
            log('  B %s rel(%s): R2=%.4f r=%.4f resid_max=%.4f@e%d (p_x rep %.2e)'
                % (m, rname_h, R2_B, r_B, predB[rname_h]['resid_max'], e_max, p_rep))

        r_g = float((predA['isa']['r_A'] + predA['hasa']['r_A']) / 2.0)
        r2_px = float((predB['isa']['R2_B'] + predB['hasa']['R2_B']) / 2.0)
        gate_a = bool(r_g >= G_RG)
        gate_b = bool(r2_px >= G_R2)
        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full,
                      dv_rep=dict(cos_rl=cos_rl, norm_dv=norm_dv, frac_full=frac_full,
                                  pc_share=pc_share, med_cos=med_cos, drift=d83),
                      folds_rep=folds_rep, r_full_rep=r_full, r_full_drift=d78,
                      px_rep=px_rep, predA=predA, predB=predB,
                      r_g=r_g, r2_px=r2_px, gate_a=gate_a, gate_b=gate_b)
        log('%s: r_g=%.4f (gate %.2f: %s) | r2_px=%.4f (gate %.2f: %s)'
            % (m, r_g, G_RG, 'PASS' if gate_a else 'FAIL',
               r2_px, G_R2, 'PASS' if gate_b else 'FAIL'))
    gate_all = all(out[m]['gate_a'] and out[m]['gate_b'] for m in MODELS)
    log('GATE R_rel low-rank parameterization (r_g>=%.2f AND r2_px>=%.2f x3): %s'
        % (G_RG, G_R2, gate_all))
    return out, gate_all


# ----------------- registry v1.11 -> v1.12 + atlas_v1_13 -----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span


def finish_registry(summary, gate_all, res8, seal8, design_sha):
    reg21 = json.loads(open(SRC['registry_v111'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg21['version'] == '1.11' and len(reg21['features']) == 24
    assert len(reg21['failures']) == 18 and len(reg21['upgrade_log']) == 6
    reg22 = copy.deepcopy(reg21)
    reg22['version'] = '1.12'
    reg22['supersedes'] = 'atlas_registry_v1_11.json (84aecc21, Phase 3184)'
    pm = summary['per_model']
    fmt3 = lambda k: '/'.join('%.4f' % pm[m][k] for m in MODELS)
    f24_old = [f for f in reg21['features'] if f['id'] == 'FTR-24'][0]
    if gate_all:
        anchor = {
            'src': 'p3185', 'phase': 3185,
            'role': 'rrel_lowrank_predictive',
            'tags': ['cross_model'],
            'asserts': {
                'r_g.qwen3-4b': pm['qwen3-4b']['r_g'],
                'r_g.qwen3-14b': pm['qwen3-14b']['r_g'],
                'r_g.glm4-9b': pm['glm4-9b']['r_g'],
                'r2_px.qwen3-4b': pm['qwen3-4b']['r2_px'],
                'r2_px.qwen3-14b': pm['qwen3-14b']['r2_px'],
                'r2_px.glm4-9b': pm['glm4-9b']['r2_px'],
                'gate_r_g_min': G_RG, 'gate_r2_px_min': G_R2,
                'res_sha8': res8, 'seal_sha8': seal8}},
        f24 = copy.deepcopy(f24_old)
        f24['anchors'] = f24_old['anchors'] + [anchor]
        f24['statement'] = f24_old['statement'] + (
            '；3185 G5-B10 定量检验通过：以 Δv 为 R_rel 参数化的免拟合解析预测对 '
            '3177 关系缺口 g_e（r_g ' + fmt3('r_g') + '）与 3182 cross-null 罚 p_x '
            '实体级分布（R²_px ' + fmt3('r2_px') + '）双双过门（0.5/0.5 ×3 模型）'
            '——T2 主线确立：ΔM_rel = T_C + αΔv，α 参数拟合入下一 phase。')
        reg22['features'] = [f24 if f['id'] == 'FTR-24' else f for f in reg21['features']]
        disposition = 'v1.11_to_v1.12_ftr24_predictive_anchor'
    else:
        ce = ('3185 低秩参数化预测检验定判 FAIL：免拟合解析预测对 3177 关系缺口 g_e '
              '（r_g ' + fmt3('r_g') + '）与 3182 cross-null 罚 p_x 实体级分布 '
              '（R²_px ' + fmt3('r2_px') + '）未双双过门（0.5/0.5 ×3）——R_rel 低秩'
              '但非线性叠加，T2 路线修正；F19 入失败账本。')
        f24 = copy.deepcopy(f24_old)
        f24['counter_evidence'] = f24_old['counter_evidence'] + [ce]
        reg22['features'] = [f24 if f['id'] == 'FTR-24' else f for f in reg21['features']]
        reg22['failures'] = reg21['failures'] + [{
            'id': 'F19', 'kind': 'rrel_linear_superposition_rejected', 'phase': 3185,
            'text': 'fit-free analytic Delta-v parameterization fails to predict '
                    'entity-level 3177 gap and/or 3182 cross-null p_x distribution: '
                    'R_rel low-rank direction exists but superposition is nonlinear',
            'evidence': 'p3185 result %s/seal %s' % (res8, seal8)}]
        disposition = 'v1.11_to_v1.12_f19_nonlinear'
    # preservation asserts
    for f in reg21['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg22['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-24':
            if gate_all:
                assert b != a, ('FTR-24 must change on pass',)
                fa = json.loads(a)
                assert json.dumps(f24['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True), \
                    'FTR-24 anchors not prefix-preserved'
                assert f24['statement'].startswith(fa['statement']), 'statement not append-only'
            else:
                assert b != a, ('FTR-24 must change on fail',)
                fa = json.loads(a)
                assert f24['counter_evidence'][:-1] == fa['counter_evidence'], \
                    'counter_evidence not append-only'
        else:
            assert a == b, ('feature not preserved', f['id'])
    for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg21.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg22.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    if gate_all:
        assert json.dumps(reg21['failures'], ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg22['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    else:
        assert len(reg22['failures']) == 19
        assert json.dumps(reg21['failures'], ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg22['failures'][:18], ensure_ascii=False, sort_keys=True), \
            'failures not prefix-preserved'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_12.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg22, ensure_ascii=False, indent=1))
    reg22_sha = sha8_file(rp)
    log('registry v1.12 written sha8 %s (%s)' % (reg22_sha, disposition))

    meta_sha = {'reg22': reg22_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.13 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.13 &mdash; 语义机制图谱（registry v1.12 渲染）</h1>')
    H.append('<div class="sub">Phase 3185 G5-B10 &middot; R_rel 低秩模型定量检验（%s） &middot; '
             'registry v1.12 sha8=%s &middot; v1.11 sha8=%s &middot; v1.10 sha8=%s &middot; v1.9 sha8=%s &middot; v1.8 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3177 %s + p3178 %s + p3182 %s + p3183 %s + p3184 %s &middot; 渲染于 %s</div>' % (
                 'FTR-24 预测锚追加，T2 主线确立' if gate_all else 'F19 非线性叠加，T2 修正',
                 span('meta.registry_v112_sha8', reg22_sha),
                 span('meta.registry_v111_sha8', '84aecc21'),
                 span('meta.registry_v110_sha8', '4cf49743'),
                 span('meta.registry_v19_sha8', '59a91512'),
                 span('meta.registry_v18_sha8', 'd08e9064'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3177_result', SHA_ANCHOR['p3177_result']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3182_result', SHA_ANCHOR['p3182_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.p3184_result', SHA_ANCHOR['p3184_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg22['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg22['principles']):
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
    for f in reg22['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.12，%d 条）</h2>' % len(reg22['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg22['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg22['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg22['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.13 &middot; 渲染自 atlas_registry_v1_12.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3185 G5-B10 R_rel 低秩模型定量检验'
             '（数据源 p3157 collect %s / %s / %s + p3177 %s + p3178 %s + p3182 %s + p3183 %s + p3184 %s）'
             '&middot; 本文件为 Phase 3185 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg22_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3177_result'], SHA_ANCHOR['p3178_result'],
                 SHA_ANCHOR['p3182_result'], SHA_ANCHOR['p3183_result'],
                 SHA_ANCHOR['p3184_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_13.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v113(reg22, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_13.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3185, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_all': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg22['features']), 'failures': len(reg22['failures']),
                   'upgrades': len(reg22['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v112_file': 'atlas_registry_v1_12.json', 'registry_v112_sha8': reg22_sha,
        'html_file': 'atlas_v1_13.html', 'html_sha8': html_sha,
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


def flatten_v113(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v112_sha8'] = meta_sha['reg22']
    d['meta.registry_v111_sha8'] = '84aecc21'
    d['meta.registry_v110_sha8'] = '4cf49743'
    d['meta.registry_v19_sha8'] = '59a91512'
    d['meta.registry_v18_sha8'] = 'd08e9064'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3177_result'] = SHA_ANCHOR['p3177_result']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.p3182_result'] = SHA_ANCHOR['p3182_result']
    d['meta.p3183_result'] = SHA_ANCHOR['p3183_result']
    d['meta.p3184_result'] = SHA_ANCHOR['p3184_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    global sealed_3177, sealed_3178, sealed_3182, sealed_3183
    design_sha = freeze()
    log('== Phase 3185 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
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
    log('3157 anchors double-locked (3 per-model)')
    # double-lock sealed results 3177/3178/3182/3183/3184
    sealed_3177 = json.load(open(SRC['p3177_result'], encoding='utf-8'))
    assert sealed_3177['res_sha8'] == CONTENT_SHA_3177[0] and sealed_3177['seal_sha8'] == CONTENT_SHA_3177[1]
    sealed_3178 = json.load(open(SRC['p3178_result'], encoding='utf-8'))
    assert sealed_3178['res_sha8'] == CONTENT_SHA_3178[0] and sealed_3178['seal_sha8'] == CONTENT_SHA_3178[1]
    sealed_3182 = json.load(open(SRC['p3182_result'], encoding='utf-8'))
    assert sealed_3182['res_sha8'] == CONTENT_SHA_3182[0] and sealed_3182['seal_sha8'] == CONTENT_SHA_3182[1]
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    assert sealed_3183['res_sha8'] == CONTENT_SHA_3183[0] and sealed_3183['seal_sha8'] == CONTENT_SHA_3183[1]
    sealed_3184 = json.load(open(SRC['p3184_result'], encoding='utf-8'))
    assert sealed_3184['res_sha8'] == CONTENT_SHA_3184[0] and sealed_3184['seal_sha8'] == CONTENT_SHA_3184[1]
    log('sealed results double-locked: 3177 %s/%s 3178 %s/%s 3182 %s/%s 3183 %s/%s 3184 %s/%s'
        % (CONTENT_SHA_3177 + CONTENT_SHA_3178 + CONTENT_SHA_3182 +
           CONTENT_SHA_3183 + CONTENT_SHA_3184))

    out, gate_all = run_arm()
    if gate_all:
        verdict = ('rrel_lowrank_parameterization_confirmed (r_g/r2_px gates x3; '
                   'FTR-24 predictive anchor appended, T2 mainline pinned: '
                   'Delta-M_rel = T_C + alpha*Delta-v, registry v1.11 -> v1.12)')
    else:
        verdict = ('rrel_linear_superposition_rejected (F19 registered; FTR-24 '
                   'counter_evidence append; T2 nonlinear-superposition revision)')
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3185, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'r_g_min': G_RG, 'r2_px_min': G_R2, 'repl_tol': REPL_TOL,
                  'device_3157': True, 'gate_all': bool(gate_all)},
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
