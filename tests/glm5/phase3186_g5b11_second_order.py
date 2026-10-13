# -*- coding: utf-8 -*-
"""Phase 3186 (G5-B11): relation-offset second-order structure test
(T2 revision arm, zero GPU).

Per prereg in MEMO Phase 3185 tail, section "接续：预注册 3186" (frozen before
any observation):
  Attribute the F19 failure: single-direction Delta-v explains <30% variance
  and residuals concentrate on few entities (e15 panel-wide + 14b e12).
  Freeze-time protocol pinning (recipes verbatim from 3183/3185):
  - Panel + replication gates identical to 3185 (3157 tc device gate <1e-9;
    3177 folds / 3178 r_full / 3182 p_x / 3183 dv quantities <1e-9).
  - Arm A (spectrum): per held relation rl, entity deviation vectors
    delta_e = u_e - vh_t (u_e = mean of e's 2 cells, vh_t = unnormalized
    32-cell mean; sum_e delta_e = 0 exactly, 15-dim subspace), 16 vectors;
    SVD energy shares: top-1, top-2 ((s1^2+s2^2)/sum(s^2)), rank for 80% /
    90% energy; per-model stat = mean over isa/hasa.
    GATE (pinned pre-observation): top-2 energy >= 0.5 x3 models =>
    plane parameterization T2' (Delta-M_rel = T_C + alpha*Delta-v + beta*w2)
    registered as candidate: FTR-24 anchor append + T2' statement append
    (registry v1.12 -> v1.13, 23 legacy byte-preserved), atlas_v1_14.html.
    Otherwise (full-rank diffuse) => mean-level contraction: FTR-24
    counter_evidence append + statement contraction note, no new failure id
    (F19 already covers entity-level linear rejection).
  - Arm B (outlier LOO attribution, 3184 protocol): 16 folds, each removing
    one entity e (held cells dropped from vh_t -> 30-cell mean; prediction
    evaluation on remaining 15 entities); recompute r_g and R2_px per fold;
    register per model: LOO min/max, flip entity ids (post-removal BOTH
    r_g >= 0.5 AND R2_px >= 0.5), and whether e15 (4b/glm4 residual max)
    / e12 (14b hasa residual max, 3184 outlier) is the flip entity.
    Report-only for gate; attributive.
  - SMOKE (env P3186_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
NAME = 'g5b11_second_order'
OUTDIR = os.path.join(SRC_DIR, 'phase3186', NAME)
SMOKE = os.environ.get('P3186_SMOKE') == '1'
LOG = []
N_E = 16
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
NPZ_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}

G_T2 = 0.5
G_RG = 0.5
G_R2 = 0.5
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'top2_energy_min': G_T2, 'r_g_min': G_RG, 'r2_px_min': G_R2,
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
    'p3184_result': os.path.join(SRC_DIR, r'phase3184\g5b9_14b_marginal_loo\result.json'),
    'p3185_result': os.path.join(SRC_DIR, r'phase3185\g5b10_rrel_predictive\result.json'),
    'registry_v112': os.path.join(SRC_DIR, r'phase3185\g5b10_rrel_predictive\atlas_registry_v1_12.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3157_summary': '71959b6a', 'p3177_result': 'b2ad52ef', 'p3178_result': 'abae296e',
    'p3182_result': '15b16f37', 'p3183_result': '0ab70579', 'p3184_result': 'c88898a0',
    'p3185_result': '386f6197', 'registry_v112': '269a37c0', 'gap_v15': '4fb3f37d',
    'registry_3162': '00f15e98',
}
CONTENT_SHA_3177 = ('d451030f', '7b746eb2')
CONTENT_SHA_3178 = ('4f521d9b', '1f166645')
CONTENT_SHA_3182 = ('33d73196', 'ed51ca22')
CONTENT_SHA_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_SHA_3184 = ('54e2b9d9', '55706292')
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
    'phase': '3186', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3185 tail G5-B11 (frozen before any observation)',
    'data': 'sealed 3157 collect npz x3 + sealed 3177/3178/3182/3183/3184/3185 '
            'result.json; zero GPU',
    'arm_A': 'per held relation: delta_e = u_e - vh_t (sum-zero, 15-dim), 16 vectors; '
             'SVD top-1/top-2 energy shares + rank for 80%/90%; per-model stat = '
             'mean over isa/hasa',
    'arm_B': 'outlier LOO (3184 protocol): 16 folds drop one entity (vh_t -> 30-cell '
             'mean, evaluation on remaining 15); recompute r_g/R2_px; register LOO '
             'min/max + flip entities (post-removal BOTH gates pass) + e15/e12 '
             'attribution; report-only for gate',
    'gate': 'top-2 SVD energy >= 0.5 x3 models => plane parameterization T2\' '
            '(alpha*Delta-v + beta*w2) candidate: FTR-24 anchor append + T2\' '
            'statement append (registry v1.12->v1.13, 23 legacy byte-preserved, '
            'atlas_v1_14.html); else full-rank diffuse => mean-level contraction: '
            'FTR-24 counter_evidence append + statement note, no new failure id '
            '(F19 covers entity-level linear rejection)',
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
        json.dump({'phase': 3186, 'name': NAME, 'design_sha8': dsha[:8],
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


def xnull_folds(dCn, rl_h, rl_c=None):
    """3182 LORO cross-null verbatim: per-entity folds."""
    if rl_c is None:
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


def rg_r2_for(dCn, dvu, rl_h, drop_e=None):
    """3185 predictions A+B for held relation rl_h, optionally dropping one
    entity from baseline AND evaluation (LOO)."""
    rows_h = dC_rows(rl_h)
    rows_c = dC_rows(1 - rl_h)
    v_c = unit_mean(dCn[rows_c])
    cos_train = float((dCn[rows_c] @ v_c).mean())
    if drop_e is None:
        keep = list(range(N_E))
        vh_t = dCn[rows_h].mean(0)
    else:
        keep = [e for e in range(N_E) if e != drop_e]
        keep_rows = [r for r in rows_h
                     if r // 4 != drop_e]
        vh_t = dCn[keep_rows].mean(0)
    gbar = cos_train - float(vh_t @ v_c)
    kap = float(dvu @ v_c)
    gam = np.zeros(len(keep))
    g_meas = np.zeros(len(keep))
    g_pred = np.zeros(len(keep))
    for i, e in enumerate(keep):
        u_e = dCn[[e * 4 + rl_h * 2 + p for p in (0, 1)]].mean(0)
        gam[i] = float((u_e - vh_t) @ dvu)
        g_meas[i] = cos_train - float(u_e @ v_c)
        g_pred[i] = gbar - kap * gam[i]
    r_A = corr(g_meas, g_pred)

    # prediction B: 3182 xnull verbatim (v_x always full-panel per-fold; LOO
    # only affects the held-relation baseline/evaluation side)
    pxf = xnull_folds(dCn, rl_h)
    p_meas = np.zeros(len(keep))
    p_pred = np.zeros(len(keep))
    for i, e in enumerate(keep):
        x = pxf[e]
        u_e = dCn[[e * 4 + rl_h * 2 + p for p in (0, 1)]].mean(0)
        gam_e = float((u_e - vh_t) @ dvu)
        p_meas[i] = x['p_x_fold']
        p_pred[i] = (x['cos_calib'] - float(vh_t @ x['v_x'])) - gam_e * float(dvu @ x['v_x'])
    R2_B = r2(p_meas, p_pred)
    return r_A, R2_B


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

        # 3177/3178/3182 replications (compact, verbatim)
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

        # ---- Arm A: delta_e SVD spectrum per relation ----
        specA = {}
        for rl_h, rname in ((0, 'isa'), (1, 'hasa')):
            rows_h = dC_rows(rl_h)
            vh_t = dCn[rows_h].mean(0)
            DEL = np.stack([dCn[[e * 4 + rl_h * 2 + p for p in (0, 1)]].mean(0) - vh_t
                            for e in range(N_E)])
            assert np.isfinite(DEL).all(), ('nonfinite DEL', m, rname)
            # sum-to-zero identity (float32 tol)
            assert float(np.abs(DEL.sum(0)).max()) < 1e-6, ('sum-zero', m, rname)
            s = np.linalg.svd(DEL, compute_uv=False)
            e_sq = (s ** 2)
            tot = float(e_sq.sum())
            top1 = float(e_sq[0] / tot)
            top2 = float(e_sq[:2].sum() / tot)
            cum = np.cumsum(e_sq) / tot
            r80 = int(np.searchsorted(cum, 0.8) + 1)
            r90 = int(np.searchsorted(cum, 0.9) + 1)
            specA[rname] = dict(top1=top1, top2=top2, r80=r80, r90=r90)
            log('  A %s rel(%s): top1=%.4f top2=%.4f r80=%d r90=%d'
                % (m, rname, top1, top2, r80, r90))
        top2_mean = float((specA['isa']['top2'] + specA['hasa']['top2']) / 2.0)

        # ---- Arm B: outlier LOO (3184 protocol on 3185 predictions) ----
        looB = {}
        for rl_h, rname in ((0, 'isa'), (1, 'hasa')):
            folds = []
            for e in range(N_E):
                r_A, R2_B = rg_r2_for(dCn, dvu, rl_h, drop_e=e)
                folds.append(dict(e=e, r_A=r_A, R2_B=R2_B))
            rg_f = [f['r_A'] for f in folds]
            r2_f = [f['R2_B'] for f in folds]
            flips = [f['e'] for f in folds
                     if f['r_A'] >= G_RG and f['R2_B'] >= G_R2]
            looB[rname] = dict(folds=folds,
                               r_min=float(min(rg_f)), r_max=float(max(rg_f)),
                               r2_min=float(min(r2_f)), r2_max=float(max(r2_f)),
                               flip_entities=flips)
            log('  B %s rel(%s): r_g LOO %.4f..%.4f | R2 LOO %.4f..%.4f | flips=%s'
                % (m, rname, min(rg_f), max(rg_f), min(r2_f), max(r2_f),
                   flips if flips else 'none'))
        # full-panel 3185 values for reference
        rA_full = {}
        R2_full = {}
        for rl_h, rname in ((0, 'isa'), (1, 'hasa')):
            r_A, R2_B = rg_r2_for(dCn, dvu, rl_h, drop_e=None)
            rA_full[rname] = r_A
            R2_full[rname] = R2_B
        E85 = sealed_3185['per_model'][m]
        d85 = max(abs(rA_full['isa'] - E85['predA']['isa']['r_A']),
                  abs(rA_full['hasa'] - E85['predA']['hasa']['r_A']),
                  abs(R2_full['isa'] - E85['predB']['isa']['R2_B']),
                  abs(R2_full['hasa'] - E85['predB']['hasa']['R2_B']))
        assert d85 < REPL_TOL, ('3185 full-panel replication', m, d85)

        gate_a = bool(top2_mean >= G_T2)
        out[m] = dict(kout=KOUT, nl=NL, tc_full=tc_full,
                      dv_rep=dict(cos_rl=cos_rl, norm_dv=norm_dv, frac_full=frac_full,
                                  pc_share=pc_share, med_cos=med_cos, drift=d83),
                      folds_rep=folds_rep, r_full_rep=r_full, r_full_drift=d78,
                      px_rep=px_rep, specA=specA, top2_mean=top2_mean,
                      gate_a=gate_a, looB=looB,
                      rA_full=rA_full, R2_full=R2_full, repl_3185=d85)
        log('%s: top2_energy=%.4f (gate %.2f: %s) | rA full %.4f/%.4f R2 full %.4f/%.4f'
            % (m, top2_mean, G_T2, 'PASS' if gate_a else 'FAIL',
               rA_full['isa'], rA_full['hasa'], R2_full['isa'], R2_full['hasa']))
    gate_all = all(out[m]['gate_a'] for m in MODELS)
    log('GATE plane-parameterization candidate (top2 >= %.2f x3): %s' % (G_T2, gate_all))
    return out, gate_all


# ----------------- registry v1.12 -> v1.13 + atlas_v1_14 -----------------
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
span = p79.span


def finish_registry(summary, gate_all, res8, seal8, design_sha):
    reg22 = json.loads(open(SRC['registry_v112'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg22['version'] == '1.12' and len(reg22['features']) == 24
    assert len(reg22['failures']) == 19 and len(reg22['upgrade_log']) == 6
    reg23 = copy.deepcopy(reg22)
    reg23['version'] = '1.13'
    reg23['supersedes'] = 'atlas_registry_v1_12.json (269a37c0, Phase 3185)'
    pm = summary['per_model']
    fmt3 = lambda k: '/'.join('%.4f' % pm[m][k] for m in MODELS)
    f24_old = [f for f in reg22['features'] if f['id'] == 'FTR-24'][0]
    f24 = copy.deepcopy(f24_old)
    if gate_all:
        anchor = {
            'src': 'p3186', 'phase': 3186,
            'role': 'rrel_second_order_spectrum',
            'tags': ['cross_model'],
            'asserts': {
                'top2_energy.qwen3-4b': pm['qwen3-4b']['top2_mean'],
                'top2_energy.qwen3-14b': pm['qwen3-14b']['top2_mean'],
                'top2_energy.glm4-9b': pm['glm4-9b']['top2_mean'],
                'gate_top2_min': G_T2,
                'res_sha8': res8, 'seal_sha8': seal8}}
        f24['anchors'] = f24_old['anchors'] + [anchor]
        f24['statement'] = f24_old['statement'] + (
            '；3186 G5-B11 二阶谱检验通过：实体级偏离 δ_e 的 SVD 前 2 主成分能量占比 '
            + fmt3('top2_mean') + '（≥门 0.5 ×3 模型）——偏离集中于秩-2 平面，'
            '平面参数化 T2′（ΔM_rel = T_C + αΔv + βw₂）立项候选。')
        reg23['features'] = [f24 if f['id'] == 'FTR-24' else f for f in reg22['features']]
        disposition = 'v1.12_to_v1.13_ftr24_plane_candidate'
    else:
        ce = ('3186 二阶谱检验定判 DIFFUSE：实体级偏离 δ_e 的 SVD 前 2 主成分能量占比 '
              + fmt3('top2_mean') + '（<门 0.5 ×3 不全满足）——实体级变异满秩扩散，'
              '属个体噪声；T2 收缩为均值级参数化（只保留关系级 Δv，放弃实体级预测）。')
        f24['counter_evidence'] = f24_old['counter_evidence'] + [ce]
        f24['statement'] = f24_old['statement'] + (
            '；3186 G5-B11：δ_e 谱满秩扩散（top2 ' + fmt3('top2_mean') +
            '），T2 收缩为均值级参数化——FTR-24 的低秩性限于关系级方向，不支撑实体级预测。')
        reg23['features'] = [f24 if f['id'] == 'FTR-24' else f for f in reg22['features']]
        disposition = 'v1.12_to_v1.13_meanlevel_contraction'
    # preservation asserts
    for f in reg22['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg23['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-24':
            assert b != a, ('FTR-24 must change',)
            fa = json.loads(a)
            if gate_all:
                assert json.dumps(f24['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True), \
                    'anchors not prefix-preserved'
                assert f24['statement'].startswith(fa['statement']), 'statement not append-only'
                assert f24['counter_evidence'] == fa['counter_evidence'], 'ce changed on pass'
            else:
                assert f24['counter_evidence'][:-1] == fa['counter_evidence'], \
                    'ce not append-only'
                assert f24['statement'].startswith(fa['statement']), 'statement not append-only'
                assert len(f24['anchors']) == len(fa['anchors']), 'anchors changed on fail'
        else:
            assert a == b, ('feature not preserved', f['id'])
    for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg22.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg23.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg22['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg23['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_13.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg23, ensure_ascii=False, indent=1))
    reg23_sha = sha8_file(rp)
    log('registry v1.13 written sha8 %s (%s)' % (reg23_sha, disposition))

    meta_sha = {'reg23': reg23_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.14 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.14 &mdash; 语义机制图谱（registry v1.13 渲染）</h1>')
    H.append('<div class="sub">Phase 3186 G5-B11 &middot; 关系偏移二阶结构检验（%s） &middot; '
             'registry v1.13 sha8=%s &middot; v1.12 sha8=%s &middot; v1.11 sha8=%s &middot; v1.10 sha8=%s &middot; v1.9 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3177 %s + p3178 %s + p3182 %s + p3183 %s + p3184 %s + p3185 %s &middot; 渲染于 %s</div>' % (
                 'T2′ 平面参数化立项候选' if gate_all else '均值级收缩',
                 span('meta.registry_v113_sha8', reg23_sha),
                 span('meta.registry_v112_sha8', '269a37c0'),
                 span('meta.registry_v111_sha8', '84aecc21'),
                 span('meta.registry_v110_sha8', '4cf49743'),
                 span('meta.registry_v19_sha8', '59a91512'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3177_result', SHA_ANCHOR['p3177_result']),
                 span('meta.p3178_result', SHA_ANCHOR['p3178_result']),
                 span('meta.p3182_result', SHA_ANCHOR['p3182_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.p3184_result', SHA_ANCHOR['p3184_result']),
                 span('meta.p3185_result', SHA_ANCHOR['p3185_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg23['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg23['principles']):
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
    for f in reg23['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.13，%d 条）</h2>' % len(reg23['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg23['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg23['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg23['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.14 &middot; 渲染自 atlas_registry_v1_13.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3186 G5-B11 关系偏移二阶结构检验'
             '（数据源 p3157 collect %s / %s / %s + p3177 %s + p3178 %s + p3182 %s + p3183 %s + p3184 %s + p3185 %s）'
             '&middot; 本文件为 Phase 3186 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg23_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3177_result'], SHA_ANCHOR['p3178_result'],
                 SHA_ANCHOR['p3182_result'], SHA_ANCHOR['p3183_result'],
                 SHA_ANCHOR['p3184_result'], SHA_ANCHOR['p3185_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_14.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v114(reg23, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_14.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3186, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_all': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg23['features']), 'failures': len(reg23['failures']),
                   'upgrades': len(reg23['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v113_file': 'atlas_registry_v1_13.json', 'registry_v113_sha8': reg23_sha,
        'html_file': 'atlas_v1_14.html', 'html_sha8': html_sha,
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


def flatten_v114(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v113_sha8'] = meta_sha['reg23']
    d['meta.registry_v112_sha8'] = '269a37c0'
    d['meta.registry_v111_sha8'] = '84aecc21'
    d['meta.registry_v110_sha8'] = '4cf49743'
    d['meta.registry_v19_sha8'] = '59a91512'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3177_result'] = SHA_ANCHOR['p3177_result']
    d['meta.p3178_result'] = SHA_ANCHOR['p3178_result']
    d['meta.p3182_result'] = SHA_ANCHOR['p3182_result']
    d['meta.p3183_result'] = SHA_ANCHOR['p3183_result']
    d['meta.p3184_result'] = SHA_ANCHOR['p3184_result']
    d['meta.p3185_result'] = SHA_ANCHOR['p3185_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    global sealed_3177, sealed_3178, sealed_3182, sealed_3183, sealed_3185
    design_sha = freeze()
    log('== Phase 3186 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
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

    out, gate_all = run_arm()
    if gate_all:
        verdict = ('rrel_plane_parameterization_candidate (top2 SVD energy x3; '
                   'FTR-24 anchor appended, T2\' pinned: Delta-M_rel = T_C + '
                   'alpha*Delta-v + beta*w2, registry v1.12 -> v1.13)')
    else:
        verdict = ('rrel_meanlevel_contraction (delta_e full-rank diffuse; '
                   'FTR-24 counter_evidence + contraction statement append; '
                   'T2 keeps relation-level Delta-v only)')
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3186, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': {'top2_energy_min': G_T2, 'r_g_min': G_RG, 'r2_px_min': G_R2,
                  'repl_tol': REPL_TOL, 'device_3157': True, 'gate_all': bool(gate_all)},
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
