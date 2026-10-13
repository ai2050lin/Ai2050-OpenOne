# -*- coding: utf-8 -*-
"""Phase 3188 (G5-C1): GAP-4 residual 1.8x Delta-v directional attribution
(C-line opener, zero GPU, sealed 3169/3175/3157/3183 reuse).

Per prereg in MEMO Phase 3187 tail (frozen before any observation):
project the 3183-verbatim Delta-v direction onto the sealed 3169 OOV-panel
per-cell collapse residuals.
  (a) projection distribution of OOV-cell residual directions vs in-panel
      (seen S1-test) baseline;
  (b) direction-explained share of the OOV collapse readings (direction-level,
      NO magnitude claim);
  (c) if the projection distributions shift significantly (OOV vs seen) =>
      R_rel direction couples to the out-of-panel collapse: GAP-4 residual
      gets its first direction-level mechanism clue; otherwise decoupled =>
      counter_evidence registered.

Freeze-time protocol pinning (from sealed scripts verbatim):
  - Delta-v: 3157 collect npz, dC = H[ctx1]-H[ctx0] at KOUT=NL-1, dCn row-
    normalized, v_rl = unit_mean(dCn[rel 32 cells]), Delta-v = v_isa - v_hasa,
    dvu = unit(Delta-v). Device: tc_mean_cos anchor < 1e-9; 3183 dv
    replication (cos_rl/norm_dv/frac_full/pc_share/med_cos) < 1e-9.
  - 3169 collapse chain verbatim (B-protocol): Y = H[:,:,readout,:] fp32
    (readout == NL-1 double-locked), seeds 7/8/9, split_s1_246, ridge lam
    1e-3, cols = NE+NC+NT+1 = 780. Device: per-seed E_seen/E_oov replicate
    sealed 3169 values < 1e-9 (identical code path precedent: 3175 G_k0).
  - residual direction per (row, seed): rvec = Y[r] - (X[r] @ W) at readout;
    q = unit(rvec); proj = q @ dvu; share = (rvec @ dvu)^2 / ||rvec||^2.
  - row-level aggregates: OOV rows (292 pairs x 3 tpl = 876) mean over 3
    seeds; seen rows mean over seeds where the row is in the S1 test set.
  - (a) gate stats: two-sample KS statistic + permutation p (n_perm=20000,
    RandomState(20260388), pooled-label permutation) + median difference.
  - GATE (pinned pre-observation): COUPLED iff KS >= 0.15 AND perm_p < 0.01
    in >= 2/3 models AND sign(median_oov - median_seen) consistent in >= 2/3
    models => FTR-25 candidate E1_repeatable (registry v1.14 -> v1.15
    features append, 24 legacy byte-preserved), atlas_v1_16.html.
    Else DECOUPLED => FTR-24 counter_evidence append (direction-decoupling),
    no new feature, registry v1.14 -> v1.15 likewise, atlas_v1_16.html.
  - (b) share means (OOV vs seen vs frozen random-direction control ~ 1/D)
    report-only, no magnitude claim.
  - APPENDIX arm (3175 out-arm k=8, report-only): verbatim 3175 out-arm
    replication; anchor E_oov/E_seen per-seed vs sealed out_curve['8']
    < 1e-9; then te_oov (876 - 96 = 780 rows) residual projections, KS vs
    primary OOV distribution, report-only. If the anchor drifts >= 1e-9 the
    appendix is marked appendix_anchor_fail and EXCLUDED from any reading;
    primary gate unaffected (declared here, before observation).
  - SMOKE (env P3188_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
import phase3179_g5b4_ftr23_registration as p79

span = p79.span

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5c1_gap4_dv_attribution'
OUTDIR = os.path.join(SRC_DIR, 'phase3188', NAME)
SMOKE = os.environ.get('P3188_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
NPZ57_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
RES57_KEY = {'qwen3-4b': 'p3157_res_4b', 'qwen3-14b': 'p3157_res_14b', 'glm4-9b': 'p3157_res_glm4'}
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}
NPZ75_KEY = {'qwen3-4b': 'p3175_out_4b', 'qwen3-14b': 'p3175_out_14b', 'glm4-9b': 'p3175_out_glm4'}

G_KS = 0.15
G_P = 0.01
N_PERM = 20000
PERM_SEED = 20260388
CTRL_SEED = 20260389
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'ks_min': G_KS, 'perm_p_max': G_P, 'n_perm': N_PERM,
        'models_needed': 2, 'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}

SRC = {
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3157_res_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\result.json'),
    'p3157_res_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\result.json'),
    'p3157_res_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\result.json'),
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3175_result': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\result.json'),
    'p3175_out_4b': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\collect_out_qwen3-4b.npz'),
    'p3175_out_14b': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\collect_out_qwen3-14b.npz'),
    'p3175_out_glm4': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\collect_out_glm4-9b.npz'),
    'registry_v114': os.path.join(SRC_DIR, r'phase3187\g5b12_alpha_calibration\atlas_registry_v1_14.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3157_res_4b': 'ca863153', 'p3157_res_14b': 'e797adf9', 'p3157_res_glm4': '4030081f',
    'p3183_result': '0ab70579',
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4',
    'p3175_result': 'e0d2d459', 'p3175_out_4b': 'd4f8931b', 'p3175_out_14b': 'b9f0a96f',
    'p3175_out_glm4': 'e14c62c4',
    'registry_v114': '95fe55f5', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3175 = ('cb76a177', '9e1a2f08')

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, tc_mean_cos=0.5640953183174133),
}

# ---------------- 3169 panel verbatim ----------------
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。', 1: '{e}属于{c}这一类。', 2: '{e}，一种常见的{c}。'}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
ENT_OOV_OUT = {
    '乐器': ['长笛', '竖琴', '唢呐', '大提琴', '手风琴', '萨克斯', '木琴', '锣', '钹', '竖笛'],
    '天气': ['彩虹', '闪电', '暴雨', '微风', '寒潮', '热浪', '沙尘暴', '霜冻', '梅雨', '阴天'],
    '运动': ['网球', '排球', '跳水', '滑雪', '射箭', '击剑', '马拉松', '瑜伽', '跳高', '举重'],
    '电器': ['烤箱', '豆浆机', '加湿器', '电吹风', '热水器', '打印机', '电熨斗', '榨汁机', '路由器', '电磁炉'],
}
NT = 3
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
LAM = 1e-3
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV_PANEL)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
N_SEEN_ENT = sum(len(v) for v in ENT_SEEN.values())
assert (NE, NC, NP_) == (73, 10, 730) and N_SEEN_ENT == 41
SEEN_PAIR_SET = set((i, c) for i in range(N_SEEN_ENT) for c in range(len(CLASSES_SEEN)))
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
OOV_CLS_PAIRS = set((i, c) for i in range(NE) for c in range(len(CLASSES_SEEN), NC))
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= len(CLASSES_SEEN)]
assert len(ALL_OOV_ROWS) == 876
OUT_ENTS = [e for cl in CLASSES_OOV for e in ENT_OOV_OUT[cl]]
N_OUT_E = len(OUT_ENTS)
OUT_PAIRS = [(i, c) for i in range(N_OUT_E) for c in range(NC)]
NP_OUT = len(OUT_PAIRS)

DESIGN = {
    'phase': '3188', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3187 tail G5-C1 (frozen before any observation)',
    'data': 'sealed 3169 collect npz x3 + 3175 collect_out npz x3 + 3157 collect '
            'npz x3 + sealed 3169/3175/3183 result.json; zero GPU',
    'dv_definition': '3157 verbatim (3183 recipe): dC = H[ctx1]-H[ctx0] at '
                     'KOUT=NL-1; dCn row-normalized; v_rl = unit_mean(dCn[rel 32]); '
                     'Delta-v = v_isa - v_hasa; dvu = unit(Delta-v). Layer '
                     'compatibility pinned: 3157 readout == 3169 readout per model '
                     '(35/39/39), same residual stream slot',
    'primary_arm': '3169 B-protocol verbatim per-cell collapse residuals at '
                   'readout; per (row,seed) q = unit(Y - XW); proj = q@dvu; '
                   'row-level mean over seeds; OOV rows 876 (all 3 seeds) vs seen '
                   'S1-test rows (mean over seeds where in test)',
    'measurements': 'per model: KS(proj_oov, proj_seen), permutation p '
                    '(n_perm=20000, rng 20260388), median diff, share means '
                    '(OOV/seen/random-direction control ~1/D, report-only); '
                    'appendix: 3175 out-arm k=8 te_oov (780 rows) projections, '
                    'KS vs primary OOV, report-only',
    'gate': 'COUPLED iff KS >= 0.15 AND perm_p < 0.01 in >= 2/3 models AND '
            'median-diff sign consistent >= 2/3 => FTR-25 candidate '
            'E1_repeatable (registry v1.14->v1.15 features append, 24 legacy '
            'byte-preserved, atlas_v1_16.html); else DECOUPLED => FTR-24 '
            'counter_evidence append, no new feature, registry v1.15 likewise',
    'appendix_policy': '3175 out-arm k=8 anchor (E_oov/E_seen per-seed < 1e-9 vs '
                       'sealed out_curve) REQUIRED for appendix readings; on drift '
                       '>= 1e-9 appendix marked appendix_anchor_fail and excluded; '
                       'primary gate unaffected (declared pre-observation)',
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
        json.dump({'phase': 3188, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


# ---------------- verbatim helpers (3157/3183) ----------------
def dC_rows(rel_filter):
    out = []
    for ei in range(16):
        for rl in range(2):
            if rel_filter is not None and rl != rel_filter:
                continue
            for pl in range(2):
                out.append(ei * 4 + rl * 2 + pl)
    return out


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


# ---------------- 3169 verbatim mechanics ----------------
def split_s1_246(seed):
    pairs246 = sorted(SEEN_PAIR_SET)
    assert len(pairs246) == 246
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def ridge_primal(Xtr, Ytr, lam=LAM):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return W


def phi_cols(ne, nc, nt):
    return ne + nc + nt + 1


def phi_row(i, c, t, ne, nc, nt):
    v = np.zeros(phi_cols(ne, nc, nt), np.float32)
    v[i] = 1.0
    v[ne + c] = 1.0
    v[ne + nc + t] = 1.0
    v[-1] = 1.0
    return v


def rows_of_joint(pair_set):
    return [t * NP_ + PI_OF_PAIR[p] for t in range(NT) for p in PAIRS if p in pair_set]


def calib_panel(seed, k):
    """3172/3175 verbatim: per OOV class k panel entity global indices."""
    PANEL_ROW_BASE = {}
    _i = 0
    for cl in CLASSES:
        for e in ENT[cl]:
            PANEL_ROW_BASE[e] = _i
            _i += 1
    assert _i == NE
    out = {}
    for ci, cl in enumerate(CLASSES_OOV):
        n_c = len(ENT[cl])
        base = PANEL_ROW_BASE[ENT[cl][0]]
        if k == 0:
            out[ci] = []
        else:
            rng = np.random.RandomState(seed + 1000)
            pick = rng.choice(n_c, size=k, replace=False)
            out[ci] = sorted(base + int(j) for j in pick)
    return out


def calib_out(seed, k):
    out = {}
    for ci, cl in enumerate(CLASSES_OOV):
        n_c = len(ENT_OOV_OUT[cl])
        if k == 0:
            out[ci] = []
        else:
            rng = np.random.RandomState(seed + 1000)
            pick = rng.choice(n_c, size=k, replace=False)
            out[ci] = sorted(int(j) for j in pick)
    return out


def ks_stat(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side='right') / float(len(a))
    cdf_b = np.searchsorted(b, allv, side='right') / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def perm_p(a, b, n_perm=N_PERM, seed=PERM_SEED):
    obs = ks_stat(a, b)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pooled = np.concatenate([a, b])
    na = len(a)
    rng = np.random.RandomState(seed)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        k = ks_stat(pooled[idx[:na]], pooled[idx[na:]])
        if k >= obs:
            cnt += 1
    return obs, float((1 + cnt) / (1 + n_perm))


# ---------------- dvu per model (3157/3183 verbatim) ----------------
def build_dvu(sealed_3183):
    dvus = {}
    for m in MODELS:
        ak = ANCH_KEY[m]
        A = ANCH_3157[ak]
        z = np.load(SRC[NPZ57_KEY[m]])
        H = z['H'].astype(np.float32)
        NROWS, NH, D = H.shape
        NL = NH - 1
        KOUT = NL - 1
        assert NROWS == 128 and (NL, KOUT) == (A['nl'], A['kout']), ('layer', m)
        EI = z['ei'].astype(int)
        RL = z['rel'].astype(int)
        PL = z['pol'].astype(int)
        CX = z['ctx'].astype(int)
        IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
        assert len(IDX) == 128
        dC = []
        for ei in range(16):
            for rl in range(2):
                for pl in range(2):
                    dC.append(H[IDX[(ei, rl, pl, 1)], KOUT] - H[IDX[(ei, rl, pl, 0)], KOUT])
        dC = np.array(dC)
        assert np.isfinite(dC).all()
        dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
        cosM = dCn @ dCn.T
        tri = cosM[np.triu_indices(len(dC), 1)]
        tc_full = float(tri.mean())
        assert abs(tc_full - A['tc_mean_cos']) < DEV_TOL, ('tc drift', m)
        v_isa = unit_mean(dCn[dC_rows(0)])
        v_hasa = unit_mean(dCn[dC_rows(1)])
        cos_rl = float(v_isa @ v_hasa)
        dv = v_isa - v_hasa
        norm_dv = float(np.linalg.norm(dv))
        v_full = unit_mean(dCn)
        frac_full = float((dv @ v_full) ** 2 / max(norm_dv ** 2, 1e-300))
        U = []
        for e in range(16):
            a_e = dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0)
            b_e = dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)
            U.append(unit(a_e - b_e))
        U = np.stack(U)
        s0 = np.linalg.svd(U, compute_uv=False)
        pc_share = float(s0[0] ** 2 / (s0 ** 2).sum())
        Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
        C = Un @ Un.T
        med_cos = float(np.median(C[np.triu_indices(16, 1)]))
        E83 = sealed_3183['per_model'][m]
        d83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']),
                  abs(frac_full - E83['frac_full']), abs(pc_share - E83['pc_share']),
                  abs(med_cos - E83['med_cos']))
        assert d83 < REPL_TOL, ('3183 dv replication', m, d83)
        dvus[m] = dict(dvu=unit(dv), kout=KOUT, nl=NL, D=D, tc_full=tc_full,
                       dv_rep_drift=d83, norm_dv=norm_dv, cos_rl=cos_rl)
        log('dvu %s: tc=%.6f ok, ||dv||=%.4f, 3183 drift=%.2e' % (m, tc_full, norm_dv, d83))
        del z, H
    return dvus


# ---------------- primary arm: 3169 per-cell residual projections ----------------
def run_primary(sealed_3169, dvus):
    out = {}
    for m in MODELS:
        log('=== primary %s ===' % m)
        rd = int(sealed_3169['per_model'][m]['readout'])
        z = np.load(SRC[NPZ69_KEY[m]])
        H16 = z['H']
        NTz, NPz, NHz, Dz = H16.shape
        assert NTz == NT and NPz == NP_, ('shape', H16.shape)
        kout = NHz - 2
        assert kout == rd == dvus[m]['kout'], ('readout slot', m, kout, rd)
        D = Dz
        Y = H16[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)
        assert np.isfinite(Y).all()

        te_seen_all = {}
        tr_base = {}
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                          for p in PAIRS if p in tr_pairs]
            te_seen_all[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                              for p in PAIRS if p in te_pairs]
        oov_rows = ALL_OOV_ROWS

        def rv(r, ne):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, ne, NC, NT)

        # device: per-seed E replication vs sealed 3169
        E69 = sealed_3169['per_model'][m]['B']
        drift = 0.0
        proj_oov_rows = np.zeros(len(oov_rows))
        share_oov = []
        proj_seen_acc = {}
        share_seen = []
        ctrl = unit(np.random.RandomState(CTRL_SEED).normal(size=D))
        share_ctrl = []
        for si, s in enumerate(SEEDS_S1):
            tr_rows = tr_base[s]
            te_rows = te_seen_all[s]
            Xtr = np.stack([rv(r, NE) for r in tr_rows])
            W = ridge_primal(Xtr, Y[tr_rows])
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            Xs = np.stack([rv(r, NE) for r in te_rows])
            es = ((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk
            Xo = np.stack([rv(r, NE) for r in oov_rows])
            eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
            drift = max(drift, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                        abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
            # residual directions
            R_oov = Y[oov_rows] - Xo @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                nv = float(np.linalg.norm(rvv))
                pr = float(rvv @ dvus[m]['dvu'])
                proj_oov_rows[j] += pr / (nv + 1e-18)
                share_oov.append(pr * pr / max(nv * nv, 1e-300))
                share_ctrl.append(float(rvv @ ctrl) ** 2 / max(nv * nv, 1e-300))
            R_seen = Y[te_rows] - Xs @ W
            for j, r in enumerate(te_rows):
                rvv = R_seen[j]
                nv = float(np.linalg.norm(rvv))
                pr = float(rvv @ dvus[m]['dvu'])
                proj_seen_acc.setdefault(r, []).append(pr / (nv + 1e-18))
                share_seen.append(pr * pr / max(nv * nv, 1e-300))
            del W, Xtr
        assert drift < REPL_TOL, ('3169 per-seed E replication', m, drift)
        log('  3169 per-seed E replication drift=%.2e OK' % drift)
        proj_oov = proj_oov_rows / len(SEEDS_S1)
        proj_seen = np.array([float(np.mean(v)) for v in proj_seen_acc.values()])
        ks, pp = perm_p(proj_oov, proj_seen)
        med_oov = float(np.median(proj_oov))
        med_seen = float(np.median(proj_seen))
        res = dict(n_oov=len(proj_oov), n_seen=len(proj_seen),
                   ks=ks, perm_p=pp, med_oov=med_oov, med_seen=med_seen,
                   med_diff=med_oov - med_seen,
                   share_oov=float(np.mean(share_oov)),
                   share_seen=float(np.mean(share_seen)),
                   share_ctrl=float(np.mean(share_ctrl)),
                   floor_1_over_D=1.0 / D, drift_3169=drift,
                   kout=kout, D=D)
        gate_m = bool(ks >= G_KS and pp < G_P)
        res['gate_model'] = gate_m
        out[m] = res
        log('  %s: KS=%.4f p=%.5f medDiff=%+.4f | share oov=%.5f seen=%.5f ctrl=%.5f (1/D=%.2e) gate=%s'
            % (m, ks, pp, med_oov - med_seen, res['share_oov'], res['share_seen'],
               res['share_ctrl'], 1.0 / D, gate_m))
        del z, H16, Y
    # pooled gate
    n_ok = sum(1 for m in MODELS if out[m]['gate_model'])
    signs = set(1 if out[m]['med_diff'] > 0 else -1 for m in MODELS if out[m]['gate_model'])
    sign_cons = bool(len(signs) == 1 and n_ok >= 2)
    gate_all = bool(n_ok >= 2 and sign_cons)
    out['_gate'] = dict(n_models_pass=n_ok, sign_consistent=sign_cons, coupled=gate_all)
    log('GATE: %d/3 models pass KS+p; sign consistent=%s => %s'
        % (n_ok, sign_cons, 'COUPLED' if gate_all else 'DECOUPLED'))
    return out, gate_all


# ---------------- appendix arm: 3175 out-arm k=8 projections ----------------
def run_appendix(sealed_3169, sealed_3175, dvus):
    out = {}
    for m in MODELS:
        log('=== appendix %s ===' % m)
        rd = int(sealed_3169['per_model'][m]['readout'])
        zp = np.load(SRC[NPZ69_KEY[m]])
        H16 = zp['H']
        NHz, D = H16.shape[2], H16.shape[3]
        kout = NHz - 2
        assert kout == rd == dvus[m]['kout']
        Y = H16[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)
        zo = np.load(SRC[NPZ75_KEY[m]])
        H_out = zo['H']
        assert H_out.shape == (NT, NP_OUT, NHz, D), ('out shape', H_out.shape)
        Y_out = H_out[:, :, kout, :].reshape(NT * NP_OUT, D).astype(np.float32)

        te_seen_all = {}
        tr_base = {}
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                          for p in PAIRS if p in tr_pairs]
            te_seen_all[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                              for p in PAIRS if p in te_pairs]

        def rv(r, ne):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, ne, NC, NT)

        K = 8
        E75 = sealed_3175['per_model'][m]['out_curve'][str(K)]
        drift = 0.0
        proj_out_rows = {}
        for si, s in enumerate(SEEDS_S1):
            in_cal = calib_panel(s, K)
            in_cal_rows = []
            for ci, idxs in in_cal.items():
                c = len(CLASSES_SEEN) + ci
                for i in idxs:
                    for t in range(NT):
                        in_cal_rows.append(t * NP_ + PI_OF_PAIR[(i, c)])
            te_oov = [r for r in ALL_OOV_ROWS if r not in set(in_cal_rows)]
            assert len(te_oov) == 780, ('te_oov size', len(te_oov))
            out_cal = calib_out(s, K)
            ext_ents = []
            out_triples = []
            for ci in range(len(CLASSES_OOV)):
                c = len(CLASSES_SEEN) + ci
                for j in out_cal[ci]:
                    ext_ents.append((c, j))
                    for t in range(NT):
                        out_triples.append((j, c, t))
            ext_ents = sorted(ext_ents)
            ext_index = {}
            for rank, (c, j) in enumerate(ext_ents):
                ext_index[(c, j)] = NE + rank
            ne_ext = NE + len(ext_ents)
            X_list = [rv(r, ne_ext) for r in tr_base[s]]
            Y_list = [Y[r] for r in tr_base[s]]
            for (j, c, t) in out_triples:
                g = ext_index[(c, j)]
                v = np.zeros(phi_cols(ne_ext, NC, NT), np.float32)
                v[g] = 1.0
                v[ne_ext + c] = 1.0
                v[ne_ext + NC + t] = 1.0
                v[-1] = 1.0
                X_list.append(v)
                Y_list.append(Y_out[t * NP_OUT + j * NC + c])
            Xtr = np.stack(X_list)
            Ytr = np.stack(Y_list)
            W = ridge_primal(Xtr, Ytr)
            ref = Ytr.mean(0)
            Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
            Xo = np.stack([rv(r, ne_ext) for r in te_oov])
            eo = ((Xo @ W - Y[te_oov]) ** 2).sum(1) / Dk
            drift = max(drift, abs(float(eo.mean()) - E75['E_oov_per_seed'][si]))
            R_oov = Y[te_oov] - Xo @ W
            for j, r in enumerate(te_oov):
                rvv = R_oov[j]
                nv = float(np.linalg.norm(rvv))
                pr = float(rvv @ dvus[m]['dvu'])
                proj_out_rows.setdefault(r, []).append(pr / (nv + 1e-18))
            del W, Xtr
        anchor_ok = drift < REPL_TOL
        res = dict(k=K, drift_out_arm=drift, anchor_ok=bool(anchor_ok),
                   n_rows=len(proj_out_rows))
        if anchor_ok:
            proj_out = np.array([float(np.mean(v)) for v in proj_out_rows.values()])
            # primary OOV distribution recomputed light: reuse via sealed? we
            # pass it in from primary run through closure-free dict below
            res['proj_mean'] = float(np.mean(proj_out))
            res['proj_med'] = float(np.median(proj_out))
        else:
            res['note'] = 'appendix_anchor_fail: excluded from readings (pre-declared)'
            log('  APPENDIX ANCHOR FAIL %s drift=%.2e (>=1e-9) -> excluded' % (m, drift))
        out[m] = res
        log('  appendix %s: k=8 out-arm drift=%.2e anchor_ok=%s n_rows=%d'
            % (m, drift, anchor_ok, len(proj_out_rows)))
        del zp, H16, Y, zo, H_out, Y_out
    return out


def main():
    global sealed_3183, sealed_3169, sealed_3175
    design_sha = freeze()
    log('== Phase 3188 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    assert sealed_3183['res_sha8'] == CONTENT_3183[0] and sealed_3183['seal_sha8'] == CONTENT_3183[1]
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    assert sealed_3169['res_sha8'] == CONTENT_3169[0] and sealed_3169['seal_sha8'] == CONTENT_3169[1]
    sealed_3175 = json.load(open(SRC['p3175_result'], encoding='utf-8'))
    assert sealed_3175['res_sha8'] == CONTENT_3175[0] and sealed_3175['seal_sha8'] == CONTENT_3175[1]
    assert abs(float(sealed_3169['gate']['ratio_B']) - 2.5388114997805062) < 1e-9
    log('sealed results double-locked: 3183 %s/%s 3169 %s/%s 3175 %s/%s'
        % (CONTENT_3183 + CONTENT_3169 + CONTENT_3175))

    dvus = build_dvu(sealed_3183)
    primary, gate_all = run_primary(sealed_3169, dvus)
    appendix = run_appendix(sealed_3169, sealed_3175, dvus)
    if gate_all:
        verdict = ('dv_direction_coupled_to_oov_collapse (KS>=0.15 & p<0.01 in '
                   '%d/3 models, median-diff sign consistent; FTR-25 candidate '
                   'E1_repeatable, registry v1.14 -> v1.15)'
                   % primary['_gate']['n_models_pass'])
    else:
        verdict = ('dv_direction_decoupled_from_oov_collapse (gate fail: %d/3 '
                   'models pass; FTR-24 counter_evidence append, GAP-4 residual '
                   'decoupled from relation offset at direction level, '
                   'registry v1.14 -> v1.15)' % primary['_gate']['n_models_pass'])
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3188, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, coupled=bool(gate_all),
                      n_models_pass=primary['_gate']['n_models_pass'],
                      sign_consistent=primary['_gate']['sign_consistent']),
        'per_model': {m: primary[m] for m in MODELS},
        'appendix_3175_out_k8': appendix,
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


# ---------------- registry v1.14 -> v1.15 + atlas_v1_16 ----------------
CREATED_STR = None  # set in finish_registry (formal only)


def finish_registry(summary, gate_all, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')
    reg24 = json.loads(open(SRC['registry_v114'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg24['version'] == '1.14' and len(reg24['features']) == 24
    assert len(reg24['failures']) == 19 and len(reg24['upgrade_log']) == 6
    reg25 = copy.deepcopy(reg24)
    reg25['version'] = '1.15'
    reg25['supersedes'] = 'atlas_registry_v1_14.json (95fe55f5, Phase 3187)'
    pm = summary['per_model']
    MODELS3 = MODELS
    fmt3 = lambda key: '/'.join('%.4f' % pm[m][key] for m in MODELS3)
    if gate_all:
        f25 = {
            'id': 'FTR-25', 'family': 'context',
            'name': 'dv_direction_oov_collapse_coupling',
            'statement': ('Δv 方向与谱外崩塌耦合（3188 G5-C1 定判）：3183 verbatim Δv 方向'
                          '对 3169 谱外面板逐细胞崩塌残差方向的投影分布相对面板内 S1 测试'
                          '基线显著偏移——KS=' + fmt3('ks') + '、置换 p=' + fmt3('perm_p') +
                          '、中位差=' + fmt3('med_diff') + '（≥2/3 模型 KS≥0.15 且 p<0.01、'
                          '中位差符号一致）。方向级主张：R_rel 方向与谱外崩塌读数耦合，'
                          'GAP-4 残留 1.8× 获得首条方向级机制线索；无幅度主张。'),
            'evidence_level': 'E1_repeatable',
            'model_scope': MODELS3,
            'anchors': [{
                'src': 'p3188', 'phase': 3188,
                'role': 'dv_direction_oov_coupling',
                'tags': ['zero_gpu', 'sealed_reuse'],
                'asserts': {
                    'ks.qwen3-4b': pm['qwen3-4b']['ks'],
                    'ks.qwen3-14b': pm['qwen3-14b']['ks'],
                    'ks.glm4-9b': pm['glm4-9b']['ks'],
                    'gate_ks_min': G_KS, 'gate_perm_p_max': G_P,
                    'res_sha8': res8, 'seal_sha8': seal8}},
            ],
            'counter_evidence': [],
            'replication': '3169 per-seed E replication <1e-9; 3157 tc + 3183 dv '
                           'replication <1e-9; permutation p from n_perm=20000 '
                           'fixed-seed labels permutation',
            'scope_limits': ['direction-level only, no magnitude claim',
                             '3169 panel classes/entities only',
                             'readout-layer slot'],
            'values': None,
        }
        old = json.dumps(reg24['features'], ensure_ascii=False, sort_keys=True)
        reg25['features'] = reg24['features'] + [f25]
        assert len(reg25['features']) == 25
        disposition = 'v1.14_to_v1.15_ftr25_coupled'
    else:
        f24_old = [f for f in reg24['features'] if f['id'] == 'FTR-24'][0]
        f24 = copy.deepcopy(f24_old)
        ce = ('3188 G5-C1 解耦定判：Δv 方向对 3169 谱外逐细胞崩塌残差投影分布无显著偏移'
              '（KS=' + fmt3('ks') + '，p=' + fmt3('perm_p') + '，中位差=' + fmt3('med_diff') +
              '；门 KS≥0.15 且 p<0.01 ×≥2/3 模型未过）——GAP-4 残留 1.8× 与关系偏移方向'
              '解耦，counter_evidence 登记。')
        f24['counter_evidence'] = f24_old['counter_evidence'] + [ce]
        old = json.dumps(reg24['features'], ensure_ascii=False, sort_keys=True)
        reg25['features'] = [f24 if f['id'] == 'FTR-24' else f
                             for f in reg24['features']]
        disposition = 'v1.14_to_v1.15_ftr24_ce_decoupled'
    # preservation asserts
    new = json.dumps(reg25['features'], ensure_ascii=False, sort_keys=True)
    if gate_all:
        assert new.startswith(old[:-1]) or len(reg25['features']) == len(reg24['features']) + 1
        for f in reg24['features']:
            a = json.dumps(f, ensure_ascii=False, sort_keys=True)
            b = json.dumps([x for x in reg25['features'] if x['id'] == f['id']][0],
                           ensure_ascii=False, sort_keys=True)
            assert a == b, ('feature not preserved', f['id'])
    else:
        for f in reg24['features']:
            a = json.dumps(f, ensure_ascii=False, sort_keys=True)
            b = json.dumps([x for x in reg25['features'] if x['id'] == f['id']][0],
                           ensure_ascii=False, sort_keys=True)
            if f['id'] == 'FTR-24':
                fa = json.loads(a)
                assert b != a
                assert f24['counter_evidence'][:-1] == fa['counter_evidence'], 'ce not append-only'
                assert f24['statement'] == fa['statement'], 'statement changed'
                assert f24['anchors'] == fa['anchors'], 'anchors changed'
            else:
                assert a == b, ('feature not preserved', f['id'])
    for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg24.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg25.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg24['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg25['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_15.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg25, ensure_ascii=False, indent=1))
    reg25_sha = sha8_file(rp)
    log('registry v1.15 written sha8 %s (%s)' % (reg25_sha, disposition))

    meta_sha = {'reg25': reg25_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.16 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.16 &mdash; 语义机制图谱（registry v1.15 渲染）</h1>')
    H.append('<div class="sub">Phase 3188 G5-C1 &middot; GAP-4 残留 1.8&times; 的 Δv 方向归因（%s） &middot; '
             'registry v1.15 sha8=%s &middot; v1.14 sha8=%s &middot; v1.13 sha8=%s &middot; v1.12 sha8=%s &middot; v1.11 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3157 collect %s/%s/%s + p3169 %s + p3175 %s + p3183 %s &middot; 渲染于 %s</div>' % (
                 'FTR-25 耦合入册' if gate_all else 'FTR-24 counter_evidence 解耦登记',
                 span('meta.registry_v115_sha8', reg25_sha),
                 span('meta.registry_v114_sha8', '95fe55f5'),
                 span('meta.registry_v113_sha8', '730729f5'),
                 span('meta.registry_v112_sha8', '269a37c0'),
                 span('meta.registry_v111_sha8', '84aecc21'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3157_14b', SHA_ANCHOR['p3157_14b']),
                 span('meta.p3157_glm4', SHA_ANCHOR['p3157_glm4']),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3175_result', SHA_ANCHOR['p3175_result']),
                 span('meta.p3183_result', SHA_ANCHOR['p3183_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg25['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg25['principles']):
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
    for f in reg25['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.15，%d 条）</h2>' % len(reg25['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg25['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg25['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg25['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.16 &middot; 渲染自 atlas_registry_v1_15.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3188 G5-C1 GAP-4 残留 Δv 方向归因'
             '（数据源 p3157 collect %s / %s / %s + p3169 %s + p3175 %s + p3183 %s）'
             '&middot; 本文件为 Phase 3188 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg25_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3157_4b'], SHA_ANCHOR['p3157_14b'], SHA_ANCHOR['p3157_glm4'],
                 SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3175_result'],
                 SHA_ANCHOR['p3183_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_16.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v116(reg25, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_16.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3188, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_coupled': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg25['features']), 'failures': len(reg25['failures']),
                   'upgrades': len(reg25['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v115_file': 'atlas_registry_v1_15.json', 'registry_v115_sha8': reg25_sha,
        'html_file': 'atlas_v1_16.html', 'html_sha8': html_sha,
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


def flatten_v116(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v115_sha8'] = meta_sha['reg25']
    d['meta.registry_v114_sha8'] = '95fe55f5'
    d['meta.registry_v113_sha8'] = '730729f5'
    d['meta.registry_v112_sha8'] = '269a37c0'
    d['meta.registry_v111_sha8'] = '84aecc21'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3157_14b'] = SHA_ANCHOR['p3157_14b']
    d['meta.p3157_glm4'] = SHA_ANCHOR['p3157_glm4']
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3175_result'] = SHA_ANCHOR['p3175_result']
    d['meta.p3183_result'] = SHA_ANCHOR['p3183_result']
    d['meta.created'] = CREATED_STR
    return d


if __name__ == '__main__':
    main()
