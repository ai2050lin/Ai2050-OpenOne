# -*- coding: utf-8 -*-
"""Phase 3193 (G5-D1): subspace decomposition of the shared OOV collapse
direction u* against known structural directions (D line opening, zero GPU,
sealed 3157/3169/3183/3189-3192 reuse).

Per prereg in MEMO Phase 3192 tail (frozen before any observation):
  Component directions (all derived from SEALED collect npz, zero GPU):
    [GATE SET - external structure]
    (1) dvu  - relation-offset direction, 3157/3183 verbatim re-derivation;
    (2) v_s[c] - seen-class mean residual directions (6 classes, 3169 B-chain,
        all 738 seen rows, 3-seed mean residuals -> per-class unit mean);
    (3) u_seen - SVD top-1 of the 738 seen mean-residual row directions.
    [REPORT-ONLY - self-consistency reference]
    (4) v_c[c] - OOV class mean directions (3189 chain; 3190 already showed
        |cos(u*, v_c)| 0.72-0.85).
  PREREG DOWNGRADE (pinned at freeze, honest): the prereg text named 3165
  family-alignment directions and 3166 logic-contrast directions; the sealed
  3165/3166 results contain scalars/singular values only (no direction
  vectors) - re-deriving them would require re-running those protocols,
  outside the zero-GPU budget. Downgraded to the component set above;
  deviation registered in MEMO.
  GATE (pinned pre-observation):
    DECOMPOSED iff max |cos(u*, gate set)| >= 0.5 in >= 2/3 models =>
    GAP-4 mechanism_note second append, gap ledger v1.6 -> v1.7
    (append-only), registry v1.18 UNCHANGED, atlas_v1_21.html;
    NOVEL iff max |cos(u*, gate set)| < 0.3 in >= 2/3 models =>
    FTR-26 candidate registration (family=context E1_repeatable,
    features 25->26 append-only), registry v1.18 -> v1.19, atlas_v1_21.html;
    intermediate band => report-only (no registry/gap output).
  Device anchors: 3189/3190/3191 sealed scalars drift == 0 bitwise; 3157 tc
  anchor + 3183 dv replication; 3169 per-seed E replication < 1e-9.
  SMOKE (env P3193_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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

span = p79.span

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5d1_ustar_decomposition'
OUTDIR = os.path.join(SRC_DIR, 'phase3193', NAME)
SMOKE = os.environ.get('P3193_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}
NPZ57_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

G_DECOMP = 0.5
G_NOVEL = 0.3
DEV_TOL = 1e-9
REPL_TOL = 1e-9
RATIO_B_ANCHOR = 2.5388114997805062
GATE = {'decompose_cos': G_DECOMP, 'novel_cos': G_NOVEL, 'models_needed': 2,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}

ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, tc_mean_cos=0.5640953183174133),
}

SRC = {
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190_result': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'p3191_result': os.path.join(SRC_DIR, r'phase3191\g5c4_ftr25_heldout_upgrade\result.json'),
    'p3192_result': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\result.json'),
    'registry_v118': os.path.join(SRC_DIR, r'phase3191\g5c4_ftr25_heldout_upgrade\atlas_registry_v1_18.json'),
    'gap_v16': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3183_result': '0ab70579',
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0', 'p3190_result': '6a8893ae',
    'p3191_result': 'a218ead4',
    'p3192_result': '5d8d4858',
    'registry_v118': 'c08cbcc2', 'gap_v16': '0f6423e0', 'gap_v15': '4fb3f37d',
    'registry_3162': '00f15e98',
}
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3189 = ('3a26f541', 'fb8c1f65')
CONTENT_3190 = ('24461856', '0c975f5b')
CONTENT_3191 = ('836fe933', 'a66c0b3d')
CONTENT_3192 = ('6221fe00', '58c4f183')
CONTENT_3183 = ('c55e4e5e', '7196fa8a')

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
assert (NE, NC, NP_) == (73, 10, 730)
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= len(CLASSES_SEEN)]
assert len(ALL_OOV_ROWS) == 876
ALL_SEEN_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                 if PAIRS[pi][1] < len(CLASSES_SEEN) and PAIRS[pi][0] < 41]
assert len(ALL_SEEN_ROWS) == 738

DESIGN = {
    'phase': '3193', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3192 tail G5-D1 (frozen before any observation)',
    'data': 'sealed 3157/3169 collect npz + sealed 3169/3183/3189/3190/3191/3192 '
            'results; zero GPU',
    'residual_chain': '3188..3192 primary verbatim; u* = Vt[0] of Qn (feature '
                      'space, 3190 verbatim)',
    'component_downgrade': 'PREREG DOWNGRADE (freeze-pinned, honest): 3165/3166 '
                           'sealed results hold scalars/singular values only, no '
                           'direction vectors; re-derivation outside zero-GPU '
                           'budget. Gate set = {dvu (3157/3183 verbatim), v_s x6 '
                           '(seen-class mean residual directions), u_seen (seen '
                           'residual top-1 SVD)}; report-only = v_c x4 (OOV class '
                           'means, self-consistency)',
    'measurements': 'per model: |cos(u*, component)| for gate set + report-only '
                    'set; device anchors as in 3188-3192',
    'gate': 'DECOMPOSED iff max |cos(u*, gate set)| >= 0.5 in >= 2/3 models => '
            'GAP-4 mechanism_note second append, gap v1.6->v1.7, registry v1.18 '
            'UNCHANGED, atlas_v1_21.html; NOVEL iff max < 0.3 in >= 2/3 models '
            '=> FTR-26 candidate (family=context E1_repeatable, features '
            '25->26), registry v1.18->v1.19, atlas_v1_21.html; intermediate => '
            'report-only (no registry/gap output)',
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
        json.dump({'phase': 3193, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def split_s1_246(seed):
    SEEN_PAIR_SET = set((i, c) for i in range(41) for c in range(6))
    pairs246 = sorted(SEEN_PAIR_SET)
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


def ks_stat(a, b):
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side='right') / float(len(a))
    cdf_b = np.searchsorted(b, allv, side='right') / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def build_dvu(m, sealed_3183):
    """dvu per model, 3157/3183 verbatim (3188 recipe)."""
    A = ANCH_3157[ANCH_KEY[m]]
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
    # rows are ordered (ei, rl, pl) flattened: rl==0 (isa) at i%4 in {0,1}
    RL_rows = np.array([(i // 4, (i % 4) // 2, i % 2) for i in range(64)])
    v_isa = unit_mean(dCn[RL_rows[:, 1] == 0])
    v_hasa = unit_mean(dCn[RL_rows[:, 1] == 1])
    cos_rl = float(v_isa @ v_hasa)
    dv = v_isa - v_hasa
    norm_dv = float(np.linalg.norm(dv))
    E83 = sealed_3183['per_model'][m]
    d83 = max(abs(cos_rl - E83['cos_rl']), abs(norm_dv - E83['norm_dv']))
    assert d83 < REPL_TOL, ('3183 dv replication', m, d83)
    del z, H
    return unit(dv), tc_full, d83


def run_arm(sealed_3169, sealed_3183, sealed_3189, sealed_3190, sealed_3191, sealed_3192):
    out = {}
    pm90 = sealed_3190['per_model']
    pm91 = sealed_3191['per_model']
    for m in MODELS:
        log('=== %s ===' % m)
        dvu, tc_full, dv_drift = build_dvu(m, sealed_3183)
        rd = int(sealed_3169['per_model'][m]['readout'])
        z = np.load(SRC[NPZ69_KEY[m]])
        H16 = z['H']
        NTz, NPz, NHz, Dz = H16.shape
        kout = NHz - 2
        assert kout == rd, ('readout slot', m)
        D = Dz
        Y = H16[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)
        assert np.isfinite(Y).all()

        tr_base = {}
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                          for p in PAIRS if p in tr_pairs]
        oov_rows = ALL_OOV_ROWS
        oov_row_cls = np.array([PAIRS[r % NP_][1] for r in oov_rows])
        seen_row_cls = np.array([PAIRS[r % NP_][1] for r in ALL_SEEN_ROWS])

        def rv(r, ne):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, ne, NC, NT)

        E69 = sealed_3169['per_model'][m]['B']
        E89 = sealed_3189['per_model'][m]
        drift69 = 0.0
        Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        Rseen_rows = np.zeros((len(ALL_SEEN_ROWS), D), dtype=np.float32)
        for si, s in enumerate(SEEDS_S1):
            tr_rows = tr_base[s]
            Xtr = np.stack([rv(r, NE) for r in tr_rows])
            W = ridge_primal(Xtr, Y[tr_rows])
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            te_pairs = split_s1_246(s)[1]
            te_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                       for p in PAIRS if p in te_pairs]
            Xs = np.stack([rv(r, NE) for r in te_rows])
            es = ((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk
            Xo = np.stack([rv(r, NE) for r in oov_rows])
            eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
            drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                          abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
            R_oov = Y[oov_rows] - Xo @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
            # all-seen residuals (amplitude-preserving mean over seeds)
            Xall = np.stack([rv(r, NE) for r in ALL_SEEN_ROWS])
            Rseen_rows += (Y[ALL_SEEN_ROWS] - Xall @ W)
            del W, Xtr, Xall
        assert drift69 < REPL_TOL, ('3169 per-seed E replication', m, drift69)
        Qn = Q_rows / len(SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
        Rseen = Rseen_rows / len(SEEDS_S1)
        Sneak = Rseen / (np.linalg.norm(Rseen, axis=1, keepdims=True) + 1e-18)

        # device anchors: 3189 scalars bitwise
        s0 = np.linalg.svd(Qn, compute_uv=False)
        energy = s0 ** 2
        tot = float(energy.sum())
        top1 = float(energy[0] / tot)
        top2 = float(energy[:2].sum() / tot)
        cum = np.cumsum(energy) / tot
        r80 = int(np.searchsorted(cum, 0.8) + 1)
        r90 = int(np.searchsorted(cum, 0.9) + 1)
        cls_ids = list(range(6, 10))
        V = []
        within = []
        for c in cls_ids:
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

        # 3190/3191 replication
        U, S, Vt = np.linalg.svd(Qn, full_matrices=False)
        ustar = Vt[0]
        loo = []
        for c in cls_ids:
            keep = oov_row_cls != c
            Ql = Qn[keep]
            sl = np.linalg.svd(Ql, compute_uv=False)
            el = sl ** 2
            loo.append(float(el[0] / el.sum()))
        loo_max = float(max(abs(x - top1) for x in loo))
        med_abs = []
        for c in cls_ids:
            held = oov_row_cls == c
            v_hat = unit_mean(Qn[~held])
            cos_held = Qn[held] @ v_hat
            med_abs.append(float(np.median(np.abs(cos_held))))
        E90, E91 = pm90[m], pm91[m]
        drift90 = max(abs(loo_max - E90['loo_max_delta']),
                      abs(top1 - E90['top1']), abs(cross_med - E90['cross_med']),
                      max(abs(a - b) for a, b in zip(loo, E90['loo_top1'])))
        assert drift90 == 0.0, ('3190 scalar drift', m, drift90)
        drift91 = max(abs(float(min(med_abs)) - E91['min_med_abs_cos']),
                      max(abs(a - b) for a, b in zip(med_abs, E91['med_abs_cos_held'])))
        assert drift91 == 0.0, ('3191 scalar drift', m, drift91)

        # (GATE set) component decomposition of u*
        cos_dvu = float(ustar @ dvu)
        vs = []
        for c in range(6):
            idxs = np.where(seen_row_cls == c)[0]
            vs.append(unit_mean(Sneak[idxs]))
        cos_vs = [abs(float(ustar @ v)) for v in vs]
        su = np.linalg.svd(Sneak, compute_uv=False)
        useen = np.linalg.svd(Sneak, full_matrices=False)[2][0]
        cos_useen = abs(float(ustar @ useen))
        gate_max = max([abs(cos_dvu)] + cos_vs + [cos_useen])
        best = 'dvu' if abs(cos_dvu) == gate_max else (
            'u_seen' if cos_useen == gate_max else 'v_s[%d]' % cos_vs.index(gate_max))
        cos_vc = [abs(float(ustar @ v)) for v in V]  # report-only
        out[m] = dict(dv_drift=dv_drift, drift_3169=drift69, drift_3189=drift89,
                      drift_3190=drift90, drift_3191=drift91,
                      cos_dvu=cos_dvu, cos_vs=cos_vs, cos_useen=cos_useen,
                      gate_max=gate_max, best_component=best,
                      cos_vc_report=cos_vc, top1=top1, cross_med=cross_med)
        log('  %s: |cos(u*,dvu)|=%.4f | max|cos(u*,v_s)|=%.4f | |cos(u*,u_seen)|=%.4f | gate_max=%.4f (%s) | report-only |cos(u*,v_c)|=%s'
            % (m, abs(cos_dvu), max(cos_vs), cos_useen, gate_max, best,
               '/'.join('%.3f' % x for x in cos_vc)))
        del z, H16, Y, Q_rows, Qn, Rseen_rows, Rseen, Sneak
    gm = [out[m]['gate_max'] for m in MODELS]
    n_dec = sum(1 for x in gm if x >= G_DECOMP)
    n_nov = sum(1 for x in gm if x < G_NOVEL)
    if n_dec >= 2:
        branch = 'DECOMPOSED'
    elif n_nov >= 2:
        branch = 'NOVEL'
    else:
        branch = 'INTERMEDIATE'
    out['_gate'] = dict(n_decomposed=n_dec, n_novel=n_nov, branch=branch)
    log('GATE: decomp %d/3 (>=0.5), novel %d/3 (<0.3) => %s' % (n_dec, n_nov, branch))
    return out, branch


# ---------------- outputs per branch ----------------
CREATED_STR = None


def finish_outputs(summary, branch, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg18 = json.loads(open(SRC['registry_v118'], 'rb').read().decode('utf-8'))
    gl16 = json.loads(open(SRC['gap_v16'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg18['version'] == '1.18' and len(reg18['features']) == 25
    assert gl16['version'] == '1.6'
    pm = summary['per_model']
    fmt3 = lambda key: '/'.join('%.4f' % pm[m][key] for m in MODELS)
    reg_render, gl_render = reg18, gl16
    disposition = 'report_only_intermediate'
    reg19_sha = None
    gl17_sha = None
    if branch == 'DECOMPOSED':
        gl17 = copy.deepcopy(gl16)
        gl17['version'] = '1.7'
        gl17['supersedes'] = 'gap_ledger_v1_6.json (0f6423e0, Phase 3192)'
        g4_old = [g for g in gl16['gaps'] if g['id'] == 'GAP-4'][0]
        g4 = copy.deepcopy(g4_old)
        g4['mechanism_note'] = g4_old['mechanism_note'] + (
            ' -> 3193 G5-D1 成分分解：u* 可归约为已知机制成分（best |cos| ' +
            fmt3('gate_max') + '，门 ≥0.5 ×≥2/3 模型；降级登记：3165/3166 封存无向量，'
            '成分集 = dvu/seen 类均值/u_seen）——u*_decomposed。锚：p3193 res ' +
            res8 + ' / seal ' + seal8 + '（design ' + design_sha + '）。')
        g4['evidence'] = g4_old['evidence'] + [
            '3193 G5-D1 decomposed: max |cos(u*, gate set)| = ' + fmt3('gate_max')
            + ' >= 0.5 x >=2/3 models (zero GPU; gate set = dvu + seen-class '
            'means + u_seen; 3165/3166 downgrade registered)']
        gl17['gaps'] = [g4 if g['id'] == 'GAP-4' else g for g in gl16['gaps']]
        for g in gl16['gaps']:
            a = json.dumps(g, ensure_ascii=False, sort_keys=True)
            b = json.dumps([x for x in gl17['gaps'] if x['id'] == g['id']][0],
                           ensure_ascii=False, sort_keys=True)
            if g['id'] == 'GAP-4':
                ga = json.loads(a)
                assert b != a and g4['mechanism_note'].startswith(ga['mechanism_note'])
                assert g4['evidence'][:-1] == ga['evidence']
                assert g4['statement'] == ga['statement'] and g4['prereg'] == ga['prereg']
            else:
                assert a == b, ('gap not preserved', g['id'])
        assert gl17['appendix'] == gl16['appendix']
        gp = os.path.join(OUTDIR, 'gap_ledger_v1_7.json')
        with open(gp, 'w', encoding='utf-8', newline='') as f:
            f.write(json.dumps(gl17, ensure_ascii=False, indent=1))
        gl17_sha = sha8_file(gp)
        log('gap ledger v1.7 written sha8 %s (gap4_decomposed)' % gl17_sha)
        gl_render, disposition = gl17, 'v1_6_to_v1_7_gap4_decomposed'
    elif branch == 'NOVEL':
        reg19 = copy.deepcopy(reg18)
        reg19['version'] = '1.19'
        reg19['supersedes'] = 'atlas_registry_v1_18.json (c08cbcc2, Phase 3191)'
        pmv = summary['per_model']
        f26 = {
            'id': 'FTR-26',
            'family': 'context',
            'name': 'u*_novel_direction',
            'statement': ('谱外 OOV 崩塌共享方向 u* 是相对编码几何中的新成分：'
                          '对全部已知结构方向（关系偏移 dvu、seen 类均值残差方向、'
                          'seen 残差主方向）的最大 |cos| = ' +
                          '/'.join('%.4f' % pmv[m]['gate_max'] for m in MODELS) +
                          '（novel 门 <0.3 ×≥2/3 模型）——u* 不可归约为任何已知机制'
                          '方向，独立登记为候选特征。方向级主张，无幅度主张；'
                          'E1_repeatable，held_out/因果升格留专用臂。'),
            'evidence_level': 'E1_repeatable',
            'model_scope': MODELS,
            'anchors': [{
                'src': 'p3193', 'phase': 3193, 'role': 'novel_direction_gate',
                'tags': ['zero_gpu', 'sealed_reuse', 'candidate'],
                'asserts': {
                    'gate_max.qwen3-4b': pmv['qwen3-4b']['gate_max'],
                    'gate_max.qwen3-14b': pmv['qwen3-14b']['gate_max'],
                    'gate_max.glm4-9b': pmv['glm4-9b']['gate_max'],
                    'gate_novel_max_cos': G_NOVEL,
                    'component_downgrade': '3165/3166 sealed scalars only',
                    'res_sha8': res8, 'seal_sha8': seal8}}],
            'counter_evidence': [],
            'replication': '3189-3192 sealed scalar chain drift == 0 bitwise x3',
            'scope_limits': '方向级（unit 残差空间）；降级成分集（3165/3166 无向量）',
            'values': {}}
        reg19['features'] = reg18['features'] + [f26]
        for f in reg18['features']:
            a = json.dumps(f, ensure_ascii=False, sort_keys=True)
            b = json.dumps([x for x in reg19['features'] if x['id'] == f['id']][0],
                           ensure_ascii=False, sort_keys=True)
            assert a == b, ('feature not preserved', f['id'])
        rp = os.path.join(OUTDIR, 'atlas_registry_v1_19.json')
        with open(rp, 'w', encoding='utf-8', newline='') as f:
            f.write(json.dumps(reg19, ensure_ascii=False, indent=1))
        reg19_sha = sha8_file(rp)
        log('registry v1.19 written sha8 %s (ftr26_novel_candidate)' % reg19_sha)
        reg_render, disposition = reg19, 'v1_18_to_v1_19_ftr26_novel_candidate'
    # atlas_v1_21.html (DECOMPOSED/NOVEL only)
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.21 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.21 &mdash; 语义机制图谱</h1>')
    H.append('<div class="sub">Phase 3193 G5-D1 &middot; u* 成分分解（%s） &middot; '
             'registry %s sha8=%s &middot; gap ledger %s sha8=%s &middot; v1.18 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3169 %s + p3157 %s + p3192 %s &middot; 渲染于 %s</div>' % (
                 {'DECOMPOSED': 'u*_decomposed（gap v1.7）',
                  'NOVEL': 'FTR-26 新成分立项（registry v1.19）',
                  'INTERMEDIATE': 'report-only 中间带'}[branch],
                 reg_render['version'], span('meta.reg_render_sha8', reg19_sha or 'c08cbcc2'),
                 gl_render['version'], span('meta.gl_render_sha8', gl17_sha or '0f6423e0'),
                 span('meta.registry_v118_sha8', 'c08cbcc2'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3157_4b', SHA_ANCHOR['p3157_4b']),
                 span('meta.p3192_result', SHA_ANCHOR['p3192_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg_render['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg_render['principles']):
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
    for f in reg_render['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry %s，%d 条）</h2>' % (reg_render['version'], len(reg_render['features'])))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg_render['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg_render['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg_render['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    H.append('<h2>缺口账本 %s（GAP-4 %s）</h2>' % (gl_render['version'], 'quantified_collapse'))
    for g in gl_render['gaps']:
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
    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl_render['appendix']))
    for a in gl_render['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')
    H.append('<div class="footer">RDC/LPF atlas v1.21 &middot; 渲染自 registry %s + gap ledger %s'
             ' &middot; 增量来源: Phase 3193 G5-D1 u* 成分分解'
             '（数据源 p3169 %s + p3157 %s + p3192 %s）'
             '&middot; 本文件为 Phase 3193 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg_render['version'], gl_render['version'],
                 SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3157_4b'],
                 SHA_ANCHOR['p3192_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_21.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v121(reg_render, nodes, gl_render, reg19_sha, gl17_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_21.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3193, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'branch': branch,
                'per_model': summary['per_model']},
        'counts': {'features': len(reg_render['features']), 'failures': len(reg_render['failures']),
                   'upgrades': len(reg_render['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl_render['appendix'])},
        'registry_disposition': disposition,
        'registry_render_version': reg_render['version'],
        'registry_render_sha8': reg19_sha or 'c08cbcc2',
        'gap_render_version': gl_render['version'],
        'gap_render_sha8': gl17_sha or '0f6423e0',
        'html_file': 'atlas_v1_21.html', 'html_sha8': html_sha,
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


def flatten_v121(reg, nodes, gl, reg19_sha, gl17_sha):
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
    d['meta.reg_render_sha8'] = reg19_sha or 'c08cbcc2'
    d['meta.gl_render_sha8'] = gl17_sha or '0f6423e0'
    d['meta.registry_v118_sha8'] = 'c08cbcc2'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3157_4b'] = SHA_ANCHOR['p3157_4b']
    d['meta.p3192_result'] = SHA_ANCHOR['p3192_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3193 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    assert sealed_3169['res_sha8'] == CONTENT_3169[0] and sealed_3169['seal_sha8'] == CONTENT_3169[1]
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    assert sealed_3183['res_sha8'] == CONTENT_3183[0] and sealed_3183['seal_sha8'] == CONTENT_3183[1]
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    assert sealed_3189['res_sha8'] == CONTENT_3189[0] and sealed_3189['seal_sha8'] == CONTENT_3189[1]
    sealed_3190 = json.load(open(SRC['p3190_result'], encoding='utf-8'))
    assert sealed_3190['res_sha8'] == CONTENT_3190[0] and sealed_3190['seal_sha8'] == CONTENT_3190[1]
    sealed_3191 = json.load(open(SRC['p3191_result'], encoding='utf-8'))
    assert sealed_3191['res_sha8'] == CONTENT_3191[0] and sealed_3191['seal_sha8'] == CONTENT_3191[1]
    sealed_3192 = json.load(open(SRC['p3192_result'], encoding='utf-8'))
    assert sealed_3192['res_sha8'] == CONTENT_3192[0] and sealed_3192['seal_sha8'] == CONTENT_3192[1]
    assert abs(float(sealed_3169['gate']['ratio_B']) - RATIO_B_ANCHOR) < 1e-9
    log('sealed results double-locked: 3169/3183/3189/3190/3191/3192')

    primary, branch = run_arm(sealed_3169, sealed_3183, sealed_3189,
                              sealed_3190, sealed_3191, sealed_3192)
    verdict = ('ustar_decomposition_%s (gate_max %s; decomp %d/3 >= 0.5, novel '
               '%d/3 < 0.3; component downgrade registered: 3165/3166 sealed '
               'scalars only)'
               % (branch.lower(),
                  '/'.join('%.4f' % primary[m]['gate_max'] for m in MODELS),
                  primary['_gate']['n_decomposed'], primary['_gate']['n_novel']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3193, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, branch=branch,
                      n_decomposed=primary['_gate']['n_decomposed'],
                      n_novel=primary['_gate']['n_novel']),
        'per_model': {m: primary[m] for m in MODELS},
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
    if not SMOKE and branch in ('DECOMPOSED', 'NOVEL'):
        finish_outputs(summary, branch, res8, seal8, design_sha)
    elif not SMOKE:
        log('branch %s => report-only, no registry/gap output (final_result only)' % branch)
        final = {
            'phase': 3193, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
            'zero_gpu': True, 'branch': branch,
            'arm': {'verdict': verdict, 'per_model': summary['per_model']},
            'sources': SHA_ANCHOR, 'res_sha8': res8, 'seal_sha8': seal8,
            'registry_disposition': 'report_only_intermediate',
        }
        raw = json.dumps(final, ensure_ascii=False, indent=1, sort_keys=True)
        f_res8 = sha8(raw.encode('utf-8'))
        mid = json.dumps(dict(final, res_sha8=f_res8), ensure_ascii=False, indent=1, sort_keys=True)
        f_seal8 = sha8(mid.encode('utf-8'))
        with open(os.path.join(OUTDIR, 'final_result.json'), 'w', encoding='utf-8', newline='') as f:
            f.write(json.dumps(dict(final, res_sha8=f_res8, seal_sha8=f_seal8),
                               ensure_ascii=False, indent=1, sort_keys=True))
        log('final result res=%s seal=%s (report-only)' % (f_res8, f_seal8))
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
