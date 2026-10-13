# -*- coding: utf-8 -*-
"""Phase 3195 (G5-D3): u* content attribution deepening arm (zero GPU, sealed
3169/3189-3194 reuse; report-only closure - registry v1.20 UNCHANGED both
branches per prereg).

Per prereg in MEMO Phase 3194 tail (frozen before any observation):
  Test whether u* is spanned by candidate structure directions NOT in the
  3193 gate set: candidate span C = {dvu, v_s x6, u_seen, v_c x4,
  KOUT top-k (k=10 right singular vectors of mean-centered 876-row raw KOUT
  matrix Y; the raw manifold ridge)} per model (22 directions total).
  Metric: residual ratio = ||u* - P_C u*|| / ||u*|| (u* unit => ratio =
  sqrt(1 - ||P_C u*||^2)).
  GATE (pinned pre-observation): PARTIALLY_DECOMPOSED iff ratio < 0.5 in
  3/3 models => FTR-26 statement attribution note (recorded in ledger #347
  detail + MEMO only; registry v1.20 file byte-UNTOUCHED, no version bump,
  per prereg "registry 不动"). Otherwise NOVEL_CORE confirmed maintained +
  top contribution spectrum reported; registry v1.20 byte-UNTOUCHED.
  Control anchors (3193 protocol re-derivation, drift == 0 bitwise):
  gate_max 0.2130/0.3303/0.2457, |cos(u*,dvu)| 0.1064/0.0405/0.0240,
  cos_vs x6, cos_useen, cos_vc x4 (report-only in 3193, here re-derived).
  Device anchors: 3189/3190/3191/3193/3194 sealed scalars drift == 0
  bitwise; 3157 tc + 3183 dv; 3169 per-seed E.
  SMOKE (env P3195_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
NAME = 'g5d3_ustar_attribution'
OUTDIR = os.path.join(SRC_DIR, 'phase3195', NAME)
SMOKE = os.environ.get('P3195_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}
NPZ57_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

G_RATIO = 0.5
K_KOUT = 10
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'residual_ratio': G_RATIO, 'k_kout': K_KOUT, 'models_needed': 3,
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
    'p3193_result': os.path.join(SRC_DIR, r'phase3193\g5d1_ustar_decomposition\result.json'),
    'p3194_result': os.path.join(SRC_DIR, r'phase3194\g5d2_ftr26_heldout_upgrade\result.json'),
    'registry_v120': os.path.join(SRC_DIR, r'phase3194\g5d2_ftr26_heldout_upgrade\atlas_registry_v1_20.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3183_result': '0ab70579',
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0', 'p3190_result': '6a8893ae',
    'p3191_result': 'a218ead4', 'p3192_result': '5d8d4858', 'p3193_result': '7c841a3f',
    'p3194_result': '2a48f16f', 'registry_v120': '2aac7e5e',
    'registry_3162': '00f15e98',
}
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3189 = ('3a26f541', 'fb8c1f65')
CONTENT_3190 = ('24461856', '0c975f5b')
CONTENT_3191 = ('836fe933', 'a66c0b3d')
CONTENT_3192 = ('6221fe00', '58c4f183')
CONTENT_3193 = ('5465dd47', 'ce1d8c8d')
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

# ---------------- DESIGN (static; hashed EXCLUDING created/design_sha8) --------
DESIGN = {
    'phase': 3195, 'name': NAME, 'line': 'G5-D3', 'zero_gpu': True,
    'mode': 'FULL', 'prereg_id': 'G5-D3',
    'gate': GATE,
    'candidates': ['dvu', 'v_s_x6', 'u_seen', 'v_c_x4', 'kout_top%d' % K_KOUT],
    'metric': 'residual_ratio = ||u* - P_C u*|| / ||u*|| (u* unit)',
    'kout_recipe': ('top-%d right singular vectors of mean-centered raw 876-row '
                    'KOUT matrix Y (= H16[:, :, kout, :] flattened; the raw '
                    'manifold ridge, NOT residuals)' % K_KOUT),
    'verdicts': {
        'PARTIALLY_DECOMPOSED': 'ratio < 0.5 in 3/3 models => FTR-26 statement '
                                'attribution note in ledger+MEMO; registry v1.20 '
                                'file byte-UNTOUCHED (prereg: registry 不动)',
        'NOVEL_CORE': 'otherwise => NOVEL_CORE maintained + top contribution '
                      'spectrum reported; registry v1.20 byte-UNTOUCHED',
    },
    'control_anchors': {
        '3193_gate_set': 'gate_max / cos_dvu / cos_vs x6 / cos_useen / '
                         'cos_vc_report x4 re-derived, drift == 0 bitwise',
        '3191': 'med_abs_cos_held + min bitwise (3194-verified)',
        'device': '3189/3190/3193 sealed scalars drift == 0 bitwise; '
                  '3157 tc; 3183 dv; 3169 per-seed E',
    },
    'sources': sorted(SHA_ANCHOR.keys()),
    'sha_anchor': SHA_ANCHOR,
    'content_locks': {'p3169': CONTENT_3169, 'p3189': CONTENT_3189,
                      'p3190': CONTENT_3190, 'p3191': CONTENT_3191,
                      'p3192': CONTENT_3192, 'p3193': CONTENT_3193,
                      'p3183': CONTENT_3183},
}
_des = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1)


def log(msg):
    LOG.append('[%7.1f] %s' % (time.time() - T0, msg))
    print(LOG[-1])


def freeze():
    dsha = hashlib.sha256(_des.encode('utf-8')).hexdigest()
    exe_p = os.path.join(OUTDIR, 'execution.json')
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3195, 'name': NAME, 'design_sha8': dsha[:8],
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


def build_dvu(m, sealed_3183):
    """dvu per model, 3157/3183 verbatim (3188 recipe; 3193 row grouping)."""
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


def run_arm(sealed_3169, sealed_3183, sealed_3189, sealed_3190,
            sealed_3191, sealed_3193, sealed_3194):
    out = {}
    pm90 = sealed_3190['per_model']
    pm91 = sealed_3191['per_model']
    pm93 = sealed_3193['per_model']
    pm94 = sealed_3194['per_model']
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

        # 3190/3191/3193/3194 replication
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
        med_ustar = []
        for c in cls_ids:
            held = oov_row_cls == c
            cos_u = Qn[held] @ ustar
            med_ustar.append(float(np.median(np.abs(cos_u))))
        E90, E91, E93, E94 = pm90[m], pm91[m], pm93[m], pm94[m]
        drift90 = max(abs(loo_max - E90['loo_max_delta']),
                      abs(top1 - E90['top1']), abs(cross_med - E90['cross_med']),
                      max(abs(a - b) for a, b in zip(loo, E90['loo_top1'])))
        assert drift90 == 0.0, ('3190 scalar drift', m, drift90)
        drift91 = max(abs(float(min(med_abs)) - E91['min_med_abs_cos']),
                      max(abs(a - b) for a, b in zip(med_abs, E91['med_abs_cos_held'])))
        assert drift91 == 0.0, ('3191 control-anchor drift', m, drift91)
        # 3193 gate set re-derivation (control anchor, bitwise)
        cos_dvu = float(ustar @ dvu)
        vs = []
        for c in range(6):
            idxs = np.where(seen_row_cls == c)[0]
            vs.append(unit_mean(Sneak[idxs]))
        cos_vs = [abs(float(ustar @ v)) for v in vs]
        useen = np.linalg.svd(Sneak, full_matrices=False)[2][0]
        cos_useen = abs(float(ustar @ useen))
        gate_max = max([abs(cos_dvu)] + cos_vs + [cos_useen])
        cos_vc = [abs(float(ustar @ v)) for v in V]
        drift93 = max(abs(gate_max - E93['gate_max']),
                      abs(abs(cos_dvu) - abs(E93['cos_dvu'])),
                      abs(cos_useen - E93['cos_useen']),
                      abs(top1 - E93['top1']),
                      max(abs(a - b) for a, b in zip(cos_vs, E93['cos_vs'])),
                      max(abs(a - b) for a, b in zip(cos_vc, E93['cos_vc_report'])))
        assert drift93 == 0.0, ('3193 gate-set drift', m, drift93)
        drift94 = max(abs(float(min(med_ustar)) - E94['min_med_abs_cos_ustar']),
                      max(abs(a - b) for a, b in zip(med_ustar, E94['med_abs_cos_ustar'])))
        assert drift94 == 0.0, ('3194 arm drift', m, drift94)
        log('  device ok: drift69=%.2e drift89/90/91/93/94=0 (dv_drift=%.2e)'
            % (drift69, dv_drift))

        # (GATE arm) candidate span projection
        Cdirs = [('dvu', dvu)]
        for c in range(6):
            Cdirs.append(('v_s[%d]' % c, vs[c]))
        Cdirs.append(('u_seen', useen))
        for j, c in enumerate(cls_ids):
            Cdirs.append(('v_c[%d]' % c, V[j]))
        Yc = Y - Y.mean(0, keepdims=True)
        _, _, Vtk = np.linalg.svd(Yc, full_matrices=False)
        for k in range(K_KOUT):
            Cdirs.append(('kout_pc[%d]' % k, Vtk[k]))
        assert len(Cdirs) == 22, ('candidate count', len(Cdirs))
        Cmat = np.stack([d for _, d in Cdirs]).astype(np.float64)
        # orthonormalize (candidates may be collinear, e.g. v_c inside kout PCs)
        Qc, _ = np.linalg.qr(Cmat.T)
        Pc = ustar.astype(np.float64) @ Qc
        proj_norm2 = float((Pc ** 2).sum())
        ratio = float(np.sqrt(max(0.0, 1.0 - proj_norm2)))
        contrib = sorted([(_n, abs(float(_d @ ustar))) for _n, _d in Cdirs],
                         key=lambda t: -t[1])
        top5 = contrib[:5]
        gate_m = bool(ratio < G_RATIO)
        out[m] = dict(dv_drift=dv_drift, drift_3169=drift69, drift_3189=drift89,
                      drift_3190=drift90, drift_3191=drift91,
                      drift_3193=drift93, drift_3194=drift94,
                      residual_ratio=ratio, proj_norm=float(np.sqrt(proj_norm2)),
                      gate_model=gate_m, top_contrib=top5,
                      gate_max_3193=gate_max, cos_dvu=cos_dvu,
                      top1=top1, cross_med=cross_med)
        log('  %s: ratio=%.4f (proj %.4f, gate <%.2f)=%s | top5 %s'
            % (m, ratio, np.sqrt(proj_norm2), G_RATIO, gate_m,
               ', '.join('%s %.3f' % (n, c) for n, c in top5)))
        del z, H16, Y, Q_rows, Qn, Rseen_rows, Rseen, Sneak
    n_pass = sum(1 for m in MODELS if out[m]['gate_model'])
    decomposed = bool(n_pass >= 3)
    out['_gate'] = dict(n_models_pass=n_pass, decomposed=decomposed)
    log('GATE: %d/3 models ratio < %.2f => %s'
        % (n_pass, G_RATIO,
           'PARTIALLY_DECOMPOSED' if decomposed else 'NOVEL_CORE_MAINTAINED'))
    return out, decomposed


def main():
    design_sha = freeze()
    for k, want in SHA_ANCHOR.items():
        got = hashlib.sha256(open(SRC[k], 'rb').read()).hexdigest()[:8]
        assert got == want, ('SRC sha', k, got, want)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))

    def load_sealed(d, ckey):
        r = json.load(open(SRC[d], encoding='utf-8'))
        assert r['res_sha8'] == ckey[0] and r['seal_sha8'] == ckey[1], (d, ckey)
        return r

    sealed_3169 = load_sealed('p3169_result', CONTENT_3169)
    sealed_3183 = load_sealed('p3183_result', CONTENT_3183)
    sealed_3189 = load_sealed('p3189_result', CONTENT_3189)
    sealed_3190 = load_sealed('p3190_result', CONTENT_3190)
    sealed_3191 = load_sealed('p3191_result', CONTENT_3191)
    sealed_3192 = load_sealed('p3192_result', CONTENT_3192)
    sealed_3193 = load_sealed('p3193_result', CONTENT_3193)
    sealed_3194 = load_sealed('p3194_result', ('f4836e97', 'be9ce95b'))
    log('sealed results double-locked: 3169/3183/3189/3190/3191/3192/3193/3194')

    res, decomposed = run_arm(sealed_3169, sealed_3183, sealed_3189,
                              sealed_3190, sealed_3191, sealed_3193,
                              sealed_3194)

    if SMOKE:
        out_json = {'phase': 3195, 'name': NAME, 'mode': 'SMOKE',
                    'design_sha8': design_sha, 'per_model': res,
                    'zero_gpu': True}
        raw = json.dumps(out_json, ensure_ascii=False, indent=1, sort_keys=True)
        r8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
        mid = json.dumps(dict(out_json, res_sha8=r8), ensure_ascii=False,
                         indent=1, sort_keys=True)
        s8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
        with open(os.path.join(OUTDIR, 'smoke_result.json'), 'w',
                  encoding='utf-8', newline='') as f:
            f.write(json.dumps(dict(out_json, res_sha8=r8, seal_sha8=s8),
                               ensure_ascii=False, indent=1, sort_keys=True))
        log('SMOKE result written res=%s seal=%s' % (r8, s8))
        return

    verdict = ('ustar_attribution_%s (residual_ratio %s < 0.5 %d/3; top1 '
               '%.4f/%.4f/%.4f; registry v1.20 UNTOUCHED report-only)'
               % ('PARTIALLY_DECOMPOSED' if decomposed else 'NOVEL_CORE_MAINTAINED',
                  '/'.join('%.4f' % res[m]['residual_ratio'] for m in MODELS),
                  res['_gate']['n_models_pass'],
                  res['qwen3-4b']['top1'], res['qwen3-14b']['top1'],
                  res['glm4-9b']['top1']))
    out_json = {'phase': 3195, 'name': NAME, 'mode': 'FULL',
                'design_sha8': design_sha, 'per_model': res,
                'verdict': verdict, 'zero_gpu': True,
                'registry_disposition': 'v1_20_unchanged_report_only'}
    raw = json.dumps(out_json, ensure_ascii=False, indent=1, sort_keys=True)
    r8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    mid = json.dumps(dict(out_json, res_sha8=r8), ensure_ascii=False,
                     indent=1, sort_keys=True)
    s8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
    with open(os.path.join(OUTDIR, 'result.json'), 'w',
              encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(out_json, res_sha8=r8, seal_sha8=s8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (r8, s8))

    final = {'phase': 3195, 'name': NAME, 'mode': 'FULL',
             'design_sha8': design_sha, 'zero_gpu': True,
             'arm': {'decomposed': decomposed,
                     'n_models_pass': res['_gate']['n_models_pass'],
                     'verdict': verdict},
             'registry_disposition': 'v1_20_unchanged_report_only',
             'registry_v120_file': 'atlas_registry_v1_20.json',
             'registry_v120_sha8': SHA_ANCHOR['registry_v120'],
             'registry_v120_recheck_sha8':
                 hashlib.sha256(open(SRC['registry_v120'], 'rb').read()).hexdigest()[:8],
             'counts': {'features': 26, 'upgrades': 8, 'failures': 19,
                        'gaps': 4, 'appendix': 5},
             'sources': SHA_ANCHOR,
             'res_sha8': None, 'seal_sha8': None}
    raw_f = json.dumps(final, ensure_ascii=False, indent=1, sort_keys=True)
    f_res8 = hashlib.sha256(raw_f.encode('utf-8')).hexdigest()[:8]
    mid = json.dumps(dict(final, res_sha8=f_res8), ensure_ascii=False,
                     indent=1, sort_keys=True)
    f_seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
    with open(os.path.join(OUTDIR, 'final_result.json'), 'w',
              encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(final, res_sha8=f_res8, seal_sha8=f_seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('final result res=%s seal=%s' % (f_res8, f_seal8))


ALL_OOV_ROWS = None
NE = NC = NP_ = PAIRS = PI_OF_PAIR = None

# ---- derive row universe (verbatim 3193 recipe) ----
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

if __name__ == '__main__':
    main()
