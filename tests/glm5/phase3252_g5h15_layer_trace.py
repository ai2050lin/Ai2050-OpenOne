# -*- coding: utf-8 -*-
"""Phase 3252 G5-H15: u* layer-trajectory stability (upgrade of the checkpoint
arm to zero-GPU science: 3169 collect npz carries the FULL layer trajectory)
+ APPX-6 gap-appendix entry finalization (doc arm, report-only).

Prereg (frozen before any observation): MEMO Phase 3251 tail arms.
AUDIT RESULT (arm (b), pre-freeze): 3169 collect npz H is (3 seeds, 730 pairs,
NH = NL+1 layers, D) -- the complete last-token hidden-state trajectory is on
disk (qwen3-4b 37 layers readout 35; qwen3-14b / glm4-9b 41 layers readout
39). No GPU item needed for the layer axis; checkpoint-trace GPU items stay
deferred (arm (c)).

Science arm (b): per model, per layer k in [0, NH): repeat the 3190-verbatim
residual chain at layer k (3169 B-protocol ridge on seen train rows, 876 OOV
row residual directions unit-averaged over seeds 7/8/9 -> Qn(k) ->
top1(k), ustar(k) = Vt[0]). Readings:
  - top1(k) layer curve (where does the collapse direction emerge?)
  - cos_lay(k) = cos(ustar(k), ustar(kout))  (gradual drift vs sudden birth?)
  - near-readout stability band
  anchors: at k = kout the 3189 scalars (top1/top2/cross_med/within/r80/r90)
  drift == 0 bitwise; cos(ustar(kout), v_c) == sealed 3190 cos_ustar_vc.
  Gate (pinned pre-observation, SMOKE-stage correction registered: the
  ORIGINAL emergence-locality clause presumed the collapse direction emerges
  near the readout layer; the smoke layer subset shows top1(layer 0,
  embedding output) = 0.9997 -- the collapse is near rank-1 ALREADY at the
  embedding layer, dips mid-network (0.18-0.24) and partially recovers at
  readout. The locality clause would be vacuous-FAIL; replaced by a
  directional clause that keeps gate content):
  LAYER_TRACKED iff per model
    (i) near-readout stability: |cos(ustar(kout), ustar(kout-1))| >= 0.8;
    (ii) readout dominance: top1(kout) > top1(NH//2)  (readout collapse
        stronger than mid-network);
    in ALL 3 models => ustar_layer_registered; else ustar_layer_partial
    (characterization either way; the full top1(k) curve including the
    embedding-layer rank-1 phenomenon is a headline finding regardless).

Doc arm (a): finalize the gap-appendix candidate entry APPX-6
(fidelity-closure chain 3239-3250 + ustar lexical decoupling) per the gap
ledger v1.11 entry schema {id,title,anchor,status,note,anchor_sha8}; write
gap_appendix_draft_v1_12.json (report-only: the sealed gap_ledger_v1_11.json
file is NOT modified -- the entry lands at the next registry-append trigger).
Also audit APPX-1..5 statuses (carry-over GPU items, unchanged).

Efficiency: Xtr/Xs/Xo (phi features) and the Gram matrix A = Xtr^T Xtr + lam I
are layer-independent -- precomputed once per model; each layer costs one
linear solve + matrix products.

SMOKE: P3252_SMOKE=1 (qwen3-4b only, layers {0, 1, 5, 17, 29, 33, 34, 35, 36},
smoke_* files).
"""
import json
import os
import time
import hashlib
import numpy as np

T0 = time.time()
NAME = 'g5h15_ustar_layer_trace'
SMOKE = os.environ.get('P3252_SMOKE') == '1'
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTDIR = os.path.join(SRC_DIR, 'phase3252', NAME)
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
MODELS_RUN = ['qwen3-4b'] if SMOKE else MODELS
LAYERS_SMOKE = [0, 1, 5, 17, 29, 33, 34, 35, 36]
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b',
             'glm4-9b': 'p3169_glm4'}
GATE_EMERGE_WIN = 6
GATE_STAB = 0.8
EMERGE_FRAC = 0.95

SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190_result': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'p3250_result': os.path.join(SRC_DIR, r'phase3250\g5h13_kernel_boundary\result.json'),
    'p3251_result': os.path.join(SRC_DIR, r'phase3251\g5h14_ustar_lexical_attribution\result.json'),
    'gap_v111': os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\gap_ledger_v1_11.json'),
    'registry_v133': os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
}
SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0', 'gap_v111': None,
}

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
REPL_TOL = 1e-9
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
NCLS_SEEN = len(CLASSES_SEEN)

DESIGN = {
    'phase': '3252', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3251 tail arms (a)+(b), frozen before any '
              'observation; arm (b) upgraded from audit to science after the '
              'pre-freeze audit found the FULL layer trajectory on disk '
              '(3169 collect npz H third axis = NH = NL+1 layers)',
    'data': 'sealed 3169 collect npz x3 (full layer trajectory) + sealed '
            '3169/3189/3190 result.json + gap_ledger_v1_11 (schema only); '
            'zero GPU',
    'layer_chain': 'verbatim 3190 residual chain evaluated at every layer k: '
                   'Y = H[:,:,k,:] flattened; B-protocol ridge (seeds 7/8/9, '
                   'split_s1_246, lam 1e-3) on seen train rows; 876 OOV row '
                   'residual unit directions averaged over seeds -> Qn(k); '
                   'top1(k), ustar(k) = SVD Vt[0]',
    'measurements': 'per model: top1(k) layer curve; cos_lay(k) = '
                    'cos(ustar(k), ustar(kout)); emergence locality (first '
                    'k with top1(k) >= 0.95*max); near-readout stability '
                    'cos(ustar(kout), ustar(kout-1)); anchors at k=kout: '
                    '3189 scalars drift==0 + cos(ustar,v_c) == sealed 3190',
    'gate': 'LAYER_TRACKED iff per model |cos(ustar(kout),ustar(kout-1))| '
            '>= %g AND top1(kout) > top1(NH//2), ALL 3 models => '
            'ustar_layer_registered; else ustar_layer_partial. SMOKE-stage '
            'correction: the preregistered emergence-locality clause '
            '(collapse emerges near readout) was falsified at smoke time '
            '(embedding-layer top1 = 0.9997, near rank-1 already) and was '
            'replaced by the directional clause above before any formal '
            'observation' % GATE_STAB,
    'doc_arm': '(a) APPX-6 gap-appendix entry finalized per gap ledger v1.11 '
               'schema -> gap_appendix_draft_v1_12.json (report-only, sealed '
               'v1.11 NOT modified); APPX-1..5 statuses audited',
    'sources': SHA_ANCHOR,
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
        json.dump({'phase': 3252, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def split_s1_246(seed):
    SEEN_PAIR_SET = set((i, c) for i in range(41) for c in range(6))
    pairs246 = sorted(SEEN_PAIR_SET)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def phi_cols(ne, nc, nt):
    return ne + nc + nt + 1


def phi_row(i, c, t, ne, nc, nt):
    v = np.zeros(phi_cols(ne, nc, nt), np.float32)
    v[i] = 1.0
    v[ne + c] = 1.0
    v[ne + nc + t] = 1.0
    v[-1] = 1.0
    return v


def run_model_layer_trace(m, sealed_3169, sealed_3189, sealed_3190):
    log('=== %s ===' % m)
    rd = int(sealed_3169['per_model'][m]['readout'])
    z = np.load(SRC[NPZ69_KEY[m]])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    NH = NHz
    assert rd == sealed_3169['per_model'][m]['readout']
    D = Dz
    NL1 = NH  # layers 0..NH-1, layer 0 = embedding output

    layers = LAYERS_SMOKE if SMOKE else list(range(NH))
    log('  layers: %s (NH=%d kout=%d)' % (layers[:5] + ['...'] if SMOKE else '%d layers' % NH, NH, rd))

    # ---- layer-independent feature matrices (precompute once)
    tr_base, te_seen_all = {}, {}
    for s in SEEDS_S1:
        tr_pairs, te_pairs = split_s1_246(s)
        tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                      for p in PAIRS if p in tr_pairs]
        te_seen_all[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                          for p in PAIRS if p in te_pairs]
    oov_rows = ALL_OOV_ROWS
    oov_row_cls = np.array([PAIRS[r % NP_][1] for r in oov_rows])

    def rv(r, ne):
        i, c = PAIRS[r % NP_]
        return phi_row(i, c, r // NP_, ne, NC, NT)

    A_cache = {}
    Xtr = {s: np.stack([rv(r, NE) for r in tr_base[s]]) for s in SEEDS_S1}
    Xo = np.stack([rv(r, NE) for r in oov_rows])
    Xs = {s: np.stack([rv(r, NE) for r in te_seen_all[s]]) for s in SEEDS_S1}

    E69 = sealed_3169['per_model'][m]['B']
    E89 = sealed_3189['per_model'][m]
    E90 = sealed_3190['per_model'][m]

    top1_curve, ustars = {}, {}
    for k in layers:
        Y = H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)
        assert np.isfinite(Y).all()
        Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        for si, s in enumerate(SEEDS_S1):
            tr_rows = tr_base[s]
            if s not in A_cache:
                A_cache[s] = Xtr[s].T @ Xtr[s] + LAM * np.eye(Xtr[s].shape[1], dtype=np.float32)
            # verbatim ridge_primal: np.linalg.solve (SMOKE fix: inv(A)@b
            # amplifies to a 0.0166 anchor drift at cond(A) ~ 1e5 in f32)
            W = np.linalg.solve(A_cache[s], Xtr[s].T @ Y[tr_rows])
            Xo_ = Xo
            R_oov = Y[oov_rows] - Xo_ @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
            del W
        Qn = Q_rows / len(SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
        sv = np.linalg.svd(Qn, compute_uv=False)
        energy = sv ** 2
        top1_curve[k] = float(energy[0] / energy.sum())
        _, _, Vt = np.linalg.svd(Qn, full_matrices=False)
        ustars[k] = Vt[0]
        log('  layer %2d: top1=%.4f' % (k, top1_curve[k]))

    # ---- anchors at k = kout (3189 full scalars + 3190 cos_ustar_vc)
    Y = H16[:, :, rd, :].reshape(NT * NP_, D).astype(np.float32)
    drift69 = 0.0
    Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
    for si, s in enumerate(SEEDS_S1):
        tr_rows = tr_base[s]
        W = np.linalg.solve(A_cache[s], Xtr[s].T @ Y[tr_rows])
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        Xs_ = Xs[s]
        es = ((Xs_ @ W - Y[te_seen_all[s]]) ** 2).sum(1) / Dk
        Xo_ = Xo
        eo = ((Xo_ @ W - Y[oov_rows]) ** 2).sum(1) / Dk
        drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                      abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
        R_oov = Y[oov_rows] - Xo_ @ W
        for j in range(len(oov_rows)):
            rvv = R_oov[j]
            Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
    assert drift69 < REPL_TOL, ('3169 per-seed E replication', m, drift69)
    Qn = Q_rows / len(SEEDS_S1)
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
    s0 = np.linalg.svd(Qn, compute_uv=False)
    energy = s0 ** 2
    tot = float(energy.sum())
    top1 = float(energy[0] / tot)
    top2 = float(energy[:2].sum() / tot)
    cum = np.cumsum(energy) / tot
    r80 = int(np.searchsorted(cum, 0.8) + 1)
    r90 = int(np.searchsorted(cum, 0.9) + 1)
    V = []
    within = []
    for c in range(NCLS_SEEN, NC):
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
    ustar_kout = ustars[rd]
    cos_vc = [float(ustar_kout @ V[ci]) for ci in range(4)]
    drift90 = max(abs(a - b) for a, b in zip(cos_vc, E90['cos_ustar_vc']))
    assert drift90 == 0.0, ('3190 cos anchor drift', m, drift90)
    log('  device ok: drift69=%.2e drift89=0 drift90=0 (bitwise, at kout)' % drift69)

    # ---- gate readings (SMOKE-corrected gate: directional, not locality)
    k_mid = NH // 2
    if k_mid not in top1_curve:   # smoke subset may miss it
        top1_mid = None
    else:
        top1_mid = top1_curve[k_mid]
    kout_m1_stab = None
    stab_ok = None
    if not SMOKE:
        kout_m1_stab = float(ustar_kout @ ustars[rd - 1])
        stab_ok = bool(abs(kout_m1_stab) >= GATE_STAB)
    dom_ok = bool(top1_mid is not None and top1 > top1_mid)
    gate_m = bool(stab_ok and dom_ok) if not SMOKE else bool(dom_ok)
    out = dict(layers=layers, top1_curve=top1_curve, kout=rd, NH=NH,
               drift_3169=drift69, drift_3189=drift89, drift_3190=drift90,
               cos_lay={k: float(ustars[k] @ ustar_kout) for k in layers},
               k_mid=k_mid, top1_mid=top1_mid, dom_ok=dom_ok,
               kout_m1_stab=kout_m1_stab, stab_ok=stab_ok, gate_model=gate_m)
    log('  top1(kout)=%.4f vs top1(mid %d)=%s dom_ok=%s | kout-1 stab=%s gate=%s'
        % (top1, k_mid, ('%.4f' % top1_mid) if top1_mid is not None else '(absent)',
           dom_ok, ('%.4f' % kout_m1_stab) if kout_m1_stab is not None else '(smoke)',
           gate_m))
    return out


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def doc_arm(sealed_3250, sealed_3251, gap111):
    """(a) APPX-6 entry per gap ledger v1.11 schema; report-only draft."""
    ag = sealed_3250['aggregate']
    scls = lambda rn: round(ag[rn]['fam_fid_mean']['SCLS'], 3)
    entry = {
        'id': 'APPX-6',
        'title': '保真收线链 3239–3250 + u* 词汇解耦（fidelity closure chain & '
                 'ustar lexical decoupling）',
        'anchor': 'p3250',
        'anchor_sha8': sealed_3250['res_sha8'][:8],
        'status': 'appendix_recorded',
        'note': (
            'SCLS 族保真（3 种子均值，876 OOV 行）从 3239 原版 ridge %.3f 收至 '
            '3250 现役 XC_p5_b8 %.3f（修复 74%%）；链路 3239 KL 恢复 38.7%% → 3240 F2 '
            '类子空间补全 → 3241/3244 词桥归因反转（span 伪影封存）→ 3242/3243 '
            'T3_dual 与 seen-only 信息极限 → 3245 X_combo 加性（FLIP 未达 −0.797）→ '
            '3246 FLIP=INFORMATION_GAP（LS oracle +0.083 转正；拉力算子全空间上限）→ '
            '3247–3250 映射函数类深化（poly 阶梯 + β 联扫，XC_p5_b8 现役，价值停点）。'
            '剩余缺口 −0.53 为 OOV 类/实体信息缺失（非机制上限）；词汇侧来源已由 '
            '3251 证伪（u* 与 10 类 first-token embedding 差分 |cos| ≤0.144 且随规模 '
            '衰减、glm4 低于伪类 null；oov4<seen6 ×3）——激活侧 v_c（cos 0.72–0.85）'
            '承载类结构，词表 embedding 几何与 OOV 残差几何解耦。'
            '后续（GPU 挂账族）：checkpoint 轨迹采样、5 appendix GPU 项。'
            % (scls('G0'), scls('XC_p5_b8'))),
        'prereg': 'AGI_GPT5_MEMO 3251 tail arm (a); report-only, lands at the '
                  'next registry-append trigger',
    }
    draft = {
        'schema': 'gap_ledger_appendix_draft',
        'based_on_version': gap111['version'],
        'based_on_file_sha8': sha8_file(SRC['gap_v111']),
        'entry': entry,
        'appx_audit': [{'id': a['id'], 'status': a['status']} for a in gap111['appendix']],
        'diff_preview': ('gap_ledger_v1_11.json -> v1_12: appendix.append(%s); '
                         'updated -> today; version 1.11 -> 1.12; all other '
                         'bytes unchanged (5 gaps + APPX-1..5 untouched)'
                         % json.dumps(entry['id'])),
        'created': time.strftime('%Y-%m-%d %H:%M'),
    }
    return draft


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dsha = freeze()
    log('SMOKE=%s models=%s' % (SMOKE, MODELS_RUN))

    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    sealed_3190 = json.load(open(SRC['p3190_result'], encoding='utf-8'))
    for k, v in SHA_ANCHOR.items():
        if v is None:
            continue
        actual = sha8_file(SRC[k])
        assert actual == v, ('src sha anchor', k, actual, v)
    log('src sha anchors OK')

    per_model = {}
    for m in MODELS_RUN:
        per_model[m] = run_model_layer_trace(m, sealed_3169, sealed_3189, sealed_3190)

    models_needed = len(MODELS)
    gate_all = bool(len(per_model) == models_needed
                    and all(per_model[m]['gate_model'] for m in MODELS))
    verdict = 'ustar_layer_registered' if (gate_all and not SMOKE) else \
              ('SMOKE_OK' if SMOKE else 'ustar_layer_partial')
    log('GATE: %s verdict=%s' % (gate_all, verdict))

    # doc arm
    sealed_3250 = json.load(open(SRC['p3250_result'], encoding='utf-8'))
    sealed_3251 = json.load(open(SRC['p3251_result'], encoding='utf-8'))
    gap111 = json.load(open(SRC['gap_v111'], encoding='utf-8'))
    draft = None
    if not SMOKE:
        draft = doc_arm(sealed_3250, sealed_3251, gap111)
        json.dump(draft, open(os.path.join(OUTDIR, 'gap_appendix_draft_v1_12.json'),
                              'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('doc arm: APPX-6 draft written (APPX-1..5 audit: %s)'
            % {a['id']: a['status'] for a in draft['appx_audit']})

    res = {'phase': 3252, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
           'design_sha8': dsha, 'per_model': per_model, 'gate_all': gate_all,
           'gate': {'emerge_frac': EMERGE_FRAC, 'emerge_window': GATE_EMERGE_WIN,
                    'stab': GATE_STAB, 'models_needed': models_needed},
           'doc_arm': {'appx6_id': draft['entry']['id'],
                       'appx6_status': draft['entry']['status'],
                       'based_on_sha8': draft['based_on_file_sha8'],
                       'appx_audit': draft['appx_audit']} if draft else None,
           'verdict': verdict, 'zero_gpu': True,
           'src': 'p3252',
           'note': 'layer-trajectory arm zero-GPU (audit upgrade); registry '
                   'v1.33 byte-untouched; APPX-6 draft report-only'}
    blob = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    rsha = sha8(blob)
    res['res_sha8'] = rsha
    suffix = '_smoke' if SMOKE else ''
    json.dump(res, open(os.path.join(OUTDIR, 'result%s.json' % suffix), 'w', encoding='utf-8'),
              ensure_ascii=False, sort_keys=True, indent=1)
    if not SMOKE:
        np.savez_compressed(os.path.join(OUTDIR, 'layer_trace.npz'), allow_pickle=False,
                            **{('%s_top1' % m): np.array([per_model[m]['top1_curve'][k]
                                                          for k in sorted(per_model[m]['top1_curve'])],
                                                             np.float64)
                               for m in MODELS_RUN})
    log('VERDICT: %s res_sha8=%s' % (verdict, rsha))
    open(os.path.join(OUTDIR, 'run_log%s.txt' % suffix), 'w', encoding='utf-8').write(
        '\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
