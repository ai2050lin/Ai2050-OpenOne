# -*- coding: utf-8 -*-
"""Phase 3253 G5-H16: u* subspace mechanism arm (U-shape trajectory: what does
the mid-network dip consist of?) + registry-append trigger evaluation (arm (b),
report-only).

Prereg (frozen before any observation): MEMO Phase 3252 tail arms.
  (a) SCIENCE ARM (zero GPU): per model, repeat the 3190-verbatim residual
      chain at every layer k (as 3252), but keep the TOP-R ORTHONORMAL BASIS
      of Qn(k) (R = 8, pinned) in addition to top1(k):
        - k_mid = argmin top1(k) over k in [1, kout] (the dip layer)
        - principal-angle spectra S(a,b) = singular values of B_a @ B_b^T
          for pairs (emb=layer0, mid=k_mid, out=kout), R values each
        - sigma1(a,b) = S(a,b)[0]; ovr(a,b) = mean(S^2) (subspace overlap)
        - focus curves: ov_out(k) = |B(k) ustar_out|^2 and
          ov_emb(k) = |B(k) ustar_emb|^2 over the full layer axis
        - anchors: drift69/89/90 == 0.0 at kout (3169/3189/3190, bitwise,
          same as 3252) + drift_3252 == 0.0 (top1 curve bitwise-equal to the
          sealed 3252 result -- same verbatim chain)
      GATE (pinned pre-observation, directional): SUBSPACE_TRACKED iff per
      model
        (i)  continuity: sigma1(mid,out) >= 0.3 -- the readout collapse
             direction is PARTIALLY present at the dip (attenuation of one
             axis, not replacement by a foreign direction);
        (ii) mid-readout affinity: sigma1(mid,out) > sigma1(mid,emb);
      ALL 3 models => ustar_subspace_registered; else ustar_subspace_partial
      (characterization either way).
  (b) REGISTRY-APPEND TRIGGER EVALUATION (report-only): are the 3239-3252
      closure phases + APPX-6 draft + new incumbent enough to trigger the
      registry v1.33 -> v1.34 append?  Preregistered criteria (ALL required):
        c1: >= 6 closed ledger phases in [3239, 3252];
        c2: APPX-6 draft present (gap_appendix_draft_v1_12.json, based_on
            sealed gap v1.11 sha 5934631b, status appendix_recorded);
        c3: no blocked/failed verdict among those phases;
        c4: sealed 3250 result carries the incumbent XC_p5_b8;
        c5: registry v1.33 file byte-sha still a30215fd.
      Output: registry_append_preview_v1_34.json (planned diff ONLY -- the
      sealed registry and gap ledger are NOT modified).
  (c) 5 appendix GPU items stay deferred.

SMOKE: P3253_SMOKE=1 (qwen3-4b only, layers {0,1,5,17,18,29,33,34,35,36},
smoke_* files; gate not evaluated in smoke).
"""
import json
import os
import time
import hashlib
import numpy as np

T0 = time.time()
NAME = 'g5h16_ustar_subspace'
SMOKE = os.environ.get('P3253_SMOKE') == '1'
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTDIR = os.path.join(SRC_DIR, 'phase3253', NAME)
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
MODELS_RUN = ['qwen3-4b'] if SMOKE else MODELS
LAYERS_SMOKE = [0, 1, 5, 17, 18, 29, 33, 34, 35, 36]
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b',
             'glm4-9b': 'p3169_glm4'}
R_BASIS = 8
GATE_CONT = 0.3

SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190_result': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'p3250_result': os.path.join(SRC_DIR, r'phase3250\g5h13_kernel_boundary\result.json'),
    'p3252_result': os.path.join(SRC_DIR, r'phase3252\g5h15_ustar_layer_trace\result.json'),
    'appx6_draft': os.path.join(SRC_DIR, r'phase3252\g5h15_ustar_layer_trace\gap_appendix_draft_v1_12.json'),
    'gap_v111': os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\gap_ledger_v1_11.json'),
    'registry_v133': os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
}
SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0',
    'p3252_result': 'd45c08e9', 'appx6_draft': 'c3de8895',
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
    'phase': '3253', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3252 tail arms (a)+(b), frozen before any '
              'observation',
    'data': 'sealed 3169 collect npz x3 (full layer trajectory) + sealed '
            '3169/3189/3190/3250/3252 result.json + gap_appendix_draft_v1_12 '
            '+ gap_ledger_v1_11 + registry_v1_33 + atlas ledger; zero GPU',
    'science_arm': 'per model, 3190-verbatim residual chain at every layer k '
                   '(as 3252); keep top-R basis B(k) of Qn(k), R=%d pinned; '
                   'k_mid = argmin top1(k) over k in [1, kout]; principal '
                   'angle spectra S(a,b) = svd(B_a @ B_b^T) for pairs '
                   '(emb=0, mid, out=kout); sigma1/ovr; focus curves '
                   'ov_out(k) = |B(k) ustar_out|^2, ov_emb(k) similarly; '
                   'anchors drift69/89/90 == 0.0 at kout + drift_3252 == 0.0 '
                   '(top1 curve bitwise-equal to sealed 3252)' % R_BASIS,
    'gate': 'SUBSPACE_TRACKED iff per model sigma1(mid,out) >= %g (continuity: '
            'the readout collapse direction is partially present at the dip) '
            'AND sigma1(mid,out) > sigma1(mid,emb) (mid-readout affinity), '
            'ALL 3 models => ustar_subspace_registered; else '
            'ustar_subspace_partial' % GATE_CONT,
    'eval_arm': '(b) registry-append trigger evaluation, report-only: '
                'preregistered criteria c1 (>=6 closed ledger phases in '
                '[3239,3252]) AND c2 (APPX-6 draft based_on 5934631b) AND c3 '
                '(no blocked/failed verdict in those phases) AND c4 (sealed '
                '3250 carries incumbent XC_p5_b8) AND c5 (registry v1.33 '
                'byte sha a30215fd) -> registry_append_preview_v1_34.json '
                '(planned diff only; sealed files NOT modified)',
    'gpu_arm': '(c) 5 appendix GPU items stay deferred',
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
        json.dump({'phase': 3253, 'name': NAME, 'design_sha8': dsha[:8],
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


def unit_mean(rows_dn):
    v = rows_dn.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def run_model_subspace(m, sealed_3169, sealed_3189, sealed_3190, sealed_3252):
    log('=== %s ===' % m)
    rd = int(sealed_3169['per_model'][m]['readout'])
    z = np.load(SRC[NPZ69_KEY[m]])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    NH = NHz
    D = Dz

    layers = LAYERS_SMOKE if SMOKE else list(range(NH))
    log('  layers: %s (NH=%d kout=%d R=%d)'
        % (layers if SMOKE else '%d layers' % NH, NH, rd, R_BASIS))

    # ---- layer-independent feature matrices (verbatim 3190/3252 precompute)
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
    S52 = sealed_3252['per_model'][m]['top1_curve']

    top1_curve, basis, ustar1 = {}, {}, {}
    for k in layers:
        Y = H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)
        assert np.isfinite(Y).all()
        Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        for si, s in enumerate(SEEDS_S1):
            tr_rows = tr_base[s]
            if s not in A_cache:
                A_cache[s] = Xtr[s].T @ Xtr[s] + LAM * np.eye(Xtr[s].shape[1], dtype=np.float32)
            W = np.linalg.solve(A_cache[s], Xtr[s].T @ Y[tr_rows])
            Xo_ = Xo
            R_oov = Y[oov_rows] - Xo_ @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
            del W
        Qn = Q_rows / len(SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
        _, sv, Vt = np.linalg.svd(Qn, full_matrices=False)
        energy = sv ** 2
        top1_curve[k] = float(energy[0] / energy.sum())
        basis[k] = Vt[:R_BASIS].copy()
        ustar1[k] = Vt[0].copy()
        del Vt
        # cross-phase anchor: bitwise equality with sealed 3252 curve
        if str(k) in S52 or k in S52:
            ref = S52[str(k)] if str(k) in S52 else S52[k]
            assert top1_curve[k] == ref, ('3252 top1 drift', m, k, top1_curve[k], ref)
        log('  layer %2d: top1=%.4f' % (k, top1_curve[k]))
    drift_3252 = 0.0
    log('  device ok: top1 curve bitwise == sealed 3252 (drift_3252=0)')

    # ---- anchors at k = kout (3169/3189/3190, verbatim 3252 anchor block)
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
    ustar_kout = ustar1[rd]
    cos_vc = [float(ustar_kout @ V[ci]) for ci in range(4)]
    drift90 = max(abs(a - b) for a, b in zip(cos_vc, E90['cos_ustar_vc']))
    assert drift90 == 0.0, ('3190 cos anchor drift', m, drift90)
    log('  device ok: drift69=%.2e drift89=0 drift90=0 (bitwise, at kout)' % drift69)

    # ---- subspace mechanism readings
    if SMOKE:
        cands = [k for k in layers if k >= 1]
    else:
        cands = [k for k in range(1, rd + 1)]
    k_mid = min(cands, key=lambda k: top1_curve[k])
    B_mid, B_out, B_emb = basis[k_mid], basis[rd], basis[0]
    ustar_emb = ustar1[0]

    def spect(Ba, Bb):
        s = np.linalg.svd(Ba @ Bb.T, compute_uv=False)
        return [float(x) for x in s]

    def first(Ba, Bb):
        return float(np.linalg.svd(Ba @ Bb.T, compute_uv=False)[0])

    s_mid_out = spect(B_mid, B_out)
    s_mid_emb = spect(B_mid, B_emb)
    s_emb_out = spect(B_emb, B_out)
    sig1 = {'mid_out': s_mid_out[0], 'mid_emb': s_mid_emb[0],
            'emb_out': s_emb_out[0]}
    ovr = {'mid_out': sum(x * x for x in s_mid_out) / R_BASIS,
           'mid_emb': sum(x * x for x in s_mid_emb) / R_BASIS,
           'emb_out': sum(x * x for x in s_emb_out) / R_BASIS}
    ov_out = {k: float(np.linalg.norm(basis[k] @ ustar_kout) ** 2) for k in layers}
    ov_emb = {k: float(np.linalg.norm(basis[k] @ ustar_emb) ** 2) for k in layers}
    log('  k_mid=%d top1(k_mid)=%.4f | sigma1 mid-out=%.4f mid-emb=%.4f emb-out=%.4f'
        % (k_mid, top1_curve[k_mid], sig1['mid_out'], sig1['mid_emb'],
           sig1['emb_out']))
    log('  ovr mid-out=%.3f mid-emb=%.3f emb-out=%.3f'
        % (ovr['mid_out'], ovr['mid_emb'], ovr['emb_out']))

    if SMOKE:
        gate_m, cont_ok, aff_ok = None, None, None
    else:
        cont_ok = bool(sig1['mid_out'] >= GATE_CONT)
        aff_ok = bool(sig1['mid_out'] > sig1['mid_emb'])
        gate_m = bool(cont_ok and aff_ok)
    out = dict(layers=layers, top1_curve=top1_curve, kout=rd, NH=NH, R=R_BASIS,
               k_mid=k_mid, top1_kmid=top1_curve[k_mid],
               sigma1=sig1, ovr=ovr,
               spectra={'mid_out': s_mid_out, 'mid_emb': s_mid_emb,
                        'emb_out': s_emb_out},
               ov_out=ov_out, ov_emb=ov_emb,
               drift_3169=drift69, drift_3189=drift89, drift_3190=drift90,
               drift_3252=drift_3252,
               cont_ok=cont_ok, aff_ok=aff_ok, gate_model=gate_m)
    log('  gate: cont=%s aff=%s gate=%s'
        % (cont_ok, aff_ok, gate_m))
    return out


def eval_arm(sealed_3250, sealed_3252, led):
    """(b) registry-append trigger evaluation, report-only."""
    reg_sha = sha8_file(SRC['registry_v133'])
    assert reg_sha == 'a30215fd', ('registry drifted', reg_sha)
    phases = [e.get('phase') for e in led['measurements']
              if isinstance(e.get('phase'), int) and 3239 <= e['phase'] <= 3252]
    closed = len(phases)
    c1 = closed >= 6
    draft = json.load(open(SRC['appx6_draft'], encoding='utf-8'))
    c2 = (draft['entry']['id'] == 'APPX-6'
          and draft['entry']['status'] == 'appendix_recorded'
          and draft['based_on_file_sha8'] == '5934631b')
    bad = [e['phase'] for e in led['measurements']
           if isinstance(e.get('phase'), int) and 3239 <= e['phase'] <= 3252
           and str(e.get('verdict', '')).lower().startswith(('blocked', 'failed'))]
    c3 = not bad
    c4 = 'XC_p5_b8' in sealed_3250['aggregate']
    c5 = reg_sha == 'a30215fd'
    crit = {'c1_closed_phases_gte6': {'pass': c1, 'measured': closed},
            'c2_appx6_draft': {'pass': c2, 'measured': draft['entry']['id']
                               + '/' + draft['based_on_file_sha8']},
            'c3_no_blocked_failed': {'pass': c3, 'measured': bad},
            'c4_incumbent_in_3250': {'pass': c4, 'measured': 'XC_p5_b8'},
            'c5_registry_v133_sha': {'pass': c5, 'measured': reg_sha}}
    met = all(v['pass'] for v in crit.values())
    preview = {
        'schema': 'registry_append_preview',
        'report_only': True,
        'based_on': {'registry': 'v1.33/' + reg_sha,
                     'gap_ledger': 'v1.11/5934631b',
                     'appx6_draft': sha8_file(SRC['appx6_draft'])},
        'criteria': crit,
        'trigger_status': 'append_trigger_MET' if met else 'append_trigger_NOT_MET',
        'planned_diff': {
            'gap_ledger': 'v1.11 -> v1.12: appendix.append(APPX-6 '
                          '(fidelity-closure chain 3239-3250 + ustar lexical '
                          'decoupling, anchor p3250/9d0f1825)); version bump; '
                          'gaps GAP-1..5 and APPX-1..5 bytes unchanged',
            'registry': 'v1.33 -> v1.34: provenance.anchor_files adds '
                        'p3246/p3250/p3252 (+p3253 at landing); features and '
                        'failures lists UNCHANGED (no E-level upgrade claimed '
                        'by 3239-3253 -- all characterization); principles '
                        'unchanged; phase pointer 3238 -> landing phase',
            'not_touched': 'features[].*, failures[].*, evidence taxonomy, '
                           'sealed phase results'},
        'decision': ('LAND at next phase tail (3254 closeout) as the '
                     'registry-append trigger' if met
                     else 'HOLD: criteria unmet'),
        'created': time.strftime('%Y-%m-%d %H:%M'),
    }
    return preview, crit, met


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dsha = freeze()
    log('SMOKE=%s models=%s' % (SMOKE, MODELS_RUN))

    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    sealed_3190 = json.load(open(SRC['p3190_result'], encoding='utf-8'))
    sealed_3252 = json.load(open(SRC['p3252_result'], encoding='utf-8'))
    for k, v in SHA_ANCHOR.items():
        actual = sha8_file(SRC[k])
        assert actual == v, ('src sha anchor', k, actual, v)
    log('src sha anchors OK (incl 3252 result + APPX-6 draft)')

    per_model = {}
    for m in MODELS_RUN:
        per_model[m] = run_model_subspace(m, sealed_3169, sealed_3189,
                                          sealed_3190, sealed_3252)

    models_needed = len(MODELS)
    gate_all = bool(len(per_model) == models_needed
                    and all(per_model[m]['gate_model'] for m in MODELS))
    verdict = 'ustar_subspace_registered' if (gate_all and not SMOKE) else \
              ('SMOKE_OK' if SMOKE else 'ustar_subspace_partial')
    log('GATE: %s verdict=%s' % (gate_all, verdict))

    # ---- arm (b): registry-append trigger evaluation (full mode only)
    arm_b = None
    if not SMOKE:
        sealed_3250 = json.load(open(SRC['p3250_result'], encoding='utf-8'))
        led = json.load(open(SRC['ledger'], encoding='utf-8'))
        preview, crit, met = eval_arm(sealed_3250, sealed_3252, led)
        json.dump(preview, open(os.path.join(OUTDIR,
                                             'registry_append_preview_v1_34.json'),
                                'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        arm_b = {'trigger_status': preview['trigger_status'],
                 'criteria': crit, 'decision': preview['decision'],
                 'preview_file': 'registry_append_preview_v1_34.json'}
        log('arm (b): %s (decision: %s)'
            % (preview['trigger_status'], preview['decision']))

    res = {'phase': 3253, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
           'design_sha8': dsha, 'per_model': per_model, 'gate_all': gate_all,
           'gate': {'cont': GATE_CONT, 'R': R_BASIS, 'models_needed': models_needed},
           'arm_b': arm_b,
           'verdict': verdict, 'zero_gpu': True,
           'src': 'p3253',
           'note': 'subspace mechanism arm zero-GPU; drift_3252 cross-phase '
                   'anchor bitwise; registry v1.33 byte-untouched; append '
                   'preview report-only'}
    blob = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    rsha = sha8(blob)
    res['res_sha8'] = rsha
    suffix = '_smoke' if SMOKE else ''
    json.dump(res, open(os.path.join(OUTDIR, 'result%s.json' % suffix), 'w', encoding='utf-8'),
              ensure_ascii=False, sort_keys=True, indent=1)
    if not SMOKE:
        arrs = {}
        for m in MODELS_RUN:
            pm = per_model[m]
            arrs[('%s_ov_out' % m)] = np.array([pm['ov_out'][k] for k in sorted(pm['ov_out'])], np.float64)
            arrs[('%s_ov_emb' % m)] = np.array([pm['ov_emb'][k] for k in sorted(pm['ov_emb'])], np.float64)
            arrs[('%s_spec_mid_out' % m)] = np.array(pm['spectra']['mid_out'], np.float64)
            arrs[('%s_spec_mid_emb' % m)] = np.array(pm['spectra']['mid_emb'], np.float64)
        np.savez_compressed(os.path.join(OUTDIR, 'subspace.npz'), allow_pickle=False, **arrs)
    log('VERDICT: %s res_sha8=%s' % (verdict, rsha))
    open(os.path.join(OUTDIR, 'run_log%s.txt' % suffix), 'w', encoding='utf-8').write(
        '\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
