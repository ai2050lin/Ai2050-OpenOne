# -*- coding: utf-8 -*-
# Phase 3172 G5-A9: OOV-class port calibration curve (zero GPU intervention).
# Reuses the sealed 3169 collect npz. Intervention test of the 3171 mechanism
# note: if the collapse body is the one-hot class port having no training
# signal under class leave-out (port_missing), then adding a few calibration
# rows of an OOV class to the B-protocol train set should restore its readout.
# Protocol (pre-registered in MEMO Phase 3171 tail):
#   per OOV class c: sample k entities (RandomState(seed+1000), no replacement;
#   seed-dependent so the choice variance averages out over seeds 7/8/9);
#   their rows (all 3 templates) join the B-protocol train set;
#   test = all OOV-class rows minus ALL calibration rows (k=8 = full class
#   calibration leaves no test rows for that class - its per-class cell is
#   then absent, honestly reported as null);
#   Q03 verbatim ridge (lam=1e-3, Dk = train-row variance mean, seeds 7/8/9);
#   ratio_B(k) = E_oov(k)/E_seen(k), both recomputed per k under the same
#   W/Dk chain (calibration rows change W and Dk, so E_seen is recomputed too).
# Device gate: k=0 must reproduce the sealed 3169 B-protocol values per model
# and pooled within 1e-9 (identical code chain, zero calibration rows).
# Pre-registered gate: pooled ratio_B(k=8) < 1.5 -> port_missing_confirmed
# (mechanism note upgraded to causal_support); >= 2 -> structural_missing;
# [1.5, 2) -> borderline_partial_recovery. k=1/2/4 = sample-efficiency gradient.
# DESIGN is fully static (3169 discipline).
import hashlib
import io
import json
import os
import time

import numpy as np

T0 = time.time()
PHASE = 3172
NAME = 'g5a9_port_calibration'
SMOKE = os.environ.get('SMOKE', '') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
P3169 = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
OUTDIR = os.path.join(RDIR, 'phase3172', 'g5a9_port_calibration')
LOG = []

MODELS = [
    dict(name='qwen3-4b', npz=os.path.join(P3169, 'collect_qwen3-4b.npz')),
    dict(name='qwen3-14b', npz=os.path.join(P3169, 'collect_qwen3-14b.npz')),
    dict(name='glm4-9b', npz=os.path.join(P3169, 'collect_glm4-9b.npz')),
]
SMOKE_NPZ = os.path.join(P3169, 'collect_smoke_qwen3-4b.npz')

SHA_ANCHOR = {
    'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575', 'p3169_glm4': '9cc3f8b4',
    'p3169_result': '5b51c2c1', 'p3169_smoke_result': '466ef808',
}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


# ---------------- panel (3169 verbatim; SMOKE truncates identically) ----------
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
NT = 3
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
LAM = 1e-3
KS_FULL = [0, 1, 2, 4, 8]
KS_SMOKE = [0, 1, 2]
KS = KS_SMOKE if SMOKE else KS_FULL
if SMOKE:
    ENT_SEEN = {c: v[:2] for c, v in ENT_SEEN.items()}
    ENT_OOV = {c: v[:2] for c, v in ENT_OOV.items()}
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
N_SEEN_ENT = sum(len(v) for v in ENT_SEEN.values())
N_OOV_ENT = sum(len(v) for v in ENT_OOV.values())
N_OOV_PER_CLS = {c: len(v) for c, v in ENT_OOV.items()}
N_SEEN_PAIRS = N_SEEN_ENT * len(CLASSES_SEEN)
N_OOVCLS_PAIRS = NE * len(CLASSES_OOV)
if not SMOKE:
    assert (NE, NC, NP_) == (73, 10, 730), 'panel'
    assert (N_SEEN_PAIRS, N_OOVCLS_PAIRS) == (246, 292), 'pair sets'
SEEN_PAIR_SET = set((i, c) for i in range(N_SEEN_ENT) for c in range(len(CLASSES_SEEN)))
OOV_CLS_PAIRS = set((i, c) for i in range(NE) for c in range(len(CLASSES_SEEN), NC))
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
OOV_ROW_OF = {}
for t in range(NT):
    for p in OOV_CLS_PAIRS:
        OOV_ROW_OF[(t, p)] = t * NP_ + PI_OF_PAIR[p]
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= len(CLASSES_SEEN)]
ENT_ROW_BASE = {}
_i = 0
for cl in CLASSES:
    for e in ENT[cl]:
        ENT_ROW_BASE[e] = _i
        _i += 1
assert _i == NE

# ---------------- DESIGN (fully static) ---------------------------------------
DESIGN = dict(
    phase=PHASE, name=NAME,
    hypothesis='collapse body = one-hot class port without training signal under '
               'class leave-out (3171 mechanism note: encoding_missing rejected '
               '3/3, K_readout energy ~1, centroids inside seen cone); a few '
               'calibration rows of an OOV class should restore its readout if '
               'the port hypothesis is correct',
    intervention=dict(
        unit='per OOV class c: k entities sampled with RandomState(seed+1000), '
             'no replacement, seed-dependent (choice variance averages over '
             'seeds); their rows (all NT templates) join the B-protocol train '
             'set; k=0 = no calibration (pure 3169 replication)',
        test='all OOV-class rows minus the calibration rows themselves; a '
             'calibration removes only the sampled entities OWN rows, so a '
             'class cell keeps the remaining entities x that class rows and '
             'per-class curves are complete on the full k grid (test rows are '
             'defined by pair (entity, class) over all 73 entities)',
        k_grid_full=[0, 1, 2, 4, 8], k_grid_smoke=[0, 1, 2]),
    protocol='Q03 verbatim: phi_row one-hot[entity]+one-hot[class]+one-hot'
             '[template]+bias (cols = NE+NC+NT+1), ridge_primal lam=1e-3 fp32, '
             'Dk = train-row variance mean + 1e-9, E = normalized MSE at the '
             'readout slot; seeds 7/8/9; ratio_B(k) = E_oov(k)/E_seen(k) both '
             'recomputed per k under the same W/Dk chain',
    gate_pre_registered=dict(
        primary='pooled ratio_B(k=8): < 1.5 -> port_missing_confirmed '
                '(GAP-4 mechanism_note upgraded to causal_support); >= 2 -> '
                'structural_missing; [1.5, 2) -> borderline_partial_recovery',
        gradient='k=1/2/4 sample-efficiency; per-class curves complete on the '
                 'full k grid (a calibrated class keeps the remaining '
                 'entities rows as test)',
        device='k=0 reproduces sealed 3169 B-protocol per-model E_oov/E_seen '
               'and pooled values within 1e-9'),
    slots=dict(kout='NL-1, asserted equal to 3169 result per_model.readout'),
    dtype_chain='H float16 -> float32 (Y) -> fp32 ridge chain (3169 verbatim)',
    source_sha8=SHA_ANCHOR,
    smoke='SMOKE run uses the sealed 3169 smoke npz (truncated panel) and '
          'asserts k=0 against the sealed 3169 smoke_result gate values',
)


def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    core = {k: v for k, v in DESIGN.items() if k not in ('created', 'design_sha8')}
    d8 = hashlib.sha256(json.dumps(core, ensure_ascii=False, indent=1,
                                   sort_keys=True).encode('utf-8')).hexdigest()[:8]
    if os.path.exists(exep):
        prev = json.load(io.open(exep, encoding='utf-8'))
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        body = dict(core)
        body['created'] = time.strftime('%Y-%m-%d %H:%M:%S')
        body['design_sha8'] = d8
        with io.open(exep, 'w', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True))
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ---------------- Q03 verbatim mechanics (3169 copy) --------------------------
def split_s1_246(seed):
    allp = SEEN_PAIR_SET
    pairs246 = sorted(allp)
    assert len(pairs246) == N_SEEN_PAIRS, 'seen pairs count'
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return allp - test, test


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


def calib_entities(seed, k):
    """Per OOV class: k entity global indices, RandomState(seed+1000), no repl."""
    out = {}
    for ci, cl in enumerate(CLASSES_OOV):
        n_c = len(ENT[cl])
        base = ENT_ROW_BASE[ENT[cl][0]]
        if k == 0:
            out[ci] = []
        else:
            assert k <= n_c, ('k exceeds class entity count', k, cl)
            rng = np.random.RandomState(seed + 1000)
            pick = rng.choice(n_c, size=k, replace=False)
            out[ci] = sorted(base + int(j) for j in pick)
    return out


def run():
    d8 = freeze()
    # ---- G_anchor ----
    src = {
        'p3169_4b': MODELS[0]['npz'], 'p3169_14b': MODELS[1]['npz'],
        'p3169_glm4': MODELS[2]['npz'],
        'p3169_result': os.path.join(P3169, 'result.json'),
        'p3169_smoke_result': os.path.join(P3169, 'smoke_result.json'),
    }
    for kk, v in SHA_ANCHOR.items():
        got = sha8(src[kk])
        assert got == v, ('G_anchor', kk, got, v)
    log('G_anchor: %d files OK' % len(SHA_ANCHOR))
    R69 = json.load(io.open(src['p3169_result'], encoding='utf-8'))
    S69 = json.load(io.open(src['p3169_smoke_result'], encoding='utf-8'))
    assert R69['res_sha8'] == '49430a39' and S69['res_sha8'] == '46600d73'
    REF = S69 if SMOKE else R69

    per_model = {}
    for m in (MODELS[:1] if SMOKE else MODELS):
        mk = m['name']
        log('=== model %s ===' % mk)
        z = np.load(SMOKE_NPZ if SMOKE else m['npz'])
        H16 = z['H']
        NTz, NPz, NH, D = H16.shape
        assert NTz == NT and NPz == NP_, ('shape', H16.shape)
        assert np.isfinite(H16.astype(np.float32)).all(), 'H non-finite'
        NL = NH - 1
        kout = NL - 1
        rd = int(REF['per_model'][mk]['readout'])
        assert kout == rd, ('G_slot', kout, rd)
        log('G_slot: kout=%d == 3169 readout; NL=%d D=%d' % (kout, NL, D))
        Y = H16[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)

        te_seen_all = []
        tr_base = {}
        for s in SEEDS_S1:
            tr_pairs, te_pairs = split_s1_246(s)
            tr_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                       for p in PAIRS if p in tr_pairs]
            te_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                       for p in PAIRS if p in te_pairs]
            tr_base[s] = tr_rows
            te_seen_all.append(te_rows)

        def rv(r):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, NE, NC, NT)

        kres = {}
        for k in KS:
            per_seed = dict(E_seen=[], E_oov=[], E_newent=[],
                            E_oov_cls=[[] for _ in range(len(CLASSES_OOV))])
            for si, s in enumerate(SEEDS_S1):
                cal = calib_entities(s, k)
                cal_rows = []
                for ci, idxs in cal.items():
                    c = len(CLASSES_SEEN) + ci
                    for i in idxs:
                        for t in range(NT):
                            cal_rows.append(t * NP_ + PI_OF_PAIR[(i, c)])
                cal_set = set(cal_rows)
                assert len(cal_set) == len(cal_rows), 'calib row collision'
                assert not (cal_set & set(tr_base[s])), 'calib leaks into seen train'
                tr_rows_k = tr_base[s] + cal_rows
                te_oov_k = [r for r in ALL_OOV_ROWS if r not in cal_set]
                Xtr = np.stack([rv(r) for r in tr_rows_k])
                W = ridge_primal(Xtr, Y[tr_rows_k])
                ref = Y[tr_rows_k].mean(0)
                Dk = float(((Y[tr_rows_k] - ref) ** 2).sum(1).mean()) + 1e-9
                Xs = np.stack([rv(r) for r in te_seen_all[si]])
                es = ((Xs @ W - Y[te_seen_all[si]]) ** 2).sum(1) / Dk
                Xo = np.stack([rv(r) for r in te_oov_k])
                eo = ((Xo @ W - Y[te_oov_k]) ** 2).sum(1) / Dk
                ne_rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                           if PAIRS[pi][0] >= N_SEEN_ENT and PAIRS[pi][1] < 6]
                Xn = np.stack([rv(r) for r in ne_rows])
                en = ((Xn @ W - Y[ne_rows]) ** 2).sum(1) / Dk
                per_seed['E_seen'].append(float(es.mean()))
                per_seed['E_oov'].append(float(eo.mean()))
                per_seed['E_newent'].append(float(en.mean()))
                pos = 0
                for ci in range(len(CLASSES_OOV)):
                    rows_c = [r for r in te_oov_k
                              if PAIRS[r % NP_][1] == len(CLASSES_SEEN) + ci]
                    if rows_c:
                        Xc = np.stack([rv(r) for r in rows_c])
                        ec = ((Xc @ W - Y[rows_c]) ** 2).sum(1) / Dk
                        per_seed['E_oov_cls'][ci].append(float(ec.mean()))
                    else:
                        per_seed['E_oov_cls'][ci].append(None)
                    pos += len(rows_c)
                assert pos == len(te_oov_k)
            E_seen = float(np.mean(per_seed['E_seen']))
            E_oov = float(np.mean(per_seed['E_oov']))
            E_newent = float(np.mean(per_seed['E_newent']))
            kres[k] = dict(E_seen=E_seen, E_oov=E_oov, E_newent=E_newent,
                           ratio=E_oov / E_seen,
                           E_seen_per_seed=per_seed['E_seen'],
                           E_oov_per_seed=per_seed['E_oov'],
                           E_oov_cls_mean=[None if None in v else float(np.mean(v))
                                           for v in per_seed['E_oov_cls']])
            log('k=%d: E_seen=%.6f E_oov=%.6f ratio=%.4f E_newent=%.6f'
                % (k, E_seen, E_oov, E_oov / E_seen, E_newent))
            del Xtr, W
        # G_k0: reproduce sealed 3169 B values
        refB = REF['per_model'][mk]['B']
        for key, val in (('E_oov', kres[0]['E_oov']), ('E_seen', kres[0]['E_seen']),
                         ('E_newent', kres[0]['E_newent'])):
            rk = 'E_newent' if key == 'E_newent' else key
            d0 = abs(val - float(refB[rk]))
            assert d0 < 1e-9, ('G_k0', mk, key, val, refB[rk])
            log('G_k0 %s %s: drift=%.2e OK' % (mk, key, d0))
        per_model[mk] = dict(NL=NL, D=D, kout=kout, kcurves=kres)
        del z, H16, Y

    # ---- pooled + verdict ----
    pooled = {}
    for k in KS:
        po = float(np.mean([per_model[mk]['kcurves'][k]['E_oov'] for mk in per_model]))
        ps = float(np.mean([per_model[mk]['kcurves'][k]['E_seen'] for mk in per_model]))
        pooled[k] = dict(E_oov=po, E_seen=ps, ratio=po / ps)
        log('pooled k=%d: E_oov=%.6f E_seen=%.6f ratio=%.4f'
            % (k, po, ps, po / ps))
    # G_k0 pooled
    refp = REF['gate']
    for key in ('pooled_E_oov_B', 'pooled_E_seen_B'):
        kk = 'E_oov' if key == 'pooled_E_oov_B' else 'E_seen'
        d0 = abs(pooled[0][kk] - float(refp[key]))
        assert d0 < 1e-9, ('G_k0 pooled', key, d0)
        log('G_k0 pooled %s: drift=%.2e OK' % (key, d0))
    r8 = pooled[KS[-1]]['ratio']
    if SMOKE:
        main_cls = 'smoke'
    else:
        main_cls = ('port_missing_confirmed' if r8 < 1.5 else
                    'structural_missing' if r8 >= 2.0 else
                    'borderline_partial_recovery')
    verdict = ('g5a9_port_calibration|%s|%d_models|ratio_k%s=%.4f|%s'
               % ('smoke' if SMOKE else 'full', len(per_model), KS[-1], r8, main_cls))
    log('VERDICT: %s' % verdict)

    result = dict(phase=PHASE, name=NAME, design_sha8=d8, smoke=SMOKE,
                  k_grid=KS, per_model=per_model, pooled=pooled,
                  overall=dict(ratio_kmax=r8, main_cls=main_cls,
                               gate='pre-registered: pooled ratio_B(k=8) <1.5 '
                                    'port_missing_confirmed / >=2 '
                                    'structural_missing / else '
                                    'borderline_partial_recovery'),
                  verdict=verdict)
    raw = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
    mid = json.dumps(dict(result, res_sha8=res8), ensure_ascii=False,
                     indent=1, sort_keys=True)
    seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
    result['res_sha8'] = res8
    result['seal_sha8'] = seal8
    fname = 'smoke_result.json' if SMOKE else 'result.json'
    with io.open(os.path.join(OUTDIR, fname), 'w', encoding='utf-8',
                 newline='\r\n') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True))
    log('sealed %s res=%s seal=%s' % (fname, res8, seal8))
    lname = 'smoke_run_log.txt' if SMOKE else 'run_log.txt'
    with io.open(os.path.join(OUTDIR, lname), 'w', encoding='utf-8',
                 newline='\r\n') as f:
        f.write('\n'.join(LOG) + '\n')
    log('DONE')


if __name__ == '__main__':
    run()
