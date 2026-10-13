# -*- coding: utf-8 -*-
"""Phase 3191 (G5-C4): FTR-25 held_out upgrade arm (E1_repeatable -> E2_held_out
channel; C line, zero GPU, sealed 3169/3189/3190 reuse).

Per prereg in MEMO Phase 3190 tail (frozen before any observation):
  For each model, 4 folds: hold out one OOV class (219 row directions);
  v_hat_held = unit(mean of the other 3 classes' row directions from Qn);
  cos_held = Qn[held rows] @ v_hat_held (219 values).
  GATE (pinned pre-observation): HOLDOUT_PREDICTIVE iff median |cos_held| >= 0.3
  in >= 2/3 models => FTR-25 upgraded E1_repeatable -> E2_held_out:
  e_level change + anchors append (role=held_out) + statement append +
  upgrade_log append (chain #7), registry v1.17 -> v1.18, atlas_v1_19.html.
  Not passed => counter_evidence append, E level UNCHANGED E1_repeatable.
  Report-only: KS(|cos_held| distribution vs within-class pairwise cos
  distribution of the held class, from 3189 chain).
  Device anchors: 3189 sealed scalars (top1/top2/cross_med/within/r80/r90)
  drift == 0 bitwise; 3190 sealed scalars (top1/cross_med/within/loo_max_delta/
  ks_seen_ustar/cos_ustar_vc) drift == 0 bitwise; 3169 per-seed E replication
  < 1e-9.
  SMOKE (env P3191_SMOKE=1): zero-GPU => identical numbers; smoke_ prefix.
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
NAME = 'g5c4_ftr25_heldout_upgrade'
OUTDIR = os.path.join(SRC_DIR, 'phase3191', NAME)
SMOKE = os.environ.get('P3191_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}

G_HELD = 0.3
DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'heldout_med_abs_cos': G_HELD, 'models_needed': 2,
        'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}

SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3190_result': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\result.json'),
    'registry_v117': os.path.join(SRC_DIR, r'phase3190\g5c3_shared_direction_robustness\atlas_registry_v1_17.json'),
    'gap_v15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0', 'p3190_result': '6a8893ae',
    'registry_v117': '73923c83', 'gap_v15': '4fb3f37d', 'registry_3162': '00f15e98',
}
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3189 = ('3a26f541', 'fb8c1f65')
CONTENT_3190 = ('24461856', '0c975f5b')  # arm-stage content anchors embedded in
# p3190 result.json (final fc8be7db/f7d8571a live in p3190 final_result.json;
# content sha != file sha lesson re-confirmed, SMOKE-1 honest log)

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

DESIGN = {
    'phase': '3191', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3190 tail G5-C4 (frozen before any observation)',
    'data': 'sealed 3169 collect npz x3 + sealed 3169/3189/3190 result.json; zero GPU',
    'residual_chain': '3188/3189/3190 primary verbatim: 3169 B-protocol (seeds 7/8/9, '
                      'split_s1_246, ridge lam 1e-3, cols 780), 876 OOV rows, '
                      'row-level directions = unit(mean over 3 seeds of q)',
    'measurements': 'per model, 4 folds: hold out one OOV class (219 rows); '
                    'v_hat_held = unit(mean of other 3 classes rows of Qn); '
                    'cos_held = Qn[held] @ v_hat_held (219 values); '
                    'report-only: KS(|cos_held| vs within-class pairwise cos of '
                    'held class)',
    'gate': 'HOLDOUT_PREDICTIVE iff median |cos_held| >= 0.3 in >= 2/3 models => '
            'FTR-25 E1_repeatable -> E2_held_out (e_level change + anchors append '
            'role=held_out + statement append + upgrade_log append #7), registry '
            'v1.17->v1.18, atlas_v1_19.html; else counter_evidence append, E level '
            'UNCHANGED E1_repeatable',
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
        json.dump({'phase': 3191, 'name': NAME, 'design_sha8': dsha[:8],
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


def run_arm(sealed_3169, sealed_3189, sealed_3190):
    out = {}
    pm90 = sealed_3190['per_model']
    for m in MODELS:
        log('=== %s ===' % m)
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

        def rv(r, ne):
            i, c = PAIRS[r % NP_]
            return phi_row(i, c, r // NP_, ne, NC, NT)

        E69 = sealed_3169['per_model'][m]['B']
        E89 = sealed_3189['per_model'][m]
        drift69 = 0.0
        Q_rows = np.zeros((len(oov_rows), D), dtype=np.float32)
        for si, s in enumerate(SEEDS_S1):
            tr_rows = tr_base[s]
            Xtr = np.stack([rv(r, NE) for r in tr_rows])
            W = ridge_primal(Xtr, Y[tr_rows])
            ref = Y[tr_rows].mean(0)
            Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
            Xs = np.stack([rv(r, NE) for r in [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                                               for p in PAIRS if p in split_s1_246(s)[1]]])
            es = ((Xs @ W - Y[[t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                               for p in PAIRS if p in split_s1_246(s)[1]]]) ** 2).sum(1) / Dk
            Xo = np.stack([rv(r, NE) for r in oov_rows])
            eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
            drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                          abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
            R_oov = Y[oov_rows] - Xo @ W
            for j in range(len(oov_rows)):
                rvv = R_oov[j]
                Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
            del W, Xtr
        assert drift69 < REPL_TOL, ('3169 per-seed E replication', m, drift69)
        Qn = Q_rows / len(SEEDS_S1)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)

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

        # 3190 replication: LOO + u* readings bitwise
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
        E90 = pm90[m]
        drift90 = max(abs(loo_max - E90['loo_max_delta']),
                      abs(top1 - E90['top1']), abs(cross_med - E90['cross_med']),
                      max(abs(a - b) for a, b in zip(within, E90['within_class'])),
                      max(abs(a - b) for a, b in zip(loo, E90['loo_top1'])))
        assert drift90 == 0.0, ('3190 scalar drift', m, drift90)

        # (GATE arm) held-out prediction, 4 folds
        med_abs = []
        ks_within = []
        for ci, c in enumerate(cls_ids):
            held = oov_row_cls == c
            others = ~held
            v_hat = unit_mean(Qn[others])
            cos_held = Qn[held] @ v_hat
            med_abs.append(float(np.median(np.abs(cos_held))))
            idxs = np.where(held)[0]
            Cc = Qn[idxs] @ Qn[idxs].T
            wvals = Cc[np.triu_indices(len(idxs), 1)]
            ks_within.append(ks_stat(np.abs(cos_held), wvals))
        # model gate: held class must be predictable as a GROUP in all 4 folds
        gate_m = bool(min(med_abs) >= G_HELD)
        out[m] = dict(drift_3169=drift69, drift_3189=drift89, drift_3190=drift90,
                      med_abs_cos_held=med_abs, min_med_abs_cos=float(min(med_abs)),
                      gate_model=gate_m, ks_within=ks_within,
                      top1=top1, cross_med=cross_med, within_class=within)
        log('  %s: med|cos_held| %s (min %.4f, gate >=%.2f all 4 folds)=%s | KS(|cos|;within)=%s'
            % (m, '/'.join('%.4f' % x for x in med_abs), min(med_abs), G_HELD,
               gate_m, '/'.join('%.4f' % x for x in ks_within)))
        del z, H16, Y, Q_rows, Qn
    n_pass = sum(1 for m in MODELS if out[m]['gate_model'])
    gate_all = bool(n_pass >= 2)
    out['_gate'] = dict(n_models_pass=n_pass, predictive=gate_all)
    log('GATE: %d/3 models pass held-out => %s'
        % (n_pass, 'HOLDOUT_PREDICTIVE' if gate_all else 'NOT_PREDICTIVE'))
    return out, gate_all


# ---------------- registry v1.17 -> v1.18 + atlas_v1_19 ----------------
CREATED_STR = None


def finish_registry(summary, gate_all, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg17 = json.loads(open(SRC['registry_v117'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg17['version'] == '1.17' and len(reg17['features']) == 25
    assert len(reg17['failures']) == 19 and len(reg17['upgrade_log']) == 6
    reg18 = copy.deepcopy(reg17)
    reg18['version'] = '1.18'
    reg18['supersedes'] = 'atlas_registry_v1_17.json (73923c83, Phase 3190)'
    pm = summary['per_model']
    fmt3 = lambda key: '/'.join('%.4f' % pm[m][key] for m in MODELS)
    f25_old = [f for f in reg17['features'] if f['id'] == 'FTR-25'][0]
    f25 = copy.deepcopy(f25_old)
    if gate_all:
        anchor = {
            'src': 'p3191', 'phase': 3191,
            'role': 'held_out',
            'tags': ['zero_gpu', 'sealed_reuse', 'e_level_upgrade'],
            'asserts': {
                'min_med_abs_cos.qwen3-4b': pm['qwen3-4b']['min_med_abs_cos'],
                'min_med_abs_cos.qwen3-14b': pm['qwen3-14b']['min_med_abs_cos'],
                'min_med_abs_cos.glm4-9b': pm['glm4-9b']['min_med_abs_cos'],
                'gate_heldout_med_abs_cos': G_HELD,
                'e_level_before': 'E1_repeatable', 'e_level_after': 'E2_held_out',
                'res_sha8': res8, 'seal_sha8': seal8}}
        f25['anchors'] = f25_old['anchors'] + [anchor]
        f25['evidence_level'] = 'E2_held_out'
        f25['statement'] = f25_old['statement'] + (
            '；3191 G5-C4 held_out 升格臂：留出类 219 行方向对其余 3 类均值方向的中位 '
            '|cos| 每模型 4 折最小值 ' + fmt3('min_med_abs_cos') +
            '（门 ≥0.3 全 4 折 ×≥2/3 模型）——3 类均值方向可预测留出类行方向，'
            'E1_repeatable 升 E2_held_out（升级链 #7）。')
        reg18['features'] = [f25 if f['id'] == 'FTR-25' else f
                             for f in reg17['features']]
        reg18['upgrade_log'] = reg17['upgrade_log'] + [{
            'node': 'FTR-25', 'change': 'E1_repeatable -> E2_held_out',
            'reason': 'G5-C4 held_out arm (Phase 3191): held-out class row '
                      'directions predicted by 3-class mean direction, median '
                      '|cos| >= 0.3 in all 4 folds x 3/3 models; cross-model '
                      'held-out predictiveness established (zero GPU, sealed '
                      '3169/3189/3190 reuse)'}]
        disposition = 'v1.17_to_v1.18_ftr25_e2_held_out'
    else:
        ce = ('3191 G5-C4 held_out 升格臂未过：留出类 219 行方向对其余 3 类均值方向的'
              '中位 |cos| 4 折最小值 ' + fmt3('min_med_abs_cos') +
              '（门 ≥0.3 全 4 折 ×≥2/3 模型未过）——3 类均值方向不足以预测留出类，'
              'E1_repeatable 维持。')
        f25['counter_evidence'] = f25_old['counter_evidence'] + [ce]
        reg18['features'] = [f25 if f['id'] == 'FTR-25' else f
                             for f in reg17['features']]
        disposition = 'v1.17_to_v1.18_ftr25_ce_not_predictive'
    # preservation asserts
    for f in reg17['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg18['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-25':
            fa = json.loads(a)
            assert b != a
            if gate_all:
                assert f25['statement'].startswith(fa['statement'])
                assert f25['counter_evidence'] == fa['counter_evidence']
                assert json.dumps(f25['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \
                    json.dumps(fa['anchors'], ensure_ascii=False, sort_keys=True)
                assert f25['evidence_level'] == 'E2_held_out' and fa['evidence_level'] == 'E1_repeatable'
            else:
                assert f25['counter_evidence'][:-1] == fa['counter_evidence']
                assert f25['statement'] == fa['statement']
                assert f25['anchors'] == fa['anchors']
                assert f25['evidence_level'] == fa['evidence_level'] == 'E1_repeatable'
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg18['upgrade_log'][:-1], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg17['upgrade_log'], ensure_ascii=False, sort_keys=True) \
        if gate_all else json.dumps(reg18['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg17['upgrade_log'], ensure_ascii=False, sort_keys=True)
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg17.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg18.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg17['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg18['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_18.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg18, ensure_ascii=False, indent=1))
    reg18_sha = sha8_file(rp)
    log('registry v1.18 written sha8 %s (%s)' % (reg18_sha, disposition))

    meta_sha = {'reg18': reg18_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.19 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.19 &mdash; 语义机制图谱（registry v1.18 渲染）</h1>')
    H.append('<div class="sub">Phase 3191 G5-C4 &middot; FTR-25 held_out 升格臂（%s） &middot; '
             'registry v1.18 sha8=%s &middot; v1.17 sha8=%s &middot; v1.16 sha8=%s &middot; v1.15 sha8=%s &middot; v1.14 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3169 %s + p3189 %s + p3190 %s &middot; 渲染于 %s</div>' % (
                 'FTR-25 E1->E2_held_out 升格' if gate_all else 'FTR-25 counter_evidence 未过升格门',
                 span('meta.registry_v118_sha8', reg18_sha),
                 span('meta.registry_v117_sha8', '73923c83'),
                 span('meta.registry_v116_sha8', '114aa114'),
                 span('meta.registry_v115_sha8', '0a55d2e7'),
                 span('meta.registry_v114_sha8', '95fe55f5'),
                 span('meta.gap_v15_sha8', SHA_ANCHOR['gap_v15']),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3189_result', SHA_ANCHOR['p3189_result']),
                 span('meta.p3190_result', SHA_ANCHOR['p3190_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg18['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg18['principles']):
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
    for f in reg18['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.18，%d 条）</h2>' % len(reg18['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg18['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg18['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg18['failures']:
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
    H.append('<div class="footer">RDC/LPF atlas v1.19 &middot; 渲染自 atlas_registry_v1_18.json (%s)'
             ' + gap ledger v1.5 (%s) &middot; 增量来源: Phase 3191 G5-C4 FTR-25 held_out 升格臂'
             '（数据源 p3169 %s + p3189 %s + p3190 %s）'
             '&middot; 本文件为 Phase 3191 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg18_sha, SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3189_result'],
                 SHA_ANCHOR['p3190_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_19.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v119(reg18, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_19.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3191, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'verdict': summary['verdict'], 'gate_predictive': bool(gate_all),
                'per_model': summary['per_model']},
        'counts': {'features': len(reg18['features']), 'failures': len(reg18['failures']),
                   'upgrades': len(reg18['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v118_file': 'atlas_registry_v1_18.json', 'registry_v118_sha8': reg18_sha,
        'html_file': 'atlas_v1_19.html', 'html_sha8': html_sha,
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


def flatten_v119(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v118_sha8'] = meta_sha['reg18']
    d['meta.registry_v117_sha8'] = '73923c83'
    d['meta.registry_v116_sha8'] = '114aa114'
    d['meta.registry_v115_sha8'] = '0a55d2e7'
    d['meta.registry_v114_sha8'] = '95fe55f5'
    d['meta.gap_v15_sha8'] = SHA_ANCHOR['gap_v15']
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3189_result'] = SHA_ANCHOR['p3189_result']
    d['meta.p3190_result'] = SHA_ANCHOR['p3190_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3191 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    assert sealed_3169['res_sha8'] == CONTENT_3169[0] and sealed_3169['seal_sha8'] == CONTENT_3169[1]
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    assert sealed_3189['res_sha8'] == CONTENT_3189[0] and sealed_3189['seal_sha8'] == CONTENT_3189[1]
    sealed_3190 = json.load(open(SRC['p3190_result'], encoding='utf-8'))
    assert sealed_3190['res_sha8'] == CONTENT_3190[0] and sealed_3190['seal_sha8'] == CONTENT_3190[1]
    assert abs(float(sealed_3169['gate']['ratio_B']) - 2.5388114997805062) < 1e-9
    log('sealed results double-locked: 3169 %s/%s 3189 %s/%s 3190 %s/%s'
        % (CONTENT_3169 + CONTENT_3189 + CONTENT_3190))

    primary, gate_all = run_arm(sealed_3169, sealed_3189, sealed_3190)
    if gate_all:
        verdict = ('ftr25_heldout_predictive (min med|cos_held| %s >= 0.3 all 4 '
                   'folds in %d/3 models; FTR-25 E1_repeatable -> E2_held_out, '
                   'upgrade_log #7, registry v1.17 -> v1.18)'
                   % ('/'.join('%.4f' % primary[m]['min_med_abs_cos'] for m in MODELS),
                      primary['_gate']['n_models_pass']))
    else:
        verdict = ('ftr25_heldout_not_predictive (held-out gate fail in %d/3 '
                   'models pass; FTR-25 counter_evidence append, E level '
                   'unchanged E1_repeatable, registry v1.17 -> v1.18)'
                   % primary['_gate']['n_models_pass'])
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3191, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, predictive=bool(gate_all),
                      n_models_pass=primary['_gate']['n_models_pass']),
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
    if not SMOKE:
        finish_registry(summary, gate_all, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
