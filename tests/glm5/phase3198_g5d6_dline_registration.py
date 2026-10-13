# -*- coding: utf-8 -*-
"""Phase 3198 (G5-D6): D-line closing registration arm (registry v1.20 ->
v1.21, zero GPU, sealed 3169/3189-3197 reuse).

Per prereg in MEMO Phase 3197 tail (frozen before any observation):
  FTR-26 statement append (SIGNED_CLASS_CONTRAST qualitative closing +
  D-line five-link chain 3193-3197 summary) + anchors append
  (role=signed_closing) - registry v1.20 -> v1.21 (append-only; 25 legacy
  features byte-preserved; evidence_level UNCHANGED E2_held_out;
  upgrade_log UNCHANGED 8 entries) + atlas_v1_23.html.
  GATE (pinned pre-observation): REGISTERED iff (a) statement append
  verified (startswith + anchors prefix + all other features byte-equal),
  (b) atlas field check clean (0 missing/extra/mismatch/dups), (c) control
  anchor: sealed 3197 signed_weights v_c x4 + ls_resid (4dp, measured from
  sealed p3197 result BEFORE freeze) re-read drift == 0.
  Device anchors: sealed-file double locks (content shas) 3169/3183/3189/
  3190/3191/3192/3193/3194-result; 3157/3183 files sha-asserted.
  SMOKE (env P3198_SMOKE=1): registry/html writes skipped; checks only.
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
NAME = 'g5d6_dline_registration'
OUTDIR = os.path.join(SRC_DIR, 'phase3198', NAME)
SMOKE = os.environ.get('P3198_SMOKE') == '1'
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
NPZ69_KEY = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}
NPZ57_KEY = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}

DEV_TOL = 1e-9
REPL_TOL = 1e-9
GATE = {'append_only': True, 'field_check_clean': True,
        'models_needed': 3, 'dev_tol': DEV_TOL, 'repl_tol': REPL_TOL}

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
    'p3197_result': os.path.join(SRC_DIR, r'phase3197\g5d5_signed_ls_shares\result.json'),
    'registry_v120': os.path.join(SRC_DIR, r'phase3194\g5d2_ftr26_heldout_upgrade\atlas_registry_v1_20.json'),
    'registry_v119': os.path.join(SRC_DIR, r'phase3193\g5d1_ustar_decomposition\atlas_registry_v1_19.json'),
    'registry_3162': os.path.join(SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3183_result': '0ab70579',
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3189_result': '5bd804d0', 'p3190_result': '6a8893ae',
    'p3191_result': 'a218ead4', 'p3192_result': '5d8d4858', 'p3193_result': '7c841a3f', 'p3197_result': 'ee91f8c7',
    'registry_v120': '2aac7e5e',
    'registry_v119': '4d64a153', 'registry_3162': '00f15e98',
}
CONTENT_3169 = ('49430a39', '018c6024')
CONTENT_3189 = ('3a26f541', 'fb8c1f65')
CONTENT_3190 = ('24461856', '0c975f5b')
CONTENT_3191 = ('836fe933', 'a66c0b3d')
CONTENT_3192 = ('6221fe00', '58c4f183')
CONTENT_3193 = ('5465dd47', 'ce1d8c8d')
CONTENT_3183 = ('c55e4e5e', '7196fa8a')
CONTENT_3197 = ('dd2af224', '4311d796')

# sealed 3197 scalars (4dp; measured from sealed p3197 result BEFORE freeze)
ANCH_3197 = {
    'qwen3-4b': dict(ls_resid=0.0181, v_c6=-0.3199, v_c7=-0.2725, v_c8=-0.3360, v_c9=-0.3028),
    'qwen3-14b': dict(ls_resid=0.0174, v_c6=-0.2893, v_c7=-0.2810, v_c8=-0.3547, v_c9=-0.2756),
    'glm4-9b': dict(ls_resid=0.0181, v_c6=-0.3613, v_c7=-0.3137, v_c8=-0.3203, v_c9=-0.3020),
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
    'phase': '3198', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3197 tail G5-D6 (frozen before any observation)',
    'data': 'sealed atlas_registry_v1_20.json (2aac7e5e) + sealed p3197 result '
            '(ee91f8c7) + gap ledger v1.5; zero GPU',
    'control_anchor': 'per model, re-read sealed 3197 signed_weights v_c x4 + '
                      'ls_resid (4dp, measured from sealed p3197 result BEFORE '
                      'freeze) vs pinned 4dp constants ANCH_3197; drift == 0 x3/3',
    'measurements': 'registry v1.20 -> v1.21 reconstruction in memory: FTR-26 '
                    'statement append (SIGNED_CLASS_CONTRAST closing + D-line '
                    '3193-3197 chain summary) + anchors append (role='
                    'signed_closing); evidence_level + upgrade_log + 25 legacy '
                    'features byte-preserved; atlas_v1_23.html render + '
                    'per-field data-k verify',
    'gate': 'REGISTERED iff (a) statement append verified (startswith + anchors '
            'prefix + all other features byte-equal), (b) atlas field check '
            'clean (0 missing/extra/mismatch/dups), (c) control anchor drift '
            '== 0 x3/3; else NOT_REGISTERED, registry v1.20 unchanged',
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
        json.dump({'phase': 3198, 'name': NAME, 'design_sha8': dsha[:8],
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


def run_control(sealed_3197):
    """Control anchor: sealed 3197 signed_weights/ls_resid vs pinned 4dp."""
    out = {}
    for m in MODELS:
        pm = sealed_3197['per_model'][m]
        A = ANCH_3197[m]
        drift = abs(round(pm['ls_resid'], 4) - A['ls_resid'])
        for j, c in enumerate(range(6, 10)):
            drift = max(drift, abs(round(pm['signed_weights']['v_c[%d]' % c], 4)
                                   - A['v_c%d' % c]))
        assert drift == 0.0, ('3197 control drift', m, drift)
        out[m] = dict(drift=drift, ls_resid=pm['ls_resid'],
                      vc_weights={c: pm['signed_weights']['v_c[%d]' % c]
                                  for c in range(6, 10)})
        log('  control 3197 %s: drift=0 (ls_resid %.4f, v_c x4 all negative)'
            % (m, pm['ls_resid']))
    log('CONTROL: sealed 3197 scalars == pinned 4dp constants x3/3')
    return out


# ---------------- registry v1.19 -> v1.20 + atlas_v1_22 ----------------
CREATED_STR = None


def finish_registry(control, res8, seal8, design_sha):
    global CREATED_STR
    CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    reg20 = json.loads(open(SRC['registry_v120'], 'rb').read().decode('utf-8'))
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg20['version'] == '1.20' and len(reg20['features']) == 26
    reg21 = copy.deepcopy(reg20)
    reg21['version'] = '1.21'
    reg21['supersedes'] = 'atlas_registry_v1_20.json (2aac7e5e, Phase 3194)'
    f26_old = [f for f in reg20['features'] if f['id'] == 'FTR-26'][0]
    f26 = copy.deepcopy(f26_old)
    anchor = {
        'src': 'p3197', 'phase': 3197,
        'role': 'signed_closing',
        'tags': ['zero_gpu', 'sealed_reuse', 'qualitative_closing'],
        'asserts': {
            'ls_resid.qwen3-4b': 0.0181, 'ls_resid.qwen3-14b': 0.0174,
            'ls_resid.glm4-9b': 0.0181,
            'vc_l1_share.qwen3-4b': 0.9159, 'vc_l1_share.qwen3-14b': 0.8905,
            'vc_l1_share.glm4-9b': 0.9244,
            'vc_weights': 'all negative in [-0.40, -0.25] x3/3',
            'gate_signed_class_contrast': 0.5, 'ls_resid_gate': 0.05,
            'control_3197_drift': 0.0, 'res_sha8': res8, 'seal_sha8': seal8}}
    f26['anchors'] = f26_old['anchors'] + [anchor]
    f26['statement'] = f26_old['statement'] + (
        '；3197 G5-D5 带符号组成份额定谳（D 线收官）：22 方向候选集上无约束 LS '
        '（λ=1e-12）残差 0.0181/0.0174/0.0181（≈3195 QR 投影残差比，近完全拟合），'
        'v_c×4 权重全负 −0.273~−0.361、l1 份额 0.9159/0.8905/0.9244、次大 |w| '
        '≤0.024——u* ≈ −(w₈v_c[8]+w₆v_c[6]+w₉v_c[9]+w₇v_c[7])：反类均值对比轴'
        '（同时偏离全部 OOV 类均值）。D 线五连闭环：3193 NOVEL（单方向级，限制门集）'
        '→ 3194 E2_held_out（跨类留出可预测）→ 3195 PARTIALLY_DECOMPOSED（子空间级'
        ' proj 0.9998）→ 3196 非负锥 FAIL（nnls_resid 0.82 ≫ ratio 0.018）→ 3197 '
        '带符号定谳；方法学沉淀：子空间投影/非负份额/带符号份额三层读数必须分开'
        '报告，无符号 cos 级读数系统性夸大类均值构成。')
    reg21['features'] = [f26 if f['id'] == 'FTR-26' else f
                         for f in reg20['features']]
    disposition = 'v1_20_to_v1_21_ftr26_signed_closing'
    # preservation asserts (append-only; evidence_level + upgrade_log unchanged)
    assert f26['evidence_level'] == 'E2_held_out' == f26_old['evidence_level']
    assert f26['statement'].startswith(f26_old['statement'])
    assert json.dumps(f26['anchors'][:-1], ensure_ascii=False, sort_keys=True) == \
        json.dumps(f26_old['anchors'], ensure_ascii=False, sort_keys=True)
    for f in reg20['features']:
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg21['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        if f['id'] == 'FTR-26':
            assert b != a
        else:
            assert a == b, ('feature not preserved', f['id'])
    assert json.dumps(reg21['upgrade_log'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg20['upgrade_log'], ensure_ascii=False, sort_keys=True), 'upgrade_log changed'
    for k in ('evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created',
              'phase', 'name', 'design_sha8'):
        assert json.dumps(reg20.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg21.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    assert json.dumps(reg20['failures'], ensure_ascii=False, sort_keys=True) == \
        json.dumps(reg21['failures'], ensure_ascii=False, sort_keys=True), 'failures changed'
    rp = os.path.join(OUTDIR, 'atlas_registry_v1_21.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(reg21, ensure_ascii=False, indent=1))
    reg21_sha = sha8_file(rp)
    log('registry v1.21 written sha8 %s (%s)' % (reg21_sha, disposition))

    meta_sha = {'reg21': reg21_sha}
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.23 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.23 &mdash; 语义机制图谱（registry v1.21 渲染）</h1>')
    H.append('<div class="sub">Phase 3198 G5-D6 &middot; D 线收官入册（FTR-26 SIGNED_CLASS_CONTRAST 定谳） &middot; '
             'registry v1.21 sha8=%s &middot; v1.20 sha8=%s &middot; v1.19 sha8=%s &middot; v1.18 sha8=%s &middot; gap v1.6 sha8=%s'
             ' &middot; 源 p3169 %s + p3197 %s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v121_sha8', reg21_sha),
                 span('meta.registry_v120_sha8', '2aac7e5e'),
                 span('meta.registry_v119_sha8', '4d64a153'),
                 span('meta.registry_v118_sha8', 'c08cbcc2'),
                 span('meta.gap_v16_sha8', '0f6423e0'),
                 span('meta.p3169_result', SHA_ANCHOR['p3169_result']),
                 span('meta.p3197_result', SHA_ANCHOR['p3197_result']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg21['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg21['principles']):
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
    for f in reg21['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.21，%d 条）</h2>' % len(reg21['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg21['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg21['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg21['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    gl15 = json.loads(open(os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'), encoding='utf-8').read())
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿；v1.6 mechanism_note 见 3192 封存）</h2>')
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
    H.append('<div class="footer">RDC/LPF atlas v1.23 &middot; 渲染自 atlas_registry_v1_21.json (%s)'
             ' + gap ledger v1.5 (4fb3f37d) &middot; 增量来源: Phase 3198 G5-D6 D 线收官入册'
             '（FTR-26 SIGNED_CLASS_CONTRAST 定谳；源 p3169 %s + p3197 %s）'
             '&middot; 本文件为 Phase 3198 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 reg21_sha, SHA_ANCHOR['p3169_result'], SHA_ANCHOR['p3197_result']))
    H.append('</body></html>')
    html_text = '\n'.join(H)
    hp = os.path.join(OUTDIR, 'atlas_v1_23.html')
    with open(hp, 'w', encoding='utf-8', newline='') as f:
        f.write(html_text)
    html_sha = sha8_file(hp)
    expected = flatten_v122(reg21, nodes, gl15, meta_sha)
    v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
    log('atlas_v1_23.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
        html_sha, v['n_expected'], v['n_found'], len(v['missing']),
        len(v['extra']), len(v['mismatch']), len(v['dups'])))
    assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
    final = {
        'phase': 3198, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True,
        'arm': {'registered': True, 'control': control,
                'per_model': {m: ANCH_3197[m] for m in MODELS}},
        'counts': {'features': len(reg21['features']), 'failures': len(reg21['failures']),
                   'upgrades': len(reg21['upgrade_log']), 'gaps': 4,
                   'appendix': len(gl15['appendix'])},
        'registry_disposition': disposition,
        'registry_v121_file': 'atlas_registry_v1_21.json', 'registry_v121_sha8': reg21_sha,
        'registry_v120_sha8_ref': '2aac7e5e',
        'html_file': 'atlas_v1_23.html', 'html_sha8': html_sha,
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


def flatten_v122(reg, nodes, gl, meta_sha):
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
    d['meta.registry_v121_sha8'] = meta_sha['reg21']
    d['meta.registry_v120_sha8'] = '2aac7e5e'
    d['meta.registry_v119_sha8'] = '4d64a153'
    d['meta.registry_v118_sha8'] = 'c08cbcc2'
    d['meta.gap_v16_sha8'] = '0f6423e0'
    d['meta.p3169_result'] = SHA_ANCHOR['p3169_result']
    d['meta.p3197_result'] = SHA_ANCHOR['p3197_result']
    d['meta.created'] = CREATED_STR
    return d


def main():
    design_sha = freeze()
    log('== Phase 3198 %s %s ==' % (NAME, 'SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    sealed_3197 = json.load(open(SRC['p3197_result'], encoding='utf-8'))
    assert sealed_3197['res_sha8'] == CONTENT_3197[0] and sealed_3197['seal_sha8'] == CONTENT_3197[1]
    log('sealed result double-locked: 3197')

    control = run_control(sealed_3197)
    verdict = ('dline_closing_registered (control anchor: sealed 3197 '
               'signed_weights v_c x4 + ls_resid drift==0 x3/3; FTR-26 '
               'statement/anchors append, registry v1.20 -> v1.21, '
               'evidence_level + upgrade_log unchanged)')
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3198, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'gates': dict(GATE, registered=True),
        'control': control,
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
        finish_registry(control, res8, seal8, design_sha)
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
