# -*- coding: utf-8 -*-
# Phase 3175 independent verify. Re-implements the full two-arm chain from
# the on-disk sources (sealed 3169 collect npz + sealed out-collect npz) and
# cross-checks every sealed number; rebuilds seals byte-level; asserts gap
# ledger v1.5 increments and five-write reads. Never imports the main script.
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3175', 'g5a12_residual_arms')
P3169 = os.path.join(S, 'phase3169', 'g5a6_oov_panel')
P3172 = os.path.join(S, 'phase3172', 'g5a9_port_calibration')
P3173 = os.path.join(S, 'phase3173', 'g5a10_atlas_v13')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = []


def chk(name, cond, detail=''):
    OUT.append('%s %s %s' % ('OK  ' if cond else 'FAIL', name, detail))
    return cond


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
SM = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(PDIR, 'execution.json'), encoding='utf-8'))
R69 = json.load(io.open(os.path.join(P3169, 'result.json'), encoding='utf-8'))
R72 = json.load(io.open(os.path.join(P3172, 'result.json'), encoding='utf-8'))

# ---- 1. seal byte-level rebuild ----
core = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
r8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(core, res_sha8=r8), ensure_ascii=False, indent=1, sort_keys=True)
s8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('1.1 res_sha8 rebuild', r8 == R['res_sha8'], r8)
chk('1.2 seal_sha8 rebuild', s8 == R['seal_sha8'], s8)
coreS = {k: v for k, v in SM.items() if k not in ('res_sha8', 'seal_sha8')}
rawS = json.dumps(coreS, ensure_ascii=False, indent=1, sort_keys=True)
r8s = hashlib.sha256(rawS.encode('utf-8')).hexdigest()[:8]
chk('1.3 smoke res rebuild', r8s == SM['res_sha8'], r8s)
# execution.json design hash rebuild (core excludes created/design_sha8)
coreE = {k: v for k, v in EX.items() if k not in ('created', 'design_sha8')}
d8e = hashlib.sha256(json.dumps(coreE, ensure_ascii=False, indent=1,
                                sort_keys=True).encode('utf-8')).hexdigest()[:8]
chk('1.4 exec design rebuild', d8e == EX['design_sha8'], d8e)
chk('1.5 result design matches exec', R['design_sha8'] == EX['design_sha8'],
    R['design_sha8'])

# ---- 2. anchors ----
chk('2.1 p3169 result file', sha8_file(os.path.join(P3169, 'result.json')) == '5b51c2c1')
chk('2.2 p3172 result file', sha8_file(os.path.join(P3172, 'result.json')) == 'd8ddc481')
chk('2.3 p3169 content', R69['res_sha8'] == '49430a39')
chk('2.4 p3172 content', R72['res_sha8'] == '463f42c8' and R72['seal_sha8'] == 'cdb85525')
for fn, h in (('collect_out_qwen3-4b.npz', 'd4f8931b'),
              ('collect_out_qwen3-14b.npz', 'b9f0a96f'),
              ('collect_out_glm4-9b.npz', 'e14c62c4')):
    chk('2.5 ' + fn, sha8_file(os.path.join(PDIR, fn)) == h)

# ---- 3. k=0 gates vs sealed sources (both arms, pooled) ----
g69 = R69['gate']
for arm in ('in', 'out'):
    chk('3.1 %s k0 pooled E_oov == 3169' % arm,
        abs(R['pooled'][arm]['0']['E_oov'] - float(g69['pooled_E_oov_B'])) < 1e-9)
    chk('3.2 %s k0 pooled E_seen == 3169' % arm,
        abs(R['pooled'][arm]['0']['E_seen'] - float(g69['pooled_E_seen_B'])) < 1e-9)
    chk('3.3 %s k0 pooled ratio == 3169' % arm,
        abs(R['pooled'][arm]['0']['ratio'] - float(g69['ratio_B'])) < 1e-9)
chk('3.4 k0 arms bitwise equal',
    R['pooled']['in']['0']['ratio'] == R['pooled']['out']['0']['ratio'])

# ---- 4. in-arm replay vs sealed 3172 (pooled + per-model) ----
for k in ('0', '1', '2', '4', '8'):
    for key in ('E_oov', 'E_seen', 'ratio'):
        chk('4.1 pooled in[%s].%s == 3172' % (k, key),
            abs(R['pooled']['in'][k][key] - float(R72['pooled'][k][key])) < 1e-9)
for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    for k in ('0', '1', '2', '4', '8'):
        for key in ('E_oov', 'E_seen', 'E_newent', 'ratio'):
            chk('4.2 %s in[%s].%s == 3172' % (mk, k, key),
                abs(R['per_model'][mk]['in_curve'][k][key] -
                    float(R72['per_model'][mk]['kcurves'][k][key])) < 1e-9)

# ---- 5. full independent re-implementation (both arms, all models, all k) ----
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
SEEDS = [7, 8, 9]
FRAC = 0.2
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
assert (NE, NC, NP_) == (73, 10, 730)
SEEN_PAIR_SET = set((i, c) for i in range(N_SEEN_ENT) for c in range(6))
PI_OF = {p: j for j, p in enumerate(PAIRS)}
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= 6]
ENT_BASE = {}
_i = 0
for cl in CLASSES:
    for e in ENT[cl]:
        ENT_BASE[e] = _i
        _i += 1
OUT_ENTS = [e for cl in CLASSES_OOV for e in ENT_OOV_OUT[cl]]
N_OUT_E = len(OUT_ENTS)
assert N_OUT_E == 40
NP_OUT = N_OUT_E * NC


def split_s1(seed):
    pairs246 = sorted(SEEN_PAIR_SET)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def ridge(Xtr, Ytr):
    A = Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xtr.T @ Ytr)


def phi(i, c, t, ne):
    v = np.zeros(ne + NC + NT + 1, np.float32)
    v[i] = 1.0
    v[ne + c] = 1.0
    v[ne + NC + t] = 1.0
    v[-1] = 1.0
    return v


MODEL_NPZ = {'qwen3-4b': os.path.join(P3169, 'collect_qwen3-4b.npz'),
             'qwen3-14b': os.path.join(P3169, 'collect_qwen3-14b.npz'),
             'glm4-9b': os.path.join(P3169, 'collect_glm4-9b.npz')}
MODEL_OUTNPZ = {'qwen3-4b': os.path.join(PDIR, 'collect_out_qwen3-4b.npz'),
                'qwen3-14b': os.path.join(PDIR, 'collect_out_qwen3-14b.npz'),
                'glm4-9b': os.path.join(PDIR, 'collect_out_glm4-9b.npz')}

replay_all_ok = True
for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    z = np.load(MODEL_NPZ[mk])
    H = z['H']
    NTz, NPz, NH, D = H.shape
    kout = NH - 2
    assert kout == int(R69['per_model'][mk]['readout'])
    Y = H[:, :, kout, :].reshape(NT * NP_, D).astype(np.float32)
    zo = np.load(MODEL_OUTNPZ[mk])
    Ho = zo['H']
    assert Ho.shape == (NT, NP_OUT, NH, D)
    assert np.isfinite(Ho.astype(np.float32)).all()
    Yo = Ho[:, :, kout, :].reshape(NT * NP_OUT, D).astype(np.float32)

    te_seen_all = {}
    tr_base = {}
    for s in SEEDS:
        tr_pairs, te_pairs = split_s1(s)
        tr_base[s] = [t * NP_ + PI_OF[p] for t in range(NT) for p in PAIRS
                      if p in tr_pairs]
        te_seen_all[s] = [t * NP_ + PI_OF[p] for t in range(NT) for p in PAIRS
                          if p in te_pairs]
    ne_rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
               if PAIRS[pi][0] >= N_SEEN_ENT and PAIRS[pi][1] < 6]

    for k in (0, 1, 2, 4, 8):
        for arm in ('in', 'out'):
            per_seed = {'E_seen': [], 'E_oov': [], 'E_newent': []}
            for s in SEEDS:
                in_cal_rows = []
                for ci, cl in enumerate(CLASSES_OOV):
                    c = 6 + ci
                    if k > 0:
                        rng = np.random.RandomState(s + 1000)
                        pick = rng.choice(len(ENT[cl]), size=k, replace=False)
                        for j in sorted(int(x) for x in pick):
                            g = ENT_BASE[ENT[cl][j]]
                            for t in range(NT):
                                in_cal_rows.append(t * NP_ + PI_OF[(g, c)])
                te_oov = [r for r in ALL_OOV_ROWS if r not in set(in_cal_rows)]
                Xl = []
                Yl = []
                if arm == 'in':
                    rows = tr_base[s] + in_cal_rows
                    ne_ext = NE
                    Xl = [phi(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_, ne_ext)
                          for r in rows]
                    Yl = [Y[r] for r in rows]
                else:
                    ne_ext = NE
                    out_rows = []
                    for ci, cl in enumerate(CLASSES_OOV):
                        c = 6 + ci
                        if k > 0:
                            rng = np.random.RandomState(s + 1000)
                            pick = rng.choice(len(ENT_OOV_OUT[cl]), size=k,
                                              replace=False)
                            for j in sorted(int(x) for x in pick):
                                for t in range(NT):
                                    out_rows.append((j, c, t))
                    ext = sorted(set((c, j) for j, c, t in out_rows))
                    ext_idx = {(c, j): NE + rank for rank, (c, j) in enumerate(ext)}
                    ne_ext = NE + len(ext)
                    Xl = [phi(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_, ne_ext)
                          for r in tr_base[s]]
                    Yl = [Y[r] for r in tr_base[s]]
                    for (j, c, t) in out_rows:
                        g = ext_idx[(c, j)]
                        v = np.zeros(ne_ext + NC + NT + 1, np.float32)
                        v[g] = 1.0
                        v[ne_ext + c] = 1.0
                        v[ne_ext + NC + t] = 1.0
                        v[-1] = 1.0
                        Xl.append(v)
                        Yl.append(Yo[t * NP_OUT + j * NC + c])
                Xtr = np.stack(Xl)
                Ytr = np.stack(Yl)
                W = ridge(Xtr, Ytr)
                ref = Ytr.mean(0)
                Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
                Xs = np.stack([phi(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_, ne_ext)
                               for r in te_seen_all[s]])
                per_seed['E_seen'].append(float((((Xs @ W - Y[te_seen_all[s]]) ** 2)
                                                 .sum(1) / Dk).mean()))
                Xo2 = np.stack([phi(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_, ne_ext)
                                for r in te_oov])
                per_seed['E_oov'].append(float((((Xo2 @ W - Y[te_oov]) ** 2)
                                                .sum(1) / Dk).mean()))
                Xn = np.stack([phi(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_, ne_ext)
                               for r in ne_rows])
                per_seed['E_newent'].append(float((((Xn @ W - Y[ne_rows]) ** 2)
                                                   .sum(1) / Dk).mean()))
            for key in ('E_seen', 'E_oov', 'E_newent'):
                got = float(np.mean(per_seed[key]))
                sealed = R['per_model'][mk][arm + '_curve'][str(k)][key]
                if abs(got - sealed) >= 1e-9:
                    replay_all_ok = False
                    chk('5 %s %s k%d %s' % (mk, arm, k, key), False,
                        'got %.12f sealed %.12f' % (got, sealed))
    del z, H, Y, zo, Ho, Yo
chk('5.1 full replay in+out x3 models x5 k x3 keys (135 cells)', replay_all_ok)

# ---- 6. delta gate re-derivation ----
delta = R['pooled']['out']['8']['ratio'] - R['pooled']['in']['8']['ratio']
chk('6.1 delta recompute', abs(delta - R['delta_gate']['delta']) < 1e-12,
    '+%.6f' % delta)
cls_expect = ('port_residual_dominant' if abs(delta) <= 0.15 else
              'entity_familiarity_component_confirmed' if delta > 0 else
              'anomaly_register')
chk('6.2 verdict class re-derive', R['delta_gate']['main_cls'] == cls_expect,
    cls_expect)
chk('6.3 verdict string', R['verdict'] == R['verdict'].strip() and
    'port_residual_dominant' in R['verdict'], R['verdict'][:90])
# recovery fractions
rec_in = (R['pooled']['in']['0']['ratio'] - R['pooled']['in']['8']['ratio']) / \
    R['pooled']['in']['0']['ratio']
rec_out = (R['pooled']['out']['0']['ratio'] - R['pooled']['out']['8']['ratio']) / \
    R['pooled']['out']['0']['ratio']
chk('6.4 recovery in ~29%', abs(rec_in - 0.2909) < 0.001, '%.4f' % rec_in)
chk('6.5 recovery out ~26.5%', abs(rec_out - 0.2649) < 0.001, '%.4f' % rec_out)

# per-class delta re-derivation from sealed per-model curves
pc_ok = True
for cl_i, cl in enumerate(['乐器', '天气', '运动', '电器']):
    c = 6 + cl_i
    for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
        d = (R['per_model'][mk]['out_curve']['8']['E_oov_cls_mean'][cl_i] -
             R['per_model'][mk]['in_curve']['8']['E_oov_cls_mean'][cl_i])
        if abs(d - R['per_class_delta'][cl]['delta_per_model'][
                ('qwen3-4b', 'qwen3-14b', 'glm4-9b').index(mk)]) > 1e-12:
            pc_ok = False
chk('6.6 per-class deltas re-derive', pc_ok)

# ---- 7. gap ledger v1.5 ----
G14 = json.load(io.open(os.path.join(P3173, 'gap_ledger_v1_4.json'), encoding='utf-8'))
G15 = json.load(io.open(os.path.join(PDIR, 'gap_ledger_v1_5.json'), encoding='utf-8'))
g4o = [x for x in G14['gaps'] if x['id'] == 'GAP-4'][0]
g4n = [x for x in G15['gaps'] if x['id'] == 'GAP-4'][0]
chk('7.1 version bump', G15['version'] == '1.5', G15['version'])
chk('7.2 schema bump', G15['schema'] == 'rdc_atlas_gap_ledger_v1_5')
chk('7.3 statement extended', len(g4n['statement']) > len(g4o['statement']) and
    '3175 交叉臂定判' in g4n['statement'])
chk('7.4 evidence 7->8', len(g4o['evidence']) == 7 and len(g4n['evidence']) == 8)
chk('7.5 anchor p3175 added', g4n['anchor_sha8'].get('p3175') ==
    sha8_file(os.path.join(PDIR, 'result.json')))
chk('7.6 mechanism finalized', '3175 交叉臂定稿' in g4n['mechanism_note'] and
    g4n['mechanism_note'].startswith(g4o['mechanism_note']))
chk('7.7 status unchanged', g4n['status'] == g4o['status'] == 'quantified_collapse')
others14 = json.dumps([x for x in G14['gaps'] if x['id'] != 'GAP-4'],
                      ensure_ascii=False, sort_keys=True)
others15 = json.dumps([x for x in G15['gaps'] if x['id'] != 'GAP-4'],
                      ensure_ascii=False, sort_keys=True)
chk('7.8 other gaps byte-identical', others14 == others15)
chk('7.9 appendix byte-identical',
    json.dumps(G14['appendix'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(G15['appendix'], ensure_ascii=False, sort_keys=True))
chk('7.10 prereg field unchanged', g4n['prereg'] == g4o['prereg'])

# ---- 8. five-write reads ----
led = json.load(io.open(LEDGER, encoding='utf-8'))
e75 = [m for m in led['measurements'] if m.get('phase') == 3175]
chk('8.1 ledger n=327 has 3175', len(led['measurements']) == 327 and
    len(e75) == 1, 'n=%d' % len(led['measurements']))
if e75:
    chk('8.2 ledger 3175 verdict', e75[0]['verdict'] == R['verdict'])
    chk('8.3 ledger 3175 E2', e75[0]['evidence_level'] == 'E2_predictive')
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('8.4 MEMO 3175 section', '## Phase 3175' in memo2)
chk('8.5 MEMO has 3176 prereg', '### 接续：预注册 3176' in memo2)
chk('8.6 MEMO has res+seal', R['res_sha8'] in memo2 and R['seal_sha8'] in memo2)
chk('8.7 MEMO has gap v1.5', '4fb3f37d' in memo2)
d2 = io.open(DAILY, encoding='utf-8').read()
chk('8.8 daily has 3175 with shas', 'cb76a177/seal 9e1a2f08' in d2 and
    'port_residual_dominant' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('8.9 MEMORY has 3175 final', '3175 结构残留定位交叉臂闭环' in w2 and
    'port_residual_dominant' in w2)
chk('8.10 smoke artifact', SM['smoke'] is True and R['smoke'] is False)

n_ok = sum(1 for l in OUT if l.startswith('OK'))
n_fail = sum(1 for l in OUT if l.startswith('FAIL'))
OUT.append('TOTAL PASS=%d FAIL=%d' % (n_ok, n_fail))
with io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3175_verify_out.txt'),
             'w', encoding='utf-8') as f:
    f.write('\n'.join(OUT) + '\n')
print('verify written: PASS=%d FAIL=%d' % (n_ok, n_fail))
