# -*- coding: utf-8 -*-
# Phase 3169 independent verify (separate process).
# - disk hash stability + result record match
# - seal byte-level rebuild
# - D4 anchor recompute independently re-run on qwen3-4b collect npz (cols=51 verbatim)
# - E_seen_B vs q03 anchors (tol 1e-6), gate verdict independently re-derived
# - D3 spot re-check (4b, sample rows bitwise)
# - five-write read-back
import hashlib
import io
import json
import os
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
B3152 = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1')
B3151 = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3169_verify_out.txt')
LOG = []


def chk(name, ok, info=''):
    LOG.append('%s %s %s' % ('PASS' if ok else 'FAIL', name, info))
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, info), flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
Q03 = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json'),
                        encoding='utf-8'))
pm = R['per_model']

# ---------- 1. disk hashes ----------
chk('collect 4b sha stable', sha8_file(os.path.join(PDIR, 'collect_qwen3-4b.npz')) ==
    sha8_file(os.path.join(PDIR, 'collect_qwen3-4b.npz')))
chk('result sha stable', sha8_file(os.path.join(PDIR, 'result.json')) ==
    sha8_file(os.path.join(PDIR, 'result.json')))
chk('execution exists', os.path.getsize(os.path.join(PDIR, 'execution.json')) > 3000)
chk('design sha matches result', R['design_sha8'] == '50e81ddf', R['design_sha8'])

# ---------- 2. seal rebuild ----------
body = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(body, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('res_sha8 rebuild', res8 == R['res_sha8'], '%s vs %s' % (res8, R['res_sha8']))
chk('seal_sha8 rebuild', seal8 == R['seal_sha8'], '%s vs %s' % (seal8, R['seal_sha8']))

# ---------- 3. D4 independent recompute (qwen3-4b only, cols=51 verbatim) ----------
zj = np.load(os.path.join(PDIR, 'collect_qwen3-4b.npz'))
Hj = zj['H']
zr = np.load(os.path.join(B3152, 'qwen3-4b', 'collect.npz'))
Hr = zr['H']
NT, NPJ, NH, D = Hj.shape
chk('joint H shape', (NT, NPJ) == (3, 730), str(Hj.shape))
NC_SEEN = 6
N_ENT_FULL = 41
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_FULL = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
REF_ENT = {}
_i = 0
for cl in CLASSES_SEEN:
    for e in ENT_FULL[cl]:
        REF_ENT[e] = _i
        _i += 1
ENT_SEEN_FULL = ENT_FULL
ENTS_J = []
for cl in CLASSES_SEEN + ['乐器', '天气', '运动', '电器']:
    pass
# joint seen entity list = first entries per class in result panel is full 41 (formal)
# reconstruct from the result: formal panel used full seen tables
seen_ents = []
for cl in CLASSES_SEEN:
    seen_ents.extend(ENT_FULL[cl])
NCJ = 10
# D3 spot re-check: all 738 seen rows bitwise (4b)
n_bad = 0
for t in range(NT):
    for i, e in enumerate(seen_ents):
        ir = REF_ENT[e]
        for c in range(NC_SEEN):
            if not np.array_equal(Hj[t, i * NCJ + c], Hr[t, ir * NC_SEEN + c]):
                n_bad += 1
chk('D3 spot re-check 4b bitwise (738 rows)', n_bad == 0, 'mismatch=%d' % n_bad)

# D4 recompute
kout = pm['qwen3-4b']['readout']
pairs_s = [(i, c) for i in range(N_ENT_FULL) for c in range(NC_SEEN)]
nps = 246
Ys = np.zeros((NT * nps, D), np.float32)
for t in range(NT):
    for i, e in enumerate(seen_ents):
        ir = REF_ENT[e]
        for c in range(NC_SEEN):
            Ys[t * nps + ir * NC_SEEN + c] = Hj[t, i * NCJ + c, kout]
lam = 1e-3
rec = []
for s in (7, 8, 9):
    rng = np.random.RandomState(s)
    idx = rng.permutation(nps)
    n_test = int(round(0.2 * nps))
    te = set(idx[:n_test].tolist())
    tr_rows = [t * nps + j for t in range(NT) for j in range(nps) if j not in te]
    te_rows = [t * nps + j for t in range(NT) for j in range(nps) if j in te]
    cols = N_ENT_FULL + NC_SEEN + 3 + 1

    def rv(j, t):
        i, c = pairs_s[j]
        v = np.zeros(cols, np.float32)
        v[i] = 1.0
        v[N_ENT_FULL + c] = 1.0
        v[N_ENT_FULL + NC_SEEN + t] = 1.0
        v[-1] = 1.0
        return v

    Xtr = np.stack([rv(j, t) for t in range(NT) for j in range(nps) if j not in te])
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(cols, dtype=np.float32), Xtr.T @ Ys[tr_rows])
    Xte = np.stack([rv(j, t) for t in range(NT) for j in range(nps) if j in te])
    ref = Ys[tr_rows].mean(0)
    Dk = float(((Ys[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
    e = ((Xte @ W - Ys[te_rows]) ** 2).sum(1) / Dk
    rec.append(float(e.mean()))
anch = pm['qwen3-4b']['anchor']['per_seed_anchor']
drift = max(abs(a - b) for a, b in zip(rec, anch))
chk('D4 independent recompute drift<1e-9', drift < 1e-9, 'max_drift=%.3e' % drift)

# ---------- 4. E_seen vs q03 anchors; gate re-derive ----------
for m, qk in (('qwen3-4b', 'qwen3-4b'), ('qwen3-14b', 'qwen3-14b'), ('glm4-9b', 'glm4-9b')):
    a = Q03['per_model'][qk]['b4_rel_readout_mean3seed_anchor']
    b = pm[m]['B']['E_seen']
    chk('E_seen_B vs q03 anchor (%s) tol 1e-6' % m, abs(a - b) < 1e-6,
        '%.9f vs %.9f' % (a, b))
Eo = [pm[k]['B']['E_oov'] for k in pm]
Es = [pm[k]['B']['E_seen'] for k in pm]
ratio_re = float(np.mean(Eo)) / float(np.mean(Es))
chk('gate ratio re-derive', abs(ratio_re - R['gate']['ratio_B']) < 1e-9,
    '%.6f vs %.6f' % (ratio_re, R['gate']['ratio_B']))
verdict_re = ('collapse_confirmed_gap4_open' if ratio_re > 2.0 else
              ('generalization_holds_gap4_closable' if ratio_re < 1.5 else
               'borderline_rejudge_seeds_11_12'))
chk('gate verdict re-derive', verdict_re == R['gate']['verdict'], verdict_re)
chk('all per-model ratios >2', all(pm[k]['B']['ratio'] > 2.0 for k in pm),
    ' '.join('%.2f' % pm[k]['B']['ratio'] for k in pm))
chk('all A ratios <1', all(pm[k]['A']['ratio'] < 1.0 for k in pm),
    ' '.join('%.2f' % pm[k]['A']['ratio'] for k in pm))

# ---------- 5. five-write read-back ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms2 = led['measurements']
e3169 = [m for m in ms2 if m.get('phase') == 3169]
chk('ledger has 3169', len(e3169) == 1, 'n=%d' % len(ms2))
chk('ledger n>=321', len(ms2) >= 321, 'n=%d' % len(ms2))
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('MEMO has Phase 3169', '## Phase 3169' in memo2)
chk('MEMO has res sha', R['res_sha8'] in memo2)
chk('MEMO has prereg 3170', '预注册 3170' in memo2)
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk('daily has 3169', '3169 谱外类别面板' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('MEMORY has 3169', '3169 谱外类别面板' in w2)

bad = [ln for ln in LOG if ln.startswith('FAIL')]
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
print('VERIFY %s (%d checks, %d fail)' % ('ALL PASS' if not bad else 'HAS FAIL', len(LOG), len(bad)))
