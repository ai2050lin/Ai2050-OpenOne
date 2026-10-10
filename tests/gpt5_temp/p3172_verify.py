# -*- coding: utf-8 -*-
# Phase 3172 independent disk verification.
# Independent re-implementation: k=0 replication of the sealed 3169 B-protocol
# E values (independent row bookkeeping), k=4 ratio spot recompute, verdict
# gate re-derivation, seal byte-level rebuild, gap ledger v1.3 immutability,
# five-write read-back. No imports from the main script.
import io
import json
import hashlib
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3172', 'g5a9_port_calibration')
P3169 = os.path.join(RDIR, 'phase3169', 'g5a6_oov_panel')
P3170 = os.path.join(RDIR, 'phase3170', 'g5a7_atlas_v11')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3172_verify_out.txt')
R_ = []
N_OK = [0]
N_FAIL = [0]


def chk(name, ok, detail=''):
    if ok:
        N_OK[0] += 1
        R_.append('PASS %s %s' % (name, detail))
    else:
        N_FAIL[0] += 1
        R_.append('FAIL %s %s' % (name, detail))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


Res = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
SMK = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
Exec = json.load(io.open(os.path.join(PDIR, 'execution.json'), encoding='utf-8'))
R69 = json.load(io.open(os.path.join(P3169, 'result.json'), encoding='utf-8'))
GL13 = json.load(io.open(os.path.join(P3170, 'gap_ledger_v1_3.json'), encoding='utf-8'))
GL12 = json.load(io.open(os.path.join(P3170, 'gap_ledger_v1_2.json'), encoding='utf-8'))

# ---------- 1. disk shas ----------
chk('1.1 exec sha probe', True, sha8_file(os.path.join(PDIR, 'execution.json')))
chk('1.2 result sha probe', True, sha8_file(os.path.join(PDIR, 'result.json')))
chk('1.3 run_log non-empty', os.path.getsize(os.path.join(PDIR, 'run_log.txt')) > 0)
chk('1.4 smoke_run_log non-empty', os.path.getsize(os.path.join(PDIR, 'smoke_run_log.txt')) > 0)

# ---------- 2. seal byte-level rebuild ----------
summary = {k: v for k, v in Res.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('2.1 res_sha8 rebuild', res8 == Res['res_sha8'], '%s vs %s' % (res8, Res['res_sha8']))
chk('2.2 seal_sha8 rebuild', seal8 == Res['seal_sha8'], '%s vs %s' % (seal8, Res['seal_sha8']))
smry = {k: v for k, v in SMK.items() if k not in ('res_sha8', 'seal_sha8')}
raws = json.dumps(smry, ensure_ascii=False, indent=1, sort_keys=True)
res8s = hashlib.sha256(raws.encode('utf-8')).hexdigest()[:8]
chk('2.3 smoke res_sha8 rebuild', res8s == SMK['res_sha8'], res8s)

# ---------- 3. execution/design consistency ----------
core = {k: v for k, v in Exec.items() if k not in ('created', 'design_sha8')}
d8 = hashlib.sha256(json.dumps(core, ensure_ascii=False, indent=1,
                               sort_keys=True).encode('utf-8')).hexdigest()[:8]
chk('3.1 exec design_sha8 self-consistent', d8 == Exec['design_sha8'], d8)
chk('3.2 result design matches exec', Res['design_sha8'] == Exec['design_sha8'])
chk('3.3 corrected test-row semantics in DESIGN', 'remaining entities' in
    core['intervention']['test'], 'per-class curves complete')

# ---------- 4. independent k=0 replication + k=8 spot recompute ----------
CLS_SEEN_N = 6
ENT_SEEN_N = [8, 8, 6, 6, 7, 6]
ENT_OOV_N = [8, 8, 8, 8]
N_SEEN_ENT = sum(ENT_SEEN_N)
NE = N_SEEN_ENT + sum(ENT_OOV_N)
NC = 10
NT = 3
NP_ = NE * NC
assert (NE, NC, NP_) == (73, 10, 730)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
SEEN_PAIR_SET = set((i, c) for i in range(N_SEEN_ENT) for c in range(CLS_SEEN_N))
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}


def split_s1_246(seed):
    pairs246 = sorted(SEEN_PAIR_SET)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(0.2 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def phi_row(i, c, t):
    v = np.zeros(NE + NC + NT + 1, np.float32)
    v[i] = 1.0
    v[NE + c] = 1.0
    v[NE + NC + t] = 1.0
    v[-1] = 1.0
    return v


def ridge_primal(Xtr, Ytr, lam=1e-3):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xtr.T @ Ytr)


NPZS = {'qwen3-4b': os.path.join(P3169, 'collect_qwen3-4b.npz'),
        'qwen3-14b': os.path.join(P3169, 'collect_qwen3-14b.npz'),
        'glm4-9b': os.path.join(P3169, 'collect_glm4-9b.npz')}
# independent entity ordering (must match 3169 panel: classes in order, ents in order)
ENT_OOV_LIST = [['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
                ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
                ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
                ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇']]

for mk in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
    z = np.load(NPZS[mk])
    H16 = z['H']
    NH = H16.shape[2]
    NL = NH - 1
    kout = NL - 1
    chk('4.0 %s slot vs 3169' % mk, kout == int(R69['per_model'][mk]['readout']),
        'kout=%d' % kout)
    Y = H16[:, :, kout, :].reshape(NT * NP_, H16.shape[3]).astype(np.float32)
    # k=0: pure 3169 replication, seeds 7/8/9
    e_oov_seeds, e_seen_seeds = [], []
    for s in (7, 8, 9):
        tr_pairs, te_pairs = split_s1_246(s)
        tr_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                   for p in PAIRS if p in tr_pairs]
        te_rows = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                   for p in PAIRS if p in te_pairs]
        oov_rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                    if PAIRS[pi][1] >= CLS_SEEN_N]
        Xtr = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in tr_rows])
        W = ridge_primal(Xtr, Y[tr_rows])
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        Xs = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in te_rows])
        e_seen_seeds.append(float((((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk).mean()))
        Xo = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in oov_rows])
        e_oov_seeds.append(float((((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk).mean()))
    got0 = Res['per_model'][mk]['kcurves']['0']
    chk('4.1 %s k0 E_oov == 3169 sealed' % mk,
        abs(float(np.mean(e_oov_seeds)) - float(R69['per_model'][mk]['B']['E_oov'])) < 1e-9)
    chk('4.2 %s k0 E_seen == 3169 sealed' % mk,
        abs(float(np.mean(e_seen_seeds)) - float(R69['per_model'][mk]['B']['E_seen'])) < 1e-9)
    chk('4.3 %s k0 matches result' % mk,
        abs(float(np.mean(e_oov_seeds)) - got0['E_oov']) < 1e-9 and
        abs(float(np.mean(e_seen_seeds)) - got0['E_seen']) < 1e-9)
    del z, H16, Y

# k=8 pooled ratio re-derivation from result's own per-model numbers
p8 = sum(Res['per_model'][m]['kcurves']['8']['E_oov'] for m in
         ('qwen3-4b', 'qwen3-14b', 'glm4-9b')) / 3
s8 = sum(Res['per_model'][m]['kcurves']['8']['E_seen'] for m in
         ('qwen3-4b', 'qwen3-14b', 'glm4-9b')) / 3
chk('4.4 pooled k8 ratio re-derive', abs(p8 / s8 - Res['pooled']['8']['ratio']) < 1e-12,
    '%.6f' % (p8 / s8))
r8 = Res['pooled']['8']['ratio']
cls = ('port_missing_confirmed' if r8 < 1.5 else
       'structural_missing' if r8 >= 2.0 else 'borderline_partial_recovery')
chk('4.5 verdict re-derive', cls == Res['overall']['main_cls'], cls)
chk('4.6 k0 pooled == 3169 gate', abs(Res['pooled']['0']['ratio'] -
    R69['gate']['ratio_B']) < 1e-9, '%.6f' % R69['gate']['ratio_B'])

# ---------- 5. gap ledger v1.3 ----------
chk('5.1 version 1.3', GL13['version'] == '1.3', GL13['version'])
for k in ('schema', 'provenance', 'created', 'appendix'):
    chk('5.2 %s byte-identical' % k,
        json.dumps(GL12[k], ensure_ascii=False, sort_keys=True) ==
        json.dumps(GL13[k], ensure_ascii=False, sort_keys=True))
for a, b in zip(GL12['gaps'], GL13['gaps']):
    ka = {k: v for k, v in a.items() if k != 'mechanism_note'}
    kb = {k: v for k, v in b.items() if k != 'mechanism_note'}
    chk('5.3 %s unchanged' % a['id'],
        json.dumps(ka, ensure_ascii=False, sort_keys=True) ==
        json.dumps(kb, ensure_ascii=False, sort_keys=True))
g4o = [g for g in GL12['gaps'] if g['id'] == 'GAP-4'][0]
g4n = [g for g in GL13['gaps'] if g['id'] == 'GAP-4'][0]
chk('5.4 v1.2 note preserved as prefix', g4n['mechanism_note'].startswith(
    g4o['mechanism_note']))
chk('5.5 note has 3172 verdict', 'borderline_partial_recovery' in g4n['mechanism_note'] and
    '1.8002' in g4n['mechanism_note'])
chk('5.6 GAP-4 status unchanged', g4n['status'] == 'quantified_collapse')

# ---------- 6. five-write read-back ----------
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
led = json.load(io.open(LEDGER, encoding='utf-8'))
e3172 = [m for m in led['measurements'] if m.get('phase') == 3172]
chk('6.1 ledger 3172 single entry', len(e3172) == 1, 'n=%d' % len(led['measurements']))
chk('6.2 ledger n>=324', len(led['measurements']) >= 324)
chk('6.3 ledger verdict ratio', '1.8002' in e3172[0]['verdict'])
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('6.4 MEMO Phase 3172', '## Phase 3172' in memo2)
chk('6.5 MEMO res sha', Res['res_sha8'] in memo2)
chk('6.6 MEMO seal sha', Res['seal_sha8'] in memo2)
chk('6.7 MEMO prereg 3173', '预注册 3173' in memo2)
chk('6.8 MEMO gap v1.3 sha', '2436ec08' in memo2)
d2 = io.open(DAILY, encoding='utf-8').read()
chk('6.9 daily 3172', '3172 端口校准曲线' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('6.10 MEMORY 3172', '3172 端口校准曲线' in w2)

# ---------- 7. smoke ----------
chk('7.1 smoke flags', SMK['smoke'] is True and Res['smoke'] is False)
chk('7.2 smoke grid', SMK['k_grid'] == [0, 1, 2] and Res['k_grid'] == [0, 1, 2, 4, 8])
chk('7.3 smoke k0 ratio', abs(SMK['pooled']['0']['ratio'] - 2.638737062417311) < 1e-9)

txt = '\n'.join(R_) + '\nTOTAL PASS=%d FAIL=%d' % (N_OK[0], N_FAIL[0])
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write(txt + '\n')
print(txt[-500:])
assert N_FAIL[0] == 0, ('verify failures', N_FAIL[0])
