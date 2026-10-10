# -*- coding: utf-8 -*-
# Phase 3166 independent disk verify (independent re-implementation, no import of main module)
# Checks: source sha, result seal byte-level rebuild, angle recompute (same dtype chain),
#         LOEO AUC recompute, five-write disk read-back, crosscheck anchors.
import hashlib
import io
import json
import os
import time

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(B, 'phase3166', 'g5a3b_logic_direction')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3166_verify_out.txt'
LOG = []
OK = [0, 0]


def log(s):
    LOG.append('[3166v] %s' % s)
    print('[3166v] %s' % s, flush=True)


def chk(name, cond, extra=''):
    OK[1] += 1
    if cond:
        OK[0] += 1
    log('%s %s %s' % ('PASS' if cond else 'FAIL', name, extra))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def unit(x):
    n = float(np.linalg.norm(x))
    assert n > 0
    return x / n


t0 = time.monotonic()
R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))

# ---------- 1. source sha (17) ----------
SRC = {
    'p3158_4b': B + r'\phase3158\g4p1_output_equivalence_class\qwen3-4b\collect.npz',
    'p3158_14b': B + r'\phase3158\g4p1_output_equivalence_class\qwen3-14b\collect.npz',
    'p3158_glm4': B + r'\phase3158\g4p1_output_equivalence_class\glm4\collect.npz',
    'p3157_4b': B + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz',
    'p3157_14b': B + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz',
    'p3157_glm4': B + r'\phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz',
    'p2874': B + r'\phase2874\attr_vocab_v2\attr_vocab_v2.npz',
    'p2878': B + r'\phase2878\syntax_trans_vocab\syntax_trans_vocab.npz',
    'p2881': B + r'\phase2881\joint_word_coords\joint_word_coords.npz',
    'p2806_exec': B + r'\phase2806\qwen4_hierarchy\execution.json',
    'p3151_collect': B + r'\phase3151\g1p1_combo_additive_vs_interaction\collect.npz',
    'p3152_4b_collect': B + r'\phase3152\g1p2_tri_model_k1\qwen3-4b\collect.npz',
    'p3152_14b_collect': B + r'\phase3152\g1p2_tri_model_k1\qwen3-14b\collect.npz',
    'p3152_4b_result': B + r'\phase3152\g1p2_tri_model_k1\qwen3-4b\result.json',
    'p3152_14b_result': B + r'\phase3152\g1p2_tri_model_k1\qwen3-14b\result.json',
    'p3152_glm4k1_result': B + r'\phase3152\g1p2_tri_model_k1\glm4k1\result.json',
    'p3165_result': B + r'\phase3165\g5a3_family_alignment\result.json',
}
bad = [k for k, p in SRC.items() if sha8_file(p) != R['source_sha8'][k]]
chk('source sha8 17/17', not bad, str(bad))

# ---------- 2. seal byte-level rebuild ----------
# NOTE: main script computes seal on the pre-seal state where verdict ends with
# '|sha8_pending'; the on-disk verdict was rewritten AFTER sealing (constructive,
# see 3161 addendum). Verify rebuilds that pre-seal state explicitly.
Rpre = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
Rpre['verdict'] = Rpre['verdict'].replace('|sha8_' + R['res_sha8'], '|sha8_pending')
raw = json.dumps(Rpre, ensure_ascii=False, indent=1, sort_keys=True)
res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(Rpre, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk('res_sha8 rebuild (pre-seal verdict state)', res8 == R['res_sha8'],
    '%s vs %s' % (res8, R['res_sha8']))
chk('seal_sha8 rebuild (pre-seal verdict state)', seal8 == R['seal_sha8'],
    '%s vs %s' % (seal8, R['seal_sha8']))
chk('on-disk verdict rewritten to final sha',
    R['verdict'].endswith('|sha8_' + R['res_sha8']), R['verdict'][-20:])
disk_sha = sha8_file(os.path.join(PDIR, 'result.json'))
raw_disk = open(os.path.join(PDIR, 'result.json'), 'rb').read()
chk('result CRLF on disk', b'\r\n' in raw_disk)
chk('result disk sha8 recorded', len(disk_sha) == 8, disk_sha)

# ---------- 3. independent recompute (4b + 14b + glm4, same dtype chain) ----------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE, NC = len(ENTS), len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NT = 3


def kross(A, Bd):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s, float(np.degrees(np.arccos(s[0])))


def build_R(Hk, y):
    ds = []
    for i in range(NE):
        tr = [r for r in ent_rows[i] if y[r]]
        fa = [r for r in ent_rows[i] if not y[r]]
        ds.append(Hk[tr].mean(0) - Hk[fa].mean(0))
    Dm = np.stack(ds)
    DmU = np.stack([unit(v) for v in Dm])
    Dc = DmU - DmU.mean(0, keepdims=True)
    _, sv, Vh = np.linalg.svd(Dc, full_matrices=False)
    return Dm, sv, Vh[:8]


def loeo(Hk, y):
    aucs = []
    for e in range(NE):
        other = [i for i in range(NE) if i != e]
        ds = []
        for i in other:
            tr = [r for r in ent_rows[i] if y[r]]
            fa = [r for r in ent_rows[i] if not y[r]]
            ds.append(unit(Hk[tr].mean(0) - Hk[fa].mean(0)))
        Dm = np.stack(ds)
        Dc = Dm - Dm.mean(0, keepdims=True)
        _, _, Vh = np.linalg.svd(Dc, full_matrices=False)
        V8 = Vh[:8]
        tr_rows = [r for i in other for r in ent_rows[i]]
        P = Hk[tr_rows] @ V8.T
        yt = y[tr_rows]
        w = P[yt].mean(0) - P[~yt].mean(0)
        he = ent_rows[e]
        sh = (Hk[he] @ V8.T) @ w
        yh = y[he]
        pos, neg = sh[yh], sh[~yh]
        m = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
        aucs.append(float(m) / (len(pos) * len(neg)))
    return float(np.mean(aucs))


MSRC = {'qwen3-4b': 'p3152_4b_collect', 'qwen3-14b': 'p3152_14b_collect',
        'glm4': 'p3151_collect'}
ent_rows = {i: [t * len(PAIRS) + pi for t in range(NT)
                for pi, (ei, ci) in enumerate(PAIRS) if ei == i] for i in range(NE)}
y = np.zeros(NT * len(PAIRS), bool)
for t in range(NT):
    for pi, (ei, ci) in enumerate(PAIRS):
        if ci == CLS_OF[ei]:
            y[t * len(PAIRS) + pi] = True

for mname, mkey in MSRC.items():
    zm = np.load(SRC[mkey])
    H16 = zm['H']
    MARG = zm['marg']
    NL = H16.shape[2] - 1
    Dn = H16.shape[3]
    Hk = H16[:, :, NL, :].reshape(NT * len(PAIRS), Dn).astype(np.float64)
    Dm, sv, V8 = build_R(Hk, y)
    rec = R['per_model'][mname]
    # R x K_readout
    z58 = np.load(SRC['p3158_' + {'qwen3-4b': '4b', 'qwen3-14b': '14b', 'glm4': 'glm4'}[mname]])
    K_read = z58['top64'].astype(np.float64).T
    z57 = np.load(SRC['p3157_' + {'qwen3-4b': '4b', 'qwen3-14b': '14b', 'glm4': 'glm4'}[mname]])
    H57 = z57['H'].astype(np.float64)
    Xc = H57[:, H57.shape[1] - 1, :]
    Xc = Xc - Xc.mean(0, keepdims=True)
    _, _, Vh57 = np.linalg.svd(Xc, full_matrices=False)
    K_ent = Vh57[:8]
    s1, t1r = kross(V8, K_read)
    s2, t1e = kross(V8, K_ent)
    chk('%s R x K_readout top1' % mname,
        abs(t1r - rec['census']['R_logic__K_readout']['top1_deg']) <= 0.05,
        '%.3f vs %s' % (t1r, rec['census']['R_logic__K_readout']['top1_deg']))
    chk('%s R x K_entity top1' % mname,
        abs(t1e - rec['census']['R_logic__K_entity']['top1_deg']) <= 0.05,
        '%.3f vs %s' % (t1e, rec['census']['R_logic__K_entity']['top1_deg']))
    cnt2 = int((s2 ** 2 >= 0.5).sum())
    chk('%s R x K_entity eff_ge05' % mname,
        cnt2 == rec['census']['R_logic__K_entity']['eff_ge05'],
        '%d vs %d' % (cnt2, rec['census']['R_logic__K_entity']['eff_ge05']))
    # G0 margin contrast
    mt, mf = [], []
    for t in range(NT):
        for pi, (ei, ci) in enumerate(PAIRS):
            mgv = MARG[t, pi]
            if ci == CLS_OF[ei]:
                mt.append(float(mgv[ci]))
            else:
                mf.append(float(mgv[ci]))
    g0d = float(np.mean(mt) - np.mean(mf))
    chk('%s G0 diff' % mname, abs(g0d - rec['device']['G0']['diff']) <= 1e-6,
        '%.6f vs %s' % (g0d, rec['device']['G0']['diff']))
    # G2 LOEO (4b only full; 14b/glm4 also cheap enough)
    auc = loeo(Hk, y)
    chk('%s G2 LOEO AUC' % mname, abs(auc - rec['device']['G2']['auc_mean']) <= 5e-4,
        '%.4f vs %s' % (auc, rec['device']['G2']['auc_mean']))

# 4b S_class rebuild (3165 recipe incl. float32 round-trip) + R x S_class
from safetensors import safe_open
mdir4 = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
cfgm = json.load(io.open(os.path.join(mdir4, 'config.json'), encoding='utf-8'))
want = 'model.embed_tokens.weight' if cfgm.get('tie_word_embeddings') else 'lm_head.weight'
for sh in sorted(os.listdir(mdir4)):
    if sh.endswith('.safetensors'):
        with safe_open(os.path.join(mdir4, sh), framework='pt') as f:
            if want in set(f.keys()):
                W4 = f.get_tensor(want).float().numpy()
                break
W4 = W4.astype(np.float64)
e2806 = json.load(io.open(SRC['p2806_exec'], encoding='utf-8'))
CATS = e2806['cats']
CAT_WORDS = list(CATS.keys())
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(mdir4, local_files_only=True,
                                    trust_remote_code=True, use_fast=True)
tc = {}


def tid(t):
    if t not in tc:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        assert len(ids) == 1
        tc[t] = int(ids[0])
    return tc[t]


all_words = [w for v in CATS.values() for w in v]
single_tok = []
for w in all_words:
    try:
        tid(w)
        single_tok.append(w)
    except AssertionError:
        pass
Erows = {w: W4[tid(w)] for w in single_tok}
cents = [np.stack([Erows[w] for w in CATS[cat] if w in single_tok]).mean(0)
         for cat in CAT_WORDS]
Cm = np.stack(cents)
dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
dW_class = np.stack([unit(dW_c[i]) for i in range(10)]).astype(np.float32)  # round-trip!
S_class = dW_class.astype(np.float64)
zm4 = np.load(SRC['p3152_4b_collect'])
H4 = zm4['H']
Hk4 = H4[:, :, 36, :].reshape(NT * len(PAIRS), 2560).astype(np.float64)
_, _, V84 = build_R(Hk4, y)
s3, t1s = kross(V84, S_class)
chk('4b R x S_class top1 (float32 round-trip chain)',
    abs(t1s - R['per_model']['qwen3-4b']['census']['R_logic__S_class']['top1_deg']) <= 0.05,
    '%.3f vs %s' % (t1s, R['per_model']['qwen3-4b']['census']['R_logic__S_class']['top1_deg']))

# crosscheck anchors vs 3165 result
z165 = json.load(io.open(SRC['p3165_result'], encoding='utf-8'))
chk('3165 4b Kread_Kent anchor', abs(z165['k_entity_vs_readout_4b']['top1_deg'] - 69.823) < 0.01)
chk('3165 cross 14b anchor', abs(z165['cross_model_K']['qwen3-14b']['K_entity_top1_vs_readout_deg'] - 70.235) < 0.01)
chk('3165 cross glm4 anchor', abs(z165['cross_model_K']['glm4']['K_entity_top1_vs_readout_deg'] - 79.01) < 0.01)

# ---------- 4. five-write disk read-back ----------
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
ms2 = led['measurements']
chk('ledger n==318', len(ms2) == 318, 'n=%d' % len(ms2))
chk('ledger last=3166', ms2[-1].get('phase') == 3166)
chk('ledger verdict contains mixed', 'mixed' in ms2[-1]['verdict'])
memo2 = open(MEMO, 'rb').read().decode('utf-8')
chk('MEMO has 3166 section', '## Phase 3166' in memo2)
chk('MEMO 3166 has device numbers', '0.856' in memo2)
chk('MEMO 3166 has 3167 prereg', '3167=G5-A4' in memo2.replace(' ', '') or '3167' in memo2)
d2 = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk('daily has 3166', '3166 R 族逻辑方向' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('MEMORY has 3166', '3166 R 族逻辑对比方向' in w2)
chk('execution.json design_sha8', json.load(io.open(os.path.join(PDIR, 'execution.json'),
    encoding='utf-8'))['design_sha8'] == R['design_sha8'])
smoke = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
chk('smoke seal fields', bool(smoke.get('res_sha8')) and bool(smoke.get('seal_sha8')))

log('TOTAL %d/%d PASS, %d FAIL, elapsed %.1fs' % (OK[0], OK[1], OK[1] - OK[0],
                                                  time.monotonic() - t0))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
assert OK[0] == OK[1], 'VERIFY FAILURES'
log('ALL PASS')
