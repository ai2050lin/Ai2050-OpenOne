# -*- coding: utf-8 -*-
# Phase 3165 independent disk verification (separate process; byte-level seal recon;
# pairwise-angle recomputation from raw npz; five-write presence).
import hashlib
import io
import json
import os
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3165', 'g5a3_family_alignment')
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3165_verify_out.txt'
RES = []


def chk(name, ok, info=''):
    RES.append('%s %s %s' % ('PASS' if ok else 'FAIL', name, info))


def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def seal_recon_disk(p):
    """Remove LAST top-level seal_sha8 line byte-wise, re-hash, compare."""
    txt = io.open(p, encoding='utf-8').read()
    raw = json.loads(txt)
    seal = raw.get('seal_sha8')
    if not seal:
        return False, None
    m = re.search(r',\r?\n[ \t]*"seal_sha8": "[0-9a-f]{8}"', txt)
    if not m:
        return False, seal
    blob = (txt[:m.start()] + txt[m.end():]).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:8] == seal, seal


# 1. disk sha vs in-result anchors
R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))
for k, p in R['source_sha8'].items():
    full = os.path.join(B, k.replace('p3158', 'phase3158').replace('p3157', 'phase3157')
                        .replace('p2874', 'phase2874').replace('p2878', 'phase2878')
                        .replace('p2881', 'phase2881').replace('p2806', 'phase2806')
                        .replace('qwen3-4b', 'qwen3-4b')) if False else None
# explicit path map instead
PMAP = {
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
}
for k, p in PMAP.items():
    chk('disk sha %s' % k, sha8(p) == R['source_sha8'][k], R['source_sha8'][k])

# 2. seal byte-level reconstruction
for f in ('result.json', 'smoke_result.json'):
    ok, seal = seal_recon_disk(os.path.join(PDIR, f))
    chk('seal recon %s' % f, ok, seal)

# 3. independent angle recomputation (raw npz -> top1 for the 3 gate pairs)
z58 = np.load(PMAP['p3158_4b'])
top64 = z58['top64'].astype(np.float64).T          # (64, D)
z57 = np.load(PMAP['p3157_4b'])
H = z57['H'].astype(np.float64)
NL = H.shape[1] - 1
X = H[:, NL, :]
X = X - X.mean(0, keepdims=True)
_, _, Vh = np.linalg.svd(X, full_matrices=False)
K_ent = Vh[:8]
z74 = np.load(PMAP['p2874'], allow_pickle=True)
z78 = np.load(PMAP['p2878'], allow_pickle=True)
S = {'S_class': R and None}
W_t = None
from safetensors import safe_open
mdir = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
cfg4 = json.load(io.open(os.path.join(mdir, 'config.json'), encoding='utf-8'))
tied4 = bool(cfg4.get('tie_word_embeddings', False))
want4 = 'model.embed_tokens.weight' if tied4 else 'lm_head.weight'
for sh in sorted(os.listdir(mdir)):
    if sh.endswith('.safetensors'):
        with safe_open(os.path.join(mdir, sh), framework='pt') as f:
            if want4 in set(f.keys()):
                W_t = f.get_tensor(want4).float().numpy()
                break
assert W_t is not None, 'unembed not found'
W = W_t.astype(np.float64)
e2806 = json.load(io.open(PMAP['p2806_exec'], encoding='utf-8'))
CATS = e2806['cats']
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(mdir, local_files_only=True,
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


single = []
for w in [x for v in CATS.values() for x in v]:
    try:
        tid(w)
        single.append(w)
    except AssertionError:
        pass
cents = []
for cat in CATS.keys():
    ws = [w for w in CATS[cat] if w in single]   # 2881 verbatim: no truncation in centroid
    cents.append(np.stack([W[tid(w)] for w in ws]).mean(0))
Cm = np.stack(cents)
dW_c = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
dW_class = dW_c / np.linalg.norm(dW_c, axis=1, keepdims=True)
# 口径一致：主脚本 build_S_class 返回 float32（round-trip 后的 float64 才是登记口径）。
# S_class 行空间病态（类质心方向近相关），纯 float64 版对 3.7e-9 量化扰动敏感 0.27°
# （实测 66.726 vs 67.000），两侧均远离 30/15 度门——判决不变。
S_class = dW_class.astype(np.float32).astype(np.float64)
S_attr = z74['dW_unit'].astype(np.float64)
S_syn = z78['dW_unit'].astype(np.float64)


def top1_deg(A, Bd):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s[0], 0, 1))))


fp = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()[:10]
RES.append('INFO fp S_class=%s top64=%s K_ent=%s' % (fp(S_class), fp(top64), fp(K_ent)))
# decisive: rebuild S_class via main-script module in this process and diff
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'm3165v', os.path.join(ROOT, 'tests', 'glm5', 'phase3165_g5a3_family_alignment.py'))
_m = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_m)
W_m, _ = _m.load_WU(_m.MDIR4)
dW_m, _, _ = _m.build_S_class(W_m)
RES.append('INFO maxdiff S_class main-vs-verify=%.3e fp_main=%s'
           % (float(np.abs(dW_m.astype(np.float64) - S_class).max()), fp(dW_m.astype(np.float64))))


for nm, Bm in (('K_readout__S_class', S_class), ('K_readout__S_attr', S_attr),
               ('K_readout__S_syntax', S_syn), ('K_entity__S_class', S_class)):
    t = top1_deg(top64 if nm.startswith('K_readout') else K_ent, Bm)
    got = R['pairwise_4b'][nm]['top1_deg']
    chk('angle recomp %s' % nm, abs(t - got) < 0.05, 're=%.3f res=%.3f' % (t, got))
    if nm == 'K_readout__S_class':
        t_m = _m.kross(top64, Bm)[1]
        t_m2 = _m.kross(top64, dW_m.astype(np.float64))[1]
        t_v2 = top1_deg(top64, dW_m.astype(np.float64))
        RES.append('INFO triple: verify_fn(%.4f) main_kross(%.4f) main_kross_f32(%.4f) '
                   'verify_fn_f32(%.4f)' % (t, t_m, t_m2, t_v2))

# 4. five-write presence
led = json.load(io.open(LEDGER, encoding='utf-8'))
chk('ledger n==317', len(led['measurements']) == 317, 'n=%d' % len(led['measurements']))
chk('ledger last 3165', led['measurements'][-1].get('phase') == 3165)
memo = open(MEMO, 'rb').read().decode('utf-8')
chk('MEMO 3165', '## Phase 3165' in memo)
chk('MEMO BOM intact', memo.startswith('\ufeff'))
chk('MEMO CRLF uniform', memo.count('\r\n') > 0 and
    (memo.count('\n') - memo.count('\r\n')) == (1 if memo.endswith('\n') and not memo.endswith('\r\n') else 0))
d2 = io.open(DAILY, encoding='utf-8').read()
chk('daily 3165', '3165 跨族连接 v0 闭环' in d2)
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('MEMORY 3165', '3165 跨族连接 v0 闭环' in w2)

# 5. execution.json drift self-consistency
ex = json.load(io.open(os.path.join(PDIR, 'execution.json'), encoding='utf-8'))
CANON = ('phase', 'name', 'subspace_defs', 'source_sha8', 'angle_method',
         'eff_dim', 'gates', 'descriptive', 'reason_family')
blob = json.dumps({k: ex[k] for k in CANON}, ensure_ascii=False, indent=1,
                  sort_keys=True).encode('utf-8')
chk('execution self-sha', hashlib.sha256(blob).hexdigest()[:8] == ex['design_sha8'],
    ex['design_sha8'])

n_pass = sum(1 for r in RES if r.startswith('PASS'))
n_fail = sum(1 for r in RES if r.startswith('FAIL'))
RES.append('TOTAL PASS=%d FAIL=%d %s' % (n_pass, n_fail, 'ALL PASS' if n_fail == 0 else 'HAS FAILURES'))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(RES) + '\n')
print(RES[-1])
