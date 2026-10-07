# -*- coding: utf-8 -*-
"""Phase 3076 independent verify: recompute key
statistics from the frozen npz, recheck seals and
the closeout chain.  Prints VERIFY_OK only if all
checks pass."""
import hashlib
import io
import json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913\phase3076'
     r'\omega_p73_cross_prompt_family')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
SCRIPT = (ROOT + r'\tests\glm5\phase3076_omega_'
          r'p73_cross_prompt_family.py')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
TOL = 0.02


def sha8(path):
    return hashlib.sha256(
        io.open(path, 'rb').read()).hexdigest()[:8]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra * rb).sum()
                 / np.sqrt((ra * ra).sum()
                           * (rb * rb).sum()))


def pc(m):
    return bin(m).count('1')


res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
st = res['stats']
fam = st['families']
cross = st['cross']
npz = np.load(R + r'\omega_p73_cross_prompt_family.npz')

# --- seals ---
assert seal['npz_sha256_8'] == sha8(
    R + r'\omega_p73_cross_prompt_family.npz')
assert seal['result_sha256_8'] == sha8(
    R + r'\result.json')
assert seal['exec_sha256_8'] == sha8(
    R + r'\execution.json')
assert seal['script_sha256_8'] == sha8(SCRIPT)
assert seal['verdict'] == 'cross_prompt_unstable'

# --- per-family recomputation from npz ---
FK = ('A', 'B', 'C')
top8s = {}
r1s = {}
dahs = {}
for fk in FK:
    r1 = npz['R1_ALL32_' + fk].astype(np.float64)
    r1s[fk] = r1
    n_neg = int((r1 < 0).sum())
    assert n_neg == fam[fk]['n_neg'], (fk, n_neg)
    order = np.argsort(r1)
    topk = min(8, n_neg)
    top8 = [int(h) for h in order[:topk]]
    assert top8 == fam[fk]['top8'], (fk, top8)
    top8s[fk] = set(top8)
    assert int(npz['TOP8_SEL_OK_' + fk]) == 1
    neg_sum = float(r1[r1 < 0].sum())
    cap = abs(float(r1[top8].sum())) / abs(neg_sum)
    assert abs(cap - fam[fk]['capture8']) < 1e-12
    dahs[fk] = np.abs(npz['DAH34_MED_' + fk]).astype(
        np.float64)
    assert abs(float(npz['R_ALL_' + fk])
               - fam[fk]['r_all32']) < 1e-15
    # submodularity replay
    p4 = npz['PAIRS4_' + fk].astype(np.float64)
    assert p4.shape == (16472,)
    nv = int((p4 > TOL).sum())
    assert nv == fam[fk]['n_viol'], (fk, nv)
    assert abs(float(p4.max())
               - fam[fk]['worst_excess']) < 1e-12
    # Moebius order census
    mu = npz['MU_' + fk].astype(np.float64)
    assert mu.shape == (256,)
    for k in (2, 3, 4, 5, 6, 7, 8):
        idx = [m for m in range(0, 256) if pc(m) == k]
        mk = mu[idx]
        spec = fam[fk]['spectrum'][str(k)]
        assert int((mk > TOL).sum()) \
            == spec['n_pos'], (fk, k)
        assert abs(float(mk.max())
                   - spec['max']) < 1e-12, (fk, k)
    assert abs(float(npz['VIOL_RATE_' + fk])
               - fam[fk]['viol_rate']) < 1e-15
    assert int(npz['STABLE_' + fk]) == (
        1 if fam[fk]['stable'] else 0)

# --- cross-family recomputation ---
ov = {}
for a, b in (('A', 'B'), ('A', 'C'), ('B', 'C')):
    inter = len(top8s[a] & top8s[b])
    uni = len(top8s[a] | top8s[b])
    ov[a + b] = (inter, inter / uni)
    assert inter == cross['ov_' + a.lower()
                          + b.lower()], (a, b, inter)
    assert abs(ov[a + b][1]
               - cross['jac_' + a.lower()
                       + b.lower()]) < 1e-12
assert ov['AB'][0] == 5 and ov['AC'][0] == 4
assert ov['BC'][0] == 1
common = top8s['A'] & top8s['B'] & top8s['C']
assert common == {14}, common
sp_r1 = {
    'AB': spearman(r1s['A'], r1s['B']),
    'AC': spearman(r1s['A'], r1s['C']),
    'BC': spearman(r1s['B'], r1s['C'])}
sp_dah = {
    'AB': spearman(dahs['A'], dahs['B']),
    'AC': spearman(dahs['A'], dahs['C']),
    'BC': spearman(dahs['B'], dahs['C'])}
for k in ('AB', 'AC', 'BC'):
    assert abs(sp_r1[k] - cross['sp_r1_'
                                + k.lower()]) < 1e-9
    assert abs(sp_dah[k] - cross['sp_dah_'
                                 + k.lower()]) < 1e-9
assert sp_dah['AB'] > 0.8 and sp_r1['BC'] < 0
assert sp_r1['AC'] < 0.2

# --- verdict tree replay ---
n_stable = sum(1 for fk in FK if fam[fk]['stable'])
assert n_stable == 1
assert n_stable < 2
assert res['verdict'] == 'cross_prompt_unstable'

# --- ledger chain ---
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
assert len(led['measurements']) == 215
assert len(l14['connects']) == 183
m76 = [m for m in led['measurements']
       if m.get('phase') == 3076]
assert len(m76) == 1
assert m76[0]['verdict'] == 'cross_prompt_unstable'
stored = led['ledger_sha256_8']
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
assert stored == hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]

# --- memo / audit / wlog / memory ---
memo = io.open(MEMO, encoding='utf-8').read()
pos = memo.rfind('## Phase 3076:')
assert pos > 0
assert '## Phase 3077:' not in memo
head = memo[pos:pos + 120]
assert '[2026-09-21 21:26:53]' in head
assert 'cross_prompt_unstable' in memo[pos:]
assert '16 重跨 phase 锚全 bit 0.0' in memo[pos:]
aud = io.open(AUDIT, encoding='utf-8').read()
assert '## 三十八、3076 增补' in aud
wl = io.open(WLOG, encoding='utf-8').read()
assert 'Phase 3076' in wl
memw = io.open(MEMW, encoding='utf-8').read()
assert 'max=3076' in memw

print('VERIFY_OK')
