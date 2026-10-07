# -*- coding: utf-8 -*-
"""Phase 3077 independent verify: recompute key
statistics from the frozen npz, recheck seals and
the closeout chain.  Prints VERIFY_OK only if all
checks pass."""
import hashlib
import io
import json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = BASE + (r'\phase3077\omega_p74_write_routing')
P76 = BASE + (r'\phase3076\omega_p73_cross_prompt_'
              r'family')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
SCRIPT = (ROOT + r'\tests\glm5\phase3077_omega_'
          r'p74_write_routing.py')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-21.md'
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
FK = ('A', 'B', 'C')


def sha8(path):
    return hashlib.sha256(
        io.open(path, 'rb').read()).hexdigest()[:8]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def anova2(M):
    M = np.asarray(M, dtype=np.float64)
    gm = M.mean()
    alpha = M.mean(axis=0) - gm
    beta = M.mean(axis=1) - gm
    eps = M - gm - alpha[None, :] - beta[:, None]
    nf, nh = M.shape
    v_head = float((alpha ** 2).sum() / (nh - 1))
    v_fam = float((beta ** 2).sum() / (nf - 1))
    v_eps = float((eps ** 2).sum()
                  / ((nf - 1) * (nh - 1)))
    return {'alpha': alpha, 'eps': eps,
            'icc': v_head / (v_head + v_eps)}


res = json.load(io.open(R + r'\result.json',
                        encoding='utf-8'))
seal = json.load(io.open(R + r'\seal.json',
                         encoding='utf-8'))
st = res['stats']
z77 = np.load(R + r'\omega_p74_write_routing.npz')
z76 = np.load(P76 + r'\omega_p73_cross_prompt_family.npz')

# --- seals ---
assert seal['npz_sha256_8'] == sha8(
    R + r'\omega_p74_write_routing.npz')
assert seal['result_sha256_8'] == sha8(
    R + r'\result.json')
assert seal['exec_sha256_8'] == sha8(
    R + r'\execution.json')
assert seal['script_sha256_8'] == sha8(SCRIPT)
assert seal['verdict'] == ('observation_'
                           'causation_decoupled')
assert res['forwards'] == 0

# --- recompute ICC from npz R matrix ---
Rm = z77['R1_3x32'].astype(np.float64)
# cross-source: npz R matrix must equal 3076 R1
for fi, f in enumerate(FK):
    assert np.abs(Rm[fi] - z76['R1_ALL32_' + f]
                  .astype(np.float64)).max() == 0.0
aR = anova2(Rm)
assert abs(aR['icc'] - st['icc']['R']
           ['icc_head']) < 1e-12
Dm = z77['ABS_D34_3x32'].astype(np.float64)
for fi, f in enumerate(FK):
    assert np.abs(Dm[fi] - np.abs(
        z76['DAH34_MED_' + f].astype(np.float64))
    ).max() == 0.0
aD = anova2(Dm)
assert abs(aD['icc'] - st['icc']['absD34']
           ['icc_head']) < 1e-12
assert aD['icc'] > aR['icc'] > 0.4
assert aD['icc'] < 0.95

# --- recompute g5 migration spearman ---
r1a = z76['R1_ALL32_A'].astype(np.float64)
sp_ab = spearman(r1a, z76['R1_ALL32_B']
                 .astype(np.float64))
sp_ac = spearman(r1a, z76['R1_ALL32_C']
                 .astype(np.float64))
byid = {e['id']: e for e in st['features']}
assert abs(byid['g5']['sp']['B'] - sp_ab) < 1e-12
assert abs(byid['g5']['sp']['C'] - sp_ac) < 1e-12
assert abs(sp_ab - 0.6763196480938416) < 1e-12
assert abs(sp_ac - 0.13049853372434017) < 1e-12
assert abs(byid['g5']['mean_abs_sp']
           - (abs(sp_ab) + abs(sp_ac)) / 2) < 1e-12
assert byid['g5']['mean_abs_sp'] < 0.6
for e in st['features']:
    assert e['mean_abs_sp'] < 0.6, e['id']
    if e['id'] in ('g4', 'g5', 'g6', 'g11',
                   'g12'):
        assert 'A' not in e['sp'], e['id']

# --- recompute alpha/eps overlaps ---
T8 = {f: set(int(x) for x in z76['TOP8_' + f])
      for f in FK}
al8 = set(int(h) for h in
          np.argsort(aR['alpha'])[:8])
assert al8 == set(st['alpha_top8'])
assert len(al8 & T8['A']) == 8
assert len(al8 & T8['B']) == 5
assert len(al8 & T8['C']) == 4
for fi, f in enumerate(FK):
    e8 = set(int(h) for h in
             np.argsort(aR['eps'][fi])[:8])
    assert (len(e8 & T8[f])
            == st['eps_overlap'][f]['overlap'])

# --- sign flip replay ---
own = []
oth = []
for f in FK:
    for h in sorted(T8[f]):
        own.append(Rm[FK.index(f)][h])
        oth.extend(Rm[FK.index(g)][h]
                   for g in FK if g != f)
sf = float(np.mean(oth) - np.mean(own))
assert abs(sf - st['sign_flip']['delta']) < 1e-12
assert st['sign_flip']['p_perm'] < 0.05

# --- gates + verdict replay ---
g = st['gates']
assert g['G1'] is False and g['G2'] is False
assert g['G3'] is True and g['G4'] is False
assert res['verdict'] == ('observation_'
                          'causation_decoupled')

# --- ledger chain ---
led = json.load(io.open(LEDGER, encoding='utf-8'))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
assert len(led['measurements']) == 216
assert len(l14['connects']) == 184
m77 = [m for m in led['measurements']
       if m.get('phase') == 3077]
assert len(m77) == 1
assert m77[0]['verdict'] == ('observation_'
                             'causation_decoupled')
stored = led['ledger_sha256_8']
led.pop('ledger_sha256_8')
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
assert stored == hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]

# --- memo / audit / wlog / memory ---
memo = io.open(MEMO, encoding='utf-8').read()
pos = memo.rfind('## Phase 3077:')
assert pos > 0
assert '## Phase 3078:' not in memo
head = memo[pos:pos + 140]
assert '[2026-09-21 22:13:42]' in head
assert 'observation_causation_decoupled' \
    in memo[pos:]
assert '来源族特征不得在来源族上评估' in memo[pos:]
aud = io.open(AUDIT, encoding='utf-8').read()
assert '## 三十九、3077 增补' in aud
wl = io.open(WLOG, encoding='utf-8').read()
assert 'Phase 3077' in wl
memw = io.open(MEMW, encoding='utf-8').read()
assert 'max=3077' in memw

print('VERIFY_OK')
