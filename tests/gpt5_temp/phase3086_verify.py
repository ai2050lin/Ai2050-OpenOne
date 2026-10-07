# -*- coding: utf-8 -*-
"""Phase 3086 independent verify (bit-replay).
Checks: seal hashes, npz scalars, anchors, unit
values vs result.json AND vs source npz
re-derivation, spearman/perm bit replay, ledger,
MEMO/audit/wlog/MEMORY, smoke artifacts."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = BASE + r'\phase3086\omega_p83_continuum_test'
RS = R + r'\smoke'
S79 = BASE + (r'\phase3079\omega_p76_migration_'
              r'lock\omega_p76_migration_lock.npz')
S81 = BASE + (r'\phase3081\omega_p78_ds7b_'
              r'crossmodel\omega_p78_ds7b_'
              r'crossmodel.npz')
S82 = BASE + (r'\phase3082\omega_p79_ds7b_negative_'
              r'anatomy\omega_p79_ds7b_negative_'
              r'anatomy.npz')
S85 = BASE + (r'\phase3085\omega_p82_l34_full_'
              r'arbitration\omega_p82_l34_full_'
              r'arbitration.npz')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-22.md'
MEM = ROOT + r'\.workbuddy\memory\MEMORY.md'
OUTF = R + r'\verify_log.txt'

n_chk = [0]
n_fail = [0]
fails = []


def chk(cond, msg):
    n_chk[0] += 1
    if not cond:
        n_fail[0] += 1
        fails.append(msg)


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def perm_p(a, b, n_perm=20000, seed=3086):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    if n_perm <= 0:
        return 1.0
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (rb * ra[None, :]).sum(axis=1)
    den = np.sqrt(
        (rb * rb).sum(axis=1)
        * float((ra * ra).sum()))
    den[den == 0] = 1.0
    stats = np.abs(num / den)
    return float((stats >= obs - 1e-12)
                 .mean())


# ==== A. seal ====
res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
exe = json.load(io.open(
    R + r'\execution.json', encoding='utf-8'))
z = np.load(R + r'\omega_p83_continuum_'
            r'test.npz', allow_pickle=False)
chk(seal['npz_sha256_8'] == sha8(
    R + r'\omega_p83_continuum_test.npz'),
    'seal npz8')
chk(seal['result_sha256_8'] == sha8(
    R + r'\result.json'), 'seal result8')
chk(seal['exec_sha256_8'] == sha8(
    R + r'\execution.json'), 'seal exec8')
chk(seal['script_sha256_8'] == sha8(
    ROOT + r'\tests\glm5'
    r'\phase3086_omega_p83_continuum_test.py'),
    'seal script8')
chk(seal['verdict'] == 'continuum_confirmed',
    'seal verdict')
chk(seal['setup_ok'] is True, 'seal setup_ok')
chk(seal['phase'] == 3086, 'seal phase')
chk(seal['name'] == 'omega_p83_continuum_test',
    'seal name')

# ==== B. npz scalars ====
chk(str(z['VERDICT'])
    == 'continuum_confirmed', 'npz verdict')
chk(bool(z['SMOKE']) is False, 'npz smoke false')
chk(int(z['N_PERM']) == 20000, 'npz n_perm')
chk(int(z['SEED']) == 3086, 'npz seed')
chk(int(z['N_UNITS']) == 9, 'npz n_units')
chk(bool(z['SETUP_OK']), 'npz setup_ok')
chk(int(z['FORWARDS']) == 0, 'npz forwards 0')

# ==== C. anchors ====
chk(bool(z['ANCH_A1']), 'anchor a1')
chk(bool(z['ANCH_A2']), 'anchor a2')
chk(bool(z['ANCH_A3']), 'anchor a3')

# ==== D. source hashes ====
h = {'S79': sha8(S79), 'S81': sha8(S81),
     'S82': sha8(S82), 'S85': sha8(S85)}
for k in ('S79', 'S81', 'S82', 'S85'):
    chk(str(z['SHA' + k[1:]]) == h[k],
        'npz sha %s' % k)
    chk(res['stats']['sha'][k] == h[k],
        'result sha %s' % k)

# ==== E. rho/p vs result.json ====
st = res['stats']['tests']
chk(abs(float(z['RHO_T']) - st['rho_T']) == 0.0,
    'rho_T match')
chk(abs(float(z['P_T']) - st['p_T']) == 0.0,
    'p_T match')
chk(abs(float(z['RHO_U']) - st['rho_U']) == 0.0,
    'rho_U match')
chk(abs(float(z['P_U']) - st['p_U']) == 0.0,
    'p_U match')
chk(abs(float(z['RHO_MIG'])
        - st['rho_MIG']) == 0.0, 'rho_M match')
chk(abs(float(z['P_MIG'])
        - st['p_MIG']) == 0.0, 'p_M match')
chk(float(z['RHO_T']) > 0
    and float(z['P_T']) < 0.05,
    'verdict condition rho_T>0 p<0.05')
chk(abs(float(z['SP_MEAN_T'])
        - st['sp_mean_T']) == 0.0, 'sp_mean_T')
chk(abs(float(z['SP_MEAN_U'])
        - st['sp_mean_U']) == 0.0, 'sp_mean_U')
chk(abs(float(z['SP_MEAN_MIG'])
        - st['sp_mean_MIG']) == 0.0,
    'sp_mean_MIG')

# ==== G/H. units vs result.json + key form ====
TAGS = ['M3B_AB', 'M3B_AC', 'M3B_BC',
        'M4B_AB', 'M4B_AC', 'M4B_BC',
        'MDS7B_AB', 'MDS7B_AC', 'MDS7B_BC']
KEYS = {'S_LO_': 's_lo', 'S_MEAN_': 's_mean',
        'T_MED_': 'T_med', 'U_MED_': 'U_med',
        'MIG_': 'MIG'}
units = res['stats']['units']
chk(len(units) == 9, 'result units n=9')
for pre, uk in KEYS.items():
    for tg in TAGS:
        kk = pre + tg
        chk(kk in z.files, 'npz key %s' % kk)
        src = units[tg[1:]]
        chk(abs(float(z[kk])
                - src[uk]) == 0.0,
            'unit %s == result' % kk)
for k in z.files:
    if k.startswith(('S_LO_', 'S_MEAN_',
                     'T_MED_', 'U_MED_',
                     'MIG_M')):
        chk('DS7M4B' not in k,
            'no tag debris %s' % k)

# ==== O. independent re-derivation from sources ====
z79 = np.load(S79, allow_pickle=False)
z81 = np.load(S81, allow_pickle=False)
z82 = np.load(S82, allow_pickle=False)
z85 = np.load(S85, allow_pickle=False)
SRC = {'4B': (z79, z82, 'SP_R1_REPLAY',
              '4B_'),
       'DS7B': (z81, z82, 'MIG', 'DS7B_'),
       '3B': (z85, z85, 'MIG', '')}
FK = ('A', 'B', 'C')
for mn, (zn, zt, mref, pfk) in SRC.items():
    t3 = {fk: float(zt['E3_TOP3_CS_' + pfk
                        + fk]) for fk in FK}
    for p in ('AB', 'AC', 'BC'):
        u = units['%s_%s' % (mn, p)]
        lo = min(t3[p[0]], t3[p[1]])
        chk(abs(u['s_lo'] - lo) < 1e-15,
            'rederive s_lo %s_%s' % (mn, p))
        tm = float(np.median(
            zn['T_' + p]))
        chk(abs(u['T_med'] - tm) < 1e-15,
            'rederive T_med %s_%s' % (mn, p))
        um = float(np.median(
            zn['U_' + p]))
        chk(abs(u['U_med'] - um) < 1e-15,
            'rederive U_med %s_%s' % (mn, p))
        mg = float(zn[mref + '_' + p])
        chk(abs(u['MIG'] - mg) < 1e-15,
            'rederive MIG %s_%s' % (mn, p))

# ==== I. bit replay of the main test ====
order = sorted(units)
s_lo = np.array([units[k]['s_lo']
                 for k in order])
s_mean = np.array([units[k]['s_mean']
                   for k in order])
T_med = np.array([units[k]['T_med']
                  for k in order])
U_med = np.array([units[k]['U_med']
                  for k in order])
MIG = np.array([units[k]['MIG']
                for k in order])
r1 = spearman(s_lo, T_med)
p1 = perm_p(s_lo, T_med)
chk(r1 == float(z['RHO_T']),
    'replay rho_T bit')
chk(p1 == float(z['P_T']),
    'replay p_T bit')
r2 = spearman(s_lo, U_med)
p2 = perm_p(s_lo, U_med)
chk(r2 == float(z['RHO_U']),
    'replay rho_U bit')
chk(p2 == float(z['P_U']),
    'replay p_U bit')
r3 = spearman(s_lo, MIG)
p3 = perm_p(s_lo, MIG)
chk(r3 == float(z['RHO_MIG']),
    'replay rho_MIG bit')
chk(p3 == float(z['P_MIG']),
    'replay p_MIG bit')
smT = spearman(s_mean, T_med)
chk(smT == float(z['SP_MEAN_T']),
    'replay sp_mean_T bit')

# ==== J. execution.json ====
chk(exe.get('phase') == 3086, 'exec phase')
chk(exe.get('name')
    == 'omega_p83_continuum_test',
    'exec name')
chk(exe.get('smoke') is False,
    'exec smoke false')
chk(str(exe.get('created', '')).startswith(
    '2026-09-'), 'exec created')
chk(len(json.dumps(exe.get('prereg', {})))
    > 200, 'exec prereg present')

# ==== K. run_log ====
rl = io.open(R + r'\run_log.txt',
             encoding='utf-8').read()
chk('VERDICT: continuum_confirmed' in rl,
    'runlog verdict')
chk('SETUP_OK=True' in rl, 'runlog setup')
chk('a1 sha chain' in rl
    and '-> True' in rl, 'runlog a1')
chk('smoke=False' in rl, 'runlog full mode')

# ==== L. ledger ====
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk(len(led['measurements']) == 225,
    'ledger n=225')
m86 = [m for m in led['measurements']
       if isinstance(m, dict)
       and m.get('phase') == 3086]
chk(len(m86) == 1, 'meas3086 present')
if m86:
    mc = m86[0]
    chk(mc['meas_id']
        == 'meas3086_omega_p83_'
           'continuum_test', 'meas id')
    chk(mc['verdict']
        == 'continuum_confirmed',
        'meas verdict')
    chk(mc['hashes']['npz_sha256_8']
        == seal['npz_sha256_8'],
        'meas npz hash')
    chk(mc['hashes']['result_sha256_8']
        == seal['result_sha256_8'],
        'meas result hash')
    chk('+0.8667' in mc['claim'],
        'meas claim rho')
    chk(mc['claim'].count(
        'continuum_confirmed') >= 1,
        'meas claim verdict')
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_'
          'cross_model'][0]
chk(len(l14['connects']) == 193,
    'L14 n=193')
last = [c for c in l14['connects']
        if isinstance(c, dict)
        and c.get('phase') == 3086]
chk(len(last) == 1, 'L14 3086 entry')
if last:
    chk('Omega-P83' in
        last[0]['grade_change'],
        'L14 grade text')

# ==== M. docs ====
memo = io.open(MEMO, encoding='utf-8').read()
chk('## Phase 3086:' in memo, 'memo 3086 head')
chk('continuum_confirmed' in memo,
    'memo verdict')
chk('rho_T=+0.867' in memo,
    'memo title rho')
chk(memo.rstrip().endswith('3087 A.')
    or '3087' in memo[-3000:],
    'memo menu tail')
aud = io.open(AUDIT, encoding='utf-8').read()
chk('## 四十八、3086' in aud, 'audit 48')
chk('continuum_confirmed' in aud,
    'audit verdict')
wl = io.open(WLOG, encoding='utf-8').read()
chk('Phase 3086' in wl, 'wlog 3086')
mem = io.open(MEM, encoding='utf-8').read()
chk('max=3086' in mem, 'memory max=3086')
chk(len(mem) < 3000, 'memory len<3000')

# ==== N. smoke artifacts ====
zs = np.load(RS + r'\omega_p83_continuum_'
             r'test.npz', allow_pickle=False)
chk(str(zs['VERDICT'])
    == 'continuum_confirmed',
    'smoke verdict')
chk(int(zs['N_PERM']) == 1000,
    'smoke n_perm=1000')
chk(bool(zs['SMOKE']) is True,
    'smoke flag true')
chk(abs(float(zs['RHO_T'])
        - float(z['RHO_T'])) < 1e-12,
    'smoke rho equals full rho')

# ==== report ====
lines = ['VERIFY n=%d fail=%d'
         % (n_chk[0], n_fail[0])]
lines += ['FAIL: %s' % f for f in fails]
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(lines) + '\n')
print('VERIFY_DONE fail=%d' % n_fail[0])
