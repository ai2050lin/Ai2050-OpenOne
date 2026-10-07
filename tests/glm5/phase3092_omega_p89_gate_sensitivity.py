# -*- coding: utf-8 -*-
"""Phase 3092 - Omega-P89: G_DS gate sensitivity
formalization (forward-free).

Question: the 3085/3087/3089 mixed_absent
verdicts hinge on the preregistered sign-aware
G_DS count gate (>=4/6 positive-significant).
3089's recorded unsigned Stouffer z=3.150 is
direction-agnostic (small p counts as positive
evidence regardless of sp sign) and superficially
suggests "present".  Are the mixed_absent
verdicts gate artifacts?

Units: the four arbitration phases with complete
6-test blocks (f2_cTT x {T,U} x {AB,AC,BC}):
  3081 DS7B (ds7b_cos_absent)
  3085 3B L34 (third_mixed_absent)
  3087 A2 GLM4 L37 (fourth_mixed_absent)
  3089 GLM4 L38 (fourth_l38_mixed_absent)

Gates (preregistered decision table, frozen in
execution.json before any computation):
  G0 (actual preregistered gate):
     cnt_pos>=4 AND min_sp>0
  G1 signed Stouffer: z = sum(sign(sp)*z(p))/sqrt(6)
     present iff z > 1.645
  G2 unsigned Stouffer (3089 recorded form):
     z = sum(z(p))/sqrt(6), present iff z > 1.645
  G3 Bonferroni: any test p<0.05/6 AND sp>0
  where z(p) = NormalDist().inv_cdf(1-p).

Bit-checks: recompute cnt_pos / n_bonf / min_sp /
unsigned Stouffer from stored (sp, p) and require
bit equality with each phase's stored GDS_COUNT /
GDS_N_BONF / GDS_MIN_SP / STOUFFER_Z.

Preregistered prediction: 3089 flips ONLY under G2
(unsigned) - G0/G1/G3 all absent; no other phase
flips under any gate.

Verdict tree:
  gate_unsigned_inflates   prediction holds
                           (G2 is the only flipping
                           gate, direction: towards
                           present)
  gate_order_stable        no phase flips under
                           any gate
  gate_sensitive_substantive
                           G0/G1/G3 disagree on any
                           phase, or G2 flips any
                           other phase
"""
import hashlib
import io
import json
import os
import time
from statistics import NormalDist

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
IN = {
    'p3081': (RES + r'\phase3081'
              r'\omega_p78_ds7b_crossmodel'
              r'\omega_p78_ds7b_crossmodel.npz',
              'ds7b_cos_absent'),
    'p3085': (RES + r'\phase3085'
              r'\omega_p82_l34_full_arbitration'
              r'\omega_p82_l34_full_arbitration.npz',
              'third_mixed_absent'),
    'p3087': (RES + r'\phase3087'
              r'\omega_p85_glm4_l37_full_'
              r'arbitration'
              r'\omega_p85_glm4_l37_full_'
              r'arbitration.npz',
              'fourth_mixed_absent'),
    'p3089': (RES + r'\phase3089'
              r'\omega_p87_glm4_l38_full_'
              r'arbitration'
              r'\omega_p87_glm4_l38_full_'
              r'arbitration.npz',
              'fourth_l38_mixed_absent'),
}
NAME = 'omega_p89_gate_sensitivity'
OUT = RES + r'\phase3092' + '\\' + NAME
TESTS = (('T', 'AB'), ('T', 'AC'), ('T', 'BC'),
         ('U', 'AB'), ('U', 'AC'), ('U', 'BC'))

o = []


def log(msg):
    o.append(msg)
    print(msg)


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


# ---- load inputs ----
zs = {}
sha_in = {}
for tag, (p, exp_v) in IN.items():
    z = np.load(p, allow_pickle=False)
    zs[tag] = z
    sha_in[tag] = sha8(p)
    assert str(z['VERDICT']) == exp_v, (tag,
        str(z['VERDICT']))
    assert bool(z['SMOKE']) is False
log('inputs loaded, sha8 %s' % sha_in)

# ---- cleanup stale artifacts ----
os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'result.json',
           'execution.json', 'seal.json',
           'run_log.txt'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')

# ---- freeze design BEFORE computing ----
design = {
    'phase': 3092,
    'name': 'Omega-P89 G_DS gate '
            'sensitivity (forward-free)',
    'created': created,
    'forward_free': True,
    'inputs': sha_in,
    'gates': {
        'G0': 'cnt_pos>=4 AND min_sp>0 '
              '(preregistered decision gate)',
        'G1': 'signed Stouffer z>1.645: '
              'sum(sign(sp)*inv_cdf(1-p))/sqrt(6)',
        'G2': 'unsigned Stouffer z>1.645: '
              'sum(inv_cdf(1-p))/sqrt(6) '
              '(direction-agnostic, 3089 '
              'recorded form)',
        'G3': 'Bonferroni: any p<0.05/6 '
              'AND sp>0',
    },
    'prereg_prediction':
        '3089 flips ONLY under G2 (towards '
        'present); G0/G1/G3 unanimous absent '
        'on all four phases',
    'verdict_tree': [
        'gate_unsigned_inflates: prediction '
        'holds exactly',
        'gate_order_stable: no flips at all',
        'gate_sensitive_substantive: any '
        'other disagreement',
    ],
}
with io.open(os.path.join(OUT,
              'execution.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, ensure_ascii=False,
              indent=1)
log('execution.json frozen (before '
    'computation)')

# ---- compute ----
inv = NormalDist().inv_cdf
results = {}
bits_ok = True
for tag in ('p3081', 'p3085', 'p3087',
            'p3089'):
    z = zs[tag]
    sps, ps = [], []
    for rn, pr in TESTS:
        sps.append(float(z['E3_F2_CTT_'
                              + rn + '_'
                              + pr]))
        ps.append(float(z['E3P_F2_CTT_'
                              + rn + '_'
                              + pr]))
    # bit-checks vs stored gate quantities
    cnt = sum(1 for s, p in zip(sps, ps)
              if s > 0 and p < 0.05)
    nbonf = sum(1 for p in ps
                if p < 0.05 / 6)
    min_sp = min(sps)
    zl = [inv(min(1.0 - 1e-12, 1.0 - p))
          for p in ps]
    st_un = float(np.sum(zl)
                  / np.sqrt(len(zl)))
    b_cnt = (cnt == int(z['GDS_COUNT']))
    b_nb = (nbonf
            == int(z['GDS_N_BONF']))
    b_ms = (min_sp
            == float(z['GDS_MIN_SP']))
    b_st = (st_un
            == float(z['STOUFFER_Z']))
    bits_ok = (bits_ok and b_cnt and b_nb
               and b_ms and b_st)
    log('%s bit-checks cnt=%s bonf=%s '
        'minsp=%s stouffer=%s (cnt=%d '
        'bonf=%d minsp=%+.4f st=%.4f)'
        % (tag, b_cnt, b_nb, b_ms, b_st,
           cnt, nbonf, min_sp, st_un))
    assert b_cnt and b_nb and b_ms and b_st
    # gates
    g0 = (cnt >= 4 and min_sp > 0)
    z_sig = float(np.sum(
        [(1.0 if s > 0 else -1.0)
         * inv(min(1.0 - 1e-12, 1.0 - p))
         for s, p in zip(sps, ps)])
        / np.sqrt(6))
    g1 = (z_sig > 1.645)
    g2 = (st_un > 1.645)
    g3 = any(s > 0 and p < 0.05 / 6
             for s, p in zip(sps, ps))
    stored_cnt = int(z['GDS_COUNT'])
    g0_stored = (stored_cnt >= 4
                 and min_sp > 0)
    assert g0 == g0_stored, (tag, 'G0 '
                             'mismatch vs '
                             'stored count')
    results[tag] = {
        'stored_verdict': IN[tag][1],
        'sps': sps, 'ps': ps,
        'cnt_pos': cnt, 'n_bonf': nbonf,
        'min_sp': min_sp,
        'stouffer_unsigned': st_un,
        'stouffer_signed': z_sig,
        'G0_preregistered': g0,
        'G1_signed_stouffer': g1,
        'G2_unsigned_stouffer': g2,
        'G3_bonferroni': g3,
    }
    log('%s gates: G0=%s G1=%s G2=%s G3=%s '
        '(z_sig=%+.4f z_un=%.4f)'
        % (tag, g0, g1, g2, g3, z_sig,
           st_un))
assert bits_ok

# ---- verdict ----
g0 = [results[t]['G0_preregistered']
      for t in results]
g1 = [results[t]['G1_signed_stouffer']
      for t in results]
g2 = [results[t]['G2_unsigned_stouffer']
      for t in results]
g3 = [results[t]['G3_bonferroni']
      for t in results]
pred_3089 = (results['p3089'][
    'G2_unsigned_stouffer']
    and not results['p3089'][
        'G1_signed_stouffer']
    and not results['p3089'][
        'G0_preregistered']
    and not results['p3089'][
        'G3_bonferroni'])
others_stable = all(
    not results[t][k] for t in
    ('p3081', 'p3085', 'p3087')
    for k in ('G0_preregistered',
              'G1_signed_stouffer',
              'G2_unsigned_stouffer',
              'G3_bonferroni'))
if pred_3089 and others_stable:
    verdict = 'gate_unsigned_inflates'
elif (g0 == g1 == g2 == g3
      and not any(g0)):
    verdict = 'gate_order_stable'
else:
    verdict = 'gate_sensitive_substantive'
log('VERDICT: %s (pred_3089=%s '
    'others_stable=%s)'
    % (verdict, pred_3089, others_stable))

elapsed = time.time() - t0

# ---- npz ----
save = {
    'FORWARD_FREE': np.bool_(True),
    'SETUP_OK': np.bool_(True),
    'ANCH_BITS_ALL': np.bool_(bits_ok),
    'N_PHASES': np.int64(len(results)),
}
for tag in results:
    r = results[tag]
    save['CNT_POS_' + tag] = np.int64(
        r['cnt_pos'])
    save['N_BONF_' + tag] = np.int64(
        r['n_bonf'])
    save['MIN_SP_' + tag] = np.float64(
        r['min_sp'])
    save['ST_UN_' + tag] = np.float64(
        r['stouffer_unsigned'])
    save['ST_SIG_' + tag] = np.float64(
        r['stouffer_signed'])
    for gname in ('G0_preregistered',
                  'G1_signed_stouffer',
                  'G2_unsigned_stouffer',
                  'G3_bonferroni'):
        save['%s_%s' % (gname.upper(),
                        tag)] = np.bool_(
            r[gname])
    for i, (rn, pr) in enumerate(TESTS):
        save['SP_%s_%s_%s' % (tag, rn,
                              pr)] = \
            np.float64(r['sps'][i])
        save['P_%s_%s_%s' % (tag, rn,
                             pr)] = \
            np.float64(r['ps'][i])
save['ELAPSED'] = np.float64(elapsed)
save['VERDICT'] = np.str_(verdict)
npz_path = os.path.join(OUT,
                        NAME + '.npz')
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ---- result.json ----
result = {
    'phase': 3092,
    'name': NAME,
    'created': created,
    'forward_free': True,
    'forwards': 0,
    'elapsed': elapsed,
    'inputs_sha8': sha_in,
    'gates': design['gates'],
    'phases': results,
    'verdict': verdict,
    'verdict_tree': design['verdict_tree'],
    'prereg_prediction':
        design['prereg_prediction'],
}
rj = os.path.join(OUT, 'result.json')
with io.open(rj, 'w', encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False, indent=1)
log('result.json saved')

# ---- seal (3089 key convention) ----
seal = {
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(rj),
    'exec8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8':
        sha8(os.path.abspath(__file__)),
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec8'],
       seal['script_sha256_8'], elapsed))

with io.open(os.path.join(OUT,
              'run_log.txt'),
             'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('RUN_COMPLETE %s' % verdict)
