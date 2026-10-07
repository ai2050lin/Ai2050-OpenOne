# -*- coding: utf-8 -*-
"""Phase 3097  Omega-P95: cliff layer
decomposition (forward-free).

3096 (omega_p94) found the 14B f2
collapse is a readout-assembly event
concentrated in the final ~4 layers
(d=0.9->1.0 folded L36..L40).  Open
question: WHICH step inside the final
segment carries the cliff -- one single
block step, or a gradual slide?

Data (already sealed): F2P_{4B,14B}_{AB,
AC,BC} per-pair f2 over ALL layers
(24 x (NL+1)); 4B NL=36, 14B NL=40.
Column L<NL = head(norm(h_L)) with h_L
the residual entering block L (L=0
embedding); column L=NL = head(h_NL) =
native logits (h_NL post-final-norm).
Step L->L+1 thus bundles block L's
computation with the re-norm; the final
step (NL-1 -> NL) additionally contains
the final-norm switch from the lens
path to the native path.

Anchors (frozen):
  c1: F2P_14B final column vs 3093
      sealed F2_CTT <= 1e-9.
  c2: F2P_4B final column vs 3079
      sealed F2_CTT <= 1e-9.
  c3: F2P medians re-render the 3096
      D_GRID curves bit-equal at the
      grid layers <= 1e-12.
  c4: 3096 npz sha8 equals the value
      recorded in 3096 seal.json.

Preregistered gates (per model; last4 =
layers [NL-4, NL-1], prev = [NL-10,
NL-5]; steps are consecutive f2 drops
of the 8-pair ci median curve):
  H_C1 single_step_cliff:
      max_step(last4) >= 0.15 AND
      max_step(last4) >= 2 x
      max_step(prev)  (on 14B AC/BC
      ci1 -- the collapsed groups).
  H_C2 gradual:
      max_step(L in NL-10..NL-1) < 0.15
      on the same groups.
  H_C3 readout_amplify:
      on ci2 groups the last4 steps are
      predominantly positive (>= 3 of
      the 4 steps have delta f2 > 0).
  order: H_C1 -> H_C2 -> H_C3 -> mixed;
  verdicts fifth_cliff_single_step /
  fifth_cliff_gradual /
  fifth_cliff_readout_amplify /
  fifth_cliff_mixed / (all fail on both
  readings) fifth_cliff_inconclusive.
H_C1/H_C2 are evaluated on 14B; 4B is
the control arm (reported, not gating).

Limitations: step deltas bundle one
block computation with the re-norm (the
final-norm contribution cannot be
separated from block NL-1 without a
raw-head probe, recorded as follow-up);
medians over 8 pairs; lens caveat as in
3096.
"""
import hashlib
import io
import json
import os

import numpy as np

PHASE = 3097
NAME = 'omega_p95_cliff_layer_decomposition'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P3096 = (R13 + r'\phase3096'
         r'\omega_p94_f2_lens_layer_profile'
         r'\omega_p94_f2_lens_layer_profile.npz')
P3096RES = (R13 + r'\phase3096'
            r'\omega_p94_f2_lens_layer_profile'
            r'\result.json')
P3096SEAL = (R13 + r'\phase3096'
             r'\omega_p94_f2_lens_layer_profile'
             r'\seal.json')
P4B79 = (R13 + r'\phase3079'
         r'\omega_p76_migration_lock'
         r'\omega_p76_migration_lock.npz')
P14B = (R13 + r'\phase3093'
        r'\omega_p91_qwen14b_l37_full_arbitration'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration.npz')
OUT = (R13 + r'\phase3097' + '\\' + NAME)
LOGF = OUT + r'\run_log.txt'
LOGS = []
CIDX = np.array([k // 8 + 1
                 for k in range(24)])


def log(msg):
    LOGS.append(msg)


def h8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20),
                        b''):
            h.update(blk)
    return h.hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)

z96 = np.load(P3096, allow_pickle=False)
z4_79 = np.load(P4B79, allow_pickle=False)
z14 = np.load(P14B, allow_pickle=False)
res96 = json.load(io.open(
    P3096RES, encoding='utf-8'))
seal96 = json.load(io.open(
    P3096SEAL, encoding='utf-8'))

GATES = {
    'H_C1_single_step_cliff': {
        'max_step_last4_ge': 0.15,
        'ratio_ge': 2.0},
    'H_C2_gradual': {
        'max_step_all_last10_lt': 0.15},
    'H_C3_readout_amplify': {
        'pos_steps_ge': 3},
    'step_def': ('delta f2 of the 8-pair '
                 'ci median curve, '
                 'consecutive layers')}
exec_doc = {
    'phase': PHASE, 'name': NAME,
    'frozen_before_compute': True,
    'question': ('which step inside the '
                 'final segment carries '
                 'the 14B f2 cliff'),
    'gates': GATES,
    'decision_order': ['H_C1', 'H_C2',
                       'H_C3', 'mixed',
                       'inconclusive'],
    'anchors': {
        'c1': 'F2P_14B[:,NL] vs 3093 '
              'F2_CTT <=1e-9',
        'c2': 'F2P_4B[:,NL] vs 3079 '
              'F2_CTT <=1e-9',
        'c3': 'F2P grid medians vs 3096 '
              'result curves <=1e-12',
        'c4': '3096 npz sha8 vs its '
              'seal.json'},
    'inputs': {
        'p3096_npz8': h8(P3096),
        'p4b79_npz8': h8(P4B79),
        'p14b_npz8': h8(P14B)},
    'forward_free': True}
with io.open(OUT + r'\execution.json',
             'w', encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=1,
              ensure_ascii=False)
log('execution.json written (prereg '
    'frozen) forward_free')

# ---------- anchors ----------
c4_ok = (seal96['npz_sha256_8']
         == h8(P3096))
log('c4 3096 npz sha8 match seal.json: '
    '%s (%s)' % (c4_ok,
                 seal96['npz_sha256_8']))
c1_diff = 0.0
c2_diff = 0.0
for (side, zseal, nlf) in (
        ('14B', z14, 40), ('4B', z4_79,
                           36)):
    for key in ('AB', 'AC', 'BC'):
        f2p = z96['F2P_%s_%s' % (side,
                                 key)]
        assert f2p.shape == (24, nlf + 1), \
            (side, key, f2p.shape)
        final = f2p[:, nlf].astype(np.float64)
        sea = zseal['F2_CTT_' + key] \
            .astype(np.float64)
        d = float(np.max(np.abs(final
                                - sea)))
        if side == '14B':
            c1_diff = max(c1_diff, d)
        else:
            c2_diff = max(c2_diff, d)
c1_ok = c1_diff <= 1e-9
c2_ok = c2_diff <= 1e-9
log('c1 14B final vs 3093 sealed '
    'diff=%.3e ok=%s' % (c1_diff, c1_ok))
log('c2 4B final vs 3079 sealed '
    'diff=%.3e ok=%s' % (c2_diff, c2_ok))

# c3 grid medians vs 3096 result curves
c3_diff = 0.0
grid = res96['stats']['curves']
for ck, cv in grid.items():
    key, ci = ck.rsplit('_ci', 1)
    ci = int(ci)
    side = '14B'
    NL = 40 if side == '14B' else 36
    f2p = z96['F2P_%s_%s' % (side, key)]
    m = CIDX == ci
    for gi, d in enumerate(cv['grid']):
        Lq = int(round(d * NL))
        med = float(np.median(
            f2p[m, Lq]))
        c3_diff = max(c3_diff, abs(
            med - cv['m14B'][gi]))
c3_ok = c3_diff <= 1e-12
log('c3 grid medians vs 3096 curves '
    'diff=%.3e ok=%s' % (c3_diff, c3_ok))
assert c1_ok and c2_ok and c3_ok \
    and c4_ok
log('anchors ok')

# ---------- E1 step decomposition ----------
KEYCi = ['%s_ci%d' % (key, ci)
         for key in ('AB', 'AC', 'BC')
         for ci in (1, 2, 3)]


def med_curve(side, key, ci):
    NL = 40 if side == '14B' else 36
    f2p = z96['F2P_%s_%s' % (side, key)]
    m = CIDX == ci
    return np.array([
        float(np.median(f2p[m, L]))
        for L in range(NL + 1)])


E = {'steps': {}, 'max_last4': {},
     'max_prev': {}, 'cliff_at': {}}
for side in ('14B', '4B'):
    NL = 40 if side == '14B' else 36
    last4 = list(range(NL - 4, NL))
    prev = list(range(NL - 10, NL - 4))
    for ck in KEYCi:
        key, ci = ck.rsplit('_ci', 1)
        cur = med_curve(side, key,
                        int(ci))
        steps = np.diff(cur)
        ml4 = float(np.max(
            np.abs(steps[last4])))
        mpv = float(np.max(
            np.abs(steps[prev])))
        E['steps']['%s_%s' % (side, ck)] = \
            [float(x) for x in steps]
        E['max_last4']['%s_%s' % (side,
                                  ck)] = ml4
        E['max_prev']['%s_%s' % (side,
                                 ck)] = mpv
        arg = int(np.argmax(
            np.abs(steps[last4])))
        E['cliff_at']['%s_%s' % (side,
                                 ck)] = \
            last4[arg]
    log('E1 %s last4/prev max |step|: %s'
        % (side, ' '.join(
            '%s %.3f/%.3f@L%d'
            % (ck,
               E['max_last4']['%s_%s'
                              % (side, ck)],
               E['max_prev']['%s_%s'
                             % (side, ck)],
               E['cliff_at']['%s_%s'
                             % (side, ck)])
            for ck in KEYCi)))
log('E1 14B step vectors (L->L+1, '
    'L=30..39):')
for ck in ('AC_ci1', 'BC_ci1',
           'AC_ci2', 'AB_ci2'):
    st = E['steps']['14B_%s' % ck]
    log('  %s: %s' % (ck, ' '.join(
        '%+.3f' % st[L]
        for L in range(30, 40))))

# ---------- decision ----------
def gates_for(side):
    NL = 40 if side == '14B' else 36
    out = {}
    for ci in (1, 2, 3):
        for key in ('AC', 'BC'):
            ck = '%s_ci%d' % (key, ci)
            ml4 = E['max_last4'][
                '%s_%s' % (side, ck)]
            mpv = E['max_prev'][
                '%s_%s' % (side, ck)]
            out['c1_' + ck] = (
                ml4 >= 0.15
                and ml4 >= 2.0 * mpv)
            st = E['steps'][
                '%s_%s' % (side, ck)]
            out['c2_' + ck] = (
                max(abs(x) for x in
                    st[NL - 10:NL]) < 0.15)
    for key in ('AC', 'BC', 'AB'):
        ck = '%s_ci2' % key
        st = E['steps'][
            '%s_%s' % (side, ck)]
        pos = sum(
            1 for x in st[NL - 4:NL]
            if x > 0)
        out['c3_' + ck] = pos >= 3
    return out


g14 = gates_for('14B')
H_C1 = (g14['c1_AC_ci1']
        and g14['c1_BC_ci1'])
H_C2 = all(g14[k] for k in g14
           if k.startswith('c2_'))
H_C3 = (g14['c3_AC_ci2']
        and g14['c3_BC_ci2'])
log('decision: H_C1=%s H_C2=%s H_C3=%s'
    % (H_C1, H_C2, H_C3))
if H_C1:
    VERDICT = 'fifth_cliff_single_step'
elif H_C2:
    VERDICT = 'fifth_cliff_gradual'
elif H_C3:
    VERDICT = 'fifth_cliff_readout_' \
        'amplify'
elif not (H_C1 or H_C2 or H_C3):
    VERDICT = 'fifth_cliff_mixed'
else:
    VERDICT = 'fifth_cliff_inconclusive'
log('VERDICT: %s' % VERDICT)

# ---------- persist ----------
res = {
    'phase': PHASE, 'name': NAME,
    'verdict': VERDICT, 'forwards': 0,
    'elapsed': 0.0,
    'stats': {'E1': E,
              'gates_14B': g14,
              'decision': {
                  'H_C1': bool(H_C1),
                  'H_C2': bool(H_C2),
                  'H_C3': bool(H_C3)}},
    'anchors': {
        'c1_diff': c1_diff,
        'c2_diff': c2_diff,
        'c3_diff': c3_diff,
        'c4_ok': bool(c4_ok),
        'ok': True}}
with io.open(OUT + r'\result.json', 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=1,
              ensure_ascii=False)
save = {
    'VERDICT': np.array(VERDICT),
    'SMOKE': np.bool_(False),
    'PHASE': np.int64(PHASE),
    'C1_DIFF': np.float64(c1_diff),
    'C2_DIFF': np.float64(c2_diff),
    'C3_DIFF': np.float64(c3_diff)}
for side in ('14B', '4B'):
    for ck in KEYCi:
        save['STEPS_%s_%s'
             % (side, ck)] = np.array(
            E['steps']['%s_%s'
                       % (side, ck)])
        save['MAXL4_%s_%s'
             % (side, ck)] = np.float64(
            E['max_last4']['%s_%s'
                           % (side, ck)])
        save['MAXPV_%s_%s'
             % (side, ck)] = np.float64(
            E['max_prev']['%s_%s'
                          % (side, ck)])
        save['CLIFF_%s_%s'
             % (side, ck)] = np.int64(
            E['cliff_at']['%s_%s'
                          % (side, ck)])
npz_path = OUT + r'\%s.npz' % NAME
np.savez(npz_path, **save)
seal = {
    'npz_sha256_8': h8(npz_path),
    'result_sha256_8': h8(
        OUT + r'\result.json'),
    'script_sha256_8': h8(
        os.path.abspath(__file__)),
    'execution_sha256_8': h8(
        OUT + r'\execution.json')}
with io.open(OUT + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('sealed npz8=%s result8=%s'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8']))
with io.open(LOGF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(LOGS) + '\n')
print('RUN_COMPLETE %s' % VERDICT)
