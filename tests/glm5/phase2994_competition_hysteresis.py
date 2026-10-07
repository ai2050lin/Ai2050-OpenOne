"""Phase 2994: Omega-E competition-hysteresis operationalization
(plan v4 P3 / plan v3 Omega-E).

Question.  Plan v3 Omega-E: inject a MISLEAD direction that
competes with the readout, ramp the dose, remove it, measure
recovery; criterion = recovery residual (hysteresis area) > 0
with random-direction control; irreversible dose threshold
existence.  Output = an operationalized criterion for the
"error attractor" language.  Until this Phase is achieved,
"phase transition / potential barrier" naming is banned; even
after, "phase transition" stays banned (no energy functional).

Stateless-forward note (preregistered applicability limit).
A transformer forward has no temporal state, so temporal
dose-removal hysteresis is operationalized as SPATIAL
wash-out: the mislead is injected at context position p of a
16-token sequence (14 fillers + [func, word]) and the word
readout is measured as a function of the mislead-to-word
distance.  Order dependence (early vs late injection at
equal dose) and recovery residual at maximal distance are
the operational hysteresis analogs.  True temporal hysteresis
(KV-cache decoding) is OUT OF SCOPE here and registered as
untested in the applicability tag.

Design (preregistered; frozen BEFORE any observation).
  cells    37 en words (2977 order: F_en 0-14, C_en 30-51),
          L16 cond = 2986/2987 filler protocol verbatim.
  mislead  m = unit(mean res34(C_en) - mean res34(F_en)) at
          base (L34 decoder-layer input residual, 2560,
          d_lang_u caliber); per word the injection axis is
          a_i = s_i * m, s_i = +1 for F (pushed toward C
          pole), -1 for C -- per-word mirror built in.
  dose     delta = g_rel * n17_i * a_i at L17 self_attn
          input, in-place add (2977 protocol), n17_i =
          ||x17(word pos)|| at base; g_rel grid.
  conds    base; ramp@word g in {0.1,0.2,0.4,0.8,1.6};
          position grid @1.6 p in {0,1,4,7,10,13} (distance
          to word 15-p); early@0 @0.4; random-direction
          control (seeded unit vec per word) @word/@0/@13
          all @1.6.  16 conds x 37 words.
  readout  proj = res34 . a_i (primary, signed toward the
          wrong pole); u35 trajectory secondary.
  T1       PRIMARY gate: displacement D_i(g=1.6@word) > 0,
          one-sided sign-permutation over words (N=10000).
  T2 (C1)  recovery fraction R_i = D_i(p=0)/D_i(p=13) > 0,
          one-sided sign-permutation (floor: D(p=13) >
          1e-9*n17_i, skips registered).
  T3 (C2)  median R > median R_rand (two-sample sign
          permutation, one-sided).
  S1       secondary: dose persistence R(1.6) > R(0.4)
          paired one-sided; washout area_rel descriptive.
  verdict  anchor_fail_all_void / competition_floor_void /
           hysteresis_absent / error_attractor_
           operationalized

Anchors:
  a1   Vt8 rebuild vs 2939 < 1e-6
  a1w  words identity 2986/2987/here (exact, composite fmt)
  a2   base prof_all vs 2986 L16 bin bit < 1e-12 (37 en)
  a3   base headC vs 2987 L16N bit < 1e-12 (37 en)
  a4   determinism < 1e-4 (base + injected @word 1.6)
  a5   injection locality: op[16] base vs injected bit 0
  a6   injection efficacy guard ~ bf16 ULP of delta
  a7   single-token base74
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2986 = os.path.join(BASE, 'phase2986',
                        'length_context_drift',
                        'length_context_drift.npz')
SRC_2987 = os.path.join(BASE, 'phase2987',
                        'context_minimal_audit',
                        'context_minimal_audit.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2994',
                   'competition_hysteresis')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L34 = 34
L_INJ = 17
SEQ_LEN = 16            # 14 fillers + [func, word]
WORD_POS = 15
FUNC_POS = 14
POS_GRID = [0, 1, 4, 7, 10, 13]
G_RAMP = [0.1, 0.2, 0.4, 0.8, 1.6]
G_MAIN = 1.6
G_LOW = 0.4
FILLER_TEXT = (' The sun rises in the east and sets in the '
               'west .')
N_PERM = 10000
P_GATE = 0.05
R_FLOOR_REL = 1e-9
RNG_MAIN = 2994
BIT_TOL = 1e-12


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def sign_perm_p(vals, rng, n_perm, one_sided=True):
    v = np.asarray(vals, dtype=np.float64)
    stat = float(np.mean(v))
    cnt = 0
    for _ in range(n_perm):
        pm = rng.choice([-1.0, 1.0], size=v.size)
        if one_sided:
            if float(np.mean(v * pm)) >= stat:
                cnt += 1
        else:
            if abs(float(np.mean(v * pm))) >= abs(stat):
                cnt += 1
    # observed itself included via >= on identity perm
    return (cnt + 1) / (n_perm + 1)


def two_samp_p(a, b, rng, n_perm, one_sided=True):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pool = np.concatenate([a, b])
    na = a.size
    stat = float(a.mean() - b.mean())
    cnt = 0
    for _ in range(n_perm):
        pm = rng.permutation(pool.size)
        d = pool[pm[:na]].mean() - pool[pm[na:]].mean()
        if one_sided:
            if d >= stat:
                cnt += 1
        else:
            if abs(d) >= abs(stat):
                cnt += 1
    return (cnt + 1) / (n_perm + 1)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- execution freeze (BEFORE any compute) ----------
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2994,
                   'name': 'competition_hysteresis',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'prereg': {
                       'design': 'plan v4 P3 / v3 Omega-E: '
                                 '37 en words x L16 (2986/2987 '
                                 'filler verbatim); mislead m = '
                                 'unit(C_mean - F_mean) of base '
                                 'res34, per-word mirror axis '
                                 'a_i = s_i*m (s=+1 F, -1 C); '
                                 'injection delta = g_rel*n17*a '
                                 'at L17 self_attn input in-place '
                                 '(2977 protocol); conds: base, '
                                 'ramp@word %s, position grid %s '
                                 '@1.6 (spatial washout = '
                                 'dose-removal operationalization;'
                                 ' temporal KV-cache hysteresis '
                                 'OUT OF SCOPE, tagged untested),'
                                 ' early@0.4, random-direction '
                                 'control @word/@0/@13 @1.6; '
                                 'readout res34.a_i signed toward '
                                 'wrong pole + u35 secondary'
                                 % (G_RAMP, POS_GRID),
                       'T1_primary': 'D_i(g=1.6@word) > 0 '
                                     'one-sided sign-perm '
                                     'N=%d; p >= %.2f -> '
                                     'competition_floor_void'
                                     % (N_PERM, P_GATE),
                       'T2_C1': 'R_i = D(p=0)/D(p=13) > 0 '
                                'one-sided (floor D13 > '
                                '%.0e*n17)' % R_FLOOR_REL,
                       'T3_C2': 'median R > median R_rand '
                                'two-sample one-sided',
                       'verdict_map': ['anchor_fail_all_void',
                                       'competition_floor_void',
                                       'hysteresis_absent',
                                       'error_attractor_'
                                       'operationalized'],
                       'naming_rule': 'error-attractor '
                                      'operational criterion '
                                      'licensed iff verdict = '
                                      'error_attractor_; '
                                      '"phase transition" stays '
                                      'banned (no energy '
                                      'functional)',
                       'anchors': {
                           'a1': 'Vt8 rebuild < 1e-6',
                           'a1w': 'words identity 2986/2987/'
                                  'here exact',
                           'a2': 'base prof vs 2986 L16 bit',
                           'a3': 'base headC vs 2987 L16N bit',
                           'a4': 'determinism < 1e-4',
                           'a5': 'locality op16 bit 0',
                           'a6': 'injection guard ~ bf16 ULP',
                           'a7': 'single-token base74'},
                       'rng': RNG_MAIN,
                       'n_perm': N_PERM,
                       'g_ramp': G_RAMP,
                       'g_main': G_MAIN,
                       'pos_grid': POS_GRID,
                       'seq_len': SEQ_LEN}},
                  f, indent=1)
    log('execution.json frozen', lines)

    # ---------- cells (2977 exec verbatim) ----------
    e77 = json.load(open(EXEC_2977, encoding='utf-8'))
    F_EN = e77['cells']['F_en']
    F_FR = e77['cells']['F_fr']
    C_EN = e77['cells']['C_en']
    C_FR = e77['cells']['C_fr']
    assert (len(F_EN), len(F_FR), len(C_EN), len(C_FR)) \
        == (15, 15, 22, 22), 'cell size drift'
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_EN] \
        + [('C', 'fr', w) for w in C_FR]
    n_base = len(cells)
    iF_en = list(range(0, 15))
    iC_en = list(range(30, 52))
    i_en = iF_en + iC_en
    n_en = len(i_en)                    # 37
    s_en = np.array([1.0] * 15 + [-1.0] * 22)
    words_here_base = ['%s:%s:%s' % c for c in cells]

    # ---------- upstream artifacts ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z86 = np.load(SRC_2986, allow_pickle=True)
    z87 = np.load(SRC_2987, allow_pickle=True)
    words86 = [str(w) for w in z86['words']]
    words87 = [str(w) for w in z87['words']]
    a1w_ok = bool(words86 == words87 == words_here_base)
    log('a1w words identity 2986/2987/here: %s'
        % a1w_ok, lines)
    prof86 = z86['prof_all'].astype(np.float64)
    headC87 = z87['headC'].astype(np.float64)

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer
    import torch as _t

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)

    def tid_of(w):
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        return int(ids[0]) if len(ids) == 1 else -1

    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        t = tid_of(w)
        tid_map[w] = t
        if t != -1:
            n_single += 1
    a7_ok = bool(n_single == n_base)
    log('a7 single-token %d/%d ok=%s'
        % (n_single, n_base, a7_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])
    fill_pool = tok(FILLER_TEXT,
                    add_special_tokens=False)['input_ids']
    assert len(fill_pool) >= 1
    fill = [int(fill_pool[k % len(fill_pool)])
            for k in range(SEQ_LEN - 2)]
    assert len(fill) == SEQ_LEN - 2

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (fill_pool %d tok)'
        % len(fill_pool), lines)

    cap_op = {li: [] for li in range(NL)}
    cap_res34 = []
    cap_x17p = []
    inj_state = {'delta': None, 'pos': None, 'guard': None}
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_inj(module, args, kwargs):
        d = inj_state['delta']
        if d is None:
            return None
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        p = inj_state['pos']
        base_row = inj_state['guard']  # base row or None
        dt = _t.as_tensor(d, device=x.device,
                          dtype=x.dtype)
        x[:, p, :] += dt
        if base_row is not None:
            got = x[:, p, :].detach().float().cpu() \
                .numpy().astype(np.float64).reshape(-1)
            br = base_row.astype(np.float64).reshape(-1)
            d64 = np.asarray(d, dtype=np.float64) \
                .reshape(-1)
            inj_state['guard'] = ('measured', got, br, d64)
        return None

    def hook_res34(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_res34.append(
            x[:, -1, :].detach().float().cpu().numpy())
        return None

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, -1, :].detach().float().cpu().numpy())
            return None
        return h

    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_inj, with_kwargs=True))
    handles.append(
        layers[L34].register_forward_pre_hook(
            hook_res34, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]
        del cap_res34[:]

    def forward1(toks, delta=None, pos=None, guard_row=None):
        clear_cap()
        inj_state['delta'] = delta
        inj_state['pos'] = pos
        inj_state['guard'] = guard_row
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        g = inj_state['guard']
        inj_state['delta'] = None
        inj_state['pos'] = None
        inj_state['guard'] = None
        guard_out = None
        if isinstance(g, tuple) and g[0] == 'measured':
            _, got, br, d64 = g
            # bf16 rounding of the add is bounded by the ULP
            # of the operands: normalize by max(|br|max,
            # |d|max), NOT |d|max (run2: |d|max underestimates
            # when delta is small vs base row -> 9.2e-2
            # artifact; 2992 same lesson)
            denom = max(float(np.abs(br).max()),
                        float(np.abs(d64).max()), 1e-30)
            guard_out = float(np.abs(
                (got - br) - d64).max() / denom)
        return ({li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)},
                cap_res34[0].astype(np.float64)
                .reshape(-1),
                guard_out)

    # weight-space readout rows (2986/2987 verbatim)
    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo

    # ---------- base sweep (37 en words, no injection) ----------
    en_words = [cells[i][2] for i in i_en]
    seqs = {}
    for i in i_en:
        w = cells[i][2]
        seqs[i] = fill + [func_tid, tid_map[w]]
        assert len(seqs[i]) == SEQ_LEN
    prof_base = np.zeros((n_en, NL))
    res_base = np.zeros((n_en, 2560))
    headC_base = np.zeros((n_en, NH))
    x17_base = np.zeros((n_en, SEQ_LEN, 2560))
    Wo34 = layers[L34].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C34 = u35 @ Wo34
    for k, i in enumerate(i_en):
        ops, res, _ = forward1(seqs[i])
        for li in range(NL):
            prof_base[k, li] = float(np.dot(
                ops[li].reshape(-1), M[li]))
        res_base[k] = res
        x34 = ops[L34].reshape(-1)
        for h in range(NH):
            headC_base[k, h] = float(np.dot(
                C34[h * HD:(h + 1) * HD],
                x34[h * HD:(h + 1) * HD]))
    log('base sweep done (%d words)' % n_en, lines)

    # per-position x17 base rows (for injection guard):
    # capture L17 self_attn input at a chosen position via
    # the guard hook with zero delta is not possible; use a
    # dedicated capture hook pass
    cap_full = []

    def hook_cap17(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_full.append(
            x[0].detach().float().cpu().numpy())
        return None

    hcap = layers[L_INJ].self_attn \
        .register_forward_pre_hook(hook_cap17,
                                   with_kwargs=True)
    for k, i in enumerate(i_en):
        del cap_full[:]
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([seqs[i]], device='cuda'))
        x17_base[k] = cap_full[0]
    hcap.remove()
    n17 = np.linalg.norm(x17_base[:, WORD_POS, :], axis=1)
    log('x17 base rows captured (n17 median %.2f)'
        % float(np.median(n17)), lines)

    # ---------- mislead direction ----------
    # en-array coords: F = 0..14, C = 15..36 (i_en order)
    m = res_base[15:37].mean(0) - res_base[0:15].mean(0)
    m = m / max(float(np.linalg.norm(m)), 1e-30)
    rng_rand = np.random.default_rng(RNG_MAIN + 100)
    rand_dirs = {}
    for k in range(n_en):
        v = rng_rand.standard_normal(2560)
        rand_dirs[k] = v / float(np.linalg.norm(v))

    # ---------- conditions sweep ----------
    # cond list: (name, kind, pos, g, use_rand)
    conds = [('base', None, None, 0.0, False)]
    for g in G_RAMP:
        conds.append(('ramp%g' % g, 'mis', WORD_POS, g,
                      False))
    for p in POS_GRID:
        conds.append(('pos%d' % p, 'mis', p, G_MAIN, False))
    conds.append(('pos0_low', 'mis', 0, G_LOW, False))
    conds.append(('rand_word', 'rand', WORD_POS, G_MAIN,
                  True))
    conds.append(('rand_pos0', 'rand', 0, G_MAIN, True))
    conds.append(('rand_pos13', 'rand', 13, G_MAIN, True))
    proj = {cn: np.zeros(n_en) for cn, *_ in conds}
    proj_u = {cn: np.zeros(n_en) for cn, *_ in conds}
    guards = {}
    for cn, kind, p, g, use_rand in conds:
        if cn == 'base':
            for k in range(n_en):
                proj[cn][k] = float(
                    res_base[k] @ (s_en[k] * m))
                proj_u[cn][k] = float(res_base[k] @ u35)
            continue
        for k, i in enumerate(i_en):
            a = (rand_dirs[k] if use_rand
                 else s_en[k] * m)
            d = float(g) * float(n17[k]) * a
            grow = x17_base[k, p] if not use_rand else None
            _, res, gd = forward1(seqs[i], delta=d,
                                  pos=p, guard_row=grow)
            proj[cn][k] = float(res @ a)
            proj_u[cn][k] = float(res @ u35)
            if gd is not None:
                guards['%s_w%d' % (cn, k)] = gd
        log('cond %s done' % cn, lines)

    # ---------- anchors ----------
    op_a = forward1(seqs[i_en[0]])[0]
    op_b = forward1(seqs[i_en[0]])[0]
    a4a = float(np.abs(op_a[30] - op_b[30]).max()
                / max(float(np.abs(op_a[30]).max()),
                      1e-30))
    k0 = 0
    i0 = i_en[k0]
    a_vec0 = s_en[k0] * m
    d0 = G_MAIN * float(n17[k0]) * a_vec0
    op_c = forward1(seqs[i0], delta=d0, pos=WORD_POS)[0]
    op_d = forward1(seqs[i0], delta=d0, pos=WORD_POS)[0]
    a4b = float(np.abs(op_c[30] - op_d[30]).max()
                / max(float(np.abs(op_c[30]).max()),
                      1e-30))
    a4_ok = bool(a4a < 1e-4 and a4b < 1e-4)
    log('a4 determinism %.2e / %.2e ok=%s'
        % (a4a, a4b, a4_ok), lines)

    # a2/a3: base vs 2986/2987 (bit level, en rows)
    a2_diff = float(np.abs(
        prof_base - prof86[1][i_en]).max())
    a3_diff = float(np.abs(
        headC_base - headC87[2][i_en]).max())
    a2_ok = bool(a2_diff < BIT_TOL)
    a3_ok = bool(a3_diff < BIT_TOL)
    log('a2 prof base vs 2986 L16: %.2e ok=%s; '
        'a3 headC vs 2987: %.2e ok=%s'
        % (a2_diff, a2_ok, a3_diff, a3_ok), lines)

    # a5: locality -- layers < L_INJ identical under injection
    op_e = forward1(seqs[i0])[0]
    a5_diff = max(float(np.abs(op_e[li] - op_c[li]).max())
                  for li in range(L_INJ))
    a5_ok = bool(a5_diff < BIT_TOL)
    log('a5 locality layers<%d: %.2e ok=%s'
        % (L_INJ, a5_diff, a5_ok), lines)

    # a6: injection efficacy guard
    gvals = list(guards.values())
    a6_max = max(gvals) if gvals else -1.0
    a6_med = float(np.median(gvals)) if gvals else -1.0
    a6_ok = bool(gvals and a6_max < 0.01)
    log('a6 injection guard med %.2e max %.2e (n=%d) ok=%s'
        % (a6_med, a6_max, len(gvals), a6_ok), lines)

    anchor_ok = bool(a1_ok and a1w_ok and a2_ok and a3_ok
                     and a4_ok and a5_ok and a6_ok and a7_ok)

    # preinit (anchor_fail path stays construct-safe)
    verdict = None
    T1 = T2 = T3 = S1 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        # ---------- T1 competition gate ----------
        rng_t1 = np.random.default_rng(RNG_MAIN + 10)
        D_word = proj['ramp1.6'] - proj['base']
        p1 = sign_perm_p(D_word, rng_t1, N_PERM,
                         one_sided=True)
        n_pos1 = int(np.sum(D_word > 0))
        T1 = {'mean_D': round(float(D_word.mean()), 4),
              'median_D': round(float(np.median(D_word)),
                                4),
              'n_positive': n_pos1,
              'n_words': n_en,
              'p_one_sided': round(p1, 5)}
        log('T1 D@word g=1.6: mean %.4f median %.4f '
            'pos %d/%d p=%s'
            % (D_word.mean(), np.median(D_word), n_pos1,
               n_en, p1), lines)
        t1_pass = bool(p1 < P_GATE)

        # ---------- T2 recovery fraction ----------
        D13 = proj['pos13'] - proj['base']
        D0 = proj['pos0'] - proj['base']
        floor = R_FLOOR_REL * n17
        okmask = D13 > floor
        n_skip = int((~okmask).sum())
        R = (D0[okmask] / D13[okmask])
        rng_t2 = np.random.default_rng(RNG_MAIN + 20)
        p2 = sign_perm_p(R, rng_t2, N_PERM,
                         one_sided=True)
        T2 = {'n_used': int(okmask.sum()),
              'n_skip_floor': n_skip,
              'median_R': round(float(np.median(R)), 4),
              'p_one_sided': round(p2, 5)}
        log('T2 R recovery fraction: n=%d median %.4f '
            'p=%s' % (okmask.sum(), np.median(R), p2),
            lines)
        c1_pass = bool(p2 < P_GATE and np.median(R) > 0)

        # ---------- T3 vs random control ----------
        Rrand13 = proj['rand_pos13'] - proj['base']
        Rrand0 = proj['rand_pos0'] - proj['base']
        ok_r = Rrand13 > floor
        R_rand = Rrand0[ok_r] / Rrand13[ok_r]
        rng_t3 = np.random.default_rng(RNG_MAIN + 30)
        p3 = two_samp_p(R, R_rand, rng_t3, N_PERM,
                        one_sided=True)
        T3 = {'n_rand_used': int(ok_r.sum()),
              'median_R_rand':
                  round(float(np.median(R_rand)), 4),
              'p_one_sided': round(p3, 5)}
        log('T3 R vs rand: median %.4f vs %.4f p=%s'
            % (np.median(R), np.median(R_rand), p3), lines)
        c2_pass = bool(np.median(R) > np.median(R_rand)
                       and p3 < P_GATE)

        # ---------- S1 secondary ----------
        D0low = proj['pos0_low'] - proj['base']
        okl = np.logical_and(D13 > floor, D0low != 0)
        R_low = D0low[okl] / D13[okl]
        # paired: same words in both
        common = np.logical_and(okmask, okl)
        Rc1 = (proj['pos0'] - proj['base'])[common] \
            / D13[common]
        Rc2 = (proj['pos0_low'] - proj['base'])[common] \
            / D13[common]
        rng_s1 = np.random.default_rng(RNG_MAIN + 40)
        p_s1 = sign_perm_p(Rc1 - Rc2, rng_s1, N_PERM,
                           one_sided=True)
        # washout area (descriptive): mean over words of
        # summed |delta(pos)| over the grid, normalized by
        # the population-mean |delta(pos13)|
        area = float(np.mean([
            float(np.abs(proj['pos%d' % p]
                         - proj['base']).sum())
            for p in POS_GRID])
            / max(abs(float((proj['pos13']
                             - proj['base']).mean())),
                  1e-30))
        S1 = {'n_paired': int(common.sum()),
              'median_R_1.6': round(float(np.median(Rc1)),
                                    4),
              'median_R_0.4': round(float(np.median(Rc2)),
                                    4),
              'p_dose_persistence': round(p_s1, 5),
              'washout_area_rel': round(area, 4)}
        log('S1 dose persistence: R 1.6 %.4f vs 0.4 %.4f '
            'p=%s; area_rel %.4f'
            % (np.median(Rc1), np.median(Rc2), p_s1,
               area), lines)

        # ---------- verdict ----------
        if not t1_pass:
            verdict = 'competition_floor_void'
        elif c1_pass and c2_pass:
            verdict = 'error_attractor_operationalized'
        else:
            verdict = 'hysteresis_absent'
        log('VERDICT: %s' % verdict, lines)

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2994,
        'name': 'competition_hysteresis',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_Vt8': a1_ok,
            'a1w_words': a1w_ok,
            'a2_prof_diff': a2_diff,
            'a3_headC_diff': a3_diff,
            'a4_determinism': a4_ok,
            'a5_locality_diff': a5_diff,
            'a6_guard_max': a6_max,
            'a7_single_token': a7_ok},
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'S1': S1,
        'elapsed_s': elapsed,
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    npz_path = os.path.join(
        OUT, 'competition_hysteresis.npz')
    save = {'words': np.array(words_here_base),
            'i_en': np.array(i_en),
            's_en': s_en,
            'm': m,
            'n17': n17,
            'res_base': res_base.astype(np.float32)}
    for cn, *_ in conds:
        save['proj_%s' % cn] = proj[cn]
        save['proju_%s' % cn] = proj_u[cn]
    np.savez_compressed(npz_path, **save)
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'sealed_at':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'verdict': verdict,
                   'npz_sha256_8':
                       sha8(npz_path),
                   'result_sha256_8':
                       sha8(os.path.join(OUT,
                                         'result.json'))},
                  f, indent=1)
    log('PHASE2994 DONE elapsed=%ss verdict=%s'
        % (elapsed, verdict), lines)
    print('PHASE2994 DONE', verdict, elapsed)


if __name__ == '__main__':
    main()
