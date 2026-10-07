"""Phase 2991: carrier migration atlas (plan v4 P1b).

Question. 2987 showed the L34 F/C-gap carrier is protocol-
relative: h15 carries the gap at len-2 but dies under context
while h11/h23 become the significant carriers at L16N.  The
NATURE of this migration is undecided:
  H_smooth    contribution spectrum is continuously
              reweighted (same distributed structure, new
              argmax); new carriers were already secondary
              at L2.
  H_rearrange head identity discretely reorganizes
              (new carriers were noise-level at L2;
              spectrum decorrelates beyond label null).

Design.
  Offline   2987 npz headC (5 conds x 74 x 32) gives the full
            per-condition head atlas with NO new forwards;
            T1 full 32-head perm census on all 5 conditions
            (2987 perm_p verbatim); T2 spectrum similarity
            L2 vs L16N (cos + Spearman on 32-dim contrast
            vectors) with paired label-permutation null;
            rank-shift of h11/h23 in the L2 spectrum.
  Rerun     74 cells x 2 conds (L2, L16N) forwards with 2987
            op/x17 hooks (bit-level headC anchors) AND 2989
            down_proj hooks (MLP act at 9 registry layers);
            T3 neuron-side condition contrast: per-layer
            registry stability (top-128 from 2989 archived
            delta vs L16N recomputed) + axis spectrum cos.
  T4 synthesis: head-level vs neuron-level spectrum change.

Verdict branches (frozen, primary = T2):
  anchor_fail_all_void
  degenerate_void (spectrum norms ~ 0)
  cos12 >= 0.8 and p_cos < .05   -> carrier_smooth_reweighting
  cos12 < 0.8 and p_jacc >= .05  -> carrier_identity_rearranged
  else                           -> carrier_partial_recombination

Anchors:
  a1 words identity 2987/2989/here (exact)
  a2 rerun L2 headC vs 2987 headC[0] rel < 1e-9
  a3 rerun L16N headC vs 2987 headC[2] rel < 1e-9
  a4 rerun L2 act vs 2989 act_l rel < 1e-9 (9 layers)
  a5 offline h15c recompute vs 2987 stored (round4 tol)
     + full-head L16N contrast recompute vs fresh perm
  a6 d34 recompute vs 2987 stored (round4 tol)
  a7 determinism rel < 1e-4 (L2, L16N)
  a8 2989 c_lang/c_cls rebuild identity (exact, <1e-12)
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
SRC_2973 = os.path.join(BASE, 'phase2973', 'fr_scale_audit',
                        'fr_scale_audit.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2987 = os.path.join(BASE, 'phase2987',
                        'context_minimal_audit',
                        'context_minimal_audit.npz')
SRC_2989 = os.path.join(BASE, 'phase2989',
                        'mlp_neuron_registry',
                        'mlp_neuron_registry.npz')
OUT = os.path.join(BASE, 'phase2991', 'carrier_migration_atlas')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L34 = 34
REG_LAYERS = [6, 7, 8, 9, 10, 11, 12, 17, 34]
N_PERM = 10000        # T1 head census (2987 verbatim count)
N_PERM_NULL = 2000    # T2/T3 paired null
K_TOP = 128
RNG_MAIN = 2991
P_GATE = 0.01         # carrier significance (2987 gate)
COS_GATE = 0.8        # T2 branch gate (frozen)
ROUND_TOL = 5.1e-5    # compare vs round(.,4) stored values
BIT_TOL = 1e-9
FILLER_NEUTRAL = (' The sun rises in the east and sets in '
                  'the west .')


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def median(a):
    return float(np.median(a))


def perm_p(v, labels, rng, n_perm=N_PERM):
    obs = median(v[labels == 1]) - median(v[labels == 0])
    cnt = 0
    for _ in range(n_perm):
        pl = rng.permutation(labels)
        pv = median(v[pl == 1]) - median(v[pl == 0])
        if abs(pv) >= abs(obs):
            cnt += 1
    return float(obs), float(cnt) / float(n_perm)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- execution freeze (BEFORE any compute) ----------
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2991,
                   'name': 'carrier_migration_atlas',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'prereg': {
                       'design': 'offline 2987 headC '
                                 '(5x74x32) full-head atlas; '
                                 'rerun 74 cells x {L2,L16N} '
                                 'with op/x17/down_proj hooks; '
                                 'T1 32-head perm census all 5 '
                                 'conds (perm_p verbatim, '
                                 'N_PERM=%d); T2 spectrum cos12 '
                                 'L2 vs L16N + paired label '
                                 'null (N=%d) + rank shift; '
                                 'T3 neuron delta (2989 '
                                 'caliber) condition contrast '
                                 '+ top-128 registry overlap '
                                 'vs random; T4 synthesis'
                                 % (N_PERM, N_PERM_NULL),
                       'T2_primary': 'cos12 >= %.2f and '
                                     'p_cos<0.05 -> '
                                     'carrier_smooth_'
                                     'reweighting; cos12 < '
                                     'gate and p_cos>=0.05 '
                                     '-> carrier_identity_'
                                     'rearranged; else '
                                     'carrier_partial_'
                                     'recombination'
                                     % COS_GATE,
                       'anchors': {
                           'a1': 'Vt8 rebuild < 1e-6',
                           'a1w': 'words identity '
                                  '2987/2989/here exact',
                           'a2': 'rerun L2 headC vs 2987 '
                                 'rel < 1e-9',
                           'a3': 'rerun L16N headC vs 2987 '
                                 'rel < 1e-9',
                           'a4': 'rerun L2 act vs 2989 rel '
                                 '< 1e-9 (9 layers)',
                           'a5': 'h15 contrast recompute vs '
                                 '2987 round4 tol (p not '
                                 'gated: MC noise)',
                           'a6': 'd34 recompute vs 2987 '
                                 'round4 tol',
                           'a7': 'determinism rel < 1e-4',
                           'a8': 'c_lang/c_cls rebuild '
                                 'exact < 1e-12'},
                       'rng': RNG_MAIN,
                       'p_gate': P_GATE,
                       'cos_gate': COS_GATE,
                       'k_top': K_TOP}},
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
    n_test = len(cells)
    lang = np.array([0 if c[1] == 'en' else 1 for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1 for c in cells])

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
    z79 = np.load(SRC_2979, allow_pickle=True)
    d79_l = z79['d_lang_u'].astype(np.float64)
    d79_c = z79['d_cls_u'].astype(np.float64)
    z87 = np.load(SRC_2987, allow_pickle=True)
    z89 = np.load(SRC_2989, allow_pickle=True)
    words87 = [str(w) for w in z87['words']]
    words89 = [str(w) for w in z89['words']]
    words_here = ['%s:%s:%s' % c for c in cells]

    # a1w words identity three-way (2989 'F:en:he' caliber)
    a1w_ok = bool(words87 == words89 == words_here)
    log('a1w words identity 2987/2989/here: %s'
        % a1w_ok, lines)

    headC87 = z87['headC'].astype(np.float64)
    h15c87 = z87['h15c'].astype(np.float64)
    p_h15_87 = z87['p_h15'].astype(np.float64)
    d34_87 = z87['d34'].astype(np.float64)
    act_l89 = {li: z89['act_%d' % li].astype(np.float64)
               for li in REG_LAYERS}
    c_lang89 = {li: z89['c_lang_%d' % li].astype(np.float64)
                for li in REG_LAYERS}
    c_cls89 = {li: z89['c_cls_%d' % li].astype(np.float64)
               for li in REG_LAYERS}
    delta_l89 = {li: z89['delta_lang_%d' % li]
                 .astype(np.float64) for li in REG_LAYERS}
    delta_c89 = {li: z89['delta_cls_%d' % li]
                 .astype(np.float64) for li in REG_LAYERS}

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
    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    assert n_single == n_test, 'single-token check'
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])
    neutral_pool = tok(FILLER_NEUTRAL,
                       add_special_tokens=False)['input_ids']
    n14 = [int(neutral_pool[k % len(neutral_pool)])
           for k in range(14)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_act = {li: [] for li in REG_LAYERS}
    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_x17(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_x17.append(
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

    def hook_down(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            # word sits at last position in both conds
            # (L2 [func,w]: -1 == pos 1, 2989-identical)
            cap_act[li].append(
                x[:, -1, :].detach().float().cpu().numpy())
            return None
        return h

    handles.append(
        layers[17].self_attn
        .register_forward_pre_hook(
            hook_x17, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
        if li in REG_LAYERS:
            handles.append(
                layers[li].mlp.down_proj
                .register_forward_pre_hook(
                    hook_down(li), with_kwargs=True))

    def clear_cap():
        for li in cap_act:
            del cap_act[li][:]
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]

    def forward1(toks):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        return ({li: cap_act[li][0].astype(np.float64)
                 for li in REG_LAYERS},
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    # weight-space readout rows (2989/2987 verbatim)
    M = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(M[li]))
    Wo34 = layers[L34].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C34 = u35 @ Wo34

    # registry axes rebuild (a8 identity gate)
    a8_diff = 0.0
    Wd = {}
    c_lang = {}
    c_cls = {}
    for li in REG_LAYERS:
        Wdm = layers[li].mlp.down_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wd[li] = Wdm.T
        c_lang[li] = Wd[li] @ d79_l
        c_cls[li] = Wd[li] @ d79_c
        a8_diff = max(a8_diff,
                      float(np.abs(c_lang[li]
                                   - c_lang89[li]).max()),
                      float(np.abs(c_cls[li]
                                   - c_cls89[li]).max()))
    a8_ok = bool(a8_diff < 1e-12)
    log('a8 c_lang/c_cls rebuild vs 2989: %.2e ok=%s'
        % (a8_diff, a8_ok), lines)

    # ---------- rerun sweep: 2 conds ----------
    CONDS = [('L2', []), ('L16N', n14)]
    n_cond = 2
    headC = np.zeros((n_cond, n_test, NH))
    prof34 = np.zeros((n_cond, n_test))
    act = {li: {ci: np.zeros((n_test, 9728))
                for ci in range(n_cond)}
           for li in REG_LAYERS}
    for ci, (cname, fill) in enumerate(CONDS):
        for i, (_, _, w) in enumerate(cells):
            seq = list(fill) + [func_tid, tid_map[w]]
            assert len(seq) == len(fill) + 2
            acts, ops = forward1(seq)
            for li in REG_LAYERS:
                act[li][ci][i] = acts[li].reshape(-1)
            x34 = ops[L34].reshape(-1)
            for h in range(NH):
                headC[ci, i, h] = float(np.dot(
                    C34[h * HD:(h + 1) * HD],
                    x34[h * HD:(h + 1) * HD]))
            prof34[ci, i] = float(np.dot(x34, M[L34]))
        log('cond %s done' % cname, lines)

    # ---------- anchors ----------
    op_a = forward1([func_tid, tid_map['man']])[1]
    op_b = forward1([func_tid, tid_map['man']])[1]
    a7a = float(np.abs(op_a[30] - op_b[30]).max()
                / max(float(np.abs(op_a[30]).max()), 1e-30))
    op_c = forward1(list(n14)
                    + [func_tid, tid_map['man']])[1]
    op_d = forward1(list(n14)
                    + [func_tid, tid_map['man']])[1]
    a7b = float(np.abs(op_c[30] - op_d[30]).max()
                / max(float(np.abs(op_c[30]).max()), 1e-30))
    a7_ok = bool(a7a < 1e-4 and a7b < 1e-4)
    log('a7 determinism %.2e / %.2e ok=%s'
        % (a7a, a7b, a7_ok), lines)

    a2_rel = float(np.max(np.abs(headC[0] - headC87[0])
                          / np.maximum(np.abs(headC87[0])
                                       .max(), 1e-30)))
    a2_ok = bool(a2_rel < BIT_TOL)
    log('a2 L2 headC vs 2987: %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    a3_rel = float(np.max(np.abs(headC[1] - headC87[2])
                          / np.maximum(np.abs(headC87[2])
                                       .max(), 1e-30)))
    a3_ok = bool(a3_rel < BIT_TOL)
    log('a3 L16N headC vs 2987: %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    a4_rel = 0.0
    for li in REG_LAYERS:
        a4_rel = max(a4_rel, float(np.max(
            np.abs(act[li][0] - act_l89[li])
            / max(float(np.abs(act_l89[li]).max()), 1e-30))))
    a4_ok = bool(a4_rel < BIT_TOL)
    log('a4 L2 act vs 2989 (9 layers): %.2e ok=%s'
        % (a4_rel, a4_ok), lines)

    # a5 offline h15 contrast recompute (2987 round4 tol);
    # p-values carry Monte-Carlo noise -> reported, not gated
    rng5 = np.random.default_rng(RNG_MAIN + 1)
    h15c_re = np.zeros(5)
    p_h15_re = np.zeros(5)
    for ci in range(5):
        h15c_re[ci], p_h15_re[ci] = perm_p(
            headC87[ci, :, 15], cls, rng5)
    h15_diff = float(np.max(np.abs(h15c_re - h15c87)))
    p15_diff = float(np.max(np.abs(p_h15_re - p_h15_87)))
    a5_ok = bool(h15_diff < ROUND_TOL)
    log('a5 h15 contrast recompute vs 2987: %.2e '
        '(p MC diff %.4f, not gated) ok=%s'
        % (h15_diff, p15_diff, a5_ok), lines)

    # a6 d34 recompute
    d34_re = np.array([float(np.median(np.abs(
        z87['prof34'][ci].astype(np.float64)
        - z87['prof34'][0].astype(np.float64))))
        for ci in range(5)])
    a6_diff = float(np.max(np.abs(d34_re - d34_87)))
    a6_ok = bool(a6_diff < ROUND_TOL)
    log('a6 d34 recompute vs 2987: %.2e ok=%s'
        % (a6_diff, a6_ok), lines)

    anchor_ok = bool(a1_ok and a1w_ok and a2_ok and a3_ok
                     and a4_ok and a5_ok and a6_ok and a7_ok
                     and a8_ok)

    verdict = None
    T1 = T2 = T3 = T4 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        # ---------- T1 full head atlas, 5 conds ----------
        rng_t1b = np.random.default_rng(RNG_MAIN + 2)
        atlas = {}
        for ci, cname in enumerate(['L2', 'L3', 'L16N',
                                    'L16R', 'L16T']):
            cs = np.zeros(NH)
            ps = np.zeros(NH)
            for h in range(NH):
                cs[h], ps[h] = perm_p(
                    headC87[ci, :, h], cls, rng_t1b)
            sig = [int(h) for h in range(NH)
                   if ps[h] < P_GATE]
            atlas[cname] = {'contrast': cs, 'p': ps,
                            'sig_heads': sig}
        jac = {}
        names = list(atlas)
        for i in range(5):
            for j in range(i + 1, 5):
                si = set(atlas[names[i]]['sig_heads'])
                sj = set(atlas[names[j]]['sig_heads'])
                u = len(si | sj)
                jac['%s-%s' % (names[i], names[j])] = \
                    round(len(si & sj) / u, 4) if u else 0.0
        h15_curve = [round(float(atlas[n]['contrast'][15]),
                           4) for n in names]
        top1 = {n: int(np.argmax(np.abs(
            atlas[n]['contrast']))) for n in names}
        T1 = {'sig_heads': {n: atlas[n]['sig_heads']
                            for n in names},
              'jaccard': jac,
              'h15_contrast_curve': h15_curve,
              'top1_head': top1}
        log('T1 atlas: sig=%s top1=%s h15=%s'
            % ({n: atlas[n]['sig_heads'] for n in names},
               top1, h15_curve), lines)

        # ---------- T2 migration nature (primary) ----------
        c2 = atlas['L2']['contrast']
        c16 = atlas['L16N']['contrast']
        n2 = float(np.linalg.norm(c2))
        n16 = float(np.linalg.norm(c16))
        cos12 = float(c2 @ c16
                      / max(n2 * n16, 1e-30))
        rk2 = np.argsort(np.argsort(-np.abs(c2)))
        rk16 = np.argsort(np.argsort(-np.abs(c16)))
        rho12 = float(np.corrcoef(rk2, rk16)[0, 1])
        # paired label-permutation null for cos12
        rng_n = np.random.default_rng(RNG_MAIN + 3)
        cos_null = np.zeros(N_PERM_NULL)
        for b in range(N_PERM_NULL):
            pl = rng_n.permutation(cls)
            cc2 = np.zeros(NH)
            cc16 = np.zeros(NH)
            for h in range(NH):
                v = headC87[0, :, h]
                cc2[h] = median(v[pl == 1]) \
                    - median(v[pl == 0])
                v = headC87[2, :, h]
                cc16[h] = median(v[pl == 1]) \
                    - median(v[pl == 0])
            nn2 = float(np.linalg.norm(cc2))
            nn16 = float(np.linalg.norm(cc16))
            cos_null[b] = abs(cc2 @ cc16
                              / max(nn2 * nn16, 1e-30))
        p_cos = float((np.sum(
            np.abs(cos_null) >= abs(cos12)) + 1)
            / (N_PERM_NULL + 1))
        # jaccard(L2, L16N) vs carrier-count null
        s2 = set(atlas['L2']['sig_heads'])
        s16 = set(atlas['L16N']['sig_heads'])
        u2 = len(s2 | s16)
        jac_obs = (len(s2 & s16) / u2) if u2 else 0.0
        # rank of h11/h23 at L2
        rank11 = int(rk2[11])
        rank23 = int(rk2[23])
        rank15_16 = int(rk16[15])
        T2 = {'cos12': round(cos12, 4),
              'rho_rank12': round(rho12, 4),
              'p_cos_null': round(p_cos, 4),
              'jaccard_L2_L16N': round(jac_obs, 4),
              'rank_at_L2': {'h11': rank11, 'h23': rank23},
              'rank_of_h15_at_L16N': rank15_16,
              'sig_overlap': sorted(s2 & s16)}
        log('T2 migration: cos=%.4f rho=%.4f p_cos=%.4f '
            'jac=%.4f rank11/23@L2=%d/%d rank15@16=%d'
            % (cos12, rho12, p_cos, jac_obs, rank11,
               rank23, rank15_16), lines)

        # ---------- T3 neuron-side condition contrast ----------
        # delta caliber = 2989 verbatim:
        # D = proj[lab==0].mean(0) - proj[lab==1].mean(0)
        reg_stab = {}
        for li in REG_LAYERS:
            entry = {}
            for ax, cc, lab in (('lang', c_lang[li], lang),
                                ('cls', c_cls[li], cls)):
                a2_ = act[li][0]
                a16 = act[li][1]
                p2 = a2_ * cc[None, :]
                p16 = a16 * cc[None, :]
                D2 = (p2[lab == 0].mean(0)
                      - p2[lab == 1].mean(0))
                D16 = (p16[lab == 0].mean(0)
                       - p16[lab == 1].mean(0))
                cos_c = float(
                    D2 @ D16
                    / max(float(np.linalg.norm(D2))
                          * float(np.linalg.norm(D16)),
                          1e-30))
                # top-128 registry from 2989 archived delta
                d89 = (delta_l89[li] if ax == 'lang'
                       else delta_c89[li])
                top89 = set(np.argsort(
                    -np.abs(d89))[:K_TOP].tolist())
                top16 = set(np.argsort(
                    -np.abs(D16))[:K_TOP].tolist())
                ov = len(top89 & top16)
                rng_z = np.random.default_rng(
                    RNG_MAIN + 100 + li)
                nul = np.zeros(200)
                for b in range(200):
                    rz = set(rng_z.choice(
                        9728, size=K_TOP,
                        replace=False).tolist())
                    nul[b] = len(rz & top16)
                z = ((ov - float(nul.mean()))
                     / max(float(nul.std()), 1e-30))
                entry[ax] = {
                    'cos_D_L2_vs_L16N': round(cos_c, 4),
                    'overlap_top128_2989_vs_16N': ov,
                    'z_vs_random': round(float(z), 2)}
            reg_stab[li] = entry
        T3 = reg_stab
        log('T3 neuron registry stability: %s'
            % json.dumps(reg_stab), lines)

        # ---------- T4 synthesis (descriptive) ----------
        head_cos = abs(cos12)
        neur_cos = float(np.mean(
            [reg_stab[li]['lang']['cos_D_L2_vs_L16N']
             for li in REG_LAYERS]))
        T4 = {'head_spectrum_cos': round(head_cos, 4),
              'neuron_lang_mean_cos':
                  round(neur_cos, 4),
              'note': 'descriptive synthesis only'}
        log('T4 synthesis: head %.4f vs neuron %.4f'
            % (head_cos, neur_cos), lines)

        # ---------- verdict (primary T2) ----------
        if n2 < 1e-12 or n16 < 1e-12:
            verdict = 'degenerate_void'
            log('DEGENERATE spectrum -> void', lines)
        elif cos12 >= COS_GATE and p_cos < 0.05:
            verdict = 'carrier_smooth_reweighting'
        elif cos12 < COS_GATE and p_cos >= 0.05:
            verdict = 'carrier_identity_rearranged'
        else:
            verdict = 'carrier_partial_recombination'
        log('VERDICT: %s' % verdict, lines)

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2991,
        'name': 'carrier_migration_atlas',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_Vt8': a1_ok,
            'a1w_words': a1w_ok,
            'a2_headC_L2_rel': a2_rel,
            'a3_headC_L16N_rel': a3_rel,
            'a4_act_vs2989_rel': a4_rel,
            'a5_h15_recompute': a5_ok,
            'a6_d34_recompute': a6_ok,
            'a7_determinism': a7_ok,
            'a8_axis_rebuild': a8_ok},
        'T1': ({n: {'sig_heads': atlas[n]['sig_heads'],
                    'contrast_top5': {
                        int(h): round(float(
                            atlas[n]['contrast'][h]), 4)
                        for h in np.argsort(
                            -np.abs(atlas[n]['contrast'])
                        )[:5]}} for n in atlas}
               if anchor_ok else None),
        'T2': T2,
        'T3': T3,
        'T4': T4,
        'elapsed_s': elapsed,
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    npz_path = os.path.join(
        OUT, 'carrier_migration_atlas.npz')
    save = {'words': np.array(words_here),
            'lang': lang, 'cls': cls,
            'headC_rerun_L2': headC[0],
            'headC_rerun_L16N': headC[1],
            'prof34_rerun_L2': prof34[0],
            'prof34_rerun_L16N': prof34[1]}
    for li in REG_LAYERS:
        save['act_L2_%d' % li] = act[li][0]
        save['act_L16N_%d' % li] = act[li][1]
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
    log('PHASE2991 DONE elapsed=%ss verdict=%s'
        % (elapsed, verdict), lines)
    print('PHASE2991 DONE', verdict, elapsed)


if __name__ == '__main__':
    main()
