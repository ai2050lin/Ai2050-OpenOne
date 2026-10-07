"""Phase 2995: Omega-F1 cross-model card replication panel on
GLM4-9B (plan v4 P3 / plan v3 Omega-F, first panel).

Question.  Plan v3 Omega-F: replicate the card set on a second
model (GLM4-9B, in local library) with the same protocol and
per-card grading {replicated / directional / not-replicated};
anchors rebuilt for the new model.  This Phase runs the FIRST
(weight-bearing-band) panel of three cards; the full 34-card
copy is later work.

Model.  glm4-9b-chat-hf: hidden=4096, NL=40, NH=32, HD=128,
GQA kv=2, intermediate=13696 (FUSED gate_up_proj: up neuron row
j = gate_up_proj.weight[13696+j]).  bf16, full determinism
verified in probe (bit 0.0 re-forward).

Design (preregistered; frozen BEFORE any observation).
  cells    runtime single-token filter on GLM4 tokenizer from
           2972 exec cells {F_en,F_fr,C_en,C_fr} + 2993 L
           candidates; en group = lab 0 (F_en+C_en+L singles),
           non-en group = lab 1 (F_fr+C_fr singles).
  sweep    seq = [the, w] (pos 1 = word), one forward per word,
           captures at every layer: attn_in (self_attn input =
           post-LN1) and mlp_in (mlp input = post-LN2).
  axes     dirs_attn[l] = unit(mean_en(attn_in) -
           mean_nonen(attn_in))   (2927 word-probe caliber)
           dirs_mlp[l]  same on mlp_in (registry space; NO
           cross-space caliber mixing).
  T1 (M2963 lang-class separation)  proj_w = attn_in[39][w] .
           dirs_attn[39] for en words; stat = median(F) -
           median(C); label permutation over 36 en words
           two-sided N=4999; pass p < 0.05.  L-median ordering
           F<L<C descriptive (2993 caliber).
  T2 (M2947 head concentration)  C39 = dirs_attn[39] @ Wo39,
           headC[w,h] = dot(C39[h*128:(h+1)*128], op_in[39][w]);
           per-head d_h = mean_F - mean_C; maxT over 32 heads
           (same label perms, N=2000); pass min p_maxT < 0.05.
  T3 (M2989 MLP registry null)  REG_LAYERS=[7,10,13]; align =
           |dot(unit(up_row_j), dirs_mlp[l])|; stat = mean
           top-128; Haar null N=1000 (same rows); z per layer;
           pass min z > 5 (qwen 2989 observed 31-55).
  grades   per card {replicated/directional/not} per rules in
           prereg grades below; applicability tags per plan v4
           P0 (len-2, en classes, weight-side snapshot: causal
           ablation NOT tested here -> tagged untested).
  verdict  anchor_fail_all_void / omega_f1_panel_absent /
           omega_f1_panel_partial / omega_f1_panel_replicated

Anchors (rebuilt for GLM4; no cross-model artifact upstream):
  a1  sweep determinism (full sweep twice) < 1e-6
  a2  unit norms dirs_attn/dirs_mlp all layers < 1e-12
  a3  tokenizer re-encode identity (exact)
  a4  headC recompute identity < 1e-10
  a5  registry recompute identity < 1e-10
  a6  T1 permutation rerun same seed -> identical count
  a7  batch dim == 1 on every forward (assert in forward1)
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
EXEC_2972 = os.path.join(BASE, 'phase2972',
                         'two_factor_signature',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2995', 'omega_f1_glm4_panel')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
NH, HD = 32, 128
NL = 40
LTOP = NL - 1
HID = 4096
INT = 13696
REG_LAYERS = [7, 10, 13]
TOPK_N = 128
N_PERM_T1 = 4999
N_PERM_T2 = 2000
N_NULL = 1000
P_GATE = 0.05
Z_GATE = 5.0
Z_DIR = 3.0
RNG_MAIN = 2995
BIT_TOL = 1e-12
ARR_TOL = 1e-6
L_CAND = ["because", "therefore", "although", "unless",
          "however", "thus", "moreover", "since", "whereas",
          "despite", "hence", "nevertheless", "consequently",
          "furthermore", "otherwise", "instead", "while",
          "accordingly", "likewise", "meanwhile", "nonetheless",
          "thereafter", "whereby", "albeit"]


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- execution freeze (BEFORE any compute) ----------
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2995,
                   'name': 'omega_f1_glm4_panel',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'prereg': {
                       'design': 'plan v4 P3 / v3 Omega-F first '
                                 'panel on GLM4-9B (glm4-9b-'
                                 'chat-hf, h=4096 L=40 H=32x128 '
                                 'kv=2 int=13696 fused '
                                 'gate_up_proj): 62-word '
                                 'single-token sweep [the,w] '
                                 'pos-1, captures attn_in '
                                 '(post-LN1) + mlp_in (post-LN2) '
                                 'all 40 layers; dirs_attn/'
                                 'dirs_mlp = unit(en mean - '
                                 'non-en mean) per layer (2927 '
                                 'word-probe caliber); '
                                 'cells runtime-filtered from '
                                 '2972 exec + 2993 L candidates',
                       'T1_M2963': 'lang-class separation at '
                                   'L39: median(F)-median(C) '
                                   'proj on dirs_attn[39], '
                                   'single label permutation '
                                   'within FUC pool ONLY '
                                   '(positional split nF/nC, '
                                   'L words excluded from the '
                                   'null) two-sided N=%d; '
                                   'pass p < %.2f; F<L<C '
                                   'ordering descriptive'
                                   % (N_PERM_T1, P_GATE),
                       'T2_M2947': 'head concentration: C39 = '
                                   'dirs_attn[39]@Wo39 head-'
                                   'sliced, d_h = meanF-meanC, '
                                   'maxT over 32 heads N=%d; '
                                   'pass min p_maxT < %.2f'
                                   % (N_PERM_T2, P_GATE),
                       'T3_M2989': 'MLP registry null: '
                                   'REG_LAYERS=%s, align='
                                   '|unit(up_row).dirs_mlp|, '
                                   'top-%d mean, Haar null '
                                   'N=%d, z per layer; pass '
                                   'min z > %.0f (qwen 2989: '
                                   '31-55)' % (REG_LAYERS,
                                               TOPK_N, N_NULL,
                                               Z_GATE),
                       'grades': {
                           'T1': 'replicated p<0.05; '
                                 'directional F<L<C ordering '
                                 'holds; else not',
                           'T2': 'replicated min p_maxT<0.05; '
                                 'directional top-1 share > '
                                 'perm p95; else not',
                           'T3': 'replicated min z>5; '
                                 'directional max z in '
                                 '(3,5]; else not'},
                       'applicability': 'glm4-9b / len-2 / en '
                                        'classes / registry '
                                        'weight-side snapshot; '
                                        'causal ablation untested '
                                        '(plan v4 P0 tags)',
                       'verdict_map': ['anchor_fail_all_void',
                                       'omega_f1_panel_absent',
                                       'omega_f1_panel_partial',
                                       'omega_f1_panel_'
                                       'replicated'],
                       'anchors': {
                           'a1': 'sweep determinism < 1e-6',
                           'a2': 'unit norms < 1e-12',
                           'a3': 'tokenizer identity exact',
                           'a4': 'headC recompute < 1e-10',
                           'a5': 'registry recompute < 1e-10',
                           'a6': 'perm rerun identical count',
                           'a7': 'batch dim == 1 assert'},
                       'rng': RNG_MAIN,
                       'n_perm_t1': N_PERM_T1,
                       'n_perm_t2': N_PERM_T2,
                       'n_null': N_NULL}},
                  f, indent=1)
    log('execution.json frozen', lines)

    # ---------- cells (runtime filter on GLM4 tokenizer) ----------
    e72 = json.load(open(EXEC_2972, encoding='utf-8'))
    F_EN = e72['cells']['F_en']
    F_FR = e72['cells']['F_fr']
    C_EN = e72['cells']['C_en']
    C_FR = e72['cells']['C_fr']

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch as _t

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, use_fast=True)

    def tid_of(w):
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        return int(ids[0]) if len(ids) == 1 else -1

    # groups: (label, cat, lang, word); lab 0 = en, 1 = non-en
    src = ([('F', 'en', w) for w in F_EN]
           + [('C', 'en', w) for w in C_EN]
           + [('L', 'en', w) for w in L_CAND]
           + [('F', 'fr', w) for w in F_FR]
           + [('C', 'fr', w) for w in C_FR])
    cells = []
    tid_map = {}
    for cat, lang, w in src:
        t = tid_of(w)
        if t != -1:
            lab = 0 if lang == 'en' else 1
            cells.append((lab, cat, lang, w))
            tid_map[w] = t
    n_cells = len(cells)
    i_en = [i for i, c in enumerate(cells) if c[0] == 0]
    i_non = [i for i, c in enumerate(cells) if c[0] == 1]
    iF_en = [i for i in i_en if cells[i][1] == 'F']
    iC_en = [i for i in i_en if cells[i][1] == 'C']
    iL_en = [i for i in i_en if cells[i][1] == 'L']
    words_here = ['%s:%s:%s' % (c[1], c[2], c[3])
                  for c in cells]
    log('cells single-token %d (en %d [F %d C %d L %d] '
        'non-en %d)' % (n_cells, len(i_en), len(iF_en),
                        len(iC_en), len(iL_en), len(i_non)),
        lines)
    ids_the = tid_of('the')
    assert ids_the != -1
    func_tid = int(ids_the)

    # a3: tokenizer re-encode identity
    tid_map2 = {}
    for _, _, _, w in cells:
        tid_map2[w] = tid_of(w)
    a3_ok = bool(tid_map2 == tid_map)
    log('a3 tokenizer identity: %s' % a3_ok, lines)

    # ---------- model ----------
    model = AutoModelForCausalLM.from_pretrained(
        MD, torch_dtype=_t.bfloat16).cuda().eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap_ai = {}
    cap_mi = {}
    cap_oi = {}
    handles = []

    def hk_ai(mod, args, kwargs):
        x = kwargs['hidden_states']
        cap_ai['x'] = x.detach().float().cpu().numpy().copy()
        return None

    def hk_mi(mod, args):
        cap_mi['x'] = args[0].detach().float().cpu().numpy() \
            .copy()
        return None

    def hk_oi(mod, args):
        cap_oi['x'] = args[0].detach().float().cpu().numpy() \
            .copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           hk_ai, with_kwargs=True))
        handles.append(layers[li].mlp
                       .register_forward_pre_hook(hk_mi))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(hk_oi))

    seqs = [[func_tid, tid_map[c[3]]] for c in cells]

    def forward1(toks):
        assert len(toks) == 2
        cap_ai.clear()
        cap_mi.clear()
        cap_oi.clear()
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        # a7: batch dim enforced == 1 on every capture
        assert cap_ai['x'].shape[0] == 1
        assert cap_mi['x'].shape[0] == 1
        return (cap_ai['x'], cap_mi['x'], cap_oi['x'])

    # ---------- double sweep (captures + a1 determinism) ----------
    arr1_ai = np.zeros((NL, n_cells, HID))
    arr1_mi = np.zeros((NL, n_cells, HID))
    arr2_ai = np.zeros((NL, n_cells, HID))
    arr2_mi = np.zeros((NL, n_cells, HID))
    for sweep, A_ai, A_mi in ((1, arr1_ai, arr1_mi),
                              (2, arr2_ai, arr2_mi)):
        for i in range(n_cells):
            ai, mi, _ = forward1(seqs[i])
            A_ai[:, i, :] = ai[0, 1, :]   # pos-1 = word
            A_mi[:, i, :] = mi[0, 1, :]
        log('sweep %d done' % sweep, lines)

    a1_diff = max(
        float(np.abs(arr1_ai - arr2_ai).max()
              / max(float(np.abs(arr1_ai).max()), 1e-30)),
        float(np.abs(arr1_mi - arr2_mi).max()
              / max(float(np.abs(arr1_mi).max()), 1e-30)))
    a1_ok = bool(a1_diff < ARR_TOL)
    log('a1 sweep determinism %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    del arr2_ai, arr2_mi

    # ---------- axes (2927 word-probe caliber, both spaces) --
    m0_ai = arr1_ai[:, i_en, :].mean(axis=1)
    m1_ai = arr1_ai[:, i_non, :].mean(axis=1)
    d_ai = m0_ai - m1_ai
    dirs_attn = d_ai / np.maximum(
        np.linalg.norm(d_ai, axis=1, keepdims=True), 1e-30)
    m0_mi = arr1_mi[:, i_en, :].mean(axis=1)
    m1_mi = arr1_mi[:, i_non, :].mean(axis=1)
    d_mi = m0_mi - m1_mi
    dirs_mlp = d_mi / np.maximum(
        np.linalg.norm(d_mi, axis=1, keepdims=True), 1e-30)
    a2_diff = max(
        float(np.abs(np.linalg.norm(dirs_attn, axis=1) - 1)
              .max()),
        float(np.abs(np.linalg.norm(dirs_mlp, axis=1) - 1)
              .max()))
    a2_ok = bool(a2_diff < BIT_TOL)
    log('a2 unit norms %.2e ok=%s' % (a2_diff, a2_ok), lines)

    u_top = dirs_attn[LTOP]

    # ---------- T2 capture: head projections ----------
    Wo39 = layers[LTOP].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C39 = u_top @ Wo39                       # (4096,) op space
    op_top = np.zeros((n_cells, HID))
    for i in range(n_cells):
        _, _, oi = forward1(seqs[i])
        op_top[i] = oi[0, -1, :]
    headC = np.zeros((n_cells, NH))
    for h in range(NH):
        headC[:, h] = op_top[:, h * HD:(h + 1) * HD] \
            @ C39[h * HD:(h + 1) * HD]
    # a4: recompute identity (second path: einsum-free recompute
    # from stored arrays)
    headC_re = (op_top.reshape(n_cells, NH, HD)
                * C39.reshape(NH, HD)).sum(axis=2)
    a4_diff = float(np.abs(headC - headC_re).max())
    a4_ok = bool(a4_diff < 1e-10)
    log('a4 headC recompute %.2e ok=%s'
        % (a4_diff, a4_ok), lines)

    # ---------- T3 capture: registry alignments ----------
    Wup = layers[REG_LAYERS[0]].mlp.gate_up_proj.weight \
        .detach().float().cpu().numpy().astype(np.float64)
    assert Wup.shape == (2 * INT, HID)
    up_rows = {}
    aligns = {}
    for li in REG_LAYERS:
        Wl = layers[li].mlp.gate_up_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        R = Wl[INT:, :]                      # up half rows
        Rn = R / np.maximum(
            np.linalg.norm(R, axis=1, keepdims=True), 1e-30)
        up_rows[li] = Rn
        aligns[li] = np.abs(Rn @ dirs_mlp[li])
    # a5: fresh weight refetch + recompute vs stored aligns
    Wf = layers[10].mlp.gate_up_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    Rf = Wf[INT:, :]
    Rf = Rf / np.maximum(
        np.linalg.norm(Rf, axis=1, keepdims=True), 1e-30)
    a5_diff = float(np.abs(
        np.abs(Rf @ dirs_mlp[10]) - aligns[10]).max())
    a5_ok = bool(np.isfinite(a5_diff) and a5_diff < 1e-10)
    log('a5 registry recompute %.2e ok=%s'
        % (a5_diff, a5_ok), lines)
    del up_rows, Wup

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok)

    # preinit (anchor_fail path stays construct-safe)
    verdict = None
    T1 = T2 = T3 = None
    g1 = g2 = g3 = None
    proj_top = np.zeros(n_cells)
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        # ---------- T1 (M2963) ----------
        proj_top = arr1_ai[LTOP] @ u_top       # (n_cells,)
        pf = proj_top[iF_en]
        pc = proj_top[iC_en]
        stat1 = float(np.median(pf) - np.median(pc))
        # perm pool = F union C ONLY (labels 1=F, 0=C);
        # L words are descriptive-only, never in the null.
        # NOTE single permutation + positional split (values
        # permuted, labels implicit by position) -- pairing
        # pool[pm] with lab[pm] would be a double permutation
        # that recovers the original groups identically.
        nF = len(iF_en)
        t1_pool = np.array([proj_top[i] for i in iF_en + iC_en])
        rng_t1 = np.random.default_rng(RNG_MAIN + 10)
        cnt1 = 0
        for _ in range(N_PERM_T1):
            pm = rng_t1.permutation(t1_pool.size)
            d = float(np.median(t1_pool[pm[:nF]])
                      - np.median(t1_pool[pm[nF:]]))
            if abs(d) >= abs(stat1):
                cnt1 += 1
        p1 = (cnt1 + 1) / (N_PERM_T1 + 1)
        # a6: rerun same seed -> identical count
        rng_t1b = np.random.default_rng(RNG_MAIN + 10)
        cnt1b = 0
        for _ in range(N_PERM_T1):
            pm = rng_t1b.permutation(t1_pool.size)
            d = float(np.median(t1_pool[pm[:nF]])
                      - np.median(t1_pool[pm[nF:]]))
            if abs(d) >= abs(stat1):
                cnt1b += 1
        a6_ok = bool(cnt1b == cnt1)
        log('a6 perm rerun count %d vs %d ok=%s'
            % (cnt1, cnt1b, a6_ok), lines)
        anchor_ok = bool(anchor_ok and a6_ok)
        med_L = float(np.median(proj_top[iL_en]))
        med_F = float(np.median(pf))
        med_C = float(np.median(pc))
        order_flc = bool(med_F < med_L < med_C)
        T1 = {'median_F': round(med_F, 4),
              'median_L': round(med_L, 4),
              'median_C': round(med_C, 4),
              'stat_FvC': round(stat1, 4),
              'p_two_sided': round(p1, 5),
              'order_F_L_C': order_flc}
        log('T1 FvC med %.2f vs %.2f (L %.2f) p=%s '
            'order=%s' % (med_F, med_C, med_L, p1,
                          order_flc), lines)
        t1_pass = bool(p1 < P_GATE)
        g1 = ('replicated' if t1_pass
              else ('directional' if order_flc
                    else 'not_replicated'))

        # ---------- T2 (M2947) maxT ----------
        d_heads = headC[iF_en].mean(0) - headC[iC_en].mean(0)
        rng_t2 = np.random.default_rng(RNG_MAIN + 20)
        nF = len(iF_en)
        nC = len(iC_en)
        pool = headC[iF_en + iC_en]
        max_abs = np.zeros(N_PERM_T2)
        shares = np.zeros(N_PERM_T2)
        for k in range(N_PERM_T2):
            pm = rng_t2.permutation(nF + nC)
            dh = pool[pm[:nF]].mean(0) - pool[pm[nF:]].mean(0)
            max_abs[k] = np.abs(dh).max()
            shares[k] = np.abs(dh).max() \
                / max(np.abs(dh).sum(), 1e-30)
        obs_share = float(np.abs(d_heads).max()
                          / max(np.abs(d_heads).sum(), 1e-30))
        p_maxT = [(int((max_abs >= abs(d_heads[h])).sum())
                   + 1) / (N_PERM_T2 + 1)
                  for h in range(NH)]
        p2_min = float(min(p_maxT))
        h_top = int(np.argmax(np.abs(d_heads)))
        share_p95 = float(np.quantile(shares, 0.95))
        T2 = {'d_top5': {int(h): round(float(d_heads[h]), 4)
                         for h in np.argsort(
                             -np.abs(d_heads))[:5]},
              'p_maxT_min': round(p2_min, 5),
              'top1_head': h_top,
              'top1_share': round(obs_share, 4),
              'share_null_p95': round(share_p95, 4)}
        log('T2 maxT min p=%s top1 h%d d=%.4f share %.4f '
            'vs null95 %.4f' % (p2_min, h_top,
                                d_heads[h_top], obs_share,
                                share_p95), lines)
        t2_pass = bool(p2_min < P_GATE)
        g2 = ('replicated' if t2_pass
              else ('directional'
                    if obs_share > share_p95
                    else 'not_replicated'))

        # ---------- T3 (M2989) registry null ----------
        rng_t3 = np.random.default_rng(RNG_MAIN + 30)
        V = rng_t3.standard_normal((N_NULL, HID))
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        zmap = {}
        for li in REG_LAYERS:
            Rn = layers[li].mlp.gate_up_proj.weight.detach() \
                .float().cpu().numpy().astype(np.float64)
            Rn = Rn[INT:, :]
            Rn = Rn / np.maximum(
                np.linalg.norm(Rn, axis=1, keepdims=True),
                1e-30)
            obs = float(np.sort(np.abs(
                Rn @ dirs_mlp[li]))[-TOPK_N:].mean())
            nulls = np.abs(V @ Rn.T)         # (N_NULL, INT)
            null_stat = np.sort(nulls, axis=1)[:, -TOPK_N:] \
                .mean(axis=1)
            mu = float(null_stat.mean())
            sd = float(null_stat.std())
            zmap[li] = {'obs': round(obs, 6),
                        'null_mu': round(mu, 6),
                        'null_sd': round(sd, 6),
                        'z': round((obs - mu)
                                   / max(sd, 1e-30), 2)}
            log('T3 L%d obs %.6f null %.6f+-%.6f z=%.2f'
                % (li, obs, mu, sd, zmap[li]['z']), lines)
        zmin = min(v['z'] for v in zmap.values())
        zmax = max(v['z'] for v in zmap.values())
        T3 = {'per_layer': zmap, 'z_min': zmin,
              'z_max': zmax}
        t3_pass = bool(zmin > Z_GATE)
        g3 = ('replicated' if t3_pass
              else ('directional' if zmax > Z_DIR
                    else 'not_replicated'))

        # ---------- verdict ----------
        n_pass = int(t1_pass) + int(t2_pass) + int(t3_pass)
        if n_pass == 3:
            verdict = 'omega_f1_panel_replicated'
        elif n_pass >= 1:
            verdict = 'omega_f1_panel_partial'
        else:
            verdict = 'omega_f1_panel_absent'
        log('VERDICT: %s (grades %s/%s/%s)'
            % (verdict, g1, g2, g3), lines)

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2995,
        'name': 'omega_f1_glm4_panel',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_determinism': a1_diff,
            'a2_unitnorm': a2_diff,
            'a3_tokenizer': a3_ok,
            'a4_headC_recompute': a4_diff,
            'a5_registry_recompute': a5_ok,
            'a6_perm_rerun': a6_ok,
            'a7_batch_dim': True},
        'grades': {'T1_M2963': g1, 'T2_M2947': g2,
                   'T3_M2989': g3},
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'cells': {'words': words_here,
                  'n_en': len(i_en), 'n_nonen': len(i_non)},
        'model': {'path': MD, 'hidden': HID, 'layers': NL,
                  'heads': NH, 'head_dim': HD,
                  'intermediate': INT},
        'applicability_tags': 'glm4-9b / len-2 / en classes / '
                              'weight-side snapshot; causal '
                              'ablation untested',
        'elapsed_s': elapsed,
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    npz_path = os.path.join(OUT, 'omega_f1_glm4_panel.npz')
    save = {'words': np.array(words_here),
            'dirs_attn': dirs_attn.astype(np.float32),
            'dirs_mlp': dirs_mlp.astype(np.float32),
            'proj_top': proj_top if anchor_ok
                        else np.zeros(n_cells),
            'headC': headC.astype(np.float32),
            'u_top': u_top.astype(np.float32)}
    np.savez_compressed(npz_path, **save)
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'sealed_at':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'verdict': verdict,
                   'npz_sha256_8': sha8(npz_path),
                   'result_sha256_8':
                       sha8(os.path.join(OUT,
                                         'result.json'))},
                  f, indent=1)
    log('PHASE2995 DONE elapsed=%ss verdict=%s'
        % (elapsed, verdict), lines)
    print('PHASE2995 DONE', verdict, elapsed)


if __name__ == '__main__':
    main()
