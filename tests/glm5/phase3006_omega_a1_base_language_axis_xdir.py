# -*- coding: utf-8 -*-
"""Phase 3006: Omega-A1 Qwen3-4B-Base language axis +
xdir response + operator structure (plan v5 P1, first
of the alignment trio).

Why: 3002/3003/3004 established the chat-model mid-band
as an xdir-specific, dose-linear, sign-asymmetric SHARED
LOW-RANK redirect+amplification operator; 3005 showed the
GLM4 eraser band is per-cell scatter.  Open question
(plan v5 P1): is this machinery ALIGNED-TRAINING-CARVED
or present in the Base model?  The user-context flagged
that qwen3-4b is itself an instruct/chat model, so the
23.2x chat-vs-chat immunity gap cannot be attributed to
alignment state - a same-family Base comparison is the
only clean probe.

Design (3004 machine verbatim, all-Base-internal geometry:
NO chat-side numeric anchors exist for Base, so dirs /
Vt8 / dcks / xdir are rebuilt from the Base model in this
session; tokenizer tids are family-shared and checked
against the 2887 chat keys):
  pass1 x2 dirs_base rebuild (57 words 2887, func+word
     len-2 seqs, pos-1 attn_in, en-minus-non-en mean,
     unit) - a5 rebuild determinism < 1e-6.
  Vt8_base = SVD(dirs_base)[:8]; u35b = dirs_base[35].
  null0 tids (SEED_NULL=2896 verbatim -> same tids as
     chat); dcks_base = c8_n0 - c8_f0 (in-session);
     xdir_base = dcks_S @ Vt8_S, S_IDX=(0,1,4) verbatim.
  DOSE scan: xdir_base at L4/L15/L17, s in (1,2,3),
     K=2 - raw ratio profile (med ||dS|| / med ||dcks_S||)
     + sub-random controls at s=2 (2 in span(Vt8_base)
     per-cell orthogonal to xdir, 1 full-space).
  T4 capture: xdir_base s=2 at L4 and L17; per-cell
     per-layer Delta stored in full; CORRECT per-cell
     profiles only (axis-bug-free machine per 3004).
  A operator structure per layer: SVD r1/erank/medcos;
     row-gauss null (descriptive).
  E energy account: aligned_frac / amp_ratio medians.

Verdict (frozen, TWO-dimensional; band = L17-inj layers
18..35, degeneracy floor FIRST):
  anchor fail                  => anchor_fail_all_void
  med trk_B(band) < AMP_FLOOR  => base_collapsed_no_operator
  operator: med r1 >= 0.5 AND med medcos >= 0.6 => shared
            med r1 < 0.2 OR med medcos < 0.3 => scatter
            else                          => mixed
  function: med amp_ratio(band) >= 1.2          => amplifier
            med amp_ratio(band) < 0.5           => eraser
            else                                => neutral
  final = base_{shared|scatter|mixed}_{amplifier|eraser|neutral}

Anchors (frozen):
  a0 words == 2887 re-export (57)
  a1 Base tokenizer tids == 2887 en keys (family tokenizer)
  a2 baseline determinism < 1e-4
  a3 xdir identity < 1e-9
  a4 null0 collision-free (same tids as chat expected)
  a5 dirs_base rebuild determinism < 1e-6 (two passes)
  a6 same-session determinism < 1e-6 (all K-repeat arms)
  a7 chat protocol mirror (descriptive): Base sep_f vs
     chat 2935 s_base[func] reported, no gate

Tags: Omega-A1 / lang axis / len-2 / snapshot machine (no
ablation) / dimensionless gates / operator structure /
energy account / axis-bug-free capture / plan v5 P1.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887',
                        'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2935 = os.path.join(BASE, 'phase2935',
                        'null_amp_anatomy',
                        'null_amp_anatomy.npz')
OUT = os.path.join(BASE, 'phase3006',
                   'omega_a1_base_language_axis_xdir')
MD_B = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b-base'
NL, HID, VOCAB = 36, 2560, 151936
INJ_LAYERS = (4, 15, 17)
DOSE_S = (1.0, 2.0, 3.0)
TRK_LAYERS = (4, 17)
BAND = tuple(range(18, 36))
S_IDX = (0, 1, 4)
K_SCAN = 2
SEED_NULL = 2896          # 3002/3004 verbatim
SEED_RND = 3006
AMP_FLOOR = 0.05
R1_SHARED = 0.5
COS_SHARED = 0.6
R1_SCATTER = 0.2
COS_SCATTER = 0.3
AMP_HI = 1.2
AMP_LO = 0.5
N_SUB_RND = 2
N_FULL_RND = 1

PREREG = {
    'mode': 'qwen3-4b-BASE only; 3004 machine verbatim '
            'with all-Base-internal geometry (dirs/Vt8/'
            'dcks/xdir rebuilt in-session from the Base '
            'model; no chat-side numeric anchors exist '
            'for Base); tokenizer tids family-checked '
            'vs 2887 en keys; axis-bug-free per-cell '
            'capture',
    'question': 'is the chat qwen mid-band machinery '
                '(xdir-specific redirect/amplification '
                'operator) present in the same-family '
                'BASE model, or carved by alignment '
                'training?',
    'base_internal_note':
        '2887/2927/2935/2939/2945 anchors are all '
        'chat-side artifacts; for Base they serve only '
        'as protocol mirrors (words, tids, seeds, gates) '
        '- every geometric quantity is rebuilt from '
        'Base weights in this session',
    'dose': 'xdir_base at L4/L15/L17, s in (1,2,3), '
            'K=2; raw ratio = med ||dS|| / med ||dcks_S||;'
            ' sub-random (2 span(Vt8b) per-cell orth to '
            'xdir) + full-random controls at s=2',
    'T4': 'xdir_base s=2 at L4/L17; per-cell pos-1 '
          'attn_in at EVERY layer; Dl_i(l) stored in '
          'full; CORRECTED per-cell profiles only',
    'A': 'per layer l > L_inj: SVD of the per-cell '
         'Delta matrix (57 x 2560); r1, erank, medcos; '
         'row-gauss null descriptive',
    'E': 'per layer: aligned_frac = (Dl.uh_src)^2/'
         '||Dl||^2; amp_ratio = ||Dl||/||src||; medians',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'band = L17-inj layers 18..35; floor '
               'first: med trk_B < 0.05 => '
               'base_collapsed_no_operator; else two-'
               'dimensional: operator shared (r1>=0.5 '
               'AND medcos>=0.6) / scatter (r1<0.2 OR '
               'medcos<0.3) / mixed; function amplifier '
               '(med amp_ratio>=1.2) / eraser (<0.5) / '
               'neutral; final = base_{op}_{fn}',
    'tags': 'Omega-A1 / lang axis / len-2 / snapshot '
            'machine (no ablation) / dimensionless '
            'gates / operator structure / energy '
            'account / axis-bug-free capture / '
            'plan v5 P1',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'),
                              msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def op_structure(D):
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    e = S ** 2
    tot = float(e.sum())
    r1 = float(e[0]) / max(tot, 1e-30)
    p = e / max(tot, 1e-30)
    p = p[p > 0]
    erank = float(np.exp(-float((p * np.log(p)).sum())))
    nrm = np.linalg.norm(D, axis=1)
    cos = np.abs(D @ Vt[0]) / np.maximum(nrm, 1e-30)
    medcos = float(np.median(cos))
    return r1, erank, medcos, Vt[0]


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 3006,
                   'name':
                       'omega_a1_base_language_axis_xdir',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2887': sha8(SRC_2887),
                       's2935': sha8(SRC_2935)},
                   'model': 'qwen3-4b-base',
                   'model_path': MD_B,
                   'n_layers': NL, 'hidden': HID,
                   'vocab': VOCAB,
                   'inj_layers': list(INJ_LAYERS),
                   'dose_s': list(DOSE_S),
                   'trk_layers': list(TRK_LAYERS),
                   'band': list(BAND),
                   's_idx': list(S_IDX),
                   'k_scan': K_SCAN,
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'amp_floor': AMP_FLOOR,
                   'r1_shared': R1_SHARED,
                   'cos_shared': COS_SHARED,
                   'r1_scatter': R1_SCATTER,
                   'cos_scatter': COS_SCATTER,
                   'amp_hi': AMP_HI, 'amp_lo': AMP_LO,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    a0_ok = bool(n_words == 57)
    log('a0 words == 2887 re-export: %s (n=%d)'
        % (a0_ok, n_words), lines)
    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')

    # ---------- model (BASE) ----------
    import torch
    from transformers import AutoTokenizer, \
        AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MD_B, local_files_only=True, use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)[
                    'input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    mism = []
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            if tid_map[w] != int(ck):
                mism.append(w)
    a1_ok = bool(not mism)
    log('a1 Base tids == 2887 en keys: %s (mism=%s)'
        % (a1_ok, mism[:5]), lines)

    func_tid = tid('the')
    word_tids = set(tid_map.values())
    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a4_ok = bool(len(null0_tids) == n_words
                 and not (set(null0_tids) & word_tids))
    log('a4 null0 collision-free: %s' % a4_ok, lines)

    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i],
                        tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model = AutoModelForCausalLM.from_pretrained(
        MD_B, torch_dtype=torch.bfloat16).cuda().eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded (BASE)', lines)

    cap = {'ai': {}, 'trk': {}}
    state_fin = {'on': False}
    state_trk = {'on': False}
    fin_cap = {}
    inj = {'on': False, 'scale': 0.0, 'vec': None,
           'layer': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = kwargs.get('hidden_states')
            if x is None:
                x = args[0] if args else None
            if x is None or x.dim() < 2:
                return None
            ret = None
            if inj['on'] and inj['layer'] is not None \
                    and li == inj['layer']:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                ret = (args, nkw)
            if state_trk['on']:
                cap['trk'].setdefault(li, []).append(
                    x[:, 1, :].detach().float().cpu()
                    .numpy().copy())
            elif not inj['on']:
                cap['ai'].setdefault(li, []).append(
                    x.detach().float().cpu().numpy()
                    .copy())
            return ret
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy().copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]
        for li in cap['trk']:
            del cap['trk'][li][:]

    def forward_batch(toks_list, scale=0.0, layer=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    def forward_trk(toks_list, scale=0.0, layer=None):
        clear_cap()
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        state_trk['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_trk['on'] = False
        return {li: np.stack(cap['trk'][li]).astype(
            np.float64) for li in cap['trk']}

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

    # ---------- pass1 x2: dirs_base rebuild ----------
    def rebuild_dirs(tag):
        store = {}
        for i, (_, _, w) in enumerate(words):
            attnin_all = forward1(
                [func_tid, tid_map[w]])
            for li in range(NL):
                store[(i, li)] = \
                    attnin_all[li].astype(np.float32)
            if (i + 1) % 20 == 0:
                log('pass1-%s [%d/%d]'
                    % (tag, i + 1, n_words), lines)
        d_w = np.zeros((NL, HID))
        for li in range(NL):
            X = np.stack([store[(i, li)][0, 1]
                          for i in range(n_words)]) \
                .astype(np.float64)
            d_w[li] = X[lab_lang == 0].mean(0) \
                - X[lab_lang == 1].mean(0)
        return np.stack([unit(d_w[li])
                         for li in range(NL)])

    dirs_b1 = rebuild_dirs('a')
    dirs_b2 = rebuild_dirs('b')
    a5_diff = float(np.abs(dirs_b1 - dirs_b2).max())
    a5_ok = bool(a5_diff < 1e-6)
    dirs_base = dirs_b1
    log('a5 dirs_base rebuild determinism %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_base,
                             full_matrices=False)
    Vt8b = Vt[:8]
    u35b = dirs_base[NL - 1]

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)
    fin_n0 = forward_batch(batch['null0'])

    def reads(fin):
        return fin @ u35b, fin @ Vt8b.T

    proj_f0, c8_f0 = reads(fin_f1)
    proj_n0, c8_n0 = reads(fin_n0)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    sb35 = s_base_35[ifu35]
    sep_f_chat = float(sb35[lab_lang == 0].mean()
                       - sb35[lab_lang == 1].mean())
    a7_report = {
        'sep_f_base': round(sep_f, 4),
        'sep_null0_base': round(sep_n, 4),
        'sep_f_chat_2935': round(sep_f_chat, 4),
        'note': 'protocol mirror, no gate; chat sep '
                'recomputed from the per-word s_base '
                'with the same lab split'}
    log('a7 sep_f_base=%.3f null0=%.3f (chat 2935 '
        'sep_f=%.3f) ratio_n0f=%.3f'
        % (sep_f, sep_n, sep_f_chat,
           sep_n / max(sep_f, 1e-30)), lines)

    dcks = c8_n0 - c8_f0
    dcks_S = dcks[:, list(S_IDX)]
    Vt8_S = Vt8b[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a3_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a3_ok = bool(a3_diff < 1e-9)
    log('a3 xdir identity %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S,
                                            axis=1)))
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d med_dS=%.1f'
        % (xdir.shape[0], med_dS), lines)

    def arm(vec_t, scale, layer, k, tag, spreads):
        projs = []
        ratios = []
        inj['vec'] = vec_t
        for _ in range(k):
            fin = forward_batch(batch['func'],
                                scale=scale, layer=layer)
            p, c8 = reads(fin)
            projs.append(p)
            cs = c8[:, list(S_IDX)] \
                - c8_f0[:, list(S_IDX)]
            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        inj['vec'] = xdir_t
        P = np.stack(projs)
        spreads[tag] = float('%.2e' % float(
            np.abs(P - P.mean(0)).max()))
        p_med = np.median(P, axis=0)
        sep_m = float(p_med[lab_lang == 0].mean()
                      - p_med[lab_lang == 1].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a1_ok and a4_ok
                         and a5_ok and a2_ok and a3_ok)
    verdict = None
    spreads = {}
    a6_diff = None
    save = {}
    A = {}
    E = {}
    dose = {}
    trk_profB = {}

    if anchor_prelim:
        # ---------- DOSE scan ----------
        rng = np.random.default_rng(SEED_RND)
        xc = (Vt8b @ xdir.T).T
        xh = xc / np.linalg.norm(
            xc, axis=1, keepdims=True)
        rnd_sub = []
        for _ in range(N_SUB_RND):
            g = rng.standard_normal((n_words, 8))
            g = g - np.sum(g * xh, axis=1,
                           keepdims=True) * xh
            g = g / np.linalg.norm(g, axis=1,
                                   keepdims=True)
            rnd_sub.append(g @ Vt8b)
        rnd_full = [unit(rng.standard_normal(HID))]
        for li in INJ_LAYERS:
            per_s = {}
            for s in DOSE_S:
                _, sep_x, ratio_x = arm(
                    xdir_t, s, li, K_SCAN,
                    'xdir|%d|s%g' % (li, s), spreads)
                per_s['s%g' % s] = {
                    'ratio': round(ratio_x, 4),
                    'sep': round(sep_x, 2)}
            dose['L%d' % li] = per_s
            log('DOSE L%d ratios %s'
                % (li, {k: v['ratio']
                        for k, v in per_s.items()}),
                lines)
        ctl = {}
        for li in (15, 17):
            for j, v in enumerate(rnd_sub):
                vt = torch.tensor(
                    v, device='cuda',
                    dtype=torch.bfloat16)
                _, _, ratio_r = arm(
                    vt, 2.0, li, K_SCAN,
                    'subrnd%d|%d' % (j, li), spreads)
                ctl['subrnd%d_L%d' % (j, li)] = round(
                    ratio_r, 4)
            for j, v in enumerate(rnd_full):
                vt = torch.tensor(
                    v, device='cuda',
                    dtype=torch.bfloat16)
                _, _, ratio_r = arm(
                    vt, 2.0, li, K_SCAN,
                    'fullrnd%d|%d' % (j, li), spreads)
                ctl['fullrnd%d_L%d' % (j, li)] = round(
                    ratio_r, 4)
        dose['controls_s2'] = ctl
        log('controls s=2: %s' % json.dumps(ctl), lines)

        # ---------- T4 capture (corrected only) ----------
        base_trk = forward_trk(batch['func'])
        Dl_store = {}
        for li_inj in TRK_LAYERS:
            inj_trk = forward_trk(batch['func'],
                                  scale=2.0,
                                  layer=li_inj)
            layers_list = list(range(li_inj, NL))
            Dstack = np.stack(
                [(inj_trk[li] - base_trk[li])[0]
                 for li in layers_list])
            Dl_store['L%d' % li_inj] = Dstack
            save['Dl_L%d' % li_inj] = Dstack
            profB = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                nrB = float(np.median(
                    np.linalg.norm(Dl, axis=1))) / 2.0
                pjB = float(np.median(
                    np.sum(Dl * xdir, axis=1) / 2.0))
                profB[li] = {
                    'trk_ratio': round(nrB, 4),
                    'xdir_proj': round(pjB, 4)}
            trk_profB['L%d' % li_inj] = profB
            log('T4 L%d inj captured %d layers x %d '
                'cells (corrected per-cell)'
                % (li_inj, len(layers_list),
                   n_words), lines)
        log('T3_base key points: L4inj trk %.1f@4 -> '
            '%.2f@5 -> %.2f@35; L17inj trk %.1f@17 -> '
            '%.2f@18 -> %.2f@35; proj L17inj %.0f@17 '
            '-> %.0f@35'
            % (trk_profB['L4'][4]['trk_ratio'],
               trk_profB['L4'][5]['trk_ratio'],
               trk_profB['L4'][35]['trk_ratio'],
               trk_profB['L17'][17]['trk_ratio'],
               trk_profB['L17'][18]['trk_ratio'],
               trk_profB['L17'][35]['trk_ratio'],
               trk_profB['L17'][17]['xdir_proj'],
               trk_profB['L17'][35]['xdir_proj']),
            lines)

        # ---------- A operator structure ----------
        rngn = np.random.default_rng(SEED_NULL)
        Gn = rngn.standard_normal((n_words, HID))
        Gn = Gn / np.linalg.norm(
            Gn, axis=1, keepdims=True)
        r1_null, _, _, _ = op_structure(Gn)
        A['null_r1_rowgauss'] = round(r1_null, 6)
        A['per_inj'] = {}
        for key in ('L4', 'L17'):
            li_inj = int(key[1:])
            layers_list = list(range(li_inj, NL))
            Dstack = Dl_store[key]
            per = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                r1, erank, medcos, v1 = op_structure(Dl)
                mean_dir = unit(Dl.mean(0))
                cosv1mean = float(abs(v1 @ mean_dir))
                per[li] = {
                    'r1': round(r1, 6),
                    'erank': round(erank, 4),
                    'medcos': round(medcos, 6),
                    'cos_v1_meandelta':
                        round(cosv1mean, 6)}
            A['per_inj'][key] = per
            band = [per[li]['r1']
                    for li in range(li_inj + 1, NL)]
            bandc = [per[li]['medcos']
                     for li in range(li_inj + 1, NL)]
            log('A %s: r1 l=%d..%d med=%.4f '
                'medcos med=%.4f (null r1=%.2e)'
                % (key, li_inj + 1, NL - 1,
                   float(np.median(band)),
                   float(np.median(bandc)),
                   r1_null), lines)

        # ---------- E energy account ----------
        for key in ('L4', 'L17'):
            li_inj = int(key[1:])
            layers_list = list(range(li_inj, NL))
            Dstack = Dl_store[key]
            src = Dstack[0]
            uh = src / np.maximum(
                np.linalg.norm(src, axis=1,
                               keepdims=True), 1e-30)
            src_n = np.linalg.norm(src, axis=1)
            prof = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                nrm = np.linalg.norm(Dl, axis=1)
                aligned = np.sum(Dl * uh, axis=1) ** 2
                frac = aligned / np.maximum(
                    nrm ** 2, 1e-30)
                amp = nrm / np.maximum(src_n, 1e-30)
                prof[li] = {
                    'aligned_frac': round(
                        float(np.median(frac)), 6),
                    'amp_ratio': round(
                        float(np.median(amp)), 6)}
            E[key] = prof
            log('E %s: aligned_frac@%d=%.4f '
                'amp_ratio@%d=%.4f'
                % (key, li_inj + 1,
                   prof[li_inj + 1]['aligned_frac'],
                   li_inj + 1,
                   prof[li_inj + 1]['amp_ratio']),
                lines)

        # ---------- verdict (2-D) ----------
        band_r1 = [A['per_inj']['L17'][li]['r1']
                   for li in BAND]
        band_cos = [A['per_inj']['L17'][li]['medcos']
                    for li in BAND]
        band_trkB = [trk_profB['L17'][li]['trk_ratio']
                     for li in BAND]
        band_amp = [E['L17'][li]['amp_ratio']
                    for li in BAND]
        med_r1 = float(np.median(band_r1))
        med_cos = float(np.median(band_cos))
        med_trkB = float(np.median(band_trkB))
        med_amp = float(np.median(band_amp))
        A['L17_band'] = {
            'layers': list(BAND),
            'med_r1': round(med_r1, 6),
            'med_medcos': round(med_cos, 6),
            'med_trk_B': round(med_trkB, 6),
            'med_amp_ratio': round(med_amp, 6),
            'amp_floor': AMP_FLOOR,
            'r1_shared_gate': R1_SHARED,
            'cos_shared_gate': COS_SHARED,
            'r1_scatter_gate': R1_SCATTER,
            'cos_scatter_gate': COS_SCATTER,
            'amp_hi': AMP_HI, 'amp_lo': AMP_LO}
        a6_diff = max(spreads.values())
        a6_ok = bool(a6_diff < 1e-6)
        log('a6 same-session determinism %.2e ok=%s'
            % (a6_diff, a6_ok), lines)
        log('band medians: trk_B=%.4f r1=%.4f '
            'medcos=%.4f amp=%.4f'
            % (med_trkB, med_r1, med_cos, med_amp),
            lines)
        if not a6_ok:
            verdict = 'anchor_fail_all_void'
        elif med_trkB < AMP_FLOOR:
            verdict = 'base_collapsed_no_operator'
        else:
            if (med_r1 >= R1_SHARED
                    and med_cos >= COS_SHARED):
                op = 'shared'
            elif (med_r1 < R1_SCATTER
                    or med_cos < COS_SCATTER):
                op = 'scatter'
            else:
                op = 'mixed'
            if med_amp >= AMP_HI:
                fn = 'amplifier'
            elif med_amp < AMP_LO:
                fn = 'eraser'
            else:
                fn = 'neutral'
            verdict = 'base_%s_%s' % (op, fn)
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words': a0_ok, 'a1_tids': a1_ok,
        'a4_collision': a4_ok, 'a5_diff': a5_diff,
        'a2_rel': a2_rel, 'a3_diff': a3_diff,
        'a6_diff': a6_diff,
        'a6_ok': a6_ok if anchor_prelim else None,
    }
    res = {
        'phase': 3006,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a1_ok and a4_ok and a5_ok
            and a2_ok and a3_ok and a6_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'a7_chat_mirror': a7_report,
        'dose': dose,
        'A': A, 'E': E,
        'T3_base': trk_profB,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at the a7 chat mirror - the '
            '2935 s_base is a per-word (57,) array and '
            'the scalar conversion failed; fixed by '
            'recomputing the chat sep from the per-word '
            's_base with the same lab split (same '
            'quantity definition as sep_f); run2: full '
            'pass, verdict recorded; correction_note '
            'retro-registered then rerun as run3 for a '
            'self-contained result.json; run3: '
            'authoritative, no defects',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save['dirs_base'] = dirs_base
    save['Vt8b'] = Vt8b
    save['u35b'] = u35b
    save['xdir'] = xdir
    save['words'] = np.array(['%s:%s:%s' % w
                              for w in words],
                             dtype=object)
    save['labels_lang'] = lab_lang
    save['null0_tids'] = np.array(null0_tids)
    save['proj_base'] = proj_f0
    save['proj_null'] = proj_n0
    npz_path = os.path.join(
        OUT, 'omega_a1_base_language_axis_xdir.npz')
    np.savez_compressed(npz_path, **save)

    seal = {
        'npz_sha256_8': sha8(npz_path),
        'result_sha256_8': sha8(
            os.path.join(OUT, 'result.json')),
        'exec_sha256_8': sha8(
            os.path.join(OUT, 'execution.json')),
    }
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(seal, f, indent=2)
    log('sealed %s' % json.dumps(seal), lines)
    log('elapsed %.1fs' % elapsed, lines)


if __name__ == '__main__':
    main()
