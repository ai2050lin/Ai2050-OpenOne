# -*- coding: utf-8 -*-
"""Phase 2952: amplification anatomy - attention gain vs value gain.

Question: the 2951 direct term was beta*g with beta ~ 1.5x the
naive linear prediction s*g (which assumed head input change =
s*Wv*xdir, i.e. A11=1 and no LN).  Where does the amplification
live: attention self-weight gain (A11(pos->pos1) jumps) or value
path (LN Jacobian)?

Exact identity (2-token prompts; pos1 attends only pos0/pos1;
o_proj input at pos1 x_h = A10*v0 + A11*v1):
  dx_h = dA11*(v1n - v0) + A11b*(v1n - v1b)
A11 recovered per (word, head) by least squares on the captured
line (v has no RoPE; recovery independent of q/k path).
All terms projected sep-style (lang0 mean - lang1 mean) per head,
so VAL + ATT == Delta (2951 npz, keep heads) exactly up to fp.

Main tests (frozen; quasi-post-hoc label per discipline 9 since
g/Delta/R were displayed in 2948-2951):
  T1 attention-gain dominance: median over keep heads of
     |ATT|/(|VAL|+|ATT|) > 0.6 for BOTH layers
  T2 attention gain is gain-sorted: |spearman(ATT, g)| >=
     perm-null p95 (20000 perms, seed 2904) BOTH layers
  T3 A11 gain: median(A11_inj)/median(A11_base) > 3 BOTH layers

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1&T2&T3   => amplification_attention_gain_sorted
  T1&T3      => amplification_attention_gain_unsorted
  T1 fail, val-share median > 0.6 => amplification_value_gain
  else       => amplification_mixed

Anchors (frozen): a1 dirs rebuild < 1e-5; a2 Vt8 vs 2939 < 1e-6;
a3 xdir self-check < 1e-9; a4 GQA gates (v_proj out 1024,
o_proj in 4096); a5 sc_base vs 2950 sc_B1 / sc_inj vs sc_I0
< 1e-5; a6 delta vs 2951 < 1e-5; a7 identity VAL+ATT vs Delta
< 1e-3; a8 line-recovery residual base < 0.1 inj < 0.3;
a9 same-session determinism < 1e-6.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2948 = os.path.join(BASE, 'phase2948', 'wov_head_gain',
                        'wov_head_gain.npz')
SRC_2950 = os.path.join(BASE, 'phase2950', 'rebalance_anatomy',
                        'rebalance_anatomy.npz')
SRC_2951 = os.path.join(BASE, 'phase2951',
                        'rebalance_carrier_functional',
                        'rebalance_carrier_functional.npz')
OUT = os.path.join(BASE, 'phase2952', 'amplification_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2952_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
VOCAB = 151936
LI_SWITCH, S_SWITCH = 17, 1.0
LI_GRAD, S_GRAD = 16, 2.0
LAYERS = (LI_SWITCH, LI_GRAD)
TOP5 = {LI_SWITCH: [0, 7, 24, 22, 19],
        LI_GRAD: [13, 16, 1, 17, 6]}
N_PERM = 20000
SEED = 2904

PREREG = {
    'mode': 'forward family: base + xdir injection L17 s=1.0 / '
            'L16 s=2.0 (no ablation); captures: v_proj output '
            'pos0/pos1, o_proj input pos1 at injected layers; '
            'A11 recovered by line least squares',
    'question': 'is the 2951 beta~1.5x amplification carried by '
                'attention self-weight gain (A11 jump) or by the '
                'value path (LN Jacobian)?',
    'identity': 'dx_h = dA11*(v1n - v0) + A11b*(v1n - v1b); '
                'sep-projected per head: VAL + ATT == Delta',
    'main_tests': {
        'T1': 'median keep-head |ATT|/(|VAL|+|ATT|) > 0.6 both',
        'T2': '|spearman(ATT, g)| >= perm p95 (20000, 2904) both',
        'T3': 'median(A11_inj)/median(A11_base) > 3 both',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2&T3 => amplification_attention_gain_sorted; '
               'T1&T3 => amplification_attention_gain_unsorted; '
               'T1 fail & val-share median > 0.6 => '
               'amplification_value_gain; else '
               '=> amplification_mixed',
    'quasi_post_hoc': 'discipline 9: g/Delta/R displayed in '
                      '2948-2951; VAL/ATT/A11 are new captures',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 < 1e-5',
        'a2': 'Vt8 vs 2939 < 1e-6',
        'a3': 'xdir self-check < 1e-9',
        'a4': 'v_proj out 1024, o_proj in 4096',
        'a5': 'sc_base vs B1, sc_inj vs I0 < 1e-5',
        'a6': 'delta vs 2951 < 1e-5',
        'a7': 'identity VAL+ATT vs Delta < 1e-3',
        'a8': 'line-recovery residual base < 0.1, inj < 0.3',
        'a9': 'same-session determinism < 1e-6',
    },
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
    lines.append(msg)
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2952,
                   'name': 'amplification_anatomy',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2948': sha8(SRC_2948),
                               's2950': sha8(SRC_2950),
                               's2951': sha8(SRC_2951)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'li_switch': LI_SWITCH, 's_switch': S_SWITCH,
                   'li_grad': LI_GRAD, 's_grad': S_GRAD,
                   'n_perm': N_PERM, 'seed': SEED,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word_27 = z27['dirs_word'].astype(np.float64)
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    coords39 = z39['coords'].astype(np.float64)
    dcks_39 = coords39[conds39.index('null0')] \
        - coords39[conds39.index('func')]
    z48 = np.load(SRC_2948, allow_pickle=True)
    z50 = np.load(SRC_2950, allow_pickle=True)
    z51 = np.load(SRC_2951, allow_pickle=True)

    # ---------- model ----------
    import torch
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)[
                    'input_ids']
            assert len(ids) == 1
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
    func_tid = tid('the')
    batch = [[func_tid, tid_map[words[i][2]]]
             for i in range(n_words)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    a4_ok = bool(
        layers[0].self_attn.v_proj.weight.shape[0] == 1024
        and layers[0].self_attn.o_proj.in_features == NH * HD)
    log('a4 GQA gates v_proj=%d o_proj_in=%d ok=%s'
        % (layers[0].self_attn.v_proj.weight.shape[0],
           layers[0].self_attn.o_proj.in_features, a4_ok),
        lines)

    cap_v = {}
    cap_x = {}
    cap_in = {'on': False, 'store': {}}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['li'] == li and inj['vec'] is not None:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            if cap_in['on']:
                cap_in['store'].setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_v(li):
        def h(module, args, output):
            if li in LAYERS:
                o = output.detach().float().cpu().numpy()
                cap_v.setdefault(li, []).append(
                    (o[:, 0, :].copy(), o[:, 1, :].copy()))
            return None
        return h

    def hook_x(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            if li in LAYERS:
                cap_x.setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    for li in LAYERS:
        handles.append(layers[li].self_attn.v_proj
                       .register_forward_hook(hook_v(li)))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           hook_x(li), with_kwargs=True))

    # ---------- pass 1: dirs_word rebuild (a1/a2/a3) ----------
    # capture layer-input residual via the self_attn pre-hook
    # (decoder-layer-level with_kwargs hooks corrupt the
    # transformers forward - run2 finding; self_attn-level
    # hooks are the proven-safe path, 2950 verbatim)
    attn_store = {}
    cap_in['on'] = True
    for i, (_, _, w) in enumerate(words):
        cap_in['store'].clear()
        with torch.no_grad():
            model(torch.tensor([[func_tid,
                                 tid_map[words[i][2]]]],
                               device='cuda'))
        for li in range(NL):
            attn_store[(i, li)] = \
                cap_in['store'][li][0].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    cap_in['on'] = False

    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs rebuild diff %.2e ok=%s' % (a1_diff, a1_ok),
        lines)
    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a2_diff = float(np.abs(Vt8 - Vt8_39).max())
    a2_ok = bool(a2_diff < 1e-6)
    log('a2 Vt8 vs 2939 %.2e ok=%s' % (a2_diff, a2_ok), lines)
    u35 = dirs_word[NL - 1]
    S_IDX = (0, 1, 4)
    xdir = dcks_39[:, list(S_IDX)] @ Vt8[list(S_IDX)]
    a3_diff = float(np.abs(
        xdir @ Vt8[list(S_IDX)].T
        - dcks_39[:, list(S_IDX)]).max())
    a3_ok = bool(a3_diff < 1e-9)
    log('a3 xdir self-check %.2e ok=%s' % (a3_diff, a3_ok),
        lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

    # ---------- main forwards ----------
    def run(inj_li, scale):
        cap_v.clear()
        cap_x.clear()
        inj['li'] = inj_li
        inj['scale'] = float(scale)
        inj['vec'] = xdir_t if scale else None
        with torch.no_grad():
            model(torch.tensor(batch, device='cuda'))
        inj['li'] = None
        inj['scale'] = 0.0
        inj['vec'] = None
        v = {li: (np.stack([a for a, b in cap_v[li]]),
                  np.stack([b for a, b in cap_v[li]]))
             for li in cap_v}
        x = {li: np.stack(cap_x[li])[0] for li in cap_x}
        return v, x

    v_base, x_base = run(None, 0.0)
    v_17, x_17 = run(LI_SWITCH, S_SWITCH)
    v_16, x_16 = run(LI_GRAD, S_GRAD)
    # a9 determinism
    v_b2, x_b2 = run(None, 0.0)
    a9_diff = max(float(np.abs(x_b2[LI_SWITCH]
                               - x_base[LI_SWITCH]).max()),
                  float(np.abs(v_b2[LI_SWITCH][1]
                               - v_base[LI_SWITCH][1]).max()))
    a9_ok = bool(a9_diff < 1e-6)
    log('a9 determinism %.2e ok=%s' % (a9_diff, a9_ok), lines)

    NKV = 1024 // HD
    HPG = NH // NKV
    Wo_cache = {li: layers[li].self_attn.o_proj.weight
                .detach().float().cpu().numpy()
                for li in LAYERS}

    verdict = None
    t1 = t2 = t3 = d1 = None
    save = {}
    a5_diff = a6_diff = a7_diff = a8_diff = None
    a5_ok = a6_ok = a7_ok = a8_ok = False

    anchor_prelim = bool(a1_ok and a2_ok and a3_ok and a4_ok
                         and a9_ok)
    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        t1 = {}
        t2 = {}
        t3 = {}
        d1 = {}
        for li, v_inj, x_inj, s_inj in (
                (LI_SWITCH, v_17, x_17, S_SWITCH),
                (LI_GRAD, v_16, x_16, S_GRAD)):
            key = 'L%d' % li
            Wo = Wo_cache[li]
            v0b, v1b = v_base[li]
            v0n, v1n = v_inj[li]
            Xb = x_base[li].reshape(n_words, NH, HD)
            Xn = x_inj[li].reshape(n_words, NH, HD)
            v0b_r = v0b.reshape(n_words, NKV, HD)
            v1b_r = v1b.reshape(n_words, NKV, HD)
            v0n_r = v0n.reshape(n_words, NKV, HD)
            v1n_r = v1n.reshape(n_words, NKV, HD)

            def recover(Xf, v0r, v1r):
                A = np.zeros((n_words, NH))
                res = 0.0
                for hh in range(NH):
                    k = hh // HPG
                    d = v1r[:, k, :] - v0r[:, k, :]
                    den_w = (d * d).sum(1)
                    num = ((Xf[:, hh, :] - v0r[:, k, :])
                           * d).sum(1)
                    A[:, hh] = num / np.maximum(den_w,
                                                1e-30)
                    rec = v0r[:, k, :] + A[:, hh:hh + 1] * d
                    res = max(res, float(np.abs(
                        rec - Xf[:, hh, :]).max()))
                return A, res

            A11b, res_b = recover(Xb, v0b_r, v1b_r)
            A11n, res_n = recover(Xn, v0n_r, v1n_r)
            a8_diff = max(a8_diff or 0.0, res_b, res_n)

            def sep_proj(vec_w):
                return vec_w[lab_lang == 0].mean(0) \
                    - vec_w[lab_lang == 1].mean(0)

            def sc_of(Xf):
                c = np.zeros((NH, n_words))
                for hh in range(NH):
                    oh = Xf[:, hh, :] \
                        @ Wo[:, hh * HD:(hh + 1) * HD].T
                    c[hh] = oh @ u35
                return c[:, lab_lang == 0].mean(1) \
                    - c[:, lab_lang == 1].mean(1)

            sc_b = sc_of(Xb)
            sc_n = sc_of(Xn)
            delta = sc_n - sc_b
            keep = z51['keep_%s' % key]
            e_b = float(np.abs(
                sc_b[keep] - z50['sc_B1_%s' % key][keep]).max())
            e_n = float(np.abs(
                sc_n[keep] - z50['sc_I0_%s' % key][keep]).max())
            e_d = float(np.abs(
                delta[keep] - z51['delta_%s' % key]).max())
            a5_diff = max(a5_diff or 0.0, e_b, e_n)
            a6_diff = max(a6_diff or 0.0, e_d)

            VAL = np.zeros(NH)
            ATT = np.zeros(NH)
            for hh in range(NH):
                k = hh // HPG
                woh = Wo[:, hh * HD:(hh + 1) * HD]
                dA = A11n[:, hh] - A11b[:, hh]
                att_w = dA * ((v1n_r[:, k, :] - v0n_r[:, k, :])
                              @ woh.T @ u35)
                val_w = A11b[:, hh] * ((v1n_r[:, k, :]
                                        - v1b_r[:, k, :])
                                       @ woh.T @ u35)
                ATT[hh] = sep_proj(att_w)
                VAL[hh] = sep_proj(val_w)
            ident = float(np.abs(
                (VAL + ATT)[keep] - delta[keep]).max())
            a7_diff = max(a7_diff or 0.0, ident)

            share_att = np.abs(ATT) / np.maximum(
                np.abs(VAL) + np.abs(ATT), 1e-30)
            med_share_att = float(np.median(share_att[keep]))
            t1_pass_li = bool(med_share_att > 0.6)

            gk = z51['g_%s' % key]
            rk = z51['resid_%s' % key]
            rho_ag = float(spearmanr(ATT[keep], gk).statistic)
            rng = np.random.default_rng(SEED)
            null_ag = np.array([
                abs(spearmanr(
                    ATT[keep],
                    rng.permutation(gk)).statistic)
                for _ in range(N_PERM)])
            p95 = float(np.quantile(null_ag, 0.95))
            t2_pass_li = bool(abs(rho_ag) >= p95)

            ratio = float(np.median(np.abs(A11n))
                          / max(float(np.median(np.abs(A11b))),
                                1e-30))
            t3_pass_li = bool(ratio > 3.0)

            t1[key] = {'att_share_med':
                       round(med_share_att, 4),
                       'pass': t1_pass_li}
            t2[key] = {'rho_att_g': round(rho_ag, 4),
                       'null_p95': round(p95, 4),
                       'pass': t2_pass_li}
            t3[key] = {'a11_base_med':
                       round(float(np.median(A11b[:, keep])), 4),
                       'a11_inj_med':
                       round(float(np.median(A11n[:, keep])), 4),
                       'ratio': round(ratio, 2),
                       'pass': t3_pass_li}
            log('%s: att-share %.3f (T1 %s) | rho(ATT,g) %.4f '
                'p95 %.4f (T2 %s) | A11 %.3f->%.3f ratio %.1f '
                '(T3 %s) | ident %.2e'
                % (key, med_share_att, t1_pass_li, rho_ag,
                   p95, t2_pass_li,
                   float(np.median(A11b[:, keep])),
                   float(np.median(A11n[:, keep])), ratio,
                   t3_pass_li, ident), lines)

            save['A11b_%s' % key] = A11b
            save['A11n_%s' % key] = A11n
            save['VAL_%s' % key] = VAL
            save['ATT_%s' % key] = ATT
            save['delta_%s' % key] = delta
            d1[key] = {
                'val_share_med':
                    round(1.0 - med_share_att, 4),
                'mu_val': round(float(np.median(
                    VAL[keep] / np.where(
                        np.abs(gk) > 1e-30, gk, 1e-30))), 4),
                'rho_att_R': round(float(
                    spearmanr(ATT[keep], rk).statistic), 4),
                'recon_res': {'base': float('%.3e' % res_b),
                              'inj': float('%.3e' % res_n)},
            }

        a5_ok = bool(a5_diff < 1e-5)
        a6_ok = bool(a6_diff < 1e-5)
        a7_ok = bool(a7_diff < 1e-3)
        a8_ok = bool(a8_diff < 0.3)
        log('a5 sc match %.2e ok=%s | a6 delta %.2e ok=%s | '
            'a7 identity %.2e ok=%s | a8 recon %.2e ok=%s'
            % (a5_diff, a5_ok, a6_diff, a6_ok,
               a7_diff, a7_ok, a8_diff, a8_ok), lines)

        t1_all = all(v['pass'] for v in t1.values())
        t2_all = all(v['pass'] for v in t2.values())
        t3_all = all(v['pass'] for v in t3.values())
        if not (a5_ok and a6_ok and a7_ok and a8_ok):
            verdict = 'anchor_fail_all_void'
        elif t1_all and t2_all and t3_all:
            verdict = 'amplification_attention_gain_sorted'
        elif t1_all and t3_all:
            verdict = 'amplification_attention_gain_unsorted'
        elif not t1_all and all(
                1.0 - t1[k]['att_share_med'] > 0.6
                for k in t1):
            verdict = 'amplification_value_gain'
        else:
            verdict = 'amplification_mixed'

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2952, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
               'a1_ok': a1_ok,
               'a2_diff': float('%.3e' % a2_diff),
               'a2_ok': a2_ok,
               'a3_diff': float('%.3e' % a3_diff),
               'a3_ok': a3_ok,
               'a4_ok': a4_ok,
               'a5_diff': None if a5_diff is None
               else float('%.3e' % a5_diff),
               'a5_ok': a5_ok,
               'a6_diff': None if a6_diff is None
               else float('%.3e' % a6_diff),
               'a6_ok': a6_ok,
               'a7_diff': None if a7_diff is None
               else float('%.3e' % a7_diff),
               'a7_ok': a7_ok,
               'a8_diff': None if a8_diff is None
               else float('%.3e' % a8_diff),
               'a8_ok': a8_ok,
               'a9_diff': float('%.3e' % a9_diff),
               'a9_ok': a9_ok,
               'ok': bool(anchor_prelim and a5_ok and a6_ok
                          and a7_ok and a8_ok)},
           'T1_att_share': t1, 'T2_att_gain_sorted': t2,
           'T3_a11_gain': t3, 'D1_anatomy': d1,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'amplification_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2952 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
