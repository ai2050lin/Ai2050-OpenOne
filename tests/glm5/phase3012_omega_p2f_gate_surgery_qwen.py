# -*- coding: utf-8 -*-
"""Phase 3012: Omega-P2f L3 gate surgery operability (qwen).

Why: 3011 localized the logic-position distribution gating
to L3 KV (D_l*=0.0137 p_maxT=1e-4, 2x runner-up L31; K/V
both arms carry).  Open question: is the gate INFORMATIONAL
(the specific K/V content written by the logic token is
what gates the distribution) or CAPACITY (any K/V mass at
that position works)?  This is the white-box surgery
operability test: if mean-refill restores the distribution,
the gate is content-specific and surgically targetable.

Design (3011 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol; T2a/b/c
replaced):
  T2a PRIMARY refill contrast at L3 (from 3011 result
           l_star read and asserted in a15): per logic
           position p, three arms - ZERO (K,V scaled 0.0,
           = 3011 erasure), MEAN (K,V at p replaced by the
           per-dim mean over the SAME prompt prefill other
           positions of L3), RAND (K,V at p replaced by
           per-dim Gaussian matched to the empirical
           mean/std over other positions, seeded).  JS
           nats vs the same-prompt unintervened two-step
           baseline.  Per-position recovery
           recov_i = 1 - JS_arm_i / JS_zero_i (paired);
           PRIMARY stat = med recov over logic positions;
           sign-flip permutation N_PERM two-sided.
  T2b capacity contrast (DESCRIPTIVE): med recov_rand;
           gate: informational requires med recov_mean
           >= 0.5 AND p<0.05 AND med recov_rand < 0.25;
           capacity requires med recov_rand >= 0.5 AND
           recov_rand >= recov_mean.
  T2c dose (DESCRIPTIVE): refill contrast at s in
           (0.25,0.5,0.75) - K,V scaled s then MEAN-refill
           added mass; JS vs baseline.

Verdict (frozen):
  anchor fail                         => anchor_fail_all_
                                          void
  med recov_mean>=0.5 AND p<0.05 AND  => info_carrying_
  med recov_rand<0.25                    gate_qwen
  med recov_rand>=0.5 AND             => capacity_gate_
  recov_rand>=recov_mean                 qwen
  else                                => gate_mixed_qwen

Anchors (frozen): a0-a14 as 3011 verbatim (incl. a13 T3
drift vs 3009 49.5123 and a14 3010 integrity), plus
  a15 3011 integrity: seal match AND verdict ==
      js_layer_localized_qwen AND anchors ok AND
      l_star == 3 (L3_GATED asserted).

Tags: Omega-P2f / gate surgery / informational-vs-capacity
/ mean-vs-noise refill / paired sign-flip permutation / no
hallucination naming.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927',
                        'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935',
                        'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939',
                        'rotation_target',
                        'rotation_target.npz')
D_2993 = os.path.join(BASE, 'phase2993',
                      'logic_signature_registration')
D_3007 = os.path.join(BASE, 'phase3007',
                      'omega_p2a_generation_trajectory_'
                      'qwen')
D_3008 = os.path.join(BASE, 'phase3008',
                      'omega_p2b_logic_kv_causal_qwen')
D_3009 = os.path.join(BASE, 'phase3009',
                      'omega_p2c_kv_scale_causal_qwen')
D_3010 = os.path.join(BASE, 'phase3010',
                      'omega_p2d_logitlens_causal_qwen')
D_3011 = os.path.join(BASE, 'phase3011',
                      'omega_p2e_layer_js_localization_'
                      'qwen')
OUT = os.path.join(BASE, 'phase3012',
                   'omega_p2f_gate_surgery_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002..3011 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009/3010/3011
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
L3_GATED = 3
SCALE_DOSE = (0.25, 0.5, 0.75)
RECOV_INFO = 0.5
RECOV_RAND_MAX = 0.25
T3_3009_DRIFT = 49.5123
A13_GATE = 5e-5
GEN_PROMPTS = (
    'The weather was cold, so',
    'He studied every night because',
    'She wanted to buy the car, but',
    'The experiment failed, therefore',
    'You should take an umbrella if',
    'The meeting was long, and',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',
    'We left early because',)
LOGIC_WORDS = ('and', 'but', 'or', 'so', 'if', 'then',
               'because', 'therefore', 'however', 'while',
               'thus', 'although')
FUNC_WORDS = ('the', 'of', 'to', 'a', 'in', 'is', 'that',
              'it', 'for', 'on', 'with', 'as', 'at', 'by',
              'from', 'this', 'be', 'are', 'was', 'were',
              'has', 'had', 'have', 'will', 'would', 'can',
              'could', 'not', 'no', 'yes', 'he', 'she',
              'they', 'we', 'you', 'i', 'his', 'her',
              'their', 'our', 'my', 'when', 'where', 'who')

PREREG = {
    'mode': 'qwen3-4b; 3011 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by L3 refill contrast',
    'question': 'is the L3 logic-position distribution '
                'gate (3011) INFORMATIONAL - the specific '
                'K/V content written by the logic token - '
                'or CAPACITY - any K/V mass works? mean-'
                'refill restores => informational; noise-'
                'refill restores equally => capacity',
    'T2a': 'PRIMARY: at L3 (=3011 l_star, asserted a15), '
           'per logic position p three arms - ZERO (K,V '
           'scaled 0.0, 3011 erasure), MEAN (K,V at p '
           'replaced by per-dim mean over the same-prompt '
           'prefill other positions of L3), RAND (per-dim '
           'Gaussian matched to empirical mean/std over '
           'other positions, seed SEED_RND+50, per '
           'position draw); two-step protocol both arms '
           'identical; JS nats vs unintervened two-step '
           'baseline; per-position paired recovery '
           'recov_i = 1 - JS_arm/JS_zero (JS_zero>0 gate '
           'per position); PRIMARY stat = med recov over '
           'logic positions, sign-flip permutation '
           'N=10000 two-sided; gates nL>=8 and per-pos '
           'JS_zero>0; med JS_zero / med JS_mean / med '
           'JS_rand reported',
    'T2b': 'DESCRIPTIVE capacity contrast: med recov_rand '
           'vs med recov_mean verdict gates (frozen in '
           'verdict field)',
    'T2c': 'DESCRIPTIVE dose: s in (0.25,0.5,0.75) - K,V '
           'scaled s then MEAN-refill added at p; JS vs '
           'baseline; recovery fraction per scale',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'med recov_mean>=0.5 AND p<0.05 AND med '
               'recov_rand<0.25 => info_carrying_gate_'
               'qwen; med recov_rand>=0.5 AND recov_rand>'
               '=recov_mean => capacity_gate_qwen; else '
               '=> gate_mixed_qwen',
    'tags': 'Omega-P2f / gate surgery / informational-vs-'
            'capacity / mean-vs-noise refill / paired '
            'sign-flip permutation / no hallucination '
            'naming',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def js_nats(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask]
                            * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'),
                              msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 3012,
                   'name': 'omega_p2f_gate_surgery_qwen',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2927': sha8(SRC_2927),
                       's2935': sha8(SRC_2935),
                       's2939': sha8(SRC_2939),
                       's2993result': sha8(
                           D_2993 + r'\result.json'),
                       's2993npz': sha8(
                           D_2993 + r'\logic_signature_'
                           r'registration.npz'),
                       's3007result': sha8(
                           D_3007 + r'\result.json'),
                       's3007seal': sha8(
                           D_3007 + r'\seal.json'),
                       's3008result': sha8(
                           D_3008 + r'\result.json'),
                       's3008seal': sha8(
                           D_3008 + r'\seal.json'),
                       's3009result': sha8(
                           D_3009 + r'\result.json'),
                       's3009seal': sha8(
                           D_3009 + r'\seal.json'),
                       's3010result': sha8(
                           D_3010 + r'\result.json'),
                       's3010seal': sha8(
                           D_3010 + r'\seal.json'),
                       's3011result': sha8(
                           D_3011 + r'\result.json'),
                       's3011seal': sha8(
                           D_3011 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'scale_dose': list(SCALE_DOSE),
                   'recov_info': RECOV_INFO,
                   'recov_rand_max': RECOV_RAND_MAX,
                   't3_3009_drift': T3_3009_DRIFT,
                   'a13_gate': A13_GATE,
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'prompts': list(GEN_PROMPTS),
                   'logic_words': list(LOGIC_WORDS),
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- source integrity ----------
    r93 = json.load(open(D_2993 + r'\result.json',
                         encoding='utf-8'))
    seal93 = json.load(open(D_2993 + r'\seal.json',
                            encoding='utf-8'))
    a0_ok = bool(seal93['result_sha256_8']
                 == sha8(D_2993 + r'\result.json')
                 and r93['final_verdict']
                 == 'logic_signature_length_robust'
                 and r93['anchor_all_ok'] is True)
    log('a0 2993 integrity %s (verdict=%s)'
        % (a0_ok, r93['final_verdict']), lines)

    z93 = np.load(D_2993 + r'\logic_signature_'
                  r'registration.npz', allow_pickle=True)
    l_words = [str(w) for w in z93['l_words']]
    l_A = [str(w) for w in z93['l_A']]
    l_B = [str(w) for w in z93['l_B']]
    res34_93 = z93['res34'].astype(np.float64)
    w2_93 = z93['w2'].astype(np.float64)
    w1024_93 = z93['w1024'].astype(np.float64)
    iC = list(range(30, 52))
    iLA = [74 + l_words.index(w) for w in l_A]
    w2_rec = res34_93[0, iLA].mean(0) \
        - res34_93[0, iC].mean(0)
    w1024_rec = res34_93[4, iLA].mean(0) \
        - res34_93[4, iC].mean(0)
    sc1 = max(float(np.abs(w2_93).max()),
              float(np.abs(w1024_93).max()), 1e-30)
    a1_diff = float(max(np.abs(w2_rec - w2_93).max(),
                        np.abs(w1024_rec
                               - w1024_93).max()) / sc1)
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 w2/w1024 recompute vs 2993 %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    seal07 = json.load(open(D_3007 + r'\seal.json',
                            encoding='utf-8'))
    r07 = json.load(open(D_3007 + r'\result.json',
                         encoding='utf-8'))
    a9_ok = bool(seal07['result_sha256_8']
                 == sha8(D_3007 + r'\result.json')
                 and r07['final_verdict']
                 == 'logic_locked_perturb_divergent_qwen'
                 and r07['anchor_all_ok'] is True)
    log('a9 3007 integrity %s (verdict=%s)'
        % (a9_ok, r07['final_verdict']), lines)

    seal08 = json.load(open(D_3008 + r'\seal.json',
                            encoding='utf-8'))
    r08 = json.load(open(D_3008 + r'\result.json',
                         encoding='utf-8'))
    a11_ok = bool(seal08['result_sha256_8']
                  == sha8(D_3008 + r'\result.json')
                  and r08['final_verdict']
                  == 'logic_sig_gen_only_qwen'
                  and r08['anchor_all_ok'] is True)
    log('a11 3008 integrity %s (verdict=%s)'
        % (a11_ok, r08['final_verdict']), lines)

    seal09 = json.load(open(D_3009 + r'\seal.json',
                            encoding='utf-8'))
    r09 = json.load(open(D_3009 + r'\result.json',
                         encoding='utf-8'))
    a12_ok = bool(seal09['result_sha256_8']
                  == sha8(D_3009 + r'\result.json')
                  and r09['final_verdict']
                  == 'kv_scale_saturated_qwen'
                  and r09['anchor_all_ok'] is True)
    log('a12 3009 integrity %s (verdict=%s)'
        % (a12_ok, r09['final_verdict']), lines)

    seal10 = json.load(open(D_3010 + r'\seal.json',
                            encoding='utf-8'))
    r10 = json.load(open(D_3010 + r'\result.json',
                         encoding='utf-8'))
    a14_ok = bool(seal10['result_sha256_8']
                  == sha8(D_3010 + r'\result.json')
                  and r10['final_verdict']
                  == 'logitlens_logic_specific_qwen'
                  and r10['anchor_all_ok'] is True)
    log('a14 3010 integrity %s (verdict=%s)'
        % (a14_ok, r10['final_verdict']), lines)

    seal11 = json.load(open(D_3011 + r'\seal.json',
                            encoding='utf-8'))
    r11 = json.load(open(D_3011 + r'\result.json',
                         encoding='utf-8'))
    a15_ok = bool(seal11['result_sha256_8']
                  == sha8(D_3011 + r'\result.json')
                  and r11['final_verdict']
                  == 'js_layer_localized_qwen'
                  and r11['anchor_all_ok'] is True
                  and r11['T2a']['l_star'] == L3_GATED)
    log('a15 3011 integrity %s (verdict=%s l*=%s)'
        % (a15_ok, r11['final_verdict'],
           r11['T2a']['l_star']), lines)

    # ---------- upstream geometry ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(
        0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract \
        import load_native
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
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    # a8: l_words single-token recheck
    n_ok8 = 0
    for w in l_words:
        try:
            tid(w)
            n_ok8 += 1
        except AssertionError:
            pass
    a8_ok = bool(n_ok8 == len(l_words))
    log('a8 l_words single-token %d/%d ok=%s'
        % (n_ok8, len(l_words), a8_ok), lines)

    SRC_2887 = os.path.join(BASE, 'phase2887',
                            'language_axis_mlp',
                            'language_axis_mlp.npz')
    z88 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z88['words']]
    lab_lang = np.asarray(
        z88['labels_lang']).astype(int)
    n_words = len(words)
    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck)
    func_tid = tid('the')
    word_tids = set(tid_map.values())

    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)

    logic_tids = {}
    for w in LOGIC_WORDS:
        logic_tids[tid(w)] = w
    for w in l_words:
        logic_tids[tid(w)] = w
    func_tids = {}
    for w in FUNC_WORDS:
        func_tids[tid(w)] = w
    log('class lists: logic=%d func=%d'
        % (len(logic_tids), len(func_tids)), lines)

    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i],
                        tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}}
    state_fin = {'on': False}
    fin_cap = {}
    state_res = {'on': False}
    res_cap = {}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            cap['ai'].setdefault(li, []).append(
                x.detach().float().cpu().numpy()
                .copy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy().copy()
        return None

    def hook_res(module, args, kwargs):
        if state_res['on']:
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is not None and x.dim() >= 2:
                res_cap['x'] = x[:, -1, :].detach() \
                    .float().cpu().numpy().copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(layers[L_SIG]
                   .register_forward_pre_hook(
                       hook_res, with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]
        res_cap.pop('x', None)

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

    def forward_batch(toks_list):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1(
            [func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        d_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(d_w[li])
                          for li in range(NL)])
    a2_diff = float(np.abs(dirs_word - dirs27).max())
    a2_ok = bool(a2_diff < 1e-5)
    log('a2 dirs_word vs 2927 %.2e ok=%s'
        % (a2_diff, a2_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_word,
                             full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]

    # a7 xdir identity (3011 verbatim, S_IDX 0/1/4)
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]
    S_IDX = (0, 1, 4)
    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir identity %.2e ok=%s'
        % (a7_diff, a7_ok), lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a6_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a6_ok = bool(a6_rel < 1e-4)
    proj_f0 = fin_f1 @ u35
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0 = fin_n0 @ u35
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035])
                    .max())
    a5_ok = bool(a5_diff < 1e-4)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    log('a4 %.2e a5 %.2e a6 rel %.2e ok=%s/%s/%s '
        'sep_f=%.2f'
        % (a4_diff, a5_diff, a6_rel, a4_ok, a5_ok,
           a6_ok, sep_f), lines)

    # ---------- generation machine ----------
    def gen_coords():
        return fin_cap['x'][0].astype(np.float64)

    def kv_scale(past, p, s, li=L3_GATED):
        L = past.layers[li]
        L.keys[:, :, p, :] *= s
        L.values[:, :, p, :] *= s

    def kv_refill(past, p, mode, li=L3_GATED):
        """Replace K,V at position p in layer li.
        MEAN: per-dim mean over other positions.
        RAND: per-dim Gaussian with the empirical
        mean/std over other positions (numpy seeded
        draw, placed on device in float32)."""
        L = past.layers[li]
        T = int(L.keys.shape[2])
        others = [j for j in range(T) if j != p]
        ko = L.keys[:, :, others, :]
        vo = L.values[:, :, others, :]
        if mode == 'mean':
            L.keys[:, :, p, :] = ko.mean(dim=2) \
                .to(L.keys.dtype)
            L.values[:, :, p, :] = vo.mean(dim=2) \
                .to(L.values.dtype)
        elif mode == 'rand':
            mu_k = ko.mean(dim=2).float()
            sd_k = ko.std(dim=2).float()
            mu_v = vo.mean(dim=2).float()
            sd_v = vo.std(dim=2).float()
            shape_k = tuple(L.keys[:, :, p, :].shape)
            shape_v = tuple(
                L.values[:, :, p, :].shape)
            gk = torch.Generator()
            gk.manual_seed(SEED_RND + 50
                           + int(p))
            gv = torch.Generator()
            gv.manual_seed(SEED_RND + 60
                           + int(p))
            dev = L.keys.device
            nk = torch.randn(shape_k,
                             generator=gk).to(dev)
            nv = torch.randn(shape_v,
                             generator=gv).to(dev)
            L.keys[:, :, p, :] = (
                mu_k + sd_k * nk) \
                .to(L.keys.dtype)
            L.values[:, :, p, :] = (
                mu_v + sd_v * nv) \
                .to(L.values.dtype)

    def generate(prompt, k):
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        state_fin['on'] = True
        state_res['on'] = True
        rec = {'ids': [], 's': [], 'c8': []}
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            x0 = gen_coords()
            rec['s_pre'] = float(x0 @ u35)
            rec['c8_pre'] = x0 @ Vt8.T
            nid = int(out.logits[0, -1].argmax())
            for t in range(k):
                clear_cap()
                out = model(
                    input_ids=torch.tensor(
                        [[nid]], device='cuda'),
                    past_key_values=past,
                    use_cache=True)
                past = out.past_key_values
                x = gen_coords()
                rec['ids'].append(nid)
                rec['s'].append(float(x @ u35))
                rec['c8'].append(x @ Vt8.T)
                nid = int(out.logits[0, -1].argmax())
        state_fin['on'] = False
        state_res['on'] = False
        for key in ('ids', 's', 'c8'):
            rec[key] = np.array(rec[key])
        rec['prompt_ids'] = np.array(ids)
        return rec

    def prefill_step2(prompt, ids, past):
        out2 = model(
            input_ids=torch.tensor(
                [[int(ids[-1])]], device='cuda'),
            past_key_values=past,
            use_cache=False)
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p = np.exp(lg)
        p = p / p.sum()
        return p, int(np.argmax(lg))

    def prefill_logits(prompt):
        """TWO-STEP protocol (both arms identical): prefill
        forward, then feed ids[-1] once more and read that
        step's next-token distribution (3010/3011
        verbatim)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            p, am = prefill_step2(prompt, ids, past)
        return ids, p, am

    def prefill_logits_surgery(prompt, p_pos, mode,
                               s=None):
        """ZERO: K,V at p scaled 0.0.  MEAN/RAND: K,V at
        p replaced by refill (optionally after scaling
        by s for the dose arm - refill overwrites)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if mode == 'zero':
                kv_scale(past, p_pos,
                         0.0 if s is None else s)
            else:
                if s is not None:
                    kv_scale(past, p_pos, s)
                kv_refill(past, p_pos, mode)
            q, am = prefill_step2(prompt, ids, past)
        return q, am

    dec_cache = {}

    def classify(t_id):
        if t_id in logic_tids:
            return 'logic'
        if t_id in func_tids:
            return 'func'
        if t_id not in dec_cache:
            dec_cache[t_id] = tok.decode(
                [int(t_id)]).strip()
        txt = dec_cache[t_id]
        if txt.isalpha() and len(txt) >= 3:
            return 'content'
        return 'other'

    # ---------- baseline generation ----------
    anchor_prelim = bool(a0_ok and a1_ok and a2_ok
                         and a3_ok and a4_ok and a5_ok
                         and a6_ok and a7_ok and a8_ok
                         and a9_ok and a11_ok and a12_ok
                         and a14_ok and a15_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    if anchor_prelim:
        for pi, pr in enumerate(GEN_PROMPTS):
            recs[pi] = generate(pr, K_GEN)
            log('gen P%d done' % pi, lines)
        recs2 = generate(GEN_PROMPTS[0], K_GEN)
        ids_same = bool(np.array_equal(
            recs[0]['ids'], recs2['ids']))
        cmax = max(float(np.abs(recs[0]['c8']).max()),
                   float(np.abs(recs2['c8']).max()),
                   1e-30)
        a10_rel = float(np.abs(recs[0]['c8']
                               - recs2['c8']).max()) \
            / cmax
        a10_ok = bool(ids_same and a10_rel < 1e-4)
        log('a10 gen determinism ids_same=%s rel=%.2e '
            'ok=%s' % (ids_same, a10_rel, a10_ok), lines)

        # a13: T3 drift bit-level vs 3009
        drift_s = []
        for pi in range(len(GEN_PROMPTS)):
            rec = recs[pi]
            drift_s.append(abs(float(
                rec['s'][K_GEN - 1]
                - rec['s_pre'])))
        drift_med = round(float(np.median(drift_s)), 4)
        a13_diff = abs(drift_med - T3_3009_DRIFT)
        a13_ok = bool(a13_diff < A13_GATE)
        log('a13 T3 drift %.4f vs 3009 %.4f diff=%.2e '
            'ok=%s' % (drift_med, T3_3009_DRIFT,
                       a13_diff, a13_ok), lines)

        if not (a10_ok and a13_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- position selection ----------
            # (3009/3010/3011 protocol verbatim,
            #  seed chain SEED_RND+20)
            rng2 = np.random.default_rng(
                SEED_RND + 20)
            sel = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ids = list(recs[pi]['prompt_ids'])
                lp = [i for i, t in enumerate(ids)
                      if int(t) in logic_tids]
                cp = [i for i, t in enumerate(ids)
                      if classify(int(t)) == 'content']
                sp = [i for i, t in enumerate(ids)
                      if classify(int(t)) in
                      ('func', 'other')]
                ent = {'n_logic_pos': len(lp),
                       'n_content_pos': len(cp)}
                if not lp or len(cp) < 2:
                    ent['skipped'] = True
                    sel['P%d' % pi] = ent
                    log('T2 P%d skipped (lp=%d cp=%d)'
                        % (pi, len(lp), len(cp)), lines)
                    continue
                lp_use = lp[:2]
                cp_use = list(rng2.choice(
                    cp, size=2, replace=False)) \
                    if len(cp) >= 2 else cp[:2]
                sham_pool = [i for i in sp
                             if i not in lp_use
                             and i not in cp_use]
                if not sham_pool:
                    rest = [i for i in range(len(ids))
                            if i not in lp_use
                            and i not in cp_use]
                    sham_pool = rest
                sham = int(rng2.choice(sham_pool)) \
                    if sham_pool else None
                ent['positions'] = {
                    'logic': lp_use,
                    'content': [int(x) for x
                                in cp_use],
                    'sham': sham}
                sel['P%d' % pi] = ent

            # ---------- T2a refill contrast ----------
            base_probs = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ids, p, am = prefill_logits(pr)
                base_probs[pi] = p
            rowsZ = []
            rowsM = []
            rowsR = []
            tags = []
            shamZ = []
            shamM = []
            shamR = []
            contentM = []
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                pos = ent['positions']
                pb = base_probs[pi]
                for p_pos in pos['logic']:
                    qz, _ = prefill_logits_surgery(
                        pr, int(p_pos), 'zero')
                    qm, _ = prefill_logits_surgery(
                        pr, int(p_pos), 'mean')
                    qr, _ = prefill_logits_surgery(
                        pr, int(p_pos), 'rand')
                    jz = js_nats(pb, qz)
                    jm = js_nats(pb, qm)
                    jr = js_nats(pb, qr)
                    if jz > 0:
                        rowsZ.append(jz)
                        rowsM.append(jm)
                        rowsR.append(jr)
                        tags.append('P%d:%d'
                                    % (pi, p_pos))
                for p_pos in pos['content']:
                    qm, _ = prefill_logits_surgery(
                        pr, int(p_pos), 'mean')
                    contentM.append(js_nats(pb, qm))
                if pos['sham'] is not None:
                    qz, _ = prefill_logits_surgery(
                        pr, int(pos['sham']), 'zero')
                    qm, _ = prefill_logits_surgery(
                        pr, int(pos['sham']), 'mean')
                    qr, _ = prefill_logits_surgery(
                        pr, int(pos['sham']), 'rand')
                    shamZ.append(js_nats(pb, qz))
                    shamM.append(js_nats(pb, qm))
                    shamR.append(js_nats(pb, qr))
                log('T2a P%d done' % pi, lines)
            nL = len(rowsZ)
            gates_ok = bool(nL >= MIN_POS)
            vZ = np.array(rowsZ)
            vM = np.array(rowsM)
            vR = np.array(rowsR)
            recM = 1.0 - vM / vZ
            recR = 1.0 - vR / vZ
            med_recM = float(np.median(recM)) \
                if nL else None
            med_recR = float(np.median(recR)) \
                if nL else None
            # paired sign-flip permutation on recM
            p_rec = None
            if nL:
                rng3 = np.random.default_rng(
                    SEED_RND + 70)
                obs = abs(med_recM)
                cnt = 0
                for _ in range(N_PERM):
                    sg = rng3.choice(
                        (-1.0, 1.0), size=nL)
                    if abs(float(np.median(
                            recM * sg))) >= obs:
                        cnt += 1
                p_rec = (cnt + 1) / (N_PERM + 1)
            T2a = {
                'n_logic': nL,
                'l3': L3_GATED,
                'med_js_zero': round(
                    float(np.median(vZ)), 6)
                if nL else None,
                'med_js_mean': round(
                    float(np.median(vM)), 6)
                if nL else None,
                'med_js_rand': round(
                    float(np.median(vR)), 6)
                if nL else None,
                'med_recov_mean': round(med_recM, 4)
                if nL else None,
                'med_recov_rand': round(med_recR, 4)
                if nL else None,
                'p_recov_mean': round(p_rec, 5)
                if p_rec is not None else None,
                'med_js_sham_zero': round(
                    float(np.median(shamZ)), 6)
                if shamZ else None,
                'med_js_sham_mean': round(
                    float(np.median(shamM)), 6)
                if shamM else None,
                'med_js_sham_rand': round(
                    float(np.median(shamR)), 6)
                if shamR else None,
                'med_js_content_mean': round(
                    float(np.median(contentM)), 6)
                if contentM else None,
                'n_content': len(contentM),
                'n_sham': len(shamZ),
                'min_pos_gate': MIN_POS,
                'recov_per_pos': [round(float(x), 4)
                                  for x in recM],
                'tags': tags}
            log('T2a nL=%d zero=%.4f mean=%.4f '
                'rand=%.4f recM=%.3f recR=%.3f '
                'p=%.4f'
                % (nL, T2a['med_js_zero'],
                   T2a['med_js_mean'],
                   T2a['med_js_rand'],
                   med_recM or -1, med_recR or -1,
                   p_rec or -1), lines)

            # ---------- T2b verdict gates ----------
            T2b = {'recov_info_gate': RECOV_INFO,
                   'recov_rand_max': RECOV_RAND_MAX,
                   'gates_ok': gates_ok}
            if not gates_ok:
                verdict = 'gate_mixed_qwen'
            else:
                info = bool(med_recM >= RECOV_INFO
                            and p_rec < P_GATE
                            and med_recR
                            < RECOV_RAND_MAX)
                cap_ = bool(med_recR >= RECOV_INFO
                            and med_recR >= med_recM)
                if info:
                    verdict = \
                        'info_carrying_gate_qwen'
                elif cap_:
                    verdict = 'capacity_gate_qwen'
                else:
                    verdict = 'gate_mixed_qwen'

            # ---------- T2c dose (descriptive) ----------
            T2c = {'per_scale': {}}
            for s in SCALE_DOSE:
                rd = []
                for pi, pr in enumerate(GEN_PROMPTS):
                    ent = sel['P%d' % pi]
                    if ent.get('skipped'):
                        continue
                    pos = ent['positions']
                    pb = base_probs[pi]
                    for p_pos in pos['logic']:
                        qz, _ = \
                            prefill_logits_surgery(
                                pr, int(p_pos),
                                'zero', s=s)
                        qm, _ = \
                            prefill_logits_surgery(
                                pr, int(p_pos),
                                'mean', s=s)
                        jz = js_nats(pb, qz)
                        jm = js_nats(pb, qm)
                        if jz > 0:
                            rd.append(
                                1.0 - jm / jz)
                T2c['per_scale']['%.2f' % s] = {
                    'med_recov_mean': round(
                        float(np.median(rd)), 4)
                    if rd else None,
                    'n': len(rd)}
                log('T2c s=%.2f recM=%.3f n=%d'
                    % (s, T2c['per_scale']
                       ['%.2f' % s]
                       ['med_recov_mean'] or -1,
                       len(rd)), lines)

            # ---------- T3 drift (descriptive) ----------
            drift_w = []
            early = []
            late = []
            for pi in range(len(GEN_PROMPTS)):
                rec = recs[pi]
                drift_w.append(abs(float(
                    rec['s'][K_GEN - 1]
                    - rec['s_pre'])))
                early.append(float(
                    np.std(rec['s'][:32])))
                late.append(float(
                    np.std(rec['s'][224:])))
            T3 = {
                'med_drift_s_end': drift_med,
                'med_std_early': round(float(
                    np.median(early)), 4),
                'med_std_late': round(float(
                    np.median(late)), 4),
                'a13_vs_3009_diff':
                    round(a13_diff, 6),
                'note': 'descriptive vs 3009 '
                        '(med_drift_end 49.5123)'}
            log('T3 drift s=%.4f std e=%.3f l=%.3f'
                % (T3['med_drift_s_end'],
                   T3['med_std_early'],
                   T3['med_std_late']), lines)
    else:
        pass
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0
    anchors = {
        'a0_2993': a0_ok, 'a1_axis_diff': a1_diff,
        'a2_diff': a2_diff, 'a3_diff': a3_diff,
        'a4_diff': a4_diff, 'a5_diff': a5_diff,
        'a6_rel': a6_rel, 'a7_diff': a7_diff,
        'a8_lwords_single': a8_ok, 'a9_3007': a9_ok,
        'a10_gen_det': {'rel': a10_rel, 'ok': a10_ok},
        'a11_3008': a11_ok, 'a12_3009': a12_ok,
        'a13_t3_drift_diff': a13_diff,
        'a14_3010': a14_ok, 'a15_3011': a15_ok,
    }
    res = {
        'phase': 3012,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed in kv_refill RAND arm - '
            'KV cache is BFloat16 and numpy() does '
            'not support that ScalarType; rand '
            'noise source moved from numpy to '
            'torch.Generator (seed semantics '
            'unchanged: per-position draw with '
            'SEED_RND+50/+60+p; PREREG said numpy '
            'seeded, registered here as the only '
            'deviation, no verdict-bearing design '
            'change); MEAN arm made dtype-explicit; '
            'run2: authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35, 'w2': w2_93, 'w1024': w1024_93,
            'l_words': np.array(l_words, dtype=object),
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object),
            'rowsZ': np.array(rowsZ) if rowsZ
                     else np.zeros(0),
            'rowsM': np.array(rowsM) if rowsM
                     else np.zeros(0),
            'rowsR': np.array(rowsR) if rowsR
                     else np.zeros(0),
            'tags': np.array(tags, dtype=object),
            'shamZ': np.array(shamZ) if shamZ
                     else np.zeros(0),
            'shamM': np.array(shamM) if shamM
                     else np.zeros(0),
            'shamR': np.array(shamR) if shamR
                     else np.zeros(0),
            'contentM': np.array(contentM)
                        if contentM else np.zeros(0)}
    npz_path = os.path.join(
        OUT, 'omega_p2f_gate_surgery_qwen.npz')
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
