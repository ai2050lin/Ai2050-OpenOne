# -*- coding: utf-8 -*-
"""Phase 3013: Omega-P2g logic-position KV content
decomposition (qwen).

Why: 3012 (gate_mixed_qwen) showed the L3 logic-position
gate is NOT statistically replaceable - per-dim MEAN over
mixed other positions does not restore the distribution
(recov -0.026), while scale-matched noise restores ~25%.
Open question: WHAT content gates the distribution - a
SHARED logic-class component (across prompts / positions)
or position-specific distributed content?  3012's MEAN
mixed logic+content positions, which would dilute any
shared logic component; this phase separates them.

Design (3012 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol):
  T1 capture: one clean prefill per prompt; store L3 K,V
      (flattened over heads) at every logic / content /
      sham position selected by the 3009 chain.
  T2a PRIMARY logic-pool mean refill: K,V at logic
      position p replaced by the per-dim mean over ALL
      logic positions from ALL prompts (cross-prompt
      shared component, leave-p-out within the pool);
      paired recovery recov = 1 - JS_logic-mean /
      JS_zero vs 3012's mixed MEAN (re-run) and ZERO and
      RAND (both verbatim).  PRIMARY stat = med recov
      over logic positions, sign-flip permutation.
  T2b LOO rank-k refill: K,V at logic p replaced by
      their own top-k projection under a PCA fit on the
      OTHER logic positions (leave-p-out; k in
      (1,2,4,8)); k* = smallest k with med recov >= 0.5.
      Full-rank LOO reconstruction is the identity, so
      k* <= n_pool-1 always exists.
  T2c descriptive: per-position cosine of logic K (flat)
      to the pool mean vs content K to its pool mean -
      is the logic class tighter than content?
  T3 drift (descriptive, a13 anchored) verbatim.

Verdict (frozen):
  anchor fail                            => anchor_fail_
                                              all_void
  med recov_logic>=0.5 AND p<0.05        => logic_shared_
                                              component_qwen
  med recov_logic>=0.25                  => partial_shared_
                                              gate_qwen
  else                                   => position_specific_
                                              gate_qwen

Anchors (frozen): a0-a15 as 3012 verbatim, plus
  a16 3012 integrity: seal match AND verdict ==
      gate_mixed_qwen AND anchors ok.

Tags: Omega-P2g / KV content decomposition / shared-vs-
position-specific / LOO rank-k refill / paired sign-flip
permutation / no hallucination naming.
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
D_3012 = os.path.join(BASE, 'phase3012',
                      'omega_p2f_gate_surgery_qwen')
OUT = os.path.join(BASE, 'phase3013',
                   'omega_p2g_kv_content_decomposition_'
                   'qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002..3012 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009..3012
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
L3_GATED = 3
RANK_GRID = (1, 2, 4, 8)
RECOV_SHARED = 0.5
RECOV_PARTIAL = 0.25
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
    'mode': 'qwen3-4b; 3012 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by logic-pool refill + '
            'LOO rank-k decomposition',
    'question': 'WHAT content at the L3 logic-position KV '
                'gates the next-token distribution (3012 '
                'showed mixed-position mean does not '
                'restore it) - a SHARED logic-class '
                'component across prompts/positions, or '
                'position-specific distributed content?',
    'T1': 'capture: one clean prefill per prompt; store '
          'L3 K,V flattened over heads at every logic / '
          'content / sham position of the 3009 chain',
    'T2a': 'PRIMARY: per logic position p, arms - ZERO '
           '(K,V scaled 0.0, verbatim), MEAN_MIXED (K,V '
           'replaced by per-dim mean over same-prompt '
           'other positions, 3012 verbatim re-run), '
           'MEAN_LOGIC (K,V replaced by per-dim mean over '
           'ALL logic positions from ALL prompts, '
           'leave-p-out), RAND (3012 verbatim); two-step '
           'protocol both arms identical; JS nats vs '
           'unintervened two-step baseline; paired '
           'recovery recov_i = 1 - JS_arm/JS_zero '
           '(JS_zero>0 gate); PRIMARY stat = med recov '
           'over logic positions for MEAN_LOGIC, '
           'sign-flip permutation N=10000 two-sided; '
           'gates nL>=8 and per-pos JS_zero>0',
    'T2b': 'LOO rank-k refill: K,V at logic p replaced '
           'by their own top-k projection under PCA fit '
           'on the OTHER logic positions of the pool '
           '(leave-p-out; k in (1,2,4,8)); k* = smallest '
           'k with med recov >= 0.5; full-rank LOO is '
           'the identity so k* <= n_pool-1 exists',
    'T2c': 'DESCRIPTIVE: per-position cosine of logic K '
           '(flat) to the logic-pool mean vs content K '
           'to the content-pool mean - class tightness '
           'contrast',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'med recov_logic>=0.5 AND p<0.05 => '
               'logic_shared_component_qwen; med '
               'recov_logic>=0.25 => partial_shared_'
               'gate_qwen; else => position_specific_'
               'gate_qwen',
    'tags': 'Omega-P2g / KV content decomposition / '
            'shared-vs-position-specific / LOO rank-k '
            'refill / paired sign-flip permutation / no '
            'hallucination naming',
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
        json.dump({'phase': 3013,
                   'name': 'omega_p2g_kv_content_'
                           'decomposition_qwen',
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
                           D_3011 + r'\seal.json'),
                       's3012result': sha8(
                           D_3012 + r'\result.json'),
                       's3012seal': sha8(
                           D_3012 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'rank_grid': list(RANK_GRID),
                   'recov_shared': RECOV_SHARED,
                   'recov_partial': RECOV_PARTIAL,
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

    seal12 = json.load(open(D_3012 + r'\seal.json',
                            encoding='utf-8'))
    r12 = json.load(open(D_3012 + r'\result.json',
                         encoding='utf-8'))
    a16_ok = bool(seal12['result_sha256_8']
                  == sha8(D_3012 + r'\result.json')
                  and r12['final_verdict']
                  == 'gate_mixed_qwen'
                  and r12['anchor_all_ok'] is True)
    log('a16 3012 integrity %s (verdict=%s)'
        % (a16_ok, r12['final_verdict']), lines)

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

    # a7 xdir identity (3012 verbatim, S_IDX 0/1/4)
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

    def kv_replace(past, p, k_vec, v_vec, li=L3_GATED):
        L = past.layers[li]
        L.keys[:, :, p, :] = k_vec.view(
            L.keys[:, :, p, :].shape)
        L.values[:, :, p, :] = v_vec.view(
            L.values[:, :, p, :].shape)

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

    def prefill_capture(prompt):
        """Clean prefill; returns (ids, probs, L3 K/V
        flattened over heads at every position)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            L = past.layers[L3_GATED]
            kf = L.keys[0].detach() \
                .to(torch.float64).cpu().numpy()
            vf = L.values[0].detach() \
                .to(torch.float64).cpu().numpy()
            T = int(kf.shape[1])
            kflat = kf.transpose(1, 0, 2).reshape(
                T, -1)
            vflat = vf.transpose(1, 0, 2).reshape(
                T, -1)
            p, am = prefill_step2(prompt, ids, past)
        return ids, p, am, kflat, vflat

    def prefill_surgery(prompt, p_pos, k_new, v_new):
        """Two-step protocol with K,V at p_pos replaced
        (k_new/v_new already on-device tensors, or None
        pair meaning zero-scale)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if k_new is None:
                kv_scale(past, p_pos, 0.0)
            else:
                kv_replace(past, p_pos, k_new, v_new)
            q, am = prefill_step2(prompt, ids, past)
        return q, am

    def classify2(t_id, dec_cache):
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
                         and a14_ok and a15_ok and a16_ok)
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
            # (3009..3012 protocol verbatim,
            #  seed chain SEED_RND+20)
            rng2 = np.random.default_rng(
                SEED_RND + 20)
            sel = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ids = list(recs[pi]['prompt_ids'])
                lp = [i for i, t in enumerate(ids)
                      if int(t) in logic_tids]
                dec_cache = {}
                cp = [i for i, t in enumerate(ids)
                      if classify2(int(t),
                                   dec_cache)
                      == 'content']
                sp = [i for i, t in enumerate(ids)
                      if classify2(int(t),
                                   dec_cache)
                      in ('func', 'other')]
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

            # ---------- T1 capture ----------
            caps = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                caps[pi] = prefill_capture(pr)
                log('T1 P%d captured' % pi, lines)

            # logic-pool (cross-prompt) vectors
            pool_k = []
            pool_v = []
            tags = []
            for pi, (ids, pb, am, kf, vf) \
                    in caps.items():
                for p_pos in sel['P%d' % pi][
                        'positions']['logic']:
                    pool_k.append(kf[p_pos])
                    pool_v.append(vf[p_pos])
                    tags.append('P%d:%d'
                                % (pi, p_pos))
            pool_k = np.array(pool_k)
            pool_v = np.array(pool_v)
            nL = len(tags)
            d_kv = pool_k.shape[1]
            log('T1 pool n=%d kvdim=%d'
                % (nL, d_kv), lines)
            # content pool (for T2c tightness)
            cpool_k = []
            cpool_v = []
            for pi, (ids, pb, am, kf, vf) \
                    in caps.items():
                for p_pos in sel['P%d' % pi][
                        'positions']['content']:
                    cpool_k.append(kf[p_pos])
                    cpool_v.append(vf[p_pos])
            cpool_k = np.array(cpool_k)
            cpool_v = np.array(cpool_v)

            def refill_vecs(past, p, mode,
                            ki=None):
                """Build replacement K/V tensors for
                position p of the current prefill.
                mode: mixed-mean (3012), pool-mean
                (leave-p-out logic pool), rand (3012)."""
                L = past.layers[L3_GATED]
                T = int(L.keys.shape[2])
                others = [j for j in range(T)
                          if j != p]
                ko = L.keys[:, :, others, :]
                vo = L.values[:, :, others, :]
                dev = L.keys.device
                dt = L.keys.dtype
                if mode == 'mixed_mean':
                    mk = ko.mean(dim=2)
                    mv = vo.mean(dim=2)
                elif mode == 'pool_mean':
                    Xk = np.delete(pool_k, ki,
                                   axis=0)
                    Xv = np.delete(pool_v, ki,
                                   axis=0)
                    mk = torch.tensor(
                        Xk.mean(0), device=dev)
                    mv = torch.tensor(
                        Xv.mean(0), device=dev)
                elif mode == 'rand':
                    mu_k = ko.mean(dim=2).float()
                    sd_k = ko.std(dim=2).float()
                    mu_v = vo.mean(dim=2).float()
                    sd_v = vo.std(dim=2).float()
                    shape_k = tuple(
                        L.keys[:, :, p, :].shape)
                    shape_v = tuple(
                        L.values[:, :, p, :].shape)
                    gk = torch.Generator()
                    gk.manual_seed(SEED_RND + 50
                                   + int(p))
                    gv = torch.Generator()
                    gv.manual_seed(SEED_RND + 60
                                   + int(p))
                    nk = torch.randn(
                        shape_k, generator=gk) \
                        .to(dev)
                    nv = torch.randn(
                        shape_v, generator=gv) \
                        .to(dev)
                    mk = mu_k + sd_k * nk
                    mv = mu_v + sd_v * nv
                return mk.to(dt), mv.to(dt)

            # ---------- T2a refill contrast ----------
            rowsZ = []
            rowsMix = []
            rowsPool = []
            rowsR = []
            shamZ = []
            shamMix = []
            contentMix = []
            for ki, (pi, p_pos) in enumerate(
                    [(int(t.split(':')[0][1:]),
                      int(t.split(':')[1]))
                     for t in tags]):
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                qz, _ = prefill_surgery(
                    pr, p_pos, None, None)
                jz = js_nats(pb, qz)
                # mixed-mean needs the live prefill:
                ids = tok(pr,
                          add_special_tokens=False)[
                    'input_ids']
                clear_cap()
                with torch.no_grad():
                    out = model(torch.tensor(
                        [ids], device='cuda'),
                        use_cache=True)
                    past = out.past_key_values
                    mk, mv = refill_vecs(
                        past, p_pos, 'mixed_mean')
                qm, _ = prefill_surgery(
                    pr, p_pos, mk, mv)
                mk2, mv2 = None, None
                clear_cap()
                with torch.no_grad():
                    out = model(torch.tensor(
                        [ids], device='cuda'),
                        use_cache=True)
                    past = out.past_key_values
                    mk2, mv2 = refill_vecs(
                        past, p_pos, 'pool_mean',
                        ki=ki)
                qp, _ = prefill_surgery(
                    pr, p_pos, mk2, mv2)
                clear_cap()
                with torch.no_grad():
                    out = model(torch.tensor(
                        [ids], device='cuda'),
                        use_cache=True)
                    past = out.past_key_values
                    mk3, mv3 = refill_vecs(
                        past, p_pos, 'rand')
                qr, _ = prefill_surgery(
                    pr, p_pos, mk3, mv3)
                if jz > 0:
                    rowsZ.append(jz)
                    rowsMix.append(js_nats(pb, qm))
                    rowsPool.append(js_nats(pb, qp))
                    rowsR.append(js_nats(pb, qr))
                log('T2a ki=%d (%s) done jz=%.4f'
                    % (ki, tags[ki], jz), lines)

            # sham + content calibration arms
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                pos = ent['positions']
                pb = caps[pi][1]
                ids = tok(pr,
                          add_special_tokens=False)[
                    'input_ids']
                if pos['sham'] is not None:
                    p_sh = int(pos['sham'])
                    clear_cap()
                    with torch.no_grad():
                        out = model(torch.tensor(
                            [ids], device='cuda'),
                            use_cache=True)
                        past = out.past_key_values
                        mk, mv = refill_vecs(
                            past, p_sh,
                            'mixed_mean')
                    qz, _ = prefill_surgery(
                        pr, p_sh, None, None)
                    qm, _ = prefill_surgery(
                        pr, p_sh, mk, mv)
                    shamZ.append(js_nats(pb, qz))
                    shamMix.append(js_nats(pb, qm))
                for p_pos in pos['content']:
                    clear_cap()
                    with torch.no_grad():
                        out = model(torch.tensor(
                            [ids], device='cuda'),
                            use_cache=True)
                        past = out.past_key_values
                        mk, mv = refill_vecs(
                            past, int(p_pos),
                            'mixed_mean')
                    qm, _ = prefill_surgery(
                        pr, int(p_pos), mk, mv)
                    contentMix.append(
                        js_nats(pb, qm))

            vZ = np.array(rowsZ)
            vMix = np.array(rowsMix)
            vPool = np.array(rowsPool)
            vR = np.array(rowsR)
            recPool = 1.0 - vPool / vZ
            recMix = 1.0 - vMix / vZ
            recR = 1.0 - vR / vZ
            med_recPool = float(np.median(recPool)) \
                if nL else None
            med_recMix = float(np.median(recMix)) \
                if nL else None
            med_recR = float(np.median(recR)) \
                if nL else None
            p_rec = None
            if nL:
                rng3 = np.random.default_rng(
                    SEED_RND + 70)
                obs = abs(med_recPool)
                cnt = 0
                for _ in range(N_PERM):
                    sg = rng3.choice(
                        (-1.0, 1.0), size=nL)
                    if abs(float(np.median(
                            recPool * sg))) >= obs:
                        cnt += 1
                p_rec = (cnt + 1) / (N_PERM + 1)
            gates_ok = bool(nL >= MIN_POS)
            T2a = {
                'n_logic': nL,
                'l3': L3_GATED,
                'kv_dim': int(d_kv),
                'med_js_zero': round(
                    float(np.median(vZ)), 6)
                if nL else None,
                'med_js_mixed': round(
                    float(np.median(vMix)), 6)
                if nL else None,
                'med_js_pool': round(
                    float(np.median(vPool)), 6)
                if nL else None,
                'med_js_rand': round(
                    float(np.median(vR)), 6)
                if nL else None,
                'med_recov_pool': round(med_recPool, 4)
                if nL else None,
                'med_recov_mixed': round(med_recMix, 4)
                if nL else None,
                'med_recov_rand': round(med_recR, 4)
                if nL else None,
                'p_recov_pool': round(p_rec, 5)
                if p_rec is not None else None,
                'med_js_sham_zero': round(
                    float(np.median(shamZ)), 6)
                if shamZ else None,
                'med_js_sham_mixed': round(
                    float(np.median(shamMix)), 6)
                if shamMix else None,
                'med_js_content_mixed': round(
                    float(np.median(contentMix)), 6)
                if contentMix else None,
                'n_content': len(contentMix),
                'n_sham': len(shamZ),
                'min_pos_gate': MIN_POS,
                'recov_pool_per_pos': [round(
                    float(x), 4) for x in recPool],
                'tags': tags}
            log('T2a nL=%d zero=%.4f mixed=%.4f '
                'pool=%.4f rand=%.4f recP=%.3f '
                'recM=%.3f recR=%.3f p=%.4f'
                % (nL, T2a['med_js_zero'],
                   T2a['med_js_mixed'],
                   T2a['med_js_pool'],
                   T2a['med_js_rand'],
                   med_recPool or -1,
                   med_recMix or -1,
                   med_recR or -1,
                   p_rec or -1), lines)

            # ---------- T2b LOO rank-k ----------
            per_k = {k: [] for k in RANK_GRID}
            for ki in range(nL):
                pi = int(tags[ki].split(':')[0][1:])
                p_pos = int(tags[ki].split(':')[1])
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                qz, _ = prefill_surgery(
                    pr, p_pos, None, None)
                jz = js_nats(pb, qz)
                Xk = np.delete(pool_k, ki, axis=0)
                Xv = np.delete(pool_v, ki, axis=0)
                mu_k = Xk.mean(0)
                mu_v = Xv.mean(0)
                _, _, Vtk = np.linalg.svd(
                    Xk - mu_k, full_matrices=False)
                _, _, Vtv = np.linalg.svd(
                    Xv - mu_v, full_matrices=False)
                ids = tok(pr,
                          add_special_tokens=False)[
                    'input_ids']
                for k in RANK_GRID:
                    Uk = Vtk[:k]
                    Uv = Vtv[:k]
                    vk_hat = mu_k \
                        + (pool_k[ki] - mu_k) \
                        @ Uk.T @ Uk
                    vv_hat = mu_v \
                        + (pool_v[ki] - mu_v) \
                        @ Uv.T @ Uv
                    clear_cap()
                    with torch.no_grad():
                        out = model(torch.tensor(
                            [ids], device='cuda'),
                            use_cache=True)
                        past = out.past_key_values
                        L = past.layers[L3_GATED]
                        dev = L.keys.device
                        kt = torch.tensor(
                            vk_hat, device=dev)
                        vt = torch.tensor(
                            vv_hat, device=dev)
                    qk, _ = prefill_surgery(
                        pr, p_pos,
                        kt.view(L.keys[:, :,
                                       p_pos, :]
                                .shape).to(
                            L.keys.dtype),
                        vt.view(L.values[:, :,
                                         p_pos, :]
                                .shape).to(
                            L.values.dtype))
                    jk = js_nats(pb, qk)
                    per_k[k].append(
                        1.0 - jk / jz if jz > 0
                        else np.nan)
                log('T2b ki=%d (%s) done'
                    % (ki, tags[ki]), lines)
            T2b = {'per_k': {}}
            k_star = None
            for k in RANK_GRID:
                rk = np.array(per_k[k])
                rk = rk[~np.isnan(rk)]
                med = float(np.median(rk)) \
                    if len(rk) else None
                T2b['per_k'][str(k)] = {
                    'med_recov': round(med, 4)
                    if med is not None else None,
                    'n': len(rk)}
                if k_star is None and med \
                        is not None \
                        and med >= 0.5:
                    k_star = k
                log('T2b k=%d med=%.3f n=%d'
                    % (k, med if med is not None
                       else -1, len(rk)), lines)
            T2b['k_star'] = k_star
            T2b['n_pool'] = nL

            # ---------- T2c tightness ----------
            if nL and len(cpool_k) >= 2:
                cm_k = pool_k.mean(0)
                cc_k = cpool_k.mean(0)
                cos_l = [float(x @ cm_k)
                         / max(float(
                             np.linalg.norm(x)
                             * np.linalg.norm(cm_k)),
                             1e-30)
                         for x in pool_k]
                cos_c = [float(x @ cc_k)
                         / max(float(
                             np.linalg.norm(x)
                             * np.linalg.norm(cc_k)),
                             1e-30)
                         for x in cpool_k]
                T2c = {'med_cos_logic_to_pool':
                       round(float(np.median(cos_l)), 4),
                       'med_cos_content_to_pool':
                       round(float(np.median(cos_c)), 4),
                       'n_logic': nL,
                       'n_content': len(cos_c)}
                log('T2c cos logic=%.3f content=%.3f'
                    % (T2c['med_cos_logic_to_pool'],
                       T2c['med_cos_content_to_pool']),
                    lines)
            else:
                T2c = None

            # ---------- verdict ----------
            if not gates_ok:
                verdict = 'position_specific_' \
                          'gate_qwen'
            elif (med_recPool is not None
                  and med_recPool >= RECOV_SHARED
                  and p_rec < P_GATE):
                verdict = 'logic_shared_component_qwen'
            elif (med_recPool is not None
                  and med_recPool >= RECOV_PARTIAL):
                verdict = 'partial_shared_gate_qwen'
            else:
                verdict = 'position_specific_' \
                          'gate_qwen'

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
        'a16_3012': a16_ok,
    }
    res = {
        'phase': 3013,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at the T2a tag parse - tags '
            'are P<prompt>:<pos> and int() was called '
            'on the P-prefixed prompt field; parse fixed '
            'to strip the P prefix (T2a + T2b, no '
            'design change, crash was before any T2 '
            'computation; no verdict-bearing '
            'deviation from PREREG); run2: '
            'crashed at the pool_mean arm - the '
            'kv_replace view fix was a phantom '
            'edit (reported success, absent on '
            'disk), 1-D flat pool vector hit '
            'expanded-size mismatch [1,8,128] vs '
            '[1024]; patch reapplied via direct '
            'disk write; no design change; '
            'run3: authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35, 'w2': w2_93, 'w1024': w1024_93,
            'l_words': np.array(l_words, dtype=object),
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object),
            'pool_k': pool_k, 'pool_v': pool_v,
            'cpool_k': cpool_k, 'cpool_v': cpool_v,
            'tags': np.array(tags, dtype=object),
            'rowsZ': vZ, 'rowsMix': vMix,
            'rowsPool': vPool, 'rowsR': vR}
    npz_path = os.path.join(
        OUT, 'omega_p2g_kv_content_decomposition_'
             'qwen.npz')
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
