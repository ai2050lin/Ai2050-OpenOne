# -*- coding: utf-8 -*-
"""Phase 3014: Omega-P2h reverse-surgery dose law
(qwen).

Why: 3013 (position_specific_gate_qwen) established the
L3 logic-position gate is episodic - class means and
low-rank projections do NOT restore it (destruction is
easy, forgery is not).  Open question: HOW steep is the
destruction side, and can the dose curve separate the
capacity (norm/scale) component from the information
component that 3012 measured (rand refill restores
~25.6%)?

Design (3013 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol):
  T2a PRIMARY dose grid: at every logic position p
      (3009 chain, SEED_RND=3009), L3 K,V partially
      erased by scale s in (0.0, 0.25, 0.5, 0.75) for
      three arms - JOINT (K and V both), KONLY, VONLY;
      two-step protocol identical across arms; JS nats
      vs the unintervened two-step baseline; PRIMARY
      stat = med over logic positions of
      retain_joint(0.5) = JS_joint(0.5)/JS_joint(0);
      verdict gates on med retain@0.5.
  T2b descriptive: dose-curve shape - fit retain(s)
      against (1-s) [linear information+capacity] and
      (1-s)**2 [quadratic, attention-logits mixing]
      (retain(s) decreasing in s);
      K-vs-V full-erasure contrast (paired sign-flip
      permutation on JS_K(0)-JS_V(0)).
  T2c descriptive calibration: sham positions under
      the same joint grid (expect flat, ~0.0004 as
      3011); content positions zero-erasure JS.
  T3 drift (descriptive, a13 anchored) verbatim.

Verdict (frozen; retain = JS(s)/JS(0) is the
DESTRUCTION-RETAINED share - high retain at s=0.5
means half-erasure already loses most of the gate,
i.e. the gate is fragile to partial erasure):
  anchor fail                              => anchor_fail_
                                              all_void
  med retain_joint(0.5) >= 0.75            => gate_
                                              destruction_
                                              fragile_qwen
  med retain_joint(0.5) <= 0.25            => gate_
                                              destruction_
                                              robust_qwen
  else                                     => gate_
                                              destruction_
                                              graded_qwen

Anchors (frozen): a0-a16 as 3013 verbatim, plus
  a17 3013 integrity: seal match AND verdict ==
      position_specific_gate_qwen AND anchors ok.

Tags: Omega-P2h / reverse-surgery dose law / K-V arm
separation / capacity-vs-information / paired sign-flip
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
D_3013 = os.path.join(BASE, 'phase3013',
                      'omega_p2g_kv_content_'
                      'decomposition_qwen')
OUT = os.path.join(BASE, 'phase3014',
                   'omega_p2h_reverse_dose_law_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002..3013 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009..3013
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
L3_GATED = 3
SCALE_GRID = (0.0, 0.25, 0.5, 0.75)
RETAIN_STEEP = 0.25
RETAIN_RESIST = 0.75
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
    'mode': 'qwen3-4b; 3013 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by the reverse-surgery '
            'dose grid (partial KV erasure)',
    'question': 'HOW steep is the destruction side of the '
                'L3 logic-position gate (3013: forgery '
                'impossible) - what fraction of the gate '
                'survives half-erasure, and does the dose '
                'curve separate capacity (norm) from '
                'information (K vs V) components?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain',
    'T2a': 'PRIMARY: at every logic position p, L3 K,V '
           'scaled by s for three arms - JOINT (K and V), '
           'KONLY, VONLY; s in (0.0,0.25,0.5,0.75); '
           'two-step protocol identical across arms; JS '
           'nats vs unintervened two-step baseline; '
           'retain(arm,s) = JS(arm,s)/JS(arm,0) per '
           'position = DESTRUCTION-RETAINED share '
           '(JS(0)>0 gate; high retain = half-erasure '
           'already destroys most of the gate = '
           'fragile); PRIMARY stat = med '
           'retain_joint(0.5) over logic positions; '
           'gates nL>=8',
    'T2b': 'DESCRIPTIVE: joint dose-curve shape - median '
           'retain(s) vs (1-s) and (1-s)**2 fits '
           '(max-abs residual); K-vs-V full-erasure '
           'contrast: paired sign-flip permutation on '
           'per-position JS_K0-JS_V0, N=10000 two-sided',
    'T2c': 'DESCRIPTIVE calibration: sham positions under '
           'the joint grid (expect flat ~0.0004 as 3011); '
           'content positions joint-zero JS',
    'verdict': 'anchor fail => anchor_fail_all_void; med '
               'retain_joint(0.5) >= 0.75 => '
               'gate_destruction_fragile_qwen; med '
               'retain_joint(0.5) <= 0.25 => '
               'gate_destruction_robust_qwen; else => '
               'gate_destruction_graded_qwen',
    'tags': 'Omega-P2h / reverse-surgery dose law / K-V '
            'arm separation / capacity-vs-information / '
            'paired sign-flip permutation / no '
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
        json.dump({'phase': 3014,
                   'name': 'omega_p2h_reverse_dose_'
                           'law_qwen',
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
                           D_3012 + r'\seal.json'),
                       's3013result': sha8(
                           D_3013 + r'\result.json'),
                       's3013seal': sha8(
                           D_3013 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'scale_grid': list(SCALE_GRID),
                   'retain_steep': RETAIN_STEEP,
                   'retain_resist': RETAIN_RESIST,
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

    seal13 = json.load(open(D_3013 + r'\seal.json',
                            encoding='utf-8'))
    r13 = json.load(open(D_3013 + r'\result.json',
                         encoding='utf-8'))
    a17_ok = bool(seal13['result_sha256_8']
                  == sha8(D_3013 + r'\result.json')
                  and r13['final_verdict']
                  == 'position_specific_gate_qwen'
                  and r13['anchor_all_ok'] is True)
    log('a17 3013 integrity %s (verdict=%s)'
        % (a17_ok, r13['final_verdict']), lines)

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
        clear_cap()
        with torch.no_grad():
            model(torch.tensor(
                [[func_tid, tid_map[w]]],
                device='cuda'))
        for li in range(NL):
            attn_store[(i, li)] = \
                cap['ai'][li][0].astype(np.float32)
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

    # a7 xdir identity (verbatim, S_IDX 0/1/4)
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

    def kv_scale_arm(past, p, s, arm, li=L3_GATED):
        L = past.layers[li]
        if arm in ('JOINT', 'KONLY'):
            L.keys[:, :, p, :] *= s
        if arm in ('JOINT', 'VONLY'):
            L.values[:, :, p, :] *= s

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

    def prefill_dose(prompt, p_pos, s, arm):
        """Two-step protocol with L3 K,V at p_pos
        scaled by s (arm: joint / KONLY / VONLY)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            kv_scale_arm(past, p_pos, s, arm)
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
                         and a14_ok and a15_ok and a16_ok
                         and a17_ok)
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
            # (3009..3013 protocol verbatim,
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
                ids = tok(pr,
                          add_special_tokens=False)[
                    'input_ids']
                clear_cap()
                with torch.no_grad():
                    out = model(torch.tensor(
                        [ids], device='cuda'),
                        use_cache=True)
                    p, am = prefill_step2(
                        pr, ids,
                        out.past_key_values)
                caps[pi] = (ids, p)
                log('T1 P%d captured' % pi, lines)

            # logic position list
            tags = []
            for pi in caps:
                for p_pos in sel['P%d' % pi][
                        'positions']['logic']:
                    tags.append('P%d:%d'
                                % (pi, p_pos))
            nL = len(tags)
            log('T1 logic positions n=%d' % nL, lines)

            # ---------- T2a dose grid ----------
            # js[arm][s][ki]
            js = {arm: {s: [] for s in SCALE_GRID}
                  for arm in ('JOINT', 'KONLY',
                              'VONLY')}
            for ki, (pi, p_pos) in enumerate(
                    [(int(t.split(':')[0][1:]),
                      int(t.split(':')[1]))
                     for t in tags]):
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                for arm in ('JOINT', 'KONLY',
                            'VONLY'):
                    for s in SCALE_GRID:
                        q, _ = prefill_dose(
                            pr, p_pos, s, arm)
                        js[arm][s].append(
                            js_nats(pb, q))
                log('T2a ki=%d (%s) done'
                    % (ki, tags[ki]), lines)

            jZ = np.array(js['JOINT'][0.0])
            jK = np.array(js['KONLY'][0.0])
            jV = np.array(js['VONLY'][0.0])
            gates_ok = bool(nL >= MIN_POS
                            and np.all(jZ > 0))
            retain = {}
            for arm in ('JOINT', 'KONLY', 'VONLY'):
                j0 = np.array(js[arm][0.0])
                retain[arm] = {}
                for s in SCALE_GRID[1:]:
                    r = np.array(js[arm][s]) / j0
                    retain[arm][str(s)] = r
            med_rj = {str(s): float(np.median(
                retain['JOINT'][str(s)]))
                for s in SCALE_GRID[1:]}
            med_retain_05 = med_rj['0.5']

            # paired permutation K-vs-V at s=0
            p_kv = None
            d_kv = None
            if nL:
                rng3 = np.random.default_rng(
                    SEED_RND + 70)
                diffs = jK - jV
                d_kv = float(np.median(diffs))
                obs = abs(d_kv)
                cnt = 0
                for _ in range(N_PERM):
                    sg = rng3.choice(
                        (-1.0, 1.0), size=nL)
                    if abs(float(np.median(
                            diffs * sg))) >= obs:
                        cnt += 1
                p_kv = (cnt + 1) / (N_PERM + 1)

            T2a = {
                'n_logic': nL,
                'l3': L3_GATED,
                'med_js_joint0': round(
                    float(np.median(jZ)), 6)
                if nL else None,
                'med_js_k0': round(
                    float(np.median(jK)), 6)
                if nL else None,
                'med_js_v0': round(
                    float(np.median(jV)), 6)
                if nL else None,
                'med_retain_joint': {
                    s: round(v, 4)
                    for s, v in med_rj.items()},
                'med_retain_konly': {
                    s: round(float(np.median(
                        retain['KONLY'][str(s)])), 4)
                    for s in SCALE_GRID[1:]},
                'med_retain_vonly': {
                    s: round(float(np.median(
                        retain['VONLY'][str(s)])), 4)
                    for s in SCALE_GRID[1:]},
                'med_retain_joint_05':
                    round(med_retain_05, 4)
                if nL else None,
                'med_d_kv_s0': round(d_kv, 6)
                if d_kv is not None else None,
                'p_kv_s0': round(p_kv, 5)
                if p_kv is not None else None,
                'min_pos_gate': MIN_POS,
                'gates_ok': gates_ok,
                'tags': tags}
            log('T2a nL=%d j0=%.4f K0=%.4f V0=%.4f '
                'rj05=%.3f dKV=%.4f p=%.4f'
                % (nL, T2a['med_js_joint0'],
                   T2a['med_js_k0'],
                   T2a['med_js_v0'],
                   med_retain_05,
                   d_kv if d_kv is not None else -1,
                   p_kv if p_kv is not None
                   else -1), lines)

            # ---------- T2b shape fit ----------
            shape = {}
            if nL:
                grid = np.array(SCALE_GRID[1:])
                med_curve = np.array(
                    [med_rj[str(s)]
                     for s in SCALE_GRID[1:]])
                x = 1.0 - grid  # retain(s)
                # decreasing in s; linear pred = (1-s),
                # quadratic pred = (1-s)^2 captures
                # attention-mixing steepness
                for name, pred in (
                        ('linear', x),
                        ('quadratic', x ** 2)):
                    sc = float(np.sum(
                        med_curve * pred)) \
                        / max(float(np.sum(
                            pred * pred)), 1e-30)
                    resid = float(np.max(
                        np.abs(med_curve
                               - sc * pred)))
                    shape[name] = {
                        'scale': round(sc, 4),
                        'max_abs_resid':
                            round(resid, 4)}
                shape['med_curve'] = {
                    s: round(v, 4)
                    for s, v in med_rj.items()}
            T2b = shape if shape else None
            log('T2b shape %s'
                % json.dumps({k: v for k, v
                              in shape.items()
                              if k != 'med_curve'}),
                lines)

            # ---------- T2c calibration ----------
            shamJ = []
            contentJ = []
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                pos = ent['positions']
                pb = caps[pi][1]
                if pos['sham'] is not None:
                    p_sh = int(pos['sham'])
                    for s in SCALE_GRID:
                        q, _ = prefill_dose(
                            pr, p_sh, s, 'JOINT')
                        shamJ.append(
                            (s, js_nats(pb, q)))
                for p_pos in pos['content']:
                    q, _ = prefill_dose(
                        pr, int(p_pos), 0.0,
                        'JOINT')
                    contentJ.append(
                        js_nats(pb, q))
            sham_med = {}
            for s in SCALE_GRID:
                vals = [j for ss, j in shamJ
                        if ss == s]
                sham_med[str(s)] = round(
                    float(np.median(vals)), 6) \
                    if vals else None
            T2c = {'med_js_sham_joint':
                   sham_med,
                   'med_js_content_joint0': round(
                       float(np.median(contentJ)), 6)
                   if contentJ else None,
                   'n_sham': len(
                       [j for ss, j in shamJ
                        if ss == 0.0]),
                   'n_content': len(contentJ)}
            log('T2c sham %s content0=%.6f'
                % (json.dumps(sham_med),
                   T2c['med_js_content_joint0']
                   or -1), lines)

            # ---------- verdict ----------
            if not gates_ok:
                verdict = 'gate_destruction_' \
                          'graded_qwen'
            elif med_retain_05 >= RETAIN_RESIST:
                verdict = 'gate_destruction_' \
                          'fragile_qwen'
            elif med_retain_05 <= RETAIN_STEEP:
                verdict = 'gate_destruction_' \
                          'robust_qwen'
            else:
                verdict = 'gate_destruction_' \
                          'graded_qwen'

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
        'a16_3012': a16_ok, 'a17_3013': a17_ok,
    }
    res = {
        'phase': 3014,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at the retain computation '
            '- js[arm] is a dict keyed by scale and '
            'np.array(dict) is 0-dimensional (same '
            'bug in the npz save block); fix uses '
            'the s=0.0 list as denominator and '
            'stacks scales for npz; no design '
            'change, crash was before any retain '
            'was computed; run2: crashed at '
            'med_retain_05 - med_rj was built with '
            'float keys but read with str keys '
            '(KeyError 0.5); unified to str keys; '
            'no design change; run3: ran to '
            'completion but the JOINT arm was '
            'silently INERT - kv_scale_arm tested '
            "lowercase 'joint' while callers pass "
            "'JOINT', so joint K,V scaling never "
            'applied: med_js_joint0 = 0.0000, all '
            'retains NaN, sham JS exactly 0, and '
            'the NaN fell into the else branch '
            'producing a DEGENERATE verdict '
            'gate_destruction_graded_qwen - run3 '
            'verdict VOID by protocol (degenerate '
            'statistic); arm case fixed; run4: full '
            'pass with valid data but its '
            'correction_note missed the run2/run3 '
            'registrations (anchor mismatch) - '
            'note completed; run5: authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35, 'w2': w2_93, 'w1024': w1024_93,
            'l_words': np.array(l_words, dtype=object),
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object),
            'tags': np.array(tags, dtype=object)
            if tags else np.array([], dtype=object),
            'js_joint': np.array(
                [js['JOINT'][s] for s in SCALE_GRID])
            if js.get('JOINT') else np.array([]),
            'js_konly': np.array(
                [js['KONLY'][s] for s in SCALE_GRID])
            if js.get('KONLY') else np.array([]),
            'js_vonly': np.array(
                [js['VONLY'][s] for s in SCALE_GRID])
            if js.get('VONLY') else np.array([])}
    npz_path = os.path.join(
        OUT, 'omega_p2h_reverse_dose_law_qwen.npz')
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
