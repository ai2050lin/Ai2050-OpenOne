# -*- coding: utf-8 -*-
"""Phase 3015: Omega-P2i K-consumer head localization
(qwen).

Why: 3014 (reverse dose law) established the L3
logic-position gate destruction is carried by the K
routing channel - KONLY erasure is non-monotonic in the
dose (softmax competition) while VONLY decays linearly,
and half-strength K is more destructive than zero K.
Open question: WHO consumes the destroyed K?  The L3
key cache at position p is read only by layer-3's own
32 query heads (positions > p).  A per-head K erasure
therefore maps the functional consumers exactly.

Design (3014 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol,
SEED_RND=3009 chain):
  T2a PRIMARY per-head K erasure: at every logic
      position p (3009 chain), zero L3 K of head h
      alone, for all 32 heads; two-step protocol
      identical; JS nats vs the unintervened two-step
      baseline; share(h,p) = JS_h(p)/JS_all(p) where
      JS_all = full-K erasure (all heads, s=0) at p;
      PRIMARY stat = top1_med = med over positions of
      max_h share(h,p); permutation null: within each
      position permute the 32 share values, N=10000.
  T2b descriptive: per-head dose curves for the top-4
      impact heads (s in 0.0/0.25/0.5/0.75, retain vs
      own s=0); V-side erasure of the same top-4 heads
      (K-specificity contrast).
  T2c descriptive calibration: full-K erasure at sham
      positions and content positions (expect ~0 as
      3011/3014).
  T2d descriptive attention readout (best-effort,
      non-gating): step-2 attention of layer 3 per
      head - attention mass on p per head, softmax
      entropy per head, baseline vs full-K erasure;
      Spearman rho between per-head impact and
      per-head attention-on-p.  If the installed
      transformers cannot return attentions, T2d
      fields are null and the verdict is unaffected.
  T3 drift (descriptive, a13 anchored) verbatim.

Verdict (frozen; top1_med = median maximal head share
of the K-routing effect):
  anchor fail                              => anchor_
                                              fail_
                                              all_void
  gates fail (nL<8 or any JS_all<=0)       => k_
                                              consumer_
                                              undetermined_
                                              void
  top1_med >= 0.5 AND p_perm <= 0.05       => k_
                                              consumer_
                                              head_
                                              concentrated_
                                              qwen
  top1_med <= 0.2                          => k_
                                              consumer_
                                              distributed_
                                              qwen
  else                                     => k_
                                              consumer_
                                              mixed_qwen

Anchors (frozen): a0-a17 as 3014 verbatim, plus
  a18 3014 integrity: seal match AND verdict ==
      gate_destruction_fragile_qwen AND anchors ok.

Tags: Omega-P2i / K consumer head localization /
per-head K erasure / concentration permutation null /
attention readout descriptive / no hallucination
naming.
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
D_3014 = os.path.join(BASE, 'phase3014',
                      'omega_p2h_reverse_dose_law_qwen')
OUT = os.path.join(BASE, 'phase3015',
                   'omega_p2i_k_consumer_heads_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002..3014 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009..3014
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
L3_GATED = 3
NH_EXPECT = 8   # GQA num KV heads: 32 query heads
                # share 8 KV heads (kv head g
                # serves query heads 4g..4g+3)
SHARE_CONC = 0.5
SHARE_DIST = 0.2
DOSE_GRID = (0.0, 0.25, 0.5, 0.75)
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
    'mode': 'qwen3-4b; 3014 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by per-head L3 K '
            'erasure (functional consumer map)',
    'question': 'WHO consumes the destroyed L3 logic-'
                'position K (3014: destruction is carried '
                'by the K routing channel)?  Per-head K '
                'erasure maps the layer-3 query heads '
                'that functionally read position p; is '
                'the K consumption head-concentrated or '
                'distributed, and does impact track '
                'attention mass on p?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain',
    'T2a': 'PRIMARY: at every logic position p, zero L3 '
           'K of head h alone for all 32 heads; two-step '
           'protocol identical; JS nats vs unintervened '
           'two-step baseline; share(h,p) = '
           'JS_h(p)/JS_all(p) with JS_all = full-K '
           'erasure (all heads, s=0) at p; PRIMARY stat '
           '= top1_med = med over positions of max_h '
           'share(h,p); permutation null within position '
           '(permute 32 shares), N=10000 one-sided; '
           'gates nL>=8 and all JS_all>0',
    'T2b': 'DESCRIPTIVE: per-head dose curves for top-4 '
           'impact heads, s in (0.0,0.25,0.5,0.75), '
           'retain_h(s) = JS_h(s)/JS_h(0); V-side '
           'erasure of the same top-4 heads at s=0 '
           '(K-specificity contrast)',
    'T2c': 'DESCRIPTIVE calibration: full-K erasure '
           '(all heads, s=0) at sham positions and '
           'content positions (expect ~0 as 3011/3014)',
    'T2d': 'DESCRIPTIVE attention readout (best-effort, '
           'non-gating): step-2 layer-3 attention per '
           'head - mass on p, softmax entropy, baseline '
           'vs full-K erasure; Spearman rho between '
           'per-head impact and attention-on-p; if '
           'attentions unavailable, fields null and '
           'verdict unaffected',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => '
               'k_consumer_undetermined_void; top1_med '
               '>= 0.5 AND p_perm <= 0.05 => '
               'k_consumer_head_concentrated_qwen; '
               'top1_med <= 0.2 => '
               'k_consumer_distributed_qwen; else => '
               'k_consumer_mixed_qwen',
    'tags': 'Omega-P2i / K consumer head localization / '
            'per-head K erasure / concentration '
            'permutation null / attention readout '
            'descriptive / no hallucination naming',
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


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx < 1e-30 or sy < 1e-30:
        return None
    return float(np.mean((rx - rx.mean())
                         * (ry - ry.mean()))
                 / (sx * sy))


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
        json.dump({'phase': 3015,
                   'name': 'omega_p2i_k_consumer_'
                           'heads_qwen',
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
                           D_3013 + r'\seal.json'),
                       's3014result': sha8(
                           D_3014 + r'\result.json'),
                       's3014seal': sha8(
                           D_3014 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'nh_expect': NH_EXPECT,
                   'share_conc': SHARE_CONC,
                   'share_dist': SHARE_DIST,
                   'dose_grid': list(DOSE_GRID),
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

    seal14 = json.load(open(D_3014 + r'\seal.json',
                            encoding='utf-8'))
    r14 = json.load(open(D_3014 + r'\result.json',
                         encoding='utf-8'))
    a18_ok = bool(seal14['result_sha256_8']
                  == sha8(D_3014 + r'\result.json')
                  and r14['final_verdict']
                  == 'gate_destruction_fragile_qwen'
                  and r14['anchor_all_ok'] is True)
    log('a18 3014 integrity %s (verdict=%s)'
        % (a18_ok, r14['final_verdict']), lines)

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

    def kv_scale_arm(past, p, s, arm, h=None,
                     li=L3_GATED):
        L = past.layers[li]
        if h is None:
            if arm in ('JOINT', 'KONLY'):
                L.keys[:, :, p, :] *= s
            if arm in ('JOINT', 'VONLY'):
                L.values[:, :, p, :] *= s
        else:
            if arm in ('JOINT', 'KONLY'):
                L.keys[:, h, p, :] *= s
            if arm in ('JOINT', 'VONLY'):
                L.values[:, h, p, :] *= s

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

    def prefill_dose(prompt, p_pos, s, arm, h=None):
        """Two-step protocol with L3 K (or V) at p_pos
        scaled by s; h=None scales all heads, h=int
        scales one head (arm: JOINT / KONLY / VONLY)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            kv_scale_arm(past, p_pos, s, arm, h)
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
                         and a17_ok and a18_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T2d = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    tags = []
    nL = 0
    top4 = []
    att_ok = False
    dose_top = {}
    v_top = {}
    js_all_a = np.array([])
    js_head_a = np.array([])
    share = np.array([])
    att_to_p = np.array([])
    att_to_p_er = np.array([])
    ent_base = np.array([])
    ent_er = np.array([])
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
            # (3009..3014 protocol verbatim,
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

            # NH check from live cache shape
            ids_chk = tok(GEN_PROMPTS[0],
                          add_special_tokens=False)[
                'input_ids']
            with torch.no_grad():
                out_chk = model(torch.tensor(
                    [ids_chk], device='cuda'),
                    use_cache=True)
            NH = int(out_chk.past_key_values
                     .layers[L3_GATED].keys.shape[1])
            del out_chk
            nh_ok = bool(NH == NH_EXPECT)
            log('NH=%d expect=%d ok=%s'
                % (NH, NH_EXPECT, nh_ok), lines)

            # ---------- T2a per-head K erasure ----------
            # js_all[ki]; js_head[h][ki]
            js_all = []
            js_head = [[] for _ in range(NH)]
            for ki, (pi, p_pos) in enumerate(
                    [(int(t.split(':')[0][1:]),
                      int(t.split(':')[1]))
                     for t in tags]):
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                q, _ = prefill_dose(pr, p_pos, 0.0,
                                    'KONLY', None)
                js_all.append(js_nats(pb, q))
                for h in range(NH):
                    q, _ = prefill_dose(pr, p_pos,
                                        0.0, 'KONLY', h)
                    js_head[h].append(js_nats(pb, q))
                log('T2a ki=%d (%s) done'
                    % (ki, tags[ki]), lines)

            js_all_a = np.array(js_all,
                                dtype=np.float64)
            js_head_a = np.array(js_head,
                                 dtype=np.float64)
            gates_ok = bool(nL >= MIN_POS
                            and nh_ok
                            and np.all(js_all_a > 0))
            share = js_head_a / js_all_a[None, :]
            top1_per_pos = share.max(axis=0)
            top1_med = float(np.median(top1_per_pos))
            arg_h = share.argmax(axis=0)
            cnt_head = {}
            for h in arg_h:
                cnt_head[int(h)] = \
                    cnt_head.get(int(h), 0) + 1
            med_share_h = np.median(share, axis=1)
            order = np.argsort(-med_share_h)
            top4 = [int(h) for h in order[:4]]
            n_eff = (share > 0.05).sum(axis=0)
            med_n_eff = float(np.median(n_eff))
            pr_part = (share.sum(axis=0) ** 2) \
                / np.maximum(
                    (share ** 2).sum(axis=0), 1e-30)
            med_pr = float(np.median(pr_part))

            # permutation null (within position)
            rng3 = np.random.default_rng(
                SEED_RND + 80)
            cnt = 0
            for _ in range(N_PERM):
                vals = []
                for ki in range(nL):
                    perm = rng3.permutation(NH)
                    vals.append(
                        float(share[perm, ki].max()))
                if float(np.median(vals)) >= top1_med:
                    cnt += 1
            p_perm = (cnt + 1) / (N_PERM + 1)

            T2a = {
                'n_logic': nL, 'nh': NH,
                'l3': L3_GATED,
                'med_js_all': round(
                    float(np.median(js_all_a)), 6)
                if nL else None,
                'top1_med_share': round(top1_med, 4),
                'p_perm': round(p_perm, 5),
                'top_head_counts': {
                    str(k): int(v) for k, v
                    in sorted(cnt_head.items())},
                'med_share_top8': {
                    str(int(h)):
                        round(float(med_share_h[h]), 4)
                    for h in order[:8]},
                'top4_heads': top4,
                'med_n_eff_005': round(med_n_eff, 1),
                'med_participation': round(med_pr, 2),
                'min_pos_gate': MIN_POS,
                'gates_ok': gates_ok,
                'tags': tags}
            log('T2a nL=%d NH=%d jsAll=%.4f top1=%.3f '
                'p=%.4f top4=%s nEff=%.1f PR=%.1f'
                % (nL, NH,
                   T2a['med_js_all'],
                   top1_med, p_perm, top4,
                   med_n_eff, med_pr), lines)

            # ---------- T2b dose curves + V contrast ----------
            dose_top = {}
            v_top = {}
            if nL and gates_ok:
                for h in top4:
                    curves = []
                    for s in DOSE_GRID[1:]:
                        col = []
                        for ki, (pi, p_pos) in \
                                enumerate(
                                    [(int(t.split(':')[0][1:]),
                                      int(t.split(':')[1]))
                                     for t in tags]):
                            pr = GEN_PROMPTS[pi]
                            pb = caps[pi][1]
                            q, _ = prefill_dose(
                                pr, p_pos, float(s),
                                'KONLY', h)
                            col.append(
                                js_nats(pb, q))
                        col = np.array(col)
                        base_h = js_head_a[h]
                        keep = base_h > 0
                        r = np.full(nL, np.nan)
                        r[keep] = col[keep] \
                            / base_h[keep]
                        curves.append(
                            [float(x) for x in r])
                    dose_top[str(h)] = curves
                    vcol = []
                    for ki, (pi, p_pos) in \
                            enumerate(
                                [(int(t.split(':')[0][1:]),
                                  int(t.split(':')[1]))
                                 for t in tags]):
                        pr = GEN_PROMPTS[pi]
                        pb = caps[pi][1]
                        q, _ = prefill_dose(
                            pr, p_pos, 0.0, 'VONLY', h)
                        vcol.append(js_nats(pb, q))
                    v_top[str(h)] = [float(x)
                                     for x in vcol]
                    log('T2b head %d done' % h, lines)
            med_dose_top = {
                h: [round(float(np.nanmedian(
                    np.array(c))), 4)
                    for c in zip(*curves)]
                for h, curves in dose_top.items()}
            med_v_top = {
                h: round(float(np.median(v)), 6)
                for h, v in v_top.items()}
            T2b = {'dose_top_med_retain': med_dose_top,
                   'med_js_v_top': med_v_top,
                   'dose_grid': list(DOSE_GRID)}
            log('T2b dose %s v %s'
                % (json.dumps(med_dose_top),
                   json.dumps(med_v_top)), lines)

            # ---------- T2c calibration ----------
            shamK = []
            contentK = []
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                pos = ent['positions']
                pb = caps[pi][1]
                if pos['sham'] is not None:
                    q, _ = prefill_dose(
                        pr, int(pos['sham']), 0.0,
                        'KONLY', None)
                    shamK.append(js_nats(pb, q))
                for p_pos in pos['content']:
                    q, _ = prefill_dose(
                        pr, int(p_pos), 0.0,
                        'KONLY', None)
                    contentK.append(js_nats(pb, q))
            T2c = {
                'med_js_sham_K': round(
                    float(np.median(shamK)), 6)
                if shamK else None,
                'med_js_content_K': round(
                    float(np.median(contentK)), 6)
                if contentK else None,
                'n_sham': len(shamK),
                'n_content': len(contentK)}
            log('T2c sham %.6f content %.6f'
                % (T2c['med_js_sham_K'] or -1,
                   T2c['med_js_content_K'] or -1),
                lines)

            # ---------- T2d attention readout ----------
            # (best-effort, non-gating)
            # capture at QUERY-head granularity (32);
            # GQA group g serves query heads 4g..4g+3
            att_ok = True
            GQA_G = NH_EXPECT            # 8 KV heads
            GQA_Q = 32
            GQA_GRP = GQA_Q // GQA_G     # 4
            att_to_p = np.full((GQA_Q, nL), np.nan)
            ent_base = np.full((nL, GQA_Q), np.nan)
            att_to_p_er = np.full((GQA_Q, nL), np.nan)
            ent_er = np.full((nL, GQA_Q), np.nan)
            try:
                for ki, (pi, p_pos) in enumerate(
                        [(int(t.split(':')[0][1:]),
                          int(t.split(':')[1]))
                         for t in tags]):
                    pr = GEN_PROMPTS[pi]
                    ids = tok(pr,
                              add_special_tokens=False)[
                        'input_ids']
                    clear_cap()
                    with torch.no_grad():
                        out2 = model(
                            input_ids=torch.tensor(
                                [[int(ids[-1])]],
                                device='cuda'),
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=True)
                        # NOTE: out2 above is a warmup
                        # guard only if past missing;
                        # real readout below uses the
                        # proper prefill+step2 chain
                        out = model(torch.tensor(
                            [ids], device='cuda'),
                            use_cache=True,
                            output_attentions=True)
                        past = out.past_key_values
                        out2 = model(
                            input_ids=torch.tensor(
                                [[int(ids[-1])]],
                                device='cuda'),
                            past_key_values=past,
                            use_cache=False,
                            output_attentions=True)
                        a3 = out2.attentions[
                            L3_GATED][0, :, 0, :] \
                            .detach().float() \
                            .cpu().numpy()
                        # erased readout
                        out3 = model(torch.tensor(
                            [ids], device='cuda'),
                            use_cache=True)
                        past3 = out3.past_key_values
                        kv_scale_arm(past3, p_pos,
                                     0.0, 'KONLY', None)
                        out4 = model(
                            input_ids=torch.tensor(
                                [[int(ids[-1])]],
                                device='cuda'),
                            past_key_values=past3,
                            use_cache=False,
                            output_attentions=True)
                        a3e = out4.attentions[
                            L3_GATED][0, :, 0, :] \
                            .detach().float() \
                            .cpu().numpy()
                    att_to_p[:, ki] = a3[:, p_pos]
                    with np.errstate(divide='ignore',
                                     invalid='ignore'):
                        hb = -np.sum(
                            np.where(a3 > 0,
                                     a3 * np.log(a3),
                                     0.0), axis=1)
                        he = -np.sum(
                            np.where(a3e > 0,
                                     a3e * np.log(a3e),
                                     0.0), axis=1)
                    ent_base[ki] = hb
                    att_to_p_er[:, ki] = a3e[:, p_pos]
                    ent_er[ki] = he
                    log('T2d ki=%d done' % ki, lines)
            except Exception as exc:
                att_ok = False
                log('T2d unavailable: %s' % repr(exc)[:80],
                    lines)
            rho_att = None
            g_attp = None
            g_attp_er = None
            g_d_top1 = None
            top_qh = None
            if att_ok and nL:
                # GQA group aggregation: kv head g
                # serves query heads 4g..4g+3
                g_attp = att_to_p.reshape(
                    GQA_G, GQA_GRP, nL).mean(axis=1)
                g_attp_er = att_to_p_er.reshape(
                    GQA_G, GQA_GRP, nL).mean(axis=1)
                rho_att = spearman(
                    med_share_h,
                    np.nanmedian(g_attp, axis=1))
                g_d_top1 = float(np.nanmedian(
                    g_attp[top4[0]]
                    - g_attp_er[top4[0]]))
                qh_med = np.nanmedian(att_to_p,
                                      axis=1)
                qh_order = np.argsort(-qh_med)[:8]
                top_qh = {int(h):
                          round(float(qh_med[h]), 5)
                          for h in qh_order}
            T2d = {
                'att_ok': att_ok,
                'gqa_groups': GQA_G,
                'gqa_group': GQA_GRP,
                'spearman_impact_vs_attp':
                    round(rho_att, 4)
                    if rho_att is not None else None,
                'med_attp_top1_grp': round(
                    float(np.nanmedian(
                        g_attp[top4[0]])), 5)
                    if att_ok and nL else None,
                'med_attp_all_grp': round(
                    float(np.nanmedian(g_attp)), 5)
                    if att_ok and nL else None,
                'med_entropy_base': round(
                    float(np.nanmedian(ent_base)), 4)
                    if att_ok and nL else None,
                'med_entropy_erased': round(
                    float(np.nanmedian(ent_er)), 4)
                    if att_ok and nL else None,
                'med_d_attp_top1_grp': round(g_d_top1, 5)
                    if g_d_top1 is not None else None,
                'med_attp_query_head_top8': top_qh,
                'note': 'descriptive best-effort; '
                        'non-gating; impact is per-KV-'
                        'head, attention grouped by '
                        'GQA mapping'}
            log('T2d ok=%s rho=%s entB=%s entE=%s '
                'dAtp1=%s'
                % (att_ok, T2d['spearman_impact_vs_attp'],
                   T2d['med_entropy_base'],
                   T2d['med_entropy_erased'],
                   T2d['med_d_attp_top1_grp']), lines)

            # ---------- verdict ----------
            if not gates_ok:
                verdict = 'k_consumer_' \
                          'undetermined_void'
            elif (top1_med >= SHARE_CONC
                  and p_perm <= P_GATE):
                verdict = 'k_consumer_head_' \
                          'concentrated_qwen'
            elif top1_med <= SHARE_DIST:
                verdict = 'k_consumer_' \
                          'distributed_qwen'
            else:
                verdict = 'k_consumer_mixed_qwen'

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
        'a18_3014': a18_ok,
    }
    res = {
        'phase': 3015,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c,
        'T2d': T2d, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: ran to completion but verdict '
            'VOID by protocol (gates fail) - '
            'NH_EXPECT was frozen at 32 but the '
            'live L3 key cache is GQA with '
            'num_kv_heads=8 (32 query heads share '
            '8 KV heads), so nh_ok=False, T2b '
            'skipped, and T2d crashed on a '
            'broadcast (32-slot query-head '
            'attention into 8-slot arrays, caught '
            'as att_ok=False); fix: NH_EXPECT=8 '
            'with explicit GQA group mapping (kv '
            'head g serves query heads 4g..4g+3), '
            'T2d recast at query-head granularity '
            'with group aggregation; no '
            'measurement-design change (per-KV-'
            'head erasure was already the de facto '
            'unit in run1, its T2a shares were '
            'computed but voided by the gate); '
            'run2: full pass verdict '
            'k_consumer_mixed_qwen but its '
            'correction_note edit was a phantom '
            '(old text persisted on disk) - '
            're-registered here; run3: '
            'authoritative',
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
            'js_all': js_all_a if nL else np.array([]),
            'js_head': js_head_a
            if nL else np.array([]),
            'share': share if nL else np.array([]),
            'top4_heads': np.array(top4, dtype=int)
            if top4 else np.array([], dtype=int),
            'dose_top': np.array(
                [[np.array(dose_top[str(h)])[si]
                  for si in range(len(DOSE_GRID) - 1)]
                 for h in top4], dtype=np.float64)
            if dose_top else np.array([]),
            'v_top': np.array(
                [v_top[str(h)] for h in top4],
                dtype=np.float64)
            if v_top else np.array([]),
            'att_to_p': att_to_p
            if (att_ok and nL) else np.array([]),
            'att_to_p_erased': att_to_p_er
            if (att_ok and nL) else np.array([]),
            'att_grp': g_attp
            if (att_ok and nL) else np.array([]),
            'att_grp_erased': g_attp_er
            if (att_ok and nL) else np.array([]),
            'ent_base': ent_base
            if (att_ok and nL) else np.array([]),
            'ent_erased': ent_er
            if (att_ok and nL) else np.array([])}
    npz_path = os.path.join(
        OUT, 'omega_p2i_k_consumer_heads_qwen.npz')
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
