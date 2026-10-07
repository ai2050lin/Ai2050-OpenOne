# -*- coding: utf-8 -*-
"""Phase 3016: Omega-P2j downstream amplification trace
(qwen).

Why: 3015 (K consumer heads) established the L3 logic-
position K is consumed by a mixed economy - leading KV
head g7 (9/11 positions, med share 0.390, attention on
p 0.425) plus a sparse episodic background - and that
functional impact only moderately tracks attention mass
(rho 0.45), implying the head-level routing change is
partly amplified downstream.  Open question: WHO
amplifies the g7 K-routing destruction into the full
distribution JS - is the amplified signal carried by a
single downstream (layer, query-head) whose surgical
restoration recovers the distribution, or is it smeared
across many heads?

Design (3015 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol,
SEED_RND=3009 chain):
  T2a PRIMARY carrier restoration: per prompt and
      logic position p, run the two-step chain twice -
      baseline and g7-erased (L3 KV head 7 K zeroed at
      p, the 3015 leading consumer) - capturing at the
      step-2 query position the o_proj INPUT of every
      layer (4096 = 32 query heads x 128; head structure
      exists only on the o_proj input side); per-head
      normalized delta energy dnorm(l,qh) =
      ||o_er - o_base|| / (||o_base|| + 1e-12); carrier
      selection per position = argmax over l in 4..35,
      qh in 0..31 (selection on delta energy, labeled
      quasi-post-hoc); CAUSAL restoration test: in the
      g7-erased chain replace query head qh*'s o_proj-
      input slice at the step-2 position with the
      baseline value (in-place o_proj pre-hook surgery)
      - restoration(pos) = 1 - JS_restored/JS_g7e;
      PRIMARY stat = med restoration over positions;
      restoration is causal and NOT guaranteed by the
      selection; gates nL>=8 and all JS_g7e>0.
  T2b DESCRIPTIVE logit-lens build-up: final-norm +
      lm_head applied to the step-2 residual entering
      each layer (baseline vs g7-erased): JS_l per
      layer, profile JS_l / JS_final - immediate (L3
      output writes most of the effect) vs gradual
      (downstream growth).
  T2c DESCRIPTIVE concentration + calibration: per-
      layer top1 share and participation of the med
      dnorm across heads; per-position l* distribution;
      sham positions: g7-erasure JS (expect ~0.0002 as
      3015) and max dnorm (expect small).
  T3 drift (descriptive, a13 anchored) verbatim.

Verdict (frozen; restoration = share of the g7-erasure
JS recovered by undoing ONE carrier head's delta):
  anchor fail                              => anchor_
                                              fail_
                                              all_void
  gates fail (nL<8 or any JS_g7e<=0)       => ampli_
                                              fication_
                                              undetermined_
                                              void
  med restoration >= 0.5                   => amp_head_
                                              localized_
                                              qwen
  med restoration <= 0.2                   => amp_
                                              distributed_
                                              qwen
  else                                     => amp_mixed_
                                              qwen

Anchors (frozen): a0-a18 as 3015 verbatim, plus
  a19 3015 integrity: seal match AND verdict ==
      k_consumer_mixed_qwen AND anchors ok.

Tags: Omega-P2j / downstream amplification / carrier
restoration surgery / logit-lens build-up / quasi-post-
hoc selection labeled / no hallucination naming.
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
D_3015 = os.path.join(BASE, 'phase3015',
                      'omega_p2i_k_consumer_heads_qwen')
OUT = os.path.join(BASE, 'phase3016',
                   'omega_p2j_amplification_trace_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002..3015 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009..3015
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
L3_GATED = 3
G7_HEAD = 7               # 3015 leading consumer
HDIM = 128
NHQ = 32                  # query heads (o_proj input)
L_MIN = 4                 # carrier search from L4
REST_LOC = 0.5
REST_DIST = 0.2
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
    'mode': 'qwen3-4b; 3015 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by carrier selection '
            '(per-head o_proj-input delta energy) + '
            'causal restoration surgery + logit-lens '
            'build-up profile',
    'question': 'WHO amplifies the g7 K-routing '
                'destruction (3015 leading consumer, '
                'share 0.390, impact vs attention rho '
                '0.45) into the full distribution JS - '
                'is the amplified signal carried by a '
                'single downstream (layer, query-head) '
                'whose surgical restoration recovers '
                'the distribution, or is it smeared '
                'across many heads?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain',
    'T2a': 'PRIMARY carrier restoration: per prompt/'
           'position p, two-step chain twice - baseline '
           'and g7-erased (L3 KV head 7 K zeroed at p) '
           '- capturing at the step-2 query position '
           'the o_proj INPUT of every layer (4096 = 32 '
           'query heads x 128); dnorm(l,qh) = ||o_er - '
           'o_base|| / (||o_base|| + 1e-12); carrier = '
           'argmax over l in 4..35, qh in 0..31 '
           '(selection on delta energy, quasi-post-hoc '
           'labeled); causal restoration: in the '
           'g7-erased chain replace qh* o_proj-input '
           'slice with the baseline value (in-place '
           'o_proj pre-hook surgery); restoration = 1 - '
           'JS_restored/JS_g7e; PRIMARY stat = med '
           'restoration; restoration is causal and NOT '
           'guaranteed by selection; gates nL>=8 and '
           'all JS_g7e>0',
    'T2b': 'DESCRIPTIVE logit-lens build-up: final-norm '
           '+ lm_head on the step-2 residual entering '
           'each layer (baseline vs g7-erased): JS_l '
           'per layer, profile JS_l/JS_final - '
           'immediate vs gradual',
    'T2c': 'DESCRIPTIVE concentration + calibration: '
           'per-layer top1 share / participation of med '
           'dnorm across heads; per-position l* '
           'distribution; sham: g7-erasure JS (expect '
           '~0.0002) and max dnorm (expect small)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail (nL<8 or any JS_g7e<=0) => '
               'amplification_undetermined_void; med '
               'restoration >= 0.5 => '
               'amp_head_localized_qwen; med '
               'restoration <= 0.2 => '
               'amp_distributed_qwen; else => '
               'amp_mixed_qwen',
    'tags': 'Omega-P2j / downstream amplification / '
            'carrier restoration surgery / logit-lens '
            'build-up / quasi-post-hoc selection '
            'labeled / no hallucination naming',
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
        json.dump({'phase': 3016,
                   'name': 'omega_p2j_amplification_'
                           'trace_qwen',
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
                           D_3014 + r'\seal.json'),
                       's3015result': sha8(
                           D_3015 + r'\result.json'),
                       's3015seal': sha8(
                           D_3015 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'g7_head': G7_HEAD,
                   'hdim': HDIM, 'nhq': NHQ,
                   'l_min': L_MIN,
                   'rest_loc': REST_LOC,
                   'rest_dist': REST_DIST,
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

    seal15 = json.load(open(D_3015 + r'\seal.json',
                            encoding='utf-8'))
    r15 = json.load(open(D_3015 + r'\result.json',
                         encoding='utf-8'))
    a19_ok = bool(seal15['result_sha256_8']
                  == sha8(D_3015 + r'\result.json')
                  and r15['final_verdict']
                  == 'k_consumer_mixed_qwen'
                  and r15['anchor_all_ok'] is True)
    log('a19 3015 integrity %s (verdict=%s)'
        % (a19_ok, r15['final_verdict']), lines)

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
    o_in = {}
    state_o = {'on': False}
    state_rest = {'on': False, 'li': None, 'qh': None,
                  'vec': None}
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

    def pre_oproj(li):
        def h(module, args, kwargs):
            if state_o['on']:
                x = args[0]
                o_in.setdefault(li, []).append(
                    x[:, -1, :].detach().float()
                    .cpu().numpy().copy())
            return None
        return h

    def pre_rest(li):
        def h(module, args, kwargs):
            if (state_rest['on']
                    and state_rest['li'] == li):
                x = args[0]
                s = state_rest['qh'] * HDIM
                x[:, -1, s:s + HDIM] = \
                    state_rest['vec']
            return None
        return h

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           pre_oproj(li),
                           with_kwargs=True))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           pre_rest(li),
                           with_kwargs=True))
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

    def run_chain(pr, p_pos, erase, restore=None):
        """Two-step chain with optional g7-K erasure at
        p_pos and optional o_proj-input restoration at
        (li, qh) with baseline vector vec.  Returns
        (step2 probs, residual entering each layer at
        the step-2 position [36,2560], o_proj input per
        layer [36,4096])."""
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
        state_o['on'] = True
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if erase:
                kv_scale_arm(past, p_pos, 0.0,
                             'KONLY', G7_HEAD)
            if restore is not None:
                li, qh, vec = restore
                state_rest['on'] = True
                state_rest['li'] = li
                state_rest['qh'] = qh
                state_rest['vec'] = torch.from_numpy(
                    vec).float().cuda()
            out2 = model(
                input_ids=torch.tensor(
                    [[int(ids[-1])]], device='cuda'),
                past_key_values=past,
                use_cache=False)
            state_rest['on'] = False
        state_o['on'] = False
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        res36 = np.stack(
            [cap['ai'][li][-1][0, 0, :].astype(
                np.float64) for li in range(NL)])
        oin36 = np.stack(
            [o_in[li][-1][0].astype(np.float64)
             for li in range(NL)])
        return p_, res36, oin36

    def lens_js(res_b, res_e):
        """Logit-lens JS per layer between two residual
        stacks [36,2560]."""
        with torch.no_grad():
            hb = model.model.norm(torch.tensor(
                res_b, device='cuda')) \
                .to(model.lm_head.weight.dtype)
            he = model.model.norm(torch.tensor(
                res_e, device='cuda')) \
                .to(model.lm_head.weight.dtype)
            lb = model.lm_head(hb).double() \
                .cpu().numpy()
            le = model.lm_head(he).double() \
                .cpu().numpy()
        out = np.zeros(NL)
        for li in range(NL):
            a = lb[li] - lb[li].max()
            a = np.exp(a)
            a = a / a.sum()
            b = le[li] - le[li].max()
            b = np.exp(b)
            b = b / b.sum()
            out[li] = js_nats(a, b)
        return out

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
                         and a17_ok and a18_ok and a19_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    tags = []
    nL = 0
    lens_mat = np.array([])
    dnorm_stack = np.array([])
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
            # (3009..3015 protocol verbatim,
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

            # NHQ check from live o_proj input
            ids_chk = tok(GEN_PROMPTS[0],
                          add_special_tokens=False)[
                'input_ids']
            o_in.clear()
            state_o['on'] = True
            with torch.no_grad():
                out_chk = model(torch.tensor(
                    [ids_chk], device='cuda'),
                    use_cache=True)
            state_o['on'] = False
            ow = int(o_in[0][-1].shape[-1])
            nhq_ok = bool(ow == NHQ * HDIM)
            del out_chk
            log('o_proj input width=%d expect=%d ok=%s'
                % (ow, NHQ * HDIM, nhq_ok), lines)

            # ---------- T2a carrier restoration ----------
            rows = []
            lens_cols = []
            dnorms = []
            for ki, (pi, p_pos) in enumerate(
                    [(int(t.split(':')[0][1:]),
                      int(t.split(':')[1]))
                     for t in tags]):
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                q_b, res_b, o_b = run_chain(
                    pr, p_pos, False)
                q_e, res_e, o_e = run_chain(
                    pr, p_pos, True)
                js_e = js_nats(pb, q_e)
                dnorm = np.zeros((NL, NHQ))
                for li in range(L_MIN, NL):
                    db = o_b[li].reshape(NHQ, HDIM)
                    de = o_e[li].reshape(NHQ, HDIM)
                    num = np.linalg.norm(de - db,
                                         axis=1)
                    den = np.linalg.norm(db,
                                         axis=1) + 1e-12
                    dnorm[li] = num / den
                sub = dnorm[L_MIN:]
                idx = int(np.argmax(sub))
                li_rel, qh = np.unravel_index(
                    idx, sub.shape)
                l_star = L_MIN + int(li_rel)
                qh = int(qh)
                d_star = float(dnorm[l_star, qh])
                vec = o_b[l_star].reshape(
                    NHQ, HDIM)[qh].astype(np.float32)
                q_r, res_r, o_r = run_chain(
                    pr, p_pos, True,
                    restore=(l_star, qh, vec))
                js_r = js_nats(pb, q_r)
                rest = (1.0 - js_r / js_e
                        if js_e > 0 else np.nan)
                rows.append({'js_e': js_e,
                             'js_r': js_r,
                             'rest': rest,
                             'l': l_star, 'qh': qh,
                             'd': d_star})
                lens_cols.append(lens_js(res_b, res_e))
                dnorms.append(dnorm)
                log('T2a ki=%d (%s) jsE=%.5f l*=%d '
                    'qh*=%d d=%.3f rest=%.3f'
                    % (ki, tags[ki], js_e, l_star,
                       qh, d_star,
                       rest if not np.isnan(rest)
                       else -1), lines)

            js_e_a = np.array([r['js_e']
                               for r in rows])
            js_r_a = np.array([r['js_r']
                               for r in rows])
            rest_a = np.array([r['rest']
                               for r in rows])
            l_star_a = np.array([r['l'] for r in rows])
            qh_star_a = np.array([r['qh']
                                  for r in rows])
            d_star_a = np.array([r['d'] for r in rows])
            gates_ok = bool(nL >= MIN_POS
                            and nhq_ok
                            and np.all(js_e_a > 0))
            rest_med = (float(np.median(rest_a))
                        if nL and not np.all(
                            np.isnan(rest_a))
                        else None)
            l_cnt = {}
            for l_ in l_star_a:
                l_cnt[int(l_)] = \
                    l_cnt.get(int(l_), 0) + 1
            qh_cnt = {}
            for q_ in qh_star_a:
                qh_cnt[int(q_)] = \
                    qh_cnt.get(int(q_), 0) + 1
            T2a = {
                'n_logic': nL, 'g7': G7_HEAD,
                'l3': L3_GATED,
                'med_js_g7e': round(
                    float(np.median(js_e_a)), 6)
                if nL else None,
                'med_restoration': round(rest_med, 4)
                if rest_med is not None else None,
                'restoration_per_pos': [
                    round(float(x), 4)
                    for x in rest_a],
                'l_star_counts': {
                    str(k): int(v) for k, v
                    in sorted(l_cnt.items())},
                'qh_star_counts': {
                    str(k): int(v) for k, v
                    in sorted(qh_cnt.items())},
                'med_d_star': round(
                    float(np.median(d_star_a)), 4)
                if nL else None,
                'min_pos_gate': MIN_POS,
                'gates_ok': gates_ok,
                'note': 'carrier selection on delta '
                        'energy (quasi-post-hoc '
                        'labeled); restoration is '
                        'causal and not guaranteed',
                'tags': tags}
            log('T2a nL=%d jsE=%.5f restMed=%s lCnt=%s '
                'qhCnt=%s'
                % (nL, T2a['med_js_g7e'],
                   rest_med, json.dumps(l_cnt),
                   json.dumps(qh_cnt)), lines)

            # ---------- T2b logit-lens build-up ----------
            lens_mat = (np.stack(lens_cols, axis=1)
                        if nL else np.array([]))
            prof = {}
            if nL:
                med_lens = np.median(lens_mat,
                                     axis=1)
                js_fin = float(np.median(js_e_a))
                pick = (4, 8, 12, 16, 20, 24, 28,
                        32, 35)
                prof = {
                    'med_lens_js': {
                        str(int(l)): round(
                            float(med_lens[l]), 6)
                        for l in pick},
                    'ratio_vs_final': {
                        str(int(l)): round(
                            float(med_lens[l])
                            / max(js_fin, 1e-30), 4)
                        for l in pick},
                    'med_js_final': round(js_fin, 6)}
            T2b = prof if prof else None
            log('T2b lens ratios %s'
                % json.dumps(prof.get(
                    'ratio_vs_final', {})), lines)

            # ---------- T2c concentration + sham ----------
            conc = {}
            shamJ = []
            shamD = []
            if nL:
                dmed = np.median(np.stack(dnorms),
                                 axis=0)  # [NL, NHQ]
                best = None
                for li in range(L_MIN, NL):
                    w = dmed[li]
                    tot = float(w.sum())
                    if tot <= 1e-30:
                        continue
                    sh = float(w.max()) / tot
                    pr_ = tot / max(
                        float((w ** 2).sum()), 1e-30)
                    if best is None or sh > best[1]:
                        best = (li, sh, pr_)
                conc = {
                    'top_layer': int(best[0]),
                    'top1_share': round(best[1], 4),
                    'participation': round(best[2],
                                           2)}
                for pi, pr in enumerate(GEN_PROMPTS):
                    ent = sel['P%d' % pi]
                    if ent.get('skipped'):
                        continue
                    if ent['positions']['sham'] \
                            is not None:
                        p_sh = int(
                            ent['positions']['sham'])
                        pb = caps[pi][1]
                        q_s, res_s, o_s = run_chain(
                            pr, p_sh, True)
                        shamJ.append(
                            js_nats(pb, q_s))
                        q_b2, res_b2, o_b2 = \
                            run_chain(pr, p_sh, False)
                        dmax = 0.0
                        for li in range(L_MIN, NL):
                            db = o_b2[li].reshape(
                                NHQ, HDIM)
                            de = o_s[li].reshape(
                                NHQ, HDIM)
                            dmax = max(dmax, float(
                                np.max(np.linalg.norm(
                                    de - db,
                                    axis=1))))
                        shamD.append(dmax)
            T2c = {
                'concentration': conc if conc else None,
                'med_js_sham_g7e': round(
                    float(np.median(shamJ)), 6)
                if shamJ else None,
                'med_max_dnorm_sham': round(
                    float(np.median(shamD)), 4)
                if shamD else None,
                'n_sham': len(shamJ)}
            log('T2c conc=%s sham=%.6f dmax=%s'
                % (json.dumps(conc),
                   T2c['med_js_sham_g7e'] or -1,
                   T2c['med_max_dnorm_sham']), lines)

            # ---------- verdict ----------
            if not gates_ok:
                verdict = 'amplification_' \
                          'undetermined_void'
            elif (rest_med is not None
                  and rest_med >= REST_LOC):
                verdict = 'amp_head_localized_qwen'
            elif (rest_med is not None
                  and rest_med <= REST_DIST):
                verdict = 'amp_distributed_qwen'
            else:
                verdict = 'amp_mixed_qwen'

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
        'a18_3014': a18_ok, 'a19_3015': a19_ok,
    }
    res = {
        'phase': 3016,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at the logit-lens '
            'readout - model.norm returns fp32 '
            'while lm_head weights are bf16 '
            '(F.linear dtype mismatch '
            'double != BFloat16); fix: cast the '
            'normed residuals to '
            'lm_head.weight.dtype; crash was at '
            'the FIRST lens call, after T2a '
            'positions were computed but before '
            'any verdict - no data validity '
            'impact, design unchanged; run2: '
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
            'js_g7e': js_e_a if nL else np.array([]),
            'js_restored': js_r_a
            if nL else np.array([]),
            'restoration': rest_a
            if nL else np.array([]),
            'l_star': l_star_a
            if nL else np.array([]),
            'qh_star': qh_star_a
            if nL else np.array([]),
            'dnorm': np.stack(dnorms)
            if nL else np.array([]),
            'lens_js': lens_mat
            if lens_mat.size else np.array([])}
    npz_path = os.path.join(
        OUT, 'omega_p2j_amplification_trace_qwen.npz')
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
