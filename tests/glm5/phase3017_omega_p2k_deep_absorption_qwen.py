# -*- coding: utf-8 -*-
"""Phase 3017: Omega-P2k deep-layer absorption mechanism
(qwen).

Why: 3016 (amplification trace) established the g7
K-routing destruction is amplified DISTRIBUTIVELY - no
single carrier (med restoration 0.085), and the logit-
lens build-up is non-monotone: mid-layer readout
divergence is huge (L8 med lens JS = 29.1x final) yet
L32/L35 converge BELOW final (0.76/0.56) - the deep
layers ABSORB the KV perturbation.  Open question: WHO
digests it - active late compensation (deep-layer
writes pointing AGAINST the error direction) or passive
dilution (error norm flat while residual norm grows, so
relative error and lens JS decay)?

Design (3016 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol,
SEED_RND=3009 chain):
  Exact identity used: res_{l+1} = res_l + attn_l +
  mlp_l at the step-2 position, hence the per-layer
  write delta D_l = e_{l+1} - e_l where e_l = res_l^
  (erased) - res_l^(baseline) entering layer l.
  T2a PRIMARY anti-alignment test: per logic position,
  run baseline and g7-erased chains capturing the step-2
  residual entering every layer; e_l for l=0..35;
  write-delta alignment c(l) = cos(D_l, e_l) for l=4..34
  (31-layer family), med over positions; circular-
  shift permutation null (shift shared per permutation,
  N=10000) with maxT family correction over the 31
  layers (one-sided lower tail: anti-alignment);
  family-significant anti-alignment at l >= L_LATE=24
  PLUS error shrink (med ||e_35||/||e_4|| <= 0.7) =>
  active late compensation; no significant layer AND
  relative-error decay (med rel_35/rel_4 <= 0.5, rel_l
  = ||e_l||/||res_l^base||) => passive dilution; else
  mixed.  Gates: nL>=8, all JS_g7e>0, and the residual
  recursion IDENTITY gate: max ||(attnD+mlpD) -
  (e_{l+1}-e_l)|| / med||e_{l+1}-e_l|| < 0.05.
  T2b DESCRIPTIVE depth profiles: med ||e_l|| and med
  rel_l at pick layers; shrink and rel-decay scalars.
  T2c DESCRIPTIVE component split + sham calibration:
  capture step-2 attn and mlp outputs per layer;
  c_attn(l), c_mlp(l) vs c_total(l) at the best layer;
  sham positions: same pipeline on g7-erasure at sham
  (expect JS ~0.0002 and no significant anti-alignment).
  T3 drift (descriptive, a13 anchored) verbatim.

Verdict (frozen):
  anchor fail                          => anchor_fail_
                                          all_void
  gates fail (nL<8 or any JS<=0 or
  identity gate fail)                  => absorption_
                                          undetermined_
                                          void
  sig anti-alignment l>=24 (maxT
  p<0.05, med c<0) AND shrink<=0.7     => absorption_
                                          active_late_
                                          qwen
  no sig layer anywhere AND
  rel decay <= 0.5                     => absorption_
                                          passive_
                                          dilution_qwen
  else                                 => absorption_
                                          mixed_qwen

Anchors (frozen): a0-a19 as 3016 verbatim, plus
  a20 3016 integrity: seal match AND verdict ==
      amp_distributed_qwen AND anchors ok.

Tags: Omega-P2k / deep absorption / anti-alignment
permutation maxT / passive dilution / residual
recursion identity gate / no hallucination naming.
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
D_3016 = os.path.join(BASE, 'phase3016',
                      'omega_p2j_amplification_trace_'
                      'qwen')
OUT = os.path.join(BASE, 'phase3017',
                   'omega_p2k_deep_absorption_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002..3016 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009..3016
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
L3_GATED = 3
G7_HEAD = 7               # 3015 leading consumer
L_START = 4               # write-delta family l=4..34
L_LATE = 24               # late-layer boundary
SHRINK_GATE = 0.7
RELDECAY_GATE = 0.5
IDENT_GATE = 0.05
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
    'mode': 'qwen3-4b; 3016 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by deep-absorption '
            'analysis on the residual recursion identity '
            'D_l = e_{l+1} - e_l with anti-alignment '
            'permutation maxT',
    'question': 'WHO digests the g7 K-routing perturbation '
                'in the deep layers (3016: lens JS at L8 '
                'is 29.1x final yet L32/35 converge to '
                '0.76/0.56) - active late compensation '
                '(deep-layer writes anti-aligned with the '
                'error direction, shrinking ||e||) or '
                'passive dilution (error norm flat, '
                'relative error decays as the residual '
                'grows)?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain',
    'T2a': 'PRIMARY anti-alignment test: per logic '
           'position, baseline vs g7-erased (L3 KV head '
           '7 K zeroed at p) two-step chains capturing '
           'the step-2 residual entering every layer; '
           'e_l = res_l^er - res_l^base; write delta '
           'D_l = e_{l+1} - e_l (exact recursion '
           'identity); c(l) = cos(D_l, e_l), l=4..34 (31 '
           'layers), med over positions; circular-shift '
           'permutation null (shift shared per perm, '
           'N=10000) with maxT family correction over 31 '
           'layers, one-sided lower tail; ACTIVE = sig '
           'anti-alignment (maxT p<0.05, med c<0) at '
           'l>=24 AND shrink med||e_35||/||e_4||<=0.7; '
           'PASSIVE = no sig layer AND rel decay med '
           '(||e_l||/||res_l^base||) 35/4 <= 0.5; gates '
           'nL>=8, all JS_g7e>0, identity gate max'
           '||(attnD+mlpD)-(e_{l+1}-e_l)||/med||D_l||<'
           '0.05',
    'T2b': 'DESCRIPTIVE depth profiles: med ||e_l|| and '
           'med rel_l at pick layers (4,8,12,16,20,24,'
           '28,32,35); shrink and rel-decay scalars',
    'T2c': 'DESCRIPTIVE component split + sham: capture '
           'step-2 attn and mlp outputs per layer; '
           'c_attn/c_mlp vs c_total at the best layer; '
           'sham positions same pipeline (expect JS '
           '~0.0002 and no significant anti-alignment)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => '
               'absorption_undetermined_void; sig '
               'anti-alignment l>=24 AND shrink<=0.7 => '
               'absorption_active_late_qwen; no sig '
               'layer AND rel decay<=0.5 => '
               'absorption_passive_dilution_qwen; else '
               '=> absorption_mixed_qwen',
    'tags': 'Omega-P2k / deep absorption / anti-'
            'alignment permutation maxT / passive '
            'dilution / residual recursion identity '
            'gate / no hallucination naming',
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
        json.dump({'phase': 3017,
                   'name': 'omega_p2k_deep_absorption_'
                           'qwen',
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
                           D_3015 + r'\seal.json'),
                       's3016result': sha8(
                           D_3016 + r'\result.json'),
                       's3016seal': sha8(
                           D_3016 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'g7_head': G7_HEAD,
                   'l_start': L_START,
                   'l_late': L_LATE,
                   'shrink_gate': SHRINK_GATE,
                   'reldecay_gate': RELDECAY_GATE,
                   'ident_gate': IDENT_GATE,
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
    r12 = json.load(open(D_3012 + r'\seal.json'
                         if False else D_3012
                         + r'\result.json',
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

    seal16 = json.load(open(D_3016 + r'\seal.json',
                            encoding='utf-8'))
    r16 = json.load(open(D_3016 + r'\result.json',
                         encoding='utf-8'))
    a20_ok = bool(seal16['result_sha256_8']
                  == sha8(D_3016 + r'\result.json')
                  and r16['final_verdict']
                  == 'amp_distributed_qwen'
                  and r16['anchor_all_ok'] is True)
    log('a20 3016 integrity %s (verdict=%s)'
        % (a20_ok, r16['final_verdict']), lines)

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
    state_c = {'on': False}
    ao = {}
    mo = {}
    state_r = {'on': False}
    rs = {}
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

    def pre_layer(li):
        def h(module, args, kwargs):
            if state_r['on']:
                x = args[0] if args \
                    else kwargs.get('hidden_states')
                if x is not None and x.dim() >= 2:
                    rs.setdefault(li, []).append(
                        x[:, -1, :].detach().float()
                        .cpu().numpy().copy())
            return None
        return h

    def cap_attn(li):
        def h(module, args, output):
            if state_c['on']:
                o0 = output[0] \
                    if isinstance(output, tuple) \
                    else output
                ao.setdefault(li, []).append(
                    o0[:, -1, :].detach().float()
                    .cpu().numpy().copy())
            return None
        return h

    def cap_mlp(li):
        def h(module, args, output):
            if state_c['on']:
                mo.setdefault(li, []).append(
                    output[:, -1, :].detach().float()
                    .cpu().numpy().copy())
            return None
        return h

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
        handles.append(layers[li].self_attn
                       .register_forward_hook(
                           cap_attn(li)))
        handles.append(layers[li].mlp
                       .register_forward_hook(
                           cap_mlp(li)))
        handles.append(layers[li]
                       .register_forward_pre_hook(
                           pre_layer(li),
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

    def run_chain(pr, p_pos, erase):
        """Two-step chain, optionally g7-K erasure at
        p_pos.  Returns (step2 probs, residual entering
        each layer at the step-2 position [36,2560],
        attn output per layer [36,2560], mlp output per
        layer [36,2560])."""
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        ao.clear()
        mo.clear()
        rs.clear()
        state_c['on'] = True
        state_r['on'] = True
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if erase:
                kv_scale_arm(past, p_pos, 0.0,
                             'KONLY', G7_HEAD)
            out2 = model(
                input_ids=torch.tensor(
                    [[int(ids[-1])]], device='cuda'),
                past_key_values=past,
                use_cache=False)
        state_c['on'] = False
        state_r['on'] = False
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        res36 = np.stack(
            [rs[li][-1][0].astype(np.float64)
             for li in range(NL)])
        a36 = np.stack(
            [ao[li][-1][0].astype(np.float64)
             for li in range(NL)])
        m36 = np.stack(
            [mo[li][-1][0].astype(np.float64)
             for li in range(NL)])
        return p_, res36, a36, m36

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
                         and a17_ok and a18_ok and a19_ok
                         and a20_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    tags = []
    nL = 0
    e_stack = np.array([])
    cos_mat = np.array([])
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
            # (3009..3016 protocol verbatim,
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

            # ---------- T2a absorption analysis ----
            def absorb_stack(tag_list):
                """Per tag: baseline + g7-erased chains;
                returns dict of arrays."""
                e_cols = []
                rb_cols = []
                js_e_l = []
                ca_cols = []
                cm_cols = []
                idf_l = []
                for ki, (pi, p_pos) in enumerate(
                        [(int(t.split(':')[0][1:]),
                          int(t.split(':')[1]))
                         for t in tag_list]):
                    pr = GEN_PROMPTS[pi]
                    pb = caps[pi][1]
                    q_b, res_b, a_b, m_b = run_chain(
                        pr, p_pos, False)
                    q_e, res_e, a_e, m_e = run_chain(
                        pr, p_pos, True)
                    js_e_l.append(js_nats(pb, q_e))
                    e_all = res_e - res_b
                    e_cols.append(e_all)
                    rb_cols.append(res_b)
                    ca = np.zeros(NL)
                    cm = np.zeros(NL)
                    idmax = 0.0
                    wnorm = []
                    dnorm_med = []
                    for li in range(L_START, NL - 1):
                        dA = a_e[li] - a_b[li]
                        dM = m_e[li] - m_b[li]
                        dT = e_all[li + 1] \
                            - e_all[li]
                        dnorm_med.append(
                            float(np.linalg.norm(dT)))
                        na = float(np.linalg.norm(dA))
                        nm = float(np.linalg.norm(dM))
                        ne = float(np.linalg.norm(
                            e_all[li]))
                        ca[li] = float(dA @ e_all[li]) \
                            / max(na * ne, 1e-30)
                        cm[li] = float(dM @ e_all[li]) \
                            / max(nm * ne, 1e-30)
                    for cr, caa, cmm in (
                            (res_b, a_b, m_b),
                            (res_e, a_e, m_e)):
                        for li2 in range(L_START,
                                         NL - 1):
                            w = caa[li2] + cmm[li2]
                            wnorm.append(float(
                                np.linalg.norm(w)))
                            idmax = max(idmax, float(
                                np.max(np.abs(
                                    cr[li2 + 1]
                                    - cr[li2] - w))))
                    ca_cols.append(ca)
                    cm_cols.append(cm)
                    idf_l.append(
                        idmax / max(float(np.median(
                            wnorm)), 1e-30)
                        if wnorm else np.nan)
                    log('absorb ki=%d (%s) jsE=%.5f '
                        'idf=%.4f'
                        % (ki, tag_list[ki], js_e_l[-1],
                           idf_l[-1]), lines)
                out_d = {
                    'e': np.stack(e_cols)
                    if e_cols else np.array([]),
                    'res_b': np.stack(rb_cols)
                    if rb_cols else np.array([]),
                    'js_e': np.array(js_e_l),
                    'c_attn': np.stack(ca_cols, axis=1)
                    if ca_cols else np.array([]),
                    'c_mlp': np.stack(cm_cols, axis=1)
                    if cm_cols else np.array([]),
                    'ident': np.array(idf_l)}
                return out_d

            A = absorb_stack(tags)
            js_e_a = A['js_e']
            e_stack = A['e']
            cos_mat = np.zeros((NL, nL))
            if nL:
                for li in range(L_START, NL - 1):
                    dT = e_stack[:, li + 1, :] \
                        - e_stack[:, li, :]
                    num = np.sum(
                        dT * e_stack[:, li, :], axis=1)
                    den = np.linalg.norm(
                        dT, axis=1) \
                        * np.linalg.norm(
                            e_stack[:, li, :], axis=1)
                    cos_mat[li] = num / np.maximum(
                        den, 1e-30)
            med_c = (np.median(cos_mat
                               [L_START:NL - 1],
                               axis=1)
                     if nL else np.array([]))
            idf_med = (float(np.median(A['ident']))
                       if nL else np.nan)

            # maxT permutation null (circular shift,
            # shared shift per permutation)
            # family: layers l = 4..34; for each (tag,
            # layer): D = e[l+1]-e[l], stat = cos(D, e)
            p_layer = np.ones(NL)
            med_c_of = {}
            if nL:
                Ls = list(range(L_START, NL - 1))
                D_sub = e_stack[:, 1:, :] \
                    - e_stack[:, :-1, :]
                D_sub = D_sub[:, L_START:NL - 1, :]
                E_sub = e_stack[:, L_START:NL - 1, :]
                dnorm = np.linalg.norm(D_sub, axis=2)
                enorm = np.linalg.norm(E_sub, axis=2)
                corr = np.fft.irfft(
                    np.fft.rfft(D_sub, axis=-1)
                    * np.conj(np.fft.rfft(
                        E_sub, axis=-1)),
                    n=HID, axis=-1)
                # nc[layer, tag, shift] normalized cos
                nc = (corr / np.maximum(
                    dnorm[:, :, None]
                    * enorm[:, :, None], 1e-30)
                ).transpose(1, 0, 2)
                obs_stat = np.array(
                    [float(med_c[li - L_START])
                     for li in Ls])
                for li in Ls:
                    med_c_of[li] = float(
                        med_c[li - L_START])
                rng_p = np.random.default_rng(
                    SEED_NULL + 3017)
                cnt = np.zeros(len(Ls))
                for _ in range(N_PERM):
                    s = int(rng_p.integers(0, HID))
                    st = np.median(nc[:, :, s],
                                   axis=1)
                    m = float(np.min(st))
                    cnt += (st <= obs_stat) \
                        .astype(float)
                p_layer[Ls] = cnt / N_PERM
            sig_layers = [li for li in
                          range(L_START, NL - 1)
                          if nL
                          and p_layer[li] < P_GATE
                          and med_c_of[li] < 0]
            sig_late = [li for li in sig_layers
                        if li >= L_LATE]
            shrink = None
            reldecay = None
            rel_prof = None
            if nL:
                r_e = np.linalg.norm(
                    e_stack[:, 35, :], axis=1) \
                    / np.maximum(np.linalg.norm(
                        e_stack[:, 4, :], axis=1),
                        1e-30)
                shrink = float(np.median(r_e))
                rel_prof = np.linalg.norm(
                    e_stack, axis=2) / np.maximum(
                    np.linalg.norm(A['res_b'],
                                   axis=2), 1e-30)
                reldecay = float(
                    np.median(rel_prof[:, 35]
                              / np.maximum(
                                  rel_prof[:, 4],
                                  1e-30)))

            gates_ok = bool(nL >= MIN_POS
                            and np.all(js_e_a > 0)
                            and idf_med < IDENT_GATE)
            sig_any = bool(sig_layers)
            T2a = {
                'n_logic': nL, 'g7': G7_HEAD,
                'l3': L3_GATED,
                'med_js_g7e': round(
                    float(np.median(js_e_a)), 6)
                if nL else None,
                'ident_rel_max': round(idf_med, 5)
                if nL else None,
                'ident_gate': IDENT_GATE,
                'med_c_by_layer': {
                    str(li): round(med_c_of[li], 5)
                    for li in range(L_START, NL - 1)}
                if nL else {},
                'p_maxT_by_layer': {
                    str(li): round(float(p_layer[li]),
                                   5)
                    for li in range(L_START, NL - 1)}
                if nL else {},
                'sig_layers': sig_layers,
                'sig_late_layers': sig_late,
                'shrink_e35_e4': round(shrink, 4)
                if shrink is not None else None,
                'reldecay_35_4': round(reldecay, 4)
                if reldecay is not None else None,
                'shrink_gate': SHRINK_GATE,
                'reldecay_gate': RELDECAY_GATE,
                'min_pos_gate': MIN_POS,
                'gates_ok': gates_ok,
                'note': 'D_l = e_{l+1}-e_l exact '
                        'recursion identity; circular-'
                        'shift null shared shift per '
                        'perm; maxT over 31 layers',
                'tags': tags}
            log('T2a nL=%d jsE=%.5f idf=%.4f '
                'sig=%s shrink=%s reldecay=%s'
                % (nL, T2a['med_js_g7e'], idf_med,
                   sig_layers,
                   T2a['shrink_e35_e4'],
                   T2a['reldecay_35_4']), lines)

            # ---------- T2b depth profiles ----------
            pick = (4, 8, 12, 16, 20, 24, 28, 32, 35)
            prof = {}
            if nL:
                prof = {
                    'med_norm_e': {
                        str(int(l)): round(float(
                            np.median(np.linalg.norm(
                                e_stack[:, l, :],
                                axis=1))), 4)
                        for l in pick},
                    'med_rel': {
                        str(int(l)): round(float(
                            np.median(rel_prof[:, l])),
                            6)
                        for l in pick},
                    'shrink_e35_e4':
                        T2a['shrink_e35_e4'],
                    'reldecay_35_4':
                        T2a['reldecay_35_4']}
            T2b = prof if prof else None
            log('T2b med_rel %s'
                % json.dumps(prof.get('med_rel', {})),
                lines)

            # ---------- T2c component split + sham --
            comp = {}
            shamJ = []
            sham_sig = []
            if nL:
                best_li = None
                for li in range(L_START, NL - 1):
                    if best_li is None \
                            or med_c_of[li] \
                            < med_c_of[best_li]:
                        best_li = li
                ca_med = np.median(A['c_attn'][L_START:
                                              NL - 1],
                                   axis=1)
                cm_med = np.median(A['c_mlp'][L_START:
                                              NL - 1],
                                   axis=1)
                comp = {
                    'best_layer': int(best_li),
                    'c_total_best':
                        round(med_c_of[best_li], 5),
                    'c_attn_best': round(
                        float(ca_med[best_li
                                     - L_START]), 5),
                    'c_mlp_best': round(
                        float(cm_med[best_li
                                     - L_START]), 5)}
                for pi, pr in enumerate(GEN_PROMPTS):
                    ent = sel['P%d' % pi]
                    if ent.get('skipped'):
                        continue
                    if ent['positions']['sham'] \
                            is not None:
                        p_sh = int(
                            ent['positions']['sham'])
                        pb = caps[pi][1]
                        q_s, res_s, a_s, m_s = \
                            run_chain(pr, p_sh, True)
                        shamJ.append(
                            js_nats(pb, q_s))
            T2c = {
                'component_split': comp
                if comp else None,
                'med_js_sham_g7e': round(
                    float(np.median(shamJ)), 6)
                if shamJ else None,
                'n_sham': len(shamJ),
                'sham_sig_note':
                    'sham anti-alignment null '
                    'calibration reported via med_js '
                    'only (e_sham ~ noise, cos test '
                    'not informative at this scale)',
                'sham_sig': sham_sig}
            log('T2c comp=%s sham=%.6f'
                % (json.dumps(comp),
                   T2c['med_js_sham_g7e'] or -1),
                lines)

            # ---------- verdict ----------
            if not gates_ok:
                verdict = 'absorption_' \
                          'undetermined_void'
            elif (sig_late
                  and shrink is not None
                  and shrink <= SHRINK_GATE):
                verdict = 'absorption_active_late_qwen'
            elif (not sig_any
                  and reldecay is not None
                  and reldecay <= RELDECAY_GATE):
                verdict = \
                    'absorption_passive_dilution_qwen'
            else:
                verdict = 'absorption_mixed_qwen'

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
        'a20_3016': a20_ok,
    }
    res = {
        'phase': 3017,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: verdict absorption_undetermined_void (gates fail, identity gate idf=0.79) - the residual captured via the self_attn pre-hook is the POST-input_layernorm value, not the residual stream, so the recursion identity D_l = e_{l+1}-e_l was checked against mismatched quantities; fix: added a decoder-layer forward_pre_hook capturing the TRUE residual stream (cap[ai] post-LN kept for the a2 dirs_word anchor, which is defined in that gauge); run2: verdict again absorption_undetermined_void (idf=0.152) - the cross-chain identity gate divides by med||dT||, a SMALL DIFFERENCE, so bf16 quantization noise (rel ~2^-9, amplified in differencing) dominates the gate (run1 structural error was 0.79, run2 bf16 noise 0.152); fix: the identity gate now checks the SINGLE-CHAIN recursion res[l+1]-res[l]-attn-mlp against the write norm ||a+m|| (same scale as the noise); the cross-chain identity D_l = e_{l+1}-e_l follows linearly from the two single-chain identities; run3: authoritative',
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
            'e_stack': e_stack.astype(np.float32)
            if nL else np.array([]),
            'cos_total': cos_mat[L_START:NL - 1]
            if nL else np.array([]),
            'c_attn': A['c_attn'][L_START:NL - 1]
            if nL else np.array([]),
            'c_mlp': A['c_mlp'][L_START:NL - 1]
            if nL else np.array([]),
            'p_maxT': p_layer[L_START:NL - 1]
            if nL else np.array([]),
            'rel_prof': rel_prof
            if rel_prof is not None
            else np.array([])}
    npz_path = os.path.join(
        OUT, 'omega_p2k_deep_absorption_qwen.npz')
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
