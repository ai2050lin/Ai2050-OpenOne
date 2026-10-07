# -*- coding: utf-8 -*-
"""Phase 3021: Omega-P2o injection-end anatomy -
query-head accounting of the L4 immediate JS
(qwen).

Why: 3020 (layer-resolved logit-lens JS
trajectory) established the 30x readout gap is
INJECTION-BORN: at the FIRST divergent layer L4
the immediate JS is logic 0.0502 vs content
5.3e-05 (944x), error norm ||e4|| 134x (3.60 vs
0.027), and the downstream is an EQUALIZER
(dilutes 944x to 28.3x).  Open question: WHICH
heads at L3 carry the injection - is the
immediate readout concentrated in a few query
heads (linking 3015's KV-consumer ranking: head
7 share .39, head 1 .12), or distributed?

Design (3020 machine verbatim for anchors/
geometry/generation/position selection/two-step
protocol, SEED_RND=3009 chain):
  At L3 (the gate layer) capture the o_proj INPUT
  per query head (32 x 128, o_proj pre-hook) and
  the MLP output at the step-2 position, baseline
  vs g7-K-erased chains.  Since the residual
  entering L3 is identical in both chains, the
  identity e4 = W_o @ dConcat + dM holds EXACTLY
  in exact arithmetic (bf16 gate: ||sum_q d_q +
  dM - e4|| / ||e4|| med < 0.05, d_q = W_o^q @
  da_q, W_o^q = o_proj weight columns q*128..
  (q+1)*128).  Contribution shares p_q = d_q . e4
  / ||e4||^2, p_m = dM . e4 / ||e4||^2,
  sum_q p_q + p_m == 1.
  T2a PRIMARY concentration: conc_top3 = med over
  logic tags of (sum of the 3 largest p_q);
  within-tag permutation null on p_q (32 values,
  N_PERM=10000, one-sided p = P(null >= obs)),
  med over tags.
  T2b KV-group linkage: P_kv = sum of p_q over
  the GQA group (q // 4, 32 query heads -> 8 KV
  heads); med over tags -> ranking; Spearman
  rank correlation with the sealed 3015
  med_share_top8 (7: .39, 1: .1205, 5: .0174,
  6: .0172, 2: .0104, 0: .0103, 4: .0099,
  3: .0074) - DESCRIPTIVE; chain check a25:
  med ||e4|| (4 decimals) must equal the sealed
  3020 value 3.6017 (bit-identical chains).
  T2c DESCRIPTIVE specificity: same pipeline at
  content and sham positions; med p_m, med conc,
  activity ratio sum |p_q| logic/content.

Verdict (frozen):
  anchor fail                     => anchor_fail_
                                    all_void
  gates fail (nL<8 or any js_final<=0 or
  identity gate fail or n_content<8 or
  n_sham<8)  => injection_anatomy_undetermined_
                                    void
  conc_med >= 0.5 AND p_perm_med <= 0.01
                                  => injection_
                                    concentrated_
                                    qwen
  conc_med < 0.5 AND p_perm_med <= 0.01
                                  => injection_
                                    distributed_
                                    qwen
  else                            => injection_
                                    mixed_qwen

Anchors (frozen): a0-a23 as 3020 verbatim, plus
  a24 3020 integrity: seal match AND verdict ==
      injection_readout_asymmetric_qwen AND
      anchors ok
  a25 chain identity: med ||e4|| logic (4
      decimals) == 3.6017 (sealed 3020 T2b).

Tags: Omega-P2o / injection-end anatomy /
query-head decomposition of the L4 error /
GQA group linkage to 3015 / no hallucination
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
D_3015 = os.path.join(BASE, 'phase3015',
                      'omega_p2i_k_consumer_heads_qwen')
D_3016 = os.path.join(BASE, 'phase3016',
                      'omega_p2j_amplification_trace_'
                      'qwen')
D_3017 = os.path.join(BASE, 'phase3017',
                      'omega_p2k_deep_absorption_qwen')
D_3018 = os.path.join(BASE, 'phase3018',
                      'omega_p2l_dilution_decomposition_'
                      'qwen')
D_3019 = os.path.join(BASE, 'phase3019',
                      'omega_p2m_mlp_band_identity_qwen')
D_3020 = os.path.join(BASE, 'phase3020',
                      'omega_p2n_readout_specificity_'
                      'qwen')
OUT = os.path.join(BASE, 'phase3021',
                   'omega_p2o_injection_anatomy_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896
SEED_RND = 3009
K_GEN = 256
MIN_POS = 8
L3_GATED = 3
G7_HEAD = 7
NQ = 32              # query heads (o_proj input side)
HQ = 128             # head dim
N_PERM = 10000
CONC_GATE = 0.5
PPERM_GATE = 0.01
IDENT_GATE = 0.05
E4_3020 = 3.6017     # sealed 3020 T2b med_e4 logic
SHARE_3015 = {'7': 0.39, '1': 0.1205, '5': 0.0174,
              '6': 0.0172, '2': 0.0104, '0': 0.0103,
              '4': 0.0099, '3': 0.0074}
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
    'mode': 'qwen3-4b; 3020 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by query-head '
            'decomposition of the L4 injection error at '
            'L3: o_proj-input per head (32x128) + MLP '
            'output, baseline vs g7-K-erased; identity '
            'e4 = W_o @ dConcat + dM (bf16 gate 0.05); '
            'shares p_q = d_q.e4/||e4||^2 sum to 1',
    'question': 'WHICH heads at L3 carry the injection '
                '(3020: L4 immediate JS 944x logic vs '
                'content) - concentrated in a few query '
                'heads (linking 3015 KV-consumer '
                'ranking head 7 .39 / head 1 .12) or '
                'distributed?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain; '
          'L3 o_proj pre-hook + MLP output at step-2',
    'T2a': 'PRIMARY conc_top3 = med over logic tags of '
           'the sum of the 3 largest p_q (p_q = d_q.e4/'
           '||e4||^2, d_q = W_o^q @ da_q); within-tag '
           'permutation null on p_q N=10000 one-sided, '
           'med over tags; gates nL>=8, all js_final>0, '
           'identity med < 0.05, n_content>=8, '
           'n_sham>=8',
    'T2b': 'KV-group linkage: P_kv = sum p_q over GQA '
           'group q//4; med over tags; Spearman rank '
           'corr vs sealed 3015 med_share_top8 '
           '(descriptive); a25 chain identity: med '
           '||e4|| logic (4 dec) == 3.6017',
    'T2c': 'DESCRIPTIVE specificity: content/sham same '
           'pipeline; med p_m, med conc, activity ratio '
           'sum|p_q| logic/content',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => '
               'injection_anatomy_undetermined_void; '
               'conc_med>=0.5 and p_perm_med<=0.01 => '
               'injection_concentrated_qwen; '
               'conc_med<0.5 and p_perm_med<=0.01 => '
               'injection_distributed_qwen; else => '
               'injection_mixed_qwen',
    'tags': 'Omega-P2o / injection-end anatomy / '
            'query-head decomposition of the L4 error / '
            'GQA group linkage to 3015 / no '
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
        json.dump({'phase': 3021,
                   'name': 'omega_p2o_injection_'
                           'anatomy_qwen',
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
                           D_3016 + r'\seal.json'),
                       's3017result': sha8(
                           D_3017 + r'\result.json'),
                       's3017seal': sha8(
                           D_3017 + r'\seal.json'),
                       's3018result': sha8(
                           D_3018 + r'\result.json'),
                       's3018seal': sha8(
                           D_3018 + r'\seal.json'),
                       's3019result': sha8(
                           D_3019 + r'\result.json'),
                       's3019seal': sha8(
                           D_3019 + r'\seal.json'),
                       's3020result': sha8(
                           D_3020 + r'\result.json'),
                       's3020seal': sha8(
                           D_3020 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'g7_head': G7_HEAD,
                   'nq': NQ, 'hq': HQ,
                   'n_perm': N_PERM,
                   'conc_gate': CONC_GATE,
                   'pperm_gate': PPERM_GATE,
                   'ident_gate': IDENT_GATE,
                   'e4_3020': E4_3020,
                   'share_3015': SHARE_3015,
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
                  and r15['anchor_all_ok'] is True
                  and r15['T2a']['med_share_top8']
                  == SHARE_3015)
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

    seal17 = json.load(open(D_3017 + r'\seal.json',
                            encoding='utf-8'))
    r17 = json.load(open(D_3017 + r'\result.json',
                         encoding='utf-8'))
    a21_ok = bool(seal17['result_sha256_8']
                  == sha8(D_3017 + r'\result.json')
                  and r17['final_verdict']
                  == 'absorption_mixed_qwen'
                  and r17['anchor_all_ok'] is True)
    log('a21 3017 integrity %s (verdict=%s)'
        % (a21_ok, r17['final_verdict']), lines)

    seal18 = json.load(open(D_3018 + r'\seal.json',
                            encoding='utf-8'))
    r18 = json.load(open(D_3018 + r'\result.json',
                         encoding='utf-8'))
    a22_ok = bool(seal18['result_sha256_8']
                  == sha8(D_3018 + r'\result.json')
                  and r18['final_verdict']
                  == 'decomp_cancellation_dominant_qwen'
                  and r18['anchor_all_ok'] is True)
    log('a22 3018 integrity %s (verdict=%s)'
        % (a22_ok, r18['final_verdict']), lines)

    seal19 = json.load(open(D_3019 + r'\seal.json',
                            encoding='utf-8'))
    r19 = json.load(open(D_3019 + r'\result.json',
                         encoding='utf-8'))
    a23_ok = bool(seal19['result_sha256_8']
                  == sha8(D_3019 + r'\result.json')
                  and r19['final_verdict']
                  == 'mlp_band_distributed_qwen'
                  and r19['anchor_all_ok'] is True)
    log('a23 3019 integrity %s (verdict=%s)'
        % (a23_ok, r19['final_verdict']), lines)

    seal20 = json.load(open(D_3020 + r'\seal.json',
                            encoding='utf-8'))
    r20 = json.load(open(D_3020 + r'\result.json',
                         encoding='utf-8'))
    a24_ok = bool(seal20['result_sha256_8']
                  == sha8(D_3020 + r'\result.json')
                  and r20['final_verdict']
                  == 'injection_readout_asymmetric_qwen'
                  and r20['anchor_all_ok'] is True)
    log('a24 3020 integrity %s (verdict=%s)'
        % (a24_ok, r20['final_verdict']), lines)

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
    state_op = {'on': False}
    op_cap = {}
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

    def pre_oproj(li):
        def h(module, args, kwargs):
            if state_op['on']:
                x = args[0] if args \
                    else kwargs.get('hidden_states')
                if x is not None and x.dim() >= 2:
                    op_cap['x'] = x[:, -1, :] \
                        .detach().float().cpu() \
                        .numpy().copy()
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
    handles.append(layers[L3_GATED].self_attn.o_proj
                   .register_forward_pre_hook(
                       pre_oproj(L3_GATED),
                       with_kwargs=True))

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
        each layer at step-2 [36,2560], final-norm
        input x_fin [2560], L3 o_proj input [4096],
        L3 mlp output [2560])."""
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        ao.clear()
        mo.clear()
        rs.clear()
        fin_cap.pop('x', None)
        op_cap.pop('x', None)
        state_c['on'] = True
        state_r['on'] = True
        state_fin['on'] = True
        state_op['on'] = True
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
        state_fin['on'] = False
        state_op['on'] = False
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        res36 = np.stack(
            [rs[li][-1][0].astype(np.float64)
             for li in range(NL)])
        x_fin = fin_cap['x'][0].astype(np.float64)
        op_in = op_cap['x'][0].astype(np.float64)
        m3 = mo[3][-1][0].astype(np.float64)
        return p_, res36, x_fin, op_in, m3

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
                         and a20_ok and a21_ok and a22_ok
                         and a23_ok and a24_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    a25_diff = None
    tags = []
    nL = nC = nS = 0
    js_e_a = np.array([])
    js_final = {'logic': [], 'content': [],
                'sham': []}
    e4_store = {'logic': [], 'content': [],
                'sham': []}
    p_store = {'logic': [], 'content': [],
               'sham': []}
    pm_store = {'logic': [], 'content': [],
                'sham': []}
    conc_store = {'logic': [], 'content': [],
                  'sham': []}
    top3_store = {'logic': [], 'content': [],
                  'sham': []}
    act_store = {'logic': [], 'content': [],
                 'sham': []}
    ident_store = []
    pperm_store = []
    kv_store = []
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

            # ---------- T1 capture + baseline ------
            caps = {}
            base = {}
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
                q_b, res_b, xfin_b, op_b, m3_b = \
                    run_chain(pr, 0, False)
                base[pi] = {'res': res_b,
                            'op': op_b, 'm3': m3_b}
                log('T1 P%d captured + baseline' % pi,
                    lines)

            tags = []
            for pi in caps:
                for p_pos in sel['P%d' % pi][
                        'positions']['logic']:
                    tags.append('P%d:%d'
                                % (pi, p_pos))
            nL = len(tags)
            log('T1 logic positions n=%d' % nL, lines)

            # W_o^q columns: o_proj weight [2560, 4096]
            Wo = layers[L3_GATED].self_attn.o_proj \
                .weight.detach().double().cpu() \
                .numpy()
            assert Wo.shape == (HID, NQ * HQ), Wo.shape
            Wo_r = Wo.reshape(HID, NQ, HQ)
            rng_perm = np.random.default_rng(
                SEED_NULL + 21)

            def decompose(pi, p_pos):
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                q_e, res_e, xfin_e, op_e, m3_e = \
                    run_chain(pr, p_pos, True)
                js_f = js_nats(pb, q_e)
                e4 = res_e[4] - base[pi]['res'][4]
                dH = (op_e - base[pi]['op']) \
                    .reshape(NQ, HQ)
                dm = m3_e - base[pi]['m3']
                D = np.einsum('okt,kt->ko',
                              Wo_r, dH)
                n2 = float(e4 @ e4)
                p_q = (D @ e4) / n2
                p_m = float(dm @ e4) / n2
                ident = float(np.linalg.norm(
                    D.sum(0) + dm - e4)) \
                    / max(float(np.linalg.norm(e4)),
                          1e-30)
                order = np.argsort(p_q)[::-1]
                conc3 = float(p_q[order[:3]].sum())
                # permutation null on p_q: top-3 sum of
                # a random permutation of the 32 head
                # shares, N_PERM columns
                R = rng_perm.random((NQ, N_PERM))
                perm = np.argsort(R, axis=0)
                pv = np.take_along_axis(
                    np.broadcast_to(
                        p_q.reshape(NQ, 1),
                        (NQ, N_PERM)), perm, axis=0)
                null_c = np.sort(pv, axis=0)[-3:].sum(
                    axis=0)
                p_perm = float(np.mean(
                    null_c >= conc3))
                kv_p = np.array([
                    float(p_q[g * (NQ // 8):
                              (g + 1) * (NQ // 8)].sum())
                    for g in range(8)])
                top3 = [int(x) for x in order[:3]]
                return {'js': js_f, 'e4n':
                        float(np.linalg.norm(e4)),
                        'p_q': p_q, 'p_m': p_m,
                        'ident': ident,
                        'conc3': conc3,
                        'p_perm': p_perm,
                        'kv_p': kv_p, 'top3': top3}

            res_l = {}
            for ki, t in enumerate(tags):
                pi = int(t.split(':')[0][1:])
                p_pos = int(t.split(':')[1])
                d = decompose(pi, p_pos)
                res_l[t] = d
                js_final['logic'].append(d['js'])
                e4_store['logic'].append(d['e4n'])
                p_store['logic'].append(d['p_q'])
                pm_store['logic'].append(d['p_m'])
                conc_store['logic'].append(d['conc3'])
                top3_store['logic'].append(d['top3'])
                act_store['logic'].append(
                    float(np.abs(d['p_q']).sum()))
                ident_store.append(d['ident'])
                pperm_store.append(d['p_perm'])
                kv_store.append(d['kv_p'])
                log('dec ki=%d (%s) js=%.5f e4n=%.4f '
                    'pm=%.3f conc3=%.3f pperm=%.4f '
                    'ident=%.2e top3=%s'
                    % (ki, t, d['js'], d['e4n'],
                       d['p_m'], d['conc3'],
                       d['p_perm'], d['ident'],
                       d['top3']), lines)
            # content + sham
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                cps = ent['positions']['content']
                if cps:
                    pc = int(cps[0])
                    d = decompose(pi, pc)
                    js_final['content'].append(
                        d['js'])
                    e4_store['content'].append(
                        d['e4n'])
                    p_store['content'].append(d['p_q'])
                    pm_store['content'].append(
                        d['p_m'])
                    conc_store['content'].append(
                        d['conc3'])
                    top3_store['content'].append(
                        d['top3'])
                    act_store['content'].append(
                        float(np.abs(d['p_q']).sum()))
                    nC += 1
                p_sh = ent['positions']['sham']
                if p_sh is not None:
                    d = decompose(pi, p_sh)
                    js_final['sham'].append(d['js'])
                    e4_store['sham'].append(d['e4n'])
                    p_store['sham'].append(d['p_q'])
                    pm_store['sham'].append(d['p_m'])
                    conc_store['sham'].append(
                        d['conc3'])
                    top3_store['sham'].append(d['top3'])
                    act_store['sham'].append(
                        float(np.abs(d['p_q']).sum()))
                    nS += 1
            js_e_a = np.array(js_final['logic'])

            # ---------- T2a statistics ----------
            ident_med = float(np.median(ident_store))
            conc_med = float(np.median(
                conc_store['logic']))
            pperm_med = float(np.median(pperm_store))
            pm_med = float(np.median(
                pm_store['logic']))
            log('T2a conc3 med=%.4f pperm med=%.4f '
                'pm med=%.4f ident med=%.2e'
                % (conc_med, pperm_med, pm_med,
                   ident_med), lines)
            T2a = {
                'n_logic': nL, 'n_content': nC,
                'n_sham': nS,
                'g7': G7_HEAD, 'l3': L3_GATED,
                'nq': NQ, 'hq': HQ,
                'ident_med': float('%.6e'
                                   % ident_med),
                'ident_gate': IDENT_GATE,
                'conc_top3_med': round(conc_med, 4),
                'conc_gate': CONC_GATE,
                'p_perm_med': round(pperm_med, 4),
                'pperm_gate': PPERM_GATE,
                'p_perm_max': round(float(np.max(
                    pperm_store)), 4),
                'p_m_med': round(pm_med, 4),
                'med_js_final_logic': round(
                    float(np.median(js_e_a)), 6)
                if nL else None,
                'med_js_final_content': round(
                    float(np.median(
                        js_final['content'])), 6)
                if nC else None,
                'med_js_final_sham': round(
                    float(np.median(
                        js_final['sham'])), 6)
                if nS else None,
                'med_e4_norm': round(float(
                    np.median(e4_store['logic'])), 4),
                'conc_per_tag': [round(float(v), 4)
                                 for v in
                                 conc_store['logic']],
                'pperm_per_tag': [round(float(v), 4)
                                  for v in
                                  pperm_store],
                'top3_per_tag': top3_store['logic'],
                'tags': tags,
                'note': 'p_q = (W_o^q da_q).e4/||e4||^2 '
                        'at L3 step-2; sum p_q + p_m '
                        '== 1; conc_top3 = sum of the '
                        '3 largest p_q; within-tag '
                        'permutation null N=%d'
                        % N_PERM}

            # ---------- T2b KV-group linkage ------
            KV = np.stack(kv_store)
            kv_med = np.median(KV, axis=0)
            rank_new = np.argsort(
                np.argsort(-kv_med)) + 1
            sh_vec = np.array([SHARE_3015[str(g)]
                               for g in range(8)])
            rank_old = np.argsort(
                np.argsort(-sh_vec)) + 1
            d_rank = rank_new - rank_old
            spear = float(1.0 - 6.0
                          * float((d_rank ** 2).sum())
                          / (8 * (64 - 1)))
            top1_kv = int(np.argmax(kv_med))
            a25_diff = abs(round(float(np.median(
                e4_store['logic'])), 4) - E4_3020)
            a25_ok = bool(a25_diff < 1e-4)
            log('a25 med e4 %.4f vs 3020 %.4f diff=%.2e '
                'ok=%s' % (float(np.median(
                    e4_store['logic'])), E4_3020,
                    a25_diff, a25_ok), lines)
            T2b = {
                'kv_group_med_p': {
                    str(g): round(float(kv_med[g]), 4)
                    for g in range(8)},
                'kv_group_rank': {
                    str(g): int(rank_new[g])
                    for g in range(8)},
                'share_3015': SHARE_3015,
                'rank_3015': {
                    str(g): int(rank_old[g])
                    for g in range(8)},
                'spearman_3015': round(spear, 3),
                'top1_kv_group': top1_kv,
                'top1_kv_share': round(
                    float(kv_med[top1_kv]), 4),
                'a25_e4_diff': float('%.6e'
                                     % a25_diff),
                'note': 'P_kv = sum of p_q over GQA '
                        'group q//4; spearman vs '
                        'sealed 3015 med_share_top8 '
                        'descriptive; a25 = chain '
                        'identity vs 3020 med e4'}
            log('T2b kv med p=%s spearman=%.3f top1=%d'
                % ({g: round(float(kv_med[g]), 3)
                    for g in range(8)}, spear,
                   top1_kv), lines)

            # ---------- T2c specificity ----------
            def medv(x):
                return round(float(np.median(x)), 4) \
                    if x else None

            act_l = float(np.median(act_store['logic']))
            act_c = float(np.median(
                act_store['content'])) \
                if act_store['content'] else None
            T2c = {
                'med_conc': {
                    'logic': round(conc_med, 4),
                    'content': medv(
                        conc_store['content']),
                    'sham': medv(
                        conc_store['sham'])},
                'med_p_m': {
                    'logic': round(pm_med, 4),
                    'content': medv(
                        pm_store['content']),
                    'sham': medv(pm_store['sham'])},
                'med_act_sum_abs_p': {
                    'logic': float('%.6e' % act_l),
                    'content':
                        float('%.6e' % act_c)
                        if act_c is not None else None,
                    'sham': float('%.6e' % float(
                        np.median(
                            act_store['sham'])))
                    if act_store['sham'] else None},
                'act_ratio_logic_content':
                    round(act_l / max(act_c, 1e-30), 1)
                    if act_c else None,
                'note': 'descriptive specificity of '
                        'the injection decomposition'}
            log('T2c conc L/C/S=%s/%s/%s act '
                'L/C=%s/%s ratio=%s'
                % (T2c['med_conc']['logic'],
                   T2c['med_conc']['content'],
                   T2c['med_conc']['sham'],
                   T2c['med_act_sum_abs_p']['logic'],
                   T2c['med_act_sum_abs_p']['content'],
                   T2c['act_ratio_logic_content']),
                lines)

            # ---------- verdict ----------
            gates_ok = bool(nL >= MIN_POS
                            and np.all(js_e_a > 0)
                            and nC >= MIN_POS
                            and nS >= MIN_POS
                            and ident_med
                            < IDENT_GATE)
            if not gates_ok:
                verdict = 'injection_anatomy_' \
                          'undetermined_void'
            elif (conc_med >= CONC_GATE
                  and pperm_med <= PPERM_GATE):
                verdict = 'injection_concentrated_qwen'
            elif (conc_med < CONC_GATE
                  and pperm_med <= PPERM_GATE):
                verdict = 'injection_distributed_qwen'
            else:
                verdict = 'injection_mixed_qwen'

            # ---------- T3 drift (descriptive) --
            early = []
            late = []
            for pi in range(len(GEN_PROMPTS)):
                rec = recs[pi]
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
        'a20_3016': a20_ok, 'a21_3017': a21_ok,
        'a22_3018': a22_ok, 'a23_3019': a23_ok,
        'a24_3020': a24_ok,
        'a25_e4_chain_diff': a25_diff,
    }
    res = {
        'phase': 3021,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok
                              and a25_diff is not None
                              and a25_diff < 1e-4),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': '',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35,
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object),
            'tags': np.array(tags, dtype=object),
            'js_final_logic': js_e_a,
            'js_final_content': np.array(
                js_final['content']),
            'js_final_sham': np.array(
                js_final['sham']),
            'e4_norm': {ty: np.array(e4_store[ty])
                        for ty in e4_store},
            'p_q': {ty: np.stack(p_store[ty])
                    if p_store[ty]
                    else np.array([])
                    for ty in p_store},
            'p_m': {ty: np.array(pm_store[ty])
                    for ty in pm_store},
            'conc3': {ty: np.array(conc_store[ty])
                      for ty in conc_store},
            'top3_heads': {
                ty: np.array(top3_store[ty],
                             dtype=object)
                for ty in top3_store},
            'act_sum_abs_p': {
                ty: np.array(act_store[ty])
                for ty in act_store},
            'kv_group_p': np.stack(kv_store)
            if kv_store else np.array([]),
            'ident_vals': np.array(ident_store),
            'pperm_vals': np.array(pperm_store)}
    npz_path = os.path.join(
        OUT, 'omega_p2o_injection_anatomy_qwen.npz')
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
