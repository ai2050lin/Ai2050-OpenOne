# -*- coding: utf-8 -*-
"""Phase 3022: Omega-P2p L3 MLP relay neuron identity
via SwiGLU attribution (qwen).

Why: 3021 (injection-end anatomy) established the L4
injection error e4 = W_o @ dConcat + dM is carried 69%
by the L3 MLP channel (p_m med 0.686) vs 31% by the
query-head channel (GQA-locked to group 7, q28/29
dominant).  Open question: WHO inside the L3 MLP
carries the relay - a sparse neuron coalition or a
distributed field, is the coalition tag-consistent,
and is it the SAME neuron set as the sealed 3019
mid-band suppression field (top-32 at L10) or a new
dedicated coalition?

Design (3021 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol,
SEED_RND=3009 chain):
  SwiGLU attribution at L3: s_j = 2 dh_j (w_j . e4) /
  ||e4||^2 with dh = down_proj-input difference at L3,
  w_j the j-th column of W_down(L3), e4 = residual
  error entering L4 (identical chains give e_all[3]=0
  so e4 = da_3 + dM_3 exactly; pairing dM_3 with e4 is
  the relay's contribution to the L4 entry error).
  IDENTITY: sum_j s_j == 2 dM.e4/||e4||^2 (bf16 gate:
  |sum_s - ref| / (2||dM||||e4||/||e4||^2) med < 0.05).
  Chain anchors: a25 med ||e4|| (4 dec) == 3.6017
  (sealed 3020); a27 med p_m (4 dec) == 0.686 (sealed
  3021) - bit-identical chains.
  STATISTICS DISCIPLINE: the within-tag permutation
  null on conc_top-k is a PERMUTATION-INVARIANT
  statistic (degenerate, p==1 always - 3021 lesson) so
  it is NOT used.  Identity-sensitive tests instead:
  T2a concentration: per tag negmass/posmass, conc_
  dom = |s| share of the top-32 neurons in the
  dominant direction (neg if negmass>=posmass else
  pos); coalition set = those 32 neurons.
  T2b coalition tests: (i) cross-tag Jaccard of the
  11 L3 coalitions (55 pairs) vs random-32-subset
  null (N=2000, seed SEED_NULL+3022, one-sided
  p = P(null >= obs)); (ii) cross-phase Jaccard of
  L3 coalitions vs the sealed 3019 top-32 NEGATIVE
  sets at L10 (121 pairs, same null) - does the
  relay reuse the suppression field?
  T2c specificity: content/sham positions, med
  negmass at L3, spec_ratio = med negmass_content /
  med negmass_logic, sham JS calibration.

Verdict (frozen):
  anchor fail                     => anchor_fail_
                                    all_void
  gates fail (nL<8 or any js_final<=0 or bf16
  identity fail or n_mass<8)
                                  => relay_
                                    undetermined_void
  jac_cross_med > 2*jac_null_med AND
  p_cross <= 0.01                 => relay_shares_
                                    suppression_field_
                                    qwen
  elif jac_tag_med > 2*jac_null_med AND
  p_tag <= 0.01                   => relay_dedicated_
                                    coalition_qwen
  else                            => relay_
                                    distributed_qwen

Anchors (frozen): a0-a25 as 3021 verbatim, plus
  a26 3021 integrity: seal match AND verdict ==
      injection_mixed_qwen AND anchors ok
  a27 chain identity: med p_m (4 dec) == 0.686
      (sealed 3021 T2a).

Tags: Omega-P2p / L3 MLP relay / SwiGLU neuron
attribution / coalition Jaccard vs random null /
cross-phase linkage to 3019 suppression field / no
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
D_3021 = os.path.join(BASE, 'phase3021',
                      'omega_p2o_injection_anatomy_qwen')
F_3019NPZ = os.path.join(
    D_3019, 'omega_p2m_mlp_band_identity_qwen.npz')
OUT = os.path.join(BASE, 'phase3022',
                   'omega_p2p_l3_relay_neurons_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896
SEED_RND = 3009
K_GEN = 256
MIN_POS = 8
L3_GATED = 3
L_RELAY = 3             # attribution layer
L_ERR = 4               # error vector layer (e4)
G7_HEAD = 7
TOPK = 32
N_JAC_NULL = 2000
P_GATE = 0.01
JAC_MULT = 2.0
IDENT_GATE = 0.05
MASS_MIN = 0.05
E4_3020 = 3.6017        # sealed 3020 T2b med_e4 logic
PM_3021 = 0.686         # sealed 3021 T2a p_m_med
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
    'mode': 'qwen3-4b; 3021 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol (SEED_RND=3009 explicit '
            'rebuild); T2 replaced by SwiGLU neuron '
            'attribution of the L3 MLP relay: s_j = 2 '
            'dh_j (w_j.e4)/||e4||^2, dh = down_proj '
            'input diff at L3, e4 = residual error '
            'entering L4; sum_j s_j == 2 dM.e4/||e4||^2 '
            'identity with bf16 gate 0.05',
    'question': 'WHO inside the L3 MLP carries the 69% '
                'relay (3021 p_m 0.686) - sparse '
                'coalition or distributed field, '
                'tag-consistent, and the SAME neuron '
                'set as the sealed 3019 mid-band '
                'suppression field top-32 at L10 or a '
                'new dedicated coalition?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain; '
          'L3 down_proj input + MLP output + residual '
          'profile at step-2',
    'T2a': 'concentration: per tag negmass/posmass of '
           's_j; conc_dom = |s| share of top-32 in the '
           'dominant direction; coalition = those 32 '
           'neurons; bf16 identity gate; chain anchors '
           'a25 (med ||e4|| == 3.6017) and a27 (med '
           'p_m == 0.686); PERMUTATION-INVARIANT null '
           'explicitly banned (3021 lesson)',
    'T2b': 'coalition tests (identity-sensitive): (i) '
           'cross-tag Jaccard of L3 coalitions (55 '
           'pairs) vs random-32-subset null N=2000 '
           '(seed SEED_NULL+3022) one-sided; (ii) '
           'cross-phase Jaccard vs sealed 3019 top-32 '
           'negative sets at L10 (121 pairs, same '
           'null)',
    'T2c': 'DESCRIPTIVE specificity: content/sham '
           'positions; med negmass at L3; spec_ratio '
           '= med negmass_content / med '
           'negmass_logic; sham JS calibration',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => relay_undetermined_void; '
               'jac_cross_med > 2*jac_null_med and '
               'p_cross <= 0.01 => '
               'relay_shares_suppression_field_qwen; '
               'elif jac_tag_med > 2*jac_null_med and '
               'p_tag <= 0.01 => '
               'relay_dedicated_coalition_qwen; else '
               '=> relay_distributed_qwen',
    'tags': 'Omega-P2p / L3 MLP relay / SwiGLU neuron '
            'attribution / coalition Jaccard vs random '
            'null / cross-phase linkage to 3019 '
            'suppression field / no hallucination '
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


def jac(a, b):
    A = set(int(x) for x in a)
    B = set(int(x) for x in b)
    return len(A & B) / max(len(A | B), 1)


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
        json.dump({'phase': 3022,
                   'name': 'omega_p2p_l3_relay_'
                           'neurons_qwen',
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
                       's3019npz': sha8(F_3019NPZ),
                       's3020result': sha8(
                           D_3020 + r'\result.json'),
                       's3020seal': sha8(
                           D_3020 + r'\seal.json'),
                       's3021result': sha8(
                           D_3021 + r'\result.json'),
                       's3021seal': sha8(
                           D_3021 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'l_relay': L_RELAY,
                   'l_err': L_ERR,
                   'g7_head': G7_HEAD,
                   'topk': TOPK,
                   'n_jac_null': N_JAC_NULL,
                   'p_gate': P_GATE,
                   'jac_mult': JAC_MULT,
                   'ident_gate': IDENT_GATE,
                   'mass_min': MASS_MIN,
                   'e4_3020': E4_3020,
                   'pm_3021': PM_3021,
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

    seal21 = json.load(open(D_3021 + r'\seal.json',
                            encoding='utf-8'))
    r21 = json.load(open(D_3021 + r'\result.json',
                         encoding='utf-8'))
    a26_ok = bool(seal21['result_sha256_8']
                  == sha8(D_3021 + r'\result.json')
                  and r21['final_verdict']
                  == 'injection_mixed_qwen'
                  and r21['anchor_all_ok'] is True)
    log('a26 3021 integrity %s (verdict=%s)'
        % (a26_ok, r21['final_verdict']), lines)

    # ---------- 3019 coalition source ----------
    z19 = np.load(F_3019NPZ, allow_pickle=True)
    s19_prim = z19['s_prim'].astype(np.float64)
    tags19 = [str(t) for t in z19['tags']]
    if s19_prim.ndim == 2 and s19_prim.shape[0]:
        sets19 = [np.argpartition(s19_prim[k],
                                  TOPK - 1)[:TOPK]
                  for k in range(s19_prim.shape[0])]
    else:
        sets19 = []
    log('3019 coalition source: s_prim %s tags=%d'
        % (s19_prim.shape, len(sets19)), lines)

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
    state_h = {'on': False}
    hcap = {}
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

    def pre_down(li):
        def h(module, args, kwargs):
            if state_h['on']:
                x = args[0] if args \
                    else kwargs.get('hidden_states')
                if x is not None and x.dim() >= 2:
                    hcap.setdefault(li, []).append(
                        x[:, -1, :].detach().float()
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
    handles.append(layers[L_RELAY].mlp.down_proj
                   .register_forward_pre_hook(
                       pre_down(L_RELAY),
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
        each layer at step-2 [36,2560], L3 down_proj
        input [inter], L3 mlp output [2560])."""
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        ao.clear()
        mo.clear()
        rs.clear()
        hcap.clear()
        state_c['on'] = True
        state_r['on'] = True
        state_h['on'] = True
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
        state_h['on'] = False
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        res36 = np.stack(
            [rs[li][-1][0].astype(np.float64)
             for li in range(NL)])
        h3 = hcap[L_RELAY][-1][0].astype(np.float64)
        m3 = mo[L_RELAY][-1][0].astype(np.float64)
        return p_, res36, h3, m3

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
                         and a23_ok and a24_ok
                         and a26_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    a25_diff = None
    a27_diff = None
    tags = []
    nL = nC = nS = 0
    js_final = {'logic': [], 'content': [],
                'sham': []}
    e4_store = {'logic': [], 'content': [],
                'sham': []}
    s_store = {'logic': [], 'content': [],
               'sham': []}
    nm_store = {'logic': [], 'content': [],
                'sham': []}
    pm_store = {'logic': [], 'content': [],
                'sham': []}
    pmraw_store = {'logic': [], 'content': [],
                   'sham': []}
    conc_store = {'logic': [], 'content': [],
                  'sham': []}
    dir_store = {'logic': [], 'content': [],
                 'sham': []}
    ident_store = []
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
                q_b, res_b, h_b, m_b = \
                    run_chain(pr, 0, False)
                base[pi] = {'res': res_b,
                            'h': h_b, 'm': m_b}
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

            # W_down at L3: [2560, inter]; column j =
            # w_j; e @ W_down gives w_j.e for all j
            Wd = layers[L_RELAY].mlp.down_proj.weight
            inter = int(Wd.shape[1])
            Wf = Wd.detach().float()
            assert inter == s19_prim.shape[1], \
                (inter, s19_prim.shape)

            def attribute(pi, p_pos):
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                q_e, res_e, h_e, m_e = \
                    run_chain(pr, p_pos, True)
                js_f = js_nats(pb, q_e)
                e4 = res_e[L_ERR] \
                    - base[pi]['res'][L_ERR]
                dh = h_e - base[pi]['h']
                dm = m_e - base[pi]['m']
                ne2 = max(float(e4 @ e4), 1e-30)
                u = (torch.tensor(
                    e4, dtype=torch.float32,
                    device='cuda') @ Wf) \
                    .cpu().numpy().astype(np.float64)
                s = 2.0 * dh * u / ne2
                p_m = float(dm @ e4) / ne2
                ref = 2.0 * p_m
                scale = 2.0 * float(
                    np.linalg.norm(dm)) * float(
                    np.linalg.norm(e4)) / ne2
                ident = abs(float(np.sum(s)) - ref) \
                    / max(scale, 1e-30)
                nm = float(-np.sum(
                    np.minimum(s, 0.0)))
                pmas = float(np.sum(
                    np.maximum(s, 0.0)))
                if nm >= pmas:
                    dirt = 'neg'
                    idx = np.argpartition(
                        s, TOPK - 1)[:TOPK]
                else:
                    dirt = 'pos'
                    idx = np.argpartition(
                        -s, TOPK - 1)[:TOPK]
                mass = nm + pmas
                conc = float(np.abs(s[idx]).sum()) \
                    / max(mass, 1e-30)
                return {'js': js_f, 'e4n':
                        float(np.linalg.norm(e4)),
                        's': s, 'p_m': p_m,
                        'ident': ident, 'nm': nm,
                        'pmass': pmas, 'dir': dirt,
                        'idx': idx, 'conc': conc,
                        'mass': mass}

            res_l = {}
            for ki, t in enumerate(tags):
                pi = int(t.split(':')[0][1:])
                p_pos = int(t.split(':')[1])
                d = attribute(pi, p_pos)
                res_l[t] = d
                js_final['logic'].append(d['js'])
                e4_store['logic'].append(d['e4n'])
                s_store['logic'].append(d['s'])
                nm_store['logic'].append(d['nm'])
                pm_store['logic'].append(d['pmass'])
                pmraw_store['logic'].append(d['p_m'])
                conc_store['logic'].append(d['conc'])
                dir_store['logic'].append(d['dir'])
                ident_store.append(d['ident'])
                log('attr ki=%d (%s) js=%.5f e4n=%.4f '
                    'p_m=%.4f nm=%.4f pmass=%.4f '
                    'dir=%s conc=%.4f idf=%.2e'
                    % (ki, t, d['js'], d['e4n'],
                       d['p_m'], d['nm'],
                       d['pmass'], d['dir'],
                       d['conc'], d['ident']), lines)
            # content + sham
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                cps = ent['positions']['content']
                if cps:
                    pc = int(cps[0])
                    d = attribute(pi, pc)
                    js_final['content'].append(
                        d['js'])
                    e4_store['content'].append(
                        d['e4n'])
                    s_store['content'].append(d['s'])
                    nm_store['content'].append(
                        d['nm'])
                    pm_store['content'].append(
                        d['pmass'])
                    pmraw_store['content'].append(
                        d['p_m'])
                    conc_store['content'].append(
                        d['conc'])
                    dir_store['content'].append(
                        d['dir'])
                    nC += 1
                p_sh = ent['positions']['sham']
                if p_sh is not None:
                    d = attribute(pi, p_sh)
                    js_final['sham'].append(d['js'])
                    e4_store['sham'].append(d['e4n'])
                    s_store['sham'].append(d['s'])
                    nm_store['sham'].append(d['nm'])
                    pm_store['sham'].append(
                        d['pmass'])
                    pmraw_store['sham'].append(
                        d['p_m'])
                    conc_store['sham'].append(
                        d['conc'])
                    dir_store['sham'].append(d['dir'])
                    nS += 1
            js_e_a = np.array(js_final['logic'])

            # ---------- chain anchors ----------
            a25_diff = abs(round(float(np.median(
                e4_store['logic'])), 4) - E4_3020)
            a25_ok = bool(a25_diff < 1e-4)
            log('a25 med e4 %.4f vs 3020 %.4f diff=%.2e '
                'ok=%s' % (float(np.median(
                    e4_store['logic'])), E4_3020,
                    a25_diff, a25_ok), lines)
            a27_diff = abs(round(float(np.median(
                pmraw_store['logic'])), 4) - PM_3021)
            a27_ok = bool(a27_diff < 1e-4)
            log('a27 med p_m %.4f vs 3021 %.4f '
                'diff=%.2e ok=%s'
                % (float(np.median(
                    pmraw_store['logic'])), PM_3021,
                    a27_diff, a27_ok), lines)

            # ---------- T2a concentration ------
            ident_med = float(np.median(ident_store))
            n_mass = sum(1 for t in tags
                         if res_l[t]['mass']
                         > MASS_MIN)
            conc_med = float(np.median(
                conc_store['logic'])) \
                if conc_store['logic'] else float('nan')
            nm_med = float(np.median(
                nm_store['logic'])) \
                if nm_store['logic'] else float('nan')
            pmas_med = float(np.median(
                pm_store['logic'])) \
                if pm_store['logic'] else float('nan')
            dir_counts = {}
            for d_ in dir_store['logic']:
                dir_counts[d_] = \
                    dir_counts.get(d_, 0) + 1
            T2a = {
                'n_logic': nL, 'n_content': nC,
                'n_sham': nS,
                'l_relay': L_RELAY, 'l_err': L_ERR,
                'inter': inter, 'topk': TOPK,
                'n_mass': n_mass,
                'mass_min': MASS_MIN,
                'bf16_ident_med':
                    float('%.4e' % ident_med),
                'ident_gate': IDENT_GATE,
                'conc_dom_med': round(conc_med, 4)
                if np.isfinite(conc_med) else None,
                'med_negmass': round(nm_med, 4)
                if np.isfinite(nm_med) else None,
                'med_posmass': round(pmas_med, 4)
                if np.isfinite(pmas_med) else None,
                'direction_counts': dir_counts,
                'med_p_m_recomputed': round(
                    float(np.median(
                        pmraw_store['logic'])), 4),
                'med_e4_norm': round(float(
                    np.median(e4_store['logic'])), 4),
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
                'conc_per_tag': [round(float(v), 4)
                                 for v in
                                 conc_store['logic']],
                'negmass_per_tag': [round(float(v), 4)
                                    for v in
                                    nm_store['logic']],
                'posmass_per_tag': [round(float(v), 4)
                                    for v in
                                    pm_store['logic']],
                'directions': dir_store['logic'],
                'tags': tags,
                'note': 's_j = 2 dh_j (w_j.e4)/'
                        '||e4||^2 at L3; coalition = '
                        'top-32 in the dominant '
                        'direction; permutation-'
                        'invariant null banned '
                        '(3021 lesson)'}
            log('T2a conc=%.4f nm=%.4f pmass=%.4f '
                'dirs=%s idf=%.2e n_mass=%d'
                % (conc_med, nm_med, pmas_med,
                   dir_counts, ident_med, n_mass),
                lines)

            # ---------- T2b coalition tests ------
            sets22 = [res_l[t]['idx'] for t in tags]
            jac_tag = []
            for a_ in range(len(sets22)):
                for b_ in range(a_ + 1,
                                len(sets22)):
                    jac_tag.append(
                        jac(sets22[a_], sets22[b_]))
            jac_tag_med = float(np.median(jac_tag)) \
                if jac_tag else float('nan')
            # cross-phase vs 3019 L10 top-32 neg sets
            jac_cross = []
            jac_same = []
            tmap19 = {t: i for i, t
                      in enumerate(tags19)}
            for a_ in range(len(sets22)):
                for b_ in range(len(sets19)):
                    j_ = jac(sets22[a_], sets19[b_])
                    jac_cross.append(j_)
                    if tags[a_] in tmap19 \
                            and tmap19[tags[a_]] \
                            == b_:
                        jac_same.append(j_)
            jac_cross_med = float(
                np.median(jac_cross)) \
                if jac_cross else float('nan')
            jac_same_med = float(
                np.median(jac_same)) \
                if jac_same else None
            # random-32-subset null
            rng_j = np.random.default_rng(
                SEED_NULL + 3022)
            null_j = np.empty(N_JAC_NULL)
            for b_ in range(N_JAC_NULL):
                sa = rng_j.choice(inter, TOPK,
                                  replace=False)
                sb = rng_j.choice(inter, TOPK,
                                  replace=False)
                null_j[b_] = jac(sa, sb)
            jac_null_med = float(np.median(null_j))
            p_tag = float(np.mean(null_j
                                  >= jac_tag_med)) \
                if np.isfinite(jac_tag_med) else None
            p_cross = float(np.mean(null_j
                                    >= jac_cross_med)) \
                if np.isfinite(jac_cross_med) \
                else None
            T2b = {
                'jac_tag_med': round(jac_tag_med, 4)
                if np.isfinite(jac_tag_med)
                else None,
                'jac_cross_med': round(
                    jac_cross_med, 4)
                if np.isfinite(jac_cross_med)
                else None,
                'jac_same_tag_med': round(
                    jac_same_med, 4)
                if jac_same_med is not None
                else None,
                'jac_null_med': round(
                    jac_null_med, 4),
                'jac_mult_gate': JAC_MULT,
                'p_tag': round(p_tag, 4)
                if p_tag is not None else None,
                'p_cross': round(p_cross, 4)
                if p_cross is not None else None,
                'p_gate': P_GATE,
                'n_pairs_tag': len(jac_tag),
                'n_pairs_cross': len(jac_cross),
                'n_3019_sets': len(sets19),
                'null_seed': SEED_NULL + 3022,
                'note': 'identity-sensitive '
                        'coalition tests; null = '
                        'random 32-subsets of '
                        '[inter]'}
            log('T2b jac_tag=%.4f jac_cross=%.4f '
                'jac_same=%s null=%.4f p_tag=%s '
                'p_cross=%s'
                % (jac_tag_med, jac_cross_med,
                   jac_same_med, jac_null_med,
                   p_tag, p_cross), lines)

            # ---------- T2c specificity ----------
            def medv(x):
                return round(float(np.median(x)), 4) \
                    if x else None

            nm_c = float(np.median(
                nm_store['content'])) \
                if nm_store['content'] else None
            spec_ratio = None
            if nm_c is not None \
                    and np.isfinite(nm_med) \
                    and nm_med > 0.01:
                spec_ratio = nm_c / nm_med
            T2c = {
                'med_negmass': {
                    'logic': round(nm_med, 4)
                    if np.isfinite(nm_med) else None,
                    'content': medv(
                        nm_store['content']),
                    'sham': medv(nm_store['sham'])},
                'med_posmass': {
                    'logic': round(pmas_med, 4)
                    if np.isfinite(pmas_med)
                    else None,
                    'content': medv(
                        pm_store['content']),
                    'sham': medv(pm_store['sham'])},
                'spec_ratio': round(spec_ratio, 4)
                if spec_ratio is not None else None,
                'med_e4_norm_content': medv(
                    e4_store['content']),
                'med_e4_norm_sham': medv(
                    e4_store['sham']),
                'note': 'descriptive specificity of '
                        'the L3 relay'}
            log('T2c nm L/C/S=%s/%s/%s spec=%s'
                % (T2c['med_negmass']['logic'],
                   T2c['med_negmass']['content'],
                   T2c['med_negmass']['sham'],
                   T2c['spec_ratio']), lines)

            # ---------- verdict ----------
            gates_ok = bool(nL >= MIN_POS
                            and np.all(js_e_a > 0)
                            and len(ident_store) > 0
                            and ident_med
                            < IDENT_GATE
                            and n_mass >= MIN_POS)
            if not gates_ok:
                verdict = 'relay_undetermined_void'
            elif (np.isfinite(jac_cross_med)
                  and jac_cross_med
                  > JAC_MULT * jac_null_med
                  and p_cross is not None
                  and p_cross <= P_GATE):
                verdict = 'relay_shares_' \
                          'suppression_field_qwen'
            elif (np.isfinite(jac_tag_med)
                  and jac_tag_med
                  > JAC_MULT * jac_null_med
                  and p_tag is not None
                  and p_tag <= P_GATE):
                verdict = 'relay_dedicated_' \
                          'coalition_qwen'
            else:
                verdict = 'relay_distributed_qwen'

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
        'a24_3020': a24_ok, 'a26_3021': a26_ok,
        'a25_e4_chain_diff': a25_diff,
        'a27_pm_chain_diff': a27_diff,
    }
    a25_ok_f = bool(a25_diff is not None
                    and a25_diff < 1e-4)
    a27_ok_f = bool(a27_diff is not None
                    and a27_diff < 1e-4)
    res = {
        'phase': 3022,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok and a25_ok_f
                              and a27_ok_f),
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

    s_mat = np.stack(s_store['logic']) \
        if s_store['logic'] else np.array([])
    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35,
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object),
            'tags': np.array(tags, dtype=object),
            'tags19': np.array(tags19, dtype=object),
            'js_final_logic': js_e_a,
            'js_final_content': np.array(
                js_final['content']),
            'js_final_sham': np.array(
                js_final['sham']),
            'e4_norm': {ty: np.array(e4_store[ty])
                        for ty in e4_store},
            's_relay': s_mat.astype(np.float32)
            if s_mat.size else np.array([]),
            'p_m_recomputed': {ty:
                               np.array(pmraw_store[ty])
                               for ty in pmraw_store},
            'negmass': {ty: np.array(nm_store[ty])
                        for ty in nm_store},
            'posmass': {ty: np.array(pm_store[ty])
                        for ty in pm_store},
            'conc_dom': {ty: np.array(conc_store[ty])
                         for ty in conc_store},
            'directions': {
                ty: np.array(dir_store[ty],
                             dtype=object)
                for ty in dir_store},
            'ident_vals': np.array(ident_store),
            'jac_null': null_j if anchor_prelim
            and verdict != 'anchor_fail_all_void'
            else np.array([])}
    npz_path = os.path.join(
        OUT, 'omega_p2p_l3_relay_neurons_qwen.npz')
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
