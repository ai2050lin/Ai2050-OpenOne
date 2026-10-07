# -*- coding: utf-8 -*-
"""Phase 3030: Omega-P2x readout-level convexity probe
(qwen).

Why: 3028 found the coalition-channel dose response is
SUPERLINEAR (js at alpha=2 is 2.18x the linear
prediction).  3029 then showed the convexity is NOT in
the head carriers: zero recruitment (med rho 0.036,
deep band exactly 0.0) and sublinear per-unit growth
(med g 0.671 < 1).  Two-sided squeeze => the excess
must be created in the readout path (residual ->
final norm -> lm_head -> softmax).  Open question:
WHERE along depth does the convexity enter the lens
trajectory - immediately at the first divergent layer
(readout map is intrinsically convex, 3020's L4
instant readability) or accumulated across mid
layers?

Design (3029 machine verbatim for anchors/chains/
generation/position selection/two-step protocol; the
added residual captures are observation-only and bit-
identity is re-verified by a28/a30/a32):
  Per logic tag, three dose arms alpha in {0, 1, 2}
  (identical chains to 3029), each capturing the
  step-2 residual stack r_a (36 x 2560, decoder-layer
  pre-hook) and the step-2 final-norm input xfin_a.
  Baseline no-erase capture gives r_b / xfin_b.
  Logit-lens EXACTLY as 3020 (lens_start 4, lens_end
  35; bf16 RMSNorm + lm_head, float log_softmax; js on
  exp), 32-point trajectory per arm:
    jsl_l(alpha) = js(lens(x^alpha_l), lens(x^b_l))
  PRIMARY per tag: second-order excess per lens point
    excess_l = jsl_l(2) - 2*jsl_l(1) + jsl_l(0)
  Per-layer noise floor theta_l = med over sham
  chains of their alpha=1 trajectory (same-chain
  calibration, fixing the 3029 calibration flaw); a
  layer counts only if jsl_l(2) > 2*theta_l.
  Terminal share S = excess at the xfin endpoint /
  sum of positive excess over counted layers.
  a39: alpha=1 trajectory vs 3020 npz traj_logic
  bit-level 0.0 (chain identity across phases;
  rows aligned by 3020 traj_keys_logic - the 3020
  npz stores traj rows in dict order, not tags
  order).  a38: lens-terminal consistency
  |jsl_xfin(1) - js_erase| / js_erase < 1e-4
  (run1 measured 1.58e-6 from the float64 softmax
  normalization path, not machine drift; gate
  recalibrated from 1e-6).

Verdict (frozen):
  anchor fail                     => anchor_fail_
                                    all_void
  gates fail (nL<8 or any js_erase<=0 or counted-tag
  deficiency or a38 fail)         => readout_convex_
                                    undetermined_void
  med S >= 0.7                    => readout_terminal_
                                    convex_qwen
  med S <= 0.3                    => readout_
                                    distributed_convex_
                                    qwen
  else                            => readout_convex_
                                    mixed_qwen

Anchors (frozen): a0-a32 as 3029 verbatim (a28 erase
chain vs 3022 npz bit-level 0.0; a30 capture self-
consistency 0.0; a32 alpha=0 vs 3024 npz
js_restore_coal bit-level 0.0); a33 3028 integrity;
a34 3029 integrity (seal match AND verdict
amplify_existing_dominant_qwen AND anchors ok); a38
lens-terminal consistency; a39 lens trajectory vs
3020 npz bit-level 0.0.

Tags: Omega-P2x / readout convexity localization /
per-layer lens excess on the dose gradient /
sham-calibrated per-layer floor / no hallucination
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
F_3020NPZ = os.path.join(
    D_3020, 'omega_p2n_readout_specificity_qwen.npz')
D_3021 = os.path.join(BASE, 'phase3021',
                      'omega_p2o_injection_anatomy_qwen')
D_3022 = os.path.join(BASE, 'phase3022',
                      'omega_p2p_l3_relay_neurons_qwen')
D_3023 = os.path.join(BASE, 'phase3023',
                      'omega_p2q_relay_causal_ablation_'
                      'qwen')
D_3024 = os.path.join(BASE, 'phase3024',
                      'omega_p2r_restore_ablation_qwen')
F_3022NPZ = os.path.join(
    D_3022, 'omega_p2p_l3_relay_neurons_qwen.npz')
F_3024NPZ = os.path.join(
    D_3024, 'omega_p2r_restore_ablation_qwen.npz')
D_3028 = os.path.join(BASE, 'phase3028',
                      'omega_p2v_dose_symmetry_qwen')
D_3029 = os.path.join(BASE, 'phase3029',
                      'omega_p2w_recruitment_decomp_'
                      'qwen')
OUT = os.path.join(BASE, 'phase3030',
                   'omega_p2x_readout_convexity_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896
SEED_RND = 3009
K_GEN = 256
MIN_POS = 8
L3_GATED = 3
L_RELAY = 3
G7_HEAD = 7
TOPK = 32
HDIM = 128
NHQ = 32
L_MIN = 4
LENS_START = 4
LENS_END = 35
TERM_SHARE = 0.7
DIST_SHARE = 0.3
REST_MIN = 1e-6
LBANDS = {'seed': (0, 4), 'mid': (4, 17),
          'deep': (17, 31)}
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
    'mode': 'qwen3-4b; 3029 machine verbatim for anchors/'
            'chains/generation/position selection/'
            'two-step protocol + observation-only '
            'residual/xfin captures (bit-identity '
            're-verified by a28/a30/a32); logit-lens '
            'exactly as 3020 (start 4, end 35, bf16 '
            'norm+lm_head, float log_softmax)',
    'question': '3028 found convexity (js at alpha=2 = '
                '2.18x linear prediction) and 3029 '
                'squeezed it out of the head carriers '
                '(zero recruitment, sublinear g). '
                'WHERE along depth does the convexity '
                'enter the lens trajectory - '
                'immediately at the first divergent '
                'layers (readout map intrinsically '
                'convex) or accumulated across mid '
                'layers?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain; '
          'coalition sets reconstructed from the sealed '
          '3022 npz s_relay (per-tag top-32 positive); '
          'tags asserted equal to 3022 AND 3020 order; '
          'per-tag no-erase capture (h_base for the '
          'dose hook + o_base + residual stack r_b + '
          'xfin_b); a30 self-consistency js(pb, p0) == '
          '0.0 bit-level',
    'T2a': 'PRIMARY: arms per logic tag = ERASE (alpha'
           '=1) / DOSE alpha in {0, 2} (chains '
           'bit-identical to 3029); per-arm step-2 '
           'residual stack + xfin; lens trajectory '
           'jsl_l(alpha) = js(lens(x^a_l), lens(x^b_l)) '
           'for l in 4..34 plus xfin (32 points); '
           'second-order excess excess_l = jsl(2) - '
           '2*jsl(1) + jsl(0); per-layer floor theta_l '
           '= med sham alpha=1 trajectory (same-chain '
           'calibration); layer counted iff jsl(2)_l > '
           '2*theta_l; terminal share S = excess_xfin / '
           'sum positive counted excess; a28/a30/a32 '
           'bit-level 0.0; a39 alpha=1 trajectory vs '
           '3020 npz traj_logic bit-level 0.0 (rows '
           'aligned by 3020 traj_keys_logic); a38 '
           'lens-terminal consistency rel < 1e-4',
    'corrections': 'run1 anchors a39/a38 '
                   'miscalibrated: a39 compared rows '
                   'in tags order but 3020 npz stores '
                   'traj rows in traj_keys_logic dict '
                   'order (key-aligned diff is 0.0); '
                   'a38 gate 1e-6 was tighter than '
                   'the float64 softmax normalization '
                   'noise (measured 1.58e-6); verdict '
                   'mapping unchanged',
    'T2b': 'DESCRIPTIVE: positive-excess share per lens '
           'band (seed idx 0-3 = L4-7 / mid idx 4-16 = '
           'L8-20 / deep idx 17-30 = L21-34 / terminal '
           'idx 31 = xfin); per-band jsl(2)/jsl(1) '
           'ratio med; per-tag argmax-excess layer',
    'T2c': 'DESCRIPTIVE: content arms (tref convention) '
           'terminal lens JS and S_c; sham calibration '
           '(med js and per-layer floor)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => readout_convex_'
               'undetermined_void; med S >= 0.7 => '
               'readout_terminal_convex_qwen; med S <= '
               '0.3 => readout_distributed_convex_qwen; '
               'else => readout_convex_mixed_qwen',
    'tags': 'Omega-P2x / readout convexity '
            'localization / per-layer lens excess on '
            'the dose gradient / sham-calibrated '
            'per-layer floor / no hallucination naming',
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
        json.dump({'phase': 3030,
                   'name': 'omega_p2x_readout_'
                           'convexity_qwen',
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
                           D_3020 + r'\seal.json'),
                       's3020npz': sha8(F_3020NPZ),
                       's3021result': sha8(
                           D_3021 + r'\result.json'),
                       's3021seal': sha8(
                           D_3021 + r'\seal.json'),
                       's3022result': sha8(
                           D_3022 + r'\result.json'),
                       's3022seal': sha8(
                           D_3022 + r'\seal.json'),
                       's3022npz': sha8(F_3022NPZ),
                       's3023result': sha8(
                           D_3023 + r'\result.json'),
                       's3023seal': sha8(
                           D_3023 + r'\seal.json'),
                       's3024result': sha8(
                           D_3024 + r'\result.json'),
                       's3024seal': sha8(
                           D_3024 + r'\seal.json'),
                       's3024npz': sha8(F_3024NPZ),
                       's3028result': sha8(
                           D_3028 + r'\result.json'),
                       's3028seal': sha8(
                           D_3028 + r'\seal.json'),
                       's3029result': sha8(
                           D_3029 + r'\result.json'),
                       's3029seal': sha8(
                           D_3029 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'l_relay': L_RELAY,
                   'g7_head': G7_HEAD,
                   'topk': TOPK,
                   'hdim': HDIM, 'nhq': NHQ,
                   'l_min': L_MIN,
                   'lens_start': LENS_START,
                   'lens_end': LENS_END,
                   'term_share': TERM_SHARE,
                   'dist_share': DIST_SHARE,
                   'rest_min': REST_MIN,
                   'lens_bands': {k: list(v)
                                  for k, v
                                  in LBANDS.items()},
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

    seal22 = json.load(open(D_3022 + r'\seal.json',
                            encoding='utf-8'))
    r22 = json.load(open(D_3022 + r'\result.json',
                         encoding='utf-8'))
    a27_ok = bool(seal22['result_sha256_8']
                  == sha8(D_3022 + r'\result.json')
                  and r22['final_verdict']
                  == 'relay_dedicated_coalition_qwen'
                  and r22['anchor_all_ok'] is True)
    log('a27 3022 integrity %s (verdict=%s)'
        % (a27_ok, r22['final_verdict']), lines)

    seal23 = json.load(open(D_3023 + r'\seal.json',
                            encoding='utf-8'))
    r23 = json.load(open(D_3023 + r'\result.json',
                         encoding='utf-8'))
    a29_ok = bool(seal23['result_sha256_8']
                  == sha8(D_3023 + r'\result.json')
                  and r23['final_verdict']
                  == 'relay_causal_toxic_void'
                  and r23['anchor_all_ok'] is True)
    log('a29 3023 integrity %s (verdict=%s)'
        % (a29_ok, r23['final_verdict']), lines)

    seal24 = json.load(open(D_3024 + r'\seal.json',
                            encoding='utf-8'))
    r24 = json.load(open(D_3024 + r'\result.json',
                         encoding='utf-8'))
    a31_ok = bool(seal24['result_sha256_8']
                  == sha8(D_3024 + r'\result.json')
                  and r24['final_verdict']
                  == 'relay_restore_load_bearing_qwen'
                  and r24['anchor_all_ok'] is True)
    log('a31 3024 integrity %s (verdict=%s)'
        % (a31_ok, r24['final_verdict']), lines)

    seal28 = json.load(open(D_3028 + r'\seal.json',
                            encoding='utf-8'))
    r28 = json.load(open(D_3028 + r'\result.json',
                         encoding='utf-8'))
    a33_ok = bool(seal28['result_sha256_8']
                  == sha8(D_3028 + r'\result.json')
                  and r28['final_verdict']
                  == 'dose_superlinear_qwen'
                  and r28['anchor_all_ok'] is True)
    log('a33 3028 integrity %s (verdict=%s)'
        % (a33_ok, r28['final_verdict']), lines)

    seal29 = json.load(open(D_3029 + r'\seal.json',
                            encoding='utf-8'))
    r29 = json.load(open(D_3029 + r'\result.json',
                         encoding='utf-8'))
    a34_ok = bool(seal29['result_sha256_8']
                  == sha8(D_3029 + r'\result.json')
                  and r29['final_verdict']
                  == 'amplify_existing_dominant_qwen'
                  and r29['anchor_all_ok'] is True)
    log('a34 3029 integrity %s (verdict=%s)'
        % (a34_ok, r29['final_verdict']), lines)

    # ---------- 3022 coalition source ----------
    z22 = np.load(F_3022NPZ, allow_pickle=True)
    s22 = z22['s_relay'].astype(np.float64)
    tags22 = [str(t) for t in z22['tags']]
    js22 = z22['js_final_logic'].astype(np.float64)
    dirs_raw = z22['directions']
    if dirs_raw.shape == ():
        dirs_raw = dirs_raw.item()['logic']
    dirs22 = [str(d) for d in dirs_raw]
    coal_sets = []
    for k in range(s22.shape[0]):
        assert dirs22[k] == 'pos', dirs22[k]
        coal_sets.append(np.argpartition(
            -s22[k], TOPK - 1)[:TOPK])
    log('3022 coalition source: s_relay %s tags=%d '
        'dirs all pos=%s'
        % (s22.shape, len(coal_sets),
           all(d == 'pos' for d in dirs22)), lines)

    # ---------- 3024 dose anchor source ----------
    z24 = np.load(F_3024NPZ, allow_pickle=True)
    js24_coal = z24['js_restore_coal'].astype(
        np.float64)
    assert list(z24['tags'].astype(str)) == tags22
    log('3024 npz js_restore_coal loaded %s'
        % (js24_coal.shape,), lines)

    # ---------- 3020 lens trajectory source ------
    z20 = np.load(F_3020NPZ, allow_pickle=True)
    traj20 = z20['traj_logic'].astype(np.float64)
    tags20 = [str(t) for t in z20['tags']]
    keys20 = [str(t) for t in
              z20['traj_keys_logic']]
    log('3020 npz traj_logic loaded %s tags_match_3022='
        '%s rows_follow_keys=%s'
        % (traj20.shape, tags20 == tags22,
           keys20 != tags20), lines)

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
    state_r = {'on': False}
    rs = {}
    state_hcap = {'on': False}
    o_in = {}
    state_o = {'on': False}
    state_rest = {'active': False, 'idx': None,
                  'alpha': None, 'h_base': None,
                  'onorm': 0.0, 'mag': 0.0}
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

    def pre_oproj(li):
        def h(module, args, kwargs):
            if state_o['on']:
                x = args[0]
                o_in.setdefault(li, []).append(
                    x[:, -1, :].detach().float()
                    .cpu().numpy().copy())
            return None
        return h

    Wd = layers[L_RELAY].mlp.down_proj.weight
    inter = int(Wd.shape[1])
    assert inter == s22.shape[1], \
        (inter, s22.shape)

    def hook_hcap(module, args, kwargs):
        if state_hcap['on']:
            state_rest['h_base'] = \
                args[0].detach().clone()
        return None

    def hook_rest(module, args, output):
        if not state_rest['active']:
            return None
        h = args[0]
        idx = state_rest['idx']
        alpha = state_rest['alpha']
        cols = Wd[:, idx]
        d_cur = h[:, :, idx] @ cols.T
        d_base = state_rest['h_base'][:, :, idx] \
            @ cols.T
        if alpha == 0.0:
            patched = output - d_cur + d_base
        else:
            patched = output - d_cur \
                + (d_base + alpha
                   * (d_cur - d_base))
        dnet = (alpha - 1.0) * (d_cur - d_base)
        state_rest['mag'] = float(
            dnet[0, -1].float().norm())
        state_rest['onorm'] = float(
            output[0, -1].float().norm())
        return patched

    handles.append(layers[L_RELAY].mlp.down_proj
                   .register_forward_pre_hook(
                       hook_hcap, with_kwargs=True))
    handles.append(layers[L_RELAY].mlp.down_proj
                   .register_forward_hook(hook_rest))

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           pre_oproj(li),
                           with_kwargs=True))
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

    def grab_o():
        return np.stack(
            [o_in[li][-1][0].astype(np.float64)
             for li in range(NL)])

    def grab_r():
        return np.stack(
            [rs[li][-1][0].astype(np.float64)
             for li in range(NL)])

    def capture_base(pi, p_pos):
        """No-erase two-step chain with h_base + o_base
        + residual stack + xfin capture; returns
        (p0, o_b, r_b, xf_b)."""
        pr = GEN_PROMPTS[pi]
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
        rs.clear()
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        state_hcap['on'] = True
        state_o['on'] = True
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            state_r['on'] = True
            state_fin['on'] = True
            p0, _ = prefill_step2(pr, ids, past)
            state_r['on'] = False
            state_fin['on'] = False
        state_hcap['on'] = False
        state_o['on'] = False
        r_b = grab_r()
        xf_b = fin_cap['x'][0].astype(np.float64)
        return p0, grab_o(), r_b, xf_b

    def run_chain(pr, p_pos, erase, rest=None):
        """Two-step chain, optionally g7-K erasure at
        p_pos and/or coalition DOSE patch at L3
        down_proj (rest = (idx, alpha)).  Captures
        o_proj input + residual stack + xfin of the
        step-2 forward; returns (p_, r_stack, xfin).
        Caller must grab_o() right after (the next
        chain clears o_in)."""
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
        rs.clear()
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        state_rest['mag'] = 0.0
        state_rest['onorm'] = 0.0
        state_o['on'] = True
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if erase:
                kv_scale_arm(past, p_pos, 0.0,
                             'KONLY', G7_HEAD)
            if rest is not None:
                state_rest['active'] = True
                state_rest['idx'] = rest[0]
                state_rest['alpha'] = rest[1]
            state_r['on'] = True
            state_fin['on'] = True
            out2 = model(
                input_ids=torch.tensor(
                    [[int(ids[-1])]], device='cuda'),
                past_key_values=past,
                use_cache=False)
            state_r['on'] = False
            state_fin['on'] = False
        state_o['on'] = False
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        r_stack = grab_r()
        xf = fin_cap['x'][0].astype(np.float64)
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        return p_, r_stack, xf

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
                         and a23_ok and a24_ok and a26_ok
                         and a27_ok and a29_ok
                         and a31_ok and a33_ok
                         and a34_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    a28_diff = None
    a28_ok = False
    a30_diff = None
    a30_ok = False
    a32_diff = None
    a32_ok = False
    a38_rel = None
    a38_ok = False
    a39_diff = None
    a39_ok = False
    tags = []
    nL = nC = nS = 0
    js_er = []
    js0_all = []
    js2_all = []
    mag2_all = []
    traj0_list = []
    traj1_list = []
    traj2_list = []
    rel38_list = []
    theta_l = None
    sham_traj = np.zeros(
        (0, LENS_END - LENS_START + 1))
    js_sham_all = []
    S_list = []
    E_tot_list = []
    lstar_list = []
    band_share = {b: [] for b in LBANDS}
    band_ratio = {b: [] for b in LBANDS}
    a0_arr = np.zeros(0)
    res_c = {}
    t1c_last = []
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
                    rest2 = [i for i in
                             range(len(ids))
                             if i not in lp_use
                             and i not in cp_use]
                    sham_pool = rest2
                sham = int(rng2.choice(sham_pool)) \
                    if sham_pool else None
                ent['positions'] = {
                    'logic': lp_use,
                    'content': [int(x) for x
                                in cp_use],
                    'sham': sham}
                sel['P%d' % pi] = ent

            # ---------- T1 tags + baselines -------
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

            tags = []
            for pi in caps:
                for p_pos in sel['P%d' % pi][
                        'positions']['logic']:
                    tags.append('P%d:%d'
                                % (pi, p_pos))
            nL = len(tags)
            log('T1 logic positions n=%d' % nL, lines)
            assert tags == tags22, (tags, tags22)
            log('tag order == 3022 npz tags OK', lines)
            tags20_match = bool(tags == tags20)
            log('tag order == 3020 npz tags: %s'
                % tags20_match, lines)

            tmap = {t: k for k, t
                    in enumerate(tags22)}

            # ---------- logit-lens machine -------
            # (3020 verbatim: bf16 norm + lm_head,
            # float log_softmax)
            nrm = model.model.norm
            lmh = model.lm_head

            def lens_lp(x64):
                xt = torch.tensor(
                    x64, dtype=torch.float32,
                    device='cuda')[None, :]
                hb = xt.to(torch.bfloat16)
                with torch.no_grad():
                    hn = nrm(hb)
                    lg = lmh(hn)[0]
                lp = torch.log_softmax(
                    lg.float(), dim=-1).detach() \
                    .cpu().numpy() \
                    .astype(np.float64)
                return lp

            def js_lp(lp, lq):
                return js_nats(np.exp(lp),
                               np.exp(lq))

            def traj_of(r_a, xf_a, r_b, xf_b):
                out = np.empty(LENS_END
                               - LENS_START + 1)
                for l in range(LENS_START,
                               LENS_END):
                    out[l - LENS_START] = js_lp(
                        lens_lp(r_a[l]),
                        lens_lp(r_b[l]))
                out[-1] = js_lp(lens_lp(xf_a),
                                lens_lp(xf_b))
                return out

            # ---------- sham lens floor ----------
            # (same-chain calibration, per-layer)
            sham_traj_list = []
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                if ent['positions']['sham'] \
                        is None:
                    continue
                p_sh = int(
                    ent['positions']['sham'])
                pb = caps[pi][1]
                p0s, o_bs, r_bs, xf_bs = \
                    capture_base(pi, p_sh)
                js_sham_all.append(
                    js_nats(pb, p0s))
                q_s, r_s, xf_s = run_chain(
                    pr, p_sh, True, None)
                grab_o()
                sham_traj_list.append(
                    traj_of(r_s, xf_s,
                            r_bs, xf_bs))
                nS += 1
            if sham_traj_list:
                sham_traj = np.stack(
                    sham_traj_list)
                theta_l = np.median(
                    sham_traj, axis=0)
            log('sham n=%d med_js=%.6f theta_l[0]=%s '
                'theta_l[-1]=%s'
                % (nS,
                   float(np.median(js_sham_all))
                   if js_sham_all else -1,
                   '%.2e' % theta_l[0]
                   if theta_l is not None else 'NA',
                   '%.2e' % theta_l[-1]
                   if theta_l is not None else 'NA'),
                lines)

            # ---------- arms + anchors ----------
            def arms_full(pi, p_pos, ck, tref=None):
                """Per tag: baseline capture + 3 arms
                (alpha 1/0/2, chains bit-identical to
                3029) each with residual/xfin capture;
                returns dict with lens trajectories."""
                t = 'P%d:%d' % (pi, p_pos)
                tr = tref if tref is not None \
                    else t
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                cidx = coal_sets[tmap[tr]]
                p0, o_b, r_b, xf_b = \
                    capture_base(pi, p_pos)
                js0 = js_nats(pb, p0)
                q_e, r_e, xf_e = run_chain(
                    pr, p_pos, True, None)
                o_e = grab_o()
                out_er = js_nats(pb, q_e)
                q_0, r_0, xf_0 = run_chain(
                    pr, p_pos, True, (cidx, 0.0))
                o_0 = grab_o()
                js0d = js_nats(pb, q_0)
                q_2, r_2, xf_2 = run_chain(
                    pr, p_pos, True, (cidx, 2.0))
                o_2 = grab_o()
                js2d = js_nats(pb, q_2)
                mag2 = state_rest['mag'] \
                    / max(state_rest['onorm'],
                          1e-30)
                t1 = traj_of(r_e, xf_e, r_b, xf_b)
                t0a = traj_of(r_0, xf_0,
                              r_b, xf_b)
                t2a = traj_of(r_2, xf_2,
                              r_b, xf_b)
                rel38 = abs(t1[-1] - out_er) \
                    / max(out_er, 1e-12)
                log('%s %s js0=%.2e er=%.5f '
                    'a0=%.5f a2=%.5f mag2=%.2e '
                    'rel38=%.1e'
                    % (ck, t, js0, out_er, js0d,
                       js2d, mag2, rel38), lines)
                return {'js0': js0, 'er': out_er,
                        'a0': js0d, 'a2': js2d,
                        'mag2': mag2,
                        't0': t0a, 't1': t1,
                        't2': t2a, 'rel38': rel38}

            res_l = {}
            for t in tags:
                pi = int(t.split(':')[0][1:])
                p_pos = int(t.split(':')[1])
                res_l[t] = arms_full(pi, p_pos, 'L')
                js_er.append(res_l[t]['er'])
                js0_all.append(res_l[t]['js0'])
                js2_all.append(res_l[t]['a2'])
                mag2_all.append(res_l[t]['mag2'])
                traj0_list.append(res_l[t]['t0'])
                traj1_list.append(res_l[t]['t1'])
                traj2_list.append(res_l[t]['t2'])
                rel38_list.append(res_l[t]['rel38'])
            # a28: erase-chain identity vs 3022
            a28_diff = float(np.max(np.abs(
                np.array(js_er) - js22)))
            a28_ok = bool(a28_diff == 0.0)
            log('a28 erase chain vs 3022 js max|d|='
                '%.2e ok=%s' % (a28_diff, a28_ok),
                lines)
            # a30: capture self-consistency
            a30_diff = float(np.max(
                np.abs(np.array(js0_all))))
            a30_ok = bool(a30_diff == 0.0)
            log('a30 capture self-consistency max js0='
                '%.2e ok=%s' % (a30_diff, a30_ok),
                lines)
            # a32: alpha=0 chain vs 3024 restore
            a0_arr = np.array(
                [res_l[t]['a0'] for t in tags])
            a32_diff = float(np.max(np.abs(
                a0_arr - js24_coal)))
            a32_ok = bool(a32_diff == 0.0)
            log('a32 dose alpha0 vs 3024 js_restore_'
                'coal max|d|=%.2e ok=%s'
                % (a32_diff, a32_ok), lines)
            # a38: lens-terminal consistency
            a38_rel = float(np.max(rel38_list)) \
                if rel38_list else None
            a38_ok = bool(a38_rel is not None
                          and a38_rel < 1e-4)
            log('a38 lens-terminal max rel=%.2e '
                'ok(gate 1e-4)=%s'
                % (a38_rel, a38_ok), lines)
            # a39: alpha=1 trajectory vs 3020
            traj1 = np.stack(traj1_list)
            if tags20_match and traj1.shape \
                    == traj20.shape:
                traj20_al = np.stack([
                    traj20[keys20.index(t)]
                    for t in tags])
                a39_diff = float(np.max(
                    np.abs(traj1 - traj20_al)))
            else:
                a39_diff = float('nan')
            a39_ok = bool(a39_diff == 0.0)
            log('a39 lens traj alpha=1 vs 3020 npz '
                '(key-aligned rows) max|d|=%.2e ok=%s'
                % (a39_diff, a39_ok), lines)

            traj0 = np.stack(traj0_list)
            traj2 = np.stack(traj2_list)

            # ---------- T2a excess analysis -----
            excess = traj2 - 2.0 * traj1 + traj0
            E = np.zeros_like(excess)
            if theta_l is not None:
                incl = traj2 > (2.0
                                * theta_l[None, :])
                E = np.where(incl,
                             np.maximum(excess, 0.0),
                             0.0)
            else:
                incl = np.zeros_like(excess,
                                     dtype=bool)
            E_tot = E.sum(axis=1)
            S_arr = np.array([
                E[i, -1] / E_tot[i]
                if E_tot[i] > 0 else np.nan
                for i in range(nL)])
            lstar_arr = np.array([
                int(np.argmax(E[i]))
                if E_tot[i] > 0 else -1
                for i in range(nL)])
            S_list = list(S_arr)
            E_tot_list = list(E_tot)
            lstar_list = [int(v) for v
                          in lstar_arr]
            for b, (lo, hi) in LBANDS.items():
                for i in range(nL):
                    eb = float(E[i, lo:hi].sum())
                    band_share[b].append(
                        eb / E_tot[i]
                        if E_tot[i] > 0
                        else np.nan)
                    r_bv = traj2[i, lo:hi] \
                        / np.maximum(traj1[i, lo:hi],
                                     1e-12)
                    sel_b = incl[i, lo:hi] \
                        & (traj1[i, lo:hi] > 1e-9)
                    band_ratio[b].append(
                        float(np.median(r_bv[sel_b]))
                        if sel_b.any() else np.nan)
            term_share_arr = np.array([
                E[i, -1] / E_tot[i]
                if E_tot[i] > 0 else np.nan
                for i in range(nL)])
            med_S = float(np.nanmedian(S_arr)) \
                if nL and np.any(np.isfinite(S_arr)) \
                else None
            n_counted = int(np.sum(
                np.isfinite(S_arr)))
            mag2_med = float(np.median(mag2_all))
            T2a = {
                'n_logic': nL, 'n_sham': nS,
                'lens_start': LENS_START,
                'lens_end': LENS_END,
                'med_js_erase': round(
                    float(np.median(js_er)), 6)
                if nL else None,
                'med_js_alpha0': round(
                    float(np.median(a0_arr)), 6)
                if nL else None,
                'med_js_alpha2': round(
                    float(np.median(js2_all)), 6)
                if nL else None,
                'med_excess_terminal': round(
                    float(np.median(excess[:, -1])),
                    6) if nL else None,
                'med_E_tot': float('%.3e'
                                   % np.median(E_tot))
                if nL else None,
                'med_terminal_share': round(med_S, 4)
                if med_S is not None else None,
                'n_tags_counted': n_counted,
                'terminal_share_per_tag':
                    [round(float(v), 4)
                     if np.isfinite(v) else None
                     for v in S_arr],
                'lstar_per_tag': lstar_list,
                'lstar_terminal_frac': round(
                    float(np.mean([1 if v == 31
                                   else 0
                                   for v in lstar_list])),
                    4) if nL else None,
                'mag2_med':
                    float('%.2e' % mag2_med),
                'a28_erase_chain_diff':
                    float('%.2e' % a28_diff),
                'a30_capture_self_diff':
                    float('%.2e' % a30_diff),
                'a32_dose_alpha0_diff':
                    float('%.2e' % a32_diff),
                'a38_lens_terminal_rel':
                    float('%.2e' % a38_rel)
                    if a38_rel is not None else None,
                'a39_lens_traj_3020_diff':
                    float('%.2e' % a39_diff)
                    if np.isfinite(a39_diff)
                    else None,
                'tags': tags,
                'note': 'excess_l = jsl(2) - '
                        '2*jsl(1) + jsl(0); S = '
                        'excess_xfin / positive '
                        'counted excess; theta_l = '
                        'med sham alpha=1 trajectory '
                        '(same-chain calibration)'}

            # ---------- T2b bands ----------
            T2b = {
                'bands': {b: {
                    'pos_excess_share_med': round(
                        float(np.nanmedian(
                            band_share[b])), 4)
                    if band_share[b] else None,
                    'ratio_med': round(
                        float(np.nanmedian(
                            band_ratio[b])), 4)
                    if band_ratio[b]
                    and np.any(np.isfinite(
                        band_ratio[b])) else None}
                    for b in LBANDS},
                'terminal_share_med': round(
                    float(np.nanmedian(
                        term_share_arr)), 4)
                if nL and np.any(np.isfinite(
                    term_share_arr)) else None,
                'note': 'lens-point bands: seed '
                        'idx0-3 (L4-7) / mid idx4-16 '
                        '(L8-20) / deep idx17-30 '
                        '(L21-34) / terminal idx31 '
                        '(xfin)'}

            # ---------- T2c content + sham ------
            res_c = {}
            S_c_list = []
            t1c_last = []
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                cps = ent['positions']['content']
                if cps:
                    pc = int(cps[0])
                    lt0 = [t for t in tags
                           if t.startswith(
                               'P%d:' % pi)]
                    d = arms_full(
                        pi, pc, 'C',
                        lt0[0] if lt0
                        else None)
                    res_c['P%d:%d' % (pi, pc)] = d
                    t1c_last.append(d['t1'][-1])
                    Ec = np.where(
                        d['t2'] > (2.0 * theta_l
                                   if theta_l
                                   is not None
                                   else -np.inf),
                        np.maximum(
                            d['t2'] - 2.0 * d['t1']
                            + d['t0'], 0.0), 0.0)
                    Et = float(Ec.sum())
                    S_c_list.append(
                        float(Ec[-1]) / Et
                        if Et > 0 else np.nan)
            T2c = {
                'med_js_erase_content': round(
                    float(np.median(
                        [res_c[k]['er']
                         for k in res_c])), 6)
                if res_c else None,
                'med_t1_terminal_content': round(
                    float(np.median(t1c_last)), 6)
                if t1c_last else None,
                'med_S_content': round(
                    float(np.nanmedian(S_c_list)), 4)
                if S_c_list and np.any(np.isfinite(
                    S_c_list)) else None,
                'med_js_sham': round(
                    float(np.median(js_sham_all)),
                    6) if js_sham_all else None,
                'med_theta_l_terminal': float(
                    '%.2e' % theta_l[-1])
                if theta_l is not None else None,
                'note': 'descriptive content-side '
                        '(tref convention) + sham '
                        'calibration (per-layer '
                        'floor)'}
            log('T2c content er=%s t1c=%s S_c=%s '
                'sham=%s'
                % (T2c['med_js_erase_content'],
                   T2c['med_t1_terminal_content'],
                   T2c['med_S_content'],
                   T2c['med_js_sham']), lines)

            # ---------- verdict ----------
            gates_ok = bool(
                nL >= MIN_POS
                and np.all(np.array(js_er) > 0)
                and mag2_med > REST_MIN
                and theta_l is not None
                and n_counted >= MIN_POS
                and a38_ok)
            if not gates_ok:
                verdict = 'readout_convex_' \
                          'undetermined_void'
            elif med_S >= TERM_SHARE:
                verdict = 'readout_terminal_' \
                          'convex_qwen'
            elif med_S <= DIST_SHARE:
                verdict = 'readout_distributed_' \
                          'convex_qwen'
            else:
                verdict = 'readout_convex_' \
                          'mixed_qwen'

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
        'a27_3022': a27_ok,
        'a28_erase_chain_diff': a28_diff,
        'a29_3023': a29_ok,
        'a30_capture_self_diff': a30_diff,
        'a31_3024': a31_ok,
        'a32_dose_alpha0_diff': a32_diff,
        'a33_3028': a33_ok, 'a34_3029': a34_ok,
        'a38_lens_terminal_rel': a38_rel,
        'a39_lens_traj_3020_diff': a39_diff,
    }
    a28_ok_f = bool(a28_diff is not None
                    and a28_diff == 0.0)
    a30_ok_f = bool(a30_diff is not None
                    and a30_diff == 0.0)
    a32_ok_f = bool(a32_diff is not None
                    and a32_diff == 0.0)
    a39_ok_f = bool(a39_diff is not None
                    and a39_diff == 0.0)
    res = {
        'phase': 3030,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok and a28_ok_f
                              and a30_ok_f
                              and a32_ok_f
                              and a38_ok and a39_ok_f),
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
            'tags22': np.array(tags22, dtype=object),
            'tags20': np.array(tags20, dtype=object),
            'js_erase': np.array(js_er),
            'js_alpha0': a0_arr if a0_arr.size
            else np.array([]),
            'js_alpha2': np.array(js2_all),
            'js0_self': np.array(js0_all),
            'mag2': np.array(mag2_all),
            'traj_alpha0': np.stack(traj0_list)
            if traj0_list else np.array([]),
            'traj_alpha1': np.stack(traj1_list)
            if traj1_list else np.array([]),
            'traj_alpha2': np.stack(traj2_list)
            if traj2_list else np.array([]),
            'sham_traj': sham_traj,
            'theta_l': theta_l
            if theta_l is not None
            else np.array([]),
            'E_tot': np.array(E_tot_list, dtype=float),
            'terminal_share': np.array(
                [np.nan if v is None else v
                 for v in S_list], dtype=float),
            'lstar': np.array(lstar_list, dtype=int),
            'band_share_seed':
                np.array(band_share['seed'],
                         dtype=float),
            'band_share_mid':
                np.array(band_share['mid'],
                         dtype=float),
            'band_share_deep':
                np.array(band_share['deep'],
                         dtype=float),
            'band_ratio_seed':
                np.array(band_ratio['seed'],
                         dtype=float),
            'band_ratio_mid':
                np.array(band_ratio['mid'],
                         dtype=float),
            'band_ratio_deep':
                np.array(band_ratio['deep'],
                         dtype=float),
            'rel38': np.array(rel38_list,
                              dtype=float),
            'js_sham': np.array(js_sham_all),
            'js_erase_content': np.array(
                [res_c[k]['er'] for k in res_c])
            if res_c else np.array([]),
            't1_terminal_content': np.array(
                t1c_last, dtype=float)
            if t1c_last else np.array([])}
    npz_path = os.path.join(
        OUT, 'omega_p2x_readout_convexity_qwen.npz')
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
