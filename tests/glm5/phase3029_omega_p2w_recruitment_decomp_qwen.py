# -*- coding: utf-8 -*-
"""Phase 3029: Omega-P2w recruitment band decomposition
(qwen).

Why: 3028 showed the coalition-channel dose response is
SUPERLINEAR (js at alpha=2 is 2.18x the linear
prediction pred2 = 2*js1 - js0; med R^2 0.794 over the
5-point fit => mid-range quasi-linear, tail convex).
3016 established the downstream amplification is
distributed with deep-layer convergence.  Open
question: WHERE does the alpha>1 excess come from -
amplified responses of units already carrying the
alpha=1 signal, or NEWLY RECRUITED downstream units
(silent at alpha=1, above noise floor at alpha=2)?
And in which layer bands?

Design (3028 machine verbatim for anchors/geometry/
generation/position selection/two-step protocol +
3016 o_proj-input capture; the added pre_oproj hooks
are observation-only and bit-identity is re-verified
by a28/a30/a32):
  Per logic tag, three dose arms alpha in {0, 1, 2}
  (alpha=1 = plain ERASE, no hook; alpha=0 bit-equals
  3024 RESTORE_COAL; alpha=2 = 3028 amplifier), each
  capturing the step-2 o_proj INPUT of every layer
  (4096 = 32 query heads x 128).  dnorm_a(l,qh) =
  ||o_a - o_base|| / (||o_base|| + 1e-12), l in 4..35.
  Noise floor theta = 95th percentile of pooled
  sham-position dnorm (erase chains at sham/functoid
  positions; 3016 measured their JS ~2e-4).
  U(a) = {(l,qh): dnorm_a > theta}.
  PRIMARY per tag: recruitment energy share
  rho = sum(dnorm2^2 over U(2) minus U(1)) /
        sum(dnorm2^2 over U(2));
  growth exponent over U(1)&U(2):
  g = log2(dnorm2 / dnorm1) (g=1 linear propagation,
  g>1 superlinear).  med over tags => verdict.
  T2b band decomposition: seed L4-7 / mid L8-20 /
  deep L21-35 - recruited-energy share per band and
  g med per band.
  T2c: content descriptive (tref convention) + sham
  calibration (med js ~2e-4 expected).

Verdict (frozen):
  anchor fail                     => anchor_fail_
                                    all_void
  gates fail (nL<8 or any js_erase<=0 or theta
  degenerate or U(2) empty somewhere or mag2
  degenerate)                     => recruitment_
                                    undetermined_void
  med rho >= 0.5                  => recruit_new_units_
                                    dominant_qwen
  med rho <= 0.2                  => amplify_existing_
                                    dominant_qwen
  else                            => recruit_amplify_
                                    mixed_qwen

Anchors (frozen): a0-a32 as 3028 verbatim (a28 erase
chain vs 3022 npz bit-level 0.0; a30 capture self-
consistency 0.0; a32 alpha=0 vs 3024 npz
js_restore_coal bit-level 0.0); plus a33 3028
integrity (seal match AND verdict
dose_superlinear_qwen AND anchors ok).

Tags: Omega-P2w / recruitment band decomposition /
o_proj-input delta energy on the dose gradient /
sham-based noise floor / no hallucination naming.
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
OUT = os.path.join(BASE, 'phase3029',
                   'omega_p2w_recruitment_decomp_qwen')
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
TH_PCTL = 95.0
RHO_NEW = 0.5
RHO_AMP = 0.2
REST_MIN = 1e-6
BANDS = {'seed': (4, 8), 'mid': (8, 21),
         'deep': (21, 36)}
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
    'mode': 'qwen3-4b; 3028 machine verbatim for anchors/'
            'geometry/generation/position selection/'
            'two-step protocol + 3016 o_proj-input '
            'capture (observation-only pre_oproj hooks; '
            'bit-identity re-verified by a28/a30/a32); '
            'recruitment decomposition on the coalition '
            'dose gradient alpha in {0, 1, 2}',
    'question': 'At alpha=2 (js 2.18x the linear '
                'prediction), is the excess carried by '
                'AMPLIFIED existing carriers (units '
                'above noise floor at alpha=1, growth '
                'exponent g = log2(d2/d1)) or by '
                'NEWLY RECRUITED units (silent at '
                'alpha=1, above floor at alpha=2)? '
                'Which layer bands host the recruited '
                'energy?',
    'T1': 'capture: one clean prefill per prompt; logic/'
          'content/sham positions of the 3009 chain; '
          'coalition sets reconstructed from the sealed '
          '3022 npz s_relay (per-tag top-32 positive); '
          'tags asserted equal; per-tag no-erase capture '
          '(h_base for the dose hook + o_base for the '
          'dnom denominator); a30 self-consistency '
          'js(pb, p0) == 0.0 bit-level',
    'T2a': 'PRIMARY: arms per logic tag = ERASE (alpha'
           '=1) / DOSE alpha in {0, 2}; o_proj-input '
           'delta energy dnorm_a(l,qh) per arm; noise '
           'floor theta = 95th pct of pooled '
           'sham-position dnorm (l in 4..35); U(a) = '
           '{dnorm_a > theta}; recruitment energy '
           'share rho = sum d2^2 over U(2)\\U(1) / sum '
           'd2^2 over U(2); med over tags; growth '
           'exponent g = log2(d2/d1) over U(1)&U(2); '
           'a28 erase identity vs 3022 npz bit-level '
           '0.0; a32 alpha0 chain vs 3024 npz '
           'js_restore_coal bit-level 0.0; a30 capture '
           'self-consistency 0.0',
    'T2b': 'DESCRIPTIVE band decomposition: seed L4-7 / '
           'mid L8-20 / deep L21-35 - recruited-energy '
           'share per band; g med per band; amplified '
           '(U1&U2) energy share per band',
    'T2c': 'DESCRIPTIVE: content dose arms (tref '
           'convention) + rho_c; sham calibration med '
           'js (expect ~2e-4) and theta',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => recruitment_undetermined_'
               'void; med rho >= 0.5 => '
               'recruit_new_units_dominant_qwen; med '
               'rho <= 0.2 => '
               'amplify_existing_dominant_qwen; else '
               '=> recruit_amplify_mixed_qwen',
    'tags': 'Omega-P2w / recruitment band '
            'decomposition / o_proj-input delta energy '
            'on the dose gradient / sham-based noise '
            'floor / no hallucination naming',
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
        json.dump({'phase': 3029,
                   'name': 'omega_p2w_recruitment_'
                           'decomp_qwen',
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
                           D_3028 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'l_relay': L_RELAY,
                   'g7_head': G7_HEAD,
                   'topk': TOPK,
                   'hdim': HDIM, 'nhq': NHQ,
                   'l_min': L_MIN,
                   'th_pctl': TH_PCTL,
                   'rho_new': RHO_NEW,
                   'rho_amp': RHO_AMP,
                   'rest_min': REST_MIN,
                   'bands': {k: list(v) for k, v
                             in BANDS.items()},
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

    def capture_base(pi, p_pos):
        """No-erase two-step chain with h_base + o_base
        capture; returns step-2 probs p0."""
        pr = GEN_PROMPTS[pi]
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
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
            p0, _ = prefill_step2(pr, ids, past)
        state_hcap['on'] = False
        state_o['on'] = False
        return p0, grab_o()

    def run_chain(pr, p_pos, erase, rest=None):
        """Two-step chain, optionally g7-K erasure at
        p_pos and/or coalition DOSE patch at L3
        down_proj (rest = (idx, alpha)).  Captures
        o_proj input; caller must grab_o() right after
        (the next chain clears o_in)."""
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
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
            out2 = model(
                input_ids=torch.tensor(
                    [[int(ids[-1])]], device='cuda'),
                past_key_values=past,
                use_cache=False)
        state_o['on'] = False
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        return p_

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
                         and a31_ok and a33_ok)
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
    tags = []
    nL = nC = nS = 0
    js_er = []
    js0_all = []
    js2_all = []
    mag2_all = []
    d1_list = []
    d2_list = []
    d0_list = []
    rho_all = []
    g_all = []
    band_share = {b: [] for b in BANDS}
    band_g = {b: [] for b in BANDS}
    band_amp = {b: [] for b in BANDS}
    theta = None
    js_sham_all = []
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

            tmap = {t: k for k, t
                    in enumerate(tags22)}

            def arms_full(pi, p_pos, ck, tref=None):
                """Per tag: capture + 3 arms (alpha 1/0/2)
                with o_proj-input capture; returns dict."""
                t = 'P%d:%d' % (pi, p_pos)
                tr = tref if tref is not None \
                    else t
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                cidx = coal_sets[tmap[tr]]
                p0, o_b = capture_base(pi, p_pos)
                js0 = js_nats(pb, p0)
                q_e = run_chain(pr, p_pos,
                                True, None)
                o_e = grab_o()
                out_er = js_nats(pb, q_e)
                q_0 = run_chain(pr, p_pos, True,
                                (cidx, 0.0))
                o_0 = grab_o()
                js0d = js_nats(pb, q_0)
                q_2 = run_chain(pr, p_pos, True,
                                (cidx, 2.0))
                o_2 = grab_o()
                js2d = js_nats(pb, q_2)
                mag2 = state_rest['mag'] \
                    / max(state_rest['onorm'],
                          1e-30)
                d1m = np.zeros((NL, NHQ))
                d2m = np.zeros((NL, NHQ))
                d0m = np.zeros((NL, NHQ))
                for li in range(L_MIN, NL):
                    db = o_b[li].reshape(NHQ, HDIM)
                    num1 = np.linalg.norm(
                        o_e[li].reshape(NHQ, HDIM)
                        - db, axis=1)
                    num2 = np.linalg.norm(
                        o_2[li].reshape(NHQ, HDIM)
                        - db, axis=1)
                    num0 = np.linalg.norm(
                        o_0[li].reshape(NHQ, HDIM)
                        - db, axis=1)
                    den = np.linalg.norm(db,
                                         axis=1) + 1e-12
                    d1m[li] = num1 / den
                    d2m[li] = num2 / den
                    d0m[li] = num0 / den
                log('%s %s js0=%.2e er=%.5f '
                    'a0=%.5f a2=%.5f mag2=%.2e'
                    % (ck, t, js0, out_er, js0d,
                       js2d, mag2), lines)
                return {'js0': js0, 'er': out_er,
                        'a0': js0d, 'a2': js2d,
                        'mag2': mag2,
                        'd1': d1m, 'd2': d2m,
                        'd0': d0m}

            res_l = {}
            for t in tags:
                pi = int(t.split(':')[0][1:])
                p_pos = int(t.split(':')[1])
                res_l[t] = arms_full(pi, p_pos, 'L')
                js_er.append(res_l[t]['er'])
                js0_all.append(res_l[t]['js0'])
                js2_all.append(res_l[t]['a2'])
                mag2_all.append(res_l[t]['mag2'])
                d1_list.append(res_l[t]['d1'])
                d2_list.append(res_l[t]['d2'])
                d0_list.append(res_l[t]['d0'])
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

            # ---------- sham noise floor ----------
            sham_dnorm = []
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
                p0s, o_bs = capture_base(pi, p_sh)
                js_sham_all.append(
                    js_nats(pb, p0s))
                q_s = run_chain(pr, p_sh, True,
                                None)
                o_s = grab_o()
                nS += 1
                for li in range(L_MIN, NL):
                    db = o_bs[li].reshape(NHQ,
                                          HDIM)
                    num = np.linalg.norm(
                        o_s[li].reshape(NHQ, HDIM)
                        - db, axis=1)
                    den = np.linalg.norm(db,
                                         axis=1) \
                        + 1e-12
                    sham_dnorm.extend(
                        (num / den).tolist())
            pool = np.array(sham_dnorm)
            theta = float(np.percentile(
                pool, TH_PCTL)) \
                if pool.size else None
            log('sham n=%d med_js=%.6f theta(%.0f%%)='
                '%.4f'
                % (nS,
                   float(np.median(js_sham_all))
                   if js_sham_all else -1,
                   TH_PCTL,
                   theta if theta is not None
                   else -1), lines)

            # ---------- T2a recruitment ----------
            for t in tags:
                d1v = res_l[t]['d1'][L_MIN:].ravel()
                d2v = res_l[t]['d2'][L_MIN:].ravel()
                U2 = d2v > theta
                U1 = d1v > theta
                rec = U2 & (~U1)
                e2 = float((d2v[U2] ** 2).sum())
                rho_tag = float(
                    (d2v[rec] ** 2).sum()) / e2 \
                    if e2 > 0 else np.nan
                rho_all.append(rho_tag)
                both = U2 & U1
                g_tag = float(np.median(
                    np.log2(d2v[both]
                            / np.maximum(
                                d1v[both], 1e-30)))) \
                    if both.any() else np.nan
                g_all.append(g_tag)
                for b, (lo, hi) in BANDS.items():
                    sl = slice(
                        (lo - L_MIN) * NHQ,
                        (hi - L_MIN) * NHQ)
                    eb_r = float(
                        (d2v[rec][sl] ** 2).sum())
                    eb_a = float(
                        (d2v[both][sl] ** 2)
                        .sum())
                    band_share[b].append(
                        eb_r / e2 if e2 > 0
                        else np.nan)
                    band_amp[b].append(
                        eb_a / e2 if e2 > 0
                        else np.nan)
                    gb = np.log2(
                        d2v[both][sl]
                        / np.maximum(
                            d1v[both][sl],
                            1e-30))
                    band_g[b].append(
                        float(np.median(gb))
                        if gb.size else np.nan)
                log('%s rho=%.3f g=%.3f '
                    'bandR=%s'
                    % (t, rho_tag, g_tag,
                       {b: round(float(
                           band_share[b][-1]), 3)
                        for b in BANDS}), lines)

            rho_a = np.array(rho_all, dtype=float)
            g_a = np.array(g_all, dtype=float)
            med_rho = float(np.nanmedian(rho_a)) \
                if rho_a.size else None
            med_g = float(np.nanmedian(g_a)) \
                if g_a.size else None
            n_rec = int(np.sum(rho_a > RHO_NEW)) \
                if rho_a.size else 0
            mag2_med = float(np.median(mag2_all))
            T2a = {
                'n_logic': nL, 'n_sham': nS,
                'l_relay': L_RELAY, 'inter': inter,
                'topk': TOPK,
                'th_pctl': TH_PCTL,
                'theta': round(theta, 6)
                if theta is not None else None,
                'med_js_erase': round(
                    float(np.median(js_er)), 6)
                if nL else None,
                'med_js_alpha0': round(
                    float(np.median(a0_arr)), 6)
                if nL else None,
                'med_js_alpha2': round(
                    float(np.median(js2_all)), 6)
                if nL else None,
                'med_rho_recruit_share':
                    round(med_rho, 4)
                if med_rho is not None else None,
                'med_growth_exponent':
                    round(med_g, 4)
                if med_g is not None else None,
                'n_tags_rho_new_dominant': n_rec,
                'rho_per_tag':
                    [round(float(v), 4)
                     for v in rho_a],
                'g_per_tag':
                    [round(float(v), 4)
                     for v in g_a],
                'mag2_med':
                    float('%.2e' % mag2_med),
                'a28_erase_chain_diff':
                    float('%.2e' % a28_diff),
                'a30_capture_self_diff':
                    float('%.2e' % a30_diff),
                'a32_dose_alpha0_diff':
                    float('%.2e' % a32_diff),
                'tags': tags,
                'note': 'rho = recruited (U2\\U1) '
                        'energy share of U2 at '
                        'alpha=2; g = log2(d2/d1) '
                        'median over U1&U2; theta = '
                        'sham-pooled 95th pct'}
            log('T2a med_rho=%s med_g=%s n_rec=%d '
                'mag2=%.2e'
                % (med_rho, med_g, n_rec,
                   mag2_med), lines)

            # ---------- T2b bands ----------
            T2b = {
                'bands': {b: {
                    'recruit_share_med': round(
                        float(np.nanmedian(
                            band_share[b])), 4)
                    if band_share[b] else None,
                    'amp_share_med': round(
                        float(np.nanmedian(
                            band_amp[b])), 4)
                    if band_amp[b] else None,
                    'g_med': round(
                        float(np.nanmedian(
                            band_g[b])), 4)
                    if band_g[b] else None}
                    for b in BANDS},
                'note': 'seed L4-7 / mid L8-20 / '
                        'deep L21-35; recruit share '
                        'of the alpha=2 delta energy'}
            log('T2b bands %s'
                % json.dumps(T2b['bands']), lines)

            # ---------- T2c content + sham ----------
            res_c = {}
            rho_c_list = []
            js_er_c = []
            js_d0_c = []
            js_d2_c = []
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
                    js_er_c.append(d['er'])
                    js_d0_c.append(d['a0'])
                    js_d2_c.append(d['a2'])
                    d1v = d['d1'][L_MIN:].ravel()
                    d2v = d['d2'][L_MIN:].ravel()
                    U2 = d2v > theta
                    U1 = d1v > theta
                    recm = U2 & (~U1)
                    e2 = float(
                        (d2v[U2] ** 2).sum())
                    rho_c_list.append(
                        float((d2v[recm] ** 2)
                              .sum()) / e2
                        if e2 > 0 else np.nan)
            col_c = None
            if js_er_c:
                ec = np.array(js_er_c)
                c0 = np.array(js_d0_c)
                c2 = np.array(js_d2_c)
                col0 = float(np.median(
                    1.0 - c0 / ec))
                col2 = float(np.median(
                    (c2 - ec) / ec))
                col_c = (col0, col2)
            T2c = {
                'med_js_erase_content': round(
                    float(np.median(js_er_c)), 6)
                if js_er_c else None,
                'med_js_alpha0_content': round(
                    float(np.median(js_d0_c)), 6)
                if js_er_c else None,
                'med_js_alpha2_content': round(
                    float(np.median(js_d2_c)), 6)
                if js_er_c else None,
                'collapse0_content': round(col_c[0],
                                           4)
                if col_c is not None else None,
                'amp2_content': round(col_c[1], 4)
                if col_c is not None else None,
                'med_rho_content': round(
                    float(np.nanmedian(
                        rho_c_list)), 4)
                if rho_c_list else None,
                'med_js_sham': round(
                    float(np.median(js_sham_all)),
                    6) if js_sham_all else None,
                'note': 'descriptive content-side '
                        '(tref convention) + sham '
                        'calibration'}
            log('T2c content er=%s d0=%s d2=%s '
                'rho_c=%s sham=%s'
                % (T2c['med_js_erase_content'],
                   T2c['med_js_alpha0_content'],
                   T2c['med_js_alpha2_content'],
                   T2c['med_rho_content'],
                   T2c['med_js_sham']), lines)

            # ---------- verdict ----------
            gates_ok = bool(
                nL >= MIN_POS
                and np.all(np.array(js_er) > 0)
                and theta is not None
                and theta > 0
                and mag2_med > REST_MIN
                and rho_a.size == nL
                and np.all(np.isfinite(rho_a)))
            if not gates_ok:
                verdict = 'recruitment_' \
                          'undetermined_void'
            elif med_rho >= RHO_NEW:
                verdict = 'recruit_new_units_' \
                          'dominant_qwen'
            elif med_rho <= RHO_AMP:
                verdict = 'amplify_existing_' \
                          'dominant_qwen'
            else:
                verdict = 'recruit_amplify_' \
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
        'a33_3028': a33_ok,
    }
    a28_ok_f = bool(a28_diff is not None
                    and a28_diff == 0.0)
    a30_ok_f = bool(a30_diff is not None
                    and a30_diff == 0.0)
    a32_ok_f = bool(a32_diff is not None
                    and a32_diff == 0.0)
    res = {
        'phase': 3029,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok and a28_ok_f
                              and a30_ok_f
                              and a32_ok_f),
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
            'js_erase': np.array(js_er),
            'js_alpha0': a0_arr if a0_arr.size
            else np.array([]),
            'js_alpha2': np.array(js2_all),
            'js0_self': np.array(js0_all),
            'mag2': np.array(mag2_all),
            'd1': np.stack(d1_list)
            if d1_list else np.array([]),
            'd2': np.stack(d2_list)
            if d2_list else np.array([]),
            'd0': np.stack(d0_list)
            if d0_list else np.array([]),
            'rho': np.array(rho_all, dtype=float),
            'g_exp': np.array(g_all, dtype=float),
            'band_share_seed':
                np.array(band_share['seed'],
                         dtype=float),
            'band_share_mid':
                np.array(band_share['mid'],
                         dtype=float),
            'band_share_deep':
                np.array(band_share['deep'],
                         dtype=float),
            'band_amp_seed':
                np.array(band_amp['seed'],
                         dtype=float),
            'band_amp_mid':
                np.array(band_amp['mid'],
                         dtype=float),
            'band_amp_deep':
                np.array(band_amp['deep'],
                         dtype=float),
            'band_g_seed':
                np.array(band_g['seed'],
                         dtype=float),
            'band_g_mid':
                np.array(band_g['mid'],
                         dtype=float),
            'band_g_deep':
                np.array(band_g['deep'],
                         dtype=float),
            'theta': np.float64(theta)
            if theta is not None else np.float64(0),
            'js_sham': np.array(js_sham_all),
            'js_erase_content': np.array(js_er_c),
            'js_alpha0_content': np.array(js_d0_c),
            'js_alpha2_content': np.array(js_d2_c),
            'rho_content': np.array(rho_c_list,
                                    dtype=float)}
    npz_path = os.path.join(
        OUT, 'omega_p2w_recruitment_decomp_qwen.npz')
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
