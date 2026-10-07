# -*- coding: utf-8 -*-
"""Phase 3027: Omega-P2u coalition baseline function
and downstream consumer-head localization (qwen).

Why: 3023 showed the L3 coalition (top-32 relay
neurons, 0.33% of the layer) carries 55.5% of the
L3 MLP output norm and its baseline output is
consumed at ALL positions (zero-ablation toxicity
at content too).  3024 confirmed causal load-
bearing via baseline-restoration.  This phase
localizes WHO consumes the coalition baseline
output: per-query-head (32x128, o_proj input side)
delta accounts at L4 = L3+1, the first layer
downstream of the relay.

Design (3023 machine verbatim for run_chain/
hook_abl/erase; 3026 machine for anchors/capture/
generation/position selection, SEED_RND=3009):
  Per logic tag: capture_base chain (p0 self-
  consistency a30; L4 base capture), ERASE chain
  (a28 vs 3022 bit-level 0.0; L4 erase capture;
  e4 = res36[4] erase - base), ABL_COAL chain
  (zero-ablation of the coalition 32 neurons at
  L3 down_proj, NO erase; a35 vs sealed 3023 npz
  js_abl_only bit-level 0.0; L4 abl capture),
  4 random-32 null chains (non-coalition pool,
  SEED_RND+28; L4 null capture).
  Per content position (first content pos per
  prompt): 'ab' chain (zero-ablation, clean
  baseline-consumption readout) + 'co' chain
  (erase+ablation; a36 vs sealed 3023 npz
  js_coal_content bit-level 0.0).
  Head delta d (32,128) = abl - base at p_pos;
  p_h = per-head norm share; conc8 = top-8 share.
  T2a PRIMARY: conc8 obs vs random-null median
  (4 draws per tag), diff = obs - null_med,
  exact binomial sign test.  Reachability
  pre-checked: random-subset conc8 concentrates
  near-uniform ~0.25-0.4, coalition expected far
  above if consumption is head-specialized; no
  permutation-invariant statistic.
  T2b DESCRIPTIVE: GQA group shares (8) vs 3015
  K-consumer med_share_top8 Spearman (category
  mismatch noted: 3015 = erase response, here =
  baseline consumption); top-8 head-set cross-tag
  Jaccard; Spearman(p_h_abl, p_h_erase).
  T2c DESCRIPTIVE: per-head residual contribution
  c_q = einsum(W_o^L4.view(2560,32,128), d) ->
  (32,2560), cos(c_q, e4_unit) - which heads
  transmit erase-aligned signal.

Verdict (frozen):
  anchor fail                     => anchor_fail_
                                    all_void
  gates fail (nL<8 or any js_abl<=0 or abl
  magnitude degenerate)           => consumption_
                                    undetermined_void
  diff med > 0 AND p_binom(n_pos) <= 0.05
                                  => consumption_
                                    concentrated_qwen
  diff med < 0 AND p_binom(n_neg) <= 0.05
                                  => consumption_
                                    spread_qwen
  else                            => consumption_
                                    mixed_qwen

Anchors (frozen): a0-a27 as 3026 verbatim, a28
erase chain vs 3022 js bit-level 0.0, a29 3023
integrity, a30 capture self-consistency js(pb,p0)
0.0 bit-level, a31 3024 integrity, a33 3025
integrity, a35 ABL_COAL vs sealed 3023 npz
js_abl_only bit-level 0.0, a36 content co-chain
vs sealed 3023 npz js_coal_content bit-level 0.0.

Tags: Omega-P2u / coalition baseline function /
downstream consumer-head localization / conc8 vs
random-32 null / GQA group Spearman vs 3015 / no
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
                      'omega_p2l_dilution_'
                      'decomposition_qwen')
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
D_3025 = os.path.join(BASE, 'phase3025',
                      'omega_p2s_rebalance_decomp_qwen')
F_3022NPZ = os.path.join(
    D_3022, 'omega_p2p_l3_relay_neurons_qwen.npz')
F_3023NPZ = os.path.join(
    D_3023, 'omega_p2q_relay_causal_ablation_qwen'
            '.npz')
OUT = os.path.join(BASE, 'phase3027',
                   'omega_p2u_consumer_heads_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
SEED_NULL = 2896
SEED_RND = 3009
K_GEN = 256
MIN_POS = 8
L3_GATED = 3
L4_CONS = 4
G7_HEAD = 7
TOPK = 32
N_NULL = 4
NH_Q, HD = 32, 128
P_GATE = 0.05
ABL_MIN = 1e-6
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
               'because', 'therefore', 'however',
               'while', 'thus', 'although')
FUNC_WORDS = ('the', 'of', 'to', 'a', 'in', 'is',
              'that', 'it', 'for', 'on', 'with', 'as',
              'at', 'by', 'from', 'this', 'be', 'are',
              'was', 'were', 'has', 'had', 'have',
              'will', 'would', 'can', 'could', 'not',
              'no', 'yes', 'he', 'she', 'they', 'we',
              'you', 'i', 'his', 'her', 'their',
              'our', 'my', 'when', 'where', 'who')

PREREG = {
    'mode': 'qwen3-4b; 3023 machine verbatim for '
            'run_chain/hook_abl/erase + 3026 machine '
            'for anchors/capture/generation/position '
            'selection (SEED_RND=3009 explicit '
            'rebuild); localizes consumers of the L3 '
            'coalition baseline output: per logic '
            'tag capture_base + ERASE (a28) + '
            'ABL_COAL zero-ablation (a35 vs 3023 '
            'js_abl_only bit-level 0.0) + 4 random-'
            '32 null chains (non-coalition pool, '
            'SEED_RND+28); L4 o_proj-input per-head '
            'delta (32x128) p_h shares; T2a PRIMARY '
            'conc8 vs null median sign test; per '
            'content position ab chain (clean '
            'baseline consumption) + co chain (a36 '
            'vs 3023 js_coal_content bit-level '
            '0.0)',
    'question': 'Is the baseline output of the L3 '
                'coalition consumed by a specialized '
                'set of L4 query heads (conc8 of the '
                'per-head delta exceeds random-32 '
                'nulls), and do those heads overlap '
                'the erase-response heads (3015/'
                '3016)?',
    'T1': 'capture: one clean prefill per prompt; '
          'logic positions of the 3009 chain; '
          'coalition from sealed 3022 npz s_relay '
          'top-32 positive; tags asserted equal to '
          '3022; per tag: capture_base (a30 js0 '
          'self-consistency 0.0 bit-level, L4 base '
          'capture), ERASE (a28, e4, L4 erase '
          'capture), ABL_COAL (a35, L4 abl '
          'capture), 4 null chains',
    'T2a': 'PRIMARY: p_h = per-head norm share of '
           'd = abl - base at p_pos (32,128); '
           'conc8 = sum of top-8 shares; null = '
           '4 random-32 non-coalition zero-'
           'ablation chains per tag; diff = '
           'conc8 - null_med; exact binomial '
           'one-sided sign test across tags',
    'T2b': 'DESCRIPTIVE: GQA group shares (8) vs '
           '3015 T2a med_share_top8 Spearman '
           '(category mismatch noted: 3015 erase '
           'response vs here baseline '
           'consumption); top-8 head-set cross-tag '
           'Jaccard median; Spearman(p_h_abl, '
           'p_h_erase) per tag',
    'T2c': 'DESCRIPTIVE: per-head residual '
           'contribution c_q = einsum(W_o^L4.view('
           '2560,32,128), d) -> (32,2560); cos(c_q, '
           'e4_unit) per head; top-3 heads and '
           'median cos; content ab-chain conc8 '
           'contrast',
    'verdict': 'anchor fail => anchor_fail_all_'
               'void; gates fail => consumption_'
               'undetermined_void; diff med > 0 '
               'and p_binom(n_pos) <= 0.05 => '
               'consumption_concentrated_qwen; '
               'diff med < 0 and p_binom(n_neg) '
               '<= 0.05 => consumption_spread_'
               'qwen; else => consumption_mixed_'
               'qwen',
    'tags': 'Omega-P2u / coalition baseline '
            'function / downstream consumer-head '
            'localization / conc8 vs random-32 '
            'null / GQA group Spearman vs 3015 / '
            'no hallucination naming',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20),
                          b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def js_nats(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask]
                            * np.log(a[mask]
                                     / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def binom_ge(k_obs, n):
    from math import comb
    return float(sum(comb(n, k) for k in
                     range(k_obs, n + 1))) / 2 ** n


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def log(msg, lines):
    lines.append('[%s] %s'
                 % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'),
              'w', encoding='utf-8') as f:
        json.dump({'phase': 3027,
                   'name': 'omega_p2u_consumer_'
                           'heads_qwen',
                   'created':
                       time.strftime(
                           '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(
                           __file__)),
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
                       's3023npz': sha8(F_3023NPZ),
                       's3024result': sha8(
                           D_3024 + r'\result.json'),
                       's3024seal': sha8(
                           D_3024 + r'\seal.json'),
                       's3025result': sha8(
                           D_3025 + r'\result.json'),
                       's3025seal': sha8(
                           D_3025 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'min_pos': MIN_POS,
                   'l3_gated': L3_GATED,
                   'l4_cons': L4_CONS,
                   'g7_head': G7_HEAD,
                   'topk': TOPK,
                   'n_null': N_NULL,
                   'nh_q': NH_Q, 'hd': HD,
                   'p_gate': P_GATE,
                   'abl_min': ABL_MIN,
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
                  r'registration.npz',
                  allow_pickle=True)
    l_words = [str(w) for w in z93['l_words']]
    l_A = [str(w) for w in z93['l_A']]
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
                               - w1024_93).max())
                    / sc1)
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
                 == 'logic_locked_perturb_divergent_'
                    'qwen'
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
                  == 'decomp_cancellation_dominant_'
                     'qwen'
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
                  == 'injection_readout_asymmetric_'
                     'qwen'
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

    seal25 = json.load(open(D_3025 + r'\seal.json',
                            encoding='utf-8'))
    r25 = json.load(open(D_3025 + r'\result.json',
                         encoding='utf-8'))
    a33_ok = bool(seal25['result_sha256_8']
                  == sha8(D_3025 + r'\result.json')
                  and r25['final_verdict']
                  == 'rebalance_noncoal_protective_'
                     'qwen'
                  and r25['anchor_all_ok'] is True)
    log('a33 3025 integrity %s (verdict=%s)'
        % (a33_ok, r25['final_verdict']), lines)

    # ---- 3022 coalition + 3023 npz + 3015 share --
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
    z23 = np.load(F_3023NPZ, allow_pickle=True)
    tags23 = [str(t) for t in z23['tags']]
    js23_abl = z23['js_abl_only'].astype(np.float64)
    js23_coalc = z23['js_coal_content'] \
        .astype(np.float64)
    assert tags23 == tags22
    share15 = [float(v) for k, v in sorted(
        r15['T2a']['med_share_top8'].items(),
        key=lambda kv: int(kv[0]))]
    log('3022 s_relay %s dirs pos OK; 3023 npz tags '
        'match js_abl %s js_coalc %s; 3015 share15 '
        '%s'
        % (s22.shape, js23_abl.shape,
           js23_coalc.shape, share15), lines)

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
        MD, local_files_only=True,
        trust_remote_code=True, use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t,
                      add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(t,
                          add_special_tokens=False)[
                    'input_ids']
            assert len(ids) == 1, \
                '%s -> %s' % (t, ids)
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

    batch = {'func': [[func_tid,
                       tid_map[words[i][2]]]
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
    state_r = {'on': False}
    rs = {}
    state_abl = {'on': False, 'idx': None,
                 'norm': 0.0, 'mnorm': 0.0}
    Wd = layers[L3_GATED].mlp.down_proj.weight
    inter = int(Wd.shape[1])
    assert inter == s22.shape[1], \
        (inter, s22.shape)
    Wd_buf = {'idx': None, 'cols': None}

    def hook_abl(module, args, output):
        if state_abl['on'] \
                and state_abl['idx'] is not None:
            idx = state_abl['idx']
            if Wd_buf['idx'] is not idx:
                Wd_buf['idx'] = idx
                Wd_buf['cols'] = Wd[:, idx].detach()
            h_in = args[0]
            delta = h_in[:, :, idx] \
                @ Wd_buf['cols'].T
            out_n = output - delta
            state_abl['norm'] = float(
                delta[0, -1].float().norm())
            state_abl['mnorm'] = float(
                output[0, -1].float().norm())
            return out_n
        return None

    state_l4 = {'mode': None}
    l4_cap = {}

    def pre_o4(module, args, kwargs):
        if state_l4['mode'] is not None:
            l4_cap.setdefault(
                state_l4['mode'], []).append(
                args[0].detach().float().cpu()
                .numpy().copy())
        return None

    handles = []
    handles.append(layers[L3_GATED].mlp.down_proj
                   .register_forward_hook(hook_abl))
    handles.append(
        layers[L4_CONS].self_attn.o_proj
        .register_forward_pre_hook(
            pre_o4, with_kwargs=True))

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

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args                 else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            cap['ai'].setdefault(li, []).append(
                x.detach().float().cpu().numpy()
                .copy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :] \
                .detach().float().cpu().numpy() \
                .copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li),
                           with_kwargs=True))
        handles.append(layers[li]
                       .register_forward_pre_hook(
                           pre_layer(li),
                           with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]
        l4_cap.clear()
        state_l4['mode'] = None

    def forward_batch(toks_list):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list,
                               device='cuda'))
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
    a7_diff = float(np.abs(xdir @ Vt8_S.T
                           - dcks_S).max())
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
    log('a4 %.2e a5 %.2e a6 rel %.2e '
        'ok=%s/%s/%s sep_f=%.2f'
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
        ids = tok(prompt,
                  add_special_tokens=False)[
            'input_ids']
        clear_cap()
        state_fin['on'] = True
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
                nid = int(
                    out.logits[0, -1].argmax())
        state_fin['on'] = False
        for key in ('ids', 's', 'c8'):
            rec[key] = np.array(rec[key])
        rec['prompt_ids'] = np.array(ids)
        return rec

    def prefill_step2(pr, ids, past):
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
        return p

    def run_chain(pr, p_pos, erase, abl_idx=None,
                  l4mode=None):
        """3023 verbatim two-step chain + L4
        o_proj-input capture mode."""
        ids = tok(pr,
                  add_special_tokens=False)[
            'input_ids']
        clear_cap()
        rs.clear()
        state_abl['on'] = abl_idx is not None
        state_abl['idx'] = None
        state_abl['norm'] = 0.0
        state_abl['mnorm'] = 0.0
        state_l4['mode'] = l4mode
        state_r['on'] = True
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if erase:
                kv_scale_arm(past, p_pos, 0.0,
                             'KONLY', G7_HEAD)
            if abl_idx is not None:
                state_abl['idx'] = abl_idx
            out2 = model(
                input_ids=torch.tensor(
                    [[int(ids[-1])]],
                    device='cuda'),
                past_key_values=past,
                use_cache=False)
        state_r['on'] = False
        state_abl['on'] = False
        state_abl['idx'] = None
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        res36 = np.stack(
            [rs[li][-1][0].astype(np.float64)
             for li in range(NL)])
        return p_, res36

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
                         and a27_ok and a29_ok and a31_ok
                         and a33_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    a13_ok = False
    a28_diff = None
    a28_ok = False
    a30_diff = None
    a30_ok = False
    a35_diff = None
    a35_ok = False
    a36_diff = None
    a36_ok = False
    tags = []
    nL = 0
    nC = 0
    js_er = []
    js_abl = []
    js0_all = []
    js_c_ab = []
    js_c_co = []
    conc8_all = []
    conc8_c_all = []
    conc8_null_all = []
    p_h_all = []
    p_h_erase_all = []
    p_h_null_all = []
    p_h_c_ab = []
    gqa_all = []
    e4cos_all = []
    abl_rel_all = []
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
        log('a10 gen determinism ids_same=%s '
            'rel=%.2e ok=%s'
            % (ids_same, a10_rel, a10_ok), lines)

        drift_s = []
        for pi in range(len(GEN_PROMPTS)):
            rec = recs[pi]
            drift_s.append(abs(float(
                rec['s'][K_GEN - 1]
                - rec['s_pre'])))
        drift_med = round(float(np.median(drift_s)),
                          4)
        a13_diff = abs(drift_med - T3_3009_DRIFT)
        a13_ok = bool(a13_diff < A13_GATE)
        log('a13 T3 drift %.4f vs 3009 %.4f '
            'diff=%.2e ok=%s'
            % (drift_med, T3_3009_DRIFT, a13_diff,
               a13_ok), lines)

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
                    log('T2 P%d skipped (lp=%d '
                        'cp=%d)'
                        % (pi, len(lp), len(cp)),
                        lines)
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

            # ---------- T1 tags + baselines ------
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
                    p = prefill_step2(
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
            log('T1 logic positions n=%d' % nL,
                lines)
            assert tags == tags22, (tags, tags22)
            log('tag order == 3022 npz tags OK',
                lines)

            tmap = {t: k for k, t
                    in enumerate(tags22)}
            rng_null = np.random.default_rng(
                SEED_RND + 28)
            Wo = layers[L4_CONS].self_attn \
                .o_proj.weight
            Wo_r = Wo.detach().float().view(
                HID, NH_Q, HD)

            def grab(mode):
                # step-2 forward is a single
                # token: [-1] IS the p_pos slot;
                # MUST be called right after the
                # chain (clear_cap wipes l4_cap)
                return l4_cap[mode][-1][
                    0, -1].astype(np.float64)

            def shares_of(d):
                d = np.asarray(d).reshape(
                    NH_Q, HD)
                nn = np.linalg.norm(d, axis=1)
                p_h = nn / max(float(nn.sum()),
                               1e-30)
                conc8 = float(
                    np.sort(p_h)[::-1][:8].sum())
                return p_h, conc8

            def per_tag(pi, p_pos):
                t = 'P%d:%d' % (pi, p_pos)
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                k22 = tmap[t]
                cidx = coal_sets[k22]
                nc = np.setdiff1d(
                    np.arange(inter), cidx)
                p0, res36_b = run_chain(
                    pr, p_pos, False, None,
                    'base')
                base_a = grab('base')
                q_e, res36_e = run_chain(
                    pr, p_pos, True, None,
                    'erase')
                erase_a = grab('erase')
                q_a, _ = run_chain(
                    pr, p_pos, False, cidx, 'abl')
                abl_a = grab('abl')
                p_h, conc8 = shares_of(
                    abl_a - base_a)
                p_he, _ = shares_of(
                    erase_a - base_a)
                d = (abl_a - base_a).reshape(
                    NH_Q, HD)
                e4 = res36_e[4] - res36_b[4]
                e4u = unit(e4)
                Ct = torch.einsum(
                    'oqh,qh->qo', Wo_r,
                    torch.tensor(
                        d, dtype=torch.float32,
                        device='cuda'))
                Cn = Ct.cpu().numpy() \
                    .astype(np.float64)
                cnorm = np.linalg.norm(Cn, axis=1)
                e4cos = (Cn @ e4u) \
                    / np.maximum(cnorm, 1e-30)
                ph_null = []
                c8_null = []
                for m_ in range(N_NULL):
                    ridx = nc[rng_null.choice(
                        len(nc), TOPK,
                        replace=False)]
                    run_chain(pr, p_pos, False,
                              ridx, 'null')
                    null_a = grab('null')
                    ph_n, c8_n = shares_of(
                        null_a - base_a)
                    ph_null.append(ph_n)
                    c8_null.append(c8_n)
                abl_rel = state_abl['norm'] \
                    / max(state_abl['mnorm'],
                          1e-30)
                log('%s js0=%.2e er=%.5f '
                    'abl=%.5f conc8=%.3f '
                    'nullmed=%.3f abl_rel=%.2e '
                    'e4cos_top3=%s'
                    % (t, js_nats(pb, p0),
                       js_nats(pb, q_e),
                       js_nats(pb, q_a), conc8,
                       float(np.median(c8_null)),
                       abl_rel,
                       np.argsort(
                           -np.abs(e4cos))[:3]
                       .tolist()), lines)
                return {'base_a': base_a,
                        'js0': js_nats(pb, p0),
                        'js_e': js_nats(pb, q_e),
                        'js_a': js_nats(pb, q_a),
                        'p_h': p_h,
                        'conc8': conc8,
                        'p_he': p_he,
                        'ph_null': ph_null,
                        'c8_null': c8_null,
                        'e4cos': e4cos,
                        'abl_rel': abl_rel,
                        'gqa': p_h.reshape(
                            8, 4).sum(1)}

            res_l = {}
            for t in tags:
                pi = int(t.split(':')[0][1:])
                p_pos = int(t.split(':')[1])
                res_l[t] = per_tag(pi, p_pos)
                js0_all.append(res_l[t]['js0'])
                js_er.append(res_l[t]['js_e'])
                js_abl.append(res_l[t]['js_a'])
                conc8_all.append(
                    res_l[t]['conc8'])
                conc8_null_all.append(
                    res_l[t]['c8_null'])
                p_h_all.append(res_l[t]['p_h'])
                p_h_erase_all.append(
                    res_l[t]['p_he'])
                p_h_null_all.append(
                    res_l[t]['ph_null'])
                gqa_all.append(res_l[t]['gqa'])
                e4cos_all.append(
                    res_l[t]['e4cos'])
                abl_rel_all.append(
                    res_l[t]['abl_rel'])

            # content arms: ab (clean) + co (a36)
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
                    kref = tmap[lt0[0]] \
                        if lt0 else 0
                    cidx = coal_sets[kref]
                    pb = caps[pi][1]
                    base_ref = (res_l[lt0[0]][
                        'base_a'] if lt0
                        else None)
                    q_ab, _ = run_chain(
                        pr, pc, False, cidx,
                        'ablc')
                    ablc_a = grab('ablc')
                    p_hc, c8c = shares_of(
                        ablc_a - base_ref)
                    p_h_c_ab.append(p_hc)
                    conc8_c_all.append(c8c)
                    js_c_ab.append(js_nats(pb,
                                           q_ab))
                    q_co, _ = run_chain(
                        pr, pc, True, cidx, None)
                    js_c_co.append(js_nats(pb,
                                           q_co))
                    nC += 1
                    log('C P%d:%d ab=%.5f '
                        'co=%.5f conc8=%.3f'
                        % (pi, pc, js_c_ab[-1],
                           js_c_co[-1], c8c),
                        lines)

            # a28/a30/a35/a36
            a28_diff = float(np.max(np.abs(
                np.array(js_er) - js22))) \
                if js_er else None
            a28_ok = bool(a28_diff is not None
                          and a28_diff == 0.0)
            log('a28 erase chain vs 3022 js '
                'max|d|=%s ok=%s'
                % (a28_diff, a28_ok), lines)
            a30_diff = float(np.max(
                np.abs(np.array(js0_all)))) \
                if js0_all else None
            a30_ok = bool(a30_diff is not None
                          and a30_diff == 0.0)
            log('a30 capture self-consistency '
                'max js0=%s ok=%s'
                % (a30_diff, a30_ok), lines)
            a35_diff = float(np.max(np.abs(
                np.array(js_abl) - js23_abl))) \
                if js_abl else None
            a35_ok = bool(a35_diff is not None
                          and a35_diff == 0.0)
            log('a35 ABL_COAL vs 3023 js_abl_only '
                'max|d|=%s ok=%s'
                % (a35_diff, a35_ok), lines)
            a36_diff = float(np.max(np.abs(
                np.array(js_c_co) - js23_coalc))) \
                if js_c_co else None
            a36_ok = bool(a36_diff is not None
                          and a36_diff == 0.0)
            log('a36 content co vs 3023 '
                'js_coal_content max|d|=%s ok=%s'
                % (a36_diff, a36_ok), lines)

            # ---------- T2a ----------
            conc8_a = np.array(conc8_all)
            null_med = np.array(
                [float(np.median(row))
                 for row in conc8_null_all])
            diff = conc8_a - null_med
            n_pos = int(np.sum(diff > 0))
            n_neg = int(np.sum(diff < 0))
            p_pos_ = binom_ge(n_pos, nL) if nL \
                else None
            p_neg = binom_ge(n_neg, nL) if nL \
                else None
            T2a = {
                'n_logic': nL,
                'n_null_per_tag': N_NULL,
                'conc8_med': round(float(
                    np.median(conc8_a)), 4),
                'null_med_med': round(float(
                    np.median(null_med)), 4),
                'diff_med': round(float(
                    np.median(diff)), 4),
                'n_pos': n_pos, 'n_neg': n_neg,
                'p_binom_pos': round(p_pos_, 4)
                if p_pos_ is not None else None,
                'p_binom_neg': round(p_neg, 4)
                if p_neg is not None else None,
                'p_gate': P_GATE,
                'conc8_per_tag': [round(
                    float(v), 4)
                    for v in conc8_a],
                'null_med_per_tag': [round(
                    float(v), 4)
                    for v in null_med],
                'diff_per_tag': [round(
                    float(v), 4) for v in diff],
                'tags': tags,
                'note': 'conc8 = sum of top-8 '
                        'per-head norm shares of '
                        'the L4 o_proj-input delta '
                        '(abl - base); null = '
                        'random-32 non-coalition '
                        'zero-ablation chains; '
                        'diff = conc8 - null_med; '
                        'exact binomial one-sided'}
            log('T2a conc8 med=%.4f null=%.4f '
                'diff=%.4f n_pos=%d n_neg=%d '
                'p_pos=%s p_neg=%s'
                % (T2a['conc8_med'],
                   T2a['null_med_med'],
                   T2a['diff_med'], n_pos, n_neg,
                   p_pos_, p_neg), lines)

            # ---------- T2b ----------
            gqa_med = np.median(
                np.array(gqa_all), axis=0)
            rho15 = spearman(gqa_med, share15)
            top8_sets = [set(np.argsort(
                -row)[:8].tolist())
                for row in p_h_all]
            pair_j = []
            for i in range(nL):
                for j in range(i + 1, nL):
                    u = top8_sets[i] \
                        | top8_sets[j]
                    v = top8_sets[i] \
                        & top8_sets[j]
                    pair_j.append(
                        len(v) / max(len(u), 1))
            rho_er = [spearman(
                p_h_all[i], p_h_erase_all[i])
                for i in range(nL)]
            T2b = {
                'gqa_share_med': [round(
                    float(v), 4)
                    for v in gqa_med],
                'share15_3015': [round(
                    float(v), 4)
                    for v in share15],
                'spearman_gqa_vs_3015':
                    round(rho15, 4),
                'top8_pair_jaccard_med': round(
                    float(np.median(pair_j)), 4)
                if pair_j else None,
                'spearman_abl_vs_erase_med':
                    round(float(np.median(
                        rho_er)), 4),
                'note': 'GQA group shares = 4 '
                        'query heads per KV group; '
                        '3015 med_share_top8 is '
                        'the K-eraser impact share '
                        'per KV head (category '
                        'mismatch: erase response '
                        'vs baseline consumption); '
                        'rho_er = per-tag Spearman '
                        'between baseline-'
                        'consumption and erase-'
                        'response head profiles'}
            log('T2b rho15=%.4f top8 jac=%.4f '
                'rho_er_med=%.4f'
                % (rho15,
                   T2b['top8_pair_jaccard_med']
                   if pair_j else -1.0,
                   T2b['spearman_abl_vs_erase_med']
                   ), lines)

            # ---------- T2c ----------
            e4c = np.array(e4cos_all)
            top3_all = [np.argsort(
                -np.abs(row))[:3].tolist()
                for row in e4c]
            c8c_med = float(np.median(
                conc8_c_all)) if conc8_c_all \
                else None
            c8l_med = float(np.median(conc8_a))
            T2c = {
                'e4cos_abs_med': round(float(
                    np.median(np.abs(e4c))), 4),
                'e4cos_max': round(float(
                    np.max(np.abs(e4c))), 4),
                'top3_heads_mode': [
                    int(x) for x in np.bincount(
                        [h for row in top3_all
                         for h in row],
                        minlength=NH_Q)
                        .argsort()[::-1][:3]],
                'content_conc8_med': round(
                    c8c_med, 4)
                if c8c_med is not None
                else None,
                'logic_conc8_med': round(
                    c8l_med, 4),
                'note': 'per-head residual '
                        'contribution c_q = '
                        'einsum(W_o^L4.view('
                        '2560,32,128), d); '
                        'e4cos = cos(c_q, e4_'
                        'unit) per head; content '
                        'conc8 from the ab-chain '
                        'at the first content '
                        'position'}
            log('T2c e4cos med=%.4f max=%.4f '
                'content conc8=%s'
                % (T2c['e4cos_abs_med'],
                   T2c['e4cos_max'],
                   T2c['content_conc8_med']),
                lines)

            # ---------- verdict ----------
            abl_rel_med = float(np.median(
                abl_rel_all))
            gates_ok = bool(nL >= MIN_POS
                            and np.all(
                                np.array(js_abl)
                                > 0)
                            and abl_rel_med
                            > ABL_MIN)
            if not gates_ok:
                verdict = \
                    'consumption_undetermined_void'
            elif (T2a['diff_med'] > 0
                  and p_pos_ is not None
                  and p_pos_ <= P_GATE):
                verdict = 'consumption_' \
                          'concentrated_qwen'
            elif (T2a['diff_med'] < 0
                  and p_neg is not None
                  and p_neg <= P_GATE):
                verdict = 'consumption_spread_' \
                          'qwen'
            else:
                verdict = 'consumption_mixed_' \
                          'qwen'

            # ---------- T3 (descriptive) ------
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
        'a10_gen_det': {'rel': a10_rel,
                        'ok': a10_ok},
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
        'a33_3025': a33_ok,
        'a35_abl_chain_diff': a35_diff,
        'a36_content_co_diff': a36_diff,
    }
    a28_ok_f = bool(a28_diff is not None
                    and a28_diff == 0.0)
    a30_ok_f = bool(a30_diff is not None
                    and a30_diff == 0.0)
    a35_ok_f = bool(a35_diff is not None
                    and a35_diff == 0.0)
    a36_ok_f = bool(a36_diff is not None
                    and a36_diff == 0.0)
    res = {
        'phase': 3027,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim
                              and a10_ok and a13_ok
                              and a28_ok_f
                              and a30_ok_f
                              and a35_ok_f
                              and a36_ok_f),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c,
        'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': '',
    }
    with open(os.path.join(OUT, 'result.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2,
                  ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35,
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object),
            'tags': np.array(tags, dtype=object),
            'tags22': np.array(tags22,
                               dtype=object),
            'js_erase': np.array(js_er),
            'js_abl': np.array(js_abl),
            'js0_self': np.array(js0_all),
            'js_content_ab': np.array(js_c_ab),
            'js_content_co': np.array(js_c_co),
            'conc8': np.array(conc8_all),
            'conc8_content': np.array(
                conc8_c_all),
            'p_h': np.array(p_h_all),
            'p_h_erase': np.array(
                p_h_erase_all),
            'p_h_null': np.array(p_h_null_all),
            'p_h_content': np.array(p_h_c_ab),
            'gqa_share': np.array(gqa_all),
            'e4_cos': np.array(e4cos_all),
            'abl_rel': np.array(abl_rel_all)}
    npz_path = os.path.join(
        OUT, 'omega_p2u_consumer_heads_qwen.npz')
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
