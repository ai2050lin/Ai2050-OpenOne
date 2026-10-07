# -*- coding: utf-8 -*-
"""Phase 3032: Omega-P2z deep secondary-peak anatomy
(qwen).

Why: 3030 located the dose convexity in the
distributed readout (terminal share 0.45 pct, mid
52.5 pct, deep 28.7 pct).  The deep band L21-34 keeps
an interior local peak in the 3020 lens js profile
for 11/11 tags (P2 peaks at L32, js 0.336) and holds
28.7 pct of the positive lens excess - is that deep
peak carried by a sparse carrier (heads or MLP
neurons, like the L3 relay coalition) or is it
distributed (like the mid equalizer field)?

Design (3030 machine verbatim for chains/generation/
position selection/two-step protocol/lens; added
observation-only captures re-verified bit-level):
  Per logic tag, arms alpha in {0, 1, 2} (chains bit-
  identical to 3029/3030) + baseline capture, each
  grabbing the step-2 o_proj input (per head), the
  residual stack, xfin, AND the deep-band MLP captures
  (down_proj input h + MLP output m at L3 and L21-34).
  PRIMARY per tag: at the deep peak layer ldp (argmax
  counted positive lens excess within idx 17-30):
    head_top8_share = sum of top-8 head delta
      energies ||o_2 - o_b||^2 / total (alpha=2 arm)
    mlp_top32_share = sum of top-32 |s_l| / sum |s_l|,
      s_l = 2*dh*(e @ W_down)/||e||^2 with
      e = residual delta entering layer l+1 (3022
      attribution machine verbatim)
  Verdict on medians over deep-counted tags
  (pos_deep > 0, gate >= 5 tags).

Anchors (frozen):
  a0-a35 integrity chain (as 3030 + a35 = 3030)
  a1-a8 geometry/anchor rebuilds (3030 verbatim)
  a10/a13 generation determinism + 3009 drift
  a28 erase chain vs 3022 js bit-level 0.0
  a30 capture self-consistency 0.0
  a32 alpha=0 vs 3024 js_restore_coal bit-level 0.0
  a38 lens-terminal consistency < 1e-4
  a39 alpha=1 lens trajectory vs 3030 npz
      traj_alpha1 bit-level 0.0
  a40 per-head dnorm (alpha 1/0/2) vs 3029 npz
      d1/d2/d0 bit-level 0.0
  a41 L3 SwiGLU attribution (erase arm) vs 3022
      s_relay max|d| <= 1e-6 (probe measured
      6.6e-09; NOT bit-level because the 3022 h/m
      capture dtype path differs - registered)

Verdict (frozen):
  anchor fail                  => anchor_fail_all_void
  gates fail (nL<8 or any js_erase<=0 or
  n_deep_counted<5 or a38 fail)
                               => deep_peak_undetermined_
                                  void
  med head_top8_share >= 0.5  => deep_peak_head_
                                  concentrated_qwen
  elif med mlp_top32_share >= 0.5
                               => deep_peak_mlp_
                                  concentrated_qwen
  else                         => deep_peak_distributed_
                                  qwen

Tags: Omega-P2z / deep secondary-peak anatomy /
head vs MLP carrier at the deep lens peak / no
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
F_3029NPZ = os.path.join(
    D_3029, 'omega_p2w_recruitment_decomp_qwen.npz')
D_3030 = os.path.join(BASE, 'phase3030',
                      'omega_p2x_readout_convexity_'
                      'qwen')
F_3030NPZ = os.path.join(
    D_3030, 'omega_p2x_readout_convexity_qwen.npz')
PHASE = 3032
NAME = 'omega_p2z_deep_peak_anatomy_qwen'
OUT = os.path.join(BASE, 'phase3032', NAME)
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896
SEED_RND = 3009
K_GEN = 256
MIN_POS = 8
L3_GATED = 3
L_RELAY = 3
L_ERR = 4
G7_HEAD = 7
TOPK = 32
TOPH = 8
HDIM = 128
NHQ = 32
L_MIN = 4
LENS_START = 4
LENS_END = 35
MLP_LAYERS = [L_RELAY] + list(range(8, 35))
DEEP_LO, DEEP_HI = 17, 31
LATE_LO, LATE_HI = 27, 31
MID_LO, MID_HI = 4, 17
HEAD_GATE = 0.5
MLP_GATE = 0.5
N_DEEP_GATE = 5
A41_GATE = 1e-6
REST_MIN = 1e-6
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
    'mode': 'qwen3-4b; 3030 machine verbatim for '
            'anchors/chains/generation/position '
            'selection/two-step protocol/logit-lens; '
            'added observation-only deep MLP captures '
            '(down_proj input h + MLP output m at L3 '
            'and L21-34) with dedicated state_mcap '
            'gate (state_hcap stays capture_base-only '
            'so hook_hcap h_base is NOT overwritten - '
            'otherwise the alpha=0/2 patch would '
            'neutralize); bit-identity re-verified by '
            'a28/a30/a32/a39/a40',
    'question': 'the deep band L21-34 holds 28.7 pct '
                'of the positive lens excess and '
                'every 3020 profile has an interior '
                'deep local peak - is the deep '
                'secondary peak carried by a sparse '
                'carrier (top-8 heads or top-32 MLP '
                'neurons, like the L3 relay '
                'coalition) or distributed (like '
                'the mid equalizer field)?',
    'T1': 'sources: 3022 npz (coalition + s_relay + '
          'js), 3024 npz (a32), 3029 npz (d1/d2/d0 + '
          'theta), 3030 npz (traj_alpha1 + js arms); '
          'per tag arms alpha in {0,1,2} bit-identical '
          'to 3029/3030 + baseline capture; captures: '
          'o_proj input per head, residual stack, '
          'xfin, deep MLP h/m',
    'T2a': 'PRIMARY: per tag deep peak layer ldp = '
           'argmax counted positive lens excess '
           'within idx 17-30 (E matrix as 3030: '
           'excess_l = jsl(2)-2*jsl(1)+jsl(0), '
           'counted iff jsl(2)_l > 2*theta_l); at '
           'ldp: head_top8_share (alpha=2 arm, '
           'sum top-8 ||o_2-o_b||^2 / total) and '
           'mlp_top32_share (3022 attribution '
           'machine: s_l = 2*dh*(e @ W_down)/'
           '||e||^2, e = residual delta entering '
           'layer l+1; sum top-32 |s| / sum |s|); '
           'verdict on medians over deep-counted '
           'tags (pos_deep > 0, gate >= 5): med '
           'head >= 0.5 => head_concentrated; '
           'elif med mlp >= 0.5 => '
           'mlp_concentrated; else distributed',
    'T2b': 'DESCRIPTIVE: late share E[idx27-30]/'
           'pos_deep per tag; interior deep peak '
           'flags; deep-vs-mid carrier contrast '
           '(same shares at mid peak); rho_deep '
           'recruitment (3029 machine, deep layers '
           '21-35, theta from 3029 npz); lens '
           'decode top-5 tokens at ldp (alpha=2 '
           'vs base)',
    'T2c': 'DESCRIPTIVE: sham calibration stats; '
           'attribution ident metric at L3 and '
           'deep peak',
    'corrections': 'a41 gate 1e-6 not bit-level: '
                   '3022 h/m capture dtype path '
                   'differs (probe measured 6.6e-09 '
                   'on P0:5); probe also confirmed '
                   'run_chain MUST re-prefill (the '
                   'step-2 forward pollutes the KV '
                   'cache even with use_cache=False). '
                   'run1: 2993 npz path missing '
                   'subdir. run2: mid-peak contrast '
                   'layers not in MLP capture set - '
                   'extended to L8-34. run3: a32 '
                   'compared the CAPTURE SELF-js '
                   'array (all 0) instead of the '
                   'alpha=0 ARM js (3020-style '
                   'js0_all vs a0_arr separation '
                   'lost) - false failure 1.86e-02 = '
                   'max(js_restore_coal); fixed by '
                   'storing js_alpha0_all; also '
                   'med_js_sham now records the true '
                   'sham erase js (3029/3030 defect '
                   'not repeated). run4: T2c ident '
                   'denominator missed /ne2 (3022 '
                   'scale formula) - ident '
                   'mis-normalized 0.78, descriptive '
                   'only, fixed; a41 and verdict '
                   'unaffected',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'gates fail => '
               'deep_peak_undetermined_void; med '
               'head_top8 >= 0.5 => '
               'deep_peak_head_concentrated_qwen; '
               'elif med mlp_top32 >= 0.5 => '
               'deep_peak_mlp_concentrated_qwen; '
               'else => deep_peak_distributed_qwen',
    'tags': 'Omega-P2z / deep secondary-peak '
            'anatomy / head vs MLP carrier at the '
            'deep lens peak / no hallucination '
            'naming',
}

INTEG = [
    ('phase2993/logic_signature_registration',
     'logic_signature_length_robust'),
    ('phase3007/omega_p2a_generation_trajectory_qwen',
     'logic_locked_perturb_divergent_qwen'),
    ('phase3008/omega_p2b_logic_kv_causal_qwen',
     'logic_sig_gen_only_qwen'),
    ('phase3009/omega_p2c_kv_scale_causal_qwen',
     'kv_scale_saturated_qwen'),
    ('phase3010/omega_p2d_logitlens_causal_qwen',
     'logitlens_logic_specific_qwen'),
    ('phase3012/omega_p2f_gate_surgery_qwen',
     'gate_mixed_qwen'),
    ('phase3013/omega_p2g_kv_content_'
     'decomposition_qwen', 'position_specific_gate_'
     'qwen'),
    ('phase3014/omega_p2h_reverse_dose_law_qwen',
     'gate_destruction_fragile_qwen'),
    ('phase3015/omega_p2i_k_consumer_heads_qwen',
     'k_consumer_mixed_qwen'),
    ('phase3016/omega_p2j_amplification_trace_qwen',
     'amp_distributed_qwen'),
    ('phase3017/omega_p2k_deep_absorption_qwen',
     'absorption_mixed_qwen'),
    ('phase3018/omega_p2l_dilution_decomposition_'
     'qwen', 'decomp_cancellation_dominant_qwen'),
    ('phase3019/omega_p2m_mlp_band_identity_qwen',
     'mlp_band_distributed_qwen'),
    ('phase3020/omega_p2n_readout_specificity_qwen',
     'injection_readout_asymmetric_qwen'),
    ('phase3021/omega_p2o_injection_anatomy_qwen',
     'injection_mixed_qwen'),
    ('phase3022/omega_p2p_l3_relay_neurons_qwen',
     'relay_dedicated_coalition_qwen'),
    ('phase3023/omega_p2q_relay_causal_ablation_'
     'qwen', 'relay_causal_toxic_void'),
    ('phase3024/omega_p2r_restore_ablation_qwen',
     'relay_restore_load_bearing_qwen'),
    ('phase3028/omega_p2v_dose_symmetry_qwen',
     'dose_superlinear_qwen'),
    ('phase3029/omega_p2w_recruitment_decomp_qwen',
     'amplify_existing_dominant_qwen'),
    ('phase3030/omega_p2x_readout_convexity_qwen',
     'readout_distributed_convex_qwen'),
]


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
    for fn in ('execution.json', 'result.json',
               'seal.json'):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            os.remove(p)
    npz_path = os.path.join(OUT, NAME + '.npz')
    if os.path.exists(npz_path):
        os.remove(npz_path)

    script = os.path.join(
        r'D:\AI2050\Ai2050-OpenOne\tests\glm5',
        'phase%d_%s.py' % (PHASE, NAME))
    with open(os.path.join(OUT, 'execution.json'),
              'w', encoding='utf-8') as f:
        json.dump({'phase': PHASE, 'name': NAME,
                   'created':
                       time.strftime(
                           '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'model': 'qwen3-4b',
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'prompts': list(GEN_PROMPTS),
                   'mlp_layers': MLP_LAYERS,
                   'deep_band': [DEEP_LO, DEEP_HI],
                   'late_band': [LATE_LO, LATE_HI],
                   'head_gate': HEAD_GATE,
                   'mlp_gate': MLP_GATE,
                   'n_deep_gate': N_DEEP_GATE,
                   'a41_gate': A41_GATE,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- integrity chain ----------
    integ_ok = True
    for rel, expect in INTEG:
        d = os.path.join(BASE, rel)
        seal = json.load(open(d + r'\seal.json',
                              encoding='utf-8'))
        r = json.load(open(d + r'\result.json',
                           encoding='utf-8'))
        ok = bool(seal['result_sha256_8']
                  == sha8(d + r'\result.json')
                  and r['final_verdict'] == expect
                  and r['anchor_all_ok'] is True)
        if not ok:
            integ_ok = False
            log('integrity FAIL %s (verdict=%s)'
                % (rel, r['final_verdict']), lines)
    a0_ok = integ_ok
    log('a0-a35 integrity chain ok=%s' % a0_ok,
        lines)

    # ---------- 3011 l_star special ----------
    r11 = json.load(open(
        os.path.join(BASE, 'phase3011',
                     'omega_p2e_layer_js_localization_'
                     'qwen', 'result.json'),
        encoding='utf-8'))
    a15_ok = bool(r11['T2a']['l_star'] == L3_GATED)
    log('a15 3011 l_star=%s ok=%s'
        % (r11['T2a']['l_star'], a15_ok), lines)

    # ---------- 2993 signature ----------
    z93 = np.load(os.path.join(
        BASE, 'phase2993',
        'logic_signature_registration',
        'logic_signature_registration.npz'),
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
    log('a1 w2/w1024 recompute %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    # ---------- sealed sources ----------
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
    log('3022 source s_relay %s dirs all pos=%s'
        % (s22.shape,
           all(d == 'pos' for d in dirs22)), lines)

    z24 = np.load(F_3024NPZ, allow_pickle=True)
    js24_coal = z24['js_restore_coal'].astype(
        np.float64)
    assert list(z24['tags'].astype(str)) == tags22

    z29 = np.load(F_3029NPZ, allow_pickle=True)
    d1_29 = z29['d1'].astype(np.float64)
    d2_29 = z29['d2'].astype(np.float64)
    d0_29 = z29['d0'].astype(np.float64)
    theta29 = float(z29['theta'])
    assert [str(t) for t in z29['tags']] == tags22

    z30 = np.load(F_3030NPZ, allow_pickle=True)
    traj1_30 = z30['traj_alpha1'].astype(np.float64)
    tags30 = [str(t) for t in z30['tags']]
    assert tags30 == tags22

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
            assert len(ids) == 1, '%s -> %s' \
                % (t, ids)
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

    # ---------- hooks ----------
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
    state_mcap = {'on': False}
    mlp_h = {}
    mlp_m = {}
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
            fin_cap['x'] = args[0][:, -1, :] \
                .detach().float().cpu().numpy() \
                .copy()
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
        cols = Wd3[:, idx]
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

    def pre_down_mcap(li):
        def h(module, args, kwargs):
            if state_mcap['on']:
                x = args[0]
                mlp_h.setdefault(li, []).append(
                    x[:, -1, :].detach().float()
                    .cpu().numpy().copy())
            return None
        return h

    def post_mlp_mcap(li):
        def h(module, args, kwargs, output):
            if state_mcap['on']:
                mlp_m.setdefault(li, []).append(
                    output[:, -1, :].detach()
                    .float().cpu().numpy().copy())
            return None
        return h

    Wd3 = layers[L_RELAY].mlp.down_proj.weight
    inter = int(Wd3.shape[1])
    assert inter == s22.shape[1], (inter, s22.shape)
    Wf3 = Wd3.detach().float()
    Wf_deep = {li: layers[li].mlp.down_proj.weight
               .detach().float()
               for li in MLP_LAYERS}

    handles.append(layers[L_RELAY].mlp.down_proj
                   .register_forward_pre_hook(
                       hook_hcap, with_kwargs=True))
    handles.append(layers[L_RELAY].mlp.down_proj
                   .register_forward_hook(hook_rest))
    for li in MLP_LAYERS:
        handles.append(layers[li].mlp.down_proj
                       .register_forward_pre_hook(
                           pre_down_mcap(li),
                           with_kwargs=True))
        handles.append(layers[li].mlp
                       .register_forward_hook(
                           post_mlp_mcap(li),
                           with_kwargs=True))
    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li),
                           with_kwargs=True))
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
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S)
                    .max())
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
    log('a4 %.2e a5 %.2e a6 rel %.2e ok=%s/%s/%s'
        % (a4_diff, a5_diff, a6_rel, a4_ok, a5_ok,
           a6_ok), lines)

    def gen_coords():
        return fin_cap['x'][0].astype(np.float64)

    def kv_scale_arm(past, p, s, arm, h=None,
                     li=L3_GATED):
        L = past.layers[li]
        if h is None:
            if arm in ('JOINT', 'KONLY'):
                L.keys[:, :, p, :] *= s
        else:
            if arm in ('JOINT', 'KONLY'):
                L.keys[:, h, p, :] *= s

    def generate(prompt, k):
        ids = tok(prompt, add_special_tokens=False)[
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
                nid = int(out.logits[0, -1].argmax())
        state_fin['on'] = False
        rec['ids'] = np.array(rec['ids'])
        rec['s'] = np.array(rec['s'])
        rec['c8'] = np.array(rec['c8'])
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

    def grab_mlps():
        h = {li: mlp_h[li][-1][0]
             .astype(np.float64)
             for li in MLP_LAYERS}
        m = {li: mlp_m[li][-1][0]
             .astype(np.float64)
             for li in MLP_LAYERS}
        return h, m

    def capture_base(pi, p_pos):
        pr = GEN_PROMPTS[pi]
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
        rs.clear()
        mlp_h.clear()
        mlp_m.clear()
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        state_hcap['on'] = True
        state_o['on'] = True
        state_mcap['on'] = True
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
        state_mcap['on'] = False
        r_b = grab_r()
        xf_b = fin_cap['x'][0].astype(np.float64)
        h_b, m_b = grab_mlps()
        return p0, grab_o(), r_b, xf_b, h_b, m_b

    def run_chain(pr, p_pos, erase, rest=None):
        ids = tok(pr, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        o_in.clear()
        rs.clear()
        mlp_h.clear()
        mlp_m.clear()
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        state_rest['mag'] = 0.0
        state_rest['onorm'] = 0.0
        state_o['on'] = True
        state_mcap['on'] = True
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
        state_mcap['on'] = False
        state_rest['active'] = False
        state_rest['idx'] = None
        state_rest['alpha'] = None
        r_stack = grab_r()
        xf = fin_cap['x'][0].astype(np.float64)
        h_m, m_m = grab_mlps()
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        p_ = np.exp(lg)
        p_ = p_ / p_.sum()
        return p_, r_stack, xf, h_m, m_m

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
                         and a15_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = None
    a10_rel = None
    a10_ok = False
    a13_diff = None
    a13_ok = False
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
    a40_diff = None
    a40_ok = False
    a41_diff = None
    a41_ok = False
    tags = []
    nL = nS = 0
    js_er = []
    js0_all = []
    js2_all = []
    mag2_all = []
    traj0_list = []
    traj1_list = []
    traj2_list = []
    rel38_list = []
    theta_l = None
    js_sham_all = []
    a0_arr = np.zeros(0)
    E_all = None
    incl_all = None
    ldp_list = []
    head8_list = []
    mlp32_list = []
    head8_mid = []
    mlp32_mid = []
    late_list = []
    rho_deep_list = []
    dec_list = []
    ident_list = []
    d1_list = []
    d2_list = []
    d0_list = []
    a41_all = []
    E_rows = []
    incl_rows = []
    s_deep_store = []
    ldp_layer_list = []
    js_alpha0_all = []
    if anchor_prelim:
        for pi, pr in enumerate(GEN_PROMPTS):
            recs[pi] = generate(pr, K_GEN)
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
        log('a10 ids_same=%s rel=%.2e ok=%s; a13 '
            'drift %.4f diff=%.2e ok=%s'
            % (ids_same, a10_rel, a10_ok, drift_med,
               a13_diff, a13_ok), lines)

        if not (a10_ok and a13_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- positions ----------
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
                    continue
                lp_use = lp[:2]
                cp_use = list(rng2.choice(
                    cp, size=2, replace=False)) \
                    if len(cp) >= 2 else cp[:2]
                sham_pool = [i for i in sp
                             if i not in lp_use
                             and i not in cp_use]
                if not sham_pool:
                    sham_pool = [i for i in
                                 range(len(ids))
                                 if i not in lp_use
                                 and i not in cp_use]
                sham = int(rng2.choice(sham_pool)) \
                    if sham_pool else None
                ent['positions'] = {
                    'logic': lp_use,
                    'content': [int(x) for x
                                in cp_use],
                    'sham': sham}
                sel['P%d' % pi] = ent

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

            tags = []
            for pi in caps:
                for p_pos in sel['P%d' % pi][
                        'positions']['logic']:
                    tags.append('P%d:%d'
                                % (pi, p_pos))
            nL = len(tags)
            log('T1 logic positions n=%d' % nL, lines)
            assert tags == tags22, (tags, tags22)

            tmap = {t: k for k, t
                    in enumerate(tags22)}

            # ---------- lens machine ----------
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

            # ---------- sham floor ----------
            sham_traj_list = []
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped') \
                        or ent['positions'][
                            'sham'] is None:
                    continue
                p_sh = int(ent['positions']['sham'])
                pb = caps[pi][1]
                p0s, o_bs, r_bs, xf_bs, hb_s, \
                    mb_s = capture_base(pi, p_sh)
                q_s, r_s, xf_s, hh, mm = run_chain(
                    pr, p_sh, True, None)
                grab_o()
                js_sham_all.append(
                    js_nats(pb, q_s))
                nS += 1
                sham_traj_list.append(
                    traj_of(r_s, xf_s,
                            r_bs, xf_bs))
            if sham_traj_list:
                theta_l = np.median(
                    np.stack(sham_traj_list),
                    axis=0)
            log('sham n=%d med_js=%.6f'
                % (nS,
                   float(np.median(js_sham_all))
                   if js_sham_all else -1), lines)

            # ---------- arms ----------
            def deep_share_head(o_2, o_b, li):
                dd = np.linalg.norm(
                    o_2[li].reshape(NHQ, HDIM)
                    - o_b[li].reshape(NHQ, HDIM),
                    axis=1) ** 2
                tot = float(dd.sum())
                if tot <= 0:
                    return np.nan
                top = np.sort(dd)[::-1][:TOPH]
                return float(top.sum() / tot)

            def deep_share_mlp(s_vec):
                a = np.abs(s_vec)
                tot = float(a.sum())
                if tot <= 1e-30:
                    return np.nan
                top = np.sort(a)[::-1][:TOPK]
                return float(top.sum() / tot)

            def attrib(h_a, h_b, r_a, r_b, li):
                e = r_a[li + 1] - r_b[li + 1]
                dh = h_a[li] - h_b[li]
                ne2 = max(float(e @ e), 1e-30)
                u = (torch.tensor(
                    e, dtype=torch.float32,
                    device='cuda')
                    @ Wf_deep[li]) \
                    .cpu().numpy() \
                    .astype(np.float64)
                s = 2.0 * dh * u / ne2
                return s, float(
                    np.linalg.norm(e)), ne2

            for t in tags:
                pi = int(t.split(':')[0][1:])
                p_pos = int(t.split(':')[1])
                pr = GEN_PROMPTS[pi]
                pb = caps[pi][1]
                cidx = coal_sets[tmap[t]]
                p0, o_b, r_b, xf_b, h_b, m_b = \
                    capture_base(pi, p_pos)
                js0 = js_nats(pb, p0)
                q_e, r_e, xf_e, h_e, m_e = \
                    run_chain(pr, p_pos, True, None)
                o_e = grab_o()
                out_er = js_nats(pb, q_e)
                q_0, r_0, xf_0, h_0, m_0 = \
                    run_chain(pr, p_pos, True,
                              (cidx, 0.0))
                o_0 = grab_o()
                js0d = js_nats(pb, q_0)
                q_2, r_2, xf_2, h_2, m_2 = \
                    run_chain(pr, p_pos, True,
                              (cidx, 2.0))
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
                js_er.append(out_er)
                js0_all.append(js0)
                js_alpha0_all.append(js0d)
                js2_all.append(js2d)
                mag2_all.append(mag2)
                traj0_list.append(t0a)
                traj1_list.append(t1)
                traj2_list.append(t2a)
                rel38_list.append(rel38)

                # dnorm (3029 machine)
                k = tmap[t]
                d1m = np.zeros((NL, NHQ))
                d2m = np.zeros((NL, NHQ))
                d0m = np.zeros((NL, NHQ))
                for li in range(L_MIN, NL):
                    db = o_b[li].reshape(NHQ, HDIM)
                    d1m[li] = np.linalg.norm(
                        o_e[li].reshape(NHQ, HDIM)
                        - db, axis=1) \
                        / (np.linalg.norm(db, axis=1)
                           + 1e-12)
                    d2m[li] = np.linalg.norm(
                        o_2[li].reshape(NHQ, HDIM)
                        - db, axis=1) \
                        / (np.linalg.norm(db, axis=1)
                           + 1e-12)
                    d0m[li] = np.linalg.norm(
                        o_0[li].reshape(NHQ, HDIM)
                        - db, axis=1) \
                        / (np.linalg.norm(db, axis=1)
                           + 1e-12)
                d1_list.append(d1m)
                d2_list.append(d2m)
                d0_list.append(d0m)

                # a41: L3 attribution erase arm
                s_l3, _, _ = attrib(h_e, h_b,
                                    r_e, r_b,
                                    L_RELAY)
                a41_tag = float(
                    np.abs(s_l3 - s22[k]).max())
                a41_all.append(a41_tag)
                e4_i = r_e[L_ERR] - r_b[L_ERR]
                dm_i = m_e[L_RELAY] \
                    - m_b[L_RELAY]
                ne2_i = max(float(e4_i @ e4_i),
                            1e-30)
                ident_list.append(float(
                    abs(np.sum(s_l3)
                        - 2.0 * float(dm_i @ e4_i)
                        / ne2_i)
                    / max(2.0
                          * float(np.linalg.norm(
                              dm_i))
                          * float(np.linalg.norm(
                              e4_i)) / ne2_i,
                          1e-30)))

                # deep/mid peak analysis
                excess = t2a - 2.0 * t1 + t0a
                if theta_l is not None:
                    incl = t2a > (2.0
                                  * theta_l)
                else:
                    incl = np.zeros_like(t2a,
                                         dtype=bool)
                E_vec = np.where(
                    incl, np.maximum(excess, 0.0),
                    0.0)
                E_rows.append(E_vec)
                incl_rows.append(incl)
                pos_deep = float(
                    E_vec[DEEP_LO:DEEP_HI].sum())
                if pos_deep > 0:
                    ldp = DEEP_LO + int(
                        np.argmax(
                            E_vec[DEEP_LO:DEEP_HI]))
                else:
                    ldp = -1
                ldp_list.append(ldp)
                if ldp >= 0:
                    li_l = ldp + 4
                    head8_list.append(
                        deep_share_head(o_2, o_b,
                                        li_l))
                    s_dp, e_n, _ = attrib(
                        h_2, h_b, r_2, r_b, li_l)
                    mlp32_list.append(
                        deep_share_mlp(s_dp))
                    s_deep_store.append(
                        s_dp.astype(np.float32))
                    ldp_layer_list.append(li_l)
                    late_list.append(float(
                        E_vec[LATE_LO:LATE_HI]
                        .sum()) / pos_deep)
                    # mid contrast
                    seg = E_vec[MID_LO:MID_HI]
                    if seg.sum() > 0:
                        lmp = MID_LO + int(
                            np.argmax(seg))
                        li_m = lmp + 4
                        head8_mid.append(
                            deep_share_head(
                                o_2, o_b, li_m))
                        s_mp, _, _ = attrib(
                            h_2, h_b, r_2, r_b,
                            li_m)
                        mlp32_mid.append(
                            deep_share_mlp(s_mp))
                    else:
                        head8_mid.append(np.nan)
                        mlp32_mid.append(np.nan)
                    # lens decode at ldp
                    lp2 = lens_lp(r_2[li_l])
                    lp0 = lens_lp(r_b[li_l])
                    top2 = np.argsort(lp2)[::-1][:5]
                    top0 = np.argsort(lp0)[::-1][:5]
                    dec_list.append(json.dumps({
                        'ldp_layer': int(li_l),
                        'arm2': [str(tok.decode(
                            [int(i)]))
                            for i in top2],
                        'base': [str(tok.decode(
                            [int(i)]))
                            for i in top0]}))
                    # rho_deep (3029 machine)
                    m1 = d1m[21:36] > theta29
                    m2v = d2m[21:36] > theta29
                    new = m2v & (~m1)
                    den = float(
                        (d2m[21:36][m2v] ** 2)
                        .sum())
                    rho_deep_list.append(float(
                        (d2m[21:36][new] ** 2)
                        .sum())
                        / max(den, 1e-30))
                else:
                    head8_list.append(np.nan)
                    mlp32_list.append(np.nan)
                    s_deep_store.append(
                        np.zeros((inter,),
                                 dtype=np.float32))
                    ldp_layer_list.append(-1)
                    late_list.append(np.nan)
                    head8_mid.append(np.nan)
                    mlp32_mid.append(np.nan)
                    dec_list.append(None)
                    rho_deep_list.append(np.nan)
                log('%s js0=%.2e er=%.5f a0=%.5f '
                    'a2=%.5f mag2=%.2e ldp=%d '
                    'head8=%.3f mlp32=%.3f late=%.3f'
                    % (t, js0, out_er, js0d, js2d,
                       mag2, ldp,
                       head8_list[-1],
                       mlp32_list[-1],
                       late_list[-1]), lines)

            # ---------- anchors 28-41 --------
            a28_diff = float(np.max(np.abs(
                np.array(js_er) - js22)))
            a28_ok = bool(a28_diff == 0.0)
            a30_diff = float(np.max(
                np.abs(np.array(js0_all))))
            a30_ok = bool(a30_diff == 0.0)
            a0_arr = np.array(js_alpha0_all)
            a32_diff = float(np.max(np.abs(
                a0_arr - js24_coal)))
            a32_ok = bool(a32_diff == 0.0)
            a38_rel = float(np.max(rel38_list)) \
                if rel38_list else None
            a38_ok = bool(a38_rel is not None
                          and a38_rel < 1e-4)
            traj1 = np.stack(traj1_list)
            if traj1.shape == traj1_30.shape:
                a39_diff = float(np.max(
                    np.abs(traj1 - traj1_30)))
            else:
                a39_diff = float('nan')
            a39_ok = bool(a39_diff == 0.0)
            D1 = np.stack(d1_list)
            D2 = np.stack(d2_list)
            D0 = np.stack(d0_list)
            a40_diff = float(max(
                np.abs(D1 - d1_29).max(),
                np.abs(D2 - d2_29).max(),
                np.abs(D0 - d0_29).max()))
            a40_ok = bool(a40_diff == 0.0)
            a41_diff = float(np.max(a41_all)) \
                if a41_all else float('nan')
            a41_ok = bool(a41_diff <= A41_GATE)
            log('a28 %.2e a30 %.2e a32 %.2e a38 %.2e '
                'a39 %.2e a40 %.2e a41 %.2e'
                % (a28_diff, a30_diff, a32_diff,
                   a38_rel or -1, a39_diff,
                   a40_diff, a41_diff), lines)

            E_all = np.stack(E_rows)
            incl_all = np.stack(incl_rows)

            # ---------- T2a verdict ----------
            head_arr = np.array(head8_list,
                                dtype=float)
            mlp_arr = np.array(mlp32_list,
                               dtype=float)
            valid = np.isfinite(head_arr) \
                & np.isfinite(mlp_arr)
            n_deep = int(valid.sum())
            med_head = float(np.nanmedian(
                head_arr[valid])) if n_deep \
                else None
            med_mlp = float(np.nanmedian(
                mlp_arr[valid])) if n_deep \
                else None
            gates_ok = bool(nL >= 8
                            and min(js_er) > 0
                            and n_deep
                            >= N_DEEP_GATE
                            and a38_ok)
            if not gates_ok:
                verdict = \
                    'deep_peak_undetermined_void'
            elif med_head >= HEAD_GATE:
                verdict = \
                    'deep_peak_head_concentrated_qwen'
            elif med_mlp >= MLP_GATE:
                verdict = \
                    'deep_peak_mlp_concentrated_qwen'
            else:
                verdict = \
                    'deep_peak_distributed_qwen'
            T2a = {
                'n_logic': nL,
                'n_sham': nS,
                'n_deep_counted': n_deep,
                'med_head_top8_share':
                    round(med_head, 4)
                    if med_head is not None
                    else None,
                'med_mlp_top32_share':
                    round(med_mlp, 4)
                    if med_mlp is not None
                    else None,
                'head_top8_share_per_tag':
                    [None if not np.isfinite(v)
                     else round(float(v), 4)
                     for v in head_arr],
                'mlp_top32_share_per_tag':
                    [None if not np.isfinite(v)
                     else round(float(v), 4)
                     for v in mlp_arr],
                'ldp_idx_per_tag': ldp_list,
                'ldp_layer_per_tag':
                    ldp_layer_list,
                'med_js_erase': round(
                    float(np.median(js_er)), 6),
                'mag2_med':
                    float('%.2e'
                          % float(np.median(
                              mag2_all))),
                'gates_ok': gates_ok}

            # ---------- T2b ----------
            late_arr = np.array(late_list,
                                dtype=float)
            hd8m = np.array(head8_mid, dtype=float)
            ml32m = np.array(mlp32_mid,
                             dtype=float)
            rho_arr = np.array(rho_deep_list,
                               dtype=float)
            T2b = {
                'late_share_per_tag':
                    [None if not np.isfinite(v)
                     else round(float(v), 4)
                     for v in late_arr],
                'med_late_share': round(
                    float(np.nanmedian(late_arr)),
                    4)
                if np.any(np.isfinite(late_arr))
                else None,
                'med_head8_mid': round(
                    float(np.nanmedian(hd8m)), 4)
                if np.any(np.isfinite(hd8m))
                else None,
                'med_mlp32_mid': round(
                    float(np.nanmedian(ml32m)),
                    4)
                if np.any(np.isfinite(ml32m))
                else None,
                'med_rho_deep': round(
                    float(np.nanmedian(rho_arr)),
                    4)
                if np.any(np.isfinite(rho_arr))
                else None,
                'decode_per_tag': dec_list,
                'note': 'deep shares at alpha=2 '
                        'arm; mid contrast at the '
                        'mid excess peak'}

            # ---------- T2c ----------
            T2c = {
                'med_js_sham': round(
                    float(np.median(js_sham_all)),
                    6) if js_sham_all else None,
                'a41_ident_med': round(
                    float(np.median(ident_list)),
                    6) if ident_list else None,
                'note': 'sham chain self-js '
                        'calibration; ident = '
                        'attribution identity '
                        'metric (3022 machine)'}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0
    anchors = {
        'a0_a35_integrity': a0_ok,
        'a15_3011_lstar': a15_ok,
        'a1_signature': a1_diff,
        'a2_dirs': a2_diff, 'a3_vt8': a3_diff,
        'a4_func': a4_diff, 'a5_null0': a5_diff,
        'a6_det_rel': a6_rel, 'a7_xdir': a7_diff,
        'a8_lwords': a8_ok,
        'a10_gen_det': {'rel': a10_rel,
                        'ok': a10_ok},
        'a13_drift_diff': a13_diff,
        'a28_erase_chain_diff': a28_diff,
        'a30_capture_self_diff': a30_diff,
        'a32_dose_alpha0_diff': a32_diff,
        'a38_lens_terminal_rel': a38_rel,
        'a39_traj_3030_diff': a39_diff,
        'a40_dnorm_3029_diff': a40_diff,
        'a41_s_relay_diff': a41_diff,
    }
    res = {
        'phase': PHASE,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            anchor_prelim and a10_ok and a13_ok
            and a28_ok and a30_ok and a32_ok
            and a38_ok and a39_ok and a40_ok
            and a41_ok),
        'anchors': anchors,
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': '',
    }
    with open(os.path.join(OUT, 'result.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2,
                  ensure_ascii=False)

    save = {
        'tags': np.array(tags, dtype=object),
        'js_erase': np.array(js_er),
        'js_alpha0': a0_arr,
        'js_alpha2': np.array(js2_all),
        'js0_self': np.array(js0_all),
        'mag2': np.array(mag2_all),
        'traj_alpha0': np.stack(traj0_list)
        if traj0_list else np.zeros((0, 32)),
        'traj_alpha1': np.stack(traj1_list)
        if traj1_list else np.zeros((0, 32)),
        'traj_alpha2': np.stack(traj2_list)
        if traj2_list else np.zeros((0, 32)),
        'E': E_all if E_all is not None
        else np.zeros((0, 32)),
        'incl': incl_all if incl_all is not None
        else np.zeros((0, 32), dtype=bool),
        'theta_l': theta_l
        if theta_l is not None
        else np.zeros(0),
        'ldp_idx': np.array(ldp_list, dtype=int),
        'ldp_layer': np.array(ldp_layer_list,
                              dtype=int),
        'head_top8': np.array(head8_list,
                              dtype=float),
        'mlp_top32': np.array(mlp32_list,
                              dtype=float),
        'head_top8_mid': np.array(head8_mid,
                                  dtype=float),
        'mlp_top32_mid': np.array(mlp32_mid,
                                  dtype=float),
        'late_share': np.array(late_list,
                               dtype=float),
        'rho_deep': np.array(rho_deep_list,
                             dtype=float),
        's_ldp': np.stack(s_deep_store)
        if s_deep_store else np.zeros((0, 0)),
        'ldp_mlp_layer': np.array(
            [l for l in ldp_layer_list if l >= 0],
            dtype=int),
        'd2': np.stack(d2_list)
        if d2_list else np.zeros((0, 0, 0)),
        'a41_per_tag': np.array(a41_all,
                                dtype=float),
        'decode': np.array(
            [d if d is not None else ''
             for d in dec_list], dtype=object),
    }
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
