# -*- coding: utf-8 -*-
"""Phase 3010: Omega-P2d logit-lens distribution-distance
causal readout (qwen).

Why: 3009 proved token-divergence is a DEAD causal readout -
saturation is driven by the EXISTENCE of a unit-position KV
perturbation, not its strength (s=0.2 sham still 26/32,
headroom gate never met at any scale).  Preregistered fix:
KEEP the intervention, CHANGE the readout - first-decode-step
next-token distribution JS distance (continuous, bounded by
ln 2, no saturation floor).

Design (3009 machine verbatim for anchors/geometry/
generation; T2 replaced):
  prompts  3009 GEN_PROMPTS verbatim (12), K_GEN=256 greedy
           baseline + determinism re-gen (T3 anchor a13).
  T2 PRIMARY logit-lens: per prompt, positions logic
           (<=2, in order) / 2 rng content / 1 sham selected
           with the 3009 protocol VERBATIM (seed chain
           SEED_RND+20 with SEED_RND=3009, explicit rebuild
           for position-level comparability with 3009);
           per scale s in (0.0, 0.25, 0.5, 0.75): prefill
           forward with K,V at position p scaled by s (all
           36 layers), read logits[0,-1]; JS := Jensen-
           Shannon distance (nats) vs the UNINTERVENED
           baseline next-token distribution of the same
           prompt; also argmax-flip indicator (descriptive).
           PRIMARY scale s=0.0: D = med JS_L - med JS_C
           pooled; perm two-sided N_PERM=10000,
           p_fam=p_raw; gates nL>=8 nC>=8; dose-response of
           D_s and sham calibration across scales
           descriptive (no headroom gate needed - JS is
           continuous).
  T3       DESCRIPTIVE 256-token drift (3009 verbatim) +
           a13 bit-level anchor vs 3009 stored T3.

Verdict (frozen):
  anchor fail                        => anchor_fail_all_void
  s=0: D>0 AND p_fam<0.05 AND gates  => logitlens_logic_
                                        specific_qwen
  s=0: D<0 AND p_fam<0.05 AND gates  => logitlens_content_
                                        specific_qwen
  else                               => logitlens_null_qwen

Anchors (frozen):
  a0  2993 integrity (seal + verdict + anchors)
  a1  w2/w1024 recompute vs 2993 npz (raw diff, rel < 1e-6)
  a2  dirs_word rebuild vs 2927 < 1e-5
  a3  Vt8 vs 2939 < 1e-6
  a4  proj_func vs 2935 < 1e-4 (bit-level)
  a5  proj_null0 vs 2935 < 1e-4 (bit-level)
  a6  baseline determinism < 1e-4
  a7  xdir identity < 1e-9
  a8  l_words(2993) all single-token here
  a9  3007 integrity (seal + verdict + anchors)
  a10 generation determinism: prompt 0 twice => identical
      ids AND c8 rel < 1e-4
  a11 3008 integrity: seal match AND verdict ==
      logic_sig_gen_only_qwen AND anchors ok
  a12 3009 integrity: seal match AND verdict ==
      kv_scale_saturated_qwen AND anchors ok
  a13 T3 med_drift_s_end vs 3009 stored 49.5123
      (|diff| < 5e-5, 4dp rounding gate)

Tags: Omega-P2d / readout-swap-not-strength-swap /
distribution JS distance / argmax-flip descriptive /
no hallucination naming.
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
OUT = os.path.join(BASE, 'phase3010',
                   'omega_p2d_logitlens_causal_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002/3007/3008/3009 verbatim
SEED_RND = 3009           # explicit: position selection
                          # chain identical to 3009
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
SCALES = (0.0, 0.25, 0.5, 0.75)
PRIMARY_S = 0.0
T3_3009_DRIFT = 49.5123   # 3009 stored med_drift_s_end
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
    'mode': 'qwen3-4b; 3009 machine verbatim for anchors/'
            'geometry/generation; T2 replaced by '
            'logit-lens JS readout; position selection '
            'chain identical to 3009 (SEED_RND=3009 '
            'explicit rebuild)',
    'question': 'with a continuous distribution readout '
                '(JS distance of the first-decode-step '
                'next-token distribution), do logic '
                'prompt positions carry more causal '
                'weight than matched content positions?',
    'T2': 'PRIMARY: TWO-STEP protocol both arms - prefill '
          'forward then feed ids[-1] once and read that '
          'step next-token distribution; intervened arm '
          'scales K,V at position p (all 36 layers) by s '
          'between the two forwards; JS nats vs the same-'
          'prompt unintervened two-step baseline; '
          'positions logic (<=2, in order) vs 2 rng '
          'content vs 1 sham, 3009 protocol verbatim; '
          'scales (0.0,0.25,0.5,0.75); PRIMARY s=0.0: '
          'D = med JS_L - med JS_C pooled, perm two-sided '
          'N=10000, p_fam=p_raw, gates nL>=8 nC>=8; '
          'dose-response D_s + sham calibration + '
          'argmax-flip rates descriptive',
    'T3': 'DESCRIPTIVE 256-token drift (3009 verbatim) + '
          'a13 anchor vs 3009 stored 49.5123',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               's=0 D>0 AND p<0.05 AND nL>=8 AND nC>=8 => '
               'logitlens_logic_specific_qwen; s=0 D<0 '
               'AND p<0.05 AND gates => '
               'logitlens_content_specific_qwen; else => '
               'logitlens_null_qwen',
    'tags': 'Omega-P2d / readout-swap-not-strength-swap / '
            'distribution JS distance / argmax-flip '
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
        json.dump({'phase': 3010,
                   'name': 'omega_p2d_logitlens_causal_'
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
                           D_3009 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'scales': list(SCALES),
                   'primary_s': PRIMARY_S,
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

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

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
        attnin_all = forward1(
            [func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
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

    # a7 xdir identity (3008/3009 verbatim, S_IDX 0/1/4)
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

    def kv_scale(past, p, s):
        for L in past.layers:
            L.keys[:, :, p, :] *= s
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

    def prefill_logits(prompt):
        """TWO-STEP protocol (both arms identical): prefill
        forward, then feed ids[-1] once more and read that
        step's next-token distribution.  The intervened arm
        scales K,V between the two forwards, so both arms
        are off-by-one in the same way - a clean paired
        contrast.  Returns (ids, probs64, argmax)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
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
        return ids, p, int(np.argmax(lg))

    def prefill_logits_abl(prompt, p_pos, s):
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            kv_scale(past, p_pos, s)
            out2 = model(
                input_ids=torch.tensor(
                    [[int(ids[-1])]], device='cuda'),
                past_key_values=past,
                use_cache=False)
        lg = out2.logits[0, -1].detach() \
            .double().cpu().numpy()
        lg = lg - lg.max()
        q = np.exp(lg)
        q = q / q.sum()
        return q, int(np.argmax(lg))

    dec_cache = {}

    def classify(t_id):
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
                         and a9_ok and a11_ok and a12_ok)
    recs = {}
    verdict = None
    T2 = T3 = None
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
            # (3009 protocol verbatim, seed chain
            #  SEED_RND+20 with SEED_RND=3009)
            rng2 = np.random.default_rng(
                SEED_RND + 20)
            sel = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ids = list(recs[pi]['prompt_ids'])
                lp = [i for i, t in enumerate(ids)
                      if int(t) in logic_tids]
                cp = [i for i, t in enumerate(ids)
                      if classify(int(t)) == 'content']
                sp = [i for i, t in enumerate(ids)
                      if classify(int(t)) in
                      ('func', 'other')]
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

            # ---------- T2 logit-lens JS ----------
            base_probs = {}
            base_amax = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ids, p, am = prefill_logits(pr)
                base_probs[pi] = p
                base_amax[pi] = am
            js_store = {s: {'L': [], 'C': [], 'S': []}
                        for s in SCALES}
            flip_store = {s: {'L': [], 'C': [], 'S': []}
                          for s in SCALES}
            for s in SCALES:
                for pi, pr in enumerate(GEN_PROMPTS):
                    ent = sel['P%d' % pi]
                    if ent.get('skipped'):
                        continue
                    pos = ent['positions']
                    pb = base_probs[pi]
                    for tag, key in (('logic', 'L'),
                                     ('content', 'C')):
                        for p_pos in pos[tag]:
                            q, am = prefill_logits_abl(
                                pr, int(p_pos), s)
                            js_store[s][key].append(
                                js_nats(pb, q))
                            flip_store[s][key].append(
                                int(am != base_amax[
                                    pi]))
                    if pos['sham'] is not None:
                        q, am = prefill_logits_abl(
                            pr, int(pos['sham']), s)
                        js_store[s]['S'].append(
                            js_nats(pb, q))
                        flip_store[s]['S'].append(
                            int(am != base_amax[pi]))
                log('T2 s=%.2f done: medJS L=%s C=%s '
                    'S=%s flip %s/%s/%s'
                    % (s,
                       float(np.median(
                           js_store[s]['L']))
                       if js_store[s]['L'] else None,
                       float(np.median(
                           js_store[s]['C']))
                       if js_store[s]['C'] else None,
                       float(np.median(
                           js_store[s]['S']))
                       if js_store[s]['S'] else None,
                       float(np.mean(
                           flip_store[s]['L']))
                       if flip_store[s]['L'] else None,
                       float(np.mean(
                           flip_store[s]['C']))
                       if flip_store[s]['C'] else None,
                       float(np.mean(
                           flip_store[s]['S']))
                       if flip_store[s]['S'] else None),
                    lines)
            # per-scale stats
            T2 = {'per_scale': {},
                  'primary_s': PRIMARY_S,
                  'min_pos_gate': MIN_POS,
                  'p_gate': P_GATE}
            rng3 = np.random.default_rng(
                SEED_RND + 30)
            for s in SCALES:
                ent = {}
                for key, tag in (('L', 'logic'),
                                 ('C', 'content'),
                                 ('S', 'sham')):
                    ent['n_' + tag] = len(
                        js_store[s][key])
                    ent['med_js_' + tag] = round(
                        float(np.median(
                            js_store[s][key])), 6) \
                        if js_store[s][key] else None
                    ent['flip_rate_' + tag] = round(
                        float(np.mean(
                            flip_store[s][key])), 4) \
                        if flip_store[s][key] else None
                nL = ent['n_logic']
                nC = ent['n_content']
                if nL >= MIN_POS and nC >= MIN_POS:
                    vL = np.array(js_store[s]['L'])
                    vC = np.array(js_store[s]['C'])
                    pool = np.concatenate([vL, vC])
                    n1 = vL.size
                    D = float(np.median(vL)
                              - np.median(vC))
                    cnt = 0
                    for _ in range(N_PERM):
                        pm = rng3.permutation(
                            pool.size)
                        d = np.median(
                            pool[pm[:n1]]) \
                            - np.median(
                            pool[pm[n1:]])
                        if abs(d) >= abs(D):
                            cnt += 1
                    p_fam = (cnt + 1) / (N_PERM + 1)
                    ent['D'] = round(D, 6)
                    ent['p_fam'] = round(p_fam, 5)
                T2['per_scale']['%.2f' % s] = ent
                log('T2 s=%.2f nL=%d nC=%d medJS '
                    '%s/%s sham=%s D=%s p=%s'
                    % (s, nL, nC,
                       ent.get('med_js_logic'),
                       ent.get('med_js_content'),
                       ent.get('med_js_sham'),
                       ent.get('D'),
                       ent.get('p_fam')), lines)
            # verdict at PRIMARY scale
            e0 = T2['per_scale']['%.2f' % PRIMARY_S]
            nL = e0['n_logic']
            nC = e0['n_content']
            D = e0.get('D')
            pf = e0.get('p_fam')
            gates = bool(nL >= MIN_POS
                         and nC >= MIN_POS
                         and D is not None
                         and pf is not None
                         and pf < P_GATE)
            if D is not None and D > 0 and gates:
                verdict = \
                    'logitlens_logic_specific_qwen'
            elif D is not None and D < 0 and gates:
                verdict = \
                    'logitlens_content_specific_qwen'
            else:
                verdict = 'logitlens_null_qwen'

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
                        '(med_drift_end 49.5123, std '
                        '25.80/34.84)'}
            log('T3 drift s=%.4f std e=%.3f l=%.3f'
                % (T3['med_drift_s_end'],
                   T3['med_std_early'],
                   T3['med_std_late']), lines)
    else:
        a10_rel = None
        a10_ok = False
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
    }
    res = {
        'phase': 3010,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2': T2, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': 'first run',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35, 'w2': w2_93, 'w1024': w1024_93,
            'l_words': np.array(l_words, dtype=object),
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object)}
    for s in SCALES:
        for key in ('L', 'C', 'S'):
            save['js_%.2f_%s' % (s, key)] = np.array(
                js_store[s][key], dtype=np.float64)
            save['flip_%.2f_%s' % (s, key)] = np.array(
                flip_store[s][key], dtype=np.int64)
    npz_path = os.path.join(
        OUT, 'omega_p2d_logitlens_causal_qwen.npz')
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
