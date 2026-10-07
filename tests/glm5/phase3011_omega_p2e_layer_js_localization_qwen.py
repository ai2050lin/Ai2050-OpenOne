# -*- coding: utf-8 -*-
"""Phase 3011: Omega-P2e layer x JS localization (qwen).

Why: 3010 established that logic-position KV erasure shifts
the first-decode-step next-token distribution ~11x more than
matched content positions (JS readout; D=0.0297 p_fam=0.0135
at s=0, four-scale D>0).  Open question: WHICH layers carry
the gating?  3010 scaled all 36 layers jointly; this phase
localizes the carrier layer(s) - the white-box surgery
target.

Design (3010 machine verbatim for anchors/geometry/
generation/position selection; T2 replaced):
  T2a PRIMARY layer sweep: per layer l in 0..35, K+V at
           position p scaled by s=0.0 (erasure) in layer l
           ONLY; JS nats vs the same-prompt unintervened
           two-step baseline (two-step protocol both arms
           identical to 3010); positions logic (<=2, in
           order) / 2 rng content / 1 sham selected with
           the 3009/3010 protocol VERBATIM (SEED_RND+20,
           SEED_RND=3009) for position-level comparability;
           per-layer D_l = med JS_L - med JS_C pooled;
           family of 36 tests handled by maxT permutation
           (N_PERM=10000, two-sided, same permuted labels
           across layers per draw): p_maxT for the layer
           with max |D_l|; gates nL>=8 nC>=8.
  T2b arm split (DESCRIPTIVE, quasi-post-hoc - labeled):
           at l* = argmax |D_l|: K-only and V-only s=0.0
           sweeps, D and p_raw each (no family correction).
  T2c dose (DESCRIPTIVE): at l*, s in (0.25,0.5,0.75)
           K+V, D_s + sham calibration.

Verdict (frozen):
  anchor fail                          => anchor_fail_all_
                                          void
  p_maxT < 0.05 AND D_l* > 0 AND gates => js_layer_
                                          localized_qwen
  p_maxT < 0.05 AND D_l* < 0 AND gates => js_layer_
                                          content_
                                          localized_qwen
  else                                 => js_layer_diffuse_
                                          qwen

Anchors (frozen): a0-a13 as 3010 verbatim (incl. a13 T3
drift vs 3009 stored 49.5123), plus
  a14 3010 integrity: seal match AND verdict ==
      logitlens_logic_specific_qwen AND anchors ok.

Tags: Omega-P2e / carrier-layer localization / maxT family
correction / K-vs-V arm split descriptive / no hallucination
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
OUT = os.path.join(BASE, 'phase3011',
                   'omega_p2e_layer_js_localization_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002/3007/3008/3009/3010 verbatim
SEED_RND = 3009           # position selection chain
                          # identical to 3009/3010
K_GEN = 256
N_PERM = 10000
P_GATE = 0.05
MIN_POS = 8
SCALE_MAIN = 0.0
SCALE_DOSE = (0.25, 0.5, 0.75)
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
    'mode': 'qwen3-4b; 3010 machine verbatim for anchors/'
            'geometry/generation/position selection '
            '(SEED_RND=3009 explicit rebuild); T2 replaced '
            'by per-layer localization sweep',
    'question': 'which transformer layer(s) carry the '
                'logic-position distribution gating found '
                'in 3010 - is the JS effect localized to '
                'specific layers or diffuse across depth?',
    'T2a': 'PRIMARY: per layer l in 0..35, K,V at position '
           'p scaled by s=0.0 in layer l ONLY (two-step '
           'protocol both arms, 3010 verbatim); JS nats vs '
           'same-prompt unintervened two-step baseline; '
           'positions logic (<=2, in order) vs 2 rng '
           'content vs 1 sham, 3009/3010 protocol verbatim; '
           'D_l = med JS_L - med JS_C pooled; family of 36 '
           '=> maxT permutation N=10000 two-sided (same '
           'permuted labels across layers per draw); '
           'gates nL>=8 nC>=8; per-layer raw p and sham '
           'JS descriptive',
    'T2b': 'DESCRIPTIVE quasi-post-hoc: at l* = argmax '
           '|D_l|, K-only and V-only s=0.0 sweeps; D and '
           'p_raw each, no family correction',
    'T2c': 'DESCRIPTIVE: at l*, s in (0.25,0.5,0.75) K+V; '
           'D_s + sham calibration',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'p_maxT<0.05 AND D_l*>0 AND nL>=8 AND nC>=8 '
               '=> js_layer_localized_qwen; p_maxT<0.05 '
               'AND D_l*<0 AND gates => '
               'js_layer_content_localized_qwen; else => '
               'js_layer_diffuse_qwen',
    'tags': 'Omega-P2e / carrier-layer localization / '
            'maxT family correction / K-vs-V arm split '
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
        json.dump({'phase': 3011,
                   'name': 'omega_p2e_layer_js_'
                           'localization_qwen',
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
                           D_3010 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_pos': MIN_POS,
                   'scale_main': SCALE_MAIN,
                   'scale_dose': list(SCALE_DOSE),
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

    # a7 xdir identity (3010 verbatim, S_IDX 0/1/4)
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

    def kv_scale_layers(past, p, s, li_list, arm):
        for li in li_list:
            L = past.layers[li]
            if arm in ('kv', 'k'):
                L.keys[:, :, p, :] *= s
            if arm in ('kv', 'v'):
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

    def prefill_logits(prompt):
        """TWO-STEP protocol (both arms identical): prefill
        forward, then feed ids[-1] once more and read that
        step's next-token distribution (3010 verbatim)."""
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            p, am = prefill_step2(prompt, ids, past)
        return ids, p, am

    def prefill_logits_abl(prompt, p_pos, s,
                           li_list=None, arm='kv'):
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            kv_scale_layers(past, p_pos, s,
                            li_list if li_list
                            is not None else range(NL),
                            arm)
            q, am = prefill_step2(prompt, ids, past)
        return q, am

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
                         and a9_ok and a11_ok and a12_ok
                         and a14_ok)
    recs = {}
    verdict = None
    T2a = T2b = T2c = T3 = None
    a10_rel = None
    a10_ok = False
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
            # (3009/3010 protocol verbatim, seed chain
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

            # ---------- T2a layer sweep ----------
            base_probs = {}
            base_amax = {}
            for pi, pr in enumerate(GEN_PROMPTS):
                ids, p, am = prefill_logits(pr)
                base_probs[pi] = p
                base_amax[pi] = am
            jsL = np.full((NL, 0), np.nan)
            jsC = np.full((NL, 0), np.nan)
            jsS = np.full((NL, 0), np.nan)
            tagsL, tagsC, tagsS = [], [], []
            L_store = [[] for _ in range(NL)]
            C_store = [[] for _ in range(NL)]
            S_store = [[] for _ in range(NL)]
            for pi, pr in enumerate(GEN_PROMPTS):
                ent = sel['P%d' % pi]
                if ent.get('skipped'):
                    continue
                pos = ent['positions']
                pb = base_probs[pi]
                for p_pos in pos['logic']:
                    row = np.zeros(NL)
                    for li in range(NL):
                        q, _ = prefill_logits_abl(
                            pr, int(p_pos), SCALE_MAIN,
                            [li], 'kv')
                        row[li] = js_nats(pb, q)
                    for li in range(NL):
                        L_store[li].append(row[li])
                    tagsL.append('P%d:%d' % (pi, p_pos))
                for p_pos in pos['content']:
                    row = np.zeros(NL)
                    for li in range(NL):
                        q, _ = prefill_logits_abl(
                            pr, int(p_pos), SCALE_MAIN,
                            [li], 'kv')
                        row[li] = js_nats(pb, q)
                    for li in range(NL):
                        C_store[li].append(row[li])
                    tagsC.append('P%d:%d' % (pi, p_pos))
                if pos['sham'] is not None:
                    row = np.zeros(NL)
                    for li in range(NL):
                        q, _ = prefill_logits_abl(
                            pr, int(pos['sham']),
                            SCALE_MAIN, [li], 'kv')
                        row[li] = js_nats(pb, q)
                    for li in range(NL):
                        S_store[li].append(row[li])
                    tagsS.append('P%d:%d'
                                 % (pi, pos['sham']))
                log('T2a P%d swept' % pi, lines)
            nL = len(L_store[0])
            nC = len(C_store[0])
            nS = len(S_store[0])
            L_mat = np.array(L_store) \
                if nL else np.zeros((NL, 0))
            C_mat = np.array(C_store) \
                if nC else np.zeros((NL, 0))
            S_mat = np.array(S_store) \
                if nS else np.zeros((NL, 0))
            D_l = np.zeros(NL)
            p_raw = np.ones(NL)
            gates_ok = bool(nL >= MIN_POS
                            and nC >= MIN_POS)
            if gates_ok:
                pool_all = np.concatenate(
                    [L_mat, C_mat], axis=1)
                n1 = nL
                for li in range(NL):
                    v = pool_all[li]
                    D_l[li] = float(np.median(v[:n1])
                                    - np.median(v[n1:]))
                # maxT: same permuted labels across
                # layers per draw
                rng3 = np.random.default_rng(
                    SEED_RND + 30)
                cnt_max = 0
                for _ in range(N_PERM):
                    pm = rng3.permutation(
                        pool_all.shape[1])
                    dmax = 0.0
                    for li in range(NL):
                        v = pool_all[li][pm]
                        d = np.median(v[:n1]) \
                            - np.median(v[n1:])
                        if abs(d) > dmax:
                            dmax = abs(d)
                    if dmax >= float(
                            np.abs(D_l).max()):
                        cnt_max += 1
                p_maxT = (cnt_max + 1) / (N_PERM + 1)
                for li in range(NL):
                    v = pool_all[li]
                    cnt = 0
                    for _ in range(N_PERM):
                        pm = rng3.permutation(v.size)
                        d = np.median(v[pm[:n1]]) \
                            - np.median(v[pm[n1:]])
                        if abs(d) >= abs(D_l[li]):
                            cnt += 1
                    p_raw[li] = (cnt + 1) \
                        / (N_PERM + 1)
            else:
                p_maxT = None
            l_star = int(np.argmax(np.abs(D_l))) \
                if gates_ok else None
            log('T2a done nL=%d nC=%d nS=%d l*=%s '
                'D_l*=%s p_maxT=%s'
                % (nL, nC, nS, l_star,
                   float(D_l[l_star])
                   if l_star is not None else None,
                   p_maxT), lines)
            top = np.argsort(-np.abs(D_l))[:6] \
                if gates_ok else []
            for li in top:
                log('T2a L%d D=%.5f p_raw=%.4f '
                    'medS=%.5f'
                    % (li, D_l[li], p_raw[li],
                       float(np.median(S_mat[li]))
                       if nS else None), lines)

            # ---------- T2b arm split at l* ----------
            T2b = None
            if l_star is not None:
                T2b = {}
                for arm in ('k', 'v'):
                    ja = {key: []
                          for key in ('L', 'C')}
                    for pi, pr in enumerate(
                            GEN_PROMPTS):
                        ent = sel['P%d' % pi]
                        if ent.get('skipped'):
                            continue
                        pos = ent['positions']
                        pb = base_probs[pi]
                        for p_pos in pos['logic']:
                            q, _ = prefill_logits_abl(
                                pr, int(p_pos),
                                SCALE_MAIN, [l_star],
                                arm)
                            ja['L'].append(
                                js_nats(pb, q))
                        for p_pos in pos['content']:
                            q, _ = prefill_logits_abl(
                                pr, int(p_pos),
                                SCALE_MAIN, [l_star],
                                arm)
                            ja['C'].append(
                                js_nats(pb, q))
                    va = np.array(ja['L'])
                    vc = np.array(ja['C'])
                    pool = np.concatenate([va, vc])
                    n1a = va.size
                    Da = float(np.median(va)
                               - np.median(vc))
                    rng4 = np.random.default_rng(
                        SEED_RND + 40)
                    cnt = 0
                    for _ in range(N_PERM):
                        pm = rng4.permutation(
                            pool.size)
                        d = np.median(
                            pool[pm[:n1a]]) \
                            - np.median(
                            pool[pm[n1a:]])
                        if abs(d) >= abs(Da):
                            cnt += 1
                    T2b[arm] = {
                        'D': round(Da, 6),
                        'p_raw': round(
                            (cnt + 1)
                            / (N_PERM + 1), 5),
                        'med_js_logic': round(
                            float(np.median(va)), 6),
                        'med_js_content': round(
                            float(np.median(vc)), 6)}
                    log('T2b arm=%s D=%.5f p=%.4f'
                        % (arm, Da, T2b[arm]['p_raw']),
                        lines)

            # ---------- T2c dose at l* ----------
            T2c = None
            if l_star is not None:
                T2c = {'per_scale': {}}
                for s in SCALE_DOSE:
                    jd = {key: []
                          for key in ('L', 'C', 'S')}
                    for pi, pr in enumerate(
                            GEN_PROMPTS):
                        ent = sel['P%d' % pi]
                        if ent.get('skipped'):
                            continue
                        pos = ent['positions']
                        pb = base_probs[pi]
                        for p_pos in pos['logic']:
                            q, _ = prefill_logits_abl(
                                pr, int(p_pos), s,
                                [l_star], 'kv')
                            jd['L'].append(
                                js_nats(pb, q))
                        for p_pos in pos['content']:
                            q, _ = prefill_logits_abl(
                                pr, int(p_pos), s,
                                [l_star], 'kv')
                            jd['C'].append(
                                js_nats(pb, q))
                        if pos['sham'] is not None:
                            q, _ = \
                                prefill_logits_abl(
                                    pr,
                                    int(pos['sham']),
                                    s, [l_star], 'kv')
                            jd['S'].append(
                                js_nats(pb, q))
                    va = np.array(jd['L'])
                    vc = np.array(jd['C'])
                    T2c['per_scale']['%.2f' % s] = {
                        'D': round(float(
                            np.median(va)
                            - np.median(vc)), 6),
                        'med_js_logic': round(
                            float(np.median(va)), 6),
                        'med_js_content': round(
                            float(np.median(vc)), 6),
                        'med_js_sham': round(
                            float(np.median(jd['S'])),
                            6) if jd['S'] else None}
                    log('T2c s=%.2f D=%.5f sham=%.5f'
                        % (s, T2c['per_scale']
                           ['%.2f' % s]['D'],
                           T2c['per_scale']
                           ['%.2f' % s]
                           ['med_js_sham']), lines)

            # ---------- verdict ----------
            if not gates_ok:
                verdict = 'js_layer_diffuse_qwen'
            else:
                Ds = float(D_l[l_star])
                if p_maxT < P_GATE and Ds > 0:
                    verdict = \
                        'js_layer_localized_qwen'
                elif p_maxT < P_GATE and Ds < 0:
                    verdict = \
                        'js_layer_content_localized_qwen'
                else:
                    verdict = \
                        'js_layer_diffuse_qwen'

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

            T2a = {
                'n_logic': nL, 'n_content': nC,
                'n_sham': nS, 'scale_main': SCALE_MAIN,
                'p_maxT': round(p_maxT, 5)
                if p_maxT is not None else None,
                'l_star': l_star,
                'D_l': [round(float(x), 6)
                        for x in D_l],
                'p_raw': [round(float(x), 5)
                          for x in p_raw],
                'med_js_logic': [round(float(
                    np.median(L_mat[li])), 6)
                    if nL else None
                    for li in range(NL)],
                'med_js_content': [round(float(
                    np.median(C_mat[li])), 6)
                    if nC else None
                    for li in range(NL)],
                'med_js_sham': [round(float(
                    np.median(S_mat[li])), 6)
                    if nS else None
                    for li in range(NL)],
                'min_pos_gate': MIN_POS}
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
        'a14_3010': a14_ok,
    }
    res = {
        'phase': 3011,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok
                              and a13_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T2a': T2a, 'T2b': T2b, 'T2c': T2c, 'T3': T3,
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
                                dtype=object),
            'L_mat': L_mat, 'C_mat': C_mat,
            'S_mat': S_mat,
            'tagsL': np.array(tagsL, dtype=object),
            'tagsC': np.array(tagsC, dtype=object),
            'tagsS': np.array(tagsS, dtype=object),
            'D_l': D_l, 'p_raw': p_raw}
    npz_path = os.path.join(
        OUT, 'omega_p2e_layer_js_localization_qwen.npz')
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
