# -*- coding: utf-8 -*-
"""Phase 3008: Omega-P2b generation-state logic signature +
logic-position KV causal probe (qwen).

Why: 3007 recorded the first autoregressive trajectory and
showed u35-axis step-locking (logic 0.457x content).  Two
upgrades now preregistered:  (i) port the 2993 logic
signature machinery (w2/w1024 axes built from L34 residual,
split-half L_A/L_B, length-robust in static prompts) into
GENERATION state - does the held-out logic axis separate
logic from content tokens while the model is generating?
(ii) first causal probe on record: ablate the KV entry of a
logic-word position in the prompt vs a matched content
position - are logic positions more load-bearing for the
continuation?

Design (3007 machine verbatim for anchors and generation;
2993 axes loaded bit-level):
  prompts  3007 GEN_PROMPTS verbatim (12), K=256 greedy
           KV-cache decode, batch 1.
  capture  per decode step: L34 decoder-layer input
           residual (2993 res34 caliber, last position),
           projected on w2/w1024 (2993 npz, identity
           re-verified a1) plus final-norm s/c8 (3007).
  classes  logic := single-token tids of LOGIC_WORDS(3007)
           UNION l_words(2993, 23 words, single-token
           recheck a8); func := FUNC_WORDS(3007); content
           := decoded alpha len>=3 not in logic/func;
           other := rest (3007 classify verbatim).
  T1 PRIMARY generation-state signature: per axis w2/w1024,
     signed projection values of logic vs content generated
     tokens; D_obs = med_L - med_C; permutation N_PERM=10000
     two-sided on pooled values; p_fam = p_raw x 2
     (Bonferroni, family declared = 2 axes); sig := p_fam <
     0.05 AND logic_n >= 30 AND content_n >= 100 (margin
     discipline); |dw| step-locking medians descriptive.
  T2 PRIMARY KV causal: per prompt scan prompt ids for
     logic positions (up to 2, in order), 2 content
     positions (rng sample), 1 sham (func/other pool);
     ablation := zero K and V at that position across all
     36 layers after prefill (transformers 5.x DynamicCache
     layers[i].keys/.values), decode 32 steps; div := token
     diffs vs baseline; D_c8 median over t>=16 descriptive;
     stat: med div_L - med div_C over pooled positions,
     permutation two-sided N_PERM=10000, p_fam = p_raw
     (family 1); kv_dominant := D>0 AND p_fam<0.05 AND
     logic_pos_n >= 8 AND content_pos_n >= 8.
  T3 DESCRIPTIVE: 256-token drift (|s end - s_pre|, late/
     early window std) vs 3007 64-token baseline; no gate.

Verdict (frozen):
  anchor fail                        => anchor_fail_all_void
  T1 insufficient AND T2 insufficient=> insufficient_sample_
                                        both
  sig_gen AND kv_dominant            => logic_sig_gen_kv_
                                        causal_qwen
  sig_gen AND kv_reverse             => logic_sig_gen_kv_
                                        reverse_qwen
  sig_gen only                       => logic_sig_gen_only_
                                        qwen
  kv_dominant only                   => kv_causal_only_qwen
  else                               => logic_gen_null_qwen

Anchors (frozen):
  a0  2993 integrity: seal result hash match AND verdict ==
      logic_signature_length_robust AND anchors ok
  a1  w2/w1024 recomputed from 2993 npz res34 (float32
      stored) vs stored axes < 1e-6
  a2  dirs_word rebuild vs 2927 < 1e-5
  a3  Vt8 vs 2939 < 1e-6
  a4  proj_func vs 2935 < 1e-4 (bit-level)
  a5  proj_null0 vs 2935 < 1e-4 (bit-level)
  a6  baseline determinism < 1e-4
  a7  xdir identity < 1e-9
  a8  l_words(2993) all single-token here
  a9  3007 integrity: seal match AND verdict ==
      logic_locked_perturb_divergent_qwen
  a10 generation determinism: prompt 0 twice => identical
      ids AND c8 rel < 1e-4

Tags: Omega-P2b / generation-state signature port / KV
ablation causal / split-half held-out axis / no hallucination
naming / descriptive drift.
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
OUT = os.path.join(BASE, 'phase3008',
                   'omega_p2b_logic_kv_causal_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
L_SIG = 34
SEED_NULL = 2896          # 3002/3007 verbatim
SEED_RND = 3008
K_GEN = 256
K_CA = 32
N_PERM = 10000
P_GATE = 0.05
MIN_LOGIC = 30
MIN_CONTENT = 100
MIN_POS = 8
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
    'mode': 'qwen3-4b; 3007 generation machine verbatim; '
            '2993 axes loaded bit-level (a1); res34 = L34 '
            'decoder-layer input residual, last position, '
            'per decode step',
    'question': 'does the 2993 held-out logic axis '
                'separate logic from content tokens in '
                'GENERATION state, and are logic positions '
                'causally load-bearing (KV ablation) vs '
                'matched content positions?',
    'T1': 'PRIMARY: per axis w2/w1024 signed proj of logic '
          'vs content generated tokens; D=med_L-med_C; '
          'perm N=10000 two-sided; p_fam=p_raw x 2; sig '
          ':= p_fam<0.05 AND logic_n>=30 AND content_n>=100',
    'T2': 'PRIMARY: KV zero at position p (all 36 layers) '
          'after prefill, decode 32; logic pos (<=2, in '
          'order) vs 2 rng content pos vs 1 sham; div = '
          'token diffs; D=med_L-med_C pooled positions; '
          'perm two-sided; kv_dominant := D>0 AND '
          'p_fam<0.05 AND nL>=8 AND nC>=8',
    'T3': 'DESCRIPTIVE 256-token drift vs 3007 64-token',
    'verdict': 'anchor fail => anchor_fail_all_void; T1 '
               'insuff AND T2 insuff => '
               'insufficient_sample_both; sig AND dominant '
               '=> logic_sig_gen_kv_causal_qwen; sig AND '
               'reverse => logic_sig_gen_kv_reverse_qwen; '
               'sig only => logic_sig_gen_only_qwen; '
               'dominant only => kv_causal_only_qwen; else '
               '=> logic_gen_null_qwen',
    'tags': 'Omega-P2b / generation-state signature port / '
            'KV ablation causal / split-half held-out '
            'axis / no hallucination naming',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


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
        json.dump({'phase': 3008,
                   'name': 'omega_p2b_logic_kv_causal_qwen',
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
                           D_3007 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'k_gen': K_GEN, 'k_ca': K_CA,
                   'n_perm': N_PERM, 'p_gate': P_GATE,
                   'min_logic': MIN_LOGIC,
                   'min_content': MIN_CONTENT,
                   'min_pos': MIN_POS,
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'prompts': list(GEN_PROMPTS),
                   'logic_words': list(LOGIC_WORDS),
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- 2993 / 3007 source integrity ----------
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

    # a7 xdir identity (3007 verbatim, S_IDX 0/1/4)
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

    def kv_zero(past, p):
        for L in past.layers:
            L.keys[:, :, p, :] = 0
            L.values[:, :, p, :] = 0

    def generate(prompt, k, abl_pos=None):
        ids = tok(prompt, add_special_tokens=False)[
            'input_ids']
        clear_cap()
        state_fin['on'] = True
        state_res['on'] = True
        rec = {'ids': [], 's': [], 'c8': [],
               'w2': [], 'w1024': []}
        with torch.no_grad():
            out = model(torch.tensor([ids],
                                     device='cuda'),
                        use_cache=True)
            past = out.past_key_values
            if abl_pos is not None:
                kv_zero(past, abl_pos)
            x0 = gen_coords()
            r0 = res_cap['x'].astype(np.float64) \
                .reshape(-1)
            rec['s_pre'] = float(x0 @ u35)
            rec['c8_pre'] = x0 @ Vt8.T
            rec['w2_pre'] = float(r0 @ w2_93)
            rec['w_pre'] = float(r0 @ w1024_93)
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
                r = res_cap['x'].astype(np.float64) \
                    .reshape(-1)
                rec['ids'].append(nid)
                rec['s'].append(float(x @ u35))
                rec['c8'].append(x @ Vt8.T)
                rec['w2'].append(float(r @ w2_93))
                rec['w1024'].append(
                    float(r @ w1024_93))
                nid = int(out.logits[0, -1].argmax())
        state_fin['on'] = False
        state_res['on'] = False
        for key in ('ids', 's', 'c8', 'w2', 'w1024'):
            rec[key] = np.array(rec[key])
        rec['prompt_ids'] = np.array(ids)
        return rec

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

    # ---------- T1 baseline generation ----------
    anchor_prelim = bool(a0_ok and a1_ok and a2_ok
                         and a3_ok and a4_ok and a5_ok
                         and a6_ok and a7_ok and a8_ok
                         and a9_ok)
    recs = {}
    verdict = None
    T1 = T2 = T3 = None
    sig_gen = None
    kv_dom = None
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

        if not a10_ok:
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1 ----------
            rng1 = np.random.default_rng(
                SEED_RND + 10)
            proj = {'w2': {}, 'w1024': {}}
            cls_seq = {}
            dw_cls = {'logic': [], 'content': []}
            for key in ('w2', 'w1024'):
                proj[key] = {'logic': [], 'content': []}
            for pi in range(len(GEN_PROMPTS)):
                rec = recs[pi]
                w_prev = rec['w_pre']
                for t in range(K_GEN):
                    c = classify(int(rec['ids'][t]))
                    cls_seq.setdefault(c, 0)
                    cls_seq[c] += 1
                    if c in ('logic', 'content'):
                        proj['w2'][c].append(
                            rec['w2'][t])
                        proj['w1024'][c].append(
                            rec['w1024'][t])
                        dw_cls[c].append(
                            abs(rec['w1024'][t]
                                - w_prev))
                    w_prev = rec['w1024'][t]
            T1 = {'axes': {}, 'n': dict(cls_seq),
                  'dw_med': {k: round(float(
                      np.median(v)), 4)
                      for k, v in dw_cls.items()}}
            sig_any = False
            for key in ('w2', 'w1024'):
                vL = np.array(proj[key]['logic'])
                vC = np.array(proj[key]['content'])
                ent = {'n_logic': int(vL.size),
                       'n_content': int(vC.size),
                       'med_L': round(float(
                           np.median(vL)), 4),
                       'med_C': round(float(
                           np.median(vC)), 4)}
                ok_n = bool(vL.size >= MIN_LOGIC
                            and vC.size >= MIN_CONTENT)
                ent['n_gate_ok'] = ok_n
                if ok_n:
                    pool = np.concatenate([vL, vC])
                    n1 = vL.size
                    D = float(np.median(vL)
                              - np.median(vC))
                    cnt = 0
                    for _ in range(N_PERM):
                        pm = rng1.permutation(
                            pool.size)
                        d = np.median(
                            pool[pm[:n1]]) \
                            - np.median(
                            pool[pm[n1:]])
                        if abs(d) >= abs(D):
                            cnt += 1
                    p_raw = (cnt + 1) / (N_PERM + 1)
                    p_fam = min(1.0, p_raw * 2)
                    ent['D'] = round(D, 4)
                    ent['p_raw'] = round(p_raw, 5)
                    ent['p_fam'] = round(p_fam, 5)
                    ent['sig'] = bool(
                        p_fam < P_GATE)
                    if ent['sig']:
                        sig_any = True
                T1['axes'][key] = ent
                log('T1 %s: nL=%d nC=%d med %.3f/%.3f '
                    'p_fam=%s sig=%s'
                    % (key, ent['n_logic'],
                       ent['n_content'],
                       ent['med_L'], ent['med_C'],
                       ent.get('p_fam'),
                       ent.get('sig')), lines)
            T1['sig_gen'] = sig_any
            gate_any = bool(
                T1['axes']['w2'].get('n_gate_ok')
                or T1['axes']['w1024']
                .get('n_gate_ok'))
            sig_gen = bool(sig_any) if gate_any \
                else None
            log('T1 sig_gen=%s (gate_any=%s)'
                % (sig_gen, gate_any), lines)

            # ---------- T2 KV causal ----------
            rng2 = np.random.default_rng(
                SEED_RND + 20)
            t2 = {}
            pos_L = []
            pos_C = []
            pos_S = []
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
                    t2['P%d' % pi] = ent
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
                base = recs[pi]
                ent['positions'] = {
                    'logic': lp_use,
                    'content': [int(x) for x
                                in cp_use],
                    'sham': sham}
                for tag, plist in (
                        ('logic', lp_use),
                        ('content', cp_use)):
                    divs = []
                    dc8 = []
                    for p in plist:
                        rp = generate(pr, K_CA,
                                      abl_pos=int(p))
                        divs.append(int(np.sum(
                            rp['ids']
                            != base['ids'][:K_CA])))
                        D = np.linalg.norm(
                            rp['c8']
                            - base['c8'][:K_CA],
                            axis=1)
                        dc8.append(round(float(
                            np.median(D[16:])), 4))
                    ent[tag] = {'div': divs,
                                'd_c8_med': dc8}
                if sham is not None:
                    rp = generate(pr, K_CA,
                                  abl_pos=sham)
                    ent['sham'] = {
                        'div': int(np.sum(
                            rp['ids']
                            != base['ids'][:K_CA]))}
                for p in lp_use:
                    pos_L.append(ent['logic']['div'][
                        lp_use.index(p)])
                for x_ in ent['content']['div']:
                    pos_C.append(x_)
                t2['P%d' % pi] = ent
                log('T2 P%d logic div=%s content '
                    'div=%s sham=%s'
                    % (pi, ent['logic']['div'],
                       ent['content']['div'],
                       ent.get('sham')), lines)
            nL, nC = len(pos_L), len(pos_C)
            T2 = {'per_prompt': t2,
                  'n_logic_pos': nL,
                  'n_content_pos': nC,
                  'min_pos_gate': MIN_POS}
            if nL >= MIN_POS and nC >= MIN_POS:
                vL = np.array(pos_L, dtype=float)
                vC = np.array(pos_C, dtype=float)
                pool = np.concatenate([vL, vC])
                n1 = vL.size
                D = float(np.median(vL)
                          - np.median(vC))
                cnt = 0
                for _ in range(N_PERM):
                    pm = rng2.permutation(pool.size)
                    d = np.median(pool[pm[:n1]]) \
                        - np.median(pool[pm[n1:]])
                    if abs(d) >= abs(D):
                        cnt += 1
                p_fam = (cnt + 1) / (N_PERM + 1)
                T2['D'] = round(D, 4)
                T2['med_L'] = round(
                    float(np.median(vL)), 4)
                T2['med_C'] = round(
                    float(np.median(vC)), 4)
                T2['p_fam'] = round(p_fam, 5)
                kv_dom = bool(D > 0
                              and p_fam < P_GATE)
                T2['kv_dominant'] = kv_dom
                T2['kv_reverse'] = bool(
                    D < 0 and p_fam < P_GATE)
                log('T2 pooled nL=%d nC=%d med '
                    '%.2f/%.2f D=%.2f p_fam=%.5f '
                    'dominant=%s'
                    % (nL, nC, T2['med_L'],
                       T2['med_C'], D, p_fam,
                       kv_dom), lines)
            else:
                T2['insufficient'] = True
                log('T2 insufficient pos nL=%d nC=%d'
                    % (nL, nC), lines)

            # ---------- T3 drift ----------
            drift_s = []
            drift_w = []
            early = []
            late = []
            for pi in range(len(GEN_PROMPTS)):
                rec = recs[pi]
                drift_s.append(abs(float(
                    rec['s'][K_GEN - 1]
                    - rec['s_pre'])))
                drift_w.append(abs(float(
                    rec['w1024'][K_GEN - 1]
                    - rec['w_pre'])))
                early.append(float(
                    np.std(rec['s'][:32])))
                late.append(float(
                    np.std(rec['s'][224:])))
            T3 = {
                'med_drift_s_end': round(float(
                    np.median(drift_s)), 4),
                'med_drift_w_end': round(float(
                    np.median(drift_w)), 4),
                'med_std_early': round(float(
                    np.median(early)), 4),
                'med_std_late': round(float(
                    np.median(late)), 4),
                'note': 'descriptive vs 3007 64-token '
                        '(med_drift_end 45.97, std '
                        '25.51/31.58)'}
            log('T3 drift s=%.3f w=%.3f std '
                'e=%.3f l=%.3f'
                % (T3['med_drift_s_end'],
                   T3['med_drift_w_end'],
                   T3['med_std_early'],
                   T3['med_std_late']), lines)

            # ---------- verdict ----------
            kv_ins = T2.get('insufficient', False)
            if sig_gen is None and kv_ins:
                verdict = 'insufficient_sample_both'
            elif sig_gen and kv_dom:
                verdict = 'logic_sig_gen_kv_causal_qwen'
            elif sig_gen \
                    and T2.get('kv_reverse', False):
                verdict = \
                    'logic_sig_gen_kv_reverse_qwen'
            elif sig_gen:
                verdict = 'logic_sig_gen_only_qwen'
            elif kv_dom:
                verdict = 'kv_causal_only_qwen'
            else:
                verdict = 'logic_gen_null_qwen'
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
    }
    res = {
        'phase': 3008,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2)},
        'T1': T1, 'T2': T2, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: anchor_fail_all_void - a1 recomputed '
            'w2/w1024 with unit() but 2993 stored the '
            'UNNORMALIZED mean-difference vectors (unit is '
            'applied only inside 2993 T1); recomputation '
            'fixed to raw mean-difference with relative '
            'gate (float32 res34 rounding), no other '
            'changes; run2: authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35, 'w2': w2_93, 'w1024': w1024_93,
            'l_words': np.array(l_words, dtype=object),
            'l_A': np.array(l_A, dtype=object),
            'l_B': np.array(l_B, dtype=object),
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object)}
    for pi, rec in recs.items():
        save['ids_%d' % pi] = rec['ids']
        save['s_%d' % pi] = rec['s']
        save['w2_%d' % pi] = rec['w2']
        save['w_%d' % pi] = rec['w1024']
    npz_path = os.path.join(
        OUT, 'omega_p2b_logic_kv_causal_qwen.npz')
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
