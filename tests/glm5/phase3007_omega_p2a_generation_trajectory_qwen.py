# -*- coding: utf-8 -*-
"""Phase 3007: Omega-P2a generation trajectory recorder
(qwen).

Why: plan v5-P2 (battle two).  All 145 ledger cards are
single-forward anatomy or single-step injection; no
autoregressive trajectory is on record.  Before any
"hallucination bifurcation" naming, operationalize the
recorder: greedy KV-cache generation, per-token final-norm
coordinates in the u35/Vt8 basis (3002 readout machine,
bit-level anchors), word-class locking statistics (logic vs
function vs content positions), and a first perturb-recover
probe (P2b seed): mid-generation xdir injection at L17 vs
span(Vt8) orthogonal random control.

Design (qwen 3002 machine verbatim for anchors: 57 words
2887, dirs_word rebuild a1 vs 2927, Vt8 a3 vs 2939, u35
readout, a4/a5 bit-level vs 2935, xdir identity a7, bf16):
  T1 locking: 12 fixed prompts x K=64 greedy tokens; per
     generated token record final-norm input coords
     (s = u35 proj, c8 = Vt8 coords); classify each token
     logic/function/content/other via frozen single-token
     lists (registered below); per-class med |Delta s|
     between consecutive positions; locked := med_logic
     < 0.5 * med_content (gates: logic_n >= 20,
     content_n >= 40, else insufficient_lock_sample).
  T2 perturb-recover (P2b seed, prompts 0-3): at decode
     step t0=16 inject v = unit(mean xdir over 57 cells)
     at L17 attn_in pos -1, scale 2.0; control = random
     unit vec in span(Vt8) orthogonal to v (SEED_RND);
     D(t) = ||c8_pert(t) - c8_base(t)||; recovered per
     prompt := D_final(med over t>=t0+16) <= 0.5 *
     D_peak(max over t in [t0, t0+8]); recov := >=3/4
     prompts recovered; token divergence step recorded.
  T3 drift (descriptive): per prompt |s(t)-s_prefill|
     growth and late/early window std ratio.

Verdict (frozen):
  anchor fail                     => anchor_fail_all_void
  logic_n < 20 or content_n < 40  => insufficient_lock_
                                     sample
  locked AND recov                => logic_locked_perturb_
                                     recovering_qwen
  locked AND NOT recov            => logic_locked_perturb_
                                     divergent_qwen
  NOT locked                      => unlocked_drift_qwen

Anchors (frozen):
  a0 words == 2887 re-export (57)
  a1 dirs_word vs 2927 < 1e-5
  a2 baseline determinism < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 s_base[func] < 1e-4 (bit-level)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4 (bit-level)
  a6 sep_f > 0
  a7 xdir identity < 1e-9
  a8 null0 collision-free
  a9 3002 source integrity: result hash == seal AND
     verdict == context_entangled_qwen
  a10 generation determinism: prompt 0 run twice =>
      identical token ids AND coords rel diff < 1e-4

Tags: Omega-P2a / autoregressive / KV-cache / greedy decode
/ generation-state readout / locking statistics /
perturb-recover seed / descriptive drift / dimensionless
gates / no hallucination naming.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887',
                        'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927',
                        'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935',
                        'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939',
                        'rotation_target',
                        'rotation_target.npz')
D_3002 = os.path.join(BASE, 'phase3002',
                      'omega_g2_robustness_source_qwen')
OUT = os.path.join(BASE, 'phase3007',
                   'omega_p2a_generation_trajectory_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
S_IDX = (0, 1, 4)
SEED_NULL = 2896          # 3002 verbatim
SEED_RND = 3007
INJ_LAYER = 17
S_SCAN = 2.0
K_GEN = 64
T0_INJ = 16
N_INJ_PROMPTS = 4
LOCK_RATIO = 0.5
MIN_LOGIC = 20
MIN_CONTENT = 40
RECOV_NEEDED = 3          # of 4 prompts
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
    'mode': 'qwen3-4b only; 3002 readout machine verbatim '
            'for anchors (57 words 2887, dirs_word a1 vs '
            '2927, Vt8 a3 vs 2939, u35 readout, a4/a5 '
            'bit-level vs 2935, xdir identity a7, bf16); '
            'generation = greedy KV-cache decode, batch 1, '
            'prompt-end + per-token final-norm capture',
    'question': 'in generation state (not single-forward '
                'anatomy), do logic-word positions lock '
                'the language-axis readout (smaller step '
                'deltas than content positions), and does '
                'a mid-generation xdir perturbation '
                'recover or diverge?  First recorder for '
                'plan v5-P2/P2b.',
    'T1': '12 fixed prompts x K=64 greedy tokens; per '
          'generated token final-norm coords (s=u35 proj, '
          'c8=Vt8 coords); token class via frozen '
          'single-token lists LOGIC_WORDS/FUNC_WORDS + '
          'alpha-len>=3 content (registered in this file, '
          'single-token filter applied with dropped words '
          'logged); per-class med |Delta s| between '
          'consecutive positions; locked = med_logic < '
          '0.5*med_content; gates logic_n>=20, '
          'content_n>=40',
    'T2': 'perturb-recover seed on prompts 0-3: decode '
          'step t0=16 inject v=unit(mean xdir) at L17 '
          'attn_in pos -1 scale 2.0; control = random '
          'unit vec in span(Vt8) orth to v (SEED_RND); '
          'D(t)=||c8_pert-c8_base||; recovered := '
          'D_final(med t>=t0+16) <= 0.5*D_peak(max '
          '[t0,t0+8]); recov := >=3/4 prompts; token '
          'divergence step recorded; specificity ratio '
          'D_x/D_r descriptive',
    'T3': 'DESCRIPTIVE drift: per prompt |s(t)-s_prefill| '
          'at t=K-1, late/early window std ratio of s; no '
          'verdict branch',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'logic_n<20 or content_n<40 => '
               'insufficient_lock_sample; locked AND '
               'recov => logic_locked_perturb_recovering_'
               'qwen; locked AND NOT recov => '
               'logic_locked_perturb_divergent_qwen; NOT '
               'locked => unlocked_drift_qwen',
    'tags': 'Omega-P2a / autoregressive / KV-cache / '
            'greedy decode / generation-state readout / '
            'locking statistics / perturb-recover seed / '
            'descriptive drift / dimensionless gates / '
            'no hallucination naming',
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
        json.dump({'phase': 3007,
                   'name':
                       'omega_p2a_generation_trajectory_'
                       'qwen',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2887': sha8(SRC_2887),
                       's2927': sha8(SRC_2927),
                       's2935': sha8(SRC_2935),
                       's2939': sha8(SRC_2939),
                       's3002result': sha8(
                           D_3002 + r'\result.json'),
                       's3002seal': sha8(
                           D_3002 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'n_layers': NL, 'hidden': HID,
                   'vocab': VOCAB,
                   'k_gen': K_GEN, 't0_inj': T0_INJ,
                   'n_inj_prompts': N_INJ_PROMPTS,
                   'inj_layer': INJ_LAYER,
                   's_scan': S_SCAN,
                   'lock_ratio': LOCK_RATIO,
                   'min_logic': MIN_LOGIC,
                   'min_content': MIN_CONTENT,
                   'recov_needed': RECOV_NEEDED,
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'prompts': list(GEN_PROMPTS),
                   'logic_words': list(LOGIC_WORDS),
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    a0_ok = bool(n_words == 57)
    log('a0 words == 2887 re-export: %s (n=%d)'
        % (a0_ok, n_words), lines)

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]

    # a9 3002 source integrity
    seal02 = json.load(open(D_3002 + r'\seal.json',
                            encoding='utf-8'))
    a9_ok = bool(seal02['result_sha256_8']
                 == sha8(D_3002 + r'\result.json'))
    r02 = json.load(open(D_3002 + r'\result.json',
                         encoding='utf-8'))
    a9_ok = a9_ok and bool(
        r02['final_verdict'] == 'context_entangled_qwen'
        and r02['anchor_all_ok'] is True)
    log('a9 3002 integrity %s (verdict=%s)'
        % (a9_ok, r02['final_verdict']), lines)

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

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck), \
                'key mismatch %s' % w
    func_tid = tid('the')
    word_tids = set(tid_map.values())

    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n_words
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    # word-class single-token lists (dropped logged)
    logic_tids = {}
    dropped = []
    for w in LOGIC_WORDS:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) == 1:
            logic_tids[int(ids[0])] = w
        else:
            dropped.append(w)
    func_tids = {}
    for w in FUNC_WORDS:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) == 1:
            func_tids[int(ids[0])] = w
        else:
            dropped.append(w)
    log('class lists: logic=%d func=%d dropped=%s'
        % (len(logic_tids), len(func_tids), dropped),
        lines)

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

    cap = {'ai': {}, 'trk': {}}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None,
           'layer': None}
    inj_gen = {'on': False, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            ret = None
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
                if args:
                    ret = ((x,) + tuple(args[1:]),
                           kwargs)
                else:
                    nkw = dict(kwargs)
                    nkw['hidden_states'] = x
                    ret = (args, nkw)
            if li == INJ_LAYER and inj_gen['on']:
                x = x.clone()
                x[:, -1, :] = x[:, -1, :] \
                    + inj_gen['scale'] * inj_gen['vec']
                if args:
                    ret = ((x,) + tuple(args[1:]),
                           kwargs)
                else:
                    nkw = dict(kwargs)
                    nkw['hidden_states'] = x
                    ret = (args, nkw)
            if inj['coef'] is None:
                cap['ai'].setdefault(li, []).append(
                    x.detach().float().cpu().numpy()
                    .copy())
            return ret
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy().copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]

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
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        d_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(d_w[li])
                          for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word vs 2927 %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_word,
                             full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir identity %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S,
                                            axis=1)))

    # ---------- baselines (readout bit-level) ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T

    proj_f0, _ = reads(fin_f1)
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, _ = reads(fin_n0)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035])
                    .max())
    a5_ok = bool(a5_diff < 1e-4)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('a4 %.2e a5 %.2e ok=%s/%s; a6 sep_f=%.2f '
        '(null0 %.2f) ok=%s'
        % (a4_diff, a5_diff, a4_ok, a5_ok, sep_f,
           sep_n, a6_ok), lines)

    # ---------- generation machine ----------
    v_x = unit(xdir.mean(0))
    rngg = np.random.default_rng(SEED_RND)
    g = rngg.standard_normal(8)
    xhat = unit(Vt8 @ v_x)
    g = g - float(g @ xhat) * xhat
    g = unit(g)
    v_r = unit(g @ Vt8)
    log('gen vectors armed: v_x=unit(mean xdir) '
        'norm=%.4f; v_r=span(Vt8) orth rnd'
        % float(np.linalg.norm(xdir.mean(0))), lines)

    def gen_coords():
        return fin_cap['x'][0].astype(np.float64)

    def generate(prompt, k, vec=None, t0=None):
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
            s_pre = gen_coords() @ u35
            c8_pre = gen_coords() @ Vt8.T
            rec['s_pre'] = float(s_pre)
            rec['c8_pre'] = c8_pre
            nid = int(out.logits[0, -1].argmax())
            for t in range(k):
                clear_cap()
                if vec is not None and t == t0:
                    inj_gen['on'] = True
                    inj_gen['scale'] = S_SCAN
                    inj_gen['vec'] = torch.tensor(
                        vec, device='cuda',
                        dtype=torch.bfloat16)
                out = model(
                    input_ids=torch.tensor(
                        [[nid]], device='cuda'),
                    past_key_values=past,
                    use_cache=True)
                past = out.past_key_values
                if inj_gen['on']:
                    inj_gen['on'] = False
                x = gen_coords()
                rec['ids'].append(nid)
                rec['s'].append(float(x @ u35))
                rec['c8'].append(x @ Vt8.T)
                nid = int(out.logits[0, -1].argmax())
        state_fin['on'] = False
        rec['s'] = np.array(rec['s'])
        rec['c8'] = np.array(rec['c8'])
        rec['ids'] = np.array(rec['ids'])
        return rec

    def classify(t_id):
        if t_id in logic_tids:
            return 'logic'
        if t_id in func_tids:
            return 'func'
        txt = tok.decode([int(t_id)]).strip()
        if txt.isalpha() and len(txt) >= 3:
            return 'content'
        return 'other'

    # ---------- T1 baseline generation ----------
    anchor_prelim = bool(a0_ok and a1_ok and a2_ok
                         and a3_ok and a4_ok and a5_ok
                         and a6_ok and a7_ok and a8_ok
                         and a9_ok)
    recs = {}
    dS_cls = {'logic': [], 'func': [], 'content': [],
              'other': []}
    drift = []
    if anchor_prelim:
        for pi, pr in enumerate(GEN_PROMPTS):
            recs[pi] = generate(pr, K_GEN)
        # a10 generation determinism (prompt 0 twice)
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

        # per-class locking
        for pi in range(len(GEN_PROMPTS)):
            rec = recs[pi]
            s_prev = rec['s_pre']
            for t in range(K_GEN):
                cls = classify(int(rec['ids'][t]))
                dS_cls[cls].append(
                    abs(float(rec['s'][t] - s_prev)))
                s_prev = float(rec['s'][t])
            drift.append(abs(float(rec['s'][K_GEN - 1]
                                    - rec['s_pre'])))
        med = {k: (float(np.median(v)) if v else None)
               for k, v in dS_cls.items()}
        logic_n = len(dS_cls['logic'])
        content_n = len(dS_cls['content'])
        locked = None
        lock_ratio_obs = None
        if med['logic'] is not None \
                and med['content'] is not None \
                and med['content'] > 0:
            lock_ratio_obs = med['logic'] \
                / med['content']
            locked = bool(lock_ratio_obs < LOCK_RATIO)
        T1 = {'n': {'logic': logic_n,
                    'func': len(dS_cls['func']),
                    'content': content_n,
                    'other': len(dS_cls['other'])},
              'med_dS': {k: (round(v, 4) if v is not None
                             else None)
                         for k, v in med.items()},
              'lock_ratio_obs': (round(lock_ratio_obs, 4)
                                 if lock_ratio_obs
                                 is not None else None),
              'lock_ratio_gate': LOCK_RATIO,
              'locked': locked,
              'dropped_class_words': dropped}
        log('T1 locking: med_dS logic=%s func=%s '
            'content=%s ratio=%s locked=%s (n %d/%d)'
            % (med['logic'], med['func'],
               med['content'], lock_ratio_obs, locked,
               logic_n, content_n), lines)

        # ---------- T2 perturb-recover ----------
        t2 = {}
        recov_count = 0
        for pi in range(N_INJ_PROMPTS):
            base = recs[pi]
            pr = {}
            for tag, vec in [('xdir', v_x),
                             ('ctrl', v_r)]:
                rp = generate(GEN_PROMPTS[pi], K_GEN,
                              vec=vec, t0=T0_INJ)
                D = np.linalg.norm(rp['c8']
                                   - base['c8'],
                                   axis=1)
                peak = float(
                    D[T0_INJ:T0_INJ + 9].max())
                fin_m = float(
                    np.median(D[T0_INJ + 16:]))
                first_div = None
                for t in range(T0_INJ, K_GEN):
                    if int(rp['ids'][t]) \
                            != int(base['ids'][t]):
                        first_div = t
                        break
                ok = bool(fin_m <= 0.5 * peak)
                if tag == 'xdir' and ok:
                    recov_count += 1
                pr[tag] = {
                    'D_peak': round(peak, 4),
                    'D_final': round(fin_m, 4),
                    'recovered': ok,
                    'first_token_div': first_div,
                    'n_token_div': int(np.sum(
                        rp['ids'] != base['ids']))}
            t2['P%d' % pi] = pr
            log('T2 P%d xdir peak=%.3f fin=%.3f rec=%s '
                'div@%s (n=%d); ctrl peak=%.3f '
                'fin=%.3f rec=%s'
                % (pi, pr['xdir']['D_peak'],
                   pr['xdir']['D_final'],
                   pr['xdir']['recovered'],
                   pr['xdir']['first_token_div'],
                   pr['xdir']['n_token_div'],
                   pr['ctrl']['D_peak'],
                   pr['ctrl']['D_final'],
                   pr['ctrl']['recovered']), lines)
        recov = bool(recov_count >= RECOV_NEEDED)
        T2 = {'per_prompt': t2,
              'recov_count': recov_count,
              'recov_needed': RECOV_NEEDED,
              'recov': recov,
              't0': T0_INJ, 'scale': S_SCAN,
              'layer': INJ_LAYER}
        log('T2 recov %d/%d => %s'
            % (recov_count, N_INJ_PROMPTS, recov),
            lines)

        # ---------- T3 drift (descriptive) ----------
        late = []
        early = []
        for pi in range(len(GEN_PROMPTS)):
            s = recs[pi]['s']
            early.append(float(np.std(s[:16])))
            late.append(float(np.std(s[48:])))
        T3 = {'med_drift_end': round(
                  float(np.median(drift)), 4),
              'med_std_early': round(
                  float(np.median(early)), 4),
              'med_std_late': round(
                  float(np.median(late)), 4),
              'note': 'descriptive; no verdict branch'}
        log('T3 drift end med=%.3f std early=%.3f '
            'late=%.3f'
            % (T3['med_drift_end'],
               T3['med_std_early'],
               T3['med_std_late']), lines)

        if not a10_ok:
            verdict = 'anchor_fail_all_void'
        elif logic_n < MIN_LOGIC \
                or content_n < MIN_CONTENT:
            verdict = 'insufficient_lock_sample'
        elif locked and recov:
            verdict = \
                'logic_locked_perturb_recovering_qwen'
        elif locked:
            verdict = \
                'logic_locked_perturb_divergent_qwen'
        else:
            verdict = 'unlocked_drift_qwen'
    else:
        a10_ok = False
        a10_rel = None
        T1 = T2 = T3 = None
        logic_n = content_n = 0
        recov_count = 0
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words': a0_ok, 'a8_collision': a8_ok,
        'a1_diff': a1_diff, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a4_diff': a4_diff,
        'a5_diff': a5_diff, 'a6_sep_f': sep_f,
        'a7_diff': a7_diff, 'a9_ok': a9_ok,
        'a10_gen_det': {'ids_same_and_rel': a10_rel,
                        'ok': a10_ok},
    }
    res = {
        'phase': 3007,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim and a10_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T1': T1, 'T2': T2, 'T3': T3,
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: crashed at first decode step - '
            'fin_cap stores (1,hid) batch-view, gen '
            'coords needed [0] squeeze; no verdict '
            'computed; run2: authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save = {'dirs_word': dirs_word, 'Vt8': Vt8,
            'u35': u35, 'xdir': xdir, 'v_x': v_x,
            'v_r': v_r,
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'null0_tids': np.array(null0_tids),
            'prompts': np.array(GEN_PROMPTS,
                                dtype=object)}
    for pi, rec in recs.items():
        save['ids_%d' % pi] = rec['ids']
        save['s_%d' % pi] = rec['s']
        save['c8_%d' % pi] = rec['c8']
    npz_path = os.path.join(
        OUT, 'omega_p2a_generation_trajectory_qwen.npz')
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
