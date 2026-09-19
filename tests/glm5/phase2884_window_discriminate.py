# -*- coding: utf-8 -*-
"""Phase 2884: window-length discrimination for the DS7B carrier deviation.

2883 found: mlp carrier replicates on DS7B qualitatively (acc 0.575,
3.5x null) but deviates from qwen3-4b (0.875, delta=-0.30).  Candidate
causes (2883, post-hoc): (i) window length 8L vs 10L, (ii) model factor
(reasoning-distill training / untied lm_head geometry), (iii) window
position.  This phase discriminates them.

Key efficiency fact: g_direct is defined per LAYER, windows are slices.
So each model is measured ONCE on ALL layers (hooks already capture
every layer), and every window arm is a slice of that matrix.  One
forward per word per model total.

Arms (frozen):
  qwen3-4b (L=36):
    q_base10  [26,36)  10L  baseline (must reproduce 0.875, C0)
    q_early8  [26,34)   8L  drop last 2 layers
    q_late8   [28,36)   8L  drop first 2 layers
    q_end34   [24,34)  10L  same length, shifted end
  DS7B (L=28):
    d_base8   [20,28)   8L  baseline recompute (must reproduce 0.575, C0)
    d_long10  [18,28)  10L  lengthened window

Prereg (frozen before any forward):
  D1  one full-layer measurement per model; windows = slices; same
      vocab/context/E-row/hook protocol verbatim from 2883.
  D2  window set exactly as listed above; no other windows reported
      as gated (extra slices may be reported descriptively only).
  W3  rebuild consistency per model: v1=max|LN2(ln2in)-mlpin|<0.05 and
      v2=max|mlp(ln2in)-mlpout|<0.05; violation voids all verdicts.
  C0  determinism: acc(q_base10) == 0.875 exactly AND
      acc(d_base8) == 0.575 exactly (same GPU/code path, bf16
      deterministic for identical shapes).  Failure voids everything.
  C1  qwen length effect: max(|acc(q_early8)-0.875|,
      |acc(q_late8)-0.875|) >= 0.15
  C2  ds7b length effect: acc(d_long10) - 0.575 >= +0.15
  C3  position effect: |acc(q_early8) - acc(q_late8)| >= 0.15
  attribution (frozen precedence):
      window_length   iff C1 and C2
      model_factor    iff not C1 and not C2 and not C3
      position_matters iff C3 and not (C1 and C2)
      mixed           otherwise
  Nulls: 200 label permutations per window, SEED=2884; each acc also
      reported against its own null p95 (signal sanity, not gated).
Output: result/rdc_query_construction_20260913/phase2884/
        window_discriminate/{execution.json, result.json,
        window_discriminate.npz}
"""
import hashlib
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2884', 'window_discriminate')
SRC_2806_EXEC = os.path.join(BASE, 'phase2806', 'qwen4_hierarchy',
                             'execution.json')
SEED = 2884
EPS = 1.0
MAX_WORDS = 8
REF_Q = 0.875      # 2867 class B3 acc, qwen3-4b [26,36)
REF_D = 0.575      # 2883 class B3 acc, DS7B [20,28)
BAND = 0.15        # frozen effect-size threshold

MODELS = {
    'qwen3-4b': {
        'key': 'qwen4',
        'mdir': r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b',
        'windows': {'q_base10': (26, 36), 'q_early8': (26, 34),
                    'q_late8': (28, 36), 'q_end34': (24, 34)},
        'ref': {'q_base10': REF_Q},
    },
    'deepseek-r1-distill-qwen-7b': {
        'key': 'ds7',
        'mdir': (r'D:\AI2050\Ai2050-OpenOne\models\hf'
                 r'\deepseek-r1-distill-qwen-7b'),
        'windows': {'d_base8': (20, 28), 'd_long10': (18, 28)},
        'ref': {'d_base8': REF_D},
    },
}

T0 = time.monotonic()
LOG = []


def log(s):
    LOG.append(str(s))
    print(s, flush=True)


def unit(x):
    return x / max(float(np.linalg.norm(x)), 1e-30)


def loo_nn_acc(C, labels):
    n = len(labels)
    C = C.copy()
    np.fill_diagonal(C, -2.0)
    a = 0
    for i in range(n):
        j = int(np.argmax(C[i]))
        a += int(labels[j] == labels[i])
    return a / n


def sha8(p):
    return hashlib.sha256(io.open(p, 'rb').read()).hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)

PREREG = {
    'D1': 'one full-layer g_direct measurement per model; windows are '
          'slices; 2883 vocab/context/E-row protocol verbatim',
    'D2': 'window set frozen: qwen {[26,36),[26,34),[28,36),[24,34)}; '
          'ds7b {[20,28),[18,28)}',
    'W3': 'per model v1=max|LN2(ln2in)-mlpin|<0.05 and v2=max|mlp(ln2in)'
          '-mlpout|<0.05 else void',
    'C0': 'acc(q_base10)==0.875 exactly and acc(d_base8)==0.575 exactly '
          'else void',
    'C1': 'qwen length effect: max(|acc(q_early8)-0.875|,'
          '|acc(q_late8)-0.875|) >= 0.15',
    'C2': 'ds7b length effect: acc(d_long10)-0.575 >= +0.15',
    'C3': 'position effect: |acc(q_early8)-acc(q_late8)| >= 0.15',
    'attribution': 'window_length iff C1 and C2; model_factor iff not '
                   'C1 and not C2 and not C3; position_matters iff C3 '
                   'and not (C1 and C2); else mixed',
    'nulls': '200 label permutations per window, SEED=2884',
}
exec_doc = {
    'phase': 2884,
    'name': 'window_discriminate',
    'seed': SEED,
    'models': {m: v['key'] for m, v in MODELS.items()},
    'windows': {m: v['windows'] for m, v in MODELS.items()},
    'prereg': PREREG,
    'frozen_at_s': round(T0, 1),
}
with io.open(os.path.join(OUT, 'execution.json'), 'w',
             encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=2, ensure_ascii=False)
log('prereg frozen')

import torch
from transformers import AutoTokenizer
from safetensors import safe_open
from phase2662_symmetric_mapping_contract import load_native

exec2806 = json.load(io.open(SRC_2806_EXEC, encoding='utf-8'))
CATS = exec2806['cats']
CAT_WORDS = list(CATS.keys())

store = {}
voided = False

for mname, mcfg in MODELS.items():
    log('---- model %s ----' % mname)
    tok = AutoTokenizer.from_pretrained(
        mcfg['mdir'], local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t, _tok=tok, _tc=tc):
        if t not in _tc:
            ids = _tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = _tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            _tc[t] = int(ids[0])
        return _tc[t]

    single_tok = []
    for w in [x for v in CATS.values() for x in v]:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    targets = {c: [w for w in CATS[c] if w in single_tok][:MAX_WORDS]
               for c in CAT_WORDS}
    n_words = sum(len(targets[c]) for c in CAT_WORDS)
    target_list = [(c, w) for c in CAT_WORDS for w in targets[c]]
    labels = np.array([CAT_WORDS.index(c) for c, _ in target_list])
    log('%s: %d words (classes_ge5=%d)'
        % (mname, n_words,
           sum(1 for c in CAT_WORDS if len(targets[c]) >= 5)))

    model, _ = load_native(mcfg['key'])
    model.eval()

    # E rows: safetensors direct read (2883 lesson: lm_head may be
    # offloaded to meta by accelerate)
    idx = json.load(io.open(os.path.join(mcfg['mdir'],
                                         'model.safetensors.index.json'),
                            encoding='utf-8'))
    wmap = idx['weight_map']
    # tied models (qwen3-4b) have no lm_head row in the checkpoint;
    # embed_tokens is then the unembed by definition (tie semantics)
    _key = None
    for k in ('lm_head.weight', 'model.lm_head.weight',
              'model.embed_tokens.weight'):
        if k in wmap:
            _key = k
            break
    assert _key is not None, 'no unembed row in %s' % mcfg['mdir']
    with safe_open(os.path.join(mcfg['mdir'], wmap[_key]),
                   framework='pt') as sf:
        W_U = sf.get_tensor(_key).float().numpy()
    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    del W_U
    cents = []
    for c in CAT_WORDS:
        ws = [w for w in CATS[c] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])

    rng = np.random.default_rng(SEED)  # reserved for label permutations

    cap = {'mlp': {}, 'ln2in': {}, 'mlpin': {}}

    def make_out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o[0].detach().float().cpu().numpy())
        return hook

    def make_in_hook(kind, li):
        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            cap[kind].setdefault(li, []).append(
                x.detach()[0].float().cpu().numpy())
        return pre_hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))
        handles.append(layer.post_attention_layernorm
                       .register_forward_pre_hook(
                           make_in_hook('ln2in', li), with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(
            make_in_hook('mlpin', li), with_kwargs=True))

    def clear_cap():
        for d in cap:
            for li in cap[d]:
                del cap[d][li][:]

    layers = model.model.layers
    NL = len(layers)
    vdt = next(layers[0].mlp.parameters()).dtype

    def mlp_batch(li, X):
        t = torch.tensor(X[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            o = layers[li].mlp(t)
        return o[0].detach().float().cpu().numpy().astype(np.float64)

    def ln2_call(li, x):
        t = torch.tensor(x[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            o = layers[li].post_attention_layernorm(t)
        return o[0].detach().float().cpu().numpy().astype(np.float64)

    g_direct = np.zeros((n_words, NL))
    g_comp = np.zeros((n_words, NL))
    v1_all = []
    v2_all = []

    for i, (cat, w) in enumerate(target_list):
        ci = CAT_WORDS.index(cat)
        cdir = dW_unit[ci]
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        toks = [tid(same_cat[0]), w_tid]
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        ln2in_b = np.stack([cap['ln2in'][li][0][1] for li in range(NL)])
        mlpin_b = np.stack([cap['mlpin'][li][0][1] for li in range(NL)])
        mlpout_b = np.stack([cap['mlp'][li][0][1] for li in range(NL)])
        for l in range(NL):
            ln_x = ln2_call(l, ln2in_b[l])
            dln = ln2_call(l, ln2in_b[l] + EPS * cdir) - ln_x
            outs = mlp_batch(l, np.stack([ln_x, ln_x + dln,
                                          mlpin_b[l] + EPS * cdir]))
            out_b, out_comp, out_direct = outs
            g_comp[i, l] = float((out_comp - out_b) @ cdir) / EPS
            g_direct[i, l] = float((out_direct - out_b) @ cdir) / EPS
            v1_all.append(float(np.abs(ln_x - mlpin_b[l]).max()))
            v2_all.append(float(np.abs(out_b - mlpout_b[l]).max()))
        if (i + 1) % 20 == 0:
            log('%s words [%d/%d]' % (mname, i + 1, n_words))

    for h in handles:
        h.remove()
    v1_chk = float(np.max(v1_all))
    v2_chk = float(np.max(v2_all))
    w3_ok = bool(v1_chk < 0.05 and v2_chk < 0.05)
    log('%s W3: v1=%.6f v2=%.6f pass=%s' % (mname, v1_chk, v2_chk, w3_ok))
    if not w3_ok:
        voided = True

    store[mname] = {
        'g_direct': g_direct, 'g_comp': g_comp, 'labels': labels,
        'target_list': target_list, 'dW_unit': dW_unit,
        'W3': {'v1_max': round(v1_chk, 6), 'v2_max': round(v2_chk, 6),
               'verdict': w3_ok},
        'tc': dict(tc),
    }

    del model
    torch.cuda.empty_cache()

# ---------- window arms ----------
rng = np.random.default_rng(SEED)
arms = {}
for mname, mcfg in MODELS.items():
    st = store[mname]
    g = st['g_direct']
    labels = st['labels']
    n_words = len(labels)
    for wname, (lo, hi) in mcfg['windows'].items():
        B3 = np.stack([unit(g[i, lo:hi]) for i in range(n_words)])
        C3 = B3 @ B3.T
        acc = loo_nn_acc(C3, labels)
        null_acc = []
        for _ in range(200):
            pl = rng.permutation(labels)
            null_acc.append(loo_nn_acc(C3, pl))
        p95 = float(np.percentile(null_acc, 95))
        arms[wname] = {
            'model': mname, 'window': [lo, hi],
            'acc': round(float(acc), 4), 'null_p95': round(p95, 4),
            'null_mean': round(float(np.mean(null_acc)), 4),
            'profile_mean': [round(float(x), 4)
                             for x in g.mean(0)[lo:hi]],
        }
        log('%s [%d,%d): acc=%.4f null_p95=%.4f'
            % (wname, lo, hi, acc, p95))

# ---------- gates ----------
c0 = bool(arms['q_base10']['acc'] == REF_Q
          and arms['d_base8']['acc'] == REF_D)
c1 = bool(max(abs(arms['q_early8']['acc'] - REF_Q),
              abs(arms['q_late8']['acc'] - REF_Q)) >= BAND)
c2 = bool(arms['d_long10']['acc'] - REF_D >= BAND)
c3 = bool(abs(arms['q_early8']['acc'] - arms['q_late8']['acc']) >= BAND)
if c1 and c2:
    attribution = 'window_length'
elif (not c1) and (not c2) and (not c3):
    attribution = 'model_factor'
elif c3 and not (c1 and c2):
    attribution = 'position_matters'
else:
    attribution = 'mixed'
verdict = 'voided' if (voided or not c0) else attribution
log('C0=%s C1=%s C2=%s C3=%s -> %s' % (c0, c1, c2, c3, verdict))

res = {
    'phase': 2884,
    'prereg': PREREG,
    'W3_per_model': {m: store[m]['W3'] for m in store},
    'C0': {'verdict': c0, 'refs': {'q_base10': REF_Q,
                                   'd_base8': REF_D}},
    'arms': arms,
    'C1_qwen_length': {'verdict': c1, 'band': BAND},
    'C2_ds7b_length': {'verdict': c2, 'band': BAND},
    'C3_position': {'verdict': c3, 'band': BAND},
    'attribution': attribution,
    'final_verdict': 'window_discriminate=%s (C0=%s C1=%s C2=%s C3=%s)'
                     % (verdict, c0, c1, c2, c3),
    'runtime_s': round(time.monotonic() - T0, 1),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
np.savez_compressed(
    os.path.join(OUT, 'window_discriminate.npz'),
    **{('%s__g_direct' % m): store[m]['g_direct'].astype(np.float32)
       for m in store},
    **{('%s__g_comp' % m): store[m]['g_comp'].astype(np.float32)
       for m in store},
    **{('%s__labels' % m): store[m]['labels'] for m in store},
    **{('%s__targets' % m): np.array([json.dumps(t) for t in
                                      store[m]['target_list']])
       for m in store},
    **{('%s__dW_unit' % m): store[m]['dW_unit'].astype(np.float32)
       for m in store})
log('==== VERDICT: %s ====' % res['final_verdict'])
with io.open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
