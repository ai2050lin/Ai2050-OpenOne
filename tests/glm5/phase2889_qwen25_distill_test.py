# -*- coding: utf-8 -*-
"""Phase 2889: DS7B distill-hypothesis test (qwen2-arch non-distill
control).

Candidate A from 2888: DS7B's -0.30 carrier deviation (0.575 vs the
0.875 qwen3-4b anchor) narrowed to reasoning-distill training after
window/position (2884) and untied-geometry (2885, 2888) attributions
were excluded.  A qwen2-architecture NON-distill control exists
locally: qwen2.5-3b-instruct (Qwen2ForCausalLM, 36 layers, hidden
2048, tied embeddings, vocab 151936).  This phase runs the 2883
class-mlp protocol on it verbatim to test the distill hypothesis.

Discriminator logic (pre-frozen):
  - acc in band vs 0.875 (|acc-0.875|<=0.15)  => arch + small-size
    exonerated at 3B => distill_hypothesis_strengthened (size caveat
    3B-vs-7B registered as residual confound).
  - acc within 0.15 of DS7B 0.575             => deviation tracks
    arch/size, not training => distill_hypothesis_weakened.
  - else => inconclusive (both hypotheses survive).

Deviations (all pre-frozen):
  D1  same-context only (2883/2888 precedent).
  D2  window scaling frozen rule: WIN=[floor(26/36*L), L); L=36 ->
      [26,36) = 10 layers (identical to qwen3-4b).
  D3  2806 class list, per-model single-token filter, targets[:8].
  D4  E rows from model.embed_tokens.weight read directly from
      safetensors shard (tie_word_embeddings=true; 2884 precedent:
      mathematically identical for tied models).
  D5  manual device_map (2888 Gen2 precedent: window layers 26-35 +
      embed + norm on cuda, remaining 26 layers on cpu).
  D6  AutoModelForCausalLM load (qwen2.5-3b-instruct not in the
      load_native registry).

Prereg (frozen before any model/vocab statistic):
  V1  vocab_legal iff >= 8/10 classes retain >= 5 single-token words
      AND total >= 50; else quarantine.
  W3  Gen2 relative budget: v1_rel = max|LN2(ln2in)-mlpin|/max|mlpin|
      and v2_rel = max|mlp(ln2in)-mlpout|/max|mlpout| both < 5e-3
      (2888 Gen2 precedent; hidden-2048 scale).  Absolute reported.
      Violation voids all verdicts.
  W1  retrieval: LOO-NN same-class acc on B3 > null p95 (200 label
      permutations, SEED=2889) => mlp_carries_class_axis_qwen25_3b.
  W2  descriptive band: |acc-0.875| <= 0.15 => replicates_within_band
      else carrier_deviation (2883/2888 identical wording).
  W4  distill discriminator (frozen): 'distill_hypothesis_strengthened'
      if |acc-0.875|<=0.15; 'distill_hypothesis_weakened' if
      |acc-0.575|<=0.15; else 'inconclusive'.
  verdict mlp_carrier_replicates iff V1 and W3 and W1.
Output: phase2889/qwen25_distill_test/{execution.json, result.json,
        qwen25_distill_test.npz}
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
OUT = os.path.join(BASE, 'phase2889', 'qwen25_distill_test')
SRC_2806_EXEC = os.path.join(BASE, 'phase2806', 'qwen4_hierarchy',
                             'execution.json')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen2.5-3b-instruct'
MODEL_NAME = 'qwen2.5-3b-instruct'
SEED = 2889
EPS = 1.0
MAX_WORDS = 8

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


os.makedirs(OUT, exist_ok=True)

# ---------- architecture facts (config only) ----------
cfg = json.load(io.open(os.path.join(MD, 'config.json'),
                        encoding='utf-8'))
L = int(cfg['num_hidden_layers'])
WIN_LO = int(np.floor(26 / 36.0 * L))
WIN_HI = L
NL = L
PREREG = {
    'D1': 'same-context only; 2861 patch machinery omitted (not '
          'needed for g_direct; qwen3-specific)',
    'D2': 'window scaling frozen: WIN=[floor(26/36*L), L); L=%d -> '
          '[%d,%d)=%dL' % (L, WIN_LO, WIN_HI, WIN_HI - WIN_LO),
    'D3': '2806 class list, per-model single-token filter, '
          'targets[:8]',
    'D4': 'E rows from model.embed_tokens.weight (tied=true; 2884 '
          'precedent: mathematically identical), safetensors direct',
    'D5': 'manual device_map: window layers %d-%d + embed + norm on '
          'cuda, remaining %d layers on cpu (2888 Gen2 precedent: '
          'device_map=auto disk-offloads into META tensors)'
          % (WIN_LO, WIN_HI - 1, L - (WIN_HI - WIN_LO)),
    'D6': 'AutoModelForCausalLM load (qwen2.5-3b-instruct not in '
          'load_native registry)',
    'V1': 'vocab_legal iff >= 8/10 classes keep >= 5 single-token '
          'words and total >= 50; else quarantine',
    'W3': 'Gen2 relative budget: v1_rel=max|LN2(ln2in)-mlpin|/'
          'max|mlpin| and v2_rel=max|mlp(ln2in)-mlpout|/max|mlpout| '
          'both < 5e-3 (2888 Gen2 precedent, hidden-2048 scale). '
          'Absolute values reported alongside. Violation voids all.',
    'W1': 'LOO-NN acc(B3) > null p95 (200 perms, SEED=2889) => '
          'mlp_carries_class_axis_qwen25_3b',
    'W2': 'descriptive band frozen: |acc-0.875|<=0.15 => '
          'replicates_within_band else carrier_deviation',
    'W4': 'distill discriminator (frozen): distill_hypothesis_'
          'strengthened if |acc-0.875|<=0.15; distill_hypothesis_'
          'weakened if |acc-0.575|<=0.15; else inconclusive',
    'verdict': 'mlp_carrier_replicates iff V1 and W3 and W1',
}
exec_doc = {
    'phase': 2889,
    'name': 'qwen25_distill_test',
    'seed': SEED,
    'model': MODEL_NAME,
    'purpose': 'DS7B carrier-deviation attribution: qwen2-arch '
               'non-distill control (distill-hypothesis test, '
               'candidate A from 2888)',
    'arch': {'model_type': cfg.get('model_type'), 'layers': L,
             'hidden': cfg.get('hidden_size'),
             'heads': cfg.get('num_attention_heads'),
             'kv_heads': cfg.get('num_key_value_heads'),
             'vocab': cfg.get('vocab_size'),
             'tie_word_embeddings': cfg.get('tie_word_embeddings')},
    'prereg': PREREG,
    'frozen_at_s': round(T0, 1),
}
with io.open(os.path.join(OUT, 'execution.json'), 'w',
             encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=2, ensure_ascii=False)
log('prereg frozen (L=%d WIN=[%d,%d))' % (L, WIN_LO, WIN_HI))

# ---------- vocab (zero-forward, frozen gates) ----------
exec2806 = json.load(io.open(SRC_2806_EXEC, encoding='utf-8'))
CATS = exec2806['cats']
CAT_WORDS = list(CATS.keys())

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    MD, local_files_only=True, trust_remote_code=True, use_fast=True)

tc = {}


def tid(t):
    if t not in tc:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        assert len(ids) == 1, '%s -> %s' % (t, ids)
        tc[t] = int(ids[0])
    return tc[t]


single_tok = []
for w in [x for v in CATS.values() for x in v]:
    try:
        tid(w)
        single_tok.append(w)
    except AssertionError:
        pass
targets = {c: [w for w in CATS[c] if w in single_tok][:MAX_WORDS]
           for c in CAT_WORDS}
classes_ge5 = sum(1 for c in CAT_WORDS if len(targets[c]) >= 5)
n_words = sum(len(targets[c]) for c in CAT_WORDS)
v1 = bool(classes_ge5 >= 8 and n_words >= 50)
log('V1: classes_ge5=%d n_words=%d vocab_legal=%s'
    % (classes_ge5, n_words, v1))
target_list = [(c, w) for c in CAT_WORDS for w in targets[c]]
labels = np.array([CAT_WORDS.index(c) for c, _ in target_list])

if not v1:
    res = {'phase': 2889, 'prereg': PREREG, 'V1': {'verdict': False,
           'classes_ge5': classes_ge5, 'n_words': n_words},
           'final_verdict': 'vocab_legal=false -> quarantine, spectrum '
           'not run', 'runtime_s': round(time.monotonic() - T0, 1)}
    with io.open(os.path.join(OUT, 'result.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log('==== VERDICT: vocab_legal=false ====')
    sys.exit(0)

# ---------- class directions (embed_tokens from shard, D4) ----------
import torch
from transformers import AutoModelForCausalLM

# tied model: lm_head.weight absent from shards; embed_tokens is the
# unembedding (2884 precedent)
idx = json.load(io.open(os.path.join(MD, 'model.safetensors.index.json'),
                        encoding='utf-8'))
wmap = idx['weight_map']
_key = 'model.embed_tokens.weight'
if _key not in wmap:
    _key = 'lm_head.weight' if 'lm_head.weight' in wmap \
        else 'model.lm_head.weight'
from safetensors import safe_open
with safe_open(os.path.join(MD, wmap[_key]), framework='pt') as sf:
    W_U = sf.get_tensor(_key).float().numpy()
log('unembed rows read from %s (%s)' % (wmap[_key], _key))
Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
cents = []
for c in CAT_WORDS:
    ws = [w for w in CATS[c] if w in single_tok]
    cents.append(np.stack([Erows[w] for w in ws]).mean(0))
Cm = np.stack(cents)
dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
dW_unit = np.stack([unit(dW[i]) for i in range(10)])

rng = np.random.default_rng(SEED)
vocab_size = int(W_U.shape[0])
word_tids = set(tc.values())
null_tids = {}
while len(null_tids) < n_words:
    r = int(rng.integers(0, vocab_size))
    if r not in word_tids and r > 0:
        null_tids[len(null_tids)] = r
func_tid = tid('the')
log('dirs ready; func=%d vocab=%d' % (func_tid, vocab_size))

# ---------- model load (D5/D6) ----------
dm = {'model.embed_tokens': 0, 'model.norm': 0}
if 'lm_head' in wmap or True:
    dm['lm_head'] = 0
for li_ in range(L):
    dm['model.layers.%d' % li_] = 0 if WIN_LO <= li_ < WIN_HI else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    MD, local_files_only=True, device_map=dm)
model.eval()
layers = model.model.layers
log('model loaded: %d layers (window %d-%d on cuda)'
    % (len(layers), WIN_LO, WIN_HI - 1))

# meta check (2888 Gen1 lesson): every window-layer param must be real
for li_ in range(WIN_LO, WIN_HI):
    for p in layers[li_].mlp.parameters():
        assert p.device.type != 'meta' and p.is_floating_point(), \
            'meta tensor in window layer %d' % li_
log('meta check: window layers all real tensors')

# ---------- hooks (same-context only, D1) ----------
cap = {'mlp': {}, 'ln2in': {}, 'mlpin': {}}


def make_out_hook(kind, li):
    def hook(module, args, output):
        o = output[0] if isinstance(output, tuple) else output
        cap[kind].setdefault(li, []).append(
            o[0].detach().float().cpu().numpy())
    return hook


def make_in_hook(kind, li):
    def pre_hook(module, args, kwargs):
        x = args[0] if args else kwargs.get('hidden_states')
        if x is None:
            return
        cap[kind].setdefault(li, []).append(
            x.detach()[0].float().cpu().numpy())
    return pre_hook


handles = []
for li, layer in enumerate(layers):
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


vdt = next(layers[0].mlp.parameters()).dtype


def layer_dev(li):
    return next(layers[li].mlp.parameters()).device


def mlp_batch(li, X):
    t = torch.tensor(X[None, :], device=layer_dev(li), dtype=vdt)
    with torch.no_grad():
        o = layers[li].mlp(t)
    if isinstance(o, tuple):
        o = o[0]
    return o[0].detach().float().cpu().numpy().astype(np.float64)


def ln2_call(li, x):
    t = torch.tensor(x[None, :], device=layer_dev(li), dtype=vdt)
    with torch.no_grad():
        o = layers[li].post_attention_layernorm(t)
    return o[0].detach().float().cpu().numpy().astype(np.float64)


n_win = WIN_HI - WIN_LO
g_direct = np.zeros((n_words, n_win))
g_comp = np.zeros((n_words, n_win))
v1_all = []
v2_all = []
mlpin_all = []
mlpout_all = []

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

    for q, l in enumerate(range(WIN_LO, WIN_HI)):
        ln_x = ln2_call(l, ln2in_b[l])
        dln = ln2_call(l, ln2in_b[l] + EPS * cdir) - ln_x
        outs = mlp_batch(l, np.stack([ln_x, ln_x + dln,
                                      mlpin_b[l] + EPS * cdir]))
        out_b, out_comp, out_direct = outs
        g_comp[i, q] = float((out_comp - out_b) @ cdir) / EPS
        g_direct[i, q] = float((out_direct - out_b) @ cdir) / EPS
        v1_all.append(float(np.abs(ln_x - mlpin_b[l]).max()))
        v2_all.append(float(np.abs(out_b - mlpout_b[l]).max()))
        mlpin_all.append(mlpin_b[l])
        mlpout_all.append(mlpout_b[l])
    if (i + 1) % 20 == 0:
        log('words [%d/%d]' % (i + 1, n_words))

v1_chk = float(np.max(v1_all))
v2_chk = float(np.max(v2_all))
den_v1 = float(np.max([np.abs(x).max() for x in mlpin_all]))
den_v2 = float(np.max([np.abs(x).max() for x in mlpout_all]))
v1_rel = v1_chk / max(den_v1, 1e-30)
v2_rel = v2_chk / max(den_v2, 1e-30)
w3 = bool(v1_rel < 5e-3 and v2_rel < 5e-3)
log('W3 Gen2: v1_abs=%.6f v2_abs=%.6f v1_rel=%.3e v2_rel=%.3e pass=%s'
    % (v1_chk, v2_chk, v1_rel, v2_rel, w3))

# ---------- B3 + retrieval ----------
B3 = np.stack([unit(g_direct[i]) for i in range(n_words)])
C3 = B3 @ B3.T
acc_b3 = loo_nn_acc(C3, labels)
null_acc = []
for _ in range(200):
    pl = rng.permutation(labels)
    null_acc.append(loo_nn_acc(C3, pl))
null_p95 = float(np.percentile(null_acc, 95))
w1 = bool(acc_b3 > null_p95)
w2 = 'replicates_within_band' if abs(acc_b3 - 0.875) <= 0.15 \
    else 'carrier_deviation'
if abs(acc_b3 - 0.875) <= 0.15:
    w4 = 'distill_hypothesis_strengthened'
elif abs(acc_b3 - 0.575) <= 0.15:
    w4 = 'distill_hypothesis_weakened'
else:
    w4 = 'inconclusive'

verdict = bool(v1 and w3 and w1)
res = {
    'phase': 2889,
    'model': MODEL_NAME,
    'prereg': PREREG,
    'V1': {'classes_ge5': classes_ge5, 'n_words': n_words,
           'per_class_counts': {c: len(targets[c])
                                for c in CAT_WORDS}, 'verdict': v1},
    'window': [WIN_LO, WIN_HI],
    'W3': {'v1_abs_max': round(v1_chk, 6),
           'v2_abs_max': round(v2_chk, 6),
           'v1_rel': float('%.3e' % v1_rel),
           'v2_rel': float('%.3e' % v2_rel),
           'budget': 'rel < 5e-3 (Gen2, see prereg)',
           'verdict': w3},
    'W1': {'acc_b3': round(acc_b3, 4), 'null_p95': round(null_p95, 4),
           'null_mean': round(float(np.mean(null_acc)), 4),
           'verdict': 'mlp_carries_class_axis_qwen25_3b' if w1
                      else 'carrier_absent_qwen25_3b'},
    'W2': {'qwen3_4b_reference': 0.875, 'delta': round(acc_b3 - 0.875, 4),
           'verdict': w2},
    'W4': {'ds7b_reference': 0.575, 'delta_vs_ds7b':
           round(acc_b3 - 0.575, 4), 'verdict': w4},
    'g_direct_mean_profile': [round(float(x), 4)
                              for x in g_direct.mean(0)],
    'g_comp_mean_profile': [round(float(x), 4) for x in g_comp.mean(0)],
    'mlp_carrier_replicates': verdict,
    'final_verdict': 'mlp_carrier_replicates=%s (acc %.4f vs null p95 '
                     '%.4f; %s vs qwen3-4b 0.875; %s)'
                     % (verdict, acc_b3, null_p95, w2, w4),
    'runtime_s': round(time.monotonic() - T0, 1),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
np.savez_compressed(
    os.path.join(OUT, 'qwen25_distill_test.npz'),
    B3=B3.astype(np.float32), g_direct=g_direct.astype(np.float32),
    g_comp=g_comp.astype(np.float32),
    target_list=np.array([json.dumps(t) for t in target_list]),
    dW_unit=dW_unit.astype(np.float32), labels=labels)
log('==== VERDICT: %s ====' % res['final_verdict'])
with io.open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')

del model
torch.cuda.empty_cache()
