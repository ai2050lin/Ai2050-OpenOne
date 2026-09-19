# -*- coding: utf-8 -*-
"""Phase 2883: DS7B class-axis mlp carrier cold start (P3 cross-model).

First leg of the multi-model replication campaign: port the
2861(g_direct) + 2867(B3) protocol verbatim to
deepseek-r1-distill-qwen-7b (Qwen2 arch, 28 layers, hidden 3584,
28 heads, head_dim 128, UNTIED lm_head, vocab 152064).

Protocol (deviations declared, all pre-frozen):
  D1  same-context only: B3 needs g_direct under the 'same' context;
      the 2861 L13H30 attention-patch machinery (qwen3-specific module
      surgery) is used only for residual checks there, NOT for
      g_direct -> omitted here.  func/null contexts not needed for B3.
  D2  window scaling rule (frozen): WIN_LO_m = floor(26/36 * L_m),
      WIN_HI_m = L_m.  qwen3-4b L=36 -> [26,36) = 10 layers (verbatim
      2861); DS7B L=28 -> [20,28) = 8 layers.
  D3  vocab: same 2806 class list (10 classes x 10 words), per-model
      single-token filter, targets[:8] per class.
  D4  E rows from lm_head.weight (untied in DS7B; 2861 used lm_head
      too, so identical semantics).

Prereg (frozen before any model/vocab statistic is computed):
  V1  vocab_legal iff >= 8/10 classes retain >= 5 single-token words
      AND total words >= 50.  If false -> phase ends, spectrum not run
      (quarantine, honest negative).
  W3  consistency: v1 = max|LN2(ln2in) - mlpin| < 0.05 and
      v2 = max|mlp(ln2in) - mlpout| < 0.05 (bf16 rebuild budget;
      2861 checks were ~0, 2877 taught ~2e-3 cross-shape).  Violation
      voids all verdicts.
  W1  retrieval: LOO-NN same-class acc on B3 (cosmat, diagonal -2)
      > null p95 (200 label permutations, SEED=2883) =>
      mlp_carries_class_axis_ds7b.
  W2  replication band (descriptive, frozen pre-observation):
      |acc_ds7b - 0.875| <= 0.15 => replicates_within_band; else
      carrier_deviation (either direction, registered as measured).
  verdict mlp_carrier_replicates iff V1 and W3 and W1 (W2 banding is
  descriptive).
Output: result/rdc_query_construction_20260913/phase2883/ds7b_class_mlp/
        {execution.json, result.json, ds7b_class_mlp.npz}
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
OUT = os.path.join(BASE, 'phase2883', 'ds7b_class_mlp')
SRC_2806_EXEC = os.path.join(BASE, 'phase2806', 'qwen4_hierarchy',
                             'execution.json')
MODEL_KEY = 'ds7'
MODEL_NAME = 'deepseek-r1-distill-qwen-7b'
SEED = 2883
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

# ---------- load architecture facts (config only, no weights) ----------
cfg = json.load(io.open(
    r'D:\AI2050\Ai2050-OpenOne\models\hf\deepseek-r1-distill-qwen-7b'
    r'\config.json', encoding='utf-8'))
L = int(cfg['num_hidden_layers'])
WIN_LO = int(np.floor(26 / 36.0 * L))
WIN_HI = L
NL = L
PREREG = {
    'D1': 'same-context only; 2861 patch machinery omitted (not needed '
          'for g_direct; qwen3-specific)',
    'D2': 'window scaling frozen: WIN=[floor(26/36*L), L); qwen3-4b '
          '[26,36)=10L, DS7B L=%d -> [%d,%d)=%dL'
          % (L, WIN_LO, WIN_HI, WIN_HI - WIN_LO),
    'D3': '2806 class list, per-model single-token filter, targets[:8]',
    'D4': 'E rows from lm_head.weight (DS7B untied)',
    'V1': 'vocab_legal iff >= 8/10 classes keep >= 5 single-token words '
          'and total >= 50; else quarantine, spectrum not run',
    'W3': 'v1=max|LN2(ln2in)-mlpin|<0.05 and v2=max|mlp(ln2in)-mlpout|'
          '<0.05 else void',
    'W1': 'LOO-NN acc(B3) > null p95 (200 perms, SEED=2883) => '
          'mlp_carries_class_axis_ds7b',
    'W2': 'descriptive band frozen: |acc-0.875|<=0.15 => '
          'replicates_within_band else carrier_deviation',
    'verdict': 'mlp_carrier_replicates iff V1 and W3 and W1',
}
exec_doc = {
    'phase': 2883,
    'name': 'ds7b_class_mlp',
    'seed': SEED,
    'model': MODEL_NAME,
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
    r'D:\AI2050\Ai2050-OpenOne\models\hf' + '\\' + MODEL_NAME,
    local_files_only=True, trust_remote_code=True, use_fast=True)

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
    res = {'phase': 2883, 'prereg': PREREG, 'V1': {'verdict': False,
           'classes_ge5': classes_ge5, 'n_words': n_words},
           'final_verdict': 'vocab_legal=false -> quarantine, spectrum '
           'not run', 'runtime_s': round(time.monotonic() - T0, 1)}
    with io.open(os.path.join(OUT, 'result.json'), 'w',
                 encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log('==== VERDICT: vocab_legal=false ====')
    sys.exit(0)

# ---------- class directions ----------
import torch

from phase2662_symmetric_mapping_contract import load_native
model, _ = load_native(MODEL_KEY)
model.eval()

# lm_head offloaded to meta by accelerate (12GiB cap) -> read shard
# directly from safetensors (zero forward, D4 unchanged in semantics)
MD = (r'D:\AI2050\Ai2050-OpenOne\models\hf'
      r'\deepseek-r1-distill-qwen-7b')
idx = json.load(io.open(os.path.join(MD, 'model.safetensors.index.json'),
                        encoding='utf-8'))
wmap = idx['weight_map']
_key = 'lm_head.weight' if 'lm_head.weight' in wmap \
    else 'model.lm_head.weight'
from safetensors import safe_open
with safe_open(os.path.join(MD, wmap[_key]), framework='pt') as sf:
    W_U = sf.get_tensor(_key).float().numpy()
log('lm_head read from %s (%s)' % (wmap[_key], _key))
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


n_win = WIN_HI - WIN_LO
g_direct = np.zeros((n_words, n_win))
g_comp = np.zeros((n_words, n_win))
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
    if (i + 1) % 20 == 0:
        log('words [%d/%d]' % (i + 1, n_words))

v1_chk = float(np.max(v1_all))
v2_chk = float(np.max(v2_all))
w3 = bool(v1_chk < 0.05 and v2_chk < 0.05)
log('W3: v1=%.6f v2=%.6f pass=%s' % (v1_chk, v2_chk, w3))

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

per_class = {}
for ci, c in enumerate(CAT_WORDS):
    m = labels == ci
    if m.sum() >= 3:
        per_class[c] = round(loo_nn_acc(C3[np.ix_(m, m)],
                                        labels[m]), 4)

verdict = bool(v1 and w3 and w1)
res = {
    'phase': 2883,
    'model': MODEL_NAME,
    'prereg': PREREG,
    'V1': {'classes_ge5': classes_ge5, 'n_words': n_words,
           'per_class_counts': {c: len(targets[c])
                                for c in CAT_WORDS}, 'verdict': v1},
    'window': [WIN_LO, WIN_HI],
    'W3': {'v1_max': round(v1_chk, 6), 'v2_max': round(v2_chk, 6),
           'verdict': w3},
    'W1': {'acc_b3': round(acc_b3, 4), 'null_p95': round(null_p95, 4),
           'null_mean': round(float(np.mean(null_acc)), 4),
           'verdict': 'mlp_carries_class_axis_ds7b' if w1
                      else 'carrier_absent_ds7b'},
    'W2': {'qwen3_4b_reference': 0.875, 'delta': round(acc_b3 - 0.875, 4),
           'verdict': w2},
    'g_direct_mean_profile': [round(float(x), 4)
                              for x in g_direct.mean(0)],
    'g_comp_mean_profile': [round(float(x), 4) for x in g_comp.mean(0)],
    'mlp_carrier_replicates': verdict,
    'final_verdict': 'mlp_carrier_replicates=%s (acc %.4f vs null p95 '
                     '%.4f; %s vs qwen3-4b 0.875)'
                     % (verdict, acc_b3, null_p95, w2),
    'runtime_s': round(time.monotonic() - T0, 1),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=2, ensure_ascii=False)
np.savez_compressed(
    os.path.join(OUT, 'ds7b_class_mlp.npz'),
    B3=B3.astype(np.float32), g_direct=g_direct.astype(np.float32),
    g_comp=g_comp.astype(np.float32),
    target_list=np.array([json.dumps(t) for t in target_list]),
    dW_unit=dW_unit.astype(np.float32), labels=labels)
log('==== VERDICT: %s ====' % res['final_verdict'])
with io.open(os.path.join(OUT, 'run.log'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')

del model
torch.cuda.empty_cache()
