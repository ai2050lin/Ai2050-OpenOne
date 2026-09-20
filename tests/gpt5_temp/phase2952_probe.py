# -*- coding: utf-8 -*-
"""Phase 2952 preflight probe: A11 recovery + VAL/ATT shares.

Checks (no execution.json; observations here -> formal verdicts
labeled quasi-post-hoc per discipline 9):
  P1: GQA gates (v_proj out 1024, o_proj in 4096)
  P2: A11 least-squares recovery residual (x vs v0+A11*(v1-v0))
  P3: sc_base keep-heads vs 2950 sc_B1; sc_inj vs sc_I0
  P4: VAL/ATT decomposition shares; correlation with 2951 R
  P5: value-path amplification mu_val = median(VAL/g)
"""
import json
import os
import sys
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2948 = os.path.join(BASE, 'phase2948', 'wov_head_gain',
                        'wov_head_gain.npz')
SRC_2950 = os.path.join(BASE, 'phase2950', 'rebalance_anatomy',
                        'rebalance_anatomy.npz')
SRC_2951 = os.path.join(BASE, 'phase2951',
                        'rebalance_carrier_functional',
                        'rebalance_carrier_functional.npz')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
VOCAB = 151936
LI_SWITCH, S_SWITCH = 17, 1.0
LI_GRAD, S_GRAD = 16, 2.0
LAYERS = (LI_SWITCH, LI_GRAD)

lines = []


def log(m):
    lines.append(m)
    print(m, flush=True)


z87 = np.load(SRC_2887, allow_pickle=True)
words = [tuple(str(w).split(':')) for w in z87['words']]
lab_lang = np.asarray(z87['labels_lang']).astype(int)
n_words = len(words)
z27 = np.load(SRC_2927, allow_pickle=True)
dirs_word_27 = z27['dirs_word'].astype(np.float64)

import torch
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
from phase2662_symmetric_mapping_contract import load_native
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MD, local_files_only=True,
                                    trust_remote_code=True,
                                    use_fast=True)
tc = {}


def tid(t):
    if t not in tc:
        ids = tok(' ' + t, add_special_tokens=False)['input_ids']
        if len(ids) != 1:
            ids = tok(t, add_special_tokens=False)['input_ids']
        assert len(ids) == 1
        tc[t] = int(ids[0])
    return tc[t]


tid_map = {}
for lang, ck, w in words:
    tid_map[w] = tid(w)
func_tid = tid('the')
batch = [[func_tid, tid_map[words[i][2]]]
         for i in range(n_words)]

model, _ = load_native('qwen4')
model.eval()
layers = model.model.layers
log('model loaded')

vp_out = layers[0].self_attn.v_proj.weight.shape[0]
op_in = layers[0].self_attn.o_proj.in_features
log('P1 gates: v_proj out %d (expect 1024) | o_proj in %d '
    '(expect 4096)' % (vp_out, op_in))

cap_v = {}
cap_x = {}
inj = {'li': None, 'scale': 0.0, 'vec': None}
handles = []


def pre_attn(li):
    def h(module, args, kwargs):
        x = args[0] if args else kwargs.get('hidden_states')
        if x is None or x.dim() < 2:
            return
        if inj['li'] == li and inj['vec'] is not None:
            x = x.clone()
            x[:, 1, :] = x[:, 1, :] \
                + inj['scale'] * inj['vec']
            if args:
                return (x,) + tuple(args[1:]), kwargs
            nkw = dict(kwargs)
            nkw['hidden_states'] = x
            return args, nkw
        return None
    return h


def hook_v(li):
    def h(module, args, output):
        if li in LAYERS:
            o = output.detach().float().cpu().numpy()
            cap_v.setdefault(li, []).append((o[:, 0, :].copy(),
                                             o[:, 1, :].copy()))
        return None
    return h


def hook_x(li):
    def h(module, args, kwargs):
        x = args[0] if args else kwargs.get('input')
        if x is None or x.dim() < 2:
            return None
        if li in LAYERS:
            cap_x.setdefault(li, []).append(
                x[:, 1, :].detach().float().cpu().numpy())
        return None
    return h


for li in range(NL):
    handles.append(layers[li].self_attn
                   .register_forward_pre_hook(
                       pre_attn(li), with_kwargs=True))
for li in LAYERS:
    handles.append(layers[li].self_attn.v_proj
                   .register_forward_hook(hook_v(li)))
    handles.append(layers[li].self_attn.o_proj
                   .register_forward_pre_hook(
                       hook_x(li), with_kwargs=True))

xdir = None
z48 = np.load(SRC_2948, allow_pickle=True)
Vt8 = z48['Vt8']
dcks_S = None
z39 = np.load(os.path.join(BASE, 'phase2939', 'rotation_target',
                           'rotation_target.npz'),
              allow_pickle=True)
conds39 = [str(s) for s in z39['cond_names']]
coords39 = z39['coords'].astype(np.float64)
dcks_39 = coords39[conds39.index('null0')] \
    - coords39[conds39.index('func')]
S_IDX = (0, 1, 4)
dcks_S = dcks_39[:, list(S_IDX)]
Vt8_S = Vt8[list(S_IDX)]
xdir = dcks_S @ Vt8_S
xdir_t = torch.tensor(xdir, device='cuda', dtype=torch.bfloat16)


def run(inj_li, scale):
    cap_v.clear()
    cap_x.clear()
    inj['li'] = inj_li
    inj['scale'] = float(scale)
    inj['vec'] = xdir_t if scale else None
    with torch.no_grad():
        model(torch.tensor(batch, device='cuda'))
    inj['li'] = None
    inj['scale'] = 0.0
    inj['vec'] = None
    v = {li: (np.stack([a for a, b in cap_v[li]]),
              np.stack([b for a, b in cap_v[li]]))
         for li in cap_v}
    x = {li: np.stack(cap_x[li]) for li in cap_x}
    return v, x


t0 = time.time()
v_base, x_base = run(None, 0.0)
v_17, x_17 = run(LI_SWITCH, S_SWITCH)
v_16, x_16 = run(LI_GRAD, S_GRAD)
log('forwards done %.1fs' % (time.time() - t0))

z50 = np.load(SRC_2950, allow_pickle=True)
z51 = np.load(SRC_2951, allow_pickle=True)
Wo_cache = {li: layers[li].self_attn.o_proj.weight
            .detach().float().cpu().numpy() for li in LAYERS}
u35 = dirs_word_27[NL - 1]
NKV = vp_out // HD
HPG = NH // NKV

for li, v_inj, x_inj, s_inj in (
        (LI_SWITCH, v_17, x_17, S_SWITCH),
        (LI_GRAD, v_16, x_16, S_GRAD)):
    Wo = Wo_cache[li]
    v0b, v1b = v_base[li]
    v0n, v1n = v_inj[li]
    Xb = x_base[li].reshape(n_words, NH, HD)
    Xn = x_inj[li].reshape(n_words, NH, HD)
    v0b_r = v0b.reshape(n_words, NKV, HD)
    v1b_r = v1b.reshape(n_words, NKV, HD)
    v0n_r = v0n.reshape(n_words, NKV, HD)
    v1n_r = v1n.reshape(n_words, NKV, HD)

    # A11 recovery (base condition)
    A11b = np.zeros((n_words, NH))
    rec_res = 0.0
    for hh in range(NH):
        k = hh // HPG
        d = v1b_r[:, k, :] - v0b_r[:, k, :]
        num = ((Xb[:, hh, :] - v0b_r[:, k, :]) * d).sum(1)
        den = (d * d).sum(1)
        A11b[:, hh] = num / np.maximum(den, 1e-30)
        rec = v0b_r[:, k, :] + A11b[:, hh:hh + 1] * d
        rec_res = max(rec_res, float(np.abs(
            rec - Xb[:, hh, :]).max()))
    # A11 under injection
    A11n = np.zeros((n_words, NH))
    rec_res_n = 0.0
    for hh in range(NH):
        k = hh // HPG
        d = v1n_r[:, k, :] - v0n_r[:, k, :]
        num = ((Xn[:, hh, :] - v0n_r[:, k, :]) * d).sum(1)
        den = (d * d).sum(1)
        A11n[:, hh] = num / np.maximum(den, 1e-30)
        rec = v0n_r[:, k, :] + A11n[:, hh:hh + 1] * d
        rec_res_n = max(rec_res_n, float(np.abs(
            rec - Xn[:, hh, :]).max()))
    log('L%d P2 A11 recon residual base %.2e inj %.2e | '
        'A11 base med %.3f inj med %.3f'
        % (li, rec_res, rec_res_n,
           float(np.median(A11b)), float(np.median(A11n))))

    # sc maps (mean over words), keep heads
    keep = z51['keep_%s' % ('L17' if li == LI_SWITCH else 'L16')]
    top5 = [0, 7, 24, 22, 19] if li == LI_SWITCH \
        else [13, 16, 1, 17, 6]
    mask = np.ones(NH * HD, dtype=bool)
    for hh in top5:
        mask[hh * HD:(hh + 1) * HD] = False

    def sc_of(Xf):
        Xf = Xf.reshape(n_words, NH, HD)
        c = np.zeros((NH, n_words))
        for hh in range(NH):
            oh = Xf[:, hh, :] \
                @ Wo[:, hh * HD:(hh + 1) * HD].T
            c[hh] = oh @ u35
        return c[:, lab_lang == 0].mean(1) \
            - c[:, lab_lang == 1].mean(1)

    sc_b = sc_of(Xb)
    sc_n = sc_of(Xn)
    key = 'L%d' % li
    d_b = float(np.abs(sc_b[keep]
                       - z50['sc_B1_%s' % key][keep]).max())
    d_n = float(np.abs(sc_n[keep]
                       - z50['sc_I0_%s' % key][keep]).max())
    delta_new = sc_n - sc_b
    d_delta = float(np.abs(
        delta_new[keep]
        - z51['delta_%s' % key]).max())
    log('L%d P3 sc match: base vs B1 %.2e | inj vs I0 %.2e | '
        'delta vs 2951 %.2e' % (li, d_b, d_n, d_delta))

    # decomposition
    VAL = np.zeros(NH)
    ATT = np.zeros(NH)
    for hh in range(NH):
        k = hh // HPG
        woh = Wo[:, hh * HD:(hh + 1) * HD]
        dA = A11n[:, hh] - A11b[:, hh]
        # ATT(w) = dA11 * u35.Wo.(v1n_k - v0_k)
        att_term = dA * ((v1n_r[:, k, :] - v0n_r[:, k, :])
                         @ woh.T @ u35)
        # VAL(w) = A11b * u35.Wo.(v1n_k - v1b_k)
        val_term = A11b[:, hh] * ((v1n_r[:, k, :]
                                   - v1b_r[:, k, :])
                                  @ woh.T @ u35)
        ATT[hh] = att_term.mean()
        VAL[hh] = val_term.mean()
    ident = float(np.abs((VAL + ATT)[keep]
                         - delta_new[keep]).max())
    scale_d = float(np.abs(delta_new[keep]).max())
    log('L%d P4 identity VAL+ATT vs delta %.2e (delta max '
        '%.2f)' % (li, ident, scale_d))
    share = np.abs(VAL) / np.maximum(np.abs(VAL) + np.abs(ATT),
                                     1e-30)
    log('L%d P4 val-share median %.3f | ATT>0 heads %d | '
        'ATT med %.3f VAL med %.3f'
        % (li, float(np.median(share[keep])),
           int((ATT[keep] > 0).sum()),
           float(np.median(ATT[keep])),
           float(np.median(VAL[keep]))))
    g = z51['g_%s' % ('L17' if li == LI_SWITCH else 'L16')]
    R = z51['resid_%s' % ('L17' if li == LI_SWITCH else 'L16')]
    from scipy.stats import spearmanr
    mu_val = float(np.median(VAL[keep]
                             / np.maximum(np.abs(g), 1e-30)
                             * np.sign(g)))
    log('L%d P5 mu_val (VAL/g signed) median %.3f | '
        'rho(ATT,R) %.4f | rho(VAL,g) %.4f'
        % (li, mu_val,
           float(spearmanr(ATT[keep], R).statistic),
           float(spearmanr(VAL[keep], g).statistic)))

open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase2952_probe_report.txt', 'w',
     encoding='utf-8').write(chr(10).join(lines) + chr(10))
print('PROBE DONE')
