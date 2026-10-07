# -*- coding: utf-8 -*-
"""Probe: bf16 logit noise floor + 3044 T5 raw scales.
Decides whether 3044's logit-response statistics were
dominated by bf16 rounding-flip noise."""
import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTP = (r'D:\AI2050\Ai2050-OpenOne\gpt5_temp'
        r'\probe3045_result.txt')
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
o = []


def w(msg):
    o.append(str(msg))


z44 = np.load(os.path.join(
    BASE, 'phase3044',
    'omega_p41_field_axis_injection_qwen',
    'omega_p41_field_axis_injection_qwen.npz'),
    allow_pickle=True)

# ---------- P5: 3044 raw eff scales ----------
kind = z44['rec_kind']
body = z44['rec_body']
dlg = z44['rec_dlg']
GB = z44['G20'].mean(axis=0)
w('=== P5: 3044 raw dlg norms per kind ===')
for kd, nm in ((0, 'alpha L3'), (1, 'ubar L3'),
               (2, 'rand L3'), (3, 'ubar20'),
               (4, 'rand20')):
    m = kind == kd
    v = dlg[m]
    w('%s: n=%d med=%.4f q25=%.4f q75=%.4f max=%.2f'
      % (nm, len(v), float(np.median(v)),
         float(np.percentile(v, 25)),
         float(np.percentile(v, 75)),
         float(v.max())))
eff_u = {}
eff_r = {}
for k in range(len(kind)):
    if int(kind[k]) == 3:
        b = int(body[k])
        eff_u[b] = float(dlg[k]) / float(GB[b])
    if int(kind[k]) == 4:
        b = int(body[k])
        eff_r.setdefault(b, []).append(
            float(dlg[k]) / float(GB[b]))
raw_u = [eff_u[b] for b in sorted(eff_u)]
raw_rmed = [float(np.median(eff_r[b]))
            for b in sorted(eff_u)]
ratio = [eff_u[b] / float(np.median(eff_r[b]))
         for b in sorted(eff_u)]
w('P5: obs_t5 stored = %.6f' % float(z44['obs_t5']))
w('P5: raw med eff_u = %.6f' % float(np.median(raw_u)))
w('P5: raw med eff_r = %.6f'
  % float(np.median(raw_rmed)))
w('P5: ratio med = %.6f' % float(np.median(ratio)))

# ---------- bf16 model ----------
torch.manual_seed(3009)
np.random.seed(3009)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
W_U = model.lm_head.weight.detach().double() \
    .cpu().numpy()
cap = {}


def pre_norm(module, args, kwargs):
    cap['h'] = args[0][:, -1, :].detach() \
        .double().cpu().numpy().copy()
    return None


model.model.norm.register_forward_pre_hook(
    pre_norm, with_kwargs=True)
state = {3: {'on': False, 'pos': -1, 'delta': None},
         20: {'on': False, 'pos': -1, 'delta': None}}
SL = slice(7 * 128, 8 * 128)


def make_hook(li):
    def h(module, inp, out):
        st = state[li]
        if st['on']:
            out[0, st['pos'], SL] += st['delta']
        return out
    return h


for li in (3, 20):
    model.model.layers[li].self_attn.v_proj \
        .register_forward_hook(make_hook(li))


def forward_inj(ids, li, pos, delta):
    for lj in (3, 20):
        state[lj]['on'] = False
    d = torch.tensor(np.asarray(delta),
                     dtype=torch.bfloat16,
                     device='cuda')
    state[li].update(on=True, pos=pos, delta=d)
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
    state[li]['on'] = False
    return out.logits[0, -1].detach() \
        .double().cpu().numpy()


s = 'The weather was cold, so'
ids = [int(x) for x in tok(
    s, add_special_tokens=False)['input_ids']]
pos = ids.index(word_tok := tok(
    ' so', add_special_tokens=False)['input_ids'][0])

lg1 = forward_inj(ids, 3, pos,
                  np.zeros(128))
w('=== P1: bf16 quantization floor ===')
h = cap['h'][0]
w_norm = model.model.norm.weight.detach() \
    .double().cpu().numpy()
eps = float(getattr(model.config, 'rms_norm_eps',
                    1e-6))
hn = h / np.sqrt((h ** 2).mean() + eps)
lg64 = W_U @ (hn * w_norm)
dq = lg1 - lg64
w('P1: logit std=%.3f max|lg|=%.2f'
  % (float(lg1.std()), float(np.abs(lg1).max())))
w('P1: ||lg_bf16 - lg_fp64recompute|| = %.4f '
  '(RMS per logit %.5f)'
  % (float(np.linalg.norm(dq)),
     float(np.sqrt((dq ** 2).mean()))))

# ---------- P2/P3/P4: response curves ----------
w('=== P2: ||dlg|| vs norm/layer (random dirs) ===')
rng = np.random.default_rng(9400)
R = rng.standard_normal((6, 128))
R = R / np.linalg.norm(R, axis=1)[:, None]
w('norm | L3 | L20   (med over 6 dirs)')
for nrm in (0.1, 0.3, 1.0, 3.0):
    row = []
    for li in (3, 20):
        vals = []
        for r in range(6):
            lg_i = forward_inj(ids, li, pos,
                               nrm * R[r])
            vals.append(float(np.linalg.norm(
                lg_i - lg1)))
        row.append(float(np.median(vals)))
    w('%.2f | %.4f | %.4f' % (nrm, row[0], row[1]))

w('=== P3: one-hot 0.25 vs spread 0.25 (L3) ===')
vals_oh = []
for i in range(4):
    d = np.zeros(128)
    d[i] = 0.25
    lg_i = forward_inj(ids, 3, pos, d)
    vals_oh.append(float(np.linalg.norm(
        lg_i - lg1)))
w('one-hot med=%.4f (vals %s)'
  % (float(np.median(vals_oh)),
     ['%.3f' % v for v in vals_oh]))
vals_sp = []
for r in range(4):
    lg_i = forward_inj(ids, 3, pos, 0.25 * R[r])
    vals_sp.append(float(np.linalg.norm(
        lg_i - lg1)))
w('spread med=%.4f (vals %s)'
  % (float(np.median(vals_sp)),
     ['%.3f' % v for v in vals_sp]))

w('=== P4: +- symmetry at 0.3 (L3) ===')
css = []
for r in range(4):
    dp = forward_inj(ids, 3, pos, 0.3 * R[r]) - lg1
    dm = forward_inj(ids, 3, pos, -0.3 * R[r]) - lg1
    npn = np.linalg.norm(dp)
    nmn = np.linalg.norm(dm)
    css.append(float(dp @ dm) / (npn * nmn))
w('P4: med cos(d+,d-) = %.4f (vals %s)'
  % (float(np.median(css)),
     ['%.3f' % c for c in css]))

with open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(o) + '\n')
print('probe done')
