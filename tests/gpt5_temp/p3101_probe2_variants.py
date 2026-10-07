# -*- coding: utf-8 -*-
"""P3101 probe 2: variant test for the repV injection semantics.

V1: L37-only, base rows [0,4) <- pref rows [off, off+4)  (current 3101)
V2: full-layer replacement, identical to 3093 forward_gen
V3: L37-only, base rows [0,4) <- pref rows [0, 4)
Compare recomputed COS_LAD[0..3] against sealed COS_LAD_A.
"""
import io

import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P93 = (R13 + r'\phase3093\omega_p91_qwen14b_l37_full_'
       r'arbitration\omega_p91_qwen14b_l37_full_'
       r'arbitration.npz')
OUT = (ROOT + r'\tests\gpt5_temp'
       r'\p3101_probe2_variants.txt')
MDIR = ROOT + r'\models\hf\Qwen3-14B'
NL, HD, KVW = 40, 128, 1024
FRONT = 4
L_INJ = 37
BODIES_A = [
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus']
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')
lines = []

tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda') \
    .eval()
layers = model.model.layers

ST = {li: {'repl': None, 'mask': None}
      for li in range(NL)}
CAP = {li: {'rec': False, 'orig': None}
       for li in range(NL)}


def mk_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0] \
                .detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


for li in range(NL):
    layers[li].self_attn.v_proj \
        .register_forward_hook(
            mk_v(ST[li], CAP[li]))


def forward_gen(ids, repl=None):
    for li in range(NL):
        ST[li]['repl'] = None
        ST[li]['mask'] = None
        CAP[li]['rec'] = True
        CAP[li]['orig'] = None
    if repl is not None:
        rt = torch.tensor(
            np.ascontiguousarray(repl),
            dtype=torch.bfloat16,
            device='cuda')
        m_all = torch.ones(
            len(ids), dtype=torch.bool,
            device='cuda')
        for li in range(NL):
            ST[li]['repl'] = rt[li]
            ST[li]['mask'] = m_all
    ids_t = torch.tensor([ids],
                         device='cuda')
    with torch.no_grad():
        o = model(ids_t, use_cache=False)
    lg = o.logits[0, -1].detach() \
        .double().cpu().numpy()
    vb = np.stack([CAP[li]['orig'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    del o
    for li in range(NL):
        ST[li]['repl'] = None
        ST[li]['mask'] = None
        CAP[li]['rec'] = False
        CAP[li]['orig'] = None
    return lg, vb


assembled = []
for bi in range(8):
    for ci in range(4):
        s = (PREFIXES[ci] + ' '
             + BODIES_A[bi]) \
            if PREFIXES[ci] else BODIES_A[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
                'input_ids']]
        assembled.append(
            {'ids': ids, 'cond': ci,
             'body': bi})
idx_of = {(a['cond'], a['body']): i
          for i, a in enumerate(assembled)}
offm = {}
for a in assembled:
    if a['cond'] == 0:
        offm[(a['cond'], a['body'])] = 0
    else:
        bid = assembled[idx_of[(0,
            a['body'])]]['ids']
        offm[(a['cond'], a['body'])] = \
            len(a['ids']) - len(bid)

n_pr = len(assembled)
LG = np.zeros((n_pr, 151936))
NMAX = max(len(a['ids'])
           for a in assembled)
VB = np.zeros((n_pr, NL, NMAX, KVW))
for i, a in enumerate(assembled):
    lg, vb = forward_gen(a['ids'])
    LG[i] = lg
    VB[i, :, :vb.shape[1], :] = vb
lines.append('banks LG%s VB%s'
             % (LG.shape, VB.shape))

z93 = np.load(P93, allow_pickle=False)
LG93 = z93['LG_A'].astype(np.float64)
TT93 = z93['TT_A'].astype(np.float64)
CL93 = z93['COS_LAD_A'].astype(np.float64)
d_lg = float(np.max(np.abs(LG - LG93)))
lines.append('LG vs sealed max diff=%.3e'
             % d_lg)


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return float(a @ b) / (na * nb)


def repv_of(k, mode):
    b_ = k % 8
    base_i = idx_of[(0, b_)]
    pref_i = idx_of[(1, b_)]
    nb = len(assembled[base_i]['ids'])
    o_ = offm[(1, b_)]
    repV = VB[base_i, :, :nb, :].copy()
    src = (o_ + np.arange(FRONT)) \
        if mode in ('V1', 'V2') \
        else np.arange(FRONT)
    repV[L_INJ, np.arange(FRONT), :] = \
        VB[pref_i][L_INJ, src, :]
    return base_i, repV


for mode in ('V1', 'V2', 'V3'):
    cs = []
    for k in range(4):
        base_i, repV = repv_of(k, mode)
        if mode == 'V2':
            lg, _ = forward_gen(
                assembled[base_i]['ids'],
                repl=repV)
        else:
            rv = repV[L_INJ]
            for li in range(NL):
                ST[li]['repl'] = None
                ST[li]['mask'] = None
            ST[L_INJ]['repl'] = \
                torch.tensor(
                    np.ascontiguousarray(rv),
                    dtype=torch.bfloat16,
                    device='cuda')
            ST[L_INJ]['mask'] = torch.ones(
                rv.shape[0],
                dtype=torch.bool,
                device='cuda')
            ids_t = torch.tensor(
                [assembled[base_i]['ids']],
                device='cuda')
            with torch.no_grad():
                o = model(ids_t,
                          use_cache=False)
            lg = o.logits[0, -1].detach() \
                .double().cpu().numpy()
            del o
            ST[L_INJ]['repl'] = None
            ST[L_INJ]['mask'] = None
        base_i2 = base_i
        dlg = lg - LG[base_i2]
        cs.append(cosv(dlg, TT93[k]))
    dmax = float(np.max(np.abs(
        np.array(cs) - CL93[:4])))
    lines.append(
        '%s cos=%s sealed=%s maxdiff=%.3e'
        % (mode,
           ['%.6f' % c for c in cs],
           ['%.6f' % c for c in CL93[:4]],
           dmax))

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
