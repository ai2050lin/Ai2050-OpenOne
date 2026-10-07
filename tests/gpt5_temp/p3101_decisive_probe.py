# -*- coding: utf-8 -*-
"""P3101 decisive probe: why does d2b/d2 not hit bit 0 vs 3093 sealed?

Step 1: native lg for family A pairs 0..3 (base+pref rows) vs sealed
        LG (bit-level).
Step 2: repV-injected lg -> dlg = lg - LG[base]; cos(dlg, TT93[k])
        vs sealed COS_LAD_A[k] (k=0..3).
Step 3: also cos vs MY TT (recomputed) to factor out TT differences.
Step 4: environment info (transformers / torch versions).
"""
import io
import json

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
       r'\p3101_decisive_probe.txt')
MDIR = ROOT + r'\models\hf\Qwen3-14B'
NL, HID, HD, NQ = 40, 5120, 128, 40
KVW = 1024
FRONT = 4
L_INJ = 37
FKEY = 'A'
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


def log(s):
    lines.append(s)


import transformers
log('transformers=%s torch=%s cuda=%s'
    % (transformers.__version__, torch.__version__,
       torch.version.cuda))

tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda') \
    .eval()
layers = model.model.layers

V_ST = {'repl': None, 'mask': None}
V_CAP = {'rec': False, 'orig': None}


def mk_v(cap, st):
    def h(module, inp, out):
        if cap['rec']:
            cap['orig'] = out[0] \
                .detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


layers[L_INJ].self_attn.v_proj \
    .register_forward_hook(mk_v(V_CAP, V_ST))

z93 = np.load(P93, allow_pickle=False)
LG93 = z93['LG_' + FKEY].astype(np.float64)
TT93 = z93['TT_' + FKEY].astype(np.float64)
CL93 = z93['COS_LAD_' + FKEY] \
    .astype(np.float64)
log('sealed LG%s TT%s COS_LAD n=%d'
    % (LG93.shape, TT93.shape, CL93.size))

# build 32 prompts exactly like 3093
TARGETS = None
import re
body_toks = []
for b in BODIES_A:
    ids = tok(b, add_special_tokens=False)[
        'input_ids']
    body_toks.append([int(x) for x in ids])
assembled = []
for bi in range(8):
    for ci in range(4):
        s = (PREFIXES[ci] + ' ' + BODIES_A[bi]) \
            if PREFIXES[ci] else BODIES_A[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
                'input_ids']]
        assembled.append(
            {'ids': ids, 'cond': ci, 'body': bi})
idx_of = {(a['cond'], a['body']): i
          for i, a in enumerate(assembled)}
off = {}
for a in assembled:
    if a['cond'] == 0:
        off[(a['cond'], a['body'])] = 0
    else:
        bid = assembled[idx_of[(0,
            a['body'])]]['ids']
        off[(a['cond'], a['body'])] = \
            len(a['ids']) - len(bid)

# step 1: native lg for pairs 0..3 rows
rows = {}
for k in range(4):
    b_ = k % 8
    for cc in (0, 1):
        i = idx_of[(cc, b_)]
        if i in rows:
            continue
        ids_t = torch.tensor(
            [assembled[i]['ids']],
            device='cuda')
        with torch.no_grad():
            o = model(ids_t, use_cache=False)
        rows[i] = o.logits[0, -1].detach() \
            .float().cpu().numpy().astype(
                np.float64)
        del o
lg_mine = {}
for k in range(4):
    b_ = k % 8
    lg_mine[k] = (rows[idx_of[(1, b_)]],
                  rows[idx_of[(0, b_)]])
for k in range(4):
    cr = idx_of[(1, k % 8)]
    br = idx_of[(0, k % 8)]
    d_lg = float(np.max(np.abs(
        rows[cr] - LG93[idx_of[(1, k % 8)]])))
    d_lgb = float(np.max(np.abs(
        rows[br] - LG93[idx_of[(0, k % 8)]])))
    log('pair%d native lg pref diff=%.3e '
        'base diff=%.3e'
        % (k, d_lg, d_lgb))

# step 2: repV injection with MY captured V
# (capture V rows for pairs 0..3 rows)
vbank = {}
for i in sorted(rows.keys()):
    pass
for k in range(4):
    b_ = k % 8
    for cc in (0, 1):
        i = idx_of[(cc, b_)]
        if i in vbank:
            continue
        ids_t = torch.tensor(
            [assembled[i]['ids']],
            device='cuda')
        V_CAP['rec'] = True
        with torch.no_grad():
            o = model(ids_t, use_cache=False)
        V_CAP['rec'] = False
        vbank[i] = V_CAP['orig'].float() \
            .cpu().numpy()
        V_CAP['orig'] = None
        del o
cs_mine = np.zeros(4)
for k in range(4):
    b_ = k % 8
    br = idx_of[(0, b_)]
    pr = idx_of[(1, b_)]
    nb = len(assembled[br]['ids'])
    o_ = off[(1, b_)]
    rv = vbank[br][:nb, :].copy()
    rv[:FRONT, :] = vbank[pr][o_ + np.arange(
        FRONT), :]
    V_ST['repl'] = torch.tensor(
        np.ascontiguousarray(rv),
        dtype=torch.bfloat16, device='cuda')
    V_ST['mask'] = torch.ones(
        nb, dtype=torch.bool, device='cuda')
    ids_t = torch.tensor(
        [assembled[br]['ids']], device='cuda')
    with torch.no_grad():
        o = model(ids_t, use_cache=False)
    lgi = o.logits[0, -1].detach().float() \
        .cpu().numpy().astype(np.float64)
    del o
    V_ST['repl'] = None
    V_ST['mask'] = None
    dlg = lgi - rows[br]
    ttm = (rows[pr] - rows[br])
    a1 = float(dlg @ TT93[k]) / (
        np.linalg.norm(dlg)
        * np.linalg.norm(TT93[k]))
    a2 = float(dlg @ ttm) / (
        np.linalg.norm(dlg)
        * np.linalg.norm(ttm))
    cs_mine[k] = a1
    log('pair%d cos_vs_TT93=%.6f '
        'cos_vs_TTmine=%.6f sealed=%.6f '
        'd=%.2e dTT=%.2e'
        % (k, a1, a2, CL93[k],
           abs(a1 - CL93[k]),
           float(np.max(np.abs(
               ttm - TT93[k])))))

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
