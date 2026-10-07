# -*- coding: utf-8 -*-
"""Phase 3076 pre-flight probe: verify anchor
files, the top8 ranking criterion, and
per-family text assembly (tokenizer only, no
model load).  Output written to a txt file
(bash shim stdout is unreliable)."""
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = os.path.join(ROOT, 'tests', 'glm5',
                    'result',
                    'rdc_query_construction_20260913')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp',
                    'phase3076_probe_out.txt')
out = []

j71 = json.load(open(os.path.join(
    BASE, 'phase3071', 'omega_p68_attn_head_decomp',
    'result.json'), encoding='utf-8'))
r34 = np.array(j71['stats']['head']['r34'],
               dtype=np.float64)
top8_71 = [int(v)
           for v in j71['stats']['head']['top8']]
my8 = [int(h) for h in np.argsort(
    -np.abs(r34), kind='stable')[:8]]
out.append('r34 len=%d' % len(r34))
out.append('top8_71=%s' % top8_71)
out.append('argsort(-|r34|)[:8]=%s' % my8)
out.append('top8 criterion match=%s'
           % (my8 == top8_71))

j73 = json.load(open(os.path.join(
    BASE, 'phase3073', 'omega_p70_head_interaction',
    'result.json'), encoding='utf-8'))
st73 = j73['stats']
r1 = np.array(st73['r1'], dtype=np.float64)
out.append('3073 stats keys=%s'
           % sorted(st73.keys()))
out.append('r1 len=%d r1[0]=%.17g'
           % (len(r1), r1[0]))
out.append('max|r1 - r34[top8]|=%.3e'
           % float(np.max(np.abs(r1
                                 - r34[top8_71]))))
out.append('r_t3=%r r_u8=%r'
           % (st73.get('r_t3'), st73.get('r_u8')))

z71 = np.load(os.path.join(
    BASE, 'phase3071', 'omega_p68_attn_head_decomp',
    'omega_p68_attn_head_decomp.npz'))
out.append('npz71 files=%s' % sorted(z71.files))
z74 = np.load(os.path.join(
    BASE, 'phase3074', 'omega_p71_capacity_law',
    'omega_p71_capacity_law.npz'))
out.append('npz74 files=%s' % sorted(z74.files))
out.append('A_S shape=%s PAIRS4 shape=%s'
           % (z74['A_S'].shape,
              z74['PAIRS4'].shape))
z75 = np.load(os.path.join(
    BASE, 'phase3075',
    'omega_p72_supermodular_structure',
    'omega_p72_supermodular_structure.npz'))
out.append('npz75 files=%s' % sorted(z75.files))
out.append('MU75 shape=%s MU75[255]=%.17g'
           % (z75['MU'].shape,
              float(z75['MU'][255])))
z66 = np.load(os.path.join(
    BASE, 'phase3066',
    'omega_p63_last_layer_flip_anatomy',
    'omega_p63_last_layer_flip_anatomy.npz'))
out.append('npz66 files=%s' % sorted(z66.files))
out.append('COS_LAD shape=%s'
           % (z66['COS_LAD'].shape,))
j74 = json.load(open(os.path.join(
    BASE, 'phase3074', 'omega_p71_capacity_law',
    'result.json'), encoding='utf-8'))
hill74 = j74['stats']['fits']['hill']
out.append('3074 hill p_le2=%r' % (hill74['p_le2'],))

from transformers import AutoTokenizer
MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen3-4b')
tok = AutoTokenizer.from_pretrained(MDIR)

FAM = {
 'A': (('The weather was cold, so',
        'He studied every night because',
        'The experiment failed, therefore',
        'He missed the train, however',
        'The garden grows quickly while',
        'The price was high, yet',
        'She speaks French, although',
        'The road was closed, thus',),
       ('so', 'because', 'therefore', 'however',
        'while', 'yet', 'although', 'thus'),
       ('', 'In a formal style,',
        'In Shakespearean style,',
        'Regarding the weather,')),
 'B': (('The solution turned acidic, so',
        'The sample was heated because',
        'The catalyst degraded, therefore',
        'The vacuum leaked, however',
        'The crystals formed while',
        'The pressure dropped, yet',
        'The alloy expanded, although',
        'The circuit overheated, thus',),
       ('so', 'because', 'therefore', 'however',
        'while', 'yet', 'although', 'thus'),
       ('', 'In a formal style,',
        'In Shakespearean style,',
        'Regarding the experiment,')),
 'C': (('She felt betrayed, so',
        'He apologized because',
        'They reconciled, therefore',
        'She stormed out, however',
        'He listened quietly while',
        'The gift was cheap, yet',
        'She forgave him, although',
        'The friendship ended, thus',),
       ('so', 'because', 'therefore', 'however',
        'while', 'yet', 'although', 'thus'),
       ('', 'In a formal style,',
        'In Shakespearean style,',
        'Regarding the conversation,')),
}
for fk, (bodies, targets, prefixes) in \
        FAM.items():
    wt = {}
    ok_all = True
    for w in targets:
        wi = tok(' ' + w,
                 add_special_tokens=False)[
            'input_ids']
        if len(wi) != 1:
            ok_all = False
            out.append('[%s] target %r NOT single '
                       'token: %r' % (fk, w, wi))
        else:
            wt[w] = int(wi[0])
    nbad = 0
    for bi, body in enumerate(bodies):
        bid = [int(x) for x in tok(
            body, add_special_tokens=False)[
            'input_ids']]
        t = wt[targets[bi]]
        for ci, pre in enumerate(prefixes):
            s = (pre + ' ' + body) if pre \
                else body
            pid = [int(x) for x in tok(
                s, add_special_tokens=False)[
                'input_ids']]
            cnt = pid.count(t)
            if cnt != 1:
                ok_all = False
                nbad += 1
                out.append('[%s] b%d c%d count(%r)'
                           '=%d' % (fk, bi, ci,
                                    targets[bi],
                                    cnt))
            if ci > 0:
                off = len(pid) - len(bid)
                if off <= 0 or list(
                        pid[off + 1:]) \
                        != list(bid[1:]):
                    ok_all = False
                    nbad += 1
                    out.append('[%s] b%d c%d off '
                               'align fail off='
                               '%d' % (fk, bi, ci,
                                       off))
                else:
                    w0b = tok.decode(
                        [bid[0]]).strip()
                    w0p = tok.decode(
                        [pid[off]]).strip()
                    if w0b != w0p:
                        ok_all = False
                        nbad += 1
                        out.append(
                            '[%s] b%d c%d w0 %r vs '
                            '%r' % (fk, bi, ci, w0b,
                                    w0p))
    lens = []
    for bi, body in enumerate(bodies):
        for ci, pre in enumerate(prefixes):
            s = (pre + ' ' + body) if pre \
                else body
            lens.append(len(tok(
                s, add_special_tokens=False)[
                'input_ids']))
    out.append('[%s] assembly ok=%s bad=%d lens '
               '%d-%d' % (fk, ok_all, nbad,
                          min(lens), max(lens)))

with open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('probe done')
