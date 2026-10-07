import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
import io
import numpy as np
from transformers import AutoTokenizer

OUT = open('D:/AI2050/Ai2050-OpenOne/tests/gpt5_temp/p3124_span_out.txt',
           'w', encoding='utf-8')


def log(s):
    OUT.write(s + '\n')
    OUT.flush()


ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D13 = os.path.join(RDIR, 'phase3113',
                   'omega_p111_artifact_writein')
MDIR = os.path.join(ROOT, 'models', 'hf',
                    'glm4-9b-chat-hf')

mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
capb = np.load(os.path.join(D13, 'capture_b.npz'),
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
p2r = mat5['pair2rel']

log('n_ent=%d n_pred=%d' % (len(ents_all),
                            len(PREDS_all)))
log('ents[:6]=%s' % ents_all[:6])
log('preds[:6]=%s' % PREDS_all[:6])

hP = {}
for i in range(len(pkB)):
    if str(condB[i]) == 'P':
        hP[str(pkB[i])] = i
pks = sorted(hP.keys())
pk = pks[0]
(s, o) = (int(v) for v in pk.split('_'))
r = p2r[pk]
log('pk=%s s=%d o=%d r=%s' % (pk, s, o, r))
log('ent_s=%r ent_o=%r pred=%r'
    % (ents_all[s], ents_all[o], PREDS_all[int(r)]))

# rebuild prompt exactly as build_prompt
mat = mat5
lrel = int(r)
D = [tuple(d) for d in
     mat['distractors']['%d_%d' % (s, o)]]
k = mat['kline']['%d_%d' % (s, o)]
lines = [(s, lrel, o)] + list(D)
import random as _rnd
import zlib
rng2 = _rnd.Random(zlib.crc32(
    ('%d_%d_ord5' % (s, o)).encode('ascii')))
order = list(range(8))
rng2.shuffle(order)
lines = [lines[i] for i in order]
ci = lines.index((s, lrel, o))
lines[ci], lines[k] = lines[k], lines[ci]
text = 'Facts:'
for (ls, lr, lo) in lines:
    text += ' The %s %s the %s.' % (
        ents_all[ls], PREDS_all[lr],
        ents_all[lo])
text += (' Query: The %s %s the %s. Is this '
         'query true? Answer:'
         % (ents_all[s], PREDS_all[int(r)],
            ents_all[o]))
log('text[:220]=%r' % text[:220])

qline = 'The %s %s the %s.' % (
    ents_all[s], PREDS_all[int(r)],
    ents_all[o])
cs = text.find(qline)
log('qline=%r' % qline)
log('find(cs)=%d count=%d'
    % (cs, text.count(qline)))

tok = AutoTokenizer.from_pretrained(
    MDIR, trust_remote_code=True)
encp = tok(text, add_special_tokens=False,
           return_offsets_mapping=True)
poffs = [tuple(v) for v in
         encp['offset_mapping']]
log('n_tokens=%d' % len(poffs))
if cs != -1:
    ce = cs + len(qline)
    idxs = [kk for kk in range(len(poffs))
            if poffs[kk][0] >= cs
            and poffs[kk][1] <= ce
            and poffs[kk][1] > poffs[kk][0]]
    log('span_idx=%d..%d n=%d'
        % (idxs[0], idxs[-1], len(idxs)))
    log('first 3 tok spans in range: %s'
        % [(kk, poffs[kk],
            text[poffs[kk][0]:poffs[kk][1]])
           for kk in idxs[:3]])
    partial = [kk for kk in range(len(poffs))
               if poffs[kk][1] > cs
               and poffs[kk][0] < ce]
    log('partial-overlap n=%d' % len(partial))
    if len(idxs) != len(partial):
        log('MISMATCH detail:')
        for kk in partial:
            if kk not in idxs:
                log('  excluded tok %d span=%s '
                    'txt=%r'
                    % (kk, poffs[kk],
                       text[poffs[kk][0]:
                            poffs[kk][1]]))
log('DONE')
OUT.close()
