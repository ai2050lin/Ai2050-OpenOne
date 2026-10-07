# -*- coding: utf-8 -*-
"""Phase 3104 (Omega-P102): Relation encoding vs endpoint shortcut.

T2 main line (per 3102 plan / review S8 task B). 4B pilot.

DESIGN (pre-registered, frozen before any model observation):
  Synthetic fact network: 8 relations x 28 entities, out-degree 3
  per (relation, subject) -> 672 true triples.  PAIR-UNIQUENESS: each
  entity pair (s,o) is true under at most one relation.

  Prompt (raw completion, 2754-style):
    Facts: The A r1 B. The B r2 C. ... Query: The S R O. Is this
    query true? Answer:
  8 fact lines; line k (k ~ U{0..7}, frozen per pair) is the CRITICAL
  fact.  For a pair q=(s,r,o):
    true   condition: critical fact = (s,r,o);  facts D + q
    false_i condition: critical fact = (s,ri,o); facts D + (s,ri,o)
  D (7 distractors) IDENTICAL across the pair's 3 conditions:
    d_rel  : edge under r, pair != (s,o)   [relation-distractor]
    d_subj : edge (s,r2,o2), r2 != r       [subject-distractor]
    5x neutral edges (relations round-robin)
  => true vs false differ in EXACTLY 2 predicate tokens (critical
  fact predicate + query predicate).  Surface heuristics:
    - "query relation token appears in facts": fires on BOTH (d_rel)
    - "query endpoint pair appears in facts":  fires on BOTH (line k)
  Only joint (s,o,relation) binding separates true from false.

MEASUREMENTS (task #47 spec):
  T1: decode the CRITICAL FACT's asserted relation (8-way) from
      internal states.  Contextual positions: pos0, crit_pred,
      crit_obj, query_pred, query_obj, last; hs slots [4,8,12,16,
      20,24,28,32] + final-norm(hs[36]).  Baselines: endpoint_concat
      (E_s;E_o input embeddings), additive (E_s+E_o), bow (mean of
      all input embeddings), pos0 curve.
  T2: truth probe (binary) on same states; zero-param yes/no margin
      m = (W_yes - W_no)^T h_finalnorm(last).  Evaluated on
      ENDPOINT-MATCHED test pairs (every pair has 1 true + 2 false
      conditions by construction) + chain-completion decoys
      (surface-familiar falses: (a,r,c) with facts containing
      (a,r,b),(b,r,c)).
  Splits (pair-blocked, seed-frozen): 4 held-out entities -> TEST-E
  (new entities); remaining pairs 75/10/10 train/val/test.

ANCHORS:
  A1 determinism: 3 prompts x 2 runs bitwise identical captures.
  A2 material seal sha256 recorded before capture.
  A3 split disjointness (pair-level).
  A4 balance: 1 true : 2 false per pair; relation marginals logged.
  A5 predicates single-token (Qwen3 tokenizer).
  A6 token-multiset diff(true, false_i) == exactly {r:2, ri:-2}
     for 20 sampled pairs.
  A7 label-permutation probe sanity (T1 -> chance).

GATES (pre-registered):
  H_K1 fact_relation_beyond_endpoint:
    best contextual T1 acc on TEST >= 0.80 AND
    >= max(endpoint, additive, best pos0) + 0.30.
  H_K2 frozen_extrapolation:
    frozen best T1 acc on TEST-E >= 0.60 AND best T2 AUC on
    TEST-E >= 0.65.
  H_K3 truth_beyond_surface:
    best T2 probe AUC on TEST >= 0.80 AND pair-strict m sign
    consistency on TEST >= 0.65 AND endpoint truth AUC on
    TEST <= 0.60.
  Verdict: 3/3 relation_bound; H_K3 fail ->
  endpoint_shortcut_dominant_candidate; 2/3 relation_partial;
  else inconclusive.

SMOKE=1: 18 entities, outdeg 2, capped pairs, layers [8,16,24,32],
4 decoys -> outputs under .../smoke/.

Output dir: tests/glm5/result/rdc_query_construction_20260913/
            phase3104/omega_p102_relation_vs_endpoint/
Model: models/hf/qwen3-4b (BF16, eager, batch 1, no cache).
"""
import gc
import hashlib
import io
import json
import os
import random
import time
from collections import Counter
from datetime import datetime

import numpy as np

SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
NAME = 'omega_p102_relation_vs_endpoint'
OUT = os.path.join(R13, 'phase3104', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31040
N_REL = 8
N_ENT = 18 if SMOKE else 28
OUTDEG = 2 if SMOKE else 3
N_DECOY_REL = 4
N_DECOY_PER = 1 if SMOKE else 12
PAIR_CAP = 12 if SMOKE else None
TE_CAP = 6 if SMOKE else None
LAYERS = [8, 16, 24, 32] if SMOKE else [4, 8, 12, 16, 20,
                                        24, 28, 32]
CHUNK = 24 if SMOKE else 96
LAMBDAS = [0.01, 0.1, 1.0]
N_BOOT = 300 if SMOKE else 2000
DECOY_RELS = [0, 1, 2, 3]

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']
FN_IDX = len(LAYERS)          # slot index of final-norm
SLOTS = len(LAYERS) + 1

ENTITIES = [
    'kapira', 'molune', 'tavoka', 'birnep', 'sulora',
    'vendik', 'pomatu', 'rilsev', 'nadoka', 'ferumi',
    'gelbap', 'towina', 'zasker', 'hipuna', 'calmir',
    'dovena', 'jumtal', 'kervis', 'lomapa', 'synder',
    'vexola', 'bruven', 'timsel', 'garona', 'pelvir',
    'mondek', 'fushia', 'wrenop'][:N_ENT]
PRED_CANDIDATES = ['owns', 'sells', 'avoids', 'follows',
                   'repairs', 'ignores', 'imitates',
                   'warns', 'admires', 'replaces',
                   'greets', 'mocks', 'envies', 'trains',
                   'copies', 'shields']

LOGS = []


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    LOGS.append(line)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:8]


log('Phase 3104 Omega-P102 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 1. Material build (tokenizer access only, no forward)
# ================================================================
rng = random.Random(SEED)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(MDIR)
PREDS = []
for w in PRED_CANDIDATES:
    ids = tok.encode(' ' + w, add_special_tokens=False)
    if len(ids) == 1:
        PREDS.append(w)
    if len(PREDS) == N_REL:
        break
assert len(PREDS) == N_REL, PREDS
pred_ids = [tok.encode(' ' + w, add_special_tokens=False)[0]
            for w in PREDS]
yes_ids = tok.encode(' yes', add_special_tokens=False)
no_ids = tok.encode(' no', add_special_tokens=False)
assert len(yes_ids) == 1 and len(no_ids) == 1, (yes_ids,
                                                no_ids)
YES_ID, NO_ID = yes_ids[0], no_ids[0]
log('A5 predicates single-token: %s' % PREDS)
log('yes_id=%d no_id=%d' % (YES_ID, NO_ID))

ents = ENTITIES
NE = len(ents)
assert N_REL * NE * OUTDEG <= NE * (NE - 1), 'capacity'

# --- graph with pair-uniqueness
edges = {r: set() for r in range(N_REL)}
pair2rel = {}
used_pairs = set()
for r in range(N_REL):
    for s in range(NE):
        got = 0
        tries = 0
        while got < OUTDEG:
            tries += 1
            assert tries < 4000, 'graph build stuck'
            o = rng.randrange(NE)
            if o == s or (s, o) in used_pairs:
                continue
            used_pairs.add((s, o))
            edges[r].add((s, o))
            pair2rel[(s, o)] = r
            got += 1
assert len(used_pairs) == N_REL * NE * OUTDEG
ALL_EDGE_PAIRS = sorted(used_pairs)
log('graph: %d true triples, %d pairs, %d unused'
    % (N_REL * NE * OUTDEG, len(used_pairs),
       NE * (NE - 1) - len(used_pairs)))

# --- chain-completion decoys
decoys = []
for r in DECOY_RELS[:N_DECOY_REL]:
    e = sorted(edges[r])
    cands = []
    for (a, b) in e:
        for (b2, c) in e:
            if b2 == b and a != c \
                    and (a, c) not in edges[r]:
                cands.append((a, b, c))
    rng.shuffle(cands)
    for (a, b, c) in cands[:N_DECOY_PER]:
        decoys.append({'rel': r, 'a': a, 'b': b,
                       'c': c})
log('decoys: %d chain completions' % len(decoys))

# --- splits (pair-blocked)
pairs_all = ALL_EDGE_PAIRS[:]
rng.shuffle(pairs_all)
e4 = rng.sample(range(NE), max(2, NE // 7))
testE_pairs = [p for p in pairs_all
               if p[0] in e4 or p[1] in e4]
rest = [p for p in pairs_all
        if p not in set(testE_pairs)]
if TE_CAP:
    testE_pairs = testE_pairs[:TE_CAP]
if PAIR_CAP:
    rest = rest[:PAIR_CAP]
n_rest = len(rest)
n_tr = int(n_rest * 0.75)
n_va = max(1, int(n_rest * 0.10))
train_pairs = rest[:n_tr]
val_pairs = rest[n_tr:n_tr + n_va]
test_pairs = rest[n_tr + n_va:]
assert not (set(train_pairs) & set(val_pairs))
assert not (set(train_pairs) & set(test_pairs))
assert not (set(val_pairs) & set(test_pairs))
assert not (set(test_pairs) & set(testE_pairs))
log('A3 splits disjoint: train=%d val=%d test=%d '
    'testE=%d (held-out entities=%s)'
    % (len(train_pairs), len(val_pairs),
       len(test_pairs), len(testE_pairs), e4))

# --- distractor composition per pair (frozen)


def sample_distractors(s, o, r):
    out = []
    for _ in range(200):
        (s1, o1) = rng.choice(ALL_EDGE_PAIRS)
        if pair2rel[(s1, o1)] == r and (s1, o1) != (s, o):
            out.append((s1, r, o1))
            break
    else:
        raise AssertionError('d_rel fail')
    for _ in range(200):
        r2 = rng.randrange(N_REL)
        if r2 == r:
            continue
        cand = [(s2, o2)
                for (s2, o2) in sorted(edges[r2])
                if s2 == s and (s2, o2) != (s, o)]
        if cand:
            (s2, o2) = rng.choice(cand)
            out.append((s, r2, o2))
            break
    else:
        raise AssertionError('d_subj fail')
    seen = {(s, o), (out[0][0], out[0][2]),
            (out[1][0], out[1][2])}
    rr = rng.randrange(N_REL)
    added = 0
    guard = 0
    while added < 5:
        guard += 1
        assert guard < 4000
        r3 = (rr + added * 3 + guard) % N_REL
        (s3, o3) = rng.choice(ALL_EDGE_PAIRS)
        if pair2rel[(s3, o3)] != r3 or (s3, o3) in seen:
            continue
        seen.add((s3, o3))
        out.append((s3, r3, o3))
        added += 1
    return out


distractors = {}
kline = {}
false_rels = {}
for p in ALL_EDGE_PAIRS:
    (s, o) = p
    r = pair2rel[p]
    distractors[p] = sample_distractors(s, o, r)
    kline[p] = rng.randrange(8)
    others = [x for x in range(N_REL) if x != r]
    rng.shuffle(others)
    false_rels[p] = (others[0], others[1])


def line_spans(pos, ls, lr, lo):
    """char spans of (subject, predicate, object) for a fact
    line starting at char offset pos."""
    a_s = pos + len('The ')
    b_s = a_s + len(ents[ls])
    a_r = b_s + len(' ')
    b_r = a_r + len(PREDS[lr])
    a_o = b_r + len(' the ')
    b_o = a_o + len(ents[lo])
    return (a_s, b_s), (a_r, b_r), (a_o, b_o)


def build_prompt(s, o, crit_rel):
    """Returns (text, spans)."""
    D = distractors[(s, o)]
    lines = [(s, crit_rel, o)] + list(D)
    rng2 = random.Random(hash((s, o, 'ord'))
                         & 0xffffffff)
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    k = kline[(s, o)]
    ci = lines.index((s, crit_rel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    spans = {}
    pos = len(text)
    for li, (ls, lr, lo) in enumerate(lines):
        seg = ' The %s %s the %s.' % (ents[ls],
                                      PREDS[lr],
                                      ents[lo])
        (ss, sr, so) = line_spans(pos + 1, ls, lr, lo)
        text += seg
        pos += len(seg)
        if li == k:
            spans['crit_pred'] = sr
            spans['crit_obj'] = so
    qseg = (' Query: The %s %s the %s. Is this query '
            'true? Answer:' % (ents[s], PREDS[crit_rel],
                               ents[o]))
    text += qseg
    qs = text.rindex('The %s %s the %s.'
                     % (ents[s], PREDS[crit_rel],
                        ents[o]))
    (ss, sr, so) = line_spans(qs, s, crit_rel, o)
    spans['query_subj'] = ss
    spans['query_pred'] = sr
    spans['query_obj'] = so
    return text, spans


records = []
for split, plist in (('train', train_pairs),
                     ('val', val_pairs),
                     ('test', test_pairs),
                     ('testE', testE_pairs)):
    for (s, o) in plist:
        r = pair2rel[(s, o)]
        (ri1, ri2) = false_rels[(s, o)]
        for cond, crel, lab in (('true', r, 1),
                                ('false1', ri1, 0),
                                ('false2', ri2, 0)):
            text, spans = build_prompt(s, o, crel)
            records.append({
                'id': 'p%d' % len(records),
                'split': split, 'pair': [s, o],
                'cond': cond, 'crit_rel': crel,
                'truth': lab, 'k': kline[(s, o)],
                'text': text, 'spans': spans,
                'tag': 'main'})
for di, d in enumerate(decoys):
    (a, b, c, r) = (d['a'], d['b'], d['c'], d['rel'])
    D = []
    seen = {(a, c), (a, b), (b, c)}
    guard = 0
    while len(D) < 5:
        guard += 1
        assert guard < 4000
        r3 = rng.randrange(N_REL)
        (s3, o3) = rng.choice(ALL_EDGE_PAIRS)
        if pair2rel[(s3, o3)] != r3 or (s3, o3) in seen:
            continue
        seen.add((s3, o3))
        D.append((s3, r3, o3))
    lines = [(a, r, b), (b, r, c)] + D
    rng2 = random.Random(hash(('decoy', di))
                         & 0xffffffff)
    order = list(range(7))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    text = 'Facts:'
    spans = {}
    pos = len(text)
    for (ls, lr, lo) in lines:
        seg = ' The %s %s the %s.' % (ents[ls],
                                      PREDS[lr],
                                      ents[lo])
        (ss, sr, so) = line_spans(pos + 1, ls, lr, lo)
        text += seg
        pos += len(seg)
        if (ls, lr, lo) == (a, r, b):
            spans['crit_pred'] = sr
            spans['crit_obj'] = so
    qseg = (' Query: The %s %s the %s. Is this query '
            'true? Answer:' % (ents[a], PREDS[r],
                               ents[c]))
    text += qseg
    qs = text.rindex('The %s %s the %s.'
                     % (ents[a], PREDS[r], ents[c]))
    (ss, sr, so) = line_spans(qs, a, r, c)
    spans['query_subj'] = ss
    spans['query_pred'] = sr
    spans['query_obj'] = so
    records.append({
        'id': 'p%d' % len(records), 'split': 'decoy',
        'pair': [a, c], 'cond': 'decoy',
        'crit_rel': r, 'truth': 0, 'k': -1,
        'text': text, 'spans': spans, 'tag': 'decoy'})

# --- A4 balance + A6 multiset check
n_true = sum(1 for x in records
             if x['truth'] == 1)
n_false = sum(1 for x in records
              if x['truth'] == 0 and x['tag'] == 'main')
assert n_true * 2 == n_false, (n_true, n_false)
crit_rel_counts = Counter(x['crit_rel'] for x in records
                          if x['tag'] == 'main')
log('A4 balance: true=%d false=%d crit_rel_marginal=%s'
    % (n_true, n_false,
       dict(sorted(crit_rel_counts.items()))))
conds_by_pair = {}
for x in records:
    if x['tag'] == 'main':
        conds_by_pair.setdefault(
            tuple(x['pair']), {})[x['cond']] = x
full = [c for c in conds_by_pair.values()
        if len(c) == 3]
rng_chk = random.Random(SEED + 7)
rng_chk.shuffle(full)
n_chk = 0
for conds in full[:20]:
    tt = Counter(tok.encode(conds['true']['text'],
                            add_special_tokens=False))
    tf = Counter(tok.encode(conds['false1']['text'],
                            add_special_tokens=False))
    rt = pred_ids[conds['true']['crit_rel']]
    rf = pred_ids[conds['false1']['crit_rel']]
    assert (tt - tf) == Counter({rt: 2}), (tt - tf, rt)
    assert (tf - tt) == Counter({rf: 2}), (tf - tt, rf)
    n_chk += 1
log('A6 multiset check passed on %d pairs (diff exactly '
    '2x predicate token)' % n_chk)

mat_path = os.path.join(OUT, 'material.json')
with io.open(mat_path, 'w', encoding='utf-8') as f:
    json.dump({
        'seed': SEED, 'entities': ents,
        'predicates': PREDS, 'pred_ids': pred_ids,
        'yes_id': YES_ID, 'no_id': NO_ID,
        'edges': {str(r): sorted(edges[r])
                  for r in range(N_REL)},
        'pair2rel': {'%d_%d' % p: pair2rel[p]
                     for p in pair2rel},
        'distractors': {'%d_%d' % p: distractors[p]
                        for p in distractors},
        'kline': {'%d_%d' % p: kline[p]
                  for p in kline},
        'false_rels': {'%d_%d' % p: false_rels[p]
                       for p in false_rels},
        'splits': {
            'train': [list(p) for p in train_pairs],
            'val': [list(p) for p in val_pairs],
            'test': [list(p) for p in test_pairs],
            'testE': [list(p) for p in testE_pairs]},
        'heldout_entities': e4, 'decoys': decoys,
        'smoke': SMOKE}, f, ensure_ascii=False,
        indent=1)
MAT_SHA = sha8(mat_path)
log('A2 material seal sha8=%s records=%d'
    % (MAT_SHA, len(records)))

design = {
    'gates': {
        'H_K1': 'T1 contextual best TEST acc >= 0.80 AND '
                '>= baselines(endpoint,additive,pos0)'
                '+0.30',
        'H_K2': 'frozen T1 TESTE acc >= 0.60 AND T2 '
                'TESTE AUC >= 0.65',
        'H_K3': 'T2 probe TEST AUC >= 0.80 AND pair-'
                'strict m sign >= 0.65 AND endpoint '
                'AUC TEST <= 0.60'},
    'verdict_map': {
        '3/3': 'relation_bound',
        'H_K3 fail': 'endpoint_shortcut_dominant_'
                     'candidate',
        '2/3': 'relation_partial',
        'else': 'inconclusive'},
    'material_sha8': MAT_SHA,
    'layers_hs_index': LAYERS + ['fn(hs[36])'],
    'positions': POSITIONS, 'lambdas': LAMBDAS,
    'n_boot': N_BOOT,
    'feature_selection': 'argmax VAL; tie -> lower '
                         'slot, smaller lambda',
}
with io.open(os.path.join(OUT, 'design_seal.json'), 'w',
             encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-observation)')

# ================================================================
# 2. Capture
# ================================================================
import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
NL = len(model.model.layers)
HID = int(model.config.hidden_size)
assert NL == 36 and HID == 2560, (NL, HID)
norm_mod = model.model.norm
lm_head = model.lm_head
WU = model.lm_head.weight.detach()
w_yes = WU[YES_ID].float().cpu()
w_no = WU[NO_ID].float().cpu()
w_dn = (w_yes - w_no).numpy()
EMB = model.get_input_embeddings().weight.detach()
log('model loaded NL=%d HID=%d' % (NL, HID))

CAP_L = LAYERS + [NL]


def encode_record(rec):
    enc = tok(rec['text'], return_offsets_mapping=True,
              add_special_tokens=False)
    ids = enc['input_ids']
    offs = enc['offset_mapping']
    tok_pos = {}

    def char2tok_last(a, b):
        hit = -1
        for i, (oa, ob) in enumerate(offs):
            if oa < b and ob > a:
                hit = i
        assert hit >= 0, (a, b, rec['id'])
        return hit

    for name in ('query_subj', 'crit_pred', 'crit_obj',
                 'query_pred', 'query_obj'):
        (a, b) = rec['spans'][name]
        tok_pos[name] = char2tok_last(a, b)
    tok_pos['pos0'] = 0
    tok_pos['last'] = len(ids) - 1
    return ids, tok_pos


def capture2(rec):
    ids, tp = encode_record(rec)
    t = torch.tensor([ids], device='cuda')
    caps = {}
    with torch.inference_mode():
        out = model(t, output_hidden_states=True,
                    use_cache=False)
        hs = out.hidden_states
        for pi, pname in enumerate(POSITIONS):
            ti = tp[pname]
            for li, lidx in enumerate(CAP_L):
                if lidx < NL:
                    v = hs[lidx][0, ti, :]
                else:
                    v = norm_mod(hs[NL][0, ti, :])
                caps[(pi, li)] = \
                    v.float().cpu().numpy()
        hfn = norm_mod(hs[NL][0, tp['last'], :]).float()
        m = float((hfn.cpu().numpy() @ w_dn).item())
        logits_last = lm_head(
            norm_mod(hs[NL][0, tp['last'], :])
            .unsqueeze(0))
        nxt = int(logits_last.argmax(-1).item())
        del out, hs
    e_s = EMB[ids[tp['query_subj']]].float() \
        .cpu().numpy()
    e_o = EMB[ids[tp['query_obj']]].float() \
        .cpu().numpy()
    bow = EMB[torch.tensor(ids)].float() \
        .mean(0).cpu().numpy()
    return {'caps': caps, 'm': m, 'next': nxt,
            'e_s': e_s, 'e_o': e_o, 'bow': bow,
            'n_tok': len(ids)}


# --- A1 determinism anchor
anchor_ids = [0, len(records) // 2, len(records) - 1]
d_a1 = 0.0
for ri in anchor_ids:
    r1 = capture2(records[ri])
    r2 = capture2(records[ri])
    for key, v in r1['caps'].items():
        d_a1 = max(d_a1, float(np.abs(
            v - r2['caps'][key]).max()))
    d_a1 = max(d_a1, abs(r1['m'] - r2['m']))
log('A1 determinism: max diff = %.3e' % d_a1)
assert d_a1 == 0.0, 'determinism anchor failed'

# --- capture loop (chunked, partial saves)
N = len(records)
X_all = np.zeros((N, len(POSITIONS), SLOTS, HID),
                 dtype=np.float16)
M_all = np.zeros(N, dtype=np.float32)
NEXT_all = np.zeros(N, dtype=np.int32)
ES_all = np.zeros((N, HID), dtype=np.float16)
EO_all = np.zeros((N, HID), dtype=np.float16)
BOW_all = np.zeros((N, HID), dtype=np.float16)
t_cap = time.time()
for i0 in range(0, N, CHUNK):
    i1 = min(i0 + CHUNK, N)
    for i in range(i0, i1):
        res = capture2(records[i])
        for (pi, li), v in res['caps'].items():
            X_all[i, pi, li, :] = v.astype(np.float16)
        M_all[i] = res['m']
        NEXT_all[i] = res['next']
        ES_all[i] = res['e_s'].astype(np.float16)
        EO_all[i] = res['e_o'].astype(np.float16)
        BOW_all[i] = res['bow'].astype(np.float16)
    np.savez(os.path.join(OUT, 'capture_part%03d.npz'
                          % (i0 // CHUNK)),
             X=X_all[i0:i1], m=M_all[i0:i1],
             next_tok=NEXT_all[i0:i1],
             e_s=ES_all[i0:i1], e_o=EO_all[i0:i1],
             bow=BOW_all[i0:i1])
    log('capture %d/%d' % (i1, N))
np.savez_compressed(
    os.path.join(OUT, 'capture.npz'), X=X_all, m=M_all,
    next_tok=NEXT_all, e_s=ES_all, e_o=EO_all,
    bow=BOW_all,
    ids=np.array([r['id'] for r in records]),
    split=np.array([r['split'] for r in records]),
    cond=np.array([r['cond'] for r in records]),
    truth=np.array([r['truth'] for r in records]),
    crit_rel=np.array([r['crit_rel'] for r in records]),
    tag=np.array([r['tag'] for r in records]),
    n_tok=np.array([r['text'].count(' ')
                    for r in records]))
log('capture saved (%.1fs) X=%s'
    % (time.time() - t_cap, str(X_all.shape)))
gc.collect()
torch.cuda.empty_cache()

# ================================================================
# 3. Probes
# ================================================================
split = np.array([r['split'] for r in records])
truth = np.array([r['truth'] for r in records])
crel = np.array([r['crit_rel'] for r in records])
tag = np.array([r['tag'] for r in records])
pair_key = np.array(['%d_%d' % tuple(r['pair'])
                     for r in records])

tr = np.where((split == 'train') & (tag == 'main'))[0]
va = np.where((split == 'val') & (tag == 'main'))[0]
te = np.where((split == 'test') & (tag == 'main'))[0]
teE = np.where((split == 'testE') & (tag == 'main'))[0]
de = np.where(tag == 'decoy')[0]
log('probe sets: train=%d val=%d test=%d testE=%d '
    'decoy=%d' % (len(tr), len(va), len(te), len(teE),
                  len(de)))
assert len(tr) and len(va) and len(te) and len(teE)

XT1 = crel.astype(np.int64)
YT1 = np.eye(N_REL, dtype=np.float32)[XT1]
YT2 = (truth * 2.0 - 1.0).astype(np.float32)

Xs = {}


def get_X(key):
    if key in Xs:
        return Xs[key]
    if key[0] == 'ctx':
        (_, pname, l) = key
        p = POSITIONS.index(pname)
        X = X_all[:, p, l, :].astype(np.float32)
    elif key[0] == 'endpoint':
        X = np.concatenate([ES_all, EO_all],
                           1).astype(np.float32)
    elif key[0] == 'additive':
        X = (ES_all.astype(np.float32)
             + EO_all.astype(np.float32))
    elif key[0] == 'bow':
        X = BOW_all.astype(np.float32)
    Xs[key] = X
    return X


def fit_scaler(Xtr):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    return mu, sd


def ridge_solve(X, Y, lam):
    d = X.shape[1]
    A = ((X.T @ X) / X.shape[0]
         + lam * np.eye(d, dtype=np.float32))
    B = (X.T @ Y) / X.shape[0]
    return np.linalg.solve(A, B).astype(np.float32)


def auc_score(y, s):
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    srt = s[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) \
                and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = \
                (i + 1 + j + 1) / 2.0
        i = j + 1
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    return float((ranks[y == 1].sum()
                  - n1 * (n1 + 1) / 2.0)
                 / (n1 * n0))


def eval_probe(target, key):
    X = get_X(key)
    mu, sd = fit_scaler(X[tr])
    Z = (X - mu) / sd
    Y = YT1 if target == 'T1' \
        else YT2.reshape(-1, 1)
    best = None
    for lam_i, lam in enumerate(LAMBDAS):
        W = ridge_solve(Z[tr], Y[tr], lam)
        S = Z @ W
        if target == 'T1':
            pv = float((S[va].argmax(1)
                        == XT1[va]).mean())
        else:
            pv = auc_score(truth[va], S[va, 0])
        if best is None or pv > best[0] + 1e-12:
            best = (pv, lam_i, lam, W, S)
        elif abs(pv - best[0]) <= 1e-12 \
                and lam_i < best[1]:
            best = (pv, lam_i, lam, W, S)
    (pv, _, lam, W, S) = best
    out = {'val': pv, 'lam': lam,
           'config': '%s|%s' % (key[0], key[1])
           if len(key) > 1 else key[0]}
    if target == 'T1':
        out['acc_test'] = float(
            (S[te].argmax(1) == XT1[te]).mean())
        out['acc_testE'] = float(
            (S[teE].argmax(1) == XT1[teE]).mean())
        out['pred_test'] = S[te].argmax(1).tolist()
    else:
        out['auc_test'] = auc_score(truth[te],
                                    S[te, 0])
        out['auc_testE'] = auc_score(truth[teE],
                                     S[teE, 0])
        out['score_test'] = S[te, 0].tolist()
        out['score_decoy'] = S[de, 0].tolist()
    return out


t_probe = time.time()
T1_CTX = {}
for pname in POSITIONS:
    for li in range(SLOTS):
        T1_CTX['%s|L%d' % (pname, li)] = eval_probe(
            'T1', ('ctx', pname, li))
T1_BASE = {}
for nm, key in (('endpoint_concat', ('endpoint',)),
                ('additive', ('additive',)),
                ('bow', ('bow',))):
    T1_BASE[nm] = eval_probe('T1', key)
log('T1 sweep done (%.1fs)' % (time.time() - t_probe))

t_probe = time.time()
T2_CTX = {}
for pname in POSITIONS:
    for li in range(SLOTS):
        T2_CTX['%s|L%d' % (pname, li)] = eval_probe(
            'T2', ('ctx', pname, li))
T2_BASE = {}
for nm, key in (('endpoint_concat', ('endpoint',)),
                ('additive', ('additive',)),
                ('bow', ('bow',))):
    T2_BASE[nm] = eval_probe('T2', key)
log('T2 sweep done (%.1fs)' % (time.time() - t_probe))

# --- zero-param m margin
m_main = M_all
auc_m_test = auc_score(truth[te], -m_main[te])
auc_m_testE = auc_score(truth[teE], -m_main[teE])
pair_strict = []
for p in sorted(set(pair_key[te])):
    idx = np.where((pair_key == p)
                   & (tag == 'main'))[0]
    mt = m_main[idx[truth[idx] == 1]]
    mf = m_main[idx[truth[idx] == 0]]
    if len(mt) and len(mf):
        pair_strict.append(bool(mt[0] > mf.max()))
pair_strict_rate = float(np.mean(pair_strict))
mean_m = {
    'true': float(m_main[(tag == 'main')
                         & (truth == 1)].mean()),
    'false': float(m_main[(tag == 'main')
                          & (truth == 0)].mean()),
    'decoy': float(m_main[de].mean())}
log('m: AUC test=%.4f testE=%.4f pair_strict=%.4f '
    'mean_m=%s' % (auc_m_test, auc_m_testE,
                   pair_strict_rate,
                   json.dumps(mean_m)))

# --- A7 permutation sanity
ctx_acc = {k: v['acc_test'] for k, v in T1_CTX.items()}
best_ctx_name = max(ctx_acc, key=ctx_acc.get)
pname_b, lstr_b = best_ctx_name.split('|L')
li_b = int(lstr_b)
X = get_X(('ctx', pname_b, li_b))
mu, sd = fit_scaler(X[tr])
Z = (X - mu) / sd
lam_b = T1_CTX[best_ctx_name]['lam']
rng_np = np.random.RandomState(SEED + 1)
perm = rng_np.permutation(XT1[tr])
Wp = ridge_solve(Z[tr],
                 np.eye(N_REL, dtype=np.float32)[perm],
                 lam_b)
acc_perm = float(((Z[te] @ Wp).argmax(1)
                  == XT1[te]).mean())
log('A7 permutation sanity: acc=%.4f (config %s, '
    'lam=%g)' % (acc_perm, best_ctx_name, lam_b))

# --- decoy diagnostics (best T2 probe)
best_t2_name = max((k for k in T2_CTX),
                   key=lambda k: T2_CTX[k]['val'])
s_decoy = np.array(T2_CTX[best_t2_name]['score_decoy'])
s_test = np.array(T2_CTX[best_t2_name]['score_test'])
y_te = truth[te]
decoy_diag = {
    'config': best_t2_name,
    'score_decoy_mean': float(s_decoy.mean()),
    'score_true_mean': float(s_test[y_te == 1].mean()),
    'score_false_mean': float(s_test[y_te == 0]
                              .mean()),
    'frac_decoy_below_true_mean': float(
        (s_decoy < s_test[y_te == 1].mean()).mean())}
log('decoy: %s' % json.dumps(decoy_diag))

# --- bootstrap CIs
def boot_ci(fn, n, n_boot=N_BOOT, seed=SEED + 2):
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        v = fn(rs.randint(0, n, n))
        if v is not None:
            vals.append(v)
    if not vals:
        return [float('nan'), float('nan')]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


pred_best = np.array(
    T1_CTX[best_ctx_name]['pred_test'])
pred_ep = np.array(
    T1_BASE['endpoint_concat']['pred_test'])
ytest = XT1[te]
ci_t1 = boot_ci(lambda idx: float(
    (pred_best[idx] == ytest[idx]).mean()
    - (pred_ep[idx] == ytest[idx]).mean()), len(te))
s_best = s_test
ci_t2 = boot_ci(lambda idx: auc_score(y_te[idx],
                                      s_best[idx]),
                len(te))
pair_flags = np.array(pair_strict)
ci_m = boot_ci(lambda idx: float(
    pair_flags[idx].mean()), len(pair_flags))
log('bootstrap95: T1diff=%s T2AUC=%s m_strict=%s'
    % (ci_t1, ci_t2, ci_m))

# ================================================================
# 4. Gates & verdict
# ================================================================
pos0_best = max(v['acc_test'] for k, v in T1_CTX.items()
                if k.startswith('pos0'))
gate_baseline_max = max(
    T1_BASE['endpoint_concat']['acc_test'],
    T1_BASE['additive']['acc_test'], pos0_best)
ctx_best = ctx_acc[best_ctx_name]
H_K1 = (ctx_best >= 0.80
        and ctx_best - gate_baseline_max >= 0.30)
t2_best_auc = max(v['auc_test']
                  for v in T2_CTX.values())
t2_best_aucE = max(v['auc_testE']
                   for v in T2_CTX.values())
ep_auc = T2_BASE['endpoint_concat']['auc_test']
H_K2 = (T1_CTX[best_ctx_name]['acc_testE'] >= 0.60
        and t2_best_aucE >= 0.65)
H_K3 = (t2_best_auc >= 0.80
        and pair_strict_rate >= 0.65
        and ep_auc <= 0.60)
n_pass = sum([H_K1, H_K2, H_K3])
if n_pass == 3:
    verdict = 'relation_bound'
elif not H_K3:
    verdict = 'endpoint_shortcut_dominant_candidate'
elif n_pass == 2:
    verdict = 'relation_partial'
else:
    verdict = 'inconclusive'

results = {
    'verdict': verdict,
    'gates': {
        'H_K1': H_K1, 'H_K2': H_K2, 'H_K3': H_K3,
        'ctx_best_name': best_ctx_name,
        'ctx_best_acc_test': ctx_best,
        'gate_baseline_max': gate_baseline_max,
        'endpoint_acc_test':
            T1_BASE['endpoint_concat']['acc_test'],
        'additive_acc_test':
            T1_BASE['additive']['acc_test'],
        'bow_acc_test': T1_BASE['bow']['acc_test'],
        'pos0_best_acc_test': pos0_best,
        't1_acc_testE':
            T1_CTX[best_ctx_name]['acc_testE'],
        't2_best_auc_test': t2_best_auc,
        't2_best_auc_testE': t2_best_aucE,
        'endpoint_truth_auc_test': ep_auc,
        'm_pair_strict': pair_strict_rate,
        'm_auc_test': auc_m_test,
        'm_auc_testE': auc_m_testE},
    'm': {'auc_test': auc_m_test,
          'auc_testE': auc_m_testE,
          'pair_strict_rate': pair_strict_rate,
          'n_pairs_strict': len(pair_strict),
          'mean_m': mean_m},
    'decoy': decoy_diag,
    'sanity': {'perm_acc_test': acc_perm,
               'perm_config': best_ctx_name,
               'a1_determinism': d_a1,
               'a6_multiset_pairs': n_chk},
    'ci': {'t1_diff_vs_endpoint': ci_t1,
           't2_auc_test': ci_t2,
           'm_strict': ci_m},
    'T1_ctx': {k: {kk: vv for kk, vv in v.items()
                   if not kk.startswith('pred_')}
               for k, v in T1_CTX.items()},
    'T1_base': T1_BASE,
    'T2_ctx': {k: {kk: vv for kk, vv in v.items()
                   if not kk.startswith('score_')}
               for k, v in T2_CTX.items()},
    'T2_base': T2_BASE,
    'material_sha8': MAT_SHA, 'smoke': SMOKE,
    'n_records': N, 'n_pairs': len(used_pairs),
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S')}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1,
              ensure_ascii=False)
log('VERDICT=%s (K1=%s K2=%s K3=%s)'
    % (verdict, H_K1, H_K2, H_K3))
log('Phase 3104 done (%.1fs)' % (time.time() - T0))
print('VERDICT:', verdict)
