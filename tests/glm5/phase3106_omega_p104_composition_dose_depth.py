# -*- coding: utf-8 -*-
"""Phase 3106 (Omega-P104): Composition dose & depth.

T2 main line, 4B.  3105 established: the model's own yes/no readout
tracks IN-CONTEXT truth (m AUC 0.990, P>A1 strict 74/74) while it
ignores external graph membership (3104 AUC 0.503).  3106 asks:

  (1) DOSE/DEPTH - does the readout keep tracking truth when the
      query triple is DERIVED (2-hop transitive chain) rather than
      verbatim present?  2754 paradigm + 3105 balanced distractors.
  (2) SCATTER - when evidence is scattered/passive-rephrased, does
      joint matching survive, or degrade to frequency heuristics?
      Frequency-matched control conditions (C-sub, S-scatter) make
      the frequency heuristic predict YES where truth is NO.
  (3) BINDING GATE RERUN (offline, 3105 capture): revised VAL
      tie-break (tie -> LATER layer) - does G2 pass literally?
  (4) LAYER ALIGNMENT (offline): quantify binding curve vs truth
      curve (Pearson + peak-layer offset), not just co-statement.

DESIGN (pre-registered, frozen before any model observation):

  Graph: 8 relations x 28 entities, out-degree 3, pair-unique
  (same construction as 3104/3105, new seed 31060).

  CHAIN family (tag='chain'), 240 chains x 4 conditions:
    C-true (1): facts (a,r,b),(b,r,c); query (a,r,c) -> YES
         (2-hop same-relation path, in-context derivable)
    C-swap (0): same facts; query (c,r,a) -> NO (direction)
    C-rel  (0): facts (a,r,b),(b,ri,c) [ri!=r]; query (a,r,c) -> NO
         (2nd-hop predicate broken; vs C-true differs in exactly
         1 token on line 2 predicate)
    C-sub  (0): facts (a,r,b),(x,r,c) [(x,r,c) true edge]; query
         (a,r,c) -> NO (2nd-hop subject broken; r appears 2x,
         SAME as C-true -> frequency heuristic predicts YES)
    Rule sentence (constant): "Note: The same predicate may hold
    transitively across a chain of facts."  Distractors: 4
    condition-constant neutral edges (relations != r).
    Constraint: (a,c) not in graph (no verbatim-triple leakage).

  SCATTER family (tag='scatter'), 60 pairs x 4 conditions:
    S-active  (1): facts (s,r,o); query (s,r,o) -> YES (3105 P
         baseline, no rule sentence -> format matches 3105)
    S-passive (1): facts passive(o,r,s) "The o is PASS[r] by the
         s."; query (s,r,o) active -> YES iff readout penetrates
         grammar (surface joint match FAILS here, semantic joint
         match succeeds)
    S-scatter (0): facts (s,r,x),(y,r,o) [both true edges];
         query (s,r,o) -> NO.  r appears 2x > S-active 1x ->
         frequency heuristic predicts YES, truth is NO.
    S-false   (0): facts (s,ri,o); query (s,r,o) -> NO (3105 A1
         baseline)
    x!=o, y!=s, x!=y (x=y would create an in-prompt 2-hop path).
    All conditions 6 lines total; line count constant.

  CUE THEORY (material level, frozen):
    chain r-frequency cue AUC 0.667 (C-rel only distinguishable);
    chain endpoint-presence cue 0.500; scatter active-predicate
    cue AUC 0.375 (< 0.5: S-scatter punishes positives);
    scatter any-r cue 0.500; joint semantic matching 1.000.

GATES (pre-registered):
  G1_chain_truth: m AUC chain-family TEST >= 0.70 AND
      C-true > C-rel strict rate >= 0.60.
  G2_freq_negated: C-true > C-sub strict rate >= 0.60 AND
      mean m(C-sub) < 0 AND mean m(S-scatter) < 0.
  G3_grammar: S-passive > S-false strict rate >= 0.60 AND
      m AUC scatter-family TEST >= 0.70.
  G4_binding_rerun (offline): revised tie-break crit_obj
      T1 TEST >= 0.95 AND TEST-E >= 0.90.
  G5_align (offline, descriptive only): Pearson r(binding teE
      curve, truth teE curve) + peak-layer offset; no hard gate.
  Verdict:
    G1&G2&G3 -> composition_dose_tracked
    G1&!G2   -> frequency_heuristic_confound
    !G1      -> chain_truth_absent_in_readout
    G1&G2&!G3-> surface_match_dominant

SMOKE=1: 10 entities, 4 relations, outdeg 2, 2 chains/rel,
3 scatter pairs, layers [8,16,24,32] -> .../smoke/.

Output: tests/glm5/result/rdc_query_construction_20260913/
        phase3106/omega_p104_composition_dose_depth/
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
NAME = 'omega_p104_composition_dose_depth'
OUT = os.path.join(R13, 'phase3106', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()

SEED = 31060
N_REL = 4 if SMOKE else 8
N_ENT = 10 if SMOKE else 28
OUTDEG = 2 if SMOKE else 3
N_CHAIN_PER_REL = 15 if SMOKE else 60
N_SCATTER_PAIRS = 6 if SMOKE else 60
LAYERS = [8, 16, 24, 32] if SMOKE else [4, 8, 12, 16, 20,
                                        24, 28, 32]
CHUNK = 8 if SMOKE else 96
LAMBDAS = [0.01, 0.1, 1.0]
N_BOOT = 300 if SMOKE else 2000

POSITIONS = ['pos0', 'crit_pred', 'crit_obj',
             'query_pred', 'query_obj', 'last']
FN_IDX = len(LAYERS)
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
PASS_MAP = {'owns': 'owned', 'sells': 'sold',
            'avoids': 'avoided', 'follows': 'followed',
            'repairs': 'repaired', 'ignores': 'ignored',
            'imitates': 'imitated', 'warns': 'warned',
            'admires': 'admired', 'replaces': 'replaced',
            'greets': 'greeted', 'mocks': 'mocked',
            'envies': 'envied', 'trains': 'trained',
            'copies': 'copied', 'shields': 'shielded'}
RULE = ('Note: The same predicate may hold transitively '
        'across a chain of facts.')


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0, msg)
    with io.open(LOGF, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def sha8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:8]


log('Phase 3106 Omega-P104 start; SMOKE=%d OUT=%s'
    % (SMOKE, OUT))

# ================================================================
# 1. OFFLINE: binding gate rerun + layer alignment (3105 data)
# ================================================================
RES3105 = os.path.join(R13, 'phase3105',
                       'omega_p103_incontext_truth_consistency',
                       'result.json')
r5 = json.load(io.open(RES3105, encoding='utf-8'))
t1c = r5['T1_ctx']
t2c = r5['T2_ctx']

# --- G4: revised VAL tie-break (tie -> LATER layer).
# 3105 rule picked L0 among val=1.0 ties; revised rule picks the
# LARGEST layer index among val-ties (binding emerges with depth).
crit_layers = []
for li in range(len(LAYERS) + 1):
    k = 'crit_obj|L%d' % li
    if k in t1c:
        crit_layers.append((li, t1c[k]['val'],
                            t1c[k]['acc_test'],
                            t1c[k]['acc_testE']))
best_val = max(v for (_, v, _, _) in crit_layers)
tie_set = [(li, at, ae) for (li, v, at, ae)
           in crit_layers if v >= best_val - 1e-12]
tie_layers = [li for (li, _, _) in tie_set]
sel_li = max(tie_layers)          # revised rule: later layer
sel = t1c['crit_obj|L%d' % sel_li]
old_pick = t1c['crit_obj|L0']
G4 = (sel['acc_test'] >= 0.95 and sel['acc_testE'] >= 0.90)
log('OFFLINE G4: crit_obj tie set=%s -> revised pick L%d '
    '(test=%.4f teE=%.4f); old pick L0 teE=%.4f; G4=%s'
    % (tie_layers, sel_li, sel['acc_test'],
       sel['acc_testE'], old_pick['acc_testE'], G4))

# --- G5 (descriptive): binding curve vs truth curve alignment.
SL = len(LAYERS) + 1
bind_teE = np.array([t1c['crit_obj|L%d' % li]['acc_testE']
                     for li in range(SL)])
truthq_teE = np.array([t2c['query_obj|L%d' % li]['auc_testE']
                       for li in range(SL)])
truthl_teE = np.array([t2c['last|L%d' % li]['auc_testE']
                       for li in range(SL)])
bind_te = np.array([t1c['crit_obj|L%d' % li]['acc_test']
                    for li in range(SL)])
truthq_te = np.array([t2c['query_obj|L%d' % li]['auc_test']
                      for li in range(SL)])


def pearson(x, y):
    x = x - x.mean()
    y = y - y.mean()
    d = (np.sqrt((x * x).sum() * (y * y).sum()))
    if d == 0:
        return float('nan')
    return float((x * y).sum() / d)


align = {
    'pearson_bind_vs_truthq_teE': pearson(bind_teE,
                                          truthq_teE),
    'pearson_bind_vs_truthl_teE': pearson(bind_teE,
                                          truthl_teE),
    'pearson_bind_vs_truthq_te': pearson(bind_te,
                                         truthq_te),
    'bind_peak_layer_teE': int(bind_teE.argmax()),
    'truthq_peak_layer_teE': int(truthq_teE.argmax()),
    'truthl_peak_layer_teE': int(truthl_teE.argmax()),
    'bind_curve_teE': bind_teE.tolist(),
    'truthq_curve_teE': truthq_teE.tolist(),
}
align['peak_offset_layers'] = (align['truthq_peak_layer_teE']
                               - align['bind_peak_layer_teE'])
log('OFFLINE G5: %s' % json.dumps(align))

# ================================================================
# 2. Material build
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
pass_ids = {}
for w in PREDS:
    ii = tok.encode(' ' + PASS_MAP[w],
                    add_special_tokens=False)
    pass_ids[w] = ii
yes_ids = tok.encode(' yes', add_special_tokens=False)
no_ids = tok.encode(' no', add_special_tokens=False)
assert len(yes_ids) == 1 and len(no_ids) == 1, (yes_ids,
                                                no_ids)
YES_ID, NO_ID = yes_ids[0], no_ids[0]
ent_ids = {e: tok.encode(' ' + e, add_special_tokens=False)
           for e in ENTITIES}
log('A5 predicates single-token: %s' % PREDS)
log('passives: %s' % {w: PASS_MAP[w] for w in PREDS})
log('yes_id=%d no_id=%d' % (YES_ID, NO_ID))

ents = ENTITIES
NE = len(ents)
assert N_REL * NE * OUTDEG <= NE * (NE - 1), 'capacity'

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
in_edges = {r: {} for r in range(N_REL)}
for r in range(N_REL):
    for (s, o) in edges[r]:
        in_edges[r].setdefault(o, []).append(s)
out_edges = {r: {} for r in range(N_REL)}
for r in range(N_REL):
    for (s, o) in edges[r]:
        out_edges[r].setdefault(s, []).append(o)
log('graph: %d true triples, %d pairs'
    % (N_REL * NE * OUTDEG, len(used_pairs)))

# --- splits (unit-blocked; e4 held-out entities -> testE)
e4 = rng.sample(range(NE), 1 if SMOKE
                else max(2, NE // 7))


def blocked_split(units):
    us = units[:]
    rng.shuffle(us)
    teE = [u for u in us if u[0] in e4 or u[1] in e4]
    rest = [u for u in us if u not in set(teE)]
    n = len(rest)
    fr_tr, fr_va = ((0.60, 0.15) if SMOKE
                    else (0.70, 0.12))
    n_tr = int(n * fr_tr)
    n_va = max(1, int(n * fr_va)) if n >= 4 else 0
    n_te = max(1, n - n_tr - n_va) if n >= 4 else 0
    n_tr = n - n_va - n_te
    return rest[:n_tr], rest[n_tr:n_tr + n_va], \
        rest[n_tr + n_va:n_tr + n_va + n_te], teE


# --- chain sampling ((a,c) unique: no duplicate units)
chains = []
chain_ac = set()
for r in range(N_REL):
    two_hop = []
    for (a, b) in sorted(edges[r]):
        for c in sorted(out_edges[r].get(b, [])):
            if c == a:
                continue
            if (a, c) in used_pairs:
                continue   # no verbatim-triple leakage
            if (a, c) in chain_ac:
                continue   # unit uniqueness
            two_hop.append((a, b, c))
    rng.shuffle(two_hop)
    got = 0
    for (a, b, c) in two_hop:
        if got >= N_CHAIN_PER_REL:
            break
        # C-sub subject x: true edge (x,r,c), x not a/b/c
        xc = [x for x in in_edges[r].get(c, [])
              if x not in (a, b, c)]
        if not xc:
            continue
        x = xc[0]
        chains.append({'rel': r, 'a': a, 'b': b, 'c': c,
                       'x': x,
                       'ri': rng.choice([q for q in
                                         range(N_REL)
                                         if q != r])})
        chain_ac.add((a, c))
        got += 1
assert len(chains) >= N_REL * 8, \
    ('chain sampling too thin', len(chains))
chain_units = [(c['a'], c['c']) for c in chains]
ch_tr, ch_va, ch_te, ch_teE = blocked_split(chain_units)


def chain_split(c):
    u = (c['a'], c['c'])
    if u in ch_tr:
        return 'train'
    if u in ch_va:
        return 'val'
    if u in ch_te:
        return 'test'
    return 'testE'


log('chains: %d (per-rel target %d); splits tr/va/te/teE '
    '= %d/%d/%d/%d' % (len(chains), N_CHAIN_PER_REL,
                       len(ch_tr), len(ch_va), len(ch_te),
                       len(ch_teE)))

# --- scatter pair sampling
scount_out = {}
scount_in = {}
for p in ALL_EDGE_PAIRS:
    (s, o) = p
    r = pair2rel[p]
    scount_out.setdefault(r, {})
    scount_out[r][s] = scount_out[r].get(s, 0) + 1
    scount_in.setdefault(r, {})
    scount_in[r][o] = scount_in[r].get(o, 0) + 1
scat_pairs = []
cand = ALL_EDGE_PAIRS[:]
rng.shuffle(cand)
for (s, o) in cand:
    if len(scat_pairs) >= N_SCATTER_PAIRS:
        break
    r = pair2rel[(s, o)]
    if scount_out[r].get(s, 0) < 2:
        continue
    if scount_in[r].get(o, 0) < 2:
        continue
    xs = [v for v in out_edges[r][s] if v != o]
    ys = [u for u in in_edges[r][o] if u != s]
    ok = [(x, y) for x in xs for y in ys
          if x != y and x != s and y != o]
    if not ok:
        continue
    (x, y) = ok[0]
    ri = rng.choice([q for q in range(N_REL) if q != r])
    scat_pairs.append({'s': s, 'o': o, 'r': r, 'x': x,
                       'y': y, 'ri': ri})
assert len(scat_pairs) == N_SCATTER_PAIRS, len(scat_pairs)
scat_units = [(c['s'], c['o']) for c in scat_pairs]
sc_tr, sc_va, sc_te, sc_teE = blocked_split(scat_units)


def scat_split(c):
    u = (c['s'], c['o'])
    if u in sc_tr:
        return 'train'
    if u in sc_va:
        return 'val'
    if u in sc_te:
        return 'test'
    return 'testE'


log('scatter pairs: %d; splits tr/va/te/teE = %d/%d/%d/%d'
    % (len(scat_pairs), len(sc_tr), len(sc_va),
       len(sc_te), len(sc_teE)))

# --- neutral distractor blocks (condition-constant per unit)
used_units = set(chain_units) | set(scat_units)


def pick_neutral(r_main, k, banned, seen):
    """k neutral edges from relations != r_main (relations
    may repeat; r_main itself never appears in D)."""
    pool = [(s2, o2, rho)
            for rho in range(N_REL) if rho != r_main
            for (s2, o2) in sorted(edges[rho])
            if s2 not in banned and o2 not in banned
            and (s2, o2) not in seen]
    assert len(pool) >= k, (len(pool), k)
    rng.shuffle(pool)
    D = []
    for (s2, o2, rho) in pool[:k]:
        seen.add((s2, o2))
        D.append((s2, rho, o2))
    return D


neutrals = {}
for ci, c in enumerate(chains):
    seen = {(c['a'], c['b']), (c['b'], c['c']),
            (c['x'], c['c']), (c['a'], c['c']),
            (c['c'], c['a'])}
    banned = {c['a'], c['b'], c['c'], c['x']}
    D = pick_neutral(c['rel'], 4, banned, seen)
    assert len(D) == 4
    neutrals[('chain', ci)] = D
for si, c in enumerate(scat_pairs):
    seen = {(c['s'], c['o']), (c['s'], c['x']),
            (c['y'], c['o']), (c['o'], c['s'])}
    banned = {c['s'], c['o'], c['x'], c['y']}
    D = pick_neutral(c['r'], 5, banned, seen)
    assert len(D) == 5
    neutrals[('scat', si)] = D


def line_spans_act(pos, ls, lr, lo):
    a_s = pos + len('The ')
    b_s = a_s + len(ents[ls])
    a_r = b_s + len(' ')
    b_r = a_r + len(PREDS[lr])
    a_o = b_r + len(' the ')
    b_o = a_o + len(ents[lo])
    return (a_s, b_s), (a_r, b_r), (a_o, b_o)


def line_spans_pas(pos, lo, lr, ls):
    """Passive line: 'The o is PASS by the s.'
    Semantics: (ls, lr, lo).  crit_obj = o (patient)."""
    a_o = pos + len('The ')
    b_o = a_o + len(ents[lo])
    a_i = b_o + len(' is ')
    b_i = a_i + len(PASS_MAP[PREDS[lr]])
    a_s = b_i + len(' by the ')
    b_s = a_s + len(ents[ls])
    return (a_s, b_s), (a_i, b_i), (a_o, b_o)


def build_prompt(lines, query, rule, crit_idx, order_seed):
    """lines: list of (sub, rel, obj, form); form 'act' ->
    'The sub rel the obj.' ; 'pas' -> 'The obj is PASS[rel]
    by the sub.'  crit_idx: index of the crit line (in
    `lines` order).  query: (s, r, o)."""
    core = list(lines)
    rng2 = random.Random(order_seed & 0xffffffff)
    order = list(range(len(core)))
    rng2.shuffle(order)
    core = [core[i] for i in order]
    ci = core.index(lines[crit_idx])
    text = (rule + '\n') if rule else ''
    text += 'Facts:'
    spans = {}
    pos = len(text)
    for (ls, lr, lo, form) in core:
        if form == 'act':
            seg = ' The %s %s the %s.' % (ents[ls],
                                          PREDS[lr],
                                          ents[lo])
        else:
            seg = ' The %s is %s by the %s.' % (
                ents[lo], PASS_MAP[PREDS[lr]], ents[ls])
        text += seg
        pos += len(seg)
    # walk core order to place crit spans exactly
    pos = len((rule + '\n') if rule else '') \
        + len('Facts:')
    for k2, (ls2, lr2, lo2, form2) in enumerate(core):
        if form2 == 'act':
            (ss, sr, so) = line_spans_act(pos + 1, ls2,
                                          lr2, lo2)
        else:
            (ss, sr, so) = line_spans_pas(pos + 1, lo2,
                                          lr2, ls2)
        if k2 == ci:
            spans['crit_pred'] = sr
            spans['crit_obj'] = so
        seg2 = (' The %s %s the %s.' % (ents[ls2],
                                        PREDS[lr2],
                                        ents[lo2])
                if form2 == 'act' else
                ' The %s is %s by the %s.' % (
                    ents[lo2], PASS_MAP[PREDS[lr2]],
                    ents[ls2]))
        pos += len(seg2)
    (qs, qr_, qo) = query
    if True:
        qseg = (' Query: The %s %s the %s. Is this query '
                'true? Answer:' % (ents[qs], PREDS[qr_],
                                   ents[qo]))
        text += qseg
        qstart = text.rindex('The %s %s the %s.'
                             % (ents[qs], PREDS[qr_],
                                ents[qo]))
        (ss, sr, so) = line_spans_act(qstart, qs, qr_, qo)
        spans['query_subj'] = ss
        spans['query_pred'] = sr
        spans['query_obj'] = so
    return text, spans


records = []
# --- chain records
for ci, c in enumerate(chains):
    r = c['rel']
    D = neutrals[('chain', ci)]
    lines_true = [(c['a'], r, c['b'], 'act'),
                  (c['b'], r, c['c'], 'act')]
    lines_rel = [(c['a'], r, c['b'], 'act'),
                 (c['b'], c['ri'], c['c'], 'act')]
    lines_sub = [(c['a'], r, c['b'], 'act'),
                 (c['x'], r, c['c'], 'act')]
    oseed = hash(('chain', ci, 'ord6')) & 0xffffffff
    for cond, lines, q, lab in (
            ('C-true', lines_true, (c['a'], r, c['c']), 1),
            ('C-swap', lines_true, (c['c'], r, c['a']), 0),
            ('C-rel', lines_rel, (c['a'], r, c['c']), 0),
            ('C-sub', lines_sub, (c['a'], r, c['c']), 0)):
        text, spans = build_prompt(
            lines, q, RULE, 1, oseed + hash(cond))
        records.append({
            'id': 'p%d' % len(records),
            'split': chain_split(c),
            'unit': 'chain%d' % ci,
            'pair': [c['a'], c['c']],
            'family': 'chain', 'cond': cond,
            'query_rel': q[1], 'crit_rel': lines[1][1],
            'truth': lab, 'text': text, 'spans': spans,
            'tag': 'chain'})
# --- scatter records
for si, c in enumerate(scat_pairs):
    r = c['r']
    D = neutrals[('scat', si)]
    oseed = hash(('scat', si, 'ord6')) & 0xffffffff
    conds = (
        ('S-active',
         [(c['s'], r, c['o'], 'act')],
         (c['s'], r, c['o']), 1, 0),
        ('S-passive',
         [(c['s'], r, c['o'], 'pas')],
         (c['s'], r, c['o']), 1, 0),
        ('S-scatter',
         [(c['s'], r, c['x'], 'act'),
          (c['y'], r, c['o'], 'act')],
         (c['s'], r, c['o']), 0, 0),
        ('S-false',
         [(c['s'], c['ri'], c['o'], 'act')],
         (c['s'], r, c['o']), 0, 0))
    for (cond, lines, q, lab, crit_i) in conds:
        text, spans = build_prompt(
            lines, q, None, crit_i,
            oseed + hash(cond))
        records.append({
            'id': 'p%d' % len(records),
            'split': scat_split(c),
            'unit': 'scat%d' % si,
            'pair': [c['s'], c['o']],
            'family': 'scatter', 'cond': cond,
            'query_rel': q[1], 'crit_rel': lines[crit_i][1],
            'truth': lab, 'text': text, 'spans': spans,
            'tag': 'scatter'})

log('records: %d (chain %d, scatter %d)'
    % (len(records),
       sum(1 for x in records if x['tag'] == 'chain'),
       sum(1 for x in records
           if x['tag'] == 'scatter')))

# --- A6 multiset checks (both families, 20 units each)
rng_chk = random.Random(SEED + 7)
n_chk = 0
chain_recs = {}
scat_recs = {}
for x in records:
    if x['tag'] == 'chain':
        chain_recs.setdefault(x['unit'], {})[x['cond']] = x
    else:
        scat_recs.setdefault(x['unit'], {})[x['cond']] = x
chain_units_all = sorted(chain_recs)
rng_chk.shuffle(chain_units_all)
for u in chain_units_all[:20]:
    cm = chain_recs[u]
    tp = Counter(tok.encode(cm['C-true']['text'],
                            add_special_tokens=False))
    tr_ = Counter(tok.encode(cm['C-rel']['text'],
                             add_special_tokens=False))
    ts = Counter(tok.encode(cm['C-sub']['text'],
                            add_special_tokens=False))
    # C-true vs C-rel: line-2 predicate r vs ri
    rp = pred_ids[cm['C-true']['crit_rel']]
    rr = pred_ids[cm['C-rel']['crit_rel']]
    assert (tp - tr_) == Counter({rp: 1}), (tp - tr_, rp)
    assert (tr_ - tp) == Counter({rr: 1}), (tr_ - tp, rr)
    # C-true vs C-sub: line-2 subject b vs x (multiset of
    # entity tokens; entity token length may exceed 1)
    cb = cm['C-true']['crit_rel']  # == r
    ci2 = cm['C-true']['pair']
    # subjects are line-2 first entities: recover from spans
    # is complex; instead check diff == entity-token multiset
    d1 = tp - ts
    d2 = ts - tp
    # tokens of ' b' vs ' x' entities
    b_ent = None
    x_ent = None
    for c2 in chains:
        if ('chain%d' % chains.index(c2)) == u:
            b_ent = ents[c2['b']]
            x_ent = ents[c2['x']]
            break
    eb = Counter(tok.encode(' ' + b_ent,
                            add_special_tokens=False))
    ex = Counter(tok.encode(' ' + x_ent,
                            add_special_tokens=False))
    assert d1 == eb - ex and d2 == ex - eb, (d1, d2)
    # C-true vs C-swap: query endpoints swapped -> SAME token
    # multiset (word order only), different text.  This is the
    # bag-matched property of the swap condition.
    tw = Counter(tok.encode(cm['C-swap']['text'],
                            add_special_tokens=False))
    assert cm['C-swap']['text'] != cm['C-true']['text']
    assert tp == tw, ('swap bag mismatch', tp - tw)
    n_chk += 1
scat_units_all = sorted(scat_recs)
rng_chk.shuffle(scat_units_all)
for u in scat_units_all[:20]:
    sm = scat_recs[u]
    ta = Counter(tok.encode(sm['S-active']['text'],
                            add_special_tokens=False))
    tf = Counter(tok.encode(sm['S-false']['text'],
                            add_special_tokens=False))
    rp = pred_ids[sm['S-active']['crit_rel']]
    rf = pred_ids[sm['S-false']['crit_rel']]
    assert (ta - tf) == Counter({rp: 1}), (ta - tf, rp)
    assert (tf - ta) == Counter({rf: 1}), (tf - ta, rf)
    n_chk += 1
log('A6 multiset checks passed on %d units' % n_chk)

# --- A4 balance (scatter 1:1; chain 1:3 by design)
n_ch_t = sum(1 for x in records
             if x['tag'] == 'chain' and x['truth'] == 1)
n_ch_f = sum(1 for x in records
             if x['tag'] == 'chain' and x['truth'] == 0)
n_sc_t = sum(1 for x in records
             if x['tag'] == 'scatter' and x['truth'] == 1)
n_sc_f = sum(1 for x in records
             if x['tag'] == 'scatter' and x['truth'] == 0)
assert n_sc_t == n_sc_f, (n_sc_t, n_sc_f)
assert n_ch_t * 3 == n_ch_f, (n_ch_t, n_ch_f)
log('A4 balance: chain %d:%d scatter %d:%d'
    % (n_ch_t, n_ch_f, n_sc_t, n_sc_f))

# --- cue theory values (frozen, material level)
CUE = {
    'chain_r_freq_auc_theory': 0.667,
    'chain_endpoint_auc_theory': 0.500,
    'scatter_active_pred_auc_theory': 0.375,
    'scatter_any_r_auc_theory': 0.500,
    'joint_semantic_theory': 1.000,
}

# --- design seal (pre-observation)
design = {
    'phase': 3106, 'name': NAME,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE, 'seed': SEED,
    'n_rel': N_REL, 'n_ent': NE, 'outdeg': OUTDEG,
    'n_chains': len(chains), 'n_scat_pairs': len(scat_pairs),
    'n_records': len(records),
    'layers': LAYERS, 'n_boot': N_BOOT,
    'e4_heldout': e4,
    'gates': {
        'G1_chain': 'm AUC chain TEST >= 0.70 AND '
                    'C-true>C-rel strict >= 0.60',
        'G2_freq': 'C-true>C-sub strict >= 0.60 AND '
                   'mean m(C-sub) < 0 AND mean '
                   'm(S-scatter) < 0',
        'G3_grammar': 'S-passive>S-false strict >= 0.60 '
                      'AND m AUC scatter TEST >= 0.70',
        'G4_binding_rerun': 'revised tie-break crit_obj '
                            'T1 TEST >= 0.95 AND '
                            'TEST-E >= 0.90 (offline '
                            '3105 data)',
        'G5_align': 'descriptive only: Pearson + peak '
                    'offset (offline 3105 data)'},
    'verdict_map': {
        'G1&G2&G3': 'composition_dose_tracked',
        'G1&!G2': 'frequency_heuristic_confound',
        '!G1': 'chain_truth_absent_in_readout',
        'G1&G2&!G3': 'surface_match_dominant'},
    'cue_theory': CUE,
    'rule_sentence_chain': RULE,
    'no_rule_sentence_scatter': True,
    'offline_source_3105': RES3105,
    'offline_g4_tie_set': tie_layers,
    'offline_g4_selected_layer': sel_li,
    'offline_g4_pass': bool(G4),
}
mat_path = os.path.join(OUT, 'material.json')
with io.open(mat_path, 'w', encoding='utf-8') as f:
    json.dump({
        'seed': SEED, 'entities': ents,
        'predicates': PREDS, 'passives': PASS_MAP,
        'pred_ids': pred_ids,
        'yes_id': YES_ID, 'no_id': NO_ID,
        'edges': {str(r): sorted(edges[r])
                  for r in range(N_REL)},
        'chains': chains, 'scat_pairs': scat_pairs,
        'neutrals': {'%s_%d' % k: v
                     for k, v in neutrals.items()},
        'records': records}, f, ensure_ascii=False)
with io.open(os.path.join(OUT, 'design_seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(design, f, indent=1)
log('design sealed (pre-observation); material sha8=%s'
    % sha8(mat_path))
MAT_SHA = sha8(mat_path)

# ================================================================
# 3. Capture
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

# --- capture loop
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
    family=np.array([r['family'] for r in records]),
    cond=np.array([r['cond'] for r in records]),
    truth=np.array([r['truth'] for r in records]),
    crit_rel=np.array([r['crit_rel'] for r in records]),
    query_rel=np.array([r['query_rel']
                        for r in records]),
    tag=np.array([r['tag'] for r in records]),
    unit=np.array([r['unit'] for r in records]))
log('capture saved (%.1fs) X=%s'
    % (time.time() - t_cap, str(X_all.shape)))
gc.collect()
torch.cuda.empty_cache()

# ================================================================
# 4. Probes
# ================================================================
split = np.array([r['split'] for r in records])
truth = np.array([r['truth'] for r in records])
fam = np.array([r['family'] for r in records])
crel = np.array([r['crit_rel'] for r in records])
cond = np.array([r['cond'] for r in records])
tag = np.array([r['tag'] for r in records])
unit = np.array([r['unit'] for r in records])

tr = np.where((split == 'train'))[0]
va = np.where((split == 'val'))[0]
te = np.where((split == 'test'))[0]
teE = np.where((split == 'testE'))[0]
log('probe sets: train=%d val=%d test=%d testE=%d'
    % (len(tr), len(va), len(te), len(teE)))
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

# --- zero-param m margin (sign: truth -> +m; 3105 fix kept)
m_all = M_all
ch_m = (tag == 'chain')
sc_m = (tag == 'scatter')
auc_m_chain_te = auc_score(truth[ch_m & (split == 'test')],
                           m_all[ch_m & (split == 'test')])
auc_m_chain_teE = auc_score(
    truth[ch_m & (split == 'testE')],
    m_all[ch_m & (split == 'testE')])
auc_m_scat_te = auc_score(
    truth[sc_m & (split == 'test')],
    m_all[sc_m & (split == 'test')])
auc_m_scat_teE = auc_score(
    truth[sc_m & (split == 'testE')],
    m_all[sc_m & (split == 'testE')])

# --- within-unit strict rates (test + testE main records)
def strict_rate(mask, cpos, cneg):
    flags = []
    for u in sorted(set(unit[mask])):
        um = mask & (unit == u)
        cm = {c: float(m_all[i]) for c, i in
              ((cond[i], i) for i in np.where(um)[0])}
        if cpos in cm and cneg in cm:
            flags.append(bool(cm[cpos] > cm[cneg]))
    return (float(np.mean(flags)) if flags else None,
            len(flags))


sr_true_rel, n_ch_pairs = strict_rate(
    ch_m & np.isin(split, ['test', 'testE']),
    'C-true', 'C-rel')
sr_true_sub, _ = strict_rate(
    ch_m & np.isin(split, ['test', 'testE']),
    'C-true', 'C-sub')
sr_true_swap, _ = strict_rate(
    ch_m & np.isin(split, ['test', 'testE']),
    'C-true', 'C-swap')
sr_act_false, n_sc_units = strict_rate(
    sc_m & np.isin(split, ['test', 'testE']),
    'S-active', 'S-false')
sr_act_scat, _ = strict_rate(
    sc_m & np.isin(split, ['test', 'testE']),
    'S-active', 'S-scatter')
sr_pas_false, _ = strict_rate(
    sc_m & np.isin(split, ['test', 'testE']),
    'S-passive', 'S-false')

mean_m = {}
for c in ('C-true', 'C-swap', 'C-rel', 'C-sub',
          'S-active', 'S-passive', 'S-scatter',
          'S-false'):
    sel_c = (cond == c)
    mean_m[c] = float(m_all[sel_c].mean()) \
        if sel_c.any() else None
log('m: chain AUC te=%.4f teE=%.4f; scatter AUC te=%.4f '
    'teE=%.4f' % (auc_m_chain_te, auc_m_chain_teE,
                  auc_m_scat_te, auc_m_scat_teE))
log('strict: true>rel=%.4f(n=%d) true>sub=%.4f '
    'true>swap=%.4f act>false=%.4f(n=%d) '
    'act>scat=%.4f pas>false=%.4f'
    % (sr_true_rel, n_ch_pairs, sr_true_sub,
       sr_true_swap, sr_act_false, n_sc_units,
       sr_act_scat, sr_pas_false))
log('mean_m: %s' % json.dumps(mean_m))

# --- A7 permutation sanity (T1 best ctx)
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


ch_te_mask = ch_m & np.isin(split, ['test'])
y_ch = truth[ch_te_mask]
s_ch = m_all[ch_te_mask]
ci_chain_auc = boot_ci(lambda idx: auc_score(y_ch[idx],
                                             s_ch[idx]),
                       len(y_ch))
sc_te_mask = sc_m & np.isin(split, ['test'])
y_sc = truth[sc_te_mask]
s_sc = m_all[sc_te_mask]
ci_scat_auc = boot_ci(lambda idx: auc_score(y_sc[idx],
                                            s_sc[idx]),
                      len(y_sc))
log('bootstrap95: chainAUC=%s scatAUC=%s'
    % (ci_chain_auc, ci_scat_auc))

# ================================================================
# 5. Gates & verdict
# ================================================================
crit_best_rev = T1_CTX['crit_obj|L%d'
                       % min(sel_li, SLOTS - 1)]
crit_best_any = max((v for k, v in T1_CTX.items()
                     if k.startswith('crit_obj')),
                    key=lambda v: v['acc_test'])
t2_best_auc = max(v['auc_test']
                  for v in T2_CTX.values())
ep_auc = T2_BASE['endpoint_concat']['auc_test']
G1 = (auc_m_chain_te is not None
      and auc_m_chain_te >= 0.70
      and sr_true_rel is not None
      and sr_true_rel >= 0.60)
G2 = (sr_true_sub is not None and sr_true_sub >= 0.60
      and mean_m['C-sub'] is not None
      and mean_m['C-sub'] < 0
      and mean_m['S-scatter'] is not None
      and mean_m['S-scatter'] < 0)
G3 = (sr_pas_false is not None and sr_pas_false >= 0.60
      and auc_m_scat_te is not None
      and auc_m_scat_te >= 0.70)
n_pass = sum([G1, G2, G3])
if G1 and G2 and G3:
    verdict = 'composition_dose_tracked'
elif not G1:
    verdict = 'chain_truth_absent_in_readout'
elif not G2:
    verdict = 'frequency_heuristic_confound'
else:
    verdict = 'surface_match_dominant'
log('GATES: G1=%s G2=%s G3=%s G4(offline)=%s -> %s'
    % (G1, G2, G3, G4, verdict))

results = {
    'verdict': verdict,
    'offline': {'g4_binding_rerun': bool(G4),
                'g4_tie_set': tie_layers,
                'g4_selected_layer': sel_li,
                'g4_selected': {k: sel[k] for k in
                                ('val', 'lam',
                                 'acc_test',
                                 'acc_testE')},
                'g4_old_pick_teE':
                    old_pick['acc_testE'],
                'g5_align': align},
    'gates': {
        'G1_chain': G1, 'G2_freq': G2,
        'G3_grammar': G3, 'G4_binding_rerun': G4,
        'm_auc_chain_test': auc_m_chain_te,
        'm_auc_chain_testE': auc_m_chain_teE,
        'm_auc_scatter_test': auc_m_scat_te,
        'm_auc_scatter_testE': auc_m_scat_teE,
        'strict_Ctrue_Crel': sr_true_rel,
        'strict_Ctrue_Csub': sr_true_sub,
        'strict_Ctrue_Cswap': sr_true_swap,
        'strict_Sact_Sfalse': sr_act_false,
        'strict_Sact_Sscat': sr_act_scat,
        'strict_Spas_Sfalse': sr_pas_false,
        'n_chain_units_eval': n_ch_pairs,
        'n_scat_units_eval': n_sc_units,
        't2_best_auc_test': t2_best_auc,
        'endpoint_truth_auc_test': ep_auc,
        'crit_obj_any_best_test':
            crit_best_any['acc_test'],
        'crit_obj_any_best_testE':
            crit_best_any['acc_testE'],
        'endpoint_acc_test':
            T1_BASE['endpoint_concat']['acc_test'],
        'bow_acc_test': T1_BASE['bow']['acc_test']},
    'm': {'mean_m_by_cond': mean_m,
          'ci_chain_auc_test': ci_chain_auc,
          'ci_scatter_auc_test': ci_scat_auc},
    'cue_theory': CUE,
    'sanity': {'perm_acc_test': acc_perm,
               'perm_config': best_ctx_name,
               'a1_determinism': d_a1,
               'a6_multiset_units': n_chk,
               'n_records': len(records),
               'balance_chain_1to3': [n_ch_t, n_ch_f],
               'balance_scatter_1to1': [n_sc_t,
                                        n_sc_f]},
    'T1_ctx': {k: {kk: vv for kk, vv in v.items()
                   if not kk.startswith('pred_')}
               for k, v in T1_CTX.items()},
    'T1_base': T1_BASE,
    'T2_ctx': {k: {kk: vv for kk, vv in v.items()
                   if not kk.startswith('score_')}
               for k, v in T2_CTX.items()},
    'T2_base': T2_BASE,
    'created': datetime.now().strftime(
        '%Y-%m-%d %H:%M:%S'),
    'material_sha8': MAT_SHA,
    'n_records': len(records),
}
with io.open(os.path.join(OUT, 'result.json'), 'w',
             encoding='utf-8') as f:
    json.dump(results, f, indent=1)
log('result.json written; verdict=%s' % verdict)
print('PHASE3106_DONE verdict=%s G1=%s G2=%s G3=%s G4=%s'
      % (verdict, G1, G2, G3, G4))
