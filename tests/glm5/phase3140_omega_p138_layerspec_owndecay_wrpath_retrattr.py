# -*- coding: utf-8 -*-
"""Phase 3140 (Omega-P138): layer-spectrum
separation formalization (iinj vs cinj
full-spectrum) + own-nonspecificity decay
+ WR behavioral pathway + retrieval
failure attribution.

Preregistered in 3139 closeout (task 83,
aligned with FINGERPRINT_PARADIGM_PLAN.md
Omega-P138):
(1) layer-spectrum separation: iinj
(displaced identity) vs cinj (C(T2)
template) x SPECT_L (L17-38 step2 plus
26/32/38 densified = 14 layers) x doses
{1.0, 2.0}, SCAN_N=128 allstep -> test
L17/L26 double-peak separation (3139
baselines: iinj L17 0.328 / cinj L26
0.289); 9 trials replicate 3139 exact
chg values bit-level;
(2) own-nonspecificity deep-dive: dvec
energy decay curve in I-space (own vs
neighbors +/-1/2/4/8 vs all) + rank
curve + 3136 co36/co50/union coordinate
enrichment z-test (permutation 2000);
(3) WR behavioral pathway: WR residual
PC1 injection at L26/L29 x dose
{1,2,4} (median-norm scaled, uniform) +
PC2 row-specific projection control ->
quiet or active;
(4) retrieval attribution: 84 unseen
pairs, V1 (3139 fixed-seed distractor
prompt) vs V2 (same-distractor-family
prompt rebuilt from a bank row with the
same subject) -> retr_same_s comparison
at L17/29/38.

Bank shards REUSED from 3138 full run
(no recapture). dvecs frozen from 3135
(sha-anchored)."""
import gc
import hashlib
import io
import json
import os
import random as _rnd
import time
import zlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
NAME = ('omega_p138_layerspec_owndecay_'
        'wrpath_retrattr')
SMOKE = os.environ.get('P3140_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D35 = RDIR + r'\phase3135' \
      r'\omega_p133_conduction_' \
      r'co36ablation_window'
D36 = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_crossmatrix_w8drop'
D38 = RDIR + r'\phase3138' \
      r'\omega_p136_statebank_' \
      r'bicdecomp_probecontrast'
D39 = RDIR + r'\phase3139' \
      r'\omega_p137_idinteract_portconsume_' \
      r'rewrite_xmat'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3140', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0,
                            msg)
    with io.open(LOGF, 'a',
                 encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ---------------- crash-resume ckpt -----
import pickle  # noqa: E402
CKPTF = os.path.join(OUT, 'p138_ckpt.pkl')
CK = {'done': [], 'data': {}, 'meta': {}}
if os.path.exists(CKPTF):
    try:
        with io.open(CKPTF, 'rb') as _fh:
            CK = pickle.load(_fh)
        log('RESUME: ckpt stages=%s'
            % sorted(CK['done']))
    except Exception as _e:
        log('RESUME: ckpt load failed (%s), '
            'starting fresh' % _e)
        CK = {'done': [], 'data': {},
              'meta': {}}


def ck_save(stage, data):
    CK['data'][stage] = data
    if stage not in CK['done']:
        CK['done'].append(stage)
    _tmp = CKPTF + '.tmp'
    with io.open(_tmp, 'wb') as _fh:
        pickle.dump(CK, _fh, protocol=4)
    os.replace(_tmp, CKPTF)
    log('CKPT saved: %s' % stage)


# ---------------- frozen constants ------
NP = 672
N_T = 4
DIRS = ('P', 'A1')
ALL_L = list(range(40))
KEY_L = (17, 29, 33, 38)
# prereg: L17-38 step2 (11 odd layers)
# densified with causal-key layers
# 26/32 and write anchor 38 -> 14 layers
SPECT_L = (17, 19, 21, 23, 25, 26, 27, 29,
           31, 32, 33, 35, 37, 38)
SCAN_N = 4 if SMOKE else 128
GEN_BATCH = 4 if SMOKE else 32
N_NEW = 12
DOSES = (1.0, 2.0)
WR_DOSES = (1.0, 2.0, 4.0)
WR_QUIET = 0.10
NB_OFFS = (1, 2, 4, 8)
CO_PERM = 2000
V2_GAIN = 0.05
XMAT_RETR_MIN = 0.30
RNG_SEED = 3140

# ---------------- frozen 3139 anchors --
EXP_V39 = ('a_3138_ok|ish2_hard75_fail|'
           'ish2_soft85_below|r_retr_chance|'
           'port_I_dominant|port_c_signal|'
           'cinj_active|iinj_recorded|'
           'xmat_retr_below|'
           'xmat_auc_recorded|coverage_full')
RES39_SHA = '7b57b15b'
XPHASE39 = 1.0
# exact chg values from 3139 result.json
# (bit-level repro targets, tol 1e-9)
T39 = {
    'iinj_l17_d1.0': 0.328125,
    'iinj_l26_d1.0': 0.1015625,
    'iinj_l29_d1.0': 0.109375,
    'iinj_l32_d1.0': 0.078125,
    'cinj_l17_d2.0': 0.1875,
    'cinj_l26_d2.0': 0.2890625,
    'cinj_l29_d1.0': 0.1796875,
    'cinj_l29_d2.0': 0.2578125,
    'cinj_l32_d2.0': 0.2421875}
PORT39 = {
    '17': {'e_I_med': 0.3851623801197181,
           'e_own_med': 0.032851606607437134,
           'own_over_I': 0.08674553268642335},
    '29': {'e_I_med': 0.4460011825579544,
           'e_own_med': 0.01960926502943039,
           'own_over_I': 0.04286333097287928}}
RETR39_S = {'17': 0.05952380952380952,
            '29': 0.03571428571428571,
            '38': 0.05952380952380952}
AUC39_X = {'17': 1.0, '29': 1.0, '38': 1.0}
ISH39_17P = 0.7553799152374268
NS39_17R = 0.0876662571046181
NPAIR39 = 84
RES36_SHA = '3903af46'
# dvec sha8 (3135 npz, fp32)
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
DVEC_MEDNORM = {17: 4.253485202789307,
                29: 31.643726348876953,
                33: 44.65449905395504,
                38: 100.42410278320312}
LEDGER_N = 276

# ---------------- seal ------------------
SEAL = {
    'phase': 3140,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_T': N_T,
        'DIRS': list(DIRS),
        'ALL_L': ALL_L,
        'KEY_L': list(KEY_L),
        'SPECT_L': list(SPECT_L),
        'SCAN_N': SCAN_N,
        'GEN_BATCH': GEN_BATCH,
        'DOSES': list(DOSES),
        'WR_DOSES': list(WR_DOSES),
        'WR_QUIET': WR_QUIET,
        'NB_OFFS': list(NB_OFFS),
        'CO_PERM': CO_PERM,
        'V2_GAIN': V2_GAIN,
        'XMAT_RETR_MIN': XMAT_RETR_MIN,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res39_verdict': EXP_V39,
        'res39_sha8': RES39_SHA,
        'xphase39': XPHASE39,
        't39_chg': T39,
        'port39': PORT39,
        'retr39_s': RETR39_S,
        'auc39_x': AUC39_X,
        'ish39_17P': ISH39_17P,
        'ns39_17R': NS39_17R,
        'npair39': NPAIR39,
        'res36_sha8': RES36_SHA,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3139 closeout task 83 + '
               'FINGERPRINT_PARADIGM_PLAN '
               'Omega-P138: (1) layer-spectrum '
               'separation iinj vs cinj x '
               'SPECT_L(14) x doses 1/2, 9 '
               'trials bit-replicate 3139; '
               '(2) own decay curve own/nb/'
               'all + rank curve + co36/co50/'
               'union enrichment z (perm '
               '2000); (3) WR PC1 injection '
               'L26/29 x dose 1/2/4 + PC2 '
               'row-specific control; (4) '
               'retrieval attribution V1 vs '
               'V2 same-distractor-family '
               'prompt, retr_same_s L17/29/'
               '38. Bank reused from 3138. '
               'Frozen before observation.')}
SEALF = os.path.join(OUT, 'design_seal.json')
if os.path.exists(SEALF):
    prev = json.load(io.open(SEALF,
                             encoding='utf-8'))
    seal_rt = json.loads(json.dumps(SEAL))
    assert prev['constants'] == \
        seal_rt['constants'], 'seal drift'
    assert prev['anchors'] == \
        seal_rt['anchors'], 'seal drift'
    log('seal ok (existing, constants match)')
else:
    with io.open(SEALF, 'w',
                 encoding='utf-8') as f:
        json.dump(SEAL, f,
                  ensure_ascii=False,
                  indent=1)
    log('seal written (constants frozen)')

# ---------------- frozen inputs ---------
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
assert z26['mlg_s0_P'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
z35 = np.load(D35 + r'\p133_readout.npz',
              allow_pickle=False)
dvec = {}
for l in KEY_L:
    a = z35['dvec%d' % l].astype(np.float32)
    got = hashlib.sha256(
        a.tobytes()).hexdigest()[:8]
    assert got == DVEC_SHA[l], (l, got)
    assert a.shape == (NP, 4096)
    dvec[l] = a
    medn = float(np.median(
        np.linalg.norm(a, axis=1)))
    drift = abs(medn - DVEC_MEDNORM[l])
    log('frozen dvec L%02d sha8 %s '
        'med||d||=%.4f (drift %.2e)'
        % (l, DVEC_SHA[l], medn, drift))
    assert drift < 1e-3, drift
res39 = json.load(io.open(
    D39 + r'\result.json', encoding='utf-8'))
assert res39['smoke'] is False
res36 = json.load(io.open(
    D36 + r'\result.json', encoding='utf-8'))
assert res36['smoke'] is False
z136 = np.load(D36 + r'\p134_readout.npz',
               allow_pickle=False)
CO_SETS = {}
for cn in ('co36', 'co50', 'union'):
    _c = np.asarray(z136[cn]).astype(np.int64)
    assert _c.ndim == 1, (cn, _c.shape)
    assert _c.min() >= 0 and _c.max() < 4096
    CO_SETS[cn] = _c
    log('co set %s: len=%d range=[%d,%d]'
        % (cn, len(_c), int(_c.min()),
           int(_c.max())))
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/35/36/39 + '
    'ledger n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3139 link asserts
# ================================================================
log('== PART A: 3139 link asserts ==')
V39 = res39['verdict']
assert V39 == EXP_V39, V39
raw39 = io.open(
    D39 + r'\result.json', 'rb').read()
sha39 = hashlib.sha256(raw39).hexdigest()[:8]
assert sha39 == RES39_SHA, sha39
pe39 = res39['part_e']
pd39 = res39['part_d']
pc39 = res39['part_c']
pf39 = res39['part_f']
assert abs(float(pe39['xphase'])
           - XPHASE39) < 1e-12
for tn, v in T39.items():
    got = float(pe39['trials'][tn]['chg'])
    assert abs(got - v) < 1e-9, (tn, got, v)
for l, dd in PORT39.items():
    for k, v in dd.items():
        got = float(pd39['port'][l][k])
        assert abs(got - v) < 1e-9, (l, k)
for l, v in RETR39_S.items():
    got = float(pf39['retr_s'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
for l, v in AUC39_X.items():
    got = float(pf39['auc_xmat'][l])
    assert abs(got - v) < 1e-9, (l, got, v)
assert abs(float(pc39['ish2']['17']['P'])
           - ISH39_17P) < 1e-9
assert abs(float(
    pc39['norm_share']['17']['R'])
    - NS39_17R) < 1e-9
assert int(pf39['n_pairs']) == NPAIR39
raw36 = io.open(
    D36 + r'\result.json', 'rb').read()
sha36 = hashlib.sha256(raw36).hexdigest()[:8]
assert sha36 == RES36_SHA, sha36
assert res36['verdict'].startswith('a_3135_ok')
log('A hard asserts ok (3139 sha8 %s, '
    'xphase/9-trial/port/retr/auc/ish/ns '
    'anchors; 3136 sha8 %s)' % (sha39, sha36))

# ================================================================
# materials factory (3137 lineage)
# ================================================================
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
capb = np.load(RDIR + r'\phase3113'
               r'\omega_p111_artifact_writein'
               r'\capture_b.npz',
               allow_pickle=False)
pkB = capb['pk']
condB = capb['cond']
hP = {}
for i in range(len(pkB)):
    pk = str(pkB[i])
    if str(condB[i]) == 'P':
        hP[pk] = i
pks = sorted(hP.keys())
assert len(pks) == NP


def build_prompt(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o))
        .encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents[s], PREDS[qrel], ents[o]))
    return text


def make_materials(tok, gens):
    texts = {}
    PID_T = {}
    for dc in DIRS:
        texts[dc] = {}
        ids_l = []
        for pk in pks:
            (s, o) = (int(v)
                      for v in pk.split('_'))
            r = p2r[pk]
            ri1, ri2 = frel[pk]
            if dc == 'P':
                (qrel, lrel) = (r, r)
            else:
                (qrel, lrel) = (r, ri1)
            t_ = build_prompt(mat5, s, o,
                              lrel, qrel)
            texts[dc][pk] = t_
            ids_l.append(list(
                tok(t_,
                    add_special_tokens=False)
                ['input_ids']))
        PID_T[dc] = ids_l
    DOT = int(tok('.',
                  add_special_tokens=False)
              ['input_ids'][0])
    _line_tok = {}

    def line_tokens(s2, o2, lrel):
        key = (s2, o2, lrel)
        if key not in _line_tok:
            txt = ' The %s %s the %s.' % (
                ents_all[s2],
                PREDS_all[lrel],
                ents_all[o2])
            _line_tok[key] = list(tok(
                txt,
                add_special_tokens=False)
                ['input_ids'])
        return _line_tok[key]

    def context_triples(pk, dcode):
        (s, o) = (int(v)
                  for v in pk.split('_'))
        r = p2r[pk]
        ri1, ri2 = frel[pk]
        lrel = r if dcode == 'P' else ri1
        D = [tuple(d) for d in
             mat5['distractors']
             ['%d_%d' % (s, o)]]
        ctx = set((a, rr, b)
                  for (a, rr, b) in
                  ([(s, lrel, o)] + list(D)))
        return s, o, lrel, ctx

    def pick_replacement(pk, dcode):
        (s, o, lrel, ctx) = context_triples(
            pk, dcode)
        rng1 = _rnd.Random(zlib.crc32(
            ('s1|%s|%s' % (pk, dcode))
            .encode('ascii')))
        ents_n = len(ents_all)
        for _try in range(200):
            s2 = rng1.randrange(ents_n)
            o2 = rng1.randrange(ents_n)
            if s2 == s or o2 == o:
                continue
            if (s2, lrel, o2) in ctx:
                continue
            if s2 == o2:
                continue
            return s2, o2
        raise RuntimeError('no replacement')

    def traj_tokens(dc, j):
        ids_r = [int(t) for t in gens[dc][j]]
        ptxt = texts[dc][pks[j]]
        encp = tok(ptxt,
                   add_special_tokens=False,
                   return_offsets_mapping=True)
        poffs = [tuple(v) for v in
                 encp['offset_mapping']]
        ids2 = list(ids_r)[:N_NEW]
        while len(ids2) < N_NEW:
            ids2.append(DOT)
        return ptxt, poffs, ids2

    def find_span(dec, offs, s, o, qrel):
        qline = 'The %s %s the %s.' % (
            ents_all[s], PREDS_all[qrel],
            ents_all[o])
        ce = dec.rfind(qline)
        while ce != -1:
            cs = ce
            ce_want = cs + len(qline)
            if dec[ce_want:ce_want + 8] \
                    == ' Is this':
                idxs = [k for k in
                        range(len(offs))
                        if offs[k][0] >= cs
                        and offs[k][1]
                        <= ce_want
                        and offs[k][1]
                        > offs[k][0]]
                if len(idxs) >= 2 \
                        and idxs[0] >= 1 \
                        and idxs[-1] - idxs[0] \
                        + 1 <= N_NEW:
                    return idxs[0], idxs[-1]
            ce = dec.rfind(qline, 0, ce)
        return None

    span_idx = {dc: np.full((NP, 2), -1,
                            dtype=np.int16)
                for dc in DIRS}
    pids_all = {dc: [None] * NP
                for dc in DIRS}

    def build_pids(dc, j):
        pk = pks[j]
        prompt_ids0 = list(PID_T[dc][j])
        ptxt, poffs, _base = traj_tokens(dc, j)
        (s, o) = (int(v)
                  for v in pk.split('_'))
        r = p2r[pk]
        span = find_span(ptxt, poffs, s, o, r)
        pids = {c: list(prompt_ids0)
                for c in ('s0', 's1', 's2',
                          's3')}
        if span is not None:
            (k1, k2) = span
            Lspan = k2 - k1 + 1
            (s2, o2) = pick_replacement(
                pk, dc)
            (_, _, lrel, _) = context_triples(
                pk, dc)
            sub = line_tokens(s2, o2, lrel)
            n_pad = max(0, Lspan - len(sub))
            sub = sub[:Lspan]
            sub = sub + [DOT] * n_pad
            rng2 = _rnd.Random(zlib.crc32(
                ('s2|%s|%s' % (pk, dc))
                .encode('ascii')))
            shuf = list(sub)
            rng2.shuffle(shuf)
            pids['s1'][k1:k2 + 1] = sub
            pids['s2'][k1:k2 + 1] = shuf
            pids['s3'][k1:k2 + 1] = \
                [DOT] * Lspan
            span_idx[dc][j] = (k1, k2)
        pids_all[dc][j] = pids
        return pids, span

    for dc in DIRS:
        for j in range(NP):
            build_pids(dc, j)
    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT, 'build_pids': build_pids,
            'span_idx': span_idx,
            'pids_all': pids_all,
            'traj_tokens': traj_tokens}


log('factory defined')

# ================================================================
# model load
# ================================================================
log('== load glm4-9b ==')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

gens_g = {'P': z26['gen_base_P'],
          'A1': z26['gen_base_A1']}
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
model_g = AutoModelForCausalLM.from_pretrained(
    MDIR_G, torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    trust_remote_code=True).to('cuda').eval()
NLG = len(model_g.model.layers)
assert NLG == 40
HIDG = int(model_g.config.hidden_size)
assert HIDG == 4096
log('glm4 loaded NLG=%d HIDG=%d'
    % (NLG, HIDG))
WUG = model_g.lm_head.weight.detach()
YES_G = int(tok_g(' yes',
                  add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tok_g(' no',
                 add_special_tokens=False)
           ['input_ids'][0])
w_dn_g = (WUG[YES_G] - WUG[NO_G]) \
    .float().cpu().numpy()
DOT_G = int(tok_g('.',
                  add_special_tokens=False)
            ['input_ids'][0])
norm_g = model_g.model.norm
PADG = tok_g.pad_token_id
if PADG is None:
    PADG = model_g.config.pad_token_id
tok_g.padding_side = 'left'
matg = make_materials(tok_g, gens_g)
for dc in DIRS:
    assert np.array_equal(
        matg['span_idx'][dc],
        z26['span_idx_%s' % dc]), dc
log('span_idx asserts vs 3126 ok (672x2)')

pids_P_all = [matg['pids_all']['P'][j]['s0']
              for j in range(NP)]
pids_A1_all = [matg['pids_all']['A1'][j]['s0']
               for j in range(NP)]
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
log('gen-prefix ready (%d ids)'
    % len(PREFIX_IDS))


def pad12(ids_r):
    v = list(ids_r)[:N_NEW]
    return v + [DOT_G] * (N_NEW - len(v))


def _pad_batch(rows):
    chunk = [list(PREFIX_IDS) + list(p)
             for p in rows]
    maxlen = max(len(p) for p in chunk)
    ids_p = np.full((len(chunk), maxlen),
                    PADG, dtype=np.int64)
    mask_p = np.zeros((len(chunk), maxlen),
                      dtype=np.int64)
    for i, p in enumerate(chunk):
        ids_p[i, maxlen - len(p):] = p
        mask_p[i, maxlen - len(p):] = 1
    return ids_p, mask_p


def gen_batch_g2(pids_list,
                 swap_layers=None,
                 inj=None,
                 inj_vec=None):
    """Generate N_NEW tokens per row.
    inj_vec: per-layer transplants
    (il, dvec_batch (n,HID) fp32, scale,
    mode). mode: 'allstep' = every
    forward; int s = prompt forward +
    decode step s (3135 semantics)."""
    ids_p, mask_p = _pad_batch(pids_list)
    hooks = []
    if swap_layers:
        for l in swap_layers:
            lyr = model_g.model.layers[l]

            def _swap(mod, inp, out,
                      _l=l):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                i2 = inp[0] \
                    if isinstance(inp,
                                  tuple) \
                    else inp
                o2.copy_(i2.to(o2.dtype))

            hooks.append(
                lyr.register_forward_hook(
                    _swap))
    if inj is not None:
        inj_list = inj if isinstance(
            inj, (list, tuple)) \
            and inj and isinstance(
                inj[0], (list, tuple)) \
            else [inj]
        for (il, coords, delta, sgn,
             mode) in inj_list:
            lyr = model_g.model.layers[il]
            co_t = torch.as_tensor(
                np.asarray(coords,
                           dtype=np.int64),
                device='cuda')
            dv_t = torch.full(
                (len(coords),),
                float(delta) * float(sgn),
                device='cuda',
                dtype=torch.bfloat16)
            st = {'step': 0}

            def _inj(mod, inp, out,
                     _co=co_t, _dv=dv_t,
                     _m=mode, _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if _m == 'allstep' \
                        or _st['step'] == 0 \
                        or _st['step'] == _m:
                    o2[:, -1, _co] += _dv
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _inj))
    if inj_vec:
        for (il, dvec_batch, scale,
             mode) in inj_vec:
            lyr = model_g.model.layers[il]
            dv_t = torch.as_tensor(
                np.asarray(dvec_batch,
                           dtype=np.float32),
                device='cuda') \
                .to(torch.bfloat16) \
                * float(scale)
            st = {'step': 0}

            def _injv(mod, inp, out,
                      _dv=dv_t, _m=mode,
                      _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if isinstance(_m,
                              (list, tuple)):
                    _hit = _st['step'] in _m
                else:
                    _hit = (_m == 'allstep'
                            or _st['step'] == 0
                            or _st['step']
                            == _m)
                if _hit:
                    o2[:, -1, :] += _dv
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _injv))
    t_ids = torch.tensor(ids_p,
                         device='cuda')
    t_mask = torch.tensor(mask_p,
                          device='cuda')
    with torch.inference_mode():
        outg = model_g.generate(
            input_ids=t_ids,
            attention_mask=t_mask,
            max_new_tokens=N_NEW,
            do_sample=False, num_beams=1,
            pad_token_id=PADG)
    for hk in hooks:
        hk.remove()
    newg = outg[:, ids_p.shape[1]:]
    res = []
    for row in newg:
        ids_r = [int(t) for t in row]
        if PADG in ids_r:
            ids_r = ids_r[
                :ids_r.index(PADG)]
        for e in model_g.config.eos_token_id \
                if isinstance(
                    model_g.config
                    .eos_token_id,
                    list) else [
            model_g.config.eos_token_id]:
            if e in ids_r:
                ids_r = ids_r[
                    :ids_r.index(e)]
                break
        res.append(ids_r)
    return res


def capture_states(rows_all, cap_layers):
    """Last-position states at cap_layers.
    SINGLE-SAMPLE loop (batch=1, zero
    padding) replicating z26/3133 rev-f
    semantics exactly."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    for j in range(len(rows_all)):
        ids_r = rows_all[j]
        t_in = torch.tensor(
            [list(ids_r)], device='cuda')
        feats = {l: None
                 for l in cap_layers}
        hooks = []

        def _mk(_l):
            def hook(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                feats[_l] = \
                    o2[0, -1, :].detach()
            return hook

        for l in cap_layers:
            lyr = model_g.model.layers[l]
            hooks.append(
                lyr.register_forward_hook(
                    _mk(l)))
        with torch.inference_mode():
            model_g(t_in, use_cache=False)
            for hk in hooks:
                hk.remove()
            for l in cap_layers:
                OUTS[l][j] = \
                    feats[l].float() \
                    .cpu().numpy()
        del feats
    return OUTS


def trial_metrics(gen_l, base12_l):
    same_l = np.zeros(len(gen_l),
                      dtype=bool)
    fstep = np.full(len(gen_l), -1,
                    dtype=np.int8)
    first_l = 0
    for j in range(len(gen_l)):
        g12 = pad12(gen_l[j])
        b12 = base12_l[j]
        same_l[j] = (g12 == b12)
        if g12 != b12:
            for t in range(N_NEW):
                if g12[t] != b12[t]:
                    fstep[j] = t
                    break
        first_l += int(g12[0] != b12[0])
    chg_l = 1.0 - float(same_l.mean())
    cum = [float(((fstep >= 0)
                  & (fstep <= t)).mean())
           for t in range(N_NEW)]
    return chg_l, first_l, fstep, cum


# ================================================================
# PART B: bank reuse (3138 full shards)
# ================================================================
log('== PART B: bank reuse from 3138 ==')
H_ld = {}
H_sc = {}
for dc in DIRS:
    H_ld[dc] = {}
    H_sc[dc] = {}
    for t in range(N_T):
        fp = os.path.join(
            D38, 'bank_%s_T%d.npz'
            % (dc, t))
        assert os.path.exists(fp), fp
        zt = np.load(fp,
                     allow_pickle=False)
        _a = zt['states'].astype(np.float32)
        assert np.isfinite(_a).all(), \
            'bank shard non-finite'
        assert _a.shape == (NP, 40, HIDG)
        H_ld[dc][t] = _a
        H_sc[dc][t] = zt['scale'] \
            .astype(np.float32)
log('bank loaded (8 shards fp16, reused)')

# ================================================================
# PART C0: session baseline + xphase
# ================================================================
log('== PART C0: baseline + xphase ==')
CKM = {'smoke': SMOKE, 'scan_n': SCAN_N}
if CK['meta'] and CK['meta'] != CKM:
    log('CKPT meta mismatch -> discard')
    CK = {'done': [], 'data': {},
          'meta': {}}
CK['meta'] = CKM
rows_scan = [pids_P_all[j]
             for j in range(SCAN_N)]
_EB = CK['data'].get('e_base')
if _EB is not None:
    gen_base = _EB['gens']
    xphase = _EB['xphase']
    log('base RESUMED (xphase=%.4f)'
        % xphase)
else:
    gen_base = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        gen_base.extend(gen_batch_g2(
            rows_scan[b0:b0 + GEN_BATCH]))
    _xp = [pad12(gen_base[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(SCAN_N)]
    xphase = float(np.mean(_xp))
    log('xphase record: session-base vs '
        'z26 bit-match %.4f (%d/%d)'
        % (xphase, int(np.sum(_xp)),
           SCAN_N))
    ck_save('e_base', {
        'gens': gen_base,
        'xphase': xphase})
base12 = [pad12(gen_base[j])
          for j in range(SCAN_N)]
xphase_ok = (xphase == 1.0)

# ================================================================
# PART C1: spectrum-layer decomposition
# ================================================================
log('== PART C1: spectrum decomposition ==')
IDEINT_P = {}
CVEC = {}
PC = {}
WRPCA = {}
for l in SPECT_L:
    X = np.zeros((2, N_T, NP, HIDG),
                 dtype=np.float32)
    for di, dc in enumerate(DIRS):
        for t in range(N_T):
            X[di, t] = \
                H_ld[dc][t][:, l, :] \
                * H_sc[dc][t][l]
    B = X.mean(axis=(0, 1, 2))
    Ideint = X.mean(axis=1) - B
    C = X.mean(axis=(0, 2)) - B
    IDEINT_P[l] = Ideint[0].copy()
    CVEC[l] = C.copy()
    if l in (26, 29):
        WR = X - B[None, None, None, :] \
            - Ideint[:, None, :, :] \
            - C[None, :, None, :]
        WRm = WR.reshape(-1, HIDG)
        mu = WRm.mean(axis=0)[None, :]
        WRc = WRm - mu
        _U, _S, _Vt = np.linalg.svd(
            WRc.astype(np.float64),
            full_matrices=False)
        _var = (_S[:4] ** 2
                / np.sum(_S ** 2)).tolist()
        PC[l] = {
            'V': _Vt[:4].astype(np.float32),
            'var': _var,
            'mnorm': float(np.median(
                np.linalg.norm(
                    WRc, axis=1)))}
        WRPCA[l] = WR.reshape(
            2, N_T, NP, HIDG)
        log('C1 L%02d WR PC var %s mnorm '
            '%.3f' % (l,
                      json.dumps([round(v, 4)
                                  for v in
                                  _var]),
                      PC[l]['mnorm']))
        del WR, WRm, mu, WRc, _U, _S, _Vt
    else:
        WRPCA[l] = None
    del X, B, Ideint, C
    gc.collect()
log('C1 done (%d spectrum layers)'
    % len(SPECT_L))

# ================================================================
# PART C2: spectrum injection trials
# ================================================================
log('== PART C2: spectrum trials ==')
E_res = {}


def _run_inj_trial(tname, il, dv_batch,
                   scale, mode):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, SCAN_N, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      dv_batch[b0:b0
                               + len(batch)],
                      scale, mode)]))
    chg_l, first_l, fstep, _cum = \
        trial_metrics(gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname],
                    'fstep': fstep})


for l in SPECT_L:
    for op in ('iinj', 'cinj'):
        for dose in DOSES:
            tname = ('spec_%s_l%02d_d%.1f'
                     % (op, l, dose))
            if op == 'iinj':
                _dv = np.roll(
                    IDEINT_P[l], -1,
                    axis=0)[:SCAN_N] \
                    .astype(np.float32)
            else:
                _cv = CVEC[l][2][None, :]
                _dv = np.repeat(
                    _cv, SCAN_N,
                    axis=0) \
                    .astype(np.float32)
            _run_inj_trial(tname, l, _dv,
                           dose, 'allstep')

# 3139 bit-level repro check (same-layer
# same-dose trials vs frozen 3139 chg)
n_match = 0
repro = {}
if not SMOKE:
    for tn39, v in T39.items():
        op, rest = tn39.split('_l')
        l = int(rest[:2])
        dose = float(rest.split('_d')[1])
        tn = 'spec_%s_l%02d_d%.1f' % (
            op, l, dose)
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_match += int(m)
        repro[tn39] = {'got': got,
                       'want': v,
                       'match': bool(m)}
    log('C2 3139 repro: %d/%d bit-match'
        % (n_match, len(T39)))
else:
    log('C2 3139 repro SKIPPED (smoke)')

# spectrum verdicts
iinj_d1 = {l: E_res['spec_iinj_l%02d_'
                    'd1.0' % l]['chg']
           for l in SPECT_L}
iinj_d2 = {l: E_res['spec_iinj_l%02d_'
                    'd2.0' % l]['chg']
           for l in SPECT_L}
cinj_d1 = {l: E_res['spec_cinj_l%02d_'
                    'd1.0' % l]['chg']
           for l in SPECT_L}
cinj_d2 = {l: E_res['spec_cinj_l%02d_'
                    'd2.0' % l]['chg']
           for l in SPECT_L}
peak_iinj = max(iinj_d1,
                key=iinj_d1.get)
peak_cinj = max(cinj_d2,
                key=cinj_d2.get)
sep = abs(peak_iinj - peak_cinj)
mono_rows = []
for dd in (iinj_d1, cinj_d1):
    for l in SPECT_L:
        hi = E_res['spec_%s_l%02d_d2.0'
                   % ('iinj'
                      if dd is iinj_d1
                      else 'cinj',
                      l)]['chg']
        mono_rows.append(hi >= dd[l])
mono_rate = float(np.mean(mono_rows))
dose_mono = mono_rate >= 0.6
log('C2-SOFT: iinj d1 spec %s; cinj d2 '
    'spec %s; peaks iinj L%02d cinj L%02d '
    '(sep %d); mono rate %.2f'
    % (json.dumps({str(l): round(v, 4)
                   for l, v in
                   iinj_d1.items()}),
       json.dumps({str(l): round(v, 4)
                   for l, v in
                   cinj_d2.items()}),
       peak_iinj, peak_cinj, sep,
       mono_rate))

# ================================================================
# PART D: own decay + rank + co enrich
# ================================================================
log('== PART D: own decay geometry ==')
OWN = {}
for l in KEY_L:
    Ide = IDEINT_P[l]
    In2 = np.sum(Ide ** 2, axis=1)
    Iown = Ide / (np.linalg.norm(
        Ide, axis=1, keepdims=True) + 1e-12)
    D_ = dvec[l]
    dn2 = np.sum(D_ ** 2, axis=1)
    e_own = np.sum(
        D_ * Iown, axis=1) ** 2 / dn2
    nb_vals = []
    for k in NB_OFFS:
        for sgn in (1, -1):
            idx = (np.arange(NP)
                   + sgn * k) % NP
            nb_vals.append(np.sum(
                D_ * Iown[idx],
                axis=1) ** 2 / dn2)
    e_nb = np.mean(nb_vals, axis=0)
    # I-subspace total projection (GS:
    # qB -> C-perp -> I-perp, 3139
    # semantics; needs B/C at KEY_L)
    X = np.zeros((2, N_T, NP, HIDG),
                 dtype=np.float32)
    for di, dc in enumerate(DIRS):
        for t in range(N_T):
            X[di, t] = \
                H_ld[dc][t][:, l, :] \
                * H_sc[dc][t][l]
    B = X.mean(axis=(0, 1, 2))
    C = X.mean(axis=(0, 2)) - B
    qB = B / (np.linalg.norm(B) + 1e-12)
    Cperp = C - np.outer(
        C @ qB, qB)
    Q_C = Cperp / (np.linalg.norm(
        Cperp, axis=1, keepdims=True)
        + 1e-12)
    Ip = Ide - np.outer(
        Ide @ qB, qB) - (Ide @ Q_C.T) @ Q_C
    Q_I, _R = np.linalg.qr(
        Ip.T.astype(np.float64))
    Q_I = Q_I[:, :Ip.shape[0]]
    projI = (D_ @ Q_I) @ Q_I.T
    e_all = np.sum(projI ** 2,
                   axis=1) / dn2
    # rank of own among all rows
    sim = D_ @ Iown.T
    diag = np.diag(sim)[:, None]
    rank = 1.0 + np.sum(
        sim > diag, axis=1).astype(
        np.float64)
    mean_rank = float(np.mean(rank))
    OWN[l] = {
        'e_own_med': float(np.median(
            e_own)),
        'e_nb_med': float(np.median(
            e_nb)),
        'e_all_med': float(np.median(
            e_all)),
        'mean_rank': mean_rank,
        'rank_frac': mean_rank / NP}
    log('D L%02d: own %.5f | nb %.5f | '
        'all(I) %.5f | rank %.1f (%.3f of '
        'NP)' % (l, OWN[l]['e_own_med'],
                 OWN[l]['e_nb_med'],
                 OWN[l]['e_all_med'],
                 mean_rank,
                 OWN[l]['rank_frac']))
    del X, B, C, Ip, Q_I, projI, sim, rank
    gc.collect()
own_ratio = float(np.median(
    [OWN[l]['e_own_med']
     / max(OWN[l]['e_nb_med'], 1e-12)
     for l in KEY_L]))
own_below_nb = own_ratio < 2.0
rk_frac_med = float(np.median(
    [OWN[l]['rank_frac'] for l in KEY_L]))
if rk_frac_med >= 0.8:
    rk_tag = 'own_rank_chance'
elif rk_frac_med <= 0.2:
    rk_tag = 'own_rank_top'
else:
    rk_tag = 'own_rank_mid'
log('D-SOFT: own/nb ratio med %.3f '
    '(<2: %s); rank frac med %.3f (%s)'
    % (own_ratio, own_below_nb,
       rk_frac_med, rk_tag))

# co36/co50/union enrichment z-test at
# L17 (channels vs identity energy)
log('== PART D2: co enrichment ==')
I17 = IDEINT_P[17]
fr = I17 ** 2 / np.maximum(
    np.sum(I17 ** 2, axis=1,
           keepdims=True), 1e-12)
COZ = {}
for cn, co in CO_SETS.items():
    obs = float(np.mean(
        np.sum(fr[:, co], axis=1)))
    rng_z = _rnd.Random(
        RNG_SEED + 17)
    nulls = np.zeros(CO_PERM)
    nch = fr.shape[1]
    for p in range(CO_PERM):
        cp = rng_z.sample(range(nch),
                          len(co))
        nulls[p] = float(np.mean(
            np.sum(fr[:, cp], axis=1)))
    z = float((obs - nulls.mean())
              / (nulls.std() + 1e-12))
    COZ[cn] = {'obs': obs,
               'null_mean': float(
                   nulls.mean()),
               'z': z}
    log('D2 %s (n=%d): obs %.6f vs null '
        '%.6f -> z %.2f'
        % (cn, len(co), obs,
           nulls.mean(), z))
co_best = max(COZ[cn]['z']
              for cn in CO_SETS)
co_tag = ('co_enriched'
          if co_best >= 2.0 else
          'co_depleted'
          if co_best < 0.0 else
          'co_neutral')

# ================================================================
# PART E: WR behavioral pathway
# ================================================================
log('== PART E: WR PC injections ==')
for l in (26, 29):
    _pc = PC[l]
    v1 = _pc['V'][0]
    for dose in WR_DOSES:
        _dv = np.tile(
            (v1 * _pc['mnorm']
             * dose)[None, :],
            (SCAN_N, 1)).astype(np.float32)
        _run_inj_trial(
            'wrpc1_l%02d_d%.1f' % (l, dose),
            l, _dv, 1.0, 'allstep')
    # PC2 row-specific projection control
    v2 = _pc['V'][1]
    _W = WRPCA[l][:, :, :SCAN_N, :]
    proj = np.mean(
        np.einsum('dtjh,h->dtj', _W, v2),
        axis=(0, 1))
    _dv = (v2[None, :]
           * proj[:, None] * 2.0) \
        .astype(np.float32)
    _run_inj_trial(
        'wrpc2row_l%02d_d2.0' % l, l,
        _dv, 1.0, 'allstep')
wr_keys = [k for k in E_res
           if k.startswith('wr')]
wr_chgs = {k: E_res[k]['chg']
           for k in sorted(wr_keys)}
wr_max = float(max(wr_chgs.values()))
wr_quiet = all(c <= WR_QUIET
               for c in wr_chgs.values())
log('E-SOFT: wr chgs %s (quiet<=%.2f: %s; '
    'max %.4f)'
    % (json.dumps({k: round(v, 4)
                   for k, v in
                   wr_chgs.items()}),
       WR_QUIET, wr_quiet, wr_max))
pc_var_rec = {
    str(l): PC[l]['var'] for l in (26, 29)}

# ================================================================
# PART F: retrieval attribution
# ================================================================
log('== PART F: retrieval attribution ==')
used_pairs = set(pks)
ents_n = len(ents_all)
new_pairs = []
for s in range(ents_n):
    for o in range(ents_n):
        if s == o:
            continue
        pk = '%d_%d' % (s, o)
        if pk not in used_pairs:
            new_pairs.append(pk)
new_pairs = sorted(new_pairs)
if SMOKE:
    new_pairs = new_pairs[:6]
XMAT_N = len(new_pairs)
log('F new pairs: %d unseen (s,o) pairs'
    % XMAT_N)


def build_prompt_new(s, o, lrel, qrel):
    """V1 (3139-identical): fixed-seed
    distractor lines, query first
    candidate, k=0."""
    rng3 = _rnd.Random(zlib.crc32(
        ('xmat|%d_%d_%d_%d'
         % (s, o, lrel, qrel))
        .encode('ascii')))
    lines = [(s, lrel, o)]
    seen = {(s, lrel, o)}
    while len(lines) < 8:
        a = rng3.randrange(ents_n)
        b = rng3.randrange(ents_n)
        rr = rng3.randrange(len(PREDS_all))
        if a == b or (a, rr, b) in seen:
            continue
        seen.add((a, rr, b))
        lines.append((a, rr, b))
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents_all[ls], PREDS_all[lr],
            ents_all[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents_all[s], PREDS_all[qrel],
                ents_all[o]))
    return text


def build_prompt_v2(s, o, lrel, qrel):
    """V2: same-distractor-family prompt.
    Rebuild using the distractor lines of
    a bank row sharing the subject (s,o0),
    filtering any line mentioning o, so
    the context style matches the 672-row
    identity bank."""
    o0 = None
    for pk in pks:
        (s2, o2) = (int(v) for v in
                    pk.split('_'))
        if s2 == s and o2 != o:
            o0 = o2
            break
    assert o0 is not None, (s, o)
    D0 = [tuple(d) for d in
          mat5['distractors']
          ['%d_%d' % (s, o0)]]
    Df = [d for d in D0
          if o not in (d[0], d[2])]
    lines = [(s, lrel, o)] + Df[:7]
    _grng = _rnd.Random(zlib.crc32(
        ('v2fill|%d_%d' % (s, o))
        .encode('ascii')))
    while len(lines) < 8:
        a = _grng.randrange(ents_n)
        b = _grng.randrange(ents_n)
        rr = _grng.randrange(
            len(PREDS_all))
        if a == b or (a, rr, b) in lines:
            continue
        if o in (a, b):
            continue
        lines.append((a, rr, b))
    k0 = int(mat5['kline']
             ['%d_%d' % (s, o0)]) % 8
    rng4 = _rnd.Random(zlib.crc32(
        ('v2|%d_%d' % (s, o))
        .encode('ascii')))
    order = list(range(8))
    rng4.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k0] = \
        lines[k0], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents_all[ls], PREDS_all[lr],
            ents_all[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents_all[s], PREDS_all[qrel],
                ents_all[o]))
    return text


rows_v = {v: [] for v in ('V1', 'V2')}
for pk in new_pairs:
    (s, o) = (int(v)
              for v in pk.split('_'))
    used_rels = {p2r.get(pk)}
    if pk in frel:
        used_rels.update(frel[pk])
    rp = None
    for rr in range(len(PREDS_all)):
        if rr not in used_rels:
            rp = rr
            break
    assert rp is not None, pk
    r = rp
    for v in ('V1', 'V2'):
        if v == 'V1':
            txt = build_prompt_new(
                s, o, r, r)
        else:
            txt = build_prompt_v2(
                s, o, r, r)
        rows_v[v].append(list(tok_g(
            txt, add_special_tokens=False)
            ['input_ids']))
_FK = CK['data'].get('retr_cap')
if _FK is not None:
    vst = {int(k): v for k, v in
           _FK['states'].items()}
    log('F capture RESUMED')
else:
    _cap = capture_states(
        rows_v['V1'] + rows_v['V2'],
        [17, 29, 38])
    _half = len(rows_v['V1'])
    vst = {}
    for l in (17, 29, 38):
        vst[l] = {'V1': _cap[l][:_half],
                  'V2': _cap[l][_half:]}
    ck_save('retr_cap', {
        'states': {
            str(l): {'V1': vst[l]['V1'],
                     'V2': vst[l]['V2']}
            for l in (17, 29, 38)}})
RETR = {}
for v in ('V1', 'V2'):
    retr_s = {}
    for l in (17, 29, 38):
        bank = IDEINT_P[l]
        bank_n = bank / (np.linalg.norm(
            bank, axis=1, keepdims=True)
            + 1e-12)
        q = vst[l][v]
        q_n = q / (np.linalg.norm(
            q, axis=1, keepdims=True)
            + 1e-12)
        sim = q_n @ bank_n.T
        top1 = np.argmax(sim, axis=1)
        hit_s = 0
        for i, pk in enumerate(new_pairs):
            s = int(pk.split('_')[0])
            s2 = int(pks[int(top1[i])]
                     .split('_')[0])
            hit_s += int(s2 == s)
        retr_s[l] = hit_s / XMAT_N
        log('F %s retr L%02d: same-s %.4f '
            '(chance-s ~%.4f)'
            % (v, l, retr_s[l],
               1.0 / ents_n))
    RETR[v] = retr_s
v1_med = float(np.median(
    [RETR['V1'][l] for l in (17, 29, 38)]))
v2_med = float(np.median(
    [RETR['V2'][l] for l in (17, 29, 38)]))
v2_gain = v2_med - v1_med
v2_improve = v2_gain >= V2_GAIN
v2_ok = v2_med >= XMAT_RETR_MIN
if not SMOKE:
    v1_repro = all(
        abs(RETR['V1'][l]
            - RETR39_S[str(l)])
        < 1e-9 for l in (17, 29, 38))
else:
    v1_repro = None
log('F-SOFT: V1 med %.4f | V2 med %.4f '
    '(gain %+.4f, improve %s, v2>=0.30 '
    '%s); v1 3139 repro %s'
    % (v1_med, v2_med, v2_gain,
       v2_improve, v2_ok,
       'n/a(smoke)' if v1_repro is None
       else v1_repro))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3139_ok']
if not SMOKE:
    tags.append('spec_repro_3139_ok'
                if n_match == len(T39)
                else 'spec_repro_3139_drift')
else:
    tags.append('spec_repro_smoke_skipped')
tags.append('iinj_peak_l17'
            if peak_iinj == 17
            else 'iinj_peak_l%02d'
            % peak_iinj)
tags.append('cinj_peak_l26'
            if peak_cinj == 26
            else 'cinj_peak_l%02d'
            % peak_cinj)
tags.append('peaks_separated'
            if (peak_iinj != peak_cinj
                and sep >= 4)
            else 'peaks_overlapped')
tags.append('spec_dose_monotone'
            if dose_mono
            else 'spec_dose_flat')
tags.append('own_below_nb'
            if own_below_nb
            else 'own_above_nb')
tags.append(rk_tag)
tags.append(co_tag)
tags.append('wr_quiet' if wr_quiet
            else 'wr_active')
tags.append('wr_pcvar_recorded')
tags.append('retr_v2_improve'
            if v2_improve
            else 'retr_v2_flat')
tags.append('retr_v2_ok' if v2_ok
            else 'retr_v2_below')
if not SMOKE:
    tags.append('retr_v1_repro_ok'
                if v1_repro
                else 'retr_v1_drift')
else:
    tags.append('retr_v1_smoke_skipped')
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
runtime = time.time() - T0
log('VERDICT: %s' % verdict)
log('DONE (%.1fs)' % runtime)

result = {
    'phase': 3140,
    'name': NAME,
    'smoke': SMOKE,
    'verdict': verdict,
    'runtime_s': runtime,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'part_c': {
        'xphase': xphase,
        'iinj_d1': {str(l): iinj_d1[l]
                    for l in SPECT_L},
        'iinj_d2': {str(l): iinj_d2[l]
                    for l in SPECT_L},
        'cinj_d1': {str(l): cinj_d1[l]
                    for l in SPECT_L},
        'cinj_d2': {str(l): cinj_d2[l]
                    for l in SPECT_L},
        'peak_iinj': int(peak_iinj),
        'peak_cinj': int(peak_cinj),
        'peak_sep': int(sep),
        'mono_rate': mono_rate,
        'repro_3139': repro,
        'n_repro_match': (n_match
                          if not SMOKE
                          else None)},
    'part_d': {
        'own': {str(l): OWN[l]
                for l in KEY_L},
        'own_ratio_nb': own_ratio,
        'rank_frac_med': rk_frac_med,
        'co_z': COZ},
    'part_e': {
        'wr_trials': wr_chgs,
        'wr_max': wr_max,
        'wr_quiet': wr_quiet,
        'pc_var': pc_var_rec},
    'part_f': {
        'n_pairs': XMAT_N,
        'retr_v1': {str(l): RETR['V1'][l]
                    for l in (17, 29, 38)},
        'retr_v2': {str(l): RETR['V2'][l]
                    for l in (17, 29, 38)},
        'v1_med': v1_med,
        'v2_med': v2_med,
        'v2_gain': v2_gain,
        'v1_repro_3139': v1_repro}}
with io.open(os.path.join(
        OUT, 'result.json'), 'w',
        encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False, indent=1)
np.savez(os.path.join(
    OUT, 'p138_readout.npz'),
    spec_iinj_d1=np.array(
        [iinj_d1[l] for l in SPECT_L]),
    spec_iinj_d2=np.array(
        [iinj_d2[l] for l in SPECT_L]),
    spec_cinj_d1=np.array(
        [cinj_d1[l] for l in SPECT_L]),
    spec_cinj_d2=np.array(
        [cinj_d2[l] for l in SPECT_L]),
    spect_l=np.array(SPECT_L),
    own_eown=np.array(
        [OWN[l]['e_own_med']
         for l in KEY_L]),
    own_enb=np.array(
        [OWN[l]['e_nb_med']
         for l in KEY_L]),
    own_eall=np.array(
        [OWN[l]['e_all_med']
         for l in KEY_L]),
    own_rank=np.array(
        [OWN[l]['mean_rank']
         for l in KEY_L]),
    co_z=np.array([COZ[cn]['z']
                   for cn in
                   ('co36', 'co50',
                    'union')]),
    wr_chg=np.array([wr_chgs[k]
                     for k in
                     sorted(wr_keys)]),
    retr_v1=np.array(
        [RETR['V1'][l]
         for l in (17, 29, 38)]),
    retr_v2=np.array(
        [RETR['V2'][l]
         for l in (17, 29, 38)]))
log('dumps done: result.json + seal + '
    'p138_readout.npz')
if not SMOKE:
    if os.path.exists(CKPTF):
        os.remove(CKPTF)
        log('CKPT removed (data fully '
            'persisted)')
print('P3140 DONE (%.1fs)' % runtime)
